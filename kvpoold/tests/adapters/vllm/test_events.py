# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The KV events the client sends to the pool server with ``CompleteStore``.

``build_stored_events`` is tested on its own; the ``PoolOffloadingManager``
tests check what the client puts in ``CompleteStore`` and when. vLLM's own
``OffloadingEventsTracker`` puts the request state in ``ReqContext``, as the
connector does when a request starts, so each test sees the client read what the
connector wrote."""

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import GroupOffloadConfig
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpecKind, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.base import OffloadingKVEventsConfig, ReqContext, make_offload_key

from kvpoold.adapters.vllm.events import build_stored_events
from kvpoold.api.v1 import CompleteStore

BLOCK_SIZE = 16
CONFIG = OffloadingKVEventsConfig(enable_kv_cache_events=True, self_describing_kv_events=True)


def block_hash(i: int) -> BlockHash:
    return BlockHash(i.to_bytes(32, "big"))


def wire(i: int):
    """The form of ``block_hash(i)`` in a published event."""
    return maybe_convert_block_hash(block_hash(i))


def group_config(
    *, group_idx: int = 0, blocks_per_chunk: int = 2, sliding_window_chunks: int | None = None, mamba: bool = False
) -> GroupOffloadConfig:
    if mamba:
        spec = MambaSpec(block_size=BLOCK_SIZE, shapes=((1, 1),), dtypes=(torch.float32,), mamba_cache_mode="align")
        sliding_window_chunks = 1
        kind = KVCacheSpecKind.MAMBA.value
    else:
        spec = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)
        kind = KVCacheSpecKind.FULL_ATTENTION.value
    tokens_per_chunk = BLOCK_SIZE * blocks_per_chunk
    return GroupOffloadConfig(
        group_idx=group_idx,
        tokens_per_block=BLOCK_SIZE,
        tokens_per_chunk=tokens_per_chunk,
        hashes_per_chunk=blocks_per_chunk,
        kv_event_group_spec=OffloadingEventGroupSpec(kind, None),
        sliding_window_size_in_chunks=sliding_window_chunks,
        kv_cache_spec=spec,
        manager_cls=KVCacheSpecRegistry.get_manager_class(spec),
        requires_cow_source=mamba,
    )


def request(num_tokens: int):
    """The fields of vLLM's ``Request`` that the connector's tracker keeps."""
    return SimpleNamespace(
        block_hashes=[block_hash(i) for i in range(num_tokens // BLOCK_SIZE)],
        all_token_ids=list(range(1000, 1000 + num_tokens)),
        lora_request=None,
        resumable=False,
        kv_transfer_params=None,
    )


def started(req, *groups: GroupOffloadConfig) -> ReqContext:
    """A ReqContext as the connector leaves it once the request has started."""
    context = ReqContext(req_id="store")
    OffloadingEventsTracker(CONFIG).on_new_request(context, req, groups)
    return context


def chunk_key(req, group: GroupOffloadConfig, context: ReqContext, chunk_idx: int):
    """The key of a chunk, with the end position the scheduler records for it."""
    key = make_offload_key(req.block_hashes[(chunk_idx + 1) * group.hashes_per_chunk - 1], group.group_idx)
    context.set_offload_key_position(key, (chunk_idx + 1) * group.tokens_per_chunk)
    return key


def build(keys, context):
    return build_stored_events(keys, context, CONFIG)


# --- build_stored_events ---------------------------------------------------------


def test_a_chunk_of_two_blocks_is_one_event_with_its_hashes_parent_and_tokens():
    req, group = request(4 * BLOCK_SIZE), group_config()
    context = started(req, group)

    [event] = build([chunk_key(req, group, context, 1)], context)

    assert type(event) is BlockStored
    assert event.block_hashes == [wire(2), wire(3)]
    assert event.parent_block_hash == wire(1)
    assert event.token_ids == req.all_token_ids[2 * BLOCK_SIZE : 4 * BLOCK_SIZE]
    assert event.block_size == BLOCK_SIZE
    assert event.group_idx == 0
    assert event.kv_cache_spec_kind == "full_attention"


def test_the_first_chunk_has_no_parent():
    req, group = request(4 * BLOCK_SIZE), group_config()
    context = started(req, group)

    [event] = build([chunk_key(req, group, context, 0)], context)

    assert event.parent_block_hash is None
    assert event.block_hashes == [wire(0), wire(1)]


def test_one_event_per_key_in_the_order_of_the_keys():
    req, group = request(4 * BLOCK_SIZE), group_config()
    context = started(req, group)
    keys = [chunk_key(req, group, context, 1), chunk_key(req, group, context, 0)]

    events = build(keys, context)

    assert [event.block_hashes[-1] for event in events] == [wire(3), wire(1)]


def test_the_tier_owns_the_medium_locality_ownership_and_session():
    req, group = request(2 * BLOCK_SIZE), group_config()
    context = started(req, group)

    [event] = build([chunk_key(req, group, context, 0)], context)

    assert (event.medium, event.locality, event.ownership, event.session_id) == (None, None, None, None)


def test_a_windowed_group_gets_the_placeholder():
    req, group = request(2 * BLOCK_SIZE), group_config(sliding_window_chunks=2)
    context = started(req, group)

    [event] = build([chunk_key(req, group, context, 0)], context)

    assert event.block_hashes == [wire(1)]
    assert (event.parent_block_hash, event.token_ids, event.block_size) == (None, [], 0)
    assert event.kv_cache_spec_kind == "full_attention"


def test_a_recurrent_group_announces_its_state_checkpoint():
    req, group = request(4 * BLOCK_SIZE), group_config(mamba=True)
    context = started(req, group)

    [event] = build([chunk_key(req, group, context, 1)], context)

    assert event.kv_cache_spec_kind == "mamba"
    assert event.block_hashes == [wire(3)]
    assert event.parent_block_hash == wire(2)
    assert event.token_ids == []


def test_a_partial_recurrent_tail_describes_only_its_valid_hashes():
    req, group = request(4 * BLOCK_SIZE), group_config(mamba=True)
    context = started(req, group)
    # The state is valid at the third hash, inside the second chunk.
    key = make_offload_key(req.block_hashes[2], group.group_idx)
    context.set_offload_key_position(key, 3 * BLOCK_SIZE)

    [event] = build([key], context)

    assert event.kv_cache_spec_kind == "mamba"
    assert event.block_hashes == [wire(2)]
    assert event.parent_block_hash == wire(1)
    assert event.token_ids == []


def test_a_request_without_state_gets_placeholders_for_its_keys():
    [event] = build([make_offload_key(block_hash(7), 0)], ReqContext(req_id="untracked"))

    assert (event.block_hashes, event.block_size, event.group_idx) == ([wire(7)], 0, 0)


def test_the_same_call_twice_gives_the_same_events():
    """Partial-tail metadata is dropped with the call that recorded it, so no call sees another's."""
    req, group = request(4 * BLOCK_SIZE), group_config(mamba=True)
    context = started(req, group)
    key = make_offload_key(req.block_hashes[2], group.group_idx)
    context.set_offload_key_position(key, 3 * BLOCK_SIZE)

    assert [e.block_hashes for e in build([key], context)] == [e.block_hashes for e in build([key], context)]


# --- PoolOffloadingManager.complete_store ------------------------------------


@pytest.fixture
def sent(monkeypatch):
    """Watches a manager; returns the CompleteStore notices it sends, in order."""

    def watch(manager) -> list[CompleteStore]:
        notices: list[CompleteStore] = []
        notify = manager.client._session.notify

        def recording_notify(notice):
            if isinstance(notice, CompleteStore):
                notices.append(notice)
            notify(notice)

        monkeypatch.setattr(manager.client._session, "notify", recording_notify)
        return notices

    return watch


@pytest.fixture
def events_manager(engine):
    """An engine that has self-describing KV events enabled."""
    return engine(enable_kv_cache_events=True, self_describing_kv_events=True)


def test_complete_store_sends_the_keys_with_their_events(events_manager, sent):
    manager = events_manager
    notices = sent(manager)
    req, group = request(4 * BLOCK_SIZE), group_config()
    context = started(req, group)
    keys = [chunk_key(req, group, context, 1), chunk_key(req, group, context, 0)]

    manager.complete_store(keys, context)

    assert notices[0].keys == keys
    assert notices[0].success is True
    assert [event.block_hashes[-1] for event in notices[0].events] == [wire(3), wire(1)]


def test_no_events_when_the_engine_has_kv_events_disabled(engine, sent):
    manager = engine(enable_kv_cache_events=False, self_describing_kv_events=True)
    notices = sent(manager)
    req, group = request(2 * BLOCK_SIZE), group_config()
    context = started(req, group)

    manager.complete_store([chunk_key(req, group, context, 0)], context)

    assert notices[0].events is None


def test_no_events_when_the_engine_does_not_describe_its_events(engine, sent):
    manager = engine(enable_kv_cache_events=True, self_describing_kv_events=False)
    notices = sent(manager)
    req, group = request(2 * BLOCK_SIZE), group_config()
    context = started(req, group)

    manager.complete_store([chunk_key(req, group, context, 0)], context)

    assert notices[0].events is None


def test_a_failed_store_sends_no_events(events_manager, sent):
    manager = events_manager
    notices = sent(manager)
    req, group = request(2 * BLOCK_SIZE), group_config()
    context = started(req, group)
    key = chunk_key(req, group, context, 0)

    manager.complete_store([key], context, success=False)

    assert notices[0].success is False
    assert notices[0].keys == [key]
    assert notices[0].events is None


def test_the_client_yields_no_events_of_its_own(events_manager):
    manager = events_manager
    req, group = request(2 * BLOCK_SIZE), group_config()
    context = started(req, group)
    manager.complete_store([chunk_key(req, group, context, 0)], context)

    assert list(manager.take_events()) == []

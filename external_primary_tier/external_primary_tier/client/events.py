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

"""Builds the ``BlockStored`` events the client sends to the tier in ``CompleteStore``.

The tier publishes the KV events of the shared pool, because only the tier
knows when a chunk is evicted. But the tier sees a chunk only as an opaque
``OffloadKey``. An event consumer such as the llm-d KV-cache indexer needs the
chunk's block hashes, parent hash and token ids. Only the engine has them, in
its ``Request``. So at store time the engine builds the events and sends them
to the tier, which keeps them with the chunk and publishes them.
"""

from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
    _RequestEventContext,
)
from vllm.v1.kv_offload.base import (
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
)


def build_stored_events(
    keys: list[OffloadKey], req_context: ReqContext, kv_events_config: OffloadingKVEventsConfig
) -> list[BlockStored]:
    """Return one ``BlockStored`` per key, in the order of ``keys``.

    The events match what vLLM's offloading connector would publish for the same
    chunks. That's because they come from vLLM's own ``OffloadingEventsTracker``,
    run here on the stored keys. The tracker is not reimplemented.

    The request data comes from ``req_context``. When a request starts, the
    connector's tracker stores the request's block hashes, token ids, LoRA and
    KV-cache group configs on ``req_context`` as ``_RequestEventContext``. This
    function reads that state back.

    A new tracker is created on every call. A tracker caches per-key metadata
    until it sees that key's removal event. Removals happen in the tier and
    never reach this tracker, so a long-lived tracker would grow without bound.

    ``medium``, ``locality`` and ``ownership`` are left None, and the tier fills
    them in. A key gets a placeholder event carrying only its hash and group in
    these cases:

    - its group is sliding-window attention,
    - the request has no ``_RequestEventContext``,
    - the request is resumable.

    vLLM's connector publishes the same placeholder in those cases.
    """
    state = req_context.get_state(_RequestEventContext)
    group_specs = {} if state is None else {i: c.kv_event_group_spec for i, c in state.group_configs.items()}
    tracker = OffloadingEventsTracker(kv_events_config, group_specs)
    if state is not None:
        for key in keys:
            # Partial tails. A chunk normally holds tokens_per_chunk tokens and
            # is stored once it is full, so its key's end position is a
            # multiple of tokens_per_chunk. The tracker uses that position to
            # find which tokens and hashes the chunk covers. In hybrid models,
            # a request's last chunk can be stored before it is full: its key
            # ends mid-chunk.
            # The tracker emits only a placeholder for such a key unless the key is first
            # registered with record_partial_store, which is what the connector
            # does at store time and what this loop does here.
            group_config = state.group_configs.get(get_offload_group_idx(key))
            end_position = req_context.get_offload_key_position(key)
            if group_config is not None and end_position is not None and end_position % group_config.tokens_per_chunk:
                # record_partial_store expects a Request, but it reads only
                # block_hashes, all_token_ids and lora_request, which the state
                # also has.
                tracker.record_partial_store(state, group_config, end_position, key)  # type: ignore[arg-type]
    # take_events needs a medium in order to translate the event. The value is
    # discarded below.
    stored = list(
        tracker.take_events([OffloadingEvent(keys=keys, medium=Medium.CPU, removed=False, req_context=req_context)])
    )
    assert len(stored) == len(keys)
    for event in stored:
        event.medium = None
    return stored

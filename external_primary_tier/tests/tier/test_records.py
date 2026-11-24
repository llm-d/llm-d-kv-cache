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

"""The records: vLLM's CPUOffloadingManager shared by several clients, and the
clients' leases on it."""

from collections import Counter

import msgspec
import pytest
from external_primary_tier.tier.records import Records
from external_primary_tier.tier.snapshot import Snapshot
from external_primary_tier.tier.state import ClientContext
from vllm.distributed.kv_events import MEDIUM_CPU, BlockRemoved, BlockStored
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

from .fake_engine import key
from .invariants import check

LEASE = 30.0
NOW = 1000.0
A = b"\xaa" * 8
B = b"\xbb" * 8


def position(i: int) -> int:
    return 16 * (i + 1)


def make_records(num_chunks: int = 4, cache_policy: str = "lru") -> Records:
    return Records(num_chunks, cache_policy, lease_seconds=LEASE)


def add_client(records: Records, client_id: str, compat_id: bytes) -> ClientContext:
    client = ClientContext(client_id=client_id, compat_id=compat_id)
    records.clients[client_id] = client
    return client


def store(records: Records, client: ClientContext, req_id: str, ids: list[int], now: float = NOW):
    """Store the keys and complete the store; return the keys stored and their chunk ids."""
    result = records.prepare_store(client, req_id, [key(i) for i in ids], [position(i) for i in ids], now)
    assert result is not None
    records.complete_store(client, result[0], success=True)
    return result


def lookup(records: Records, client: ClientContext, req_id: str, i: int, now: float = NOW) -> LookupResult:
    return records.lookup(client, req_id, key(i), now)


@pytest.fixture
def records() -> Records:
    return make_records()


@pytest.fixture
def a(records) -> ClientContext:
    return add_client(records, "engine-a", A)


@pytest.fixture
def b(records) -> ClientContext:
    return add_client(records, "engine-b", B)


# --- Pins ---------------------------------------------------------------------


def test_a_lookup_hit_pins_the_key_for_the_client(records, a):
    store(records, a, "r1", [0])
    assert lookup(records, a, "r2", 0) is LookupResult.HIT
    assert a.pins == {A + key(0): [NOW + LEASE]}
    assert records.stats(a)["num_pinned_chunks"] == 1
    check(records)


def test_a_miss_or_a_pending_hit_pins_nothing(records, a):
    assert lookup(records, a, "r1", 0) is LookupResult.MISS
    assert records.prepare_store(a, "r1", [key(1)], [position(1)], NOW) is not None
    assert lookup(records, a, "r2", 1) is LookupResult.HIT_PENDING
    assert a.pins == {}
    check(records)


def test_the_lookup_pin_carries_the_load_until_complete_load(records, a):
    _, chunk_ids = store(records, a, "r1", [0, 1])
    for i in (0, 1):
        lookup(records, a, "r2", i)
    assert records.prepare_load(a, "r2", [key(0), key(1)], [position(0), position(1)]) == chunk_ids
    assert records.stats(a)["client_pins"] == 2
    check(records)

    records.complete_load(a, [key(0), key(1)])
    assert a.pins == {}
    assert records.stats(a)["num_evictable_chunks"] == 2
    check(records)


def test_a_load_of_a_key_the_client_has_not_pinned_is_refused(records, a):
    same_model = add_client(records, "engine-c", A)
    store(records, a, "r1", [0])
    assert records.prepare_load(a, "r2", [key(0)], [position(0)]) is None
    # Another client's pin on the key is not this client's.
    lookup(records, same_model, "r3", 0)
    assert records.prepare_load(a, "r2", [key(0)], [position(0)]) is None
    check(records)


def test_release_hits_releases_that_many_pins(records, a):
    store(records, a, "r1", [0])
    lookup(records, a, "r2", 0)
    lookup(records, a, "r3", 0)
    assert len(a.pins[A + key(0)]) == 2

    records.release_hits(a, [(key(0), 1)])
    assert len(a.pins[A + key(0)]) == 1
    check(records)

    # Releasing more than the client holds releases what it holds.
    records.release_hits(a, [(key(0), 5)])
    assert a.pins == {}
    check(records)


# --- Writes -------------------------------------------------------------------


def test_a_successful_store_makes_its_keys_evictable(records, a):
    result = records.prepare_store(a, "r1", [key(0), key(1)], [position(0), position(1)], NOW)
    assert result is not None
    keys_to_store, chunk_ids = result
    assert keys_to_store == [key(0), key(1)]
    assert sorted(chunk_ids) == [0, 1]
    assert a.writes == {A + key(0): NOW + LEASE, A + key(1): NOW + LEASE}
    assert records.stats(a)["num_write_pending_chunks"] == 2
    check(records)

    records.complete_store(a, [key(0), key(1)], success=True)
    assert a.writes == {}
    assert records.stats(a)["num_evictable_chunks"] == 2
    assert lookup(records, a, "r2", 0) is LookupResult.HIT
    check(records)


def test_a_failed_store_frees_its_chunks(records, a):
    records.prepare_store(a, "r1", [key(0)], [position(0)], NOW)
    records.complete_store(a, [key(0)], success=False)
    assert lookup(records, a, "r2", 0) is LookupResult.MISS
    assert records.stats(a)["num_free_chunks"] == 4
    check(records)


def test_a_client_cannot_complete_another_clients_write(records, a):
    same_model = add_client(records, "engine-c", A)
    records.prepare_store(a, "r1", [key(0)], [position(0)], NOW)
    records.complete_store(same_model, [key(0)], success=False)
    assert A + key(0) in a.writes
    assert lookup(records, same_model, "r2", 0) is LookupResult.HIT_PENDING
    check(records)


def test_no_chunks_while_every_chunk_is_held(records, a):
    store(records, a, "r1", [0, 1, 2, 3])
    for i in range(4):
        lookup(records, a, "r2", i)
    assert records.prepare_store(a, "r3", [key(4)], [position(4)], NOW) is None
    check(records)


# --- Compatibility ids ----------------------------------------------------------


def test_clients_with_the_same_compat_id_share_chunks(records, a):
    same_model = add_client(records, "engine-c", A)
    store(records, a, "r1", [0])
    assert lookup(records, same_model, "r2", 0) is LookupResult.HIT


def test_clients_with_another_compat_id_keep_their_own_chunks(records, a, b):
    store(records, a, "r1", [0])
    assert lookup(records, b, "r2", 0) is LookupResult.MISS
    assert store(records, b, "r3", [0])[0] == [key(0)]
    assert records.stats(a)["num_evictable_chunks"] == 2
    check(records)


def test_eviction_is_shared_across_clients(records, a, b):
    store(records, a, "r1", [0, 1])
    store(records, b, "r2", [2, 3])
    # Three chunks are needed: the oldest, both a's, and one of b's.
    result = records.prepare_store(a, "r3", [key(4), key(5), key(6)], [None, None, None], NOW)
    assert result is not None
    assert [lookup(records, a, "r4", i) for i in (0, 1)] == [LookupResult.MISS, LookupResult.MISS]
    assert [lookup(records, b, "r5", i) for i in (2, 3)].count(LookupResult.HIT) == 1
    check(records)


# --- Fence ----------------------------------------------------------------------


def test_a_fenced_pool_misses_and_hands_out_no_chunk(records, a):
    store(records, a, "r1", [0])
    records.fence_until = NOW + 10
    assert lookup(records, a, "r2", 0) is LookupResult.MISS
    assert records.prepare_store(a, "r2", [key(1)], [position(1)], NOW) is None
    assert lookup(records, a, "r2", 0, now=NOW + 10) is LookupResult.HIT
    check(records)


# --- Deadlines ------------------------------------------------------------------


def test_a_pin_is_released_at_its_deadline_and_not_before(records, a):
    store(records, a, "r1", [0])
    lookup(records, a, "r2", 0)
    assert records.sweep(NOW + LEASE - 1) == 0
    assert a.pins
    assert records.sweep(NOW + LEASE) == 1
    assert a.pins == {}
    assert records.stats(a)["num_evictable_chunks"] == 1
    check(records)


def test_only_the_pins_past_their_deadlines_are_released(records, a):
    store(records, a, "r1", [0])
    lookup(records, a, "r2", 0)
    lookup(records, a, "r3", 0, now=NOW + 10)
    assert records.sweep(NOW + LEASE) == 1
    assert a.pins == {A + key(0): [NOW + 10 + LEASE]}
    check(records)


def test_a_write_fails_at_its_deadline(records, a):
    records.prepare_store(a, "r1", [key(0)], [position(0)], NOW)
    assert records.sweep(NOW + LEASE - 1) == 0
    assert records.sweep(NOW + LEASE) == 1
    assert a.writes == {}
    check(records)

    # A completion after the deadline is ignored.
    records.complete_store(a, [key(0)], success=True)
    assert lookup(records, a, "r2", 0, now=NOW + LEASE) is LookupResult.MISS
    check(records)


def test_request_finished_forgets_the_request(records, a):
    store(records, a, "r1", [0])
    assert "r1" in a.requests
    records.request_finished(a, "r1")
    assert a.requests == {}


def test_a_client_that_leaves_releases_everything(records, a, b):
    store(records, a, "r1", [0, 1])
    lookup(records, a, "r2", 0)
    records.prepare_store(a, "r3", [key(2)], [position(2)], NOW)

    records.release_client(a)
    assert list(records.clients) == ["engine-b"]
    values = records.stats(b)
    assert values["num_pinned_chunks"] == 0
    assert values["num_write_pending_chunks"] == 0
    assert values["num_evictable_chunks"] == 2
    assert values["num_free_chunks"] == 2
    check(records)


def test_a_cache_policy_is_loaded_from_its_module_path():
    records = Records(
        4,
        "LRUCachePolicy",
        lease_seconds=LEASE,
        cache_policy_module_path="vllm.v1.kv_offload.cpu.policies.lru",
    )
    assert type(records.manager._policy) is LRUCachePolicy
    a = add_client(records, "engine-a", A)
    store(records, a, "r1", [0])
    assert lookup(records, a, "r2", 0) is LookupResult.HIT
    check(records)


# --- KV events ------------------------------------------------------------------


def block_stored(i: int) -> BlockStored:
    """An engine's event for the chunk of ``key(i)``: its hash is an int, not the key."""
    return BlockStored(
        block_hashes=[1000 + i],
        parent_block_hash=None,
        token_ids=[i],
        block_size=16,
        lora_id=None,
        medium=None,
        lora_name=None,
        extra_keys=[("adapter", ["mm", i])],
        group_idx=0,
    )


def store_with_events(
    records: Records, client: ClientContext, req_id: str, ids: list[int], now: float = NOW
) -> list[bytes]:
    """Store the keys, complete the store with an event per key, and return the keys stored."""
    result = records.prepare_store(client, req_id, [key(i) for i in ids], [position(i) for i in ids], now)
    assert result is not None
    keys_to_store = result[0]
    by_key = {key(i): block_stored(i) for i in ids}
    records.complete_store(client, keys_to_store, success=True, events=[by_key[k] for k in keys_to_store])
    return keys_to_store


def stored(i: int) -> BlockStored:
    """The event published for the chunk of ``key(i)``: the engine's, with the CPU medium."""
    event = block_stored(i)
    event.medium = MEDIUM_CPU
    return event


def removed(i: int) -> BlockRemoved:
    return BlockRemoved(block_hashes=[1000 + i], medium=MEDIUM_CPU, group_idx=0)


def test_a_successful_store_buffers_a_stored_event_per_key(records, a):
    store_with_events(records, a, "r1", [0, 1])
    assert records.take_events() == [stored(0), stored(1)]
    assert records.take_events() == []
    assert records.block_stored == {A + key(0): stored(0), A + key(1): stored(1)}
    check(records)


def test_nothing_is_buffered_before_the_store_completes(records, a):
    records.prepare_store(a, "r1", [key(0)], [position(0)], NOW)
    assert records.take_events() == []
    assert records.block_stored == {}


def test_a_failed_store_buffers_nothing(records, a):
    records.prepare_store(a, "r1", [key(0), key(1)], [position(0), position(1)], NOW)
    records.complete_store(a, [key(0), key(1)], success=False, events=[block_stored(0), block_stored(1)])
    assert records.take_events() == []
    assert records.block_stored == {}
    check(records)


def test_a_store_without_events_buffers_nothing_on_store_or_removal(records, a):
    store(records, a, "r1", [0, 1, 2, 3])
    assert records.take_events() == []
    store(records, a, "r2", [4, 5])  # evicts two chunks
    assert lookup(records, a, "r3", 3) is LookupResult.MISS
    assert records.take_events() == []
    assert records.block_stored == {}


def test_eviction_in_prepare_store_buffers_a_removal_from_the_stored_event(records, a):
    store_with_events(records, a, "r1", [0, 1, 2, 3])
    records.take_events()
    store_with_events(records, a, "r2", [4, 5])
    # LRU evicts the end of a prefix first: 3, then 2. The removals come before the new chunks' events.
    assert records.take_events() == [removed(3), removed(2), stored(4), stored(5)]
    assert set(records.block_stored) == {A + key(i) for i in (0, 1, 4, 5)}
    check(records)


def test_a_removal_carries_the_stored_hashes_not_the_key(records, a):
    store_with_events(records, a, "r1", [0, 1, 2, 3])
    records.take_events()
    store_with_events(records, a, "r2", [4])
    [removal] = records.take_events()[:1]
    assert isinstance(removal, BlockRemoved)
    assert removal.block_hashes == [1003]
    assert removal.group_idx == 0
    assert removal.medium == MEDIUM_CPU
    assert removal.locality is None and removal.ownership is None


def test_a_stored_event_is_the_engines_with_the_cpu_medium(records, a):
    records.prepare_store(a, "r1", [key(0)], [position(0)], NOW)
    records.complete_store(a, [key(0)], success=True, events=[block_stored(0)])
    published = records.take_events()
    assert published[0] is records.block_stored[A + key(0)]
    assert published[0].medium == MEDIUM_CPU
    assert published[0].locality is None and published[0].ownership is None


def test_evicted_chunks_without_events_stay_silent(records, a):
    store(records, a, "r1", [0, 1])  # the engine had events off
    store_with_events(records, a, "r2", [2, 3])
    records.take_events()
    store_with_events(records, a, "r3", [4, 5])  # evicts 0 and 1
    assert records.take_events() == [stored(4), stored(5)]


def test_a_chunk_evicted_by_another_clients_store_is_removed(records, a, b):
    store_with_events(records, a, "r1", [0, 1, 2, 3])
    records.take_events()
    store_with_events(records, b, "r2", [0])
    assert records.take_events() == [removed(3), stored(0)]


def test_events_of_clients_with_the_same_compat_id_come_in_order(records, a):
    same_model = add_client(records, "engine-c", A)
    store_with_events(records, a, "r1", [0])
    store_with_events(records, same_model, "r2", [1])
    assert records.take_events() == [stored(0), stored(1)]


def test_a_pinned_chunk_is_not_evicted_and_buffers_no_removal(records, a):
    store_with_events(records, a, "r1", [0, 1, 2, 3])
    lookup(records, a, "r2", 3)  # the chunk LRU would evict first
    records.take_events()
    store_with_events(records, a, "r3", [4])
    assert records.take_events() == [removed(2), stored(4)]
    assert A + key(3) in records.block_stored


def test_a_store_that_finds_no_chunk_buffers_nothing(records, a):
    store_with_events(records, a, "r1", [0, 1, 2, 3])
    for i in range(4):
        lookup(records, a, "r2", i)
    records.take_events()
    assert records.prepare_store(a, "r3", [key(4)], [position(4)], NOW) is None
    assert records.take_events() == []


def test_a_lease_that_ends_on_a_write_buffers_nothing(records, a):
    records.prepare_store(a, "r1", [key(0)], [position(0)], NOW)
    assert records.sweep(NOW + LEASE) == 1
    records.complete_store(a, [key(0)], success=True, events=[block_stored(0)])
    assert records.take_events() == []
    assert records.block_stored == {}


def test_a_lease_that_ends_on_a_pin_keeps_the_chunk_and_its_event(records, a):
    store_with_events(records, a, "r1", [0])
    records.take_events()
    lookup(records, a, "r2", 0)
    assert records.sweep(NOW + LEASE) == 1
    assert records.take_events() == []
    assert A + key(0) in records.block_stored


def test_a_client_that_leaves_removes_nothing_that_was_stored(records, a):
    store_with_events(records, a, "r1", [0])
    lookup(records, a, "r2", 0)
    records.prepare_store(a, "r3", [key(1)], [position(1)], NOW)
    records.take_events()
    records.release_client(a)
    # The failed write was never published, and the stored chunk stays in the pool.
    assert records.take_events() == []
    assert A + key(0) in records.block_stored
    check(records)


def test_a_key_already_in_the_pool_is_not_handed_out_again(records, a):
    same_model = add_client(records, "engine-c", A)
    store_with_events(records, a, "r1", [0])
    records.take_events()
    result = records.prepare_store(same_model, "r2", [key(0)], [position(0)], NOW)
    assert result is not None
    assert result[0] == []
    # A completion for it is not among the engine's writes.
    records.complete_store(same_model, [key(0)], success=True, events=[block_stored(0)])
    assert records.take_events() == []
    assert records.block_stored == {A + key(0): stored(0)}
    check(records)


def test_a_key_being_written_is_not_handed_out_to_another_engine(records, a):
    same_model = add_client(records, "engine-c", A)
    records.prepare_store(a, "r1", [key(0)], [position(0)], NOW)
    result = records.prepare_store(same_model, "r2", [key(0)], [position(0)], NOW)
    assert result is not None
    assert result[0] == []


def test_a_repeated_completion_buffers_one_event(records, a):
    keys_to_store = store_with_events(records, a, "r1", [0])
    records.complete_store(a, keys_to_store, success=True, events=[block_stored(0)])
    assert records.take_events() == [stored(0)]


def test_a_chunk_stored_again_after_eviction_is_published_again(records, a):
    store_with_events(records, a, "r1", [0, 1, 2, 3])
    store_with_events(records, a, "r2", [4])  # evicts 3
    store_with_events(records, a, "r3", [3])  # evicts 2
    events = records.take_events()
    assert events[-4:] == [removed(3), stored(4), removed(2), stored(3)]


def test_a_wrong_number_of_events_is_refused_before_anything_changes(records, a):
    records.prepare_store(a, "r1", [key(0), key(1)], [position(0), position(1)], NOW)
    with pytest.raises(ValueError):
        records.complete_store(a, [key(0), key(1)], success=True, events=[block_stored(0)])
    assert set(a.writes) == {A + key(0), A + key(1)}
    check(records)


def snapshot_of_stored_chunks(ids: list[int]) -> tuple[Records, Snapshot]:
    records = make_records(4, "lru")
    a = add_client(records, "engine-a", A)
    store_with_events(records, a, "r1", ids)
    snapshot = records.snapshot(b"region-1")
    assert snapshot is not None
    return records, snapshot


def through_the_file(snapshot: Snapshot) -> Snapshot:
    return msgspec.msgpack.decode(msgspec.msgpack.encode(snapshot), type=Snapshot)


def test_a_snapshot_keeps_each_chunks_event_through_encoding():
    _, snapshot = snapshot_of_stored_chunks([0, 1, 2, 3])
    decoded = through_the_file(snapshot)
    assert decoded.block_stored == [(A + key(i), stored(i)) for i in range(4)]


def test_a_restored_chunk_publishes_nothing_at_restore_and_a_removal_when_evicted():
    _, snapshot = snapshot_of_stored_chunks([0, 1, 2, 3])
    restored = Records.restore(through_the_file(snapshot), 4, LEASE, lease_until=NOW + 10)
    assert set(restored.block_stored) == {A + key(i) for i in range(4)}
    assert restored.take_events() == []

    client = restored.clients["engine-a"]
    # Evicts a restored chunk, and stores a new one.
    store_with_events(restored, client, "r2", [4])
    assert restored.take_events() == [removed(3), stored(4)]
    assert set(restored.block_stored) == {A + key(i) for i in (0, 1, 2, 4)}
    check(restored)


def test_a_snapshot_without_block_stored_decodes_and_restores_chunks_that_have_none():
    _, snapshot = snapshot_of_stored_chunks([0, 1, 2, 3])
    fields = msgspec.to_builtins(snapshot, builtin_types=(bytes,))
    del fields["block_stored"]
    old = msgspec.msgpack.decode(msgspec.msgpack.encode(fields), type=Snapshot)
    assert old.block_stored == []
    restored = Records.restore(old, 4, LEASE, lease_until=NOW + 10)
    assert restored.block_stored == {}
    store_with_events(restored, restored.clients["engine-a"], "r2", [4])  # evicts a chunk with no event
    assert restored.take_events() == [stored(4)]
    check(restored)


def test_a_snapshot_with_a_field_this_reader_does_not_know_still_decodes():
    _, snapshot = snapshot_of_stored_chunks([0])
    fields = msgspec.to_builtins(snapshot, builtin_types=(bytes,))
    fields["added_later"] = 1
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(fields), type=Snapshot).num_allocated_chunks == 1


def test_a_chunk_being_written_gets_its_event_when_its_store_completes_not_when_restored():
    records, _ = snapshot_of_stored_chunks([0])
    records.prepare_store(records.clients["engine-a"], "r2", [key(1)], [position(1)], NOW)  # a write
    snapshot = records.snapshot(b"region-1")
    assert snapshot is not None
    assert [entry.ref_cnt for entry in snapshot.entries] == [0, -1]
    # Even a snapshot that lists an event for the write does not give it one.
    forged = msgspec.structs.replace(snapshot, block_stored=[*snapshot.block_stored, (A + key(1), block_stored(1))])
    restored = Records.restore(forged, 4, LEASE, lease_until=NOW + 10)
    assert set(restored.block_stored) == {A + key(0)}

    restored.complete_store(restored.clients["engine-a"], [key(1)], success=True, events=[block_stored(1)])
    assert restored.take_events() == [stored(1)]
    check(restored)


def test_a_restored_write_that_fails_publishes_nothing():
    records, _ = snapshot_of_stored_chunks([0])
    records.prepare_store(records.clients["engine-a"], "r2", [key(1)], [position(1)], NOW)
    snapshot = records.snapshot(b"region-1")
    assert snapshot is not None
    restored = Records.restore(snapshot, 4, LEASE, lease_until=NOW + 10)
    assert restored.sweep(NOW + 10) == 1
    assert restored.take_events() == []
    assert set(restored.block_stored) == {A + key(0)}
    check(restored)


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_every_chunk_is_stored_once_and_removed_once(cache_policy):
    records = make_records(4, cache_policy)
    a = add_client(records, "engine-a", A)
    live: set[int] = set()
    for n in range(12):
        store_with_events(records, a, f"r{n}", [n])
        for event in records.take_events():
            [block_hash] = event.block_hashes
            assert (block_hash in live) == isinstance(event, BlockRemoved)
            (live.discard if isinstance(event, BlockRemoved) else live.add)(block_hash)
        assert live == {kept.block_hashes[0] for kept in records.block_stored.values()}
        check(records)
    assert len(live) == 4


# --- The same answers as vLLM's manager -------------------------------------------


class _Vanilla:
    """vLLM's CPUOffloadingManager for one engine, called as vLLM's connector calls it."""

    def __init__(self, num_chunks: int, cache_policy: str) -> None:
        self.manager = CPUOffloadingManager(num_chunks, cache_policy=cache_policy)
        self.requests: dict[str, ReqContext] = {}
        self.last_stored: list[bytes] = []

    def _context(self, req_id: str) -> ReqContext:
        return self.requests.setdefault(req_id, ReqContext(req_id=req_id))

    def lookup(self, req_id, i):
        context = self._context(req_id)
        context.set_offload_key_position(key(i), position(i))
        return self.manager.lookup(key(i), context)

    def load(self, req_id, ids):
        return self.manager.prepare_load([key(i) for i in ids], self._context(req_id)).block_ids.tolist()

    def complete_load(self, req_id, ids):
        self.manager.complete_load([key(i) for i in ids], self._context(req_id))

    def end_step(self):
        pass

    def store(self, req_id, ids):
        context = self._context(req_id)
        for i in ids:
            context.set_offload_key_position(key(i), position(i))
        output = self.manager.prepare_store([key(i) for i in ids], context)
        if output is None:
            return None
        self.last_stored = list(output.keys_to_store)
        return (list(output.keys_to_store), output.store_spec.block_ids.tolist())

    def complete_store(self):
        self.manager.complete_store(self.last_stored, ReqContext(req_id=""), success=True)

    def finish(self, req_id):
        self.manager.on_request_finished(self.requests.pop(req_id))


class _Shared:
    """The records for one client, called as ExternalPrimaryTierClient calls them:
    hits not loaded in their step are released at its end."""

    def __init__(self, num_chunks: int, cache_policy: str) -> None:
        self.records = make_records(num_chunks, cache_policy)
        self.client = add_client(self.records, "engine-a", A)
        self.step_hits: Counter[bytes] = Counter()
        self.last_stored: list[bytes] = []

    def lookup(self, req_id, i):
        result = self.records.lookup(self.client, req_id, key(i), NOW)
        if result is LookupResult.HIT:
            self.step_hits[key(i)] += 1
        return result

    def load(self, req_id, ids):
        self.step_hits.subtract(key(i) for i in ids)
        return self.records.prepare_load(self.client, req_id, [key(i) for i in ids], [position(i) for i in ids])

    def complete_load(self, req_id, ids):
        self.records.complete_load(self.client, [key(i) for i in ids])

    def end_step(self):
        self.records.release_hits(self.client, [(k, n) for k, n in self.step_hits.items() if n > 0])
        self.step_hits.clear()

    def store(self, req_id, ids):
        result = self.records.prepare_store(self.client, req_id, [key(i) for i in ids], [position(i) for i in ids], NOW)
        if result is None:
            return None
        self.last_stored = result[0]
        return result

    def complete_store(self):
        self.records.complete_store(self.client, self.last_stored, success=True)

    def finish(self, req_id):
        self.records.request_finished(self.client, req_id)


def _step(req_id: str, looked_up: list[int], loaded: list[int]) -> list[tuple]:
    """One scheduler step for one request: lookups, a load, the step's end, and
    the load's completion."""
    ops: list[tuple] = [("lookup", req_id, i) for i in looked_up]
    if loaded:
        ops.append(("load", req_id, loaded))
    ops.append(("end_step",))
    if loaded:
        ops.append(("complete_load", req_id, loaded))
    return ops


SCRIPT: list[tuple] = [
    ("store", "r1", [0, 1, 2, 3]),
    ("complete_store",),
    ("finish", "r1"),
    *_step("r2", [0, 1, 2], [0, 1]),
    ("finish", "r2"),
    ("store", "r3", [4, 5]),
    ("complete_store",),
    ("finish", "r3"),
    *_step("r4", [0, 1, 2, 3, 4, 5], [0]),
    ("finish", "r4"),
    ("store", "r5", [6, 7, 0]),
    ("complete_store",),
    ("finish", "r5"),
    *_step("r6", list(range(8)), []),
    ("finish", "r6"),
    ("store", "r7", [8, 9, 10, 11]),
    ("complete_store",),
    ("finish", "r7"),
    *_step("r8", list(range(12)), []),
    ("finish", "r8"),
]


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_the_same_answers_chunks_and_evictions_as_vllms_manager(cache_policy):
    vanilla = _Vanilla(4, cache_policy)
    shared = _Shared(4, cache_policy)
    for op, *args in SCRIPT:
        expected = getattr(vanilla, op)(*args)
        assert getattr(shared, op)(*args) == expected, (op, *args)
        check(shared.records)


# --- Snapshot -------------------------------------------------------------------


def test_a_snapshot_restores_lru_records():
    records = make_records(4, "lru")
    a = add_client(records, "engine-a", A)
    store(records, a, "r1", [0, 1, 2])
    records.request_finished(a, "r1")
    lookup(records, a, "r2", 1)  # a pin
    records.prepare_store(a, "r2", [key(3)], [position(3)], NOW)  # a write

    snapshot = records.snapshot(b"region-1")
    assert snapshot is not None
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(snapshot), type=Snapshot)
    restored = Records.restore(decoded, 4, LEASE, lease_until=NOW + 10)
    check(restored)
    restored_a = restored.clients["engine-a"]
    assert restored_a.pins == {A + key(1): [NOW + 10]}
    assert restored_a.writes == {A + key(3): NOW + 10}

    # The same answers and the same eviction afterwards.
    outcomes = []
    for r in (records, restored):
        client = r.clients["engine-a"]
        r.complete_store(client, [key(3)], success=True)
        stored = r.prepare_store(client, "r3", [key(4)], [position(4)], NOW)
        assert stored is not None
        outcomes.append((stored, [lookup(r, client, "r4", i) for i in range(5)]))
        check(r)
    assert outcomes[0] == outcomes[1]


def test_only_lru_records_are_snapshotted():
    assert make_records(4, "arc").snapshot(b"region-1") is None

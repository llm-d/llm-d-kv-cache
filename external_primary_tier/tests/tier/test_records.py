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
    return add_client(records, "engine-a@pod-a", A)


@pytest.fixture
def b(records) -> ClientContext:
    return add_client(records, "engine-b@pod-b", B)


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
    same_model = add_client(records, "engine-c@pod-c", A)
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
    same_model = add_client(records, "engine-c@pod-c", A)
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
    same_model = add_client(records, "engine-c@pod-c", A)
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
    assert list(records.clients) == ["engine-b@pod-b"]
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
    a = add_client(records, "engine-a@pod-a", A)
    store(records, a, "r1", [0])
    assert lookup(records, a, "r2", 0) is LookupResult.HIT
    check(records)


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
        self.client = add_client(self.records, "engine-a@pod-a", A)
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
    a = add_client(records, "engine-a@pod-a", A)
    store(records, a, "r1", [0, 1, 2])
    records.request_finished(a, "r1")
    lookup(records, a, "r2", 1)  # a pin
    records.prepare_store(a, "r2", [key(3)], [position(3)], NOW)  # a write

    snapshot = records.snapshot(b"region-1")
    assert snapshot is not None
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(snapshot), type=Snapshot)
    restored = Records.restore(decoded, 4, LEASE, lease_until=NOW + 10)
    check(restored)
    restored_a = restored.clients["engine-a@pod-a"]
    assert restored_a.pins == {A + key(1): [NOW + 10]}
    assert restored_a.writes == {A + key(3): NOW + 10}

    # The same answers and the same eviction afterwards.
    outcomes = []
    for r in (records, restored):
        client = r.clients["engine-a@pod-a"]
        r.complete_store(client, [key(3)], success=True)
        stored = r.prepare_store(client, "r3", [key(4)], [position(4)], NOW)
        assert stored is not None
        outcomes.append((stored, [lookup(r, client, "r4", i) for i in range(5)]))
        check(r)
    assert outcomes[0] == outcomes[1]


def test_only_lru_records_are_snapshotted():
    assert make_records(4, "arc").snapshot(b"region-1") is None

# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLMDKVTierManager against a real kvtierd process."""

from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import BLOCK, make_offloading_spec
from llmd_kv_tiering.vllm_tier import LLMDKVTierManager
from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    StoreTrigger,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

CTX = ReqContext(req_id="test")


def key(n: int):
    return make_offload_key(n.to_bytes(8, "big"), 0)


def primary(num_chunks: int = 8) -> np.ndarray:
    return np.zeros((num_chunks, BLOCK), dtype=np.uint8)


def fill(arr: np.ndarray, chunk: int, tag: int) -> None:
    arr[chunk] = np.frombuffer(np.random.default_rng(tag).bytes(BLOCK), dtype=np.uint8)


def expected(tag: int) -> np.ndarray:
    return np.frombuffer(np.random.default_rng(tag).bytes(BLOCK), dtype=np.uint8)


def make_tier(daemon, arr: np.ndarray, **kw) -> LLMDKVTierManager:
    kw.setdefault("timeout_s", 2.0)
    kw.setdefault("reconnect_interval_s", 0.05)
    tier = LLMDKVTierManager(
        offloading_spec=make_offloading_spec(),
        primary_kv_view=memoryview(arr),
        tier_type="LLMDKVTierManager",
        socket_path=str(daemon.socket_path),
        **kw,
    )
    return tier


@pytest.fixture
def tiers():
    made: list[LLMDKVTierManager] = []
    yield made
    for t in made:
        t.shutdown()


def job(job_id, keys, chunks, promotion=False):
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array(chunks, dtype=np.int64),
        is_promotion=promotion,
        req_context=CTX,
    )


def run(tier, j, store: bool):
    (tier.submit_store if store else tier.submit_load)(j)
    tier.drain_jobs()
    results = list(tier.get_finished_jobs())
    assert len(results) == 1 and results[0].job_id == j.job_id
    return results[0]


def wait_for(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_store_then_load_round_trip(daemon_factory, tiers):
    d = daemon_factory()
    arr = primary()
    tier = make_tier(d, arr)
    tiers.append(tier)
    assert tier.connected
    for c in range(3):
        fill(arr, c, tag=c)

    keys = [key(c) for c in range(3)]
    assert run(tier, job(1, keys, [0, 1, 2]), store=True).success
    assert all(tier.lookup(k, CTX) is LookupResult.HIT for k in keys)

    arr[:] = 0
    res = run(tier, job(2, keys, [5, 6, 7], promotion=True), store=False)
    assert res.success
    for c in range(3):
        np.testing.assert_array_equal(arr[5 + c], expected(c))

    events = list(tier.take_events())
    assert any(not e.removed and set(e.keys) == set(keys) for e in events)


def test_blocks_survive_engine_restart(daemon_factory, tiers):
    d = daemon_factory()
    arr_a = primary()
    a = make_tier(d, arr_a)
    fill(arr_a, 0, tag=42)
    assert run(a, job(1, [key(1)], [0]), store=True).success
    a.shutdown()  # the engine exits; the daemon keeps the block

    arr_b = primary()
    b = make_tier(d, arr_b)
    tiers.append(b)
    assert b.lookup(key(1), CTX) is LookupResult.HIT
    assert run(b, job(2, [key(1)], [3], promotion=True), store=False).success
    np.testing.assert_array_equal(arr_b[3], expected(42))
    # The snapshot is announced, so a KV-event consumer sees the block.
    assert any(key(1) in e.keys and not e.removed for e in b.take_events())


def test_live_engines_share_blocks(daemon_factory, tiers):
    d = daemon_factory()
    arr_a, arr_b = primary(), primary()
    a, b = make_tier(d, arr_a), make_tier(d, arr_b)
    tiers += [a, b]
    fill(arr_a, 2, tag=7)
    assert run(a, job(1, [key(9)], [2]), store=True).success
    assert wait_for(lambda: b.lookup(key(9), CTX) is LookupResult.HIT)
    assert run(b, job(2, [key(9)], [4], promotion=True), store=False).success
    np.testing.assert_array_equal(arr_b[4], expected(7))


def test_eviction_reaches_lookups_and_events(daemon_factory, tiers):
    d = daemon_factory(slots=2)
    arr = primary()
    tier = make_tier(d, arr)
    tiers.append(tier)
    for i in range(3):
        assert run(tier, job(i, [key(i)], [i]), store=True).success
    assert tier.lookup(key(0), CTX) is LookupResult.MISS
    assert tier.lookup(key(2), CTX) is LookupResult.HIT
    assert any(e.removed and key(0) in e.keys for e in tier.take_events())


def test_stale_lookup_gives_partial_load(daemon_factory, tiers):
    d = daemon_factory(slots=2)
    arr_a, arr_b = primary(), primary()
    a, b = make_tier(d, arr_a), make_tier(d, arr_b)
    tiers += [a, b]
    assert run(a, job(1, [key(1), key(2)], [0, 1]), store=True).success
    assert wait_for(lambda: b.lookup(key(1), CTX) is LookupResult.HIT)
    # b's mirror may still list key 1 after a evicts it; the load must fail
    # cleanly for that key and keep the other.
    b._mirror.add(key(1))
    assert run(a, job(2, [key(3)], [2]), store=True).success
    res = run(b, job(3, [key(1), key(2)], [4, 5], promotion=True), store=False)
    assert not res.success
    assert set(res.successful_keys) == {key(2)}


def test_daemon_down_then_up(daemon_factory, tiers):
    d = daemon_factory(start=False)
    arr = primary()
    tier = make_tier(d, arr)
    tiers.append(tier)
    assert not tier.connected
    assert tier.lookup(key(1), CTX) is LookupResult.MISS
    assert not run(tier, job(1, [key(1)], [0]), store=True).success

    d.start()
    assert wait_for(lambda: tier.connected)
    assert run(tier, job(2, [key(1)], [0]), store=True).success

    d.stop()
    assert wait_for(lambda: not tier.connected)
    assert tier.lookup(key(1), CTX) is LookupResult.MISS


def test_through_vllm_tiering_manager(daemon_factory):
    """Cascade and promotion through vLLM's own TieringOffloadingManager,
    with an engine restart in between."""
    d = daemon_factory()
    keys = [make_offload_key(str(i).encode(), 0) for i in range(3)]

    def engine():
        arr = primary(num_chunks=4)
        region = _Region(arr)
        prim = CPUPrimaryTierOffloadingManager(num_chunks=4, mmap_region=region)
        tier = make_tier(d, arr)
        mgr = TieringOffloadingManager(primary_tier=prim, secondary_tiers=[tier])
        return arr, mgr, tier

    def step(mgr):
        mgr.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
        list(mgr.take_events())

    # Engine A computes three blocks; vLLM cascades them to our tier.
    arr_a, mgr_a, tier_a = engine()
    mgr_a.on_new_request(CTX)
    out = mgr_a.prepare_store(keys, CTX, trigger=StoreTrigger.ON_COMPUTE)
    assert out is not None
    for i, cid in enumerate(out.store_spec.chunk_ids):
        fill(arr_a, int(cid), tag=100 + i)
    mgr_a.complete_store(keys, CTX, trigger=StoreTrigger.ON_COMPUTE, success=True)
    tier_a.drain_jobs()
    step(mgr_a)
    tier_a.shutdown()

    # Engine B starts cold. Its lookups promote from our tier.
    arr_b, mgr_b, tier_b = engine()
    try:
        ctx = ReqContext(req_id="after-restart")
        mgr_b.on_new_request(ctx)
        assert all(mgr_b.lookup(k, ctx) is LookupResult.HIT_PENDING for k in keys)
        assert wait_for(
            lambda: (
                (step(mgr_b), tier_b.drain_jobs(), step(mgr_b))
                and all(mgr_b.lookup(k, ctx) is LookupResult.HIT for k in keys)
            )
        )
        rows = {arr_b[c].tobytes() for c in range(4)}
        for i in range(3):
            assert expected(100 + i).tobytes() in rows
    finally:
        tier_b.shutdown()


class _Region:
    """Stands in for vLLM's SharedOffloadRegion over a numpy array."""

    def __init__(self, arr: np.ndarray) -> None:
        self._view = memoryview(arr)

    def create_kv_memoryview(self):
        return self._view

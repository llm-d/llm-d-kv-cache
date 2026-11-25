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

"""PoolOffloadingManager against a running pool server, called as vLLM's connector calls it."""

from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext

from kvpoold.adapters.vllm.manager import PoolOffloadingManager

from ...server.fake_engine import key
from .conftest import POOL_CHUNKS, kv_caches

STEP_END = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())


def store(manager: PoolOffloadingManager, ids: list[int], req_id: str = "store") -> list[int]:
    request = ReqContext(req_id=req_id)
    output = manager.prepare_store([key(i) for i in ids], request)
    assert output is not None
    manager.complete_store(output.keys_to_store, request)
    # ZMQ orders messages only within one socket. A reply on this manager's socket
    # means the server has handled the completion, so another engine's lookup that
    # follows sees the chunk as stored, not pending.
    pins(manager)
    return output.store_spec.chunk_ids.tolist()


def pins(manager: PoolOffloadingManager) -> int:
    """The pins this engine holds, as the server counts them."""
    return manager.client.stats()["client_pins"]


# --- The calls -------------------------------------------------------------------


def test_a_stored_key_is_a_hit_that_loads_from_the_chunk_it_was_stored_in(engine):
    manager = engine()
    chunk_ids = store(manager, [0, 1])

    request = ReqContext(req_id="load")
    assert [manager.lookup(key(i), request) for i in (0, 1)] == [LookupResult.HIT, LookupResult.HIT]
    spec = manager.prepare_load([key(0), key(1)], request)
    manager.on_schedule_end(STEP_END)

    assert spec.chunk_ids.tolist() == chunk_ids
    assert pins(manager) == 2
    manager.complete_load([key(0), key(1)], request)
    assert pins(manager) == 0


def test_a_key_being_stored_is_a_pending_hit(engine):
    manager = engine()
    assert manager.prepare_store([key(0)], ReqContext(req_id="store")) is not None
    assert manager.lookup(key(0), ReqContext(req_id="load")) is LookupResult.HIT_PENDING


def test_another_engine_with_the_same_model_hits_a_stored_key(engine):
    a = engine("engine-a")
    b = engine("engine-b")
    store(a, [0])
    assert b.lookup(key(0), ReqContext(req_id="load")) is LookupResult.HIT


def test_an_engine_with_another_model_misses_a_stored_key(engine):
    a = engine("engine-a", model_name="model-a")
    b = engine("engine-b", model_name="model-b")
    store(a, [0])
    assert b.lookup(key(0), ReqContext(req_id="load")) is LookupResult.MISS


def test_a_full_pool_of_pinned_chunks_skips_the_store(engine):
    manager = engine()
    store(manager, list(range(POOL_CHUNKS)))
    request = ReqContext(req_id="load")
    for i in range(POOL_CHUNKS):
        manager.lookup(key(i), request)

    assert manager.prepare_store([key(100)], ReqContext(req_id="store")) is None


# --- Releasing the hits of a step --------------------------------------------------


def test_hits_not_loaded_are_released_at_the_end_of_the_step(engine):
    manager = engine()
    store(manager, [0, 1, 2])
    request = ReqContext(req_id="load")
    for i in (0, 1, 2):
        manager.lookup(key(i), request)
    # A partial load: only the first key's slot was allocated.
    manager.prepare_load([key(0)], request)
    assert pins(manager) == 3

    manager.on_schedule_end(STEP_END)
    assert pins(manager) == 1

    manager.complete_load([key(0)], request)
    assert pins(manager) == 0


def test_a_key_looked_up_twice_in_a_step_is_released_twice(engine):
    manager = engine()
    store(manager, [0])
    # Two requests share the prefix; neither is scheduled this step.
    manager.lookup(key(0), ReqContext(req_id="a"))
    manager.lookup(key(0), ReqContext(req_id="b"))
    assert pins(manager) == 2

    manager.on_schedule_end(STEP_END)
    assert pins(manager) == 0


# --- Without the server ---------------------------------------------------------------


def test_an_engine_the_server_refused_misses_every_lookup_and_skips_every_store(make_spec):
    spec = make_spec()
    # No worker attached first, so the server refuses the scheduler.
    manager = spec.get_manager()
    try:
        assert manager.lookup(key(0), ReqContext(req_id="load")) is LookupResult.MISS
        assert manager.prepare_store([key(0)], ReqContext(req_id="store")) is None
    finally:
        manager.shutdown()


# --- Leaving -------------------------------------------------------------------------


def test_shutdown_releases_the_engines_pins_at_once(make_spec, engine):
    spec = make_spec(engine_id="engine-a")
    spec.create_worker(kv_caches()).shutdown()
    a = spec.get_manager()
    b = engine("engine-b")
    store(a, [0])
    a.lookup(key(0), ReqContext(req_id="load"))
    assert b.client.stats()["num_pinned_chunks"] == 1

    a.shutdown()

    assert b.client.stats()["num_pinned_chunks"] == 0

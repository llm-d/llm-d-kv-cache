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

"""Serving engines over ZMQ: lookups, loads, stores, deadlines, leaving and draining."""

import pytest
from external_primary_tier.protocol import (
    AttachOk,
    CompleteLoad,
    Draining,
    GetStats,
    Leave,
    LoadSpec,
    Lookup,
    NoSpace,
    PrepareLoad,
    PrepareStore,
    Refused,
    ReleaseHits,
    RequestFinished,
    UnknownClient,
)
from external_primary_tier.tier.server import ExternalPrimaryTier

from .conftest import LEASE_SECONDS, POOL_CHUNKS
from .fake_engine import KV_BYTES_PER_CHUNK, FakeEngineProcess, key
from .helpers import lookup, stats, store


def test_a_stored_key_is_a_hit_that_loads_from_its_chunk(engine):
    scheduler = engine()
    stored = store(scheduler, "r1", [0, 1])
    assert stored.keys_to_store == [key(0), key(1)]

    assert [lookup(scheduler, "r2", i) for i in (0, 1)] == ["hit", "hit"]
    loaded = scheduler.call(PrepareLoad, req_id="r2", keys=[key(0), key(1)], positions=[None, None])
    assert isinstance(loaded, LoadSpec)
    assert loaded.chunk_ids == stored.chunk_ids
    assert stats(scheduler)["client_pins"] == 2

    scheduler.send(CompleteLoad(keys=[key(0), key(1)]))
    scheduler.send(RequestFinished(req_id="r2"))
    values = stats(scheduler)
    assert values["client_pins"] == 0
    assert values["num_evictable_chunks"] == 2


def test_a_key_being_stored_is_a_pending_hit(engine):
    scheduler = engine()
    scheduler.call(PrepareStore, req_id="r1", keys=[key(0)], positions=[None])
    assert lookup(scheduler, "r2", 0) == "hit_pending"


def test_hits_not_loaded_are_released_at_the_end_of_the_step(engine):
    scheduler = engine()
    store(scheduler, "r1", [0])
    lookup(scheduler, "r2", 0)
    lookup(scheduler, "r3", 0)
    assert stats(scheduler)["client_pins"] == 2
    scheduler.send(ReleaseHits(hits=[(key(0), 2)]))
    assert stats(scheduler)["client_pins"] == 0


def test_engines_with_the_same_compat_id_share_hits(engine):
    a = engine("engine-a@pod-a")
    b = engine("engine-b@pod-b")
    store(a, "r1", [0])
    assert lookup(b, "r2", 0) == "hit"


def test_engines_with_another_compat_id_miss(engine):
    a = engine("engine-a@pod-a")
    b = engine("engine-b@pod-b", compat_id=b"\x22" * 8)
    store(a, "r1", [0])
    assert lookup(b, "r2", 0) == "miss"


def test_a_store_while_every_chunk_is_pinned_gets_no_space(engine):
    scheduler = engine()
    store(scheduler, "r1", list(range(POOL_CHUNKS)))
    assert [lookup(scheduler, "r2", i) for i in range(POOL_CHUNKS)] == ["hit"] * POOL_CHUNKS
    reply = scheduler.call(PrepareStore, req_id="r3", keys=[key(100)], positions=[None])
    assert isinstance(reply, NoSpace)


def test_a_load_of_a_key_that_is_not_pinned_is_refused(engine):
    scheduler = engine()
    store(scheduler, "r1", [0])
    assert isinstance(scheduler.call(PrepareLoad, req_id="r2", keys=[key(0)], positions=[None]), Refused)


def test_leases_are_released_at_their_deadlines(engine, clock):
    scheduler = engine()
    store(scheduler, "r1", [0])
    lookup(scheduler, "r2", 0)
    scheduler.call(PrepareStore, req_id="r3", keys=[key(1)], positions=[None])

    clock.advance(LEASE_SECONDS - 1)
    values = stats(scheduler)
    assert (values["client_pins"], values["client_writes"]) == (1, 1)

    clock.advance(1)
    values = stats(scheduler)
    assert (values["client_pins"], values["client_writes"]) == (0, 0)
    assert values["num_free_chunks"] == POOL_CHUNKS - 1


def test_an_engine_that_leaves_releases_its_leases(engine):
    a = engine("engine-a@pod-a")
    b = engine("engine-b@pod-b")
    store(a, "r1", [0])
    lookup(a, "r2", 0)
    a.call(PrepareStore, req_id="r3", keys=[key(1)], positions=[None])

    a.send(Leave())
    assert isinstance(a.call(GetStats), UnknownClient)
    values = stats(b)
    assert values["num_pinned_chunks"] == 0
    assert values["num_write_pending_chunks"] == 0
    assert values["num_clients"] == 1


def test_notices_from_a_process_that_has_not_attached_are_dropped(engine, engine_process):
    scheduler = engine()
    store(scheduler, "r1", [0])
    lookup(scheduler, "r2", 0)
    stranger = engine_process()
    stranger.send(ReleaseHits(hits=[(key(0), 1)]))
    assert isinstance(stranger.call(GetStats), UnknownClient)
    assert stats(scheduler)["client_pins"] == 1


def test_a_draining_tier_answers_requests_with_draining(tmp_path, clock):
    address = f"ipc://{tmp_path}/tier.sock"
    tier = ExternalPrimaryTier(address, str(tmp_path), pool_bytes=POOL_CHUNKS * KV_BYTES_PER_CHUNK, clock=clock)
    process = FakeEngineProcess(address)
    try:
        process.send(process.attach_message("engine-a@pod-a", rank=0))
        assert tier.poll(5000) == 1
        assert isinstance(process.receive(), AttachOk)

        process.send(Lookup(call_id=process.next_call_id(), req_id="r1", key=key(0)))
        assert tier.drain(5000) == 1
        assert isinstance(process.receive(), Draining)
    finally:
        process.close()
        tier.close()


def test_an_unknown_cache_policy_fails_at_start(tmp_path):
    with pytest.raises(ValueError):
        ExternalPrimaryTier(
            f"ipc://{tmp_path}/tier.sock",
            str(tmp_path),
            pool_bytes=POOL_CHUNKS * KV_BYTES_PER_CHUNK,
            cache_policy="no-such-policy",
        )

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

"""Restarts: the region's files outlive the External Primary Tier; its records
survive a clean stop and are fenced off after a crash."""

import os

from external_primary_tier.protocol import (
    AttachOk,
    LoadSpec,
    NoSpace,
    PrepareLoad,
    PrepareStore,
    StoreSpec,
)
from external_primary_tier.tier.region import REGION_FILE_NAME
from external_primary_tier.tier.snapshot import SNAPSHOT_FILE_NAME, SPEC_FILE_NAME

from .conftest import FENCE_SECONDS
from .fake_engine import key
from .helpers import lookup, stats, store


def region_of(engine_process, client_id="engine-z@pod-z"):
    reply = engine_process().attach(client_id, rank=0)
    assert isinstance(reply, AttachOk)
    return reply.region


def test_a_clean_restart_keeps_the_region_and_the_records(tier, start_tier, engine, engine_process):
    scheduler = engine()
    stored = store(scheduler, "r1", [0, 1])
    region = region_of(engine_process)
    tier.stop()

    start_tier()
    again = engine()
    assert region_of(engine_process) == region
    assert lookup(again, "r2", 0) == "hit"
    loaded = again.call(PrepareLoad, req_id="r2", keys=[key(0)], positions=[None])
    assert isinstance(loaded, LoadSpec)
    assert loaded.chunk_ids == stored.chunk_ids[:1]


def test_leases_in_the_snapshot_are_released_when_the_fence_ends(tier, start_tier, engine, clock):
    scheduler = engine()
    store(scheduler, "r1", [0])
    lookup(scheduler, "r2", 0)
    scheduler.call(PrepareStore, req_id="r3", keys=[key(1)], positions=[None])
    tier.stop()

    start_tier()
    again = engine()
    values = stats(again)
    assert (values["num_pinned_chunks"], values["num_write_pending_chunks"]) == (1, 1)
    assert (values["client_pins"], values["client_writes"]) == (1, 1)

    clock.advance(FENCE_SECONDS)
    values = stats(again)
    assert (values["num_pinned_chunks"], values["num_write_pending_chunks"]) == (0, 0)


def test_after_a_crash_the_region_is_kept_and_fenced(tier, start_tier, engine, engine_process, clock, tmp_path):
    scheduler = engine()
    store(scheduler, "r1", [0])
    region = region_of(engine_process)
    tier.stop()
    os.unlink(tmp_path / SNAPSHOT_FILE_NAME)  # as if the instance had crashed

    start_tier()
    again = engine()
    assert region_of(engine_process) == region
    assert lookup(again, "r2", 0) == "miss"
    assert isinstance(again.call(PrepareStore, req_id="r2", keys=[key(1)], positions=[None]), NoSpace)

    clock.advance(FENCE_SECONDS)
    assert lookup(again, "r3", 0) == "miss"
    assert isinstance(again.call(PrepareStore, req_id="r3", keys=[key(1)], positions=[None]), StoreSpec)


def test_the_snapshot_is_read_once(tier, start_tier, engine, tmp_path):
    store(engine(), "r1", [0])
    tier.stop()
    start_tier()
    assert not (tmp_path / SNAPSHOT_FILE_NAME).exists()


def test_region_files_that_do_not_match_their_spec_are_replaced(tier, start_tier, engine_process):
    region = region_of(engine_process)
    tier.stop()
    os.truncate(region.segments[0][0], 0)

    start_tier()
    replaced = region_of(engine_process)
    assert replaced.region_id != region.region_id
    for path, _, length in replaced.segments:
        assert os.stat(path).st_size == length


def test_region_files_without_a_spec_are_removed(start_tier, tmp_path):
    stray = tmp_path / REGION_FILE_NAME
    stray.write_bytes(b"left by a creation that never finished")
    start_tier()
    assert not stray.exists()
    assert not (tmp_path / SPEC_FILE_NAME).exists()

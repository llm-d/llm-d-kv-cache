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

"""Attach: the first worker creates the region; later processes must match it."""

import os
import stat

import pytest
from external_primary_tier.protocol import AttachOk, Lookup, Refused, ShmFileRegion, UnknownClient

from .conftest import POOL_CHUNKS
from .fake_engine import KV_BYTES_PER_CHUNK


def test_first_worker_attach_creates_the_region(engine_process):
    reply = engine_process().attach("engine-a@pod-a", rank=0)

    assert isinstance(reply, AttachOk)
    region = reply.region
    assert isinstance(region, ShmFileRegion)
    assert region.num_chunks == POOL_CHUNKS
    assert region.kv_bytes_per_chunk == KV_BYTES_PER_CHUNK
    assert sum(length for _, _, length in region.segments) == POOL_CHUNKS * KV_BYTES_PER_CHUNK
    for path, _, length in region.segments:
        info = os.stat(path)
        assert info.st_size == length
        assert stat.S_IMODE(info.st_mode) == 0o660


def test_other_processes_get_the_same_region(engine_process):
    first = engine_process().attach("engine-a@pod-a", rank=0)
    second = engine_process().attach("engine-a@pod-a", rank=1)
    other_engine = engine_process().attach("engine-b@pod-b", rank=0)

    assert first.region == second.region == other_engine.region


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_bytes_per_chunk": 2 * KV_BYTES_PER_CHUNK},
        {"hash_seed": "1"},
        {"hash_function": "sha256_cbor"},
        {"version": 999},
    ],
    ids=["kv_bytes_per_chunk", "hash_seed", "hash_function", "version"],
)
def test_a_layout_that_differs_from_the_pool_is_refused(engine_process, overrides):
    assert isinstance(engine_process().attach("engine-a@pod-a", rank=0), AttachOk)
    reply = engine_process().attach("engine-b@pod-b", rank=0, **overrides)
    assert isinstance(reply, Refused)


def test_a_scheduler_before_its_workers_is_refused(engine_process):
    reply = engine_process().attach("engine-a@pod-a", rank=None)
    assert isinstance(reply, Refused)


def test_a_scheduler_after_its_workers_gets_the_region(engine_process):
    worker = engine_process().attach("engine-a@pod-a", rank=0)
    scheduler = engine_process().attach("engine-a@pod-a", rank=None)
    assert isinstance(scheduler, AttachOk)
    assert scheduler.region == worker.region


def test_a_process_with_a_different_compat_id_is_refused(engine_process):
    engine_process().attach("engine-a@pod-a", rank=0)
    reply = engine_process().attach("engine-a@pod-a", rank=None, compat_id=b"\x22" * 8)
    assert isinstance(reply, Refused)


def test_a_request_from_a_process_that_has_not_attached_gets_unknown_client(engine_process):
    process = engine_process()
    call_id = process.next_call_id()
    reply = process.request(Lookup(call_id=call_id, req_id="req-1", key=b"k" * 36))
    assert isinstance(reply, UnknownClient)
    assert reply.call_id == call_id

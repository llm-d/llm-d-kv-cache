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

"""ExternalOffloadingSpec against a running External Primary Tier."""

import os

import pytest
from external_primary_tier.client.session import Session
from external_primary_tier.client.spec import ExternalOffloadingSpec, _DisabledWorker
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

from .conftest import KV_BYTES_PER_BLOCK, POOL_CHUNKS, kv_caches, offloading_config


def attach(spec: ExternalOffloadingSpec, rank: int | None = 0):
    session = Session(spec.tier_address)
    try:
        return spec.attach(session, rank)
    finally:
        session.close()


# --- create_worker ------------------------------------------------------------


def test_a_worker_maps_the_tiers_region_and_runs_vllms_worker_on_it(make_spec):
    spec = make_spec()
    worker = spec.create_worker(kv_caches())
    try:
        assert isinstance(worker, CPUOffloadingWorker)
        region = worker._mmap_region
        assert region.num_chunks == POOL_CHUNKS
        # The tier created the files; the worker only joined them.
        assert not region._creator
    finally:
        worker.shutdown()


def test_a_worker_of_another_chunk_size_is_refused_and_serves_without_the_pool(make_spec):
    make_spec(engine_id="engine-a").create_worker(kv_caches()).shutdown()

    wider = make_spec(engine_id="engine-b", kv_bytes_per_block=2 * KV_BYTES_PER_BLOCK)
    assert isinstance(wider.create_worker(kv_caches(2 * KV_BYTES_PER_BLOCK)), _DisabledWorker)


def test_a_worker_that_cannot_reach_the_tier_serves_without_the_pool(tmp_path):
    spec = ExternalOffloadingSpec(
        offloading_config(master_address=f"ipc://{tmp_path}/no-tier.sock", attach_timeout_ms=200)
    )
    assert isinstance(spec.create_worker(kv_caches()), _DisabledWorker)


def test_a_worker_whose_region_files_are_missing_serves_without_the_pool(make_spec):
    spec = make_spec()
    descriptor = attach(spec)
    for path, _, _ in descriptor.segments:
        os.unlink(path)

    assert isinstance(spec.create_worker(kv_caches()), _DisabledWorker)
    # A worker never creates the region itself.
    assert not any(os.path.exists(path) for path, _, _ in descriptor.segments)


def test_a_worker_that_created_the_region_after_its_files_vanished_removes_it(make_spec, monkeypatch):
    spec = make_spec()
    descriptor = attach(spec)
    # The files pass the size check, then vanish before SharedOffloadRegion opens them.
    original_init = SharedOffloadRegion.__init__

    def init_after_the_files_vanished(self, *args, **kwargs):
        for path, _, _ in descriptor.segments:
            os.unlink(path)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(SharedOffloadRegion, "__init__", init_after_the_files_vanished)

    assert isinstance(spec.create_worker(kv_caches()), _DisabledWorker)
    assert not any(os.path.exists(path) for path, _, _ in descriptor.segments)


def test_a_worker_whose_region_file_has_the_wrong_size_serves_without_the_pool(make_spec, monkeypatch):
    spec = make_spec()
    path, _, length = attach(spec).segments[0]
    os.truncate(path, length // 2)

    mapped = []
    monkeypatch.setattr(SharedOffloadRegion, "__init__", lambda self, *args, **kwargs: mapped.append(True))

    assert isinstance(spec.create_worker(kv_caches()), _DisabledWorker)
    # Refused before mapping: a copy into the missing half would kill the process with SIGBUS.
    assert not mapped


def test_a_worker_refuses_a_region_split_into_other_files_than_its_vllm_makes(make_spec, monkeypatch):
    spec = make_spec()
    descriptor = attach(spec)
    # As if this vLLM split the region into one more file than the tier's did.
    original_init = SharedOffloadRegion.__init__

    def init_with_another_split(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._segments = [*self._segments, ("/elsewhere", 0, 0)]

    monkeypatch.setattr(SharedOffloadRegion, "__init__", init_with_another_split)

    assert isinstance(spec.create_worker(kv_caches()), _DisabledWorker)
    # The tier's files are left as they were.
    assert all(os.path.exists(path) for path, _, _ in descriptor.segments)


def test_a_canonical_layout_without_a_mapping_for_every_layer_fails_loudly(make_spec):
    spec = make_spec(canonical_layout=True)
    with pytest.raises(RuntimeError, match="could not be certified"):
        spec.create_worker(kv_caches())


# --- The disabled worker --------------------------------------------------------


def test_the_disabled_worker_accepts_every_job_and_reports_it_failed():
    worker = _DisabledWorker()
    assert worker.submit_store(1, None, None)
    assert worker.submit_load(2, None, None)
    worker.wait({1})

    assert sorted((r.job_id, r.success) for r in worker.get_finished()) == [(1, False), (2, False)]
    assert worker.get_finished() == []


# --- attach ---------------------------------------------------------------------


def test_the_scheduler_attaches_after_its_workers_and_gets_the_same_region(make_spec):
    spec = make_spec()
    worker_region = attach(spec, rank=0)
    assert attach(spec, rank=None) == worker_region


def test_a_scheduler_with_no_attached_worker_is_refused(make_spec):
    assert attach(make_spec(), rank=None) is None


def test_the_client_id_names_the_engine_and_its_host(make_spec, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "pod-a")
    spec = make_spec(engine_id="engine-7")
    sent = []
    monkeypatch.setattr(Session, "call", lambda self, request_type, timeout_ms, **fields: sent.append(fields))

    attach(spec)

    assert sent[0]["client_id"] == "engine-7@pod-a"


# --- Compatibility id ------------------------------------------------------------


def test_engines_with_the_same_model_and_layout_share_a_compatibility_id(make_spec):
    assert make_spec(engine_id="engine-a")._compat_id == make_spec(engine_id="engine-b")._compat_id


def test_engines_with_another_model_have_another_compatibility_id(make_spec):
    assert make_spec(model_name="model-a")._compat_id != make_spec(model_name="model-b")._compat_id


def test_the_tp_size_is_in_the_compatibility_id_only_without_a_parallelism_agnostic_layout(make_spec):
    def compat_id(tp_size: int, agnostic: bool) -> bytes:
        return make_spec(tp_size=tp_size, is_parallelism_agnostic=agnostic)._compat_id

    assert compat_id(2, agnostic=True) == compat_id(4, agnostic=True)
    assert compat_id(2, agnostic=False) != compat_id(4, agnostic=False)

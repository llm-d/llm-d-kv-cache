# Copyright 2026 The llm-d Authors.
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

"""The store and load handlers share one engine completion queue.

Whichever handler vLLM polls first drains jobs of both directions, so each
TransferResult must carry the direction its job was submitted with, not the
direction of the handler that happened to poll it.
"""

import importlib.util
import logging
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_cuda_required

CONNECTOR_ROOT = Path(__file__).resolve().parents[2]

PUT = ("GPU", "SHARED_STORAGE")
GET = ("SHARED_STORAGE", "GPU")


@dataclass
class TransferResult:
    job_id: int
    success: bool
    transfer_size: int | None = None
    transfer_time: float | None = None
    transfer_type: tuple[str, str] | None = None


class OffloadingHandler:
    pass


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(mod, attr_name, value)
    return mod


def _package(name, **attrs):
    mod = _module(name, **attrs)
    mod.__path__ = []
    return mod


@contextmanager
def _temporary_modules(modules):
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, mod in previous.items():
            if mod is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _load_worker_module():
    stubs = {
        "storage_offload": _module("storage_offload"),
        "llmd_fs_backend": _package(
            "llmd_fs_backend", _logger=logging.getLogger("llmd_fs_backend_test")
        ),
        "llmd_fs_backend.file_mapper": _module(
            "llmd_fs_backend.file_mapper", FileMapper=object
        ),
        "llmd_fs_backend.mediums": _module(
            "llmd_fs_backend.mediums", SharedStorageLoadStoreSpec=object
        ),
        "vllm": _package("vllm"),
        "vllm.v1": _package("vllm.v1"),
        "vllm.v1.kv_offload": _package("vllm.v1.kv_offload"),
        "vllm.v1.kv_offload.base": _module(
            "vllm.v1.kv_offload.base",
            CanonicalKVCaches=object,
            GPULoadStoreSpec=object,
            get_offload_group_idx=lambda key: 0,
        ),
        "vllm.v1.kv_offload.worker": _package("vllm.v1.kv_offload.worker"),
        "vllm.v1.kv_offload.worker.worker": _module(
            "vllm.v1.kv_offload.worker.worker",
            OffloadingHandler=OffloadingHandler,
            TransferResult=TransferResult,
            TransferSpec=tuple,
            TransferType=tuple,
        ),
    }
    with _temporary_modules(stubs):
        spec = importlib.util.spec_from_file_location(
            "llmd_fs_backend_worker_under_test",
            CONNECTOR_ROOT / "llmd_fs_backend" / "worker.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


worker_module = _load_worker_module()


class FakeEngine:
    """Single completion queue shared by both directions, like the C++ engine."""

    def __init__(self):
        self.finished: list[tuple[int, bool]] = []

    def get_finished(self):
        done, self.finished = self.finished, []
        return done

    def wait_job(self, job_id):
        pass

    def shutdown(self):
        pass


class FakeEngineHandlers(worker_module.StorageOffloadingHandlers):
    def _create_engine(self, **kwargs):
        return FakeEngine()


def _make_handlers():
    ref = types.SimpleNamespace(tensor_idx=0, page_size_bytes=1024)
    kv_caches = types.SimpleNamespace(
        tensors=[types.SimpleNamespace(tensor=object())],
        group_data_refs=[[ref]],
    )
    return FakeEngineHandlers(
        kv_caches=kv_caches,
        file_mapper=object(),
        gpu_block_size=16,
        gpu_blocks_per_file=4,
        threads_per_gpu=2,
    )


def _poll_like_vllm(handlers):
    # vLLM's OffloadingWorker.get_finished polls handlers in registration
    # order; the spec registers the store handler before the load handler.
    results = []
    for handler in (handlers.gpu_to_storage_handler, handlers.storage_to_gpu_handler):
        results.extend(handler.get_finished())
    return {r.job_id: r for r in results}


def test_load_result_keeps_load_direction_when_store_handler_polls_first():
    handlers = _make_handlers()
    handlers.storage_to_gpu_handler._record_job(7, 4096)
    handlers.engine.finished.append((7, True))

    results = _poll_like_vllm(handlers)

    assert results[7].transfer_type == GET
    assert results[7].transfer_size == 4096


def test_mixed_completions_are_labeled_by_submitting_handler():
    handlers = _make_handlers()
    handlers.gpu_to_storage_handler._record_job(1, 1000)
    handlers.storage_to_gpu_handler._record_job(2, 2000)
    handlers.gpu_to_storage_handler._record_job(3, 3000)
    handlers.engine.finished.extend([(2, True), (1, True), (3, True)])

    results = _poll_like_vllm(handlers)

    assert {job_id: r.transfer_type for job_id, r in results.items()} == {
        1: PUT,
        2: GET,
        3: PUT,
    }
    assert {job_id: r.transfer_size for job_id, r in results.items()} == {
        1: 1000,
        2: 2000,
        3: 3000,
    }
    assert handlers.gpu_to_storage_handler._pending_jobs == {}


def test_store_result_keeps_store_direction_when_load_handler_polls_it():
    handlers = _make_handlers()
    handlers.gpu_to_storage_handler._record_job(5, 512)
    handlers.engine.finished.append((5, True))

    (result,) = handlers.storage_to_gpu_handler.get_finished()

    assert result.transfer_type == PUT
    assert result.transfer_time is not None and result.transfer_time >= 0


def test_unknown_job_is_reported_without_metrics():
    handlers = _make_handlers()
    handlers.engine.finished.append((99, False))

    (result,) = handlers.gpu_to_storage_handler.get_finished()

    assert result.job_id == 99
    assert result.success is False
    assert result.transfer_type is None
    assert result.transfer_size is None

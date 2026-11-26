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

"""Contract between the shared-storage handlers and the NIXL engine.

The handlers in llmd_fs_backend/worker.py call the engine through the
StorageEngine protocol. These tests drive the real handlers against the real
llmd_nixl.nixl_offload.StorageOffloadEngine, with nixl, torch and vllm stubbed.
The OBJ backend's constructor validation is exercised the same way.
"""

import inspect
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_cuda_required

CONNECTOR_ROOT = Path(__file__).resolve().parents[2]


class GPULoadStoreSpec:
    def __init__(self, block_ids, group_sizes, block_indices):
        self.block_ids = block_ids
        self.group_sizes = group_sizes
        self.block_indices = block_indices


class SharedStorageLoadStoreSpec:
    def __init__(self, keys):
        self.keys = keys


class StagedBackend:
    def __init__(self, io_threads, gpu_blocks_per_file, tensors, backend):
        pass


class FakeFileMapper:
    def get_file_name(self, key):
        return f"file-{key}"


@contextmanager
def temporary_modules(modules):
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def module(name, **attrs):
    mod = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(mod, attr_name, value)
    return mod


def package(name):
    mod = module(name)
    mod.__path__ = []
    return mod


def load_module(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_engine_modules():
    stubs = {
        "storage_offload": module("storage_offload"),
        "torch": module("torch", Tensor=object),
        "nixl": package("nixl"),
        "nixl._api": module("nixl._api", nixl_agent=object, nixl_agent_config=object),
        "nixl.logging": module(
            "nixl.logging", get_logger=lambda name: logging.getLogger(name)
        ),
        "vllm": package("vllm"),
        "vllm.v1": package("vllm.v1"),
        "vllm.v1.kv_offload": package("vllm.v1.kv_offload"),
        "vllm.v1.kv_offload.base": module(
            "vllm.v1.kv_offload.base",
            CanonicalKVCaches=object,
            GPULoadStoreSpec=GPULoadStoreSpec,
            get_offload_group_idx=lambda key: 0,
        ),
        "vllm.v1.kv_offload.worker": package("vllm.v1.kv_offload.worker"),
        "vllm.v1.kv_offload.worker.worker": module(
            "vllm.v1.kv_offload.worker.worker",
            OffloadingHandler=object,
            TransferResult=tuple,
            TransferSpec=tuple,
            TransferType=tuple,
        ),
        "llmd_fs_backend": module(
            "llmd_fs_backend", _logger=logging.getLogger("llmd_fs_backend")
        ),
        "llmd_fs_backend.file_mapper": module(
            "llmd_fs_backend.file_mapper", FileMapper=object
        ),
        "llmd_fs_backend.mediums": module(
            "llmd_fs_backend.mediums",
            SharedStorageLoadStoreSpec=SharedStorageLoadStoreSpec,
        ),
        "llmd_nixl": package("llmd_nixl"),
        "llmd_nixl.staged_backend": module(
            "llmd_nixl.staged_backend", _StagedBackend=StagedBackend
        ),
    }
    stubs["llmd_fs_backend"].__path__ = []
    loaded_names = [
        "llmd_fs_backend.worker",
        "llmd_nixl.nixl_offload",
        "llmd_nixl.obj_backend",
    ]
    sentinel = object()
    previous_loaded = {name: sys.modules.get(name, sentinel) for name in loaded_names}

    with temporary_modules(stubs):
        worker = load_module(
            "llmd_fs_backend.worker",
            CONNECTOR_ROOT / "llmd_fs_backend" / "worker.py",
        )
        nixl_offload = load_module(
            "llmd_nixl.nixl_offload",
            CONNECTOR_ROOT / "llmd_nixl" / "nixl_offload.py",
        )
        obj_backend = load_module(
            "llmd_nixl.obj_backend",
            CONNECTOR_ROOT / "llmd_nixl" / "obj_backend.py",
        )

    for name, previous in previous_loaded.items():
        if previous is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    return worker, nixl_offload, obj_backend


worker_module, nixl_offload_module, obj_backend_module = load_engine_modules()


class RecordingEngine(nixl_offload_module.StorageOffloadEngine):
    """Real NIXL engine entry points with the backend-specific hooks recorded."""

    def __init__(self):
        self.logger = logging.getLogger("recording-engine")
        self._transfers = []
        self.submitted = []

    def _store_gpu_blocks_to_staging(self, block_ids):
        return block_ids, None

    def _reserve_staging_for_load(self, block_ids):
        return block_ids, None

    def _submit_transfer(self, job_id, tensors, stagings, files, block_ids, op):
        self.submitted.append((job_id, files, block_ids, op))
        return True

    def _backend_params(self):
        return {}

    def _get_blocks_data(self, tensors, block_ids):
        return []

    def _open_files(self, files):
        return files

    def _build_nixl_file_entries(self, fd_list, file_idx, block_list):
        return []

    def _close_fds(self, fd_list):
        pass

    def _load_staging_to_gpu_blocks(self, stagings, block_ids):
        pass

    def _shutdown_backend(self):
        pass


def _handler(handler_cls, engine):
    return handler_cls(
        gpu_blocks_per_file=2,
        file_mapper=FakeFileMapper(),
        engine=engine,
        transfer_type=("GPU", "SHARED_STORAGE"),
        per_group_block_bytes=[1024],
    )


@pytest.mark.parametrize("method", ["async_store_gpu_blocks", "async_load_gpu_blocks"])
def test_nixl_engine_signature_matches_storage_engine_protocol(method):
    """Verify the NIXL engine takes the same parameters as the protocol."""
    expected = inspect.signature(getattr(worker_module.StorageEngine, method))
    actual = inspect.signature(
        getattr(nixl_offload_module.StorageOffloadEngine, method)
    )

    assert list(actual.parameters) == list(expected.parameters)


def test_gpu_to_storage_handler_drives_nixl_engine():
    """Verify GPUToStorageHandler can submit a PUT through the NIXL engine."""
    engine = RecordingEngine()
    handler = _handler(worker_module.GPUToStorageHandler, engine)
    gpu_spec = GPULoadStoreSpec([5, 6, 7], group_sizes=[3], block_indices=[0])
    storage_spec = SharedStorageLoadStoreSpec(["a", "b"])

    assert handler.transfer_async(1, (gpu_spec, storage_spec)) is True

    assert engine.submitted == [(1, ["file-a", "file-b"], [[5, 6], [7]], "WRITE")]


def test_storage_to_gpu_handler_drives_nixl_engine():
    """Verify StorageToGPUHandler can submit a GET through the NIXL engine."""
    engine = RecordingEngine()
    handler = _handler(worker_module.StorageToGPUHandler, engine)
    gpu_spec = GPULoadStoreSpec([5, 6, 7], group_sizes=[3], block_indices=[1])
    storage_spec = SharedStorageLoadStoreSpec(["a", "b"])

    assert handler.transfer_async(2, (storage_spec, gpu_spec)) is True

    assert engine.submitted == [(2, ["file-a", "file-b"], [[5], [6, 7]], "READ")]


OBJ_CONFIG = {
    "bucket": "kv-cache",
    "endpoint_override": "minio:9000",
    "access_key": "access",
    "secret_key": "secret",
}


def _obj_backend(num_groups, extra_config):
    return obj_backend_module.ObjBackend(
        io_threads=1,
        gpu_blocks_per_file=2,
        tensors=[],
        group_tensor_indices=[[0] for _ in range(num_groups)],
        extra_config=extra_config,
    )


def test_obj_backend_rejects_multiple_kv_cache_groups():
    """Verify ObjBackend refuses hybrid models instead of mis-sizing transfers."""
    with pytest.raises(ValueError, match="single KV cache group, got 2"):
        _obj_backend(num_groups=2, extra_config=OBJ_CONFIG)

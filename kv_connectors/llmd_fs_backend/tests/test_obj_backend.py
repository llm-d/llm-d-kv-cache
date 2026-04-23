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

"""
Integration tests for the OBJ storage backend.

Assumes the object store bucket already exists and is reachable.
Credentials are supplied via environment variables or pytest CLI options:

  OBJ_ENDPOINT   (or --obj-endpoint)   e.g. "minio.example.com:9000"
  OBJ_BUCKET     (or --obj-bucket)     e.g. "kv-cache"
  OBJ_ACCESS_KEY (or --obj-access-key) e.g. "minioadmin"
  OBJ_SECRET_KEY (or --obj-secret-key) e.g. "minioadmin"
  OBJ_SCHEME     (or --obj-scheme)     "http" or "https" (default: "http")
  OBJ_CA_BUNDLE  (or --obj-ca-bundle)  path to CA cert file (optional)

Run:
  pytest tests/test_obj_backend.py \
      --obj-endpoint minio:9000 \
      --obj-bucket kv-cache \
      --obj-access-key minioadmin \
      --obj-secret-key minioadmin
"""

import math
import os
import time

import pytest
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import StorageOffloadingHandlers
from llmd_nixl.nixl_lookup import NixlLookup
from tests.test_fs_backend import (
    make_gpu_specs,
    make_storage_specs,
    throughput_gbps,
    total_block_size_mb,
    wait_for,
)


def create_cross_layer_kv_tensor(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Create a single cross-layer KV cache tensor with shape
    (num_blocks, num_layers, 2, block_size, num_heads, head_size).

    num_blocks must be the leading dimension so that tensors[0][block_id]
    gives per-block data and stride(0) gives the per-block byte size,
    as required by StorageOffloadEngine (nixl_offload)."""
    torch.manual_seed(seed)
    shape = (num_blocks, num_layers, 2, block_size, num_heads, head_size)
    return torch.rand(shape, dtype=dtype, device="cuda")


def assert_blocks_equal_cross(
    original: torch.Tensor,
    restored: torch.Tensor,
    block_ids: list[int],
) -> None:
    """Assert that restored cross-layer blocks match the originals.
    Tensors have shape (num_blocks, num_layers, 2, ...)."""
    for b in block_ids:
        torch.testing.assert_close(original[int(b)], restored[int(b)])


# ---------------------------------------------------------------------------
# pytest CLI options (registered in conftest.py)
# ---------------------------------------------------------------------------


def _get_param(request, cli_opt: str, env_var: str, default: str = "") -> str:
    """Return value from pytest CLI option, falling back to env var, then default."""
    val = request.config.getoption(cli_opt, default=None)
    if val:
        return val
    return os.environ.get(env_var, default)


@pytest.fixture(scope="session")
def obj_config(request):
    """Session-scoped fixture that collects OBJ connection parameters.
    Skips the entire session if required credentials are not provided."""
    endpoint   = _get_param(request, "--obj-endpoint",   "OBJ_ENDPOINT")
    bucket     = _get_param(request, "--obj-bucket",     "OBJ_BUCKET")
    access_key = _get_param(request, "--obj-access-key", "OBJ_ACCESS_KEY")
    secret_key = _get_param(request, "--obj-secret-key", "OBJ_SECRET_KEY")
    scheme     = _get_param(request, "--obj-scheme",     "OBJ_SCHEME", "http")
    ca_bundle  = _get_param(request, "--obj-ca-bundle",  "OBJ_CA_BUNDLE", "")
    if not endpoint or not bucket or not access_key or not secret_key:
        pytest.skip("OBJ endpoint, bucket, access_key and secret_key must be set")

    cfg = {
        "endpoint_override": endpoint,
        "bucket":            bucket,
        "access_key":        access_key,
        "secret_key":        secret_key,
        "scheme":            scheme,
        "ca_bundle":         ca_bundle,
    }

    try:
        NixlLookup(cfg).exists("__connectivity_check__")
    except Exception as e:
        pytest.skip(f"Object store not reachable: {e}")

    return cfg


# ---------------------------------------------------------------------------
# OBJ-specific roundtrip
# ---------------------------------------------------------------------------


def roundtrip_once_obj(
    *,
    obj_config: dict,
    dtype: torch.dtype,
    num_layers: int,
    num_blocks: int,
    gpu_block_size: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    read_block_ids: list[int],
    write_block_ids: list[int],
    threads_per_gpu: int,
):
    """Write blocks to S3 then read them back, asserting bit-exact equality.

    For the OBJ backend gpu_blocks_per_file is always 1 - each GPU block maps
    to a single S3 object. FileMapper is still used to derive deterministic
    S3 key names from the block hashes.
    """
    gpu_blocks_per_file = 1  # OBJ: one S3 object per GPU block

    # Cross-layer layout: single tensor (num_layers, 2, num_blocks, block_size, num_heads, head_size)
    # nixl_offload requires len(tensors) == 1, which _get_tensors produces from this layout.
    original = create_cross_layer_kv_tensor(
        num_layers, num_blocks, block_size, num_heads, head_size, dtype
    )
    restored = torch.zeros_like(original)

    # FileMapper generates S3 key names; root_dir becomes the key prefix.
    file_mapper = FileMapper(
        root_dir="kv-test",
        model_name="test-model",
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype=str(dtype),
    )

    put_gpu_specs = make_gpu_specs(write_block_ids)
    put_num_files = math.ceil(len(write_block_ids) / gpu_blocks_per_file)
    put_storage_specs, block_hashes = make_storage_specs(put_num_files)

    # Single cross-layer entry — _get_tensors detects the extra leading dim and
    # keeps it as one tensor, satisfying nixl_offload's len(tensors) == 1 assert.
    attn_backends = {"ALL_LAYERS": FlashAttentionBackend}
    kv_caches_original = {"ALL_LAYERS": original}
    kv_caches_restored = {"ALL_LAYERS": restored}

    # PUT phase
    put_handlers = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
        backend="OBJ",
        extra_config=obj_config,
    )
    put_handler = put_handlers.gpu_to_storage_handler
    start_put = time.time()
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    put_result = wait_for(put_handler, job_id=1, timeout=30.0)
    dur_put = time.time() - start_put
    assert put_result.success, f"PUT failed: {put_result}"

    assert put_result.transfer_size is not None and put_result.transfer_size > 0
    assert put_result.transfer_time is not None and put_result.transfer_time > 0
    assert put_result.transfer_type == ("GPU", "SHARED_STORAGE")

    # GET phase
    get_handlers = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_restored,
        gpu_blocks_per_file=gpu_blocks_per_file,
        threads_per_gpu=threads_per_gpu,
        gpu_block_size=gpu_block_size,
        attn_backends=attn_backends,
        backend="OBJ",
        extra_config=obj_config,
    )
    get_handler = get_handlers.storage_to_gpu_handler

    get_gpu_specs = make_gpu_specs(read_block_ids)
    get_num_files = math.ceil(len(read_block_ids) / gpu_blocks_per_file)
    start_index = len(put_storage_specs.block_hashes) - get_num_files
    get_storage_spec = SharedStorageLoadStoreSpec(
        put_storage_specs.block_hashes[start_index:]
    )
    start_get = time.time()
    get_handler.transfer_async(job_id=2, spec=(get_storage_spec, get_gpu_specs))
    get_result = wait_for(get_handler, job_id=2, timeout=30.0)
    dur_get = time.time() - start_get
    assert get_result.success, f"GET failed: {get_result}"

    assert get_result.transfer_size is not None and get_result.transfer_size > 0
    assert get_result.transfer_time is not None and get_result.transfer_time > 0
    assert get_result.transfer_type == ("SHARED_STORAGE", "GPU")

    assert_blocks_equal_cross(original, restored, read_block_ids)

    write_total_mb = total_block_size_mb(
        num_layers, num_heads, block_size, head_size, dtype, len(write_block_ids)
    )
    read_total_mb = total_block_size_mb(
        num_layers, num_heads, block_size, head_size, dtype, len(read_block_ids)
    )
    print(
        f"\n[INFO] write blocks={len(write_block_ids)} read blocks={len(read_block_ids)} "
        f"PUT {dur_put:.4f}s ({throughput_gbps(write_total_mb, dur_put):.2f} GB/s), "
        f"GET {dur_get:.4f}s ({throughput_gbps(read_total_mb, dur_get):.2f} GB/s)"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start_idx", [0, 3])
def test_obj_backend_roundtrip(start_idx: int, obj_config, default_vllm_config):
    """End-to-end write/read roundtrip through the OBJ (S3/MinIO) backend.

    Writes num_blocks GPU blocks to S3, then reads back a subset starting at
    start_idx, and verifies bit-exact equality.
    """
    num_layers = 80
    num_blocks = 8
    block_size = 16
    num_heads = 64
    head_size = 128
    dtype = torch.float16
    threads_per_gpu = 8
    gpu_block_size = 16

    write_block_ids = list(range(num_blocks))
    read_block_ids = list(range(start_idx, num_blocks))

    roundtrip_once_obj(
        obj_config=obj_config,
        dtype=dtype,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        gpu_block_size=gpu_block_size,
        num_heads=num_heads,
        head_size=head_size,
        write_block_ids=write_block_ids,
        read_block_ids=read_block_ids,
        threads_per_gpu=threads_per_gpu,
    )

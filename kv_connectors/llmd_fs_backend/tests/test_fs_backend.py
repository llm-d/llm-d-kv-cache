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

# tests/test_fs_backend.py

import hashlib
import math
import os
import struct
import time
from collections.abc import Iterable

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import OffloadKey, make_offload_key
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import StorageOffloadingHandlers

TMP_DIR = "/tmp/shared-kv-test"

# ----------------------------
# Helpers functions
# ----------------------------


def create_dummy_kv_tensors(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Create dummy KV cache tensors [K, V] for all layers with shape
    (2, num_blocks, num_heads, block_size, head_size)."""
    torch.manual_seed(seed)
    shape = (2, num_blocks, block_size, num_heads, head_size)
    return [torch.rand(shape, dtype=dtype, device="cuda") for _ in range(num_layers)]


def get_prefix_hash(token_ids: Iterable[int]) -> BlockHash:
    """Generate a stable 64-bit hash for a list of token IDs
    by packing each as uint32."""
    buf = bytearray()
    for t in token_ids:
        buf += struct.pack("<I", int(t) & 0xFFFFFFFF)
    digest_int = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
    # Convert 64-bit int to 8-byte little-endian representation
    return BlockHash((digest_int & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))


def get_offload_key(token_ids: Iterable[int], group_idx: int = 0) -> OffloadKey:
    """Generate an OffloadKey from token IDs and group index."""
    block_hash = get_prefix_hash(token_ids)
    return make_offload_key(block_hash, group_idx)


def make_gpu_specs(block_ids: list[int]) -> GPULoadStoreSpec:
    """Create GPULoadStoreSpec objects for the given block IDs."""
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)])


def make_storage_specs(
    num_files: int,
    start_offset: int = 0,
) -> tuple[SharedStorageLoadStoreSpec, list[OffloadKey]]:
    """Create SharedStorageLoadStoreSpec objects and their keys for
    a given number of files.

    Args:
        num_files: Number of file keys to generate
        start_offset: Starting index for hash generation (prevents conflicts)
    """
    ranges = [
        (100 + (start_offset + i) * 100, 117 + (start_offset + i) * 100)
        for i in range(num_files)
    ]
    keys = [get_offload_key(range(a, b)) for (a, b) in ranges]
    return SharedStorageLoadStoreSpec(keys), keys


def cleanup_files(
    file_mapper: FileMapper,
    keys: list[OffloadKey],
) -> None:
    """Remove existing files for the provided offload keys."""
    from vllm.v1.kv_offload.abstract import get_offload_block_hash

    for key in keys:
        block_hash = get_offload_block_hash(key)
        path = file_mapper.get_file_name(block_hash)
        if os.path.exists(path):
            os.remove(path)


def throughput_gbps(total_mb: float, seconds: float) -> float:
    """Calculate throughput in GB/s given MB transferred and elapsed seconds."""
    return float("inf") if seconds <= 0 else (total_mb / 1024.0) / seconds


def assert_blocks_equal(
    original_tensors: list[torch.Tensor],
    restored_tensors: list[torch.Tensor],
    block_ids: list[int],
) -> None:
    """Assert that restored blocks match the original blocks for the given block IDs."""
    for orig, restored in zip(original_tensors, restored_tensors):
        for b in block_ids:
            torch.testing.assert_close(orig[:, int(b)], restored[:, int(b)])


def wait_for_file(file_path: str, timeout: float = 2.0) -> bool:
    """Wait for a file to exist up to timeout seconds."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(0.01)  # avoid busy-spin
    return False


def total_block_size_mb(
    num_layers: int,
    num_heads: int,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
    num_blocks: int,
) -> float:
    """Compute total block size in MB for the given model dimensions
    and number of blocks."""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    per_block_bytes = (
        num_layers * 2 * num_heads * block_size * head_size * bytes_per_elem
    )
    return (per_block_bytes * num_blocks) / (1024 * 1024)


def wait_for(
    handler,
    job_id: int,
    timeout: float = 2.0,
    _finished_cache: dict = None,
):
    """
    Wait for a specific job in handler.get_finished() up to timeout seconds.

    Args:
        handler: The handler object (put or get) to poll for finished jobs
        job_id: The specific job ID to wait for
        timeout: Max time to wait in seconds
        _finished_cache: Optional dict to cache finished jobs. Required when
            multiple handlers share the same engine, since get_finished() erases
            jobs from the map and we need to remember them across calls.

    Returns:
        The TransferResult for the completed job.
    """
    # If no cache provided, create a local one (for backward compatibility)
    if _finished_cache is None:
        _finished_cache = {}

    if job_id in _finished_cache:
        return _finished_cache[job_id]

    start = time.time()
    while time.time() - start < timeout:
        finished = handler.get_finished()
        for result in finished:
            # Cache ALL finished results (important when handlers share an engine)
            _finished_cache[result.job_id] = result
            if result.job_id == job_id:
                return result
        time.sleep(0.01)  # avoid busy-spin

    raise TimeoutError(
        f"Job {job_id} did not finish within {timeout}s. "
        f"Cached jobs: {list(_finished_cache.keys())}"
    )


def build_canonical_kv_caches(
    tensors: list[torch.Tensor],
    num_groups: int = 1,
) -> CanonicalKVCaches:
    """Build CanonicalKVCaches from FlashAttention-style tensors.

    Each tensor has shape (2, num_blocks, block_size, num_heads, head_size).
    We split K and V into separate canonical tensors, mirroring what
    vllm's register_kv_caches() does.

    Args:
        tensors: List of per-layer KV cache tensors.
        num_groups: Number of KV cache groups. Layers are split evenly
            across groups (simulating HMA models with multiple attention types).
    """
    canonical_tensors = []
    all_group_refs: list[list[CanonicalKVCacheRef]] = [[] for _ in range(num_groups)]

    layers_per_group = len(tensors) // num_groups

    for layer_idx, layer_tensor in enumerate(tensors):
        group_idx = min(layer_idx // layers_per_group, num_groups - 1)
        # FlashAttention layout: (2, num_blocks, ...)
        # Split into K and V sub-tensors, each (num_blocks, ...)
        for sub_tensor in layer_tensor:
            page_size_bytes = sub_tensor.stride(0) * sub_tensor.element_size()
            tensor_idx = len(canonical_tensors)
            canonical_tensors.append(
                CanonicalKVCacheTensor(
                    tensor=sub_tensor,
                    page_size_bytes=page_size_bytes,
                )
            )
            all_group_refs[group_idx].append(
                CanonicalKVCacheRef(
                    tensor_idx=tensor_idx,
                    page_size_bytes=page_size_bytes,
                )
            )

    return CanonicalKVCaches(
        tensors=canonical_tensors,
        group_data_refs=all_group_refs,
    )


def roundtrip_once(
    *,
    file_mapper: FileMapper,
    dtype: torch.dtype,
    num_layers: int,
    num_blocks: int,
    gpu_block_size: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    read_block_ids: list[int],
    write_block_ids: list[int],
    gpu_blocks_per_file: int,
    threads_per_gpu: int,
    num_groups: int = 1,
):
    from vllm.v1.kv_offload.abstract import get_offload_block_hash

    original = create_dummy_kv_tensors(
        num_layers, num_blocks, block_size, num_heads, head_size, dtype
    )
    restored = [torch.zeros_like(t) for t in original]

    put_gpu_specs = make_gpu_specs(write_block_ids)
    put_num_files = math.ceil(len(write_block_ids) / gpu_blocks_per_file)
    put_storage_specs, keys = make_storage_specs(put_num_files)
    cleanup_files(file_mapper, keys)

    # Build CanonicalKVCaches from test tensors
    canonical_original = build_canonical_kv_caches(original, num_groups)
    canonical_restored = build_canonical_kv_caches(restored, num_groups)

    # PUT phase
    kv_caches_original_handler = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=canonical_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
    )
    put_handler = kv_caches_original_handler.gpu_to_storage_handler
    start_put = time.time()
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    put_result = wait_for(put_handler, job_id=1, timeout=2.0)
    assert put_result.success, "PUT failed"
    dur_put = time.time() - start_put

    # Verify PUT metrics fields
    assert put_result.transfer_size is not None and put_result.transfer_size > 0
    assert put_result.transfer_time is not None and put_result.transfer_time > 0
    assert put_result.transfer_type == ("GPU", "SHARED_STORAGE")
    for key in keys:
        block_hash = get_offload_block_hash(key)
        file_path = file_mapper.get_file_name(block_hash)
        assert wait_for_file(file_path, timeout=2.0), (
            f"missing file after PUT: {file_path}"
        )

    # GET phase
    kv_caches_restored_handler = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=canonical_restored,
        gpu_blocks_per_file=gpu_blocks_per_file,
        threads_per_gpu=threads_per_gpu,
        gpu_block_size=gpu_block_size,
    )
    get_handler = kv_caches_restored_handler.storage_to_gpu_handler

    get_gpu_specs = make_gpu_specs(read_block_ids)
    get_num_files = math.ceil(len(read_block_ids) / gpu_blocks_per_file)
    start_index = len(put_storage_specs.keys) - get_num_files
    get_storage_spec = SharedStorageLoadStoreSpec(put_storage_specs.keys[start_index:])
    start_get = time.time()
    get_handler.transfer_async(job_id=2, spec=(get_storage_spec, get_gpu_specs))
    get_result = wait_for(get_handler, job_id=2, timeout=2.0)
    dur_get = time.time() - start_get
    assert get_result.success, "GET failed"

    # Verify GET metrics fields
    assert get_result.transfer_size is not None and get_result.transfer_size > 0
    assert get_result.transfer_time is not None and get_result.transfer_time > 0
    assert get_result.transfer_type == ("SHARED_STORAGE", "GPU")
    assert_blocks_equal(original, restored, read_block_ids)

    # Report
    write_total_mb = total_block_size_mb(
        num_layers, num_heads, block_size, head_size, dtype, len(write_block_ids)
    )
    read_total_mb = total_block_size_mb(
        num_layers, num_heads, block_size, head_size, dtype, len(read_block_ids)
    )
    block_hash_0 = get_offload_block_hash(keys[0])
    file_size_mb = os.path.getsize(file_mapper.get_file_name(block_hash_0)) / (
        1024 * 1024
    )
    num_files = len(keys)
    print(
        f"[INFO] group={gpu_blocks_per_file} write blocks len: "
        f"{len(write_block_ids)} read blocks len: {len(read_block_ids)} "
        f"PUT {dur_put:.4f}s ({throughput_gbps(write_total_mb, dur_put):.2f} GB/s), "
        f"GET {dur_get:.4f}s ({throughput_gbps(read_total_mb, dur_get):.2f} GB/s), "
        f"files={num_files}, sizes(MB)={file_size_mb:.2f} "
    )


# ----------------------------
# Test
# ----------------------------


@pytest.mark.parametrize("gpu_blocks_per_file", [1, 2, 4, 8])
# start_idx 0 = full from start, start_idx = 3, partial first group (e.g., 3..7)
@pytest.mark.parametrize("start_idx", [0, 3])
def test_fs_backend_roundtrip_param(
    gpu_blocks_per_file: int, start_idx: int, default_vllm_config
):
    """
    End-to-end tests for the fs (shared-storage) offloading backend.

    This suite verifies that KV-cache blocks can be:
    1. Written from GPU → Storage using the async PUT path.
    2. Read back from Storage → GPU using the async GET path.
    3. Correctly grouped into files using the configured gpu_blocks_per_file.
    4. Restored exactly (bit-matching) for selected block IDs.
    5. Run across multiple group sizes and partial block ranges.

    Each test simulates realistic Llama-style KV shapes and uses the
    StorageOffloadingHandlers to perform
    a full roundtrip with async thread-pool execution.
    """
    model_name = "llama3-70b"
    tp_size = 1
    tp_rank = 0
    dtype = torch.float16
    root_dir = TMP_DIR
    num_layers = 80
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 8
    write_block_ids = list(range(num_blocks))
    read_block_ids = list(range(start_idx, num_blocks))
    threads_per_gpu = 8
    gpu_block_size = 16
    file_mapper = FileMapper(
        root_dir=root_dir,
        model_name=model_name,
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=tp_size,
        pp_size=tp_size,
        pcp_size=tp_size,
        rank=tp_rank,
        dtype=dtype,
    )
    roundtrip_once(
        file_mapper=file_mapper,
        num_layers=num_layers,
        dtype=dtype,
        num_blocks=num_blocks,
        block_size=block_size,
        gpu_block_size=gpu_block_size,
        num_heads=num_heads,
        head_size=head_size,
        read_block_ids=read_block_ids,
        write_block_ids=write_block_ids,
        gpu_blocks_per_file=gpu_blocks_per_file,
        threads_per_gpu=threads_per_gpu,
    )

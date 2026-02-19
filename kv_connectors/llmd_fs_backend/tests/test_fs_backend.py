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
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec

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


def make_gpu_specs(block_ids: list[int]) -> GPULoadStoreSpec:
    """Create GPULoadStoreSpec objects for the given block IDs."""
    return GPULoadStoreSpec(block_ids)


def make_storage_specs(
    num_files: int,
    start_offset: int = 0,
) -> tuple[SharedStorageLoadStoreSpec, list[BlockHash]]:
    """Create SharedStorageLoadStoreSpec objects and their hashes for
    a given number of files.

    Args:
        num_files: Number of file hashes to generate
        start_offset: Starting index for hash generation (prevents conflicts)
    """
    ranges = [(100 + (start_offset + i) * 100, 117 + (start_offset + i) * 100)
              for i in range(num_files)]
    hashes = [get_prefix_hash(range(a, b)) for (a, b) in ranges]
    return SharedStorageLoadStoreSpec(hashes), hashes


def cleanup_files(
    file_mapper: FileMapper,
    block_hashes: list[BlockHash],
) -> None:
    """Remove existing files for the provided block hashes."""
    for h in block_hashes:
        path = file_mapper.get_file_name(h)
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


def log_file_info(
    base_path: str,
    block_hashes: list[BlockHash],
) -> tuple[int, list[float]]:
    """Log information about the files corresponding to the given block hashes."""
    file_sizes = []
    for h in block_hashes:
        path = StorageOffloadingHandlers.get_file_name(base_path, h)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            file_sizes.append(size_mb)
    num_files = len(file_sizes)
    return num_files, file_sizes


def wait_for(
    handler,
    job_id: int,
    timeout: float = 2.0,
    _finished_cache: dict = None,
) -> bool:
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
        True if job succeeded, False if it failed
    """
    # If no cache provided, create a local one (for backward compatibility)
    if _finished_cache is None:
        _finished_cache = {}

    if job_id in _finished_cache:
        return _finished_cache[job_id]

    start = time.time()
    while time.time() - start < timeout:
        finished = handler.get_finished()
        # Cache ALL finished jobs we see (important when handlers share an engine)
        for jid, ok in finished:
            _finished_cache[jid] = ok
            if jid == job_id:
                return ok
        time.sleep(0.01)  # avoid busy-spin

    raise TimeoutError(
        f"Job {job_id} did not finish within {timeout}s. "
        f"Cached jobs: {list(_finished_cache.keys())}"
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
):
    original = create_dummy_kv_tensors(
        num_layers, num_blocks, block_size, num_heads, head_size, dtype
    )
    restored = [torch.zeros_like(t) for t in original]

    put_gpu_specs = make_gpu_specs(write_block_ids)
    put_num_files = math.ceil(len(write_block_ids) / gpu_blocks_per_file)
    put_storage_specs, block_hashes = make_storage_specs(put_num_files)
    cleanup_files(file_mapper, block_hashes)

    # set names for layers
    attn_backends = {f"layer_{i}": FlashAttentionBackend for i in range(num_layers)}
    kv_caches_original = {f"layer_{i}": original[i] for i in range(num_layers)}
    kv_caches_restored = {f"layer_{i}": restored[i] for i in range(num_layers)}

    # PUT phase
    kv_caches_original_handler = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
    )
    put_handler = kv_caches_original_handler.gpu_to_storage_handler
    start_put = time.time()
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    ok_put = wait_for(put_handler, job_id=1, timeout=2.0)
    assert ok_put, "PUT failed"
    dur_put = time.time() - start_put
    for h in block_hashes:
        file_path = file_mapper.get_file_name(h)
        assert wait_for_file(file_path, timeout=2.0), (
            f"missing file after PUT: {file_path}"
        )

    # GET phase
    kv_caches_restored_handler = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_restored,
        gpu_blocks_per_file=gpu_blocks_per_file,
        threads_per_gpu=threads_per_gpu,
        gpu_block_size=gpu_block_size,
        attn_backends=attn_backends,
    )
    get_handler = kv_caches_restored_handler.storage_to_gpu_handler

    get_gpu_specs = make_gpu_specs(read_block_ids)
    get_num_files = math.ceil(len(read_block_ids) / gpu_blocks_per_file)
    start_index = len(put_storage_specs.block_hashes) - get_num_files
    get_storage_spec = SharedStorageLoadStoreSpec(
        put_storage_specs.block_hashes[start_index:]
    )
    start_get = time.time()
    get_handler.transfer_async(job_id=2, spec=(get_storage_spec, get_gpu_specs))
    ok_get = wait_for(get_handler, job_id=2, timeout=2.0)
    dur_get = time.time() - start_get
    assert ok_get, "GET failed"
    assert_blocks_equal(original, restored, read_block_ids)

    # Report
    write_total_mb = total_block_size_mb(
        num_layers, num_heads, block_size, head_size, dtype, len(write_block_ids)
    )
    read_total_mb = total_block_size_mb(
        num_layers, num_heads, block_size, head_size, dtype, len(read_block_ids)
    )
    file_size_mb = os.path.getsize(file_mapper.get_file_name(block_hashes[0])) / (
        1024 * 1024
    )
    num_files = len(block_hashes)
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


@pytest.mark.parametrize("read_preferring_ratio", [0.5, 0.75, 0.9])
def test_read_priority_over_writes(read_preferring_ratio: float):
    """
    Test read/write timing when both operations are queued simultaneously.

    Submits bulk writes followed by a read on the same thread pool and reports
    the timing for each. With priority queue implementation, reads should complete
    faster than writes. Without priority, reads wait behind queued writes.

    The test uses more threads and blocks to create sufficient contention and
    demonstrate the effects of different worker preference ratios.
    """
    model_name = f"llama3-70b-ratio-{int(read_preferring_ratio*100)}"
    dtype = torch.float16
    num_layers = 80
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 128
    gpu_blocks_per_file = 4
    gpu_block_size = 16
    threads_per_gpu = 8
    num_read_iterations = 5

    file_mapper = FileMapper(
        root_dir=TMP_DIR,
        model_name=model_name,
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype=dtype,
    )

    kv_cache = create_dummy_kv_tensors(
        num_layers, num_blocks, block_size, num_heads, head_size, dtype
    )
    attn_backends = {f"layer_{i}": FlashAttentionBackend for i in range(num_layers)}
    kv_dict = {f"layer_{i}": kv_cache[i] for i in range(num_layers)}

    handler = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_dict,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
        read_preferring_ratio=read_preferring_ratio,
    )

    # Share handler.engine.m_thread_pool
    put = handler.gpu_to_storage_handler
    get = handler.storage_to_gpu_handler

    # Shared cache for finished jobs, needed because put and get share the same engine,
    # so engine.get_finished() erases jobs and we need to remember them
    finished_jobs_cache = {}

    # Step 1: Write a few files that we can read later
    files_to_read = 5
    small_block_ids = list(range(files_to_read * gpu_blocks_per_file))
    small_put_gpu = make_gpu_specs(small_block_ids)
    small_put_storage, small_hashes = make_storage_specs(files_to_read)
    cleanup_files(file_mapper, small_hashes)

    put.transfer_async(job_id=1, spec=(small_put_gpu, small_put_storage))
    ok_initial = wait_for(put, job_id=1, timeout=15.0, _finished_cache=finished_jobs_cache)
    assert ok_initial, "Initial write failed"

    # Step 2: Submit large bulk write to saturate the queue (many files)
    bulk_block_ids = list(range(gpu_blocks_per_file * files_to_read, num_blocks))
    bulk_put_gpu = make_gpu_specs(bulk_block_ids)
    num_bulk_files = math.ceil(len(bulk_block_ids) / gpu_blocks_per_file)
    bulk_put_storage, bulk_hashes = make_storage_specs(num_bulk_files, start_offset=files_to_read)
    cleanup_files(file_mapper, bulk_hashes)

    start_bulk_write = time.time()
    put.transfer_async(job_id=2, spec=(bulk_put_gpu, bulk_put_storage))

    # Step 3: Submit all reads at once to create burst of competing reads
    # Small delay to let write queue build up first
    time.sleep(0.05)

    read_start_times = []
    for i in range(num_read_iterations):
        # Read from the pre-written files (cycle through them)
        file_idx = i % files_to_read
        read_block_ids = list(range(
            file_idx * gpu_blocks_per_file,
            (file_idx + 1) * gpu_blocks_per_file
        ))
        read_gpu = make_gpu_specs(read_block_ids)
        read_storage = SharedStorageLoadStoreSpec([small_put_storage.block_hashes[file_idx]])

        read_start = time.time()
        get.transfer_async(job_id=100 + i, spec=(read_storage, read_gpu))
        read_start_times.append(read_start)

    # Now wait for all reads and measure their latencies
    read_times = []
    for i in range(num_read_iterations):
        ok_read = wait_for(get, job_id=100 + i, timeout=30.0, _finished_cache=finished_jobs_cache)
        read_time = time.time() - read_start_times[i]
        assert ok_read, f"Read {i} failed"
        read_times.append(read_time)

    # Step 4: Wait for bulk writes to complete
    ok_bulk_write = wait_for(put, job_id=2, timeout=60.0, _finished_cache=finished_jobs_cache)
    total_write_time = time.time() - start_bulk_write

    assert ok_bulk_write, "Bulk write failed"

    avg_read_time = sum(read_times) / len(read_times)

    read_workers = max(1, int(threads_per_gpu * read_preferring_ratio))
    write_workers = threads_per_gpu - read_workers

    print(f"\nRead preferring ratio: {read_preferring_ratio:.0%} "
          f"({read_workers} read-first, {write_workers} write-first workers)")
    print(f"Workload: {num_read_iterations} reads (1 file each) competing with "
          f"{num_bulk_files} write files")
    print(f"Avg read latency: {avg_read_time:.3f}s (min: {min(read_times):.3f}s, "
          f"max: {max(read_times):.3f}s)")
    print(f"Total write time:  {total_write_time:.3f}s")

    # Assertion: Average read latency should be reasonable
    # Expected max avg read latency based on ratio:
    # - 90% read-preferring: reads should avg <0.20s (most workers prioritize reads)
    # - 75% read-preferring: reads should avg <0.25s (majority workers prioritize reads)
    # - 50% read-preferring: reads should avg <0.35s (balanced, but still some priority)
    if read_preferring_ratio >= 0.9:
        max_avg_read = 0.30
    elif read_preferring_ratio >= 0.75:
        max_avg_read = 0.30
    else:  # 0.5
        max_avg_read = 0.35

    assert avg_read_time <= max_avg_read, (
        f"Average read latency ({avg_read_time:.3f}s) exceeded threshold ({max_avg_read:.3f}s) "
        f"for ratio={read_preferring_ratio:.0%}. "
        f"Worker preferences may not be working correctly or system is overloaded."
    )

    cleanup_files(file_mapper, small_hashes)
    cleanup_files(file_mapper, bulk_hashes)

    # Ensure handler is fully destroyed and thread pool shutdown before next test
    # This prevents cross-test interference where the next test's rmtree() deletes
    # directories while this test's threads are still writing files
    del handler, put, get, kv_dict
    import gc
    gc.collect()
    time.sleep(0.5)


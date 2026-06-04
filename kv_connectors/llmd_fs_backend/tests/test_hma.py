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

# tests/test_hma.py
#
# HMA (Hybrid Memory Architecture) tests for the FS backend.
# Exercises the multi-group offload path: each KV cache group has its
# own subset of canonical tensors, and a single TransferSpec may span
# multiple groups.

import math
import time

import pytest
import torch

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import StorageOffloadingHandlers
from tests.test_fs_backend import (
    TMP_DIR,
    assert_blocks_equal,
    cleanup_files,
    create_dummy_kv_tensors,
    make_canonical_kv_caches,
    make_gpu_specs,
    make_storage_specs,
    wait_for,
)


@pytest.mark.parametrize("num_groups", [2, 4])
def test_hma_multi_group_roundtrip(num_groups: int, default_vllm_config):
    """PUT -> GET roundtrip with multiple KV cache groups.

    Each group has its own tensor subset (non-shared), its own subset of
    the keys list (encoded with that group's index), and its own slice
    of the flat GPULoadStoreSpec.block_ids.
    """
    model_name = f"hma-test-{num_groups}g"
    dtype = torch.float16
    num_layers_per_group = 20  # 80 total layers with num_groups=4
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 8
    gpu_blocks_per_file = 4
    gpu_block_size = 16
    threads_per_gpu = 8

    # Per-group write/read block IDs (each group uses independent GPU block
    # positions within its own tensors).
    blocks_per_group = num_blocks
    write_block_ids = list(range(blocks_per_group)) * num_groups  # flat
    read_block_ids = list(write_block_ids)
    group_sizes = [blocks_per_group] * num_groups

    file_mapper = FileMapper(
        root_dir=TMP_DIR,
        model_name=model_name,
        hash_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        dtype=dtype,
    )

    # One set of tensors per group (distinct physical memory)
    original_per_group = [
        create_dummy_kv_tensors(
            num_layers_per_group,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            dtype,
            seed=42 + g,
        )
        for g in range(num_groups)
    ]
    restored_per_group = [
        [torch.zeros_like(t) for t in group_tensors]
        for group_tensors in original_per_group
    ]

    # Flatten per-group tensors for the unified helper, which splits layers
    # evenly across num_groups in order.
    original_flat = [t for group in original_per_group for t in group]
    restored_flat = [t for group in restored_per_group for t in group]
    canonical_original = make_canonical_kv_caches(original_flat, num_groups)
    canonical_restored = make_canonical_kv_caches(restored_flat, num_groups)

    num_files_per_group = [
        math.ceil(group_sizes[g] / gpu_blocks_per_file) for g in range(num_groups)
    ]
    put_storage_specs, keys = make_storage_specs(num_files_per_group)
    cleanup_files(file_mapper, keys)

    put_gpu_specs = make_gpu_specs(write_block_ids, group_sizes=group_sizes)

    # PUT phase
    put_handlers = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=canonical_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
    )
    put_handler = put_handlers.gpu_to_storage_handler

    start_put = time.time()
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    put_result = wait_for(put_handler, job_id=1, timeout=10.0)
    assert put_result.success, "PUT failed"
    print(f"[HMA] num_groups={num_groups} PUT {time.time() - start_put:.3f}s")

    # GET phase (fresh handlers so restored tensors are targeted)
    get_handlers = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=canonical_restored,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
    )
    get_handler = get_handlers.storage_to_gpu_handler

    get_gpu_specs = make_gpu_specs(read_block_ids, group_sizes=group_sizes)
    get_storage_spec = SharedStorageLoadStoreSpec(put_storage_specs.keys)
    start_get = time.time()
    get_handler.transfer_async(job_id=2, spec=(get_storage_spec, get_gpu_specs))
    get_result = wait_for(get_handler, job_id=2, timeout=10.0)
    assert get_result.success, "GET failed"
    print(f"[HMA] num_groups={num_groups} GET {time.time() - start_get:.3f}s")

    # Verify data integrity per group — each group's tensors should be
    # restored correctly from the correct set of files.
    for g in range(num_groups):
        assert_blocks_equal(
            original_per_group[g],
            restored_per_group[g],
            list(range(num_blocks)),
        )


# block_indices=(0, 2) means group 1 starts at logical block 2 within the
# request — with gpu_blocks_per_file=4 that group is head-partial on its
# first file (head_offset=2) and tail-partial on its last file.
@pytest.mark.parametrize("block_indices", [(0, 0), (0, 2), (1, 3)])
def test_hma_head_partial_roundtrip(block_indices: tuple, default_vllm_config):
    """PUT -> GET roundtrip with two HMA groups whose first GPU block lands
    mid-file. Exercises the head_offset propagation: writes must land at the
    correct slot in the file, and reads must recover them from the same slot.
    """
    model_name = f"hma-headpartial-{block_indices[0]}-{block_indices[1]}"
    dtype = torch.float16
    num_layers_per_group = 16
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 8
    gpu_blocks_per_file = 4
    gpu_block_size = 16
    threads_per_gpu = 8
    num_groups = 2
    blocks_per_group = num_blocks

    write_block_ids = list(range(blocks_per_group)) * num_groups
    read_block_ids = list(write_block_ids)
    group_sizes = [blocks_per_group] * num_groups

    # Number of files each group spans, given its unaligned start. Matches
    # `_num_files_for_group` in the worker.
    def num_files_for_group(start: int, n: int, gpb: int) -> int:
        return (start + n - 1) // gpb - start // gpb + 1

    num_files_per_group = [
        num_files_for_group(block_indices[g], group_sizes[g], gpu_blocks_per_file)
        for g in range(num_groups)
    ]

    file_mapper = FileMapper(
        root_dir=TMP_DIR,
        model_name=model_name,
        hash_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        dtype=dtype,
    )

    original_per_group = [
        create_dummy_kv_tensors(
            num_layers_per_group,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            dtype,
            seed=42 + g,
        )
        for g in range(num_groups)
    ]
    restored_per_group = [
        [torch.zeros_like(t) for t in group_tensors]
        for group_tensors in original_per_group
    ]

    original_flat = [t for group in original_per_group for t in group]
    restored_flat = [t for group in restored_per_group for t in group]
    canonical_original = make_canonical_kv_caches(original_flat, num_groups)
    canonical_restored = make_canonical_kv_caches(restored_flat, num_groups)

    put_storage_specs, keys = make_storage_specs(num_files_per_group)
    cleanup_files(file_mapper, keys)

    put_gpu_specs = make_gpu_specs(
        write_block_ids,
        group_sizes=group_sizes,
        block_indices=list(block_indices),
    )

    # PUT phase
    put_handlers = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=canonical_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
    )
    put_handler = put_handlers.gpu_to_storage_handler
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    put_result = wait_for(put_handler, job_id=1, timeout=10.0)
    assert put_result.success, "PUT failed"

    # GET phase
    get_handlers = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=canonical_restored,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
    )
    get_handler = get_handlers.storage_to_gpu_handler

    get_gpu_specs = make_gpu_specs(
        read_block_ids,
        group_sizes=group_sizes,
        block_indices=list(block_indices),
    )
    get_storage_spec = SharedStorageLoadStoreSpec(put_storage_specs.keys)
    get_handler.transfer_async(job_id=2, spec=(get_storage_spec, get_gpu_specs))
    get_result = wait_for(get_handler, job_id=2, timeout=10.0)
    assert get_result.success, "GET failed"

    for g in range(num_groups):
        assert_blocks_equal(
            original_per_group[g],
            restored_per_group[g],
            list(range(num_blocks)),
        )

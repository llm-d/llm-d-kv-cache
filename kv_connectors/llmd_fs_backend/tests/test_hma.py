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
from vllm.v1.kv_offload.spec import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import StorageOffloadingHandlers
from tests.test_fs_backend import (
    TMP_DIR,
    assert_blocks_equal,
    cleanup_files,
    create_dummy_kv_tensors,
    make_gpu_specs,
    make_multigroup_storage_specs,
    wait_for,
)


def build_multigroup_canonical_kv_caches(
    tensors_per_group: list[list[torch.Tensor]],
) -> CanonicalKVCaches:
    """Build a CanonicalKVCaches with one group per sub-list of tensors.

    Each group gets its own subset of canonical tensors (no sharing between
    groups). This matches the typical HMA layout where different attention
    types back different layers.
    """
    canonical_tensors: list[CanonicalKVCacheTensor] = []
    group_data_refs: list[list[CanonicalKVCacheRef]] = []

    for group_tensors in tensors_per_group:
        group_refs: list[CanonicalKVCacheRef] = []
        for layer_tensor in group_tensors:
            # FlashAttention layout: (2, num_blocks, ...) — split into K/V
            for sub_tensor in layer_tensor:
                page_size_bytes = sub_tensor.stride(0) * sub_tensor.element_size()
                tensor_idx = len(canonical_tensors)
                canonical_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=sub_tensor,
                        page_size_bytes=page_size_bytes,
                    )
                )
                group_refs.append(
                    CanonicalKVCacheRef(
                        tensor_idx=tensor_idx,
                        page_size_bytes=page_size_bytes,
                    )
                )
        group_data_refs.append(group_refs)

    return CanonicalKVCaches(
        tensors=canonical_tensors,
        group_data_refs=group_data_refs,
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
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
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

    canonical_original = build_multigroup_canonical_kv_caches(original_per_group)
    canonical_restored = build_multigroup_canonical_kv_caches(restored_per_group)

    num_files_per_group = [
        math.ceil(group_sizes[g] / gpu_blocks_per_file) for g in range(num_groups)
    ]
    put_storage_specs, keys = make_multigroup_storage_specs(num_files_per_group)
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

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
# Validates that PUT -> GET roundtrips work with multiple KV cache groups,
# simulating models that mix attention types (e.g., full attention +
# sliding window + Mamba).

import pytest
import torch

from llmd_fs_backend.file_mapper import FileMapper
from tests.test_fs_backend import (
    TMP_DIR,
    roundtrip_once,
)


@pytest.mark.parametrize("num_groups", [2, 4])
def test_hma_multi_group_roundtrip(num_groups: int, default_vllm_config):
    """
    Test that PUT -> GET roundtrip works with multiple KV cache groups,
    simulating HMA models (e.g., full attention + sliding window + Mamba).

    Layers are split evenly across groups. All groups share the same block
    size (as enforced by the vllm OffloadingSpec). Verifies data integrity
    across all groups.
    """
    model_name = "hma-test-model"
    dtype = torch.float16
    num_layers = 80
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 8
    gpu_blocks_per_file = 4
    gpu_block_size = 16
    threads_per_gpu = 8

    write_block_ids = list(range(num_blocks))
    read_block_ids = list(range(num_blocks))

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
        num_groups=num_groups,
    )

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

"""Scheduler/worker config consistency tests for SharedStorageOffloadingSpec."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from llmd_fs_backend.spec import (
    DEFAULT_STORAGE_BLOCK_SIZE,
    SharedStorageOffloadingSpec,
)

pytestmark = pytest.mark.no_cuda_required

GPU_BLOCK_SIZE = 16


def make_vllm_config(extra_config: dict) -> SimpleNamespace:
    """Minimal stand-in exposing the attributes OffloadingSpec/FileMapper read."""
    return SimpleNamespace(
        model_config=SimpleNamespace(model="test/hybrid-model"),
        cache_config=SimpleNamespace(
            block_size=GPU_BLOCK_SIZE,
            cache_dtype="auto",
            enable_prefix_caching=True,
            hash_block_size=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size=1,
            rank=0,
        ),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra_config),
    )


def make_hybrid_kv_cache_config() -> KVCacheConfig:
    """Two KV cache groups (full attention + sliding window), Gemma-style."""
    attn_args = dict(
        block_size=GPU_BLOCK_SIZE,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    return KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layers.0.attn"],
                kv_cache_spec=FullAttentionSpec(**attn_args),
            ),
            KVCacheGroupSpec(
                layer_names=["layers.1.attn"],
                kv_cache_spec=SlidingWindowSpec(**attn_args, sliding_window=512),
            ),
        ],
    )


def make_spec(tmp_path, extra_config: dict) -> SharedStorageOffloadingSpec:
    extra_config = {"shared_storage_path": str(tmp_path), **extra_config}
    return SharedStorageOffloadingSpec(
        make_vllm_config(extra_config), make_hybrid_kv_cache_config()
    )


def test_default_block_size_syncs_scheduler_factor(tmp_path):
    """Without "block_size" in extra_config, vLLM's OffloadingSpec leaves
    block_size_factor at 1 while this backend defaults to
    DEFAULT_STORAGE_BLOCK_SIZE tokens per file. The spec must reconcile the
    two, or the scheduler emits one offload key per GPU block while the
    worker consumes one key per file — tripping the group_idx assertion in
    _build_transfer on hybrid models (issue #656).
    """
    spec = make_spec(tmp_path, {})
    assert spec.gpu_blocks_per_file == DEFAULT_STORAGE_BLOCK_SIZE // GPU_BLOCK_SIZE
    assert spec.block_size_factor == spec.gpu_blocks_per_file


def test_explicit_block_size_keeps_factor_in_sync(tmp_path):
    spec = make_spec(tmp_path, {"block_size": 128})
    assert spec.gpu_blocks_per_file == 128 // GPU_BLOCK_SIZE
    assert spec.block_size_factor == spec.gpu_blocks_per_file


def test_scheduler_config_sees_offloaded_block_size(tmp_path):
    """The vLLM scheduler derives per-group offloaded block sizes from
    block_size_factor; they must match the worker's per-file token span."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        SchedulerOffloadConfig,
    )

    spec = make_spec(tmp_path, {})
    config = SchedulerOffloadConfig.from_spec(spec)
    assert config.block_size_factor == spec.gpu_blocks_per_file
    for group_config in config.kv_group_configs:
        assert (
            group_config.offloaded_block_size
            == spec.gpu_blocks_per_file * group_config.gpu_block_size
        )

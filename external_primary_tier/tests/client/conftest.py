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

"""An External Primary Tier serving in a thread, and specs aimed at it."""

import threading

import pytest
import torch
from external_primary_tier.client.spec import ExternalOffloadingSpec
from external_primary_tier.tier.server import ExternalPrimaryTier
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

KV_BYTES_PER_BLOCK = 4096
"""One worker's KV bytes per block; with TP 1 and one block per chunk, also the chunk size."""
POOL_CHUNKS = 8


def offloading_config(
    *,
    master_address: str,
    engine_id: str = "engine-a",
    model_name: str = "model-a",
    tp_size: int = 1,
    is_parallelism_agnostic: bool = True,
    kv_bytes_per_block: int = KV_BYTES_PER_BLOCK,
    canonical_layout: bool = False,
    attach_timeout_ms: int = 5_000,
) -> OffloadingConfig:
    """The ``OffloadingConfig`` vLLM's connector builds for a small engine."""
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(tokens_per_block=16, layer_names=("layer-0", "layer-1"), group_id=0),),
        worker_kv_bytes_per_block=kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config={"master_address": master_address, "attach_timeout_ms": attach_timeout_ms},
        engine_id=engine_id,
        model=OffloadingModelConfig(name=model_name, dtype="float16", max_model_len=512),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=tp_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        canonical_layout=canonical_layout,
        kv_cache_layout="LBNHC" if canonical_layout else None,
    )


def kv_caches(page_size_bytes: int = KV_BYTES_PER_BLOCK) -> CanonicalKVCaches:
    """One KV cache tensor in one group, as vLLM canonicalizes a worker's caches."""
    tensor = torch.zeros((POOL_CHUNKS, page_size_bytes), dtype=torch.int8)
    return CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor, page_size_bytes)],
        group_data_refs=[[CanonicalKVCacheRef(0, page_size_bytes)]],
    )


@pytest.fixture
def tier(tmp_path):
    """An External Primary Tier with a pool of ``POOL_CHUNKS`` chunks, serving in a thread."""
    address = f"ipc://{tmp_path}/tier.sock"
    server = ExternalPrimaryTier(address, str(tmp_path), pool_bytes=POOL_CHUNKS * KV_BYTES_PER_BLOCK)
    stop = threading.Event()
    thread = threading.Thread(target=server.serve, args=(stop, 20), daemon=True)
    thread.start()
    yield address
    stop.set()
    thread.join(timeout=10)
    server.close()


@pytest.fixture
def make_spec(tier):
    """Builds specs aimed at the running tier."""

    def make(**overrides) -> ExternalOffloadingSpec:
        return ExternalOffloadingSpec(offloading_config(master_address=tier, **overrides))

    return make

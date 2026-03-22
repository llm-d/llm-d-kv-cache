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

from collections.abc import Iterator
from math import ceil, lcm

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import (
    DEFAULT_MAX_STAGING_MEMORY_GB,
    DEFAULT_READ_PREFERRING_WORKERS_RATIO,
    DEFAULT_THREADS_PER_GPU,
    StorageOffloadingHandlers,
)

DEFAULT_STORAGE_BLOCK_SIZE = 256


class SharedStorageOffloadingSpec(OffloadingSpec):
    """OffloadingSpec for shared storage backend."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self._manager: OffloadingManager | None = None
        self._handlers: StorageOffloadingHandlers | None = None

        self.threads_per_gpu = int(
            self.extra_config.get("threads_per_gpu", DEFAULT_THREADS_PER_GPU)
        )
        shared_storage_path = self.extra_config.get(
            "shared_storage_path", "/tmp/shared-kv"
        )
        self.max_staging_memory_gb = int(
            self.extra_config.get(
                "max_staging_memory_gb", DEFAULT_MAX_STAGING_MEMORY_GB
            )
        )
        if "block_size" in self.extra_config:
            self.offloaded_block_size = int(self.extra_config["block_size"])
        elif not self.hybrid_offload_enabled:
            self.offloaded_block_size = lcm(*self.gpu_block_size)

        self.gpu_block_sizes = tuple(int(block_size) for block_size in self.gpu_block_size)
        if self.hybrid_offload_enabled:
            self.gpu_blocks_per_file = tuple(
                ceil(self.offloaded_block_size / group_hash_block_size)
                for group_hash_block_size in self.group_hash_block_size
            )
        else:
            assert all(
                self.offloaded_block_size % gpu_block_size == 0
                for gpu_block_size in self.gpu_block_sizes
            ), "offloaded_block_size must be a multiple of every group gpu_block_size"
            self.gpu_blocks_per_file = tuple(
                self.offloaded_block_size // gpu_block_size
                for gpu_block_size in self.gpu_block_sizes
            )

        self.read_preferring_ratio = float(
            self.extra_config.get(
                "read_preferring_ratio", DEFAULT_READ_PREFERRING_WORKERS_RATIO
            )
        )

        parallel_config = vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        assert parallel_config.world_size == tp_size * pp_size * pcp_size

        dtype = str(vllm_config.cache_config.cache_dtype).replace("torch.", "")
        self.group_layer_names = tuple(
            tuple(group.layer_names) for group in kv_cache_config.kv_cache_groups
        )
        self.file_mappers = tuple(
            FileMapper(
                root_dir=f"{shared_storage_path}/group_{group_idx}",
                model_name=vllm_config.model_config.model,
                gpu_block_size=self.gpu_block_sizes[group_idx],
                gpu_blocks_per_file=self.gpu_blocks_per_file[group_idx],
                tp_size=tp_size,
                pp_size=pp_size,
                pcp_size=pcp_size,
                rank=parallel_config.rank,
                dtype=dtype,
            )
            for group_idx in range(len(self.group_layer_names))
        )
        self.file_mapper = self.file_mappers[0]

    def get_manager(self) -> OffloadingManager:
        assert self.vllm_config.parallel_config.rank == 0, "Scheduler rank should be 0"
        if not self._manager:
            self._manager = SharedStorageOffloadingManager(file_mapper=self.file_mappers)
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            self._handlers = StorageOffloadingHandlers(
                file_mapper=self.file_mapper,
                file_mappers=self.file_mappers,
                group_layer_names=self.group_layer_names,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                gpu_block_size=self.gpu_block_sizes,
                group_hash_block_size=self.group_hash_block_size,
                attn_backends=attn_backends,
                kv_caches=kv_caches,
                threads_per_gpu=self.threads_per_gpu,
                max_staging_memory_gb=self.max_staging_memory_gb,
            )

        assert self._handlers is not None
        yield (
            GPULoadStoreSpec,
            SharedStorageLoadStoreSpec,
            self._handlers.gpu_to_storage_handler,
        )
        yield (
            SharedStorageLoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.storage_to_gpu_handler,
        )

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

import torch
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import (DEFAULT_MAX_STAGING_MEMORY_GB,
                                    DEFAULT_MAX_THREADS_PER_GPU,
                                    StorageOffloadingHandlers)
from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class SharedStorageOffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for shared storage backend (e.g., mounted NFS, PVC).
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        self._manager: OffloadingManager | None = None
        # worker-side
        self._handlers: StorageOffloadingHandlers | None = None

        self.threads_per_gpu = int(
            self.extra_config.get("threads_per_gpu",
                                  DEFAULT_MAX_THREADS_PER_GPU))
        self.shared_storage_path = self.extra_config.get(
            "shared_storage_path", "/tmp/shared-kv")
        self.max_staging_memory_gb = int(
            self.extra_config.get(
                "max_staging_memory_gb",
                DEFAULT_MAX_STAGING_MEMORY_GB))  # Max staging CPU buffer in GB

        self.gpu_blocks_per_file = int(self.offloaded_block_size /
                                       self.gpu_block_size)
        assert self.offloaded_block_size % self.gpu_block_size == 0, "offloaded_block_size must be a multiple of gpu_block_size"

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            self._manager = SharedStorageOffloadingManager(
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                dtype=self.vllm_config.cache_config.cache_dtype,
                root_dir=self.shared_storage_path,
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec],
                        OffloadingHandler]]:

        if not self._handlers:
            self._handlers = StorageOffloadingHandlers(
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                dtype=self.vllm_config.cache_config.cache_dtype,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                root_dir=self.shared_storage_path,
                attn_backends=attn_backends,
                kv_caches=kv_caches,
                threads_per_gpu=self.threads_per_gpu,
                max_staging_memory_gb=self.max_staging_memory_gb,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._handlers.gpu_to_storage_handler
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._handlers.storage_to_gpu_handler

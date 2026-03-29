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

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
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
    """
    OffloadingSpec for shared storage backend (e.g., mounted NFS, PVC).
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self._manager: OffloadingManager | None = None
        # worker-side
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
        )  # Max staging CPU buffer in GB

        self.offloaded_block_size = int(
            self.extra_config.get("block_size", DEFAULT_STORAGE_BLOCK_SIZE)
        )

        # For HMA models, only attention groups have meaningful block sizes
        # for file-based offloading. Non-attention groups (Mamba, linear)
        # may have incompatible block sizes (e.g., entire sequence as one block).
        from vllm.v1.kv_cache_interface import AttentionSpec

        attn_block_sizes = [
            group.kv_cache_spec.block_size
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, AttentionSpec)
        ]
        if attn_block_sizes:
            gpu_block_size = attn_block_sizes[0]
        else:
            # Pure Mamba/SSM model — gpu_block_size may be very large
            # (e.g., entire sequence). Use it directly; gpu_blocks_per_file
            # will be 1 since each block is stored as a single file.
            gpu_block_size = self.gpu_block_size[0]

        self._attn_gpu_block_size = gpu_block_size

        if self.offloaded_block_size < gpu_block_size:
            # gpu_block_size exceeds offloaded_block_size (e.g., Mamba with
            # block_size=max_model_len). Store 1 gpu block per file.
            self.gpu_blocks_per_file = 1
        else:
            assert self.offloaded_block_size % gpu_block_size == 0, (
                f"offloaded_block_size ({self.offloaded_block_size}) must be "
                f"a multiple of gpu_block_size ({gpu_block_size})"
            )
            self.gpu_blocks_per_file = self.offloaded_block_size // gpu_block_size

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

        # TODO: use dtype from KVCacheConfig instead of VllmConfig.CacheConfig
        dtype = str(vllm_config.cache_config.cache_dtype).replace("torch.", "")
        self.file_mapper = FileMapper(
            root_dir=shared_storage_path,
            model_name=vllm_config.model_config.model,
            gpu_block_size=gpu_block_size,
            gpu_blocks_per_file=self.gpu_blocks_per_file,
            tp_size=tp_size,
            pp_size=pp_size,
            pcp_size=pcp_size,
            rank=parallel_config.rank,
            dtype=dtype,
        )

    def get_manager(self) -> OffloadingManager:
        assert self.vllm_config.parallel_config.rank == 0, "Scheduler rank should be 0"
        if not self._manager:
            self._manager = SharedStorageOffloadingManager(file_mapper=self.file_mapper)
        return self._manager

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            self._handlers = StorageOffloadingHandlers(
                file_mapper=self.file_mapper,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                gpu_block_size=self._attn_gpu_block_size,
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

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

import os
from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from llmd_fs_backend import get_logger
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import (
    DEFAULT_MAX_STAGING_MEMORY_GB,
    DEFAULT_MAX_WRITE_QUEUED_SECONDS,
    DEFAULT_READ_PREFERRING_WORKERS_RATIO,
    DEFAULT_THREADS_PER_GPU,
    StorageOffloadingHandlers,
)

DEFAULT_STORAGE_BLOCK_SIZE = 256

logger = get_logger()


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

        assert len(self.gpu_block_size) == 1, (
            f"Expected exactly one KV cache group, got {len(self.gpu_block_size)}"
        )

        hash_block_size = vllm_config.cache_config.block_size
        assert self.offloaded_block_size % hash_block_size == 0, (
            "offloaded_block_size must be a multiple of hash_block_size"
        )
        self.gpu_blocks_per_file = self.offloaded_block_size // hash_block_size

        self.read_preferring_ratio = float(
            self.extra_config.get(
                "read_preferring_ratio", DEFAULT_READ_PREFERRING_WORKERS_RATIO
            )
        )
        self.max_write_queued_seconds = float(
            self.extra_config.get(
                "max_write_queued_seconds", DEFAULT_MAX_WRITE_QUEUED_SECONDS
            )
        )

        # Metadata Cache Max Entries (0 = disabled)
        # Prioritize ENV variable VLLM_LLMD_FS_METADATA_CACHE_MAX_ENTRIES
        self.metadata_cache_max_entries = int(
            os.environ.get(
                "VLLM_LLMD_FS_METADATA_CACHE_MAX_ENTRIES",
                self.extra_config.get("metadata_cache_max_entries", 0),
            )
        )

        # Time-to-Live for the Metadata Cache positive entries
        # (default: 300 seconds / 5 minutes)
        # Prioritize ENV variable VLLM_LLMD_FS_METADATA_CACHE_TTL_SECS
        self.metadata_cache_ttl_secs = int(
            os.environ.get(
                "VLLM_LLMD_FS_METADATA_CACHE_TTL_SECS",
                self.extra_config.get("metadata_cache_ttl_secs", 300),
            )
        )

        parallel_config = vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        assert parallel_config.world_size == tp_size * pp_size * pcp_size

        self.file_mapper = FileMapper.from_vllm_config(
            root_dir=shared_storage_path,
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            gpu_blocks_per_file=self.gpu_blocks_per_file,
        )
        self.file_mapper.write_run_config()

        logger.info(
            "SharedStorageOffloadingSpec initialized: "
            "shared_storage_path=%s, "
            "offloaded_block_size=%d, "
            "gpu_blocks_per_file=%d, "
            "threads_per_gpu=%d, "
            "max_staging_memory_gb=%d, "
            "read_preferring_ratio=%.2f, "
            "max_write_queued_seconds=%.2f, "
            "metadata_cache_max_entries=%d, "
            "metadata_cache_ttl_secs=%ds",
            shared_storage_path,
            self.offloaded_block_size,
            self.gpu_blocks_per_file,
            self.threads_per_gpu,
            self.max_staging_memory_gb,
            self.read_preferring_ratio,
            self.max_write_queued_seconds,
            self.metadata_cache_max_entries,
            self.metadata_cache_ttl_secs,
        )

    def get_manager(self) -> OffloadingManager:
        assert self.vllm_config.parallel_config.rank == 0, "Scheduler rank should be 0"
        if not self._manager:
            backend = self.extra_config.get("backend", "POSIX")
            if backend == "OBJ":
                from llmd_nixl.manager import NixlStorageOffloadingManager

                self.extra_config.setdefault("storage_medium", "OBJECT_STORE")
                self._manager = NixlStorageOffloadingManager(
                    file_mapper=self.file_mapper,
                    extra_config=self.extra_config,
                )
            else:
                self.extra_config.setdefault("storage_medium", "SHARED_STORAGE")
                self._manager = SharedStorageOffloadingManager(
                    file_mapper=self.file_mapper,
                    extra_config=self.extra_config,
                    metadata_cache_max_entries=self.metadata_cache_max_entries,
                    metadata_cache_ttl_secs=self.metadata_cache_ttl_secs,
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            backend = self.extra_config.get("backend", "POSIX")
            if backend == "OBJ":
                from llmd_nixl.worker import NixlStorageOffloadingHandlers

                handlers_cls = NixlStorageOffloadingHandlers
            else:
                handlers_cls = StorageOffloadingHandlers
            self._handlers = handlers_cls(
                file_mapper=self.file_mapper,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                gpu_block_size=self.gpu_block_size[0],
                kv_caches=kv_caches,
                threads_per_gpu=self.threads_per_gpu,
                max_staging_memory_gb=self.max_staging_memory_gb,
                read_preferring_ratio=self.read_preferring_ratio,
                max_write_queued_seconds=self.max_write_queued_seconds,
                extra_config=self.extra_config,
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

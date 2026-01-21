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

import math
import os

import storage_offload
import torch
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec

logger = init_logger(__name__)

# ----------------------------------------------------------------------
# Base Storage Offloading Handler
# ----------------------------------------------------------------------
DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_THREADS_PER_GPU = 64


class BaseStorageOffloadingHandler(OffloadingHandler):
    """
    BaseStorageOffloadingHandler handles transfers for both directions,
    either GPU->Storage (PUT) or Storage->GPU (GET).
    """

    def __init__(
        self,
        gpu_blocks_per_file: int,
        file_mapper: FileMapper,
        engine: storage_offload.StorageOffloadEngine,
    ):
        """
        Initialize a SingleStorageDirectionOffloadingHandler.

        Args:
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            file_mapper: The FileMapper mapping blocks to files.
            engine: the storage engine.
        """
        self.file_mapper = file_mapper
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.engine = engine

    def get_finished(self) -> list[TransferResult]:
        """
        Poll finished async transfers.

        Returns:
            List of completed transfer results.
        """
        return self.engine.get_finished()

    def wait(self, job_ids: set[int]):
        """
        Block until the specified transfer jobs complete.

        Args:
            job_ids: Set of job IDs to wait for.
        """
        for job_id in job_ids:
            self.engine.wait_job(job_id)

    def _build_file_block_mapping(
        self,
        block_hashes,
        block_ids,
    ):
        """
        Build per-file block ID lists for grouped transfers.

        Returns:
            tuple[list[str], list[list[int]]]
                - file paths
                - per-file block ID lists
        """
        files = []
        per_file_block_ids = []

        # The first file in get may contain fewer blocks than gpu_blocks_per_file
        first_size = (
            len(block_ids) % self.gpu_blocks_per_file or self.gpu_blocks_per_file
        )

        start = 0
        size = first_size

        for block_hash in block_hashes:
            end = min(start + size, len(block_ids))
            block_ids_chunk = block_ids[start:end]

            # Build file path for this group of blocks
            files.append(self.file_mapper.get_file_name(block_hash))
            per_file_block_ids.append(block_ids_chunk)

            start += size
            size = self.gpu_blocks_per_file

        return files, per_file_block_ids


class GPUToStorageHandler(BaseStorageOffloadingHandler):
    """Handler for GPU -> Storage (PUT) transfers."""

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Launch an asynchronous transfer GPU -> Storage.

        Args:
            job_id: Unique identifier for the transfer job.
            spec: Transfer specification describing source and destination
                block IDs and file hashes.

        Returns:
            True if the transfer was successfully submitted.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, GPULoadStoreSpec)
        assert isinstance(dst_spec, SharedStorageLoadStoreSpec)

        dst_files, per_file_block_ids = self._build_file_block_mapping(
            block_hashes=dst_spec.block_hashes,
            block_ids=src_spec.block_ids,
        )

        # Submit async PUT transfer
        return self.engine.async_store_gpu_blocks(job_id, dst_files, per_file_block_ids)


class StorageToGPUHandler(BaseStorageOffloadingHandler):
    """Handler for asynchronous transfers from storage to GPU."""

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Launch an asynchronous transfer Storage -> GPU.

        Args:
            job_id: Unique identifier for the transfer job.
            spec: Transfer specification describing source and destination
                block IDs and file hashes.

        Returns:
            True if the transfer was successfully submitted.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, SharedStorageLoadStoreSpec)
        assert isinstance(dst_spec, GPULoadStoreSpec)

        src_files, per_file_block_ids = self._build_file_block_mapping(
            block_hashes=src_spec.block_hashes,
            block_ids=dst_spec.block_ids,
        )

        # Submit async GET transfer
        return self.engine.async_load_gpu_blocks(job_id, src_files, per_file_block_ids)


class StorageOffloadingHandlers:
    """Base handler with common helpers for Storage offloading."""

    def __init__(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        file_mapper: FileMapper,
        gpu_block_size: int,
        gpu_blocks_per_file: int,
        threads_per_gpu: int,
        max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,
    ):
        threads_per_gpu = min(threads_per_gpu, int(os.cpu_count()))
        tensors, kernel_block_size = StorageOffloadingHandlers._get_tensors(
            kv_caches, attn_backends
        )
        assert tensors
        assert gpu_block_size % kernel_block_size == 0

        kernel_blocks_per_gpu_block = gpu_block_size // kernel_block_size

        # Compute staging memory buffer size
        buffer_size_mb = self._compute_buffer_size_mb(
            tensors, gpu_blocks_per_file, kernel_blocks_per_gpu_block
        )

        # Adjust threads_per_gpu if exceeding max_staging_memory_gb
        if buffer_size_mb * threads_per_gpu > max_staging_memory_gb * 1024:
            threads_per_gpu = min(
                threads_per_gpu, int(max_staging_memory_gb * 1024 / buffer_size_mb)
            )
            logger.warning(
                f"Adjusted threads_per_gpu to {threads_per_gpu} due to "
                f"max_staging_memory_gb {max_staging_memory_gb} "
                f"limit (buffer_size_mb={buffer_size_mb})."
            )

        # Initialize storage offload resources for async transfers
        self.engine = storage_offload.StorageOffloadEngine(
            io_threads=threads_per_gpu,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
        )

        logger.info(
            f"StorageOffloadingHandlers: "
            f"threads_per_gpu={threads_per_gpu},"
            f"offloading block_size={gpu_blocks_per_file * gpu_block_size}, "
            f"staging_buffer_size_mb={buffer_size_mb}, "
            f"max_staging_memory_gb={max_staging_memory_gb}, "
        )

        self.gpu_to_storage_handler = GPUToStorageHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
        )

        self.storage_to_gpu_handler = StorageToGPUHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
        )

    def _compute_buffer_size_mb(
        self,
        tensors: list[torch.Tensor],
        gpu_blocks_per_file: int,
        kernel_blocks_per_gpu_block: int,
    ):
        """
        Estimate staging memory size in MB, applying min/max limits.

        Args:
            tensors: List of KV-cache tensors used to infer per-block memory usage.
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            kernel_blocks_per_gpu_block: Number of kernel blocks grouped into
                                         a single GPU block.

        Returns:
            Estimated staging buffer size in megabytes.
        """
        kernel_block_size_in_bytes = 0
        for tensor in tensors:
            kernel_block_size_in_bytes += tensor.stride(0) * tensor.element_size()
        kernel_blocks_per_file = kernel_blocks_per_gpu_block * gpu_blocks_per_file
        file_size_in_bytes = kernel_block_size_in_bytes * kernel_blocks_per_file
        file_size_mb = math.ceil(file_size_in_bytes / (1 << 20))
        return file_size_mb

    @staticmethod
    def _get_tensors(
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> tuple[list[torch.Tensor], int]:
        """
        Splits the given KV caches to tensors such that
            each tensor shape is (num_blocks, ...).

        Returns:
            (list_of_kv_cache_tensors, kernel_block_size)
        """
        tensors: list[torch.Tensor] = []
        kernel_block_size: int | None = None

        for layer_name, gpu_tensor in kv_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]

            # Generate a reference KV-cache shape using known parameters.
            # We compare gpu_shape with this synthetic shape to infer the layout.
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            split_k_and_v = False
            has_layers_dim = False
            if len(gpu_shape) != len(test_shape):
                # Case 1: Cross-layer tensor - an extra layer dimension exists.
                # In this case, num_blocks is the leading dimension.
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] == 1234:
                # Case 2: Standard layout - each element represents a single layer with
                # tensor shaped as (num_blocks, ...).
                # The first dimension matches num_blocks.
                pass
            else:
                # Case 3: (2, num_blocks, ...) - standard layout but with KV first:
                # (2, num_blocks, heads, block_size, head_size).
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            if split_k_and_v:
                # split tensor to k-tensor and v-tensor
                for sub_tensor in gpu_tensor:
                    tensors.append(sub_tensor)
            else:
                tensors.append(gpu_tensor)

            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim
                )
                assert len(kv_cache_stride_order) == len(gpu_shape)
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(gpu_shape)))

            # permute test_shape according to stride_order
            test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # find block_size (16) dimension index
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]

        assert len({t.stride(0) for t in tensors}) == 1, (
            "All KV-cache tensors must have the same block element stride."
        )
        assert kernel_block_size
        return tensors, kernel_block_size

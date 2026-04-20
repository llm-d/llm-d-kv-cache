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
import time

import storage_offload
import torch
from vllm.v1.kv_offload.abstract import get_offload_group_idx
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
    TransferType,
)

from llmd_fs_backend import _logger as logger
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec

# ----------------------------------------------------------------------
# Base Storage Offloading Handler
# ----------------------------------------------------------------------
DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_THREADS_PER_GPU = 64
DEFAULT_READ_PREFERRING_WORKERS_RATIO = 0.75


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
        transfer_type: TransferType,
        per_block_bytes: int,
    ):
        """
        Initialize a SingleStorageDirectionOffloadingHandler.

        Args:
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            file_mapper: The FileMapper mapping blocks to files.
            engine: the storage engine.
            transfer_type: The type of transfer (src, dst) for metrics.
            per_block_bytes: Size of a single GPU block in bytes.
        """
        self.file_mapper = file_mapper
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.engine = engine
        self.transfer_type = transfer_type
        self.per_block_bytes = per_block_bytes

        # Maps job_id -> (submit_time, transfer_size_bytes).
        # Shared across handlers via StorageOffloadingHandlers.
        self._pending_jobs: dict[int, tuple[float, int]] = {}

    def _record_job(self, job_id: int, num_blocks: int):
        """Record job submission metadata for metrics."""
        transfer_size = num_blocks * self.per_block_bytes
        self._pending_jobs[job_id] = (
            time.monotonic(),
            transfer_size,
        )

    def get_finished(self) -> list[TransferResult]:
        """
        Poll finished async transfers.

        Returns:
            List of completed transfer results.
        """
        now = time.monotonic()
        results = []
        for job_id, success in self.engine.get_finished():
            job_info = self._pending_jobs.pop(job_id, None)
            if job_info is not None:
                submit_time, transfer_size = job_info
                transfer_time = now - submit_time
                results.append(
                    TransferResult(
                        job_id=job_id,
                        success=success,
                        transfer_size=transfer_size,
                        transfer_time=transfer_time,
                        transfer_type=self.transfer_type,
                    )
                )
                logger.debug(
                    "Transfer finished: job_id=%d status=%s "
                    "size=%.2f [MB] time=%.3f [s] throughput=%.2f [GB/s] type=%s",
                    job_id,
                    "OK" if success else "FAIL",
                    transfer_size / (1 << 20),
                    transfer_time,
                    (transfer_size / transfer_time if transfer_time > 0 else 0)
                    / (1 << 30),
                    f"{self.transfer_type[0]}->{self.transfer_type[1]}",
                )
            else:
                logger.warning(
                    "Transfer finished with unknown job_id=%d, metrics unavailable",
                    job_id,
                )
                results.append(TransferResult(job_id=job_id, success=success))
        return results

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
        keys,
        block_ids,
    ):
        """
        Build per-file block ID lists for grouped transfers.

        Args:
            keys: OffloadKeys identifying the files for a single group.
            block_ids: GPU block IDs for this group.

        Returns:
            tuple[list[str], list[list[int]]]
                - file paths
                - per-file block ID lists
        """
        files = []
        per_file_block_ids = []

        # Partial first file inferred from remainder (works when the end of
        # the load is aligned to an offloaded-block boundary, which is the
        # vllm scheduler's invariant for both PUT and GET).
        first_size = (
            len(block_ids) % self.gpu_blocks_per_file or self.gpu_blocks_per_file
        )

        start = 0
        size = first_size

        for key in keys:
            end = min(start + size, len(block_ids))
            block_ids_chunk = block_ids[start:end]

            # Pass full OffloadKey so different groups get different files
            files.append(self.file_mapper.get_file_name(key))
            per_file_block_ids.append(block_ids_chunk)

            start += size
            size = self.gpu_blocks_per_file

        return files, per_file_block_ids

    def _build_transfer(
        self,
        gpu_spec: GPULoadStoreSpec,
        storage_spec: SharedStorageLoadStoreSpec,
    ) -> tuple[list[int], list[str], list[list[int]]]:
        """
        Build a flat per-file transfer from a TransferSpec that may contain
        multiple KV cache groups.

        Returns:
            tuple[list[int], list[str], list[list[int]]]:
                - group_indices[i] = KV cache group index for files[i]
                - files[i] = file path
                - per_file_block_ids[i] = GPU block IDs to transfer for files[i]
        """
        group_sizes = gpu_spec.group_sizes

        all_group_indices: list[int] = []
        all_files: list[str] = []
        all_block_ids: list[list[int]] = []

        block_offset = 0
        key_offset = 0
        for group_idx, group_size in enumerate(group_sizes):
            if group_size == 0:
                continue

            num_files = math.ceil(group_size / self.gpu_blocks_per_file)

            group_block_ids = gpu_spec.block_ids[
                block_offset : block_offset + group_size
            ]
            group_keys = storage_spec.keys[key_offset : key_offset + num_files]

            # Sanity check: first key for this slice should encode the
            # expected group_idx — catches scheduler-ordering bugs early.
            if group_keys:
                assert get_offload_group_idx(group_keys[0]) == group_idx, (
                    f"Expected group_idx={group_idx} but key encodes "
                    f"{get_offload_group_idx(group_keys[0])}"
                )

            group_files, group_per_file_block_ids = self._build_file_block_mapping(
                keys=group_keys,
                block_ids=group_block_ids,
            )

            all_group_indices.extend([group_idx] * len(group_files))
            all_files.extend(group_files)
            all_block_ids.extend(group_per_file_block_ids)

            block_offset += group_size
            key_offset += num_files

        return all_group_indices, all_files, all_block_ids


class GPUToStorageHandler(BaseStorageOffloadingHandler):
    """Handler for GPU -> Storage (PUT) transfers."""

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Launch an asynchronous transfer GPU -> Storage.

        For HMA models, builds per-file group_indices so the engine picks
        the correct tensor subset per file from a single submission.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, GPULoadStoreSpec)
        assert isinstance(dst_spec, SharedStorageLoadStoreSpec)

        group_indices, dst_files, per_file_block_ids = self._build_transfer(
            src_spec, dst_spec
        )
        if not dst_files:
            return False

        success = self.engine.async_store_gpu_blocks(
            job_id, group_indices, dst_files, per_file_block_ids
        )
        if success:
            total_blocks = sum(len(ids) for ids in per_file_block_ids)
            self._record_job(job_id, total_blocks)
        return success


class StorageToGPUHandler(BaseStorageOffloadingHandler):
    """Handler for asynchronous transfers from storage to GPU."""

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Launch an asynchronous transfer Storage -> GPU.

        For HMA models, builds per-file group_indices so the engine picks
        the correct tensor subset per file from a single submission.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, SharedStorageLoadStoreSpec)
        assert isinstance(dst_spec, GPULoadStoreSpec)

        group_indices, src_files, per_file_block_ids = self._build_transfer(
            dst_spec, src_spec
        )
        if not src_files:
            return False

        success = self.engine.async_load_gpu_blocks(
            job_id, group_indices, src_files, per_file_block_ids
        )
        if success:
            total_blocks = sum(len(ids) for ids in per_file_block_ids)
            self._record_job(job_id, total_blocks)
        return success


class StorageOffloadingHandlers:
    """Base handler with common helpers for Storage offloading."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        file_mapper: FileMapper,
        gpu_block_size: int,
        gpu_blocks_per_file: int,
        threads_per_gpu: int,
        max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,
        read_preferring_ratio: float = DEFAULT_READ_PREFERRING_WORKERS_RATIO,
    ):
        threads_per_gpu = min(threads_per_gpu, int(os.cpu_count()))
        tensors = [ct.tensor for ct in kv_caches.tensors]
        assert tensors

        # Per-group tensor indices into the flat `tensors` list.
        # For single-group models this is a single list covering all tensors
        # used by that group. For HMA models each group has its own subset.
        group_tensor_indices: list[list[int]] = [
            [ref.tensor_idx for ref in group_refs]
            for group_refs in kv_caches.group_data_refs
        ]
        assert group_tensor_indices, "CanonicalKVCaches has no groups"

        # With CanonicalKVCaches, tensors are already canonicalized with
        # shape (num_blocks, page_size_bytes). kernel_block_size is 1
        # since each "block" in the canonical tensor is already a full block.
        kernel_blocks_per_gpu_block = 1

        # Compute staging memory buffer size sized for the largest group
        # (one buffer serves any group's transfer).
        buffer_size_mb = self._compute_buffer_size_mb(
            tensors,
            group_tensor_indices,
            gpu_blocks_per_file,
            kernel_blocks_per_gpu_block,
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

        # Calculate number of read-preferring workers
        read_preferring_workers = max(1, int(threads_per_gpu * read_preferring_ratio))

        # Initialize storage offload resources for async transfers
        self.engine = storage_offload.StorageOffloadEngine(
            io_threads=threads_per_gpu,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
            group_tensor_indices=group_tensor_indices,
            read_preferring_workers=read_preferring_workers,
        )

        # Compute per-GPU-block size in bytes for metrics across all tensors.
        # For HMA models with multiple groups this is an approximation.
        kernel_block_bytes = sum(t.stride(0) * t.element_size() for t in tensors)
        per_block_bytes = kernel_block_bytes * kernel_blocks_per_gpu_block
        logger.info(
            f"StorageOffloadingHandlers: "
            f"threads_per_gpu={threads_per_gpu}, "
            f"num_groups={len(group_tensor_indices)}, "
            f"offloading block_size={gpu_blocks_per_file * gpu_block_size}, "
            f"staging_buffer_size_mb={buffer_size_mb}, "
            f"max_staging_memory_gb={max_staging_memory_gb}, "
            f"read_preferring_workers={read_preferring_workers}, "
        )

        # Shared across both handlers since the engine has a single completion queue.
        pending_jobs: dict[int, tuple[float, int, TransferType]] = {}

        self.gpu_to_storage_handler = GPUToStorageHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
            transfer_type=("GPU", "SHARED_STORAGE"),
            per_block_bytes=per_block_bytes,
        )
        self.gpu_to_storage_handler._pending_jobs = pending_jobs

        self.storage_to_gpu_handler = StorageToGPUHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
            transfer_type=("SHARED_STORAGE", "GPU"),
            per_block_bytes=per_block_bytes,
        )
        self.storage_to_gpu_handler._pending_jobs = pending_jobs

    def _compute_buffer_size_mb(
        self,
        tensors: list[torch.Tensor],
        group_tensor_indices: list[list[int]],
        gpu_blocks_per_file: int,
        kernel_blocks_per_gpu_block: int,
    ):
        """
        Estimate staging memory size in MB, sized to fit the largest group.

        Args:
            tensors: Flat list of canonical KV-cache tensors.
            group_tensor_indices: Per-group tensor indices into `tensors`.
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            kernel_blocks_per_gpu_block: Number of kernel blocks grouped into
                                         a single GPU block.

        Returns:
            Estimated staging buffer size in megabytes.
        """
        max_group_bytes = 0
        for indices in group_tensor_indices:
            group_bytes = 0
            for idx in indices:
                tensor = tensors[idx]
                group_bytes += tensor.stride(0) * tensor.element_size()
            max_group_bytes = max(max_group_bytes, group_bytes)
        kernel_blocks_per_file = kernel_blocks_per_gpu_block * gpu_blocks_per_file
        file_size_in_bytes = max_group_bytes * kernel_blocks_per_file
        file_size_mb = math.ceil(file_size_in_bytes / (1 << 20))
        return file_size_mb

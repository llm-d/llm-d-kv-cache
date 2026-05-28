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
from typing import Protocol, runtime_checkable

import storage_offload
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
    TransferType,
)

from llmd_fs_backend import _logger as logger
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec


@runtime_checkable
class StorageEngine(Protocol):
    """Common interface shared by all storage engine backends.

    Satisfied structurally by both llmd_nixl.nixl_offload.StorageOffloadEngine
    and the C++ storage_offload.StorageOffloadEngine.
    """

    def async_store_gpu_blocks(
        self,
        job_id: int,
        group_indices: list,
        files: list,
        block_ids: list,
    ) -> bool: ...
    def async_load_gpu_blocks(
        self,
        job_id: int,
        group_indices: list,
        files: list,
        block_ids: list,
    ) -> bool: ...
    def get_finished(self) -> list: ...
    def wait_job(self, job_id: int) -> None: ...
    def shutdown(self) -> None: ...


# ----------------------------------------------------------------------
# Base Storage Offloading Handler
# ----------------------------------------------------------------------
DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_THREADS_PER_GPU = 64
DEFAULT_READ_PREFERRING_WORKERS_RATIO = 0.75
DEFAULT_MAX_WRITE_QUEUED_SECONDS = 30.0


class BaseStorageOffloadingHandler(OffloadingHandler):
    """
    BaseStorageOffloadingHandler handles transfers for both directions,
    either GPU->Storage (PUT) or Storage->GPU (GET).
    """

    def __init__(
        self,
        gpu_blocks_per_file: int,
        file_mapper: FileMapper,
        engine: StorageEngine,
        transfer_type: TransferType,
        per_block_bytes: int,
        per_group_block_bytes: list[int],
    ):
        """
        Initialize a SingleStorageDirectionOffloadingHandler.

        Args:
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            file_mapper: The FileMapper mapping blocks to files.
            engine: the storage engine.
            transfer_type: The type of transfer (src, dst) for metrics.
            per_block_bytes: Max per-block bytes across groups (init logging only).
            per_group_block_bytes: Bytes/block per group (drives per-job metrics).
        """
        self.file_mapper = file_mapper
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.engine = engine
        self.transfer_type = transfer_type
        self.per_block_bytes = per_block_bytes
        self.per_group_block_bytes = per_group_block_bytes

        # Maps job_id -> (submit_time, transfer_size_bytes).
        # Shared across handlers via StorageOffloadingHandlers.
        self._pending_jobs: dict[int, tuple[float, int]] = {}

    def _record_job(self, job_id: int, transfer_size_bytes: int):
        """Record job submission metadata for metrics.

        Args:
            transfer_size_bytes: Exact number of bytes the worker will
                transfer for this job — caller computes it from the actual
                (group, blocks_per_file) it is submitting, not from a
                global per-block estimate.
        """
        self._pending_jobs[job_id] = (
            time.monotonic(),
            transfer_size_bytes,
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
        start_block_idx: int,
    ):
        """
        Build per-file block ID lists for one group's transfer, using the
        group's logical start index to handle unaligned head/tail correctly.

        A file holds `gpu_blocks_per_file` logically-consecutive GPU blocks.
        Files are aligned at multiples of `gpu_blocks_per_file` in logical
        space. A group covers logical blocks
        [start_block_idx, start_block_idx + len(block_ids)) — which may
        start AND/OR end mid-file. We split the group across the
        intersecting files accordingly.

        Args:
            keys: OffloadKeys identifying the files for this group.
                One key per file the group spans.
            block_ids: GPU block IDs for this group (in logical order).
            start_block_idx: Logical block index of the first GPU block in
                this group (from `gpu_spec.block_indices[group_idx]`).

        Returns:
            tuple[list[str], list[list[int]]]
                - file paths (len == number of files the group spans)
                - per-file block ID lists
        """
        gpb = self.gpu_blocks_per_file
        n_blocks = len(block_ids)
        if n_blocks == 0:
            return [], []

        end_block_idx = start_block_idx + n_blocks  # exclusive
        start_file_idx = start_block_idx // gpb
        end_file_idx = (end_block_idx - 1) // gpb
        num_files = end_file_idx - start_file_idx + 1
        assert len(keys) == num_files, (
            f"expected {num_files} keys for group starting at "
            f"block_idx={start_block_idx} with {n_blocks} blocks "
            f"(gpu_blocks_per_file={gpb}), got {len(keys)}"
        )

        files = []
        per_file_block_ids = []
        block_offset = 0
        for f_idx in range(num_files):
            file_logical_lo = (start_file_idx + f_idx) * gpb
            file_logical_hi = file_logical_lo + gpb  # exclusive
            # Slice of this group that lives in file f_idx.
            # max() handles head-unaligned f_idx=0 (file_logical_lo
            # is rounded down to the file boundary).
            slice_lo = max(start_block_idx, file_logical_lo)
            slice_hi = min(end_block_idx, file_logical_hi)
            slice_size = slice_hi - slice_lo

            # Pass full OffloadKey so different groups get different files
            files.append(self.file_mapper.get_file_name(keys[f_idx]))
            per_file_block_ids.append(
                block_ids[block_offset : block_offset + slice_size]
            )
            block_offset += slice_size

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
        group_block_indices = gpu_spec.block_indices

        all_group_indices: list[int] = []
        all_files: list[str] = []
        all_block_ids: list[list[int]] = []

        block_offset = 0
        key_offset = 0
        gpb = self.gpu_blocks_per_file
        for group_idx, group_size in enumerate(group_sizes):
            if group_size == 0:
                continue

            # Number of files this group spans, computed from the group's
            # logical start index (block_indices[group_idx]) so unaligned
            # heads/tails are counted correctly.
            start_block_idx = group_block_indices[group_idx]
            end_block_idx = start_block_idx + group_size  # exclusive
            num_files = (end_block_idx - 1) // gpb - start_block_idx // gpb + 1

            group_block_ids = gpu_spec.block_ids[
                block_offset : block_offset + group_size
            ]
            group_keys = storage_spec.keys[key_offset : key_offset + num_files]

            # Sanity check: key's encoded group_idx matches expected group.
            if group_keys:
                assert get_offload_group_idx(group_keys[0]) == group_idx, (
                    f"Expected group_idx={group_idx} but key encodes "
                    f"{get_offload_group_idx(group_keys[0])}"
                )

            group_files, group_per_file_block_ids = self._build_file_block_mapping(
                keys=group_keys,
                block_ids=group_block_ids,
                start_block_idx=start_block_idx,
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
        """Launch an asynchronous transfer GPU -> Storage."""
        src_spec, dst_spec = spec
        assert isinstance(src_spec, GPULoadStoreSpec)
        assert isinstance(dst_spec, SharedStorageLoadStoreSpec)

        group_indices, dst_files, per_file_block_ids = self._build_transfer(
            src_spec, dst_spec
        )
        if not dst_files:
            return False

        total_blocks = sum(len(ids) for ids in per_file_block_ids)
        # Exact bytes for this job: each file's (group, blocks_in_file) maps
        # to per_group_block_bytes[group] * blocks_in_file.
        total_bytes = sum(
            len(ids) * self.per_group_block_bytes[g]
            for g, ids in zip(group_indices, per_file_block_ids)
        )
        # INFO so it surfaces without STORAGE_LOG_LEVEL=DEBUG; "Transfer
        # finished" still requires DEBUG because completion lines are noisier
        # and only fire when get_finished() is polled.
        logger.info(
            "PUT started: job_id=%d files=%d blocks=%d size=%.2f [MB]",
            job_id,
            len(dst_files),
            total_blocks,
            total_bytes / (1 << 20),
        )
        success = self.engine.async_store_gpu_blocks(
            job_id, group_indices, dst_files, per_file_block_ids
        )
        if success:
            self._record_job(job_id, total_bytes)
        return success


class StorageToGPUHandler(BaseStorageOffloadingHandler):
    """Handler for asynchronous transfers from storage to GPU."""

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """Launch an asynchronous transfer Storage -> GPU."""
        src_spec, dst_spec = spec
        assert isinstance(src_spec, SharedStorageLoadStoreSpec)
        assert isinstance(dst_spec, GPULoadStoreSpec)

        group_indices, src_files, per_file_block_ids = self._build_transfer(
            dst_spec, src_spec
        )
        if not src_files:
            return False

        total_blocks = sum(len(ids) for ids in per_file_block_ids)
        total_bytes = sum(
            len(ids) * self.per_group_block_bytes[g]
            for g, ids in zip(group_indices, per_file_block_ids)
        )
        logger.info(
            "GET started: job_id=%d files=%d blocks=%d size=%.2f [MB]",
            job_id,
            len(src_files),
            total_blocks,
            total_bytes / (1 << 20),
        )
        success = self.engine.async_load_gpu_blocks(
            job_id, group_indices, src_files, per_file_block_ids
        )
        if success:
            self._record_job(job_id, total_bytes)
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
        max_write_queued_seconds: float = DEFAULT_MAX_WRITE_QUEUED_SECONDS,
        extra_config: dict | None = None,
    ):
        extra_config = extra_config or {}
        threads_per_gpu = min(threads_per_gpu, int(os.cpu_count()))
        tensors = [ct.tensor for ct in kv_caches.tensors]
        assert tensors

        # Source-of-truth per-group refs; derive the two int views below.
        group_data_refs: list[list] = list(kv_caches.group_data_refs)
        assert group_data_refs, "CanonicalKVCaches has no groups"

        # Identity view: which tensors per group → TensorCopier copies them.
        group_tensor_indices: list[list[int]] = [
            [ref.tensor_idx for ref in group_refs] for group_refs in group_data_refs
        ]

        # Bytes view: bytes/block per group (sum of page_size_bytes across
        # layers) → C++ engine sizes staging buffer, handlers report job bytes.
        per_group_block_bytes: list[int] = [
            sum(ref.page_size_bytes for ref in group_refs)
            for group_refs in group_data_refs
        ]
        self.per_group_block_bytes = per_group_block_bytes

        valid_gds_modes = [
            "disabled",
            "read_only",
            "write_only",
            "read_write",
            "bb_read_only",
            "bb_write_only",
            "bb_read_write",
        ]

        gds_mode = extra_config.get("gds_mode", "disabled")
        if gds_mode not in valid_gds_modes:
            logger.warning(
                f"Invalid GDS mode '{gds_mode}', defaulting to 'disabled'. "
                f"Valid options: {', '.join(valid_gds_modes)}"
            )
            gds_mode = "disabled"

        # Compute staging memory buffer size sized for the largest group
        # (one buffer serves any group's transfer).
        buffer_size_mb = self._compute_buffer_size_mb(
            per_group_block_bytes, gpu_blocks_per_file
        )

        # Adjust threads_per_gpu if exceeding max_staging_memory_gb.
        # Skip for full-GDS modes — CPU staging buffer is not used.
        _gds_uses_no_staging = gds_mode in ("read_write", "bb_read_write")
        if (
            not _gds_uses_no_staging
            and buffer_size_mb * threads_per_gpu > max_staging_memory_gb * 1024
        ):
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
        self.engine = self._create_engine(
            io_threads=threads_per_gpu,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
            group_tensor_indices=group_tensor_indices,
            per_group_block_bytes=per_group_block_bytes,
            read_preferring_workers=read_preferring_workers,
            max_write_queued_seconds=max_write_queued_seconds,
            extra_config=extra_config,
            gds_mode=gds_mode,
        )

        # Per-block bytes for throughput metrics. Handlers select the exact
        # value for the (group, blocks) they are transferring via
        # per_group_block_bytes; this float is only used for the global
        # init-time log line below.
        per_block_bytes = max(per_group_block_bytes)
        logger.info(
            f"StorageOffloadingHandlers: "
            f"threads_per_gpu={threads_per_gpu}, "
            f"num_groups={len(group_tensor_indices)}, "
            f"gds_mode={gds_mode}, "
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
            per_group_block_bytes=per_group_block_bytes,
        )
        self.gpu_to_storage_handler._pending_jobs = pending_jobs

        self.storage_to_gpu_handler = StorageToGPUHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
            transfer_type=("SHARED_STORAGE", "GPU"),
            per_block_bytes=per_block_bytes,
            per_group_block_bytes=per_group_block_bytes,
        )
        self.storage_to_gpu_handler._pending_jobs = pending_jobs

    def _compute_buffer_size_mb(
        self,
        per_group_block_bytes: list[int],
        gpu_blocks_per_file: int,
    ):
        """
        Estimate staging memory size in MB, sized to fit the largest group.

        Args:
            per_group_block_bytes: bytes-per-block per group, summed across
                the group's layers (from CanonicalKVCacheRef.page_size_bytes).
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.

        Returns:
            Estimated staging buffer size in megabytes.
        """
        max_group_bytes = max(per_group_block_bytes)
        file_size_in_bytes = max_group_bytes * gpu_blocks_per_file
        file_size_mb = math.ceil(file_size_in_bytes / (1 << 20))
        return file_size_mb

    def _create_engine(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: list,
        group_tensor_indices: list[list[int]],
        per_group_block_bytes: list[int],
        read_preferring_workers: int,
        max_write_queued_seconds: float,
        extra_config: dict,
        gds_mode: str,
    ) -> StorageEngine:
        return storage_offload.StorageOffloadEngine(
            io_threads,
            gpu_blocks_per_file,
            tensors,
            group_tensor_indices,
            per_group_block_bytes,
            read_preferring_workers,
            gds_mode,
            max_write_queued_seconds,
        )

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
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
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
        success = self.engine.async_store_gpu_blocks(
            job_id, dst_files, per_file_block_ids
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
        success = self.engine.async_load_gpu_blocks(
            job_id, src_files, per_file_block_ids
        )
        if success:
            total_blocks = sum(len(ids) for ids in per_file_block_ids)
            self._record_job(job_id, total_blocks)
        return success


class StorageOffloadingHandlers:
    """Handler factory for Storage offloading.

    Supports both single-group (standard) and multi-group (hybrid)
    models.  When ``group_layer_names`` is provided, creates
    per-group engines and handlers; otherwise falls back to the
    single-group path.
    """

    def __init__(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        file_mapper: FileMapper,
        gpu_block_size: int | tuple[int, ...],
        gpu_blocks_per_file: int | tuple[int, ...],
        threads_per_gpu: int,
        max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,
        read_preferring_ratio: float = DEFAULT_READ_PREFERRING_WORKERS_RATIO,
        # Multi-group (hybrid model) parameters:
        file_mappers: tuple[FileMapper, ...] | None = None,
        group_layer_names: tuple[tuple[str, ...], ...] | None = None,
        group_hash_block_size: tuple[int, ...] | None = None,
    ):
        threads_per_gpu = min(threads_per_gpu, int(os.cpu_count()))

        # Normalize scalar args to tuples for uniform handling.
        if isinstance(gpu_block_size, int):
            gpu_block_size = (gpu_block_size,)
        if isinstance(gpu_blocks_per_file, int):
            gpu_blocks_per_file = (gpu_blocks_per_file,)
        if file_mappers is None:
            file_mappers = (file_mapper,)
        if group_layer_names is None:
            group_layer_names = (tuple(kv_caches.keys()),)
        if group_hash_block_size is None:
            group_hash_block_size = gpu_block_size

        num_groups = len(group_layer_names)
        threads_per_group = max(1, threads_per_gpu // num_groups)
        group_budget_mb = max_staging_memory_gb * 1024 / num_groups

        # Build per-group engines and handlers.
        store_handlers: list[GPUToStorageHandler] = []
        load_handlers: list[StorageToGPUHandler] = []
        pending_jobs: dict[int, tuple[float, int, TransferType]] = {}

        for gi in range(num_groups):
            group_kv = {
                ln: kv_caches[ln]
                for ln in group_layer_names[gi]
                if ln in kv_caches
            }
            if not group_kv:
                continue
            group_attn = {
                ln: attn_backends[ln] for ln in group_kv
            }
            gbs = gpu_block_size[gi]
            gbpf = gpu_blocks_per_file[gi]
            mapper = (
                file_mappers[gi]
                if gi < len(file_mappers)
                else file_mappers[0]
            )

            tensors, kbs = self._get_tensors(group_kv, group_attn, gbs)
            assert tensors
            assert gbs % kbs == 0
            kb_per_gb = gbs // kbs

            buf_mb = self._compute_buffer_size_mb(tensors, gbpf, kb_per_gb)
            grp_threads = threads_per_group
            if buf_mb * grp_threads > group_budget_mb:
                grp_threads = max(1, int(group_budget_mb / buf_mb))

            rpw = max(1, int(grp_threads * read_preferring_ratio))
            # kb_per_gb is the canonical grouping factor: one vLLM
            # page = kb_per_gb consecutive kernel blocks.  Passing
            # this to the C++ engine ensures the on-disk format is
            # page-aligned and portable across GPUs with different
            # kernel block sizes (e.g. RTX 4080 kb=32 vs 5090 kb=64).
            engine = storage_offload.StorageOffloadEngine(
                io_threads=grp_threads,
                gpu_blocks_per_file=gbpf,
                tensors=tensors,
                read_preferring_workers=rpw,
                kernel_blocks_per_canonical_block=kb_per_gb,
            )

            kb_bytes = sum(
                t.stride(0) * t.element_size() for t in tensors
            )
            per_block = kb_bytes * kb_per_gb

            logger.info(
                "StorageOffloadingHandlers group=%d: "
                "threads=%d block_size=%d kb_per_canonical=%d "
                "staging_mb=%d",
                gi, grp_threads, gbpf * gbs, kb_per_gb, buf_mb,
            )

            sh = GPUToStorageHandler(
                engine=engine,
                file_mapper=mapper,
                gpu_blocks_per_file=gbpf,
                transfer_type=("GPU", "SHARED_STORAGE"),
                per_block_bytes=per_block,
            )
            sh._pending_jobs = pending_jobs
            store_handlers.append(sh)

            lh = StorageToGPUHandler(
                engine=engine,
                file_mapper=mapper,
                gpu_blocks_per_file=gbpf,
                transfer_type=("SHARED_STORAGE", "GPU"),
                per_block_bytes=per_block,
            )
            lh._pending_jobs = pending_jobs
            load_handlers.append(lh)

        # For single-group, expose handlers directly for
        # backward compatibility with the upstream interface.
        if len(store_handlers) == 1:
            self.gpu_to_storage_handler = store_handlers[0]
            self.storage_to_gpu_handler = load_handlers[0]
        else:
            # Multi-group: the OffloadingConnector already handles
            # per-group block splitting via TransferSpec, so the
            # first group's handler is sufficient as the entry point.
            # TODO: proper multi-group handler that dispatches to
            # the correct per-group engine based on block IDs.
            self.gpu_to_storage_handler = store_handlers[0]
            self.storage_to_gpu_handler = load_handlers[0]
            self._store_handlers = store_handlers
            self._load_handlers = load_handlers

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
        fallback_block_size: int | None = None,
    ) -> tuple[list[torch.Tensor], int]:
        """Extract per-layer tensors with block dim at position 0.

        For hybrid models, each layer may use a different attention
        backend with a different tensor layout.  Backends that don't
        expose ``get_kv_cache_shape`` (e.g., mamba) fall back to
        ``fallback_block_size``.

        Returns:
            (list_of_kv_cache_tensors, kernel_block_size)
        """
        tensors: list[torch.Tensor] = []
        kernel_block_size: int | None = None

        for layer_name, gpu_tensor in kv_caches.items():
            attn_backend = attn_backends[layer_name]
            # Hybrid models may store multi-tensor state per layer
            gpu_tensor_items = (
                list(gpu_tensor)
                if isinstance(gpu_tensor, (list, tuple))
                else [gpu_tensor]
            )

            for tensor_item in gpu_tensor_items:
                gpu_shape = tensor_item.shape
                split_k_and_v = False
                has_layers_dim = False
                block_dim = 0
                block_size_value = (
                    fallback_block_size
                    if fallback_block_size is not None
                    else gpu_shape[0]
                )

                try:
                    test_shape = attn_backend.get_kv_cache_shape(
                        num_blocks=1234,
                        block_size=16,
                        num_kv_heads=8,
                        head_size=256,
                    )

                    if len(gpu_shape) != len(test_shape):
                        assert len(gpu_shape) == len(test_shape) + 1
                        has_layers_dim = True
                        test_shape = (80,) + test_shape

                    block_dim = test_shape.index(1234)

                    if test_shape[0] == 1234:
                        pass
                    else:
                        assert test_shape[0] == 2
                        assert test_shape[1] == 1234
                        assert gpu_shape[0] == 2
                        split_k_and_v = True

                    try:
                        kv_cache_stride_order = (
                            attn_backend.get_kv_cache_stride_order(
                                include_num_layers_dimension=(
                                    has_layers_dim
                                )
                            )
                        )
                        assert len(kv_cache_stride_order) == len(
                            gpu_shape
                        )
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(
                            range(len(gpu_shape))
                        )

                    test_shape = tuple(
                        test_shape[i] for i in kv_cache_stride_order
                    )
                    block_size_idx = test_shape.index(16)
                    block_size_value = gpu_shape[block_size_idx]
                except NotImplementedError:
                    block_dim = 0

                if split_k_and_v and tensor_item.is_contiguous():
                    for sub_tensor in tensor_item:
                        tensors.append(sub_tensor)
                else:
                    normalized = (
                        torch.movedim(tensor_item, block_dim, 0)
                        if block_dim != 0
                        else tensor_item
                    )
                    tensors.append(normalized)

                if kernel_block_size is not None:
                    assert kernel_block_size == block_size_value
                else:
                    kernel_block_size = block_size_value

        assert kernel_block_size is not None

        # Cross-GPU portability is handled in the C++ TensorCopier
        # via kernel_blocks_per_canonical_block.  The Python side
        # passes kb_per_gb (page_size // kernel_block_size) to the
        # engine constructor; the C++ copy loop expands each canonical
        # block_id into the appropriate number of consecutive kernel
        # blocks for the local GPU's attention backend layout.

        return tensors, kernel_block_size

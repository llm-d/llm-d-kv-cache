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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import storage_offload
import torch
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.v1.kv_offload.worker.worker import (OffloadingHandler,
                                              TransferResult, TransferSpec)

logger = init_logger(__name__)


@dataclass
class KVCacheLayout:
    """
    Describes the KV-cache tensor layout.
    """
    num_blocks_idx: List[int]
    kv_before_num_blocks: List[bool]
    layers_before_num_blocks: List[bool]


# ----------------------------------------------------------------------
# Base Storage Offloading Handler
# ----------------------------------------------------------------------
DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_MAX_THREADS_PER_GPU = 64


class SingleStorageDirectionOffloadingHandler(OffloadingHandler):
    """
    SingleStorageDirectionOffloadingHandler handles transfers for a single direction,
    either GPU->Storage (PUT) or Storage->GPU (GET).
    """

    def __init__(
        self,
        direction: str,
        base_path: Path,
        gpu_blocks_per_file: int,
        tensors: list[torch.Tensor],
        engine: storage_offload.StorageOffloadEngine,
    ):
        """
        Initialize a SingleStorageDirectionOffloadingHandler.

        Args:
            direction: Transfer direction, either "put" (GPU->Storage) or "get" (Storage->GPU).
            base_path: Base directory for KV-cache files on shared storage.
            gpu_blocks_per_file: Number of GPU blocks grouped into a single storage file.
            tensors: List of KV-cache tensors participating in the transfer.
                For PUT, these are source GPU tensors.
                For GET, these are destination GPU tensors.
        """
        assert direction in ("put", "get")

        self.direction = direction
        self.base_path = base_path
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.tensors = tensors
        self.engine = engine

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Launch an asynchronous transfer for this direction.
        Prepare arrays of file paths, block IDs, and destination/source tensors.

        Args:
            job_id: Unique identifier for the transfer job.
            spec: Transfer specification describing source and destination
                block IDs and file hashes.

        Returns:
            True if the transfer was successfully submitted.
        """
        src_spec, dst_spec = spec

        # ------------------------------------------------------------
        # GPU -> Storage (PUT)
        # ------------------------------------------------------------
        if self.direction == "put":
            if dst_spec is None or len(dst_spec.block_hashes) == 0:
                return True

            dst_files, per_file_block_ids = self._build_file_block_mapping(
                block_hashes=dst_spec.block_hashes,
                block_ids=src_spec.block_ids,
            )

            # Submit async PUT transfer
            return self.engine.transfer_async_put(
                job_id,
                dst_files,
                self.tensors,
                per_file_block_ids,
            )

        # ------------------------------------------------------------
        # Storage -> GPU (GET)
        # ------------------------------------------------------------
        if src_spec is None or len(src_spec.block_hashes) == 0:
            return True

        src_files, per_file_block_ids = self._build_file_block_mapping(
            block_hashes=src_spec.block_hashes,
            block_ids=dst_spec.block_ids,
        )

        # Submit async GET transfer
        return self.engine.transfer_async_get(job_id, src_files,
                                              per_file_block_ids, self.tensors)

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
        first_size = (len(block_ids) % self.gpu_blocks_per_file
                      or self.gpu_blocks_per_file)

        start = 0
        size = first_size

        for block_hash in block_hashes:
            end = min(start + size, len(block_ids))
            block_ids_chunk = block_ids[start:end]

            # Build file path for this group of blocks
            files.append(
                str(
                    StorageOffloadingHandlers.get_file_name(
                        self.base_path, block_hash)))
            per_file_block_ids.append(block_ids_chunk)

            start += size
            size = self.gpu_blocks_per_file

        return files, per_file_block_ids


class StorageOffloadingHandlers:
    """Base handler with common helpers for Storage offloading."""

    def __init__(
            self,
            model_name: str,
            tp_size: int,
            tp_rank: int,
            dtype: torch.dtype,
            gpu_blocks_per_file: int,
            root_dir: str,
            attn_backends: dict[str, type[AttentionBackend]],
            kv_caches: Dict[str, torch.Tensor],
            threads_per_gpu: int,
            max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,  # in GB
    ):

        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.base_path = self.get_kv_cache_base_path(model_name, tp_size,
                                                     tp_rank, dtype, root_dir)
        self.threads_per_gpu = min(threads_per_gpu, int(os.cpu_count()),
                                   DEFAULT_MAX_THREADS_PER_GPU)
        self.max_staging_memory_gb = max_staging_memory_gb

        self.attn_backends = attn_backends
        self.tensors = list(kv_caches.values())
        # Determine KV cache layout parameters
        kv_cache_layout = self.get_kv_cache_layout(kv_caches)
        # Compute staging memory buffer size
        self.buffer_size_mb = self.compute_buffer_size_mb(
            self.tensors,
            gpu_blocks_per_file,
            num_blocks_idx=kv_cache_layout.num_blocks_idx[0])
        # Adjust threads_per_gpu if exceeding max_staging_memory_gb
        if self.buffer_size_mb * self.threads_per_gpu > self.max_staging_memory_gb * 1024:
            self.threads_per_gpu = min(
                self.threads_per_gpu,
                int(self.max_staging_memory_gb * 1024 / self.buffer_size_mb))
            logger.warning(
                f"Adjusted threads_per_gpu to {self.threads_per_gpu} due to max_staging_memory_gb {self.max_staging_memory_gb} limit "
                + f" (buffer_size_mb={self.buffer_size_mb}).")

        # Initialize storage offload resources for async transfers (assuming all layers have the same layout)
        self.engine = storage_offload.StorageOffloadEngine(
            io_threads=self.threads_per_gpu,
            staging_buffer_size_mb=self.buffer_size_mb,
            max_staging_memory_gb=self.max_staging_memory_gb,
            tp_rank=self.tp_rank,
            gpu_blocks_per_file=self.gpu_blocks_per_file,
            tensors=self.tensors,
            kv_before_blocks=kv_cache_layout.kv_before_num_blocks[0],
            layers_before_blocks=kv_cache_layout.layers_before_num_blocks[0],
            num_blocks_dimension=kv_cache_layout.num_blocks_idx[0],
        )

        logger.info(
            f"StorageOffloadingHandlers: "
            f"number_of_gpu={self.tp_size},"
            f"tp_rank={self.tp_rank},"
            f"threads_per_gpu={self.threads_per_gpu},"
            f"staging_buffer_size_mb={self.buffer_size_mb}, "
            f"max_staging_memory_gb={self.max_staging_memory_gb}, "
            f"root_dir={self.base_path},"
            f"kv_before_num_blocks={kv_cache_layout.kv_before_num_blocks[0]}, "
            f"layers_before_num_blocks={kv_cache_layout.layers_before_num_blocks[0]}, "
            f"num_blocks_dimension={kv_cache_layout.num_blocks_idx[0]}")

        self.gpu_to_storage_handler = SingleStorageDirectionOffloadingHandler(
            direction="put",
            base_path=self.base_path,
            gpu_blocks_per_file=self.gpu_blocks_per_file,
            tensors=self.tensors,
            engine=self.engine,
        )

        self.storage_to_gpu_handler = SingleStorageDirectionOffloadingHandler(
            direction="get",
            base_path=self.base_path,
            gpu_blocks_per_file=self.gpu_blocks_per_file,
            tensors=self.tensors,
            engine=self.engine,
        )

    # ----------------------------
    # Shared path helpers
    # ----------------------------
    @staticmethod
    def get_kv_cache_base_path(model_name: str, tp_size: int, tp_rank: int,
                               dtype: torch.dtype, root_dir: str) -> Path:
        """
        Build the KV-cache base path:
        <root_dir>/<model_name>/tp_<tp_size>/rank_<tp_rank>/<dtype>.

        Args:
            model_name: Model identifier.
            tp_size: Tensor-parallel world size.
            tp_rank: Tensor-parallel rank of the current process.
            dtype: Torch dtype of the KV-cache tensors.
            root_dir: Root directory for shared storage.

        Returns:
            Base path under which KV-cache files are stored or loaded.
        """
        dtype_str = str(dtype).replace("torch.", "")
        base_path = Path(
            f"{root_dir}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}")
        return base_path

    @staticmethod
    def get_file_name(base_path: Path, block_hash: int) -> Path:
        """
        Return the file path for a block. The path is built using hash-based subdirectories:
        <base>/<hhh>/<hh>/<hash>.bin, to limit directory fan-out.

        Args:
            base_path: Base KV-cache directory.
            block_hash: Hash identifying the KV-cache block (int or bytes).

        Returns:
            Full file path for the given block.
        """
        if isinstance(block_hash, bytes):  # convert bytes to int
            block_hash = int.from_bytes(block_hash, "little")
        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        full_path = base_path / subfolder1 / subfolder2 / f"{block_hash_hex}.bin"
        return full_path

    def compute_buffer_size_mb(self, tensors: list[torch.Tensor],
                               gpu_blocks_per_file: int, num_blocks_idx: int):
        """
        Estimate staging memory size in MB, applying min/max limits.

        Args:
            tensors: List of KV-cache tensors used to infer per-block memory usage.
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            num_blocks_idx: Index of the num_blocks dimension in the tensor layout.

        Returns:
            Estimated staging buffer size in megabytes.
        """
        ref = tensors[0]
        # extract exactly one KV block (block 0) using index_select and then remove(squeeze) the block dimension
        per_block = ref.index_select(
            num_blocks_idx,
            torch.tensor([0], device=ref.device)).squeeze(num_blocks_idx)

        per_block_elems = per_block.numel()
        block_elems = per_block_elems * gpu_blocks_per_file  # multiply by blocks per file
        total_elems = block_elems * len(
            tensors)  # multiply by number of tensors
        total_bytes = total_elems * ref.element_size()
        mb = math.ceil(total_bytes / (1024 * 1024))
        return mb

    def get_kv_cache_layout(
            self, gpu_caches: dict[str, torch.Tensor]) -> KVCacheLayout:
        """
        Determine KV cache layout parameters according to the attention backend

        Args:
            gpu_caches: Mapping from layer name to the corresponding KV-cache tensor
                allocated on GPU.

        Returns:
            KVCacheLayout object containing per-layer layout metadata.
        """
        # These lists collect per-layer metadata about the KV-cache layout.
        list_num_blocks_idx = []
        list_kv_before_num_blocks = []
        list_layers_before_num_blocks = []

        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = self.attn_backends[layer_name]

            # Generate a reference KV-cache shape using known parameters.
            # We compare gpu_shape with this synthetic shape to infer the layout.
            test_shape = attn_backend.get_kv_cache_shape(num_blocks=1234,
                                                         block_size=16,
                                                         num_kv_heads=8,
                                                         head_size=256)

            # Case 1: Cross-layer tensor - an extra layer dimension exists on each tensor.
            # In this case, num_blocks is the leading dimension.
            if len(gpu_shape) != len(test_shape):
                assert len(gpu_shape) == len(test_shape) + 1
                num_blocks_idx = 0
                kv_before_num_blocks = False
                layers_before_num_blocks = False

            # Case 2: Standard layout - each element represents a single layer with
            # tensor shaped as (num_blocks, ...). The first dimension matches num_blocks.
            elif test_shape[0] == 1234:
                num_blocks_idx = 0
                kv_before_num_blocks = False
                layers_before_num_blocks = True

            # Case 3: (2, num_blocks, ...) - standard layout but with KV first:
            # (2, num_blocks, heads, block_size, head_size).
            else:
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2

                num_blocks_idx = 1
                kv_before_num_blocks = True
                layers_before_num_blocks = True

            # Store layout metadata for this layer.
            list_num_blocks_idx.append(num_blocks_idx)
            list_kv_before_num_blocks.append(kv_before_num_blocks)
            list_layers_before_num_blocks.append(layers_before_num_blocks)

        return KVCacheLayout(
            num_blocks_idx=list_num_blocks_idx,
            kv_before_num_blocks=list_kv_before_num_blocks,
            layers_before_num_blocks=list_layers_before_num_blocks,
        )

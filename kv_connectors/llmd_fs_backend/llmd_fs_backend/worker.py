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
from collections.abc import Sequence
from dataclasses import dataclass

import storage_offload
import torch
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec

logger = init_logger(__name__)

DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_THREADS_PER_GPU = 64
DEFAULT_READ_PREFERRING_WORKERS_RATIO = 0.75


@dataclass
class GroupOffloadResources:
    file_mapper: FileMapper
    engine: storage_offload.StorageOffloadEngine


class GroupedStorageOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        gpu_blocks_per_file: Sequence[int],
        group_resources: Sequence[GroupOffloadResources],
        direction: str,
    ):
        assert direction in {"store", "load"}
        self.gpu_blocks_per_file = tuple(gpu_blocks_per_file)
        self.group_resources = tuple(group_resources)
        self.direction = direction
        self._next_internal_job_id = 0
        self._external_to_internal: dict[int, list[tuple[int, int]]] = {}
        self._internal_to_external: dict[int, int] = {}
        self._pending_internal_jobs: dict[int, int] = {}
        self._external_success: dict[int, bool] = {}

    def _generate_internal_job_id(self) -> int:
        job_id = self._next_internal_job_id
        self._next_internal_job_id += 1
        return job_id

    def _split_group_block_ids(
        self,
        gpu_spec: GPULoadStoreSpec,
    ) -> list[list[int]]:
        flat_block_ids = gpu_spec.block_ids.tolist()
        group_block_ids: list[list[int]] = []
        start = 0
        for group_size in gpu_spec.group_sizes:
            end = start + group_size
            group_block_ids.append(flat_block_ids[start:end])
            start = end
        assert start == len(flat_block_ids)
        return group_block_ids

    def _build_file_block_mapping(
        self,
        file_mapper: FileMapper,
        block_hashes,
        block_ids: list[int],
        gpu_blocks_per_file: int,
    ) -> tuple[list[str], list[list[int]]]:
        files = []
        per_file_block_ids = []

        first_size = len(block_ids) % gpu_blocks_per_file or gpu_blocks_per_file
        start = 0
        size = first_size

        for block_hash in block_hashes:
            end = min(start + size, len(block_ids))
            files.append(file_mapper.get_file_name(block_hash))
            per_file_block_ids.append(block_ids[start:end])
            start += size
            size = gpu_blocks_per_file

        return files, per_file_block_ids

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        if self.direction == "store":
            src_spec, dst_spec = spec
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, SharedStorageLoadStoreSpec)
            group_block_ids = self._split_group_block_ids(src_spec)
        else:
            src_spec, dst_spec = spec
            assert isinstance(src_spec, SharedStorageLoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            group_block_ids = self._split_group_block_ids(dst_spec)

        assert len(group_block_ids) == len(self.group_resources), (
            "Number of KV block groups must match number of llm-d offload groups."
        )

        internal_jobs: list[tuple[int, int]] = []
        success = True
        for group_index, (resources, block_ids) in enumerate(
            zip(self.group_resources, group_block_ids)
        ):
            group_gpu_blocks_per_file = self.gpu_blocks_per_file[group_index]
            if self.direction == "store":
                files, per_file_block_ids = self._build_file_block_mapping(
                    resources.file_mapper,
                    dst_spec.block_hashes,
                    block_ids,
                    group_gpu_blocks_per_file,
                )
                submit_ok = resources.engine.async_store_gpu_blocks(
                    self._generate_internal_job_id(), files, per_file_block_ids
                )
            else:
                files, per_file_block_ids = self._build_file_block_mapping(
                    resources.file_mapper,
                    src_spec.block_hashes,
                    block_ids,
                    group_gpu_blocks_per_file,
                )
                submit_ok = resources.engine.async_load_gpu_blocks(
                    self._generate_internal_job_id(), files, per_file_block_ids
                )

            internal_job_id = self._next_internal_job_id - 1
            internal_jobs.append((group_index, internal_job_id))
            success = success and submit_ok

        if not success:
            return False

        self._external_to_internal[job_id] = internal_jobs
        self._pending_internal_jobs[job_id] = len(internal_jobs)
        self._external_success[job_id] = True
        for _, internal_job_id in internal_jobs:
            self._internal_to_external[internal_job_id] = job_id
        return True

    def get_finished(self) -> list[TransferResult]:
        finished_results: list[TransferResult] = []
        for _, resources in enumerate(self.group_resources):
            for internal_job_id, success in resources.engine.get_finished():
                external_job_id = self._internal_to_external.pop(internal_job_id, None)
                if external_job_id is None:
                    continue
                self._external_success[external_job_id] = (
                    self._external_success[external_job_id] and success
                )
                self._pending_internal_jobs[external_job_id] -= 1
                if self._pending_internal_jobs[external_job_id] == 0:
                    finished_results.append(
                        TransferResult(
                            job_id=external_job_id,
                            success=self._external_success.pop(external_job_id),
                        )
                    )
                    del self._pending_internal_jobs[external_job_id]
                    del self._external_to_internal[external_job_id]
        return finished_results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            internal_jobs = self._external_to_internal.get(job_id, [])
            for group_index, internal_job_id in internal_jobs:
                self.group_resources[group_index].engine.wait_job(internal_job_id)


class StorageOffloadingHandlers:
    """Base handler with common helpers for Storage offloading."""

    def __init__(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        file_mapper: FileMapper,
        gpu_block_size: int | Sequence[int],
        gpu_blocks_per_file: int | Sequence[int],
        threads_per_gpu: int,
        max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,
        read_preferring_ratio: float = DEFAULT_READ_PREFERRING_WORKERS_RATIO,
        file_mappers: Sequence[FileMapper] | None = None,
        group_layer_names: Sequence[Sequence[str]] | None = None,
    ):
        threads_per_gpu = min(threads_per_gpu, int(os.cpu_count()))

        if file_mappers is None:
            file_mappers = ()
        if group_layer_names is None:
            group_layer_names = ()

        if not file_mappers:
            file_mappers = (file_mapper,)
        if not group_layer_names:
            group_layer_names = (tuple(kv_caches.keys()),)
        if isinstance(gpu_block_size, int):
            gpu_block_sizes = tuple(gpu_block_size for _ in group_layer_names)
        else:
            gpu_block_sizes = tuple(gpu_block_size)
        if isinstance(gpu_blocks_per_file, int):
            gpu_blocks_per_files = tuple(gpu_blocks_per_file for _ in group_layer_names)
        else:
            gpu_blocks_per_files = tuple(gpu_blocks_per_file)

        active_groups: list[tuple[FileMapper, dict[str, torch.Tensor], dict[str, type[AttentionBackend]]]] = []
        for group_index, layer_names in enumerate(group_layer_names):
            group_kv_caches = {
                layer_name: kv_caches[layer_name]
                for layer_name in layer_names
                if layer_name in kv_caches
            }
            if not group_kv_caches:
                continue
            group_attn_backends = {
                layer_name: attn_backends[layer_name] for layer_name in group_kv_caches
            }
            mapper = file_mappers[group_index] if group_index < len(file_mappers) else file_mappers[0]
            active_groups.append((mapper, group_kv_caches, group_attn_backends))

        assert active_groups, "At least one KV cache group is required for llm-d offload"

        threads_per_group = max(1, threads_per_gpu // len(active_groups))
        group_budget_mb = max_staging_memory_gb * 1024 / len(active_groups)

        group_resources: list[GroupOffloadResources] = []
        group_gpu_blocks_per_file_values: list[int] = []
        for group_index, (mapper, group_kv_caches, group_attn_backends) in enumerate(active_groups):
            group_gpu_block_size = gpu_block_sizes[group_index]
            group_gpu_blocks_per_file = gpu_blocks_per_files[group_index]
            tensors, kernel_block_size = StorageOffloadingHandlers._get_tensors(
                group_kv_caches, group_attn_backends, group_gpu_block_size
            )
            assert tensors
            print(
                f"LLMD group={group_index} tensor layouts=" + str([
                    {
                        "shape": tuple(int(dim) for dim in tensor.shape),
                        "stride": tuple(int(dim) for dim in tensor.stride()),
                        "contiguous": bool(tensor.is_contiguous()),
                        "dtype": str(tensor.dtype),
                    }
                    for tensor in tensors
                ]),
                flush=True,
            )
            assert group_gpu_block_size % kernel_block_size == 0

            kernel_blocks_per_gpu_block = group_gpu_block_size // kernel_block_size
            buffer_size_mb = self._compute_buffer_size_mb(
                tensors, group_gpu_blocks_per_file, kernel_blocks_per_gpu_block
            )
            group_threads = threads_per_group
            if buffer_size_mb * group_threads > group_budget_mb:
                group_threads = max(1, int(group_budget_mb / buffer_size_mb))
                logger.warning(
                    "Adjusted llm-d group %s threads_per_gpu to %s due to staging memory budget %s MB",
                    group_index,
                    group_threads,
                    int(group_budget_mb),
                )

            read_preferring_workers = max(
                1, int(group_threads * read_preferring_ratio)
            )
            engine = storage_offload.StorageOffloadEngine(
                io_threads=group_threads,
                gpu_blocks_per_file=group_gpu_blocks_per_file,
                tensors=tensors,
                read_preferring_workers=read_preferring_workers,
            )
            logger.info(
                "StorageOffloadingHandlers group=%s threads=%s block_size=%s staging_buffer_size_mb=%s read_preferring_workers=%s",
                group_index,
                group_threads,
                group_gpu_blocks_per_file * group_gpu_block_size,
                buffer_size_mb,
                read_preferring_workers,
            )
            group_gpu_blocks_per_file_values.append(group_gpu_blocks_per_file)
            group_resources.append(GroupOffloadResources(file_mapper=mapper, engine=engine))

        self.gpu_to_storage_handler = GroupedStorageOffloadingHandler(
            gpu_blocks_per_file=group_gpu_blocks_per_file_values,
            group_resources=group_resources,
            direction="store",
        )
        self.storage_to_gpu_handler = GroupedStorageOffloadingHandler(
            gpu_blocks_per_file=group_gpu_blocks_per_file_values,
            group_resources=group_resources,
            direction="load",
        )

    def _compute_buffer_size_mb(
        self,
        tensors: list[torch.Tensor],
        gpu_blocks_per_file: int,
        kernel_blocks_per_gpu_block: int,
    ):
        kernel_block_size_in_bytes = 0
        for tensor in tensors:
            kernel_block_size_in_bytes += tensor.stride(0) * tensor.element_size()
        kernel_blocks_per_file = kernel_blocks_per_gpu_block * gpu_blocks_per_file
        file_size_in_bytes = kernel_block_size_in_bytes * kernel_blocks_per_file
        return math.ceil(file_size_in_bytes / (1 << 20))

    @staticmethod
    def _get_tensors(
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        fallback_block_size: int,
    ) -> tuple[list[torch.Tensor], int]:
        tensors: list[torch.Tensor] = []
        kernel_block_size: int | None = None

        for layer_name, gpu_tensor in kv_caches.items():
            attn_backend = attn_backends[layer_name]
            gpu_tensor_items = (
                list(gpu_tensor)
                if isinstance(gpu_tensor, (list, tuple))
                else [gpu_tensor]
            )

            for tensor_item in gpu_tensor_items:
                gpu_shape = tensor_item.shape
                split_k_and_v = False
                has_layers_dim = False
                block_size_value = fallback_block_size
                block_dim = 0

                try:
                    test_shape = attn_backend.get_kv_cache_shape(
                        num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
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
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                            include_num_layers_dimension=has_layers_dim
                        )
                        assert len(kv_cache_stride_order) == len(gpu_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(gpu_shape)))

                    test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)
                    block_size_idx = test_shape.index(16)
                    block_size_value = gpu_shape[block_size_idx]
                except NotImplementedError:
                    # Some hybrid backends do not expose a static KV shape API.
                    # Fall back to the vLLM group block size, which implies that
                    # the backend is not using virtual block splitting.
                    block_dim = 0

                if split_k_and_v and tensor_item.is_contiguous():
                    for sub_tensor in tensor_item:
                        tensors.append(sub_tensor)
                else:
                    normalized_tensor = (
                        torch.movedim(tensor_item, block_dim, 0)
                        if block_dim != 0
                        else tensor_item
                    )
                    tensors.append(normalized_tensor)

                if kernel_block_size is not None:
                    assert kernel_block_size == block_size_value
                else:
                    kernel_block_size = block_size_value

        assert kernel_block_size is not None
        return tensors, kernel_block_size

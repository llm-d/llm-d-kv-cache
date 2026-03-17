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

"""Staged backend mixin - CPU pinned-buffer staging for OBJ and POSIX backends."""

import queue
from abc import ABC
from typing import List

import torch

from llmd_fs_backend.nixl_offload import StorageOffloadEngine


class _StagedBackend(StorageOffloadEngine, ABC):
    """
    Mixin for backends that stage data through pinned CPU buffers.
    GPU blocks are D2H-copied to pre-registered NIXL buffers before
    the NIXL transfer, and H2D-copied back on READ completion.
    """

    def __init__(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: List[torch.Tensor],
        backend: str,
    ):
        super().__init__(io_threads, gpu_blocks_per_file, tensors, backend)
        self._d2h_stream = torch.cuda.Stream()  # GPU --> CPU for WRITE staging
        self._h2d_stream = torch.cuda.Stream()  # CPU --> GPU for READ completion
        self._staging_pool: queue.Queue = queue.Queue()
        # tensors[0] shape: (num_blocks, *block_dims)  e.g. (256, 80, 2, 16, 128)
        # buf shape:        (1,          *block_dims)  - slice keeps the leading dim
        #                                                for NIXL memory registration.
        # buf[0] shape:     (*block_dims)              - used for GPU copies to match
        #                                                tensors[0][block_id] shape.
        block_shape = tensors[0][0:1]
        for _ in range(io_threads * 8*4):  # over-provision to avoid pool exhaustion
            buf = torch.empty_like(block_shape, device='cpu').pin_memory()
            reg = self.agent.register_memory([buf])
            self._staging_pool.put((buf, reg))

    def _acquire_staging_slot(self):
        try:
            return self._staging_pool.get_nowait()
        except queue.Empty:
            self._drain_until_slot_available()
            return self._staging_pool.get_nowait()

    def _get_blocks_data(self, tensors: List[torch.Tensor], _block_ids: List) -> list:
        # tensors is one staging buffer per block (flattened); build one NIXL
        # descriptor per buffer
        blocks_data = []
        for tensor in tensors:
            assert tensor.is_cpu
            blocks_data.append((tensor.data_ptr(), self._block_size, 0))
        return blocks_data

    def _prepare_store(self, block_ids: List) -> tuple:
        # block_ids is a list of lists; acquire one staging slot per block
        stagings, tensors = [], []
        with torch.cuda.stream(self._d2h_stream):
            for block_list in block_ids:
                for block_id in block_list:
                    staging = self._acquire_staging_slot()
                    buf, _ = staging
                    buf[0].copy_(self.tensors[0][block_id], non_blocking=True)
                    stagings.append(staging)
                    tensors.append(buf)
        return tensors, stagings

    def _prepare_load(self, block_ids: List) -> tuple:
        # block_ids is a list of lists; acquire one staging slot per block
        stagings, tensors = [], []
        for block_list in block_ids:
            for _ in block_list:
                staging = self._acquire_staging_slot()
                stagings.append(staging)
                tensors.append(staging[0])
        return tensors, stagings

    def _sync_before_transfer(self) -> None:
        self._d2h_stream.synchronize()

    def _complete_read(self, stagings: list, block_ids: List) -> None:
        with torch.cuda.stream(self._h2d_stream):
            for block_list, (buf, _) in zip(block_ids, stagings):
                for block_id in block_list:
                    self.tensors[0][block_id].copy_(buf[0], non_blocking=True)
        self._h2d_stream.synchronize()

    def _shutdown_backend(self) -> None:
        while not self._staging_pool.empty():
            _, reg = self._staging_pool.get_nowait()
            self.agent.deregister_memory(reg)

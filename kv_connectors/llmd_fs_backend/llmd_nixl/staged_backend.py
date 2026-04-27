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

from llmd_nixl.nixl_offload import StorageOffloadEngine


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
        total_block_bytes = len(tensors) * self._block_size
        for _ in range(io_threads * 8):  # over-provision to avoid pool exhaustion
            buf = torch.empty(total_block_bytes, dtype=torch.uint8, device='cpu').pin_memory()
            reg = self.agent.register_memory([buf])
            self._staging_pool.put((buf, reg))

    def _get_blocks_data(self, tensors: List[torch.Tensor], _block_ids: List) -> list:
        # tensors is one staging buffer per block (flattened); build one NIXL
        # descriptor per buffer
        blocks_data = []
        for tensor in tensors:
            assert tensor.is_cpu
            blocks_data.append((tensor.data_ptr(), len(self.tensors) * self._block_size, 0))
        return blocks_data

    def _get_staging_and_copy(self, block_ids: List) -> tuple:
        # block_ids is a list of lists; acquire one staging slot per block
        num_blocks = sum(len(bl) for bl in block_ids)
        assert self._staging_pool.qsize() >= num_blocks, (
            f"Staging pool exhausted: need {num_blocks} slots, "
            f"have {self._staging_pool.qsize()} (pool size={self.io_threads * 8})"
        )
        stagings, tensors = [], []
        with torch.cuda.stream(self._d2h_stream):
            for block_list in block_ids:
                for block_id in block_list:
                    staging = self._staging_pool.get_nowait()
                    buf, _ = staging
                    offset = 0
                    for tensor in self.tensors:
                        buf[offset:offset + self._block_size].copy_(
                            tensor[block_id].view(torch.uint8).flatten(), non_blocking=True)
                        offset += self._block_size
                    stagings.append(staging)
                    tensors.append(buf)
        return tensors, stagings

    def _get_staging(self, block_ids: List) -> tuple:
        # block_ids is a list of lists; acquire one staging slot per block
        num_blocks = sum(len(bl) for bl in block_ids)
        assert self._staging_pool.qsize() >= num_blocks, (
            f"Staging pool exhausted: need {num_blocks} slots, "
            f"have {self._staging_pool.qsize()} (pool size={self.io_threads * 8})"
        )
        stagings, tensors = [], []
        for block_list in block_ids:
            for _ in block_list:
                staging = self._staging_pool.get_nowait()
                stagings.append(staging)
                tensors.append(staging[0])
        return tensors, stagings

    def _sync_before_transfer(self) -> None:
        self._d2h_stream.synchronize()

    def _complete_read(self, stagings: list, block_ids: List) -> None:
        with torch.cuda.stream(self._h2d_stream):
            # block_ids is nested (one list per file), but stagings is flat (one
            # entry per block), matching the order produced by _get_staging.
            # Flatten block_ids so each staging slot pairs with its block_id.
            # For obj backend gpu_blocks_per_file == 1, but this is not
            # guaranteed for future backends (like nixl posix).
            flat_block_ids = [b for block_list in block_ids for b in block_list]
            for (buf, _), block_id in zip(stagings, flat_block_ids):
                offset = 0
                for tensor in self.tensors:
                    tensor[block_id].view(torch.uint8).flatten().copy_(
                        buf[offset:offset + self._block_size], non_blocking=True)
                    offset += self._block_size
        self._h2d_stream.synchronize()

    def _complete_transfer(self, entry) -> None:
        """Copy READ data to GPU, release NIXL resources, return staging slots."""
        if entry.read_block_ids is not None:
            self._complete_read(entry.stagings, entry.read_block_ids)
        super()._complete_transfer(entry)
        for s in entry.stagings:
            self._staging_pool.put(s)

    def _on_submit_error(self, stagings) -> None:
        """Return staging slots on submit failure."""
        if stagings is not None:
            for s in stagings:
                self._staging_pool.put(s)

    def _shutdown_backend(self) -> None:
        try:
            while not self._staging_pool.empty():
                _, reg = self._staging_pool.get_nowait()
                self.agent.deregister_memory(reg)
        except AttributeError:
            pass  # _staging_pool not initialized (failed __init__)

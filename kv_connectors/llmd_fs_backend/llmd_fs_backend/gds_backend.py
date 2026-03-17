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

""" GDS and GDS_MT (GPUDirect Storage) backends - VRAM to file transfers """

import os
from typing import List

import torch

from llmd_fs_backend.nixl_offload import StorageOffloadEngine


class GdsBackend(StorageOffloadEngine):
    nixl_source = "VRAM"
    nixl_dest = "FILE"

    _open_flags = os.O_RDWR | os.O_CREAT | os.O_DIRECT  # O_DIRECT required by cuFile API

    def __init__(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: List[torch.Tensor],
        backend: str = "GDS",
        **_,
    ):
        super().__init__(io_threads, gpu_blocks_per_file, tensors, backend)
        self._nixl_reg_dlist = self.agent.register_memory(tensors)

    def _backend_params(self) -> dict:
        return {}

    def _prepare_store(self, *_) -> tuple:
        return self.tensors, None

    def _prepare_load(self, *_) -> tuple:
        return self.tensors, None

    def _get_blocks_data(self, tensors: List[torch.Tensor], block_ids: List) -> list:
        assert len(tensors) == 1 and tensors[0].is_cuda
        base_addr = tensors[0].data_ptr()
        device_id = tensors[0].device.index
        return [
            (base_addr + block * self._block_size, self._block_size, device_id)
            for block_list in block_ids
            for block in block_list
        ]

    def _open_files(self, files: List[str]) -> list:
        fds = []
        for f in files:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            fds.append(os.open(f, self._open_flags))
        return fds

    def _build_nixl_file_entry(self, fd_list, file_idx, intra_offset) -> tuple:
        return (self._block_size * intra_offset, self._block_size, fd_list[file_idx], "not_obj")

    def _close_fds(self, fd_list: list) -> None:
        for fd in fd_list:
            os.close(fd)

    def _complete_read(self, *_) -> None:
        pass  # NIXL writes directly to VRAM - no staging copy needed

    def _shutdown_backend(self) -> None:
        self.agent.deregister_memory(self._nixl_reg_dlist)


class GdsMtBackend(GdsBackend):
    """Multi-threaded GDS - identical to GdsBackend with a different NIXL backend name."""

    def __init__(self, io_threads: int, gpu_blocks_per_file: int, tensors: List[torch.Tensor], **_):
        super().__init__(io_threads, gpu_blocks_per_file, tensors, backend="GDS_MT")

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

"""POSIX (regular file) storage backend."""

import os
from typing import List

import torch

from llmd_fs_backend.staged_backend import _StagedBackend


class PosixBackend(_StagedBackend):
    nixl_source = "DRAM"
    nixl_dest = "FILE"

    def __init__(self, io_threads: int, gpu_blocks_per_file: int, tensors: List[torch.Tensor], **_):
        super().__init__(io_threads, gpu_blocks_per_file, tensors, "POSIX")
        self.logger.info("in POSIX __init__")

    def _backend_params(self) -> dict:
        return {}

    def _open_files(self, files: List[str]) -> list:
        fds = []
        for f in files:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            fds.append(os.open(f, os.O_RDWR | os.O_CREAT))
        return fds

    def _build_nixl_file_entry(self, fd_list, file_idx, intra_offset) -> tuple:
        return (self._block_size * intra_offset, self._block_size, fd_list[file_idx], "posix")

    def _close_fds(self, fd_list: list) -> None:
        for fd in fd_list:
            os.close(fd)

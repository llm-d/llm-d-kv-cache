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

from llmd_fs_backend.worker import StorageEngine, StorageOffloadingHandlers
from llmd_nixl.obj_backend import ObjBackend


class NixlStorageOffloadingHandlers(StorageOffloadingHandlers):
    """StorageOffloadingHandlers backed by the NIXL OBJ engine."""

    def _create_engine(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: list,
        read_preferring_workers: int,
        max_write_queued_seconds: float,
        extra_config: dict,
        gds_mode: str,
    ) -> StorageEngine:
        return ObjBackend(
            io_threads=io_threads,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
            extra_config=extra_config,
        )

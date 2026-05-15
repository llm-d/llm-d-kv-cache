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

"""OBJ (S3) storage backend."""

import hashlib

import torch

from llmd_nixl.staged_backend import _StagedBackend


def obj_key_to_dev_id(obj_key: str) -> int:
    return int(hashlib.md5(obj_key.encode()).hexdigest(), 16) % (2**31)


class ObjBackend(_StagedBackend):
    nixl_source = "DRAM"
    nixl_dest = "OBJ"

    def __init__(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: list[torch.Tensor],
        extra_config: dict | None = None,
    ):
        assert gpu_blocks_per_file == 1, (
            "OBJ backend: multiple blocks per object not yet supported"
        )

        cfg = extra_config or {}
        required = ["bucket", "endpoint_override", "access_key", "secret_key"]
        missing = [k for k in required if not cfg.get(k)]
        if missing:
            raise ValueError(f"OBJ backend requires: {', '.join(missing)}")

        # Store before super().__init__() so _backend_params() can use them
        self._bucket = cfg["bucket"]
        self._endpoint_override = cfg["endpoint_override"]
        self._scheme = cfg.get("scheme", "http")
        self._access_key = cfg["access_key"]
        self._secret_key = cfg["secret_key"]
        self._ca_bundle = cfg.get("ca_bundle", "")
        self._io_threads = io_threads
        super().__init__(io_threads, gpu_blocks_per_file, tensors, "OBJ")

    def _backend_params(self) -> dict:
        params = {
            "bucket": self._bucket,
            "endpoint_override": self._endpoint_override,
            "scheme": self._scheme,
            "access_key": self._access_key,
            "secret_key": self._secret_key,
            "num_threads": str(self._io_threads),
        }
        if self._ca_bundle:
            params["ca_bundle"] = self._ca_bundle
        return params

    def _open_files(self, files: list[str]) -> list:
        return list(files)  # S3 keys - no real FDs

    def _build_nixl_file_entry(self, fd_list, file_idx, _intra_offset) -> tuple:
        # OBJ: block size == GPU block size, so intra_offset is always 0
        key = fd_list[file_idx]
        return (0, len(self.tensors) * self._block_size, obj_key_to_dev_id(key), key)

    def _close_fds(self, *_) -> None:
        pass  # S3 keys - nothing to close

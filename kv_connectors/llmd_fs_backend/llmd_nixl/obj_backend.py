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

    def _staging_bytes_per_slot(self) -> int:
        return self.gpu_blocks_per_file * len(self.tensors) * self._block_size

    def _get_staging_and_copy(self, block_ids: list) -> tuple:
        num_files = len(block_ids)
        shortfall = num_files - self._staging_pool.qsize()
        if shortfall > 0:
            self._extend_staging_pool(shortfall)
        stagings, tensors_out = [], []
        with torch.cuda.stream(self._d2h_stream):
            for block_list in block_ids:
                staging = self._staging_pool.get_nowait()
                buf, _ = staging
                offset = 0
                for block_id in block_list:
                    for tensor in self.tensors:
                        buf[offset : offset + self._block_size].copy_(
                            tensor[block_id].view(torch.uint8).flatten(),
                            non_blocking=True,
                        )
                        offset += self._block_size
                stagings.append(staging)
                tensors_out.append(buf)
        return tensors_out, stagings

    def _get_staging(self, block_ids: list) -> tuple:
        num_files = len(block_ids)
        shortfall = num_files - self._staging_pool.qsize()
        if shortfall > 0:
            self._extend_staging_pool(shortfall)
        stagings, tensors_out = [], []
        for _ in block_ids:
            staging = self._staging_pool.get_nowait()
            stagings.append(staging)
            tensors_out.append(staging[0])
        return tensors_out, stagings

    def _get_blocks_data(self, tensors: list[torch.Tensor], block_ids: list) -> list:
        # One DRAM descriptor per file; size matches the actual transfer bytes
        # so that partial first-file reads use a correctly-sized DRAM region.
        return [
            (t.data_ptr(), len(bl) * len(self.tensors) * self._block_size, 0)
            for t, bl in zip(tensors, block_ids)
        ]

    def _complete_read(self, stagings: list, block_ids: list) -> None:
        with torch.cuda.stream(self._h2d_stream):
            for (buf, _), block_list in zip(stagings, block_ids):
                offset = 0
                for block_id in block_list:
                    for tensor in self.tensors:
                        tensor[block_id].view(torch.uint8).flatten().copy_(
                            buf[offset : offset + self._block_size],
                            non_blocking=True,
                        )
                        offset += self._block_size
        self._h2d_stream.synchronize()

    def _open_files(self, files: list[str]) -> list:
        return list(files)  # S3 keys - no real FDs

    def _build_nixl_file_entries(self, fd_list, file_idx, block_list) -> list[tuple]:
        key = fd_list[file_idx]
        file_bytes = len(block_list) * len(self.tensors) * self._block_size
        # For a partial first file the relevant blocks sit at the tail of the
        # object; use a non-zero offset so NIXL issues a range read/write.
        skip = self.gpu_blocks_per_file - len(block_list)
        file_offset = skip * len(self.tensors) * self._block_size
        return [(file_offset, file_bytes, obj_key_to_dev_id(key), key)]

    def _close_fds(self, *_) -> None:
        pass  # S3 keys - nothing to close

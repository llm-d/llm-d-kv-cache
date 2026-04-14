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

"""Factory for instantiating storage offload engine backends.

Backend names and their meanings:
  "POSIX"           - POSIX engine, GDS disabled
  "POSIX_GDS_READ"  - GDS for reads only
  "POSIX_GDS_WRITE" - GDS for writes only
  "POSIX_GDS"       - GDS for reads and writes
  "POSIX_BB_READ"   - bounce-buffer GDS for reads only
  "POSIX_BB_WRITE"  - bounce-buffer GDS for writes only
  "POSIX_BB"        - bounce-buffer GDS for reads and writes
  "OBJ"             - NIXL S3 object store backend
"""

import storage_offload
from llmd_nixl.obj_backend import ObjBackend

# Maps backend name -> gds_mode string for C++ POSIX engine variants.
_POSIX_GDS_MODES = {
    "POSIX":           "disabled",
    "GDS_READ":  "read_only",
    "GDS_WRITE": "write_only",
    "GDS":       "read_write",
    "GDS_BB_READ":   "bb_read_only",
    "GDS_BB_WRITE":  "bb_write_only",
    "GDS_BB":        "bb_read_write",
}

# C++ POSIX backends that do not use a CPU staging buffer (full GDS path).
_POSIX_NO_STAGING = {"GDS", "GDS_BB"}


def posix_uses_no_staging(backend: str) -> bool:
    """Return True if the backend uses full GDS (no CPU staging buffer)."""
    return backend in _POSIX_NO_STAGING


def make_storage_engine(
    backend: str,
    io_threads: int,
    gpu_blocks_per_file: int,
    tensors: list,
    read_preferring_workers: int = 1,
    extra_config: dict | None = None,
):
    """
    Instantiate the correct storage engine for the given backend name.
    Args:
        backend: See module docstring for valid values.
        io_threads: Number of I/O threads.
        gpu_blocks_per_file: Number of GPU blocks per file/object.
        tensors: KV-cache tensors.
        read_preferring_workers: Number of read-preferring workers (POSIX only).
        extra_config: Backend-specific configuration (parsed by the backend).
    """
    if backend in _POSIX_GDS_MODES:
        return storage_offload.StorageOffloadEngine(
            io_threads,
            gpu_blocks_per_file,
            tensors,
            read_preferring_workers,
            _POSIX_GDS_MODES[backend],
        )

    if backend == "OBJ":
        return ObjBackend(
            io_threads=io_threads,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
            extra_config=extra_config or {},
        )

    raise ValueError(
        f"Unknown backend {backend!r}. "
        f"Valid options: {list(_POSIX_GDS_MODES) + ['OBJ']}"
    )

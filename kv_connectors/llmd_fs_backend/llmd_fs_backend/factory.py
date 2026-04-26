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

Valid backends:
  "POSIX"  - C++ POSIX engine; GDS behaviour controlled by gds_mode parameter
  "OBJ"    - NIXL S3 object store backend

Valid gds_mode values for the POSIX backend (default: "disabled"):
  "disabled"      - no GDS
  "read_only"     - GDS for reads only
  "write_only"    - GDS for writes only
  "read_write"    - GDS for reads and writes (no CPU staging buffer)
  "bb_read_only"  - bounce-buffer GDS for reads only
  "bb_write_only" - bounce-buffer GDS for writes only
  "bb_read_write" - bounce-buffer GDS for reads and writes (no CPU staging buffer)
"""

import storage_offload
from llmd_nixl.obj_backend import ObjBackend

# GDS modes that bypass the CPU staging buffer entirely.
_GDS_NO_STAGING = {"read_write", "bb_read_write"}

_VALID_GDS_MODES = {
    "disabled",
    "read_only",
    "write_only",
    "read_write",
    "bb_read_only",
    "bb_write_only",
    "bb_read_write",
}


def posix_uses_no_staging(gds_mode: str) -> bool:
    """Return True if the gds_mode uses full GDS (no CPU staging buffer)."""
    return gds_mode in _GDS_NO_STAGING


def make_storage_engine(
    backend: str,
    io_threads: int,
    gpu_blocks_per_file: int,
    tensors: list,
    read_preferring_workers: int = 1,
    extra_config: dict | None = None,
    gds_mode: str = "disabled",
):
    """
    Instantiate the correct storage engine for the given backend name.
    Args:
        backend: "POSIX" or "OBJ".
        io_threads: Number of I/O threads.
        gpu_blocks_per_file: Number of GPU blocks per file/object.
        tensors: KV-cache tensors.
        read_preferring_workers: Number of read-preferring workers (POSIX only).
        extra_config: Backend-specific configuration (parsed by the backend).
        gds_mode: GDS mode for the POSIX backend (see module docstring).
    """
    if backend == "POSIX":
        if gds_mode not in _VALID_GDS_MODES:
            raise ValueError(
                f"Unknown gds_mode {gds_mode!r}. Valid: {sorted(_VALID_GDS_MODES)}"
            )
        return storage_offload.StorageOffloadEngine(
            io_threads,
            gpu_blocks_per_file,
            tensors,
            read_preferring_workers,
            gds_mode,
        )

    if backend == "OBJ":
        return ObjBackend(
            io_threads=io_threads,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
            extra_config=extra_config or {},
        )

    raise ValueError(f"Unknown backend {backend!r}. Valid: POSIX, OBJ")

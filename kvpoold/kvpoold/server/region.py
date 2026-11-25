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

"""Creating the shared region, with vLLM's SharedOffloadRegion.

The pool server is the region's creator: it creates the segment files
with vLLM's own segment split and pre-faults every page, so the region's memory
is allocated in RAM and charged to the pool server's pod. The files
keep their pages with no process mapping them, so the server keeps neither a
mapping nor a file descriptor of them. It never calls ``cleanup()``, which would
unlink the files; the files outlive the process, and a new instance reuses them.
"""

import contextlib
import os
import uuid

import msgspec
from vllm.v1.kv_offload.cpu.shared_offload_region import (
    REGION_SEGMENTS,
    SharedOffloadRegion,
)

from kvpoold.api.v1 import ShmFileRegion

REGION_FILE_NAME = "kvpoold.mmap"


class MemorySpec(msgspec.Struct, frozen=True, kw_only=True):
    """The pool's layout, fixed by the first accepted attach and kept beside the
    region's files."""

    region: ShmFileRegion
    hash_function: str
    hash_seed: str | None

    @property
    def kv_bytes_per_chunk(self) -> int:
        return self.region.kv_bytes_per_chunk


# Engines in other pods open the files through a group they share with the
# pool server's pod.
_FILE_MODE = 0o660


def create_region(region_dir: str, pool_bytes: int, kv_bytes_per_chunk: int) -> ShmFileRegion:
    """Create and pre-fault the region's files; return their descriptor.

    SharedOffloadRegion closes its file descriptor only in ``cleanup()``, so
    this closes it; dropping the region object on return unmaps the files.
    """
    num_chunks = pool_bytes // kv_bytes_per_chunk
    if num_chunks < 1:
        raise ValueError(f"pool of {pool_bytes} bytes holds no chunk of {kv_bytes_per_chunk} bytes")
    region = SharedOffloadRegion(
        engine_id="",  # unused with mmap_path
        num_chunks=num_chunks,
        rank=None,  # no rank: pre-fault the whole region
        kv_bytes_per_chunk=kv_bytes_per_chunk,
        cpu_page_size=kv_bytes_per_chunk,
        num_segments=REGION_SEGMENTS,
        mmap_path=os.path.join(region_dir, REGION_FILE_NAME),
    )
    assert region.fd is not None
    os.close(region.fd)
    # The segments are SharedOffloadRegion's own split; engines check theirs
    # against this list.
    segments = [(path, offset, length) for path, offset, length in region._segments]
    for path, _, _ in segments:
        os.chmod(path, _FILE_MODE)
    descriptor = ShmFileRegion(
        region_id=uuid.uuid4().bytes,
        num_chunks=num_chunks,
        kv_bytes_per_chunk=kv_bytes_per_chunk,
        segments=segments,
    )
    return descriptor


def region_files_match(descriptor: ShmFileRegion) -> bool:
    """Whether every segment file exists with its length. The files keep their
    pages whether or not a process maps them, so a new instance only checks them."""
    for path, _, length in descriptor.segments:
        try:
            if os.stat(path).st_size != length:
                return False
        except FileNotFoundError:
            return False
    return True


def remove_region_files(region_dir: str) -> None:
    """Remove the region's segment files from the directory."""
    with os.scandir(region_dir) as entries:
        for entry in entries:
            if entry.name == REGION_FILE_NAME or entry.name.startswith(f"{REGION_FILE_NAME}."):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(entry.path)

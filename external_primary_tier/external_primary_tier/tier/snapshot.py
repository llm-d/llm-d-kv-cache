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

"""The files the External Primary Tier keeps beside the region's files.

- The spec file holds the pool's ``MemorySpec``. It is written when the region is
  created, before any engine is told where the region is, so region files
  without a spec file were never handed to an engine.
- The snapshot file holds the records at a clean stop. A new instance reads it
  once and removes it, so the records of an instance that later crashes are
  never loaded. The snapshot has no version number: it is a map, so a field
  added with a default decodes from a snapshot without it, and an older reader
  skips a field it does not know.

Each file is written to a temporary file and renamed over the previous one, so a
write cut short leaves the previous file or none.
"""

import contextlib
import os

import msgspec
from vllm.distributed.kv_events import BlockStored
from vllm.logger import init_logger

from external_primary_tier.tier.state import MemorySpec

logger = init_logger(__name__)

SPEC_FILE_NAME = "external-primary-tier.spec"
SNAPSHOT_FILE_NAME = "external-primary-tier.snapshot"
_FILE_MODE = 0o600


class ChunkEntry(msgspec.Struct, frozen=True, array_like=True):
    """One resident key of the LRU policy."""

    key: bytes
    """The pool key: the client's compatibility id followed by vLLM's key."""
    chunk_id: int
    ref_cnt: int


class ClientSnapshot(msgspec.Struct, frozen=True):
    client_id: str
    compat_id: bytes
    pins: list[tuple[bytes, int]]
    """(pool key, number of pins)."""
    writes: list[bytes]
    """Pool keys the client fills."""


class Snapshot(msgspec.Struct, frozen=True):
    region_id: bytes
    """The region the records describe."""
    num_allocated_chunks: int
    free_list: list[int]
    entries: list[ChunkEntry]
    """Every resident key, in LRU eviction order."""
    clients: list[ClientSnapshot]
    block_stored: list[tuple[bytes, BlockStored]] = []
    """(pool key, ``BlockStored``) of each readable chunk whose engine sent one. Values
    nested in an ``extra_keys`` entry decode as lists where they were tuples: fine to
    publish, not to hash. A snapshot written by a tier version that did not keep
    ``BlockStored`` decodes with none: that tier never published its chunks, so the
    indexer has no entries for them, and they restore without a ``BlockStored``."""


def write_spec(region_dir: str, spec: MemorySpec) -> None:
    _write(os.path.join(region_dir, SPEC_FILE_NAME), msgspec.msgpack.encode(spec))


def read_spec(region_dir: str) -> MemorySpec | None:
    return _read(os.path.join(region_dir, SPEC_FILE_NAME), MemorySpec)


def write_snapshot(region_dir: str, snapshot: Snapshot) -> None:
    _write(os.path.join(region_dir, SNAPSHOT_FILE_NAME), msgspec.msgpack.encode(snapshot))


def take_snapshot(region_dir: str) -> Snapshot | None:
    """Read the snapshot, if there is one, and remove its file."""
    path = os.path.join(region_dir, SNAPSHOT_FILE_NAME)
    snapshot = _read(path, Snapshot)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    return snapshot


def remove_spec_and_snapshot(region_dir: str) -> None:
    for name in (SPEC_FILE_NAME, SNAPSHOT_FILE_NAME):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(os.path.join(region_dir, name))


def _write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE), "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read[T](path: str, type_: type[T]) -> T | None:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    try:
        return msgspec.msgpack.decode(data, type=type_)
    except (msgspec.DecodeError, msgspec.ValidationError):
        logger.warning("Ignoring %s: it does not decode", path, exc_info=True)
        return None

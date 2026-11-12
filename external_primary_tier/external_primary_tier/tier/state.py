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

"""The External Primary Tier's records of the pool and of each engine."""

from dataclasses import dataclass, field

from external_primary_tier.protocol import AnyRegion, OffloadKey


@dataclass(frozen=True)
class MemorySpec:
    """The pool's layout, fixed by the first accepted attach."""

    region: AnyRegion
    hash_function: str
    hash_seed: str | None
    protocol_version: int

    @property
    def kv_bytes_per_chunk(self) -> int:
        return self.region.kv_bytes_per_chunk


@dataclass
class Session:
    """One engine process's DEALER to the External Primary Tier's ROUTER."""

    routing_id: bytes


@dataclass
class SchedulerSession(Session):
    """The scheduler's session: its requests and notices."""


@dataclass
class WorkerSession(Session):
    """A GPU worker's session: its attach."""

    rank: int


@dataclass
class ClientContext:
    """One engine, from its first accepted attach."""

    client_id: str
    """vLLM's engine id and the pod's hostname."""
    compat_id: bytes
    """8 bytes; prefixes this client's keys."""
    scheduler: SchedulerSession | None = None
    """None until the scheduler attaches."""
    workers: list[WorkerSession] = field(default_factory=list)
    pins: dict[OffloadKey, list[float]] = field(default_factory=dict)
    """One deadline per pin; the client's share of each key's ref_cnt."""
    writes: dict[OffloadKey, float] = field(default_factory=dict)
    """Keys with ref_cnt -1 the client fills, and their deadlines."""

    def worker(self, rank: int) -> WorkerSession | None:
        return next((w for w in self.workers if w.rank == rank), None)

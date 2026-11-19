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

import msgspec
from vllm.v1.kv_offload.base import ReqContext

from external_primary_tier.protocol import OffloadKey, ShmFileRegion


class MemorySpec(msgspec.Struct, frozen=True, kw_only=True):
    """The pool's layout, fixed by the first accepted attach and kept beside the
    region's files."""

    region: ShmFileRegion
    hash_function: str
    hash_seed: str | None

    @property
    def kv_bytes_per_chunk(self) -> int:
        return self.region.kv_bytes_per_chunk


@dataclass
class ClientContext:
    """One engine, from its first accepted attach."""

    client_id: str
    """vLLM's engine id."""
    compat_id: bytes
    """8 bytes; prefixes this client's keys."""
    scheduler: bytes | None = None
    """The scheduler process's ZMQ identity; None until it attaches."""
    pins: dict[bytes, list[float]] = field(default_factory=dict)
    """Pool key to one deadline per pin, earliest first; the client's share of
    the key's ref_cnt."""
    writes: dict[bytes, float] = field(default_factory=dict)
    """Pool keys with ref_cnt -1 the client fills, and their deadlines."""
    requests: dict[str, ReqContext] = field(default_factory=dict)
    """Request id to what vLLM's manager and cache policy keep for the request,
    and the positions of its keys."""

    def pool_key(self, key: OffloadKey) -> bytes:
        """The key the pool keeps vLLM's ``key`` under for this client."""
        return self.compat_id + key

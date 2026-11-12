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

"""Descriptions of the shared region, sent in the attach reply.

Descriptors are data only: the External Primary Tier builds them, and the
engine's spec decides how to map the region each one describes. Each subclass
is tagged with its class name, so the decoder maps the tag on the wire back to
the subclass.
"""

import msgspec


class RegionDescriptor(msgspec.Struct, tag=True, kw_only=True, frozen=True):
    """The fields every region has. Only its subclasses are instantiated."""

    region_id: bytes
    """Identifies the region's contents; a snapshot loads only onto the same region."""

    num_chunks: int

    kv_bytes_per_chunk: int
    """The row stride: bytes of one chunk, across every rank of an engine."""

    def __post_init__(self) -> None:
        if type(self) is RegionDescriptor:
            raise TypeError("RegionDescriptor is abstract; use a subclass")


class ShmFileRegion(RegionDescriptor):
    """The region as segment files in the host's /dev/shm."""

    segments: list[tuple[str, int, int]]
    """One (path, offset, length) per segment file, in region order."""


# Every region kind the protocol carries.
AnyRegion = ShmFileRegion

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

"""Messages between engine processes and the External Primary Tier.

Every message is a tagged msgspec Struct, so the decoder maps the tag on the
wire to its class. Each scheduler and GPU worker has one session, a DEALER to
the External Primary Tier's ROUTER; every message below travels on it. The
DEALER uses ZMQ's heartbeat, so engines see the External Primary Tier go away;
the External Primary Tier gives every pin and write a deadline instead of
tracking whether engines are alive.

- Requests get exactly one reply. A request carries a ``call_id`` that its reply
  repeats, so a reply that arrives after its request timed out is not taken for
  the reply to a later request.
- Notices get no reply.
- Any request can be answered with ``Draining`` or ``UnknownClient`` instead of
  its usual reply.
"""

from typing import Literal

import msgspec

from external_primary_tier.protocol.region import AnyRegion

PROTOCOL_VERSION = 1

# vLLM's OffloadKey: the block hash followed by the KV cache group index.
OffloadKey = bytes


class _Message(msgspec.Struct, tag=True, kw_only=True, frozen=True, array_like=True):
    pass


class _Call(_Message):
    """A request, or the reply to one; the reply repeats the request's ``call_id``."""

    call_id: int


# --- Requests, from engine processes ---------------------------------------


class Attach(_Call):
    """From every engine process: each GPU worker before it maps the region, and
    the scheduler after its workers have attached."""

    version: int
    client_id: str
    """vLLM's engine id and the pod's hostname."""
    rank: int | None
    """The GPU worker's rank; None for the scheduler."""
    kv_bytes_per_chunk: int
    hash_function: str
    hash_seed: str | None
    """The engine's PYTHONHASHSEED, or None when it is not set."""
    compat_id: bytes
    """8 bytes: the first 8 bytes of SHA-256 over FileMapper's run config."""


class Lookup(_Call):
    """One key of one request. A hit is pinned for the request."""

    req_id: str
    key: OffloadKey


class PrepareLoad(_Call):
    """Keys the request got as hits this step and now loads."""

    req_id: str
    keys: list[OffloadKey]


class PrepareStore(_Call):
    req_id: str
    keys: list[OffloadKey]


class GetStats(_Call):
    pass


# --- Notices, from engine processes ----------------------------------------


class CompleteLoad(_Message):
    req_id: str
    keys: list[OffloadKey]


class CompleteStore(_Message):
    req_id: str
    keys: list[OffloadKey]
    success: bool


class RequestFinished(_Message):
    """The request's keys, grouped as vLLM's cache policies expect them."""

    req_id: str
    key_groups: list[list[OffloadKey]]


class ReleaseHits(_Message):
    """At the end of a step: the step's hits that were not loaded."""

    hits: list[tuple[OffloadKey, int]]
    """(key, count): a key hit by several requests is released once per hit."""


class Leave(_Message):
    """The engine is stopping and its copies have finished, so its holds are
    released without waiting for their deadlines."""


# --- Replies, from the External Primary Tier --------------------------------


class AttachOk(_Call):
    """The region, for every process. The scheduler maps nothing; it reads
    ``region.num_chunks``."""

    region: AnyRegion


class Refused(_Call):
    """An attach that does not match the pool, or a scheduler's attach before
    any worker of its engine has attached."""

    reason: str


class LookupOk(_Call):
    result: Literal["miss", "hit", "hit_pending"]


class LoadSpec(_Call):
    """The chunk ids for the keys of a PrepareLoad, in the same order."""

    chunk_ids: list[int]


class StoreSpec(_Call):
    """The keys to store, their chunk ids in the same order, and the keys evicted for them."""

    keys_to_store: list[OffloadKey]
    chunk_ids: list[int]
    evicted_keys: list[OffloadKey]


class Stats(_Call):
    values: dict[str, int]


class Draining(_Call):
    """The External Primary Tier is stopping; the request was not served."""


class UnknownClient(_Call):
    """The sender has not attached to this External Primary Tier instance."""


# --- What each side decodes -------------------------------------------------

Request = Attach | Lookup | PrepareLoad | PrepareStore | GetStats
"""Every message that gets a reply."""

Notice = CompleteLoad | CompleteStore | RequestFinished | ReleaseHits | Leave
"""Every message that gets no reply."""

ToTier = (
    Attach
    | Lookup
    | PrepareLoad
    | PrepareStore
    | GetStats
    | CompleteLoad
    | CompleteStore
    | RequestFinished
    | ReleaseHits
    | Leave
)
"""Every message the External Primary Tier receives."""

FromTier = AttachOk | Refused | LookupOk | LoadSpec | StoreSpec | Stats | Draining | UnknownClient
"""Every message an engine process receives."""

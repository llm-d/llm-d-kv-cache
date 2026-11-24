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
wire to its class. Each engine process sends them on its own DEALER to the
External Primary Tier's ROUTER.

- Requests get exactly one reply. A request carries a ``call_id`` that its reply
  repeats, so a reply that arrives after its request timed out is not taken for
  the reply to a later request.
- Notices get no reply.
- Any request can be answered with ``Draining`` or ``UnknownClient`` instead of
  its usual reply.

The one vLLM type on the wire is ``BlockStored``, which ``CompleteStore`` carries
so that the External Primary Tier can publish KV events; importing this module
imports vLLM.
"""

from typing import Literal

import msgspec
from vllm.distributed.kv_events import BlockStored

from external_primary_tier.protocol.region import ShmFileRegion

PROTOCOL_VERSION = 2

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
    """vLLM's engine id."""
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
    """Keys the request got as hits this step and now loads. Each one's pin from
    its lookup carries the load."""

    req_id: str
    keys: list[OffloadKey]
    positions: list[int | None]
    """Each key's end-token position, from the request's context in vLLM; None
    when it has none. vLLM's cache policies order a request's keys by it."""


class PrepareStore(_Call):
    req_id: str
    keys: list[OffloadKey]
    positions: list[int | None]
    """As in ``PrepareLoad``."""


class GetStats(_Call):
    pass


# --- Notices, from engine processes ----------------------------------------


class CompleteLoad(_Message):
    keys: list[OffloadKey]


class CompleteStore(_Message):
    keys: list[OffloadKey]
    success: bool
    events: list[BlockStored] | None = None
    """One vLLM ``BlockStored`` per key, in the same order, for the External Primary
    Tier to publish; None when the engine has KV events disabled. The External
    Primary Tier overwrites the publisher-level fields (``medium``, ``locality``,
    ``ownership``) and keeps the rest with the chunk, so that it can derive the
    chunk's ``BlockRemoved`` from ``block_hashes`` and ``group_idx``."""


class RequestFinished(_Message):
    """vLLM's ``on_request_finished``: the cache policy applies the request's
    accesses, which the External Primary Tier recorded from its calls."""

    req_id: str


class ReleaseHits(_Message):
    """At the end of a step: the step's hits that were not loaded."""

    hits: list[tuple[OffloadKey, int]]
    """(key, count): a key hit by several requests is released once per hit."""


class Leave(_Message):
    """The engine is stopping and its copies have finished, so its leases are
    released without waiting for their deadlines."""


# --- Replies, from the External Primary Tier --------------------------------


class AttachOk(_Call):
    """The region, for every process. The scheduler maps nothing; it reads
    ``region.num_chunks``."""

    region: ShmFileRegion


class Refused(_Call):
    """A request that is not served: an attach that does not match the pool, a
    scheduler's attach before any worker of its engine has attached, or a
    PrepareLoad of a key the client holds no pin on."""

    reason: str


class LookupOk(_Call):
    result: Literal["miss", "hit", "hit_pending"]


class LoadSpec(_Call):
    """The chunk ids for the keys of a PrepareLoad, in the same order."""

    chunk_ids: list[int]


class StoreSpec(_Call):
    """The keys to store and their chunk ids, in the same order."""

    keys_to_store: list[OffloadKey]
    chunk_ids: list[int]


class NoSpace(_Call):
    """No chunks for a PrepareStore: the pool cannot evict enough, or it is fenced
    after a restart. vLLM's ``prepare_store`` returns None."""


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

ToTier = Request | Notice
"""Every message the External Primary Tier receives."""

FromTier = AttachOk | Refused | LookupOk | LoadSpec | StoreSpec | NoSpace | Stats | Draining | UnknownClient
"""Every message an engine process receives."""

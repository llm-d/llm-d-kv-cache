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

"""``Client``: one engine process's side of the protocol with the pool server.

The client speaks in keys, request ids and positions; an engine adapter
translates its engine's calls into them. Every engine process makes a client:

- A GPU worker calls ``attach`` with its rank to learn where the region is, and
  closes the client. It needs no connection after that, because the server gives
  every pin and write a deadline instead of watching engines.
- The scheduler attaches with no rank and keeps the client for the engine's
  lifetime. Each call that reads or changes the pool's records goes to the
  server: ``lookup``, ``prepare_load`` and ``prepare_store`` wait for a reply;
  the other calls are notices.

Pins: the server pins every hit, so no other engine evicts it before this engine
loads it. ``prepare_load`` keeps the pin for the copy and ``complete_load``
releases it. The client counts the hits of each step, takes off the keys it
loads, and releases the rest at ``end_step``.

While the server is unreachable or has forgotten this engine, the scheduler's
client is detached: every lookup misses and every store is skipped. It attaches
again at the end of a step once the server is back, so a step either runs
attached throughout or not at all, and its hits are released to the server that
granted them.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from kvpoold.api.v1 import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    CompatDescriptor,
    CompleteLoad,
    CompleteStore,
    Draining,
    FromServer,
    GetStats,
    Leave,
    LoadSpec,
    Lookup,
    LookupOk,
    NoSpace,
    OffloadKey,
    PrepareLoad,
    PrepareStore,
    Refused,
    ReleaseHits,
    RequestFinished,
    ShmFileRegion,
    Stats,
    StoreSpec,
    UnknownClient,
)
from kvpoold.api.v1.messages import BlockStored
from kvpoold.client.session import Session

logger = logging.getLogger(__name__)

DEFAULT_ATTACH_TIMEOUT_MS = 300_000
"""The first attach waits while the server creates and pre-faults the whole pool."""

DEFAULT_CALL_TIMEOUT_MS = 1000
"""How long the scheduler waits for one reply. The server answers in well under a
millisecond; a reply this late means the server is gone, and the step goes on
without the pool."""

LookupResult = Literal["miss", "hit", "hit_pending"]


class Client:
    def __init__(
        self,
        address: str,
        client_id: str,
        compat: CompatDescriptor,
        *,
        attach_timeout_ms: int = DEFAULT_ATTACH_TIMEOUT_MS,
        call_timeout_ms: int = DEFAULT_CALL_TIMEOUT_MS,
    ) -> None:
        """``client_id`` is the engine's id, the same in every process of the engine."""
        self.address = address
        self.client_id = client_id
        self.compat = compat
        self._attach_timeout_ms = attach_timeout_ms
        self._call_timeout_ms = call_timeout_ms
        self._session = Session(address)
        self._attached = False
        """Whether this is a scheduler's client, attached to the server now."""
        self._unloaded_hits: Counter[OffloadKey] = Counter()
        """This step's hits not yet loaded: the pins to release at the step's end."""

    @property
    def attached(self) -> bool:
        return self._attached

    def attach(self, rank: int | None) -> ShmFileRegion | None:
        """Attach this process: a GPU worker by its rank, the scheduler with no
        rank. Return the pool's region, or None when the server refuses or does
        not answer. A scheduler's client is attached from then on."""
        reply = self._session.call(
            Attach,
            timeout_ms=self._attach_timeout_ms,
            version=PROTOCOL_VERSION,
            client_id=self.client_id,
            rank=rank,
            compat=self.compat,
        )
        if isinstance(reply, AttachOk):
            self._attached = rank is None
            return reply.region
        if reply is None:
            _warn_without_pool(f"no answer from {self.address} within {self._attach_timeout_ms} ms")
        elif isinstance(reply, Refused):
            _warn_without_pool(f"the pool server refused the attach: {reply.reason}")
        else:
            _warn_without_pool(f"the pool server answered the attach with {type(reply).__name__}")
        return None

    # --- Loads ------------------------------------------------------------------

    def lookup(self, req_id: str, key: OffloadKey) -> LookupResult:
        """One key of one request. A hit is pinned until it is loaded or the step ends."""
        if not self._attached:
            return "miss"
        reply = self._call(Lookup, req_id=req_id, key=key)
        if not isinstance(reply, LookupOk):
            return "miss"
        if reply.result == "hit":
            self._unloaded_hits[key] += 1
        return reply.result

    def prepare_load(self, req_id: str, keys: Sequence[OffloadKey], positions: Sequence[int | None]) -> list[int]:
        """The chunk ids of keys this step got as hits, in the same order.

        Raises ``RuntimeError`` when the server does not prepare the load. The
        lookups' pins hold the chunks until their deadlines, so the chunk ids
        cannot have changed, but the client keeps no copy of them, and an engine
        that cannot fail a load cannot go on correctly: it would compute on KV
        that was never loaded.
        """
        keys = list(keys)
        for key in keys:
            assert self._unloaded_hits[key] > 0, f"{key!r} was not a hit of this step"
            self._unloaded_hits[key] -= 1
        reply = self._call(PrepareLoad, req_id=req_id, keys=keys, positions=list(positions))
        if not isinstance(reply, LoadSpec):
            raise RuntimeError(f"the pool server did not prepare the load: {reply!r}")
        return reply.chunk_ids

    def complete_load(self, keys: Sequence[OffloadKey]) -> None:
        self._session.notify(CompleteLoad(keys=list(keys)))

    def end_step(self) -> None:
        """Release the step's hits that were not loaded, and attach again if detached.

        The step's last call: a step either runs attached throughout or not at
        all, so its hits are released to the server that granted them.
        """
        hits = [(key, count) for key, count in self._unloaded_hits.items() if count > 0]
        self._unloaded_hits.clear()
        if hits:
            self._session.notify(ReleaseHits(hits=hits))
        if not self._attached and self._session.connected() and self.attach(rank=None) is not None:
            logger.info("Attached to the pool server again")

    # --- Stores -----------------------------------------------------------------

    def prepare_store(
        self, req_id: str, keys: Sequence[OffloadKey], positions: Sequence[int | None]
    ) -> tuple[list[OffloadKey], list[int]] | None:
        """The keys to store and their chunk ids, in the same order; None when the
        store is skipped: the pool has no space, or the client is detached."""
        if not self._attached:
            return None
        reply = self._call(PrepareStore, req_id=req_id, keys=list(keys), positions=list(positions))
        if isinstance(reply, StoreSpec):
            return reply.keys_to_store, reply.chunk_ids
        if not isinstance(reply, NoSpace):
            logger.warning("Skipped a store: the pool server answered %r", reply)
        return None

    def complete_store(
        self, keys: Sequence[OffloadKey], success: bool, events: list[BlockStored] | None = None
    ) -> None:
        """``events`` is one ``BlockStored`` per key for the server to publish, or
        None when the engine has KV events off or the store failed."""
        self._session.notify(CompleteStore(keys=list(keys), success=success, events=events))

    # --- Requests and lifetime ----------------------------------------------------

    def request_finished(self, req_id: str) -> None:
        self._session.notify(RequestFinished(req_id=req_id))

    def stats(self) -> dict[str, int] | None:
        """The pool's counters and this client's share of them; None on no answer."""
        reply = self._call(GetStats)
        return reply.values if isinstance(reply, Stats) else None

    def close(self) -> None:
        """Close the connection. An attached scheduler's client first sends
        ``Leave``, which releases the engine's pins and in-flight stores at once
        instead of at their deadlines, so it must be closed only once no copy is
        running. A GPU worker's client sends nothing: its ``Leave`` would release
        the whole engine's leases."""
        if self._attached:
            self._session.notify(Leave())
            self._attached = False
        self._session.close()

    # --- Talking to the server ---------------------------------------------------

    def _call(self, request_type: type, **fields) -> FromServer | None:
        """Send a request; on no reply, or on an answer that means the server lost
        this engine, mark the client detached."""
        reply = self._session.call(request_type, timeout_ms=self._call_timeout_ms, **fields)
        if reply is None or isinstance(reply, UnknownClient | Draining):
            if self._attached:
                logger.warning("Serving without the shared CPU pool until the pool server is back")
            self._attached = False
        return reply


def _warn_without_pool(reason: str) -> None:
    logger.warning("Serving without the shared CPU pool: %s", reason)

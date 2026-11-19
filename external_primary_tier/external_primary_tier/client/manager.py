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

"""``ExternalPrimaryTierClient``: the engine's offloading manager, a client of the External Primary Tier.

vLLM's ``OffloadingConnector`` calls it as it calls ``CPUOffloadingManager``. It
keeps no records of the pool: each call that reads or changes them goes to the
tier. ``lookup``, ``prepare_load`` and ``prepare_store`` wait for a reply; the
other calls are notices.

Pins: the tier pins every hit, so no other engine evicts it before this engine
loads it. ``prepare_load`` keeps the pin for the copy and ``complete_load``
releases it. The client counts the hits of each step, takes off the keys it
loads, and releases the rest at ``on_schedule_end``.

While the tier is unreachable or has forgotten this engine, every lookup misses
and every store is skipped. The client attaches again at the start of a step
once the tier is back.
"""

from collections import Counter
from collections.abc import Collection
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from external_primary_tier.client.session import Session
from external_primary_tier.protocol import (
    CompleteLoad,
    CompleteStore,
    Draining,
    Leave,
    LoadSpec,
    Lookup,
    LookupOk,
    NoSpace,
    PrepareLoad,
    PrepareStore,
    ReleaseHits,
    RequestFinished,
    StoreSpec,
    UnknownClient,
)

if TYPE_CHECKING:
    from external_primary_tier.client.spec import ExternalOffloadingSpec

logger = init_logger(__name__)

DEFAULT_CALL_TIMEOUT_MS = 1000
"""How long the scheduler waits for one reply. The tier answers in well under a
millisecond; a reply this late means the tier is gone, and the step goes on
without the pool."""

_LOOKUP_RESULTS = {"miss": LookupResult.MISS, "hit": LookupResult.HIT, "hit_pending": LookupResult.HIT_PENDING}


class ExternalPrimaryTierClient(OffloadingManager):
    def __init__(self, spec: "ExternalOffloadingSpec") -> None:
        self._spec = spec
        self._call_timeout_ms = int(spec.extra_config.get("call_timeout_ms", DEFAULT_CALL_TIMEOUT_MS))
        self._session = Session(spec.tier_address)
        self._attached = spec.attach(self._session, rank=None) is not None
        self._unloaded_hits: Counter[OffloadKey] = Counter()
        """This step's hits not yet loaded: the pins to release at the step's end."""

    # --- Loads ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if not self._attached:
            return LookupResult.MISS
        reply = self._call(Lookup, req_id=req_context.req_id, key=key)
        if not isinstance(reply, LookupOk):
            return LookupResult.MISS
        result = _LOOKUP_RESULTS[reply.result]
        if result is LookupResult.HIT:
            self._unloaded_hits[key] += 1
        return result

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        keys = list(keys)
        for key in keys:
            assert self._unloaded_hits[key] > 0, f"{key!r} was not a hit of this step"
            self._unloaded_hits[key] -= 1
        reply = self._call(PrepareLoad, req_id=req_context.req_id, keys=keys, positions=_positions(keys, req_context))
        if not isinstance(reply, LoadSpec):
            # The lookups' pins hold the chunks until their deadlines, so the
            # chunk ids cannot have changed, but this client keeps no copy of
            # them. vLLM cannot fail a load here, so the engine cannot go on
            # correctly: it would compute on KV that was never loaded.
            raise RuntimeError(f"the External Primary Tier did not prepare the load: {reply!r}")
        return CPULoadStoreSpec(reply.chunk_ids)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._session.notify(CompleteLoad(keys=list(keys)))

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        hits = [(key, count) for key, count in self._unloaded_hits.items() if count > 0]
        self._unloaded_hits.clear()
        if hits:
            self._session.notify(ReleaseHits(hits=hits))
        # The step's last manager call: a step either runs attached throughout
        # or not at all, so its hits are released to the tier that granted them.
        if not self._attached:
            self._attach()

    # --- Stores -----------------------------------------------------------------

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext) -> PrepareStoreOutput | None:
        if not self._attached:
            return None
        keys = list(keys)
        reply = self._call(PrepareStore, req_id=req_context.req_id, keys=keys, positions=_positions(keys, req_context))
        if isinstance(reply, StoreSpec):
            return PrepareStoreOutput(
                keys_to_store=reply.keys_to_store,
                store_spec=CPULoadStoreSpec(reply.chunk_ids),
                # The tier evicts for the whole node, and the connector does not
                # read this list.
                evicted_keys=[],
            )
        if not isinstance(reply, NoSpace):
            logger.warning("Skipped a store: the External Primary Tier answered %r", reply)
        return None

    def complete_store(self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True) -> None:
        self._session.notify(CompleteStore(keys=list(keys), success=success))

    # --- Requests and lifetime ----------------------------------------------------

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._session.notify(RequestFinished(req_id=req_context.req_id))

    def shutdown(self) -> None:
        # Leave releases this engine's pins and in-flight stores at once instead
        # of at their deadlines. The connector calls shutdown() after the
        # workers have stopped, so no copy is still running.
        if self._attached:
            self._session.notify(Leave())
        self._session.close()

    # --- Talking to the tier ------------------------------------------------------

    def _call(self, request_type: type, **fields):
        """Send a request; on no reply, or on an answer that means the tier lost
        this engine, mark the client detached."""
        reply = self._session.call(request_type, timeout_ms=self._call_timeout_ms, **fields)
        if reply is None or isinstance(reply, UnknownClient | Draining):
            if self._attached:
                logger.warning("Serving without the shared CPU pool until the External Primary Tier is back")
            self._attached = False
        return reply

    def _attach(self) -> None:
        if self._session.connected() and self._spec.attach(self._session, rank=None) is not None:
            logger.info("Attached to the External Primary Tier again")
            self._attached = True


def _positions(keys: list[OffloadKey], req_context: ReqContext) -> list[int | None]:
    """Each key's end-token position in the request: vLLM's cache policies order a request's keys by it."""
    return [req_context.get_offload_key_position(key) for key in keys]

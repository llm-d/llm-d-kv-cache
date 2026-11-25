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

"""``PoolOffloadingManager``: the engine's offloading manager, over a ``Client`` of the pool server.

vLLM's ``OffloadingConnector`` calls it as it calls ``CPUOffloadingManager``. It
keeps no records of the pool: it translates each call into the ``Client``'s
terms (request ids, positions, chunk ids) and the reply back into vLLM's types.
Pins, the step's hits, and serving without the pool while the server is away
are the client's (see ``kvpoold.client.client``).

KV events: the server publishes the events of the pool, so ``take_events`` yields
none. ``complete_store`` sends the content of each stored chunk for the server
to publish (see ``events``).
"""

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from kvpoold.adapters.vllm.events import build_stored_events

if TYPE_CHECKING:
    from kvpoold.adapters.vllm.spec import PoolOffloadingSpec

_LOOKUP_RESULTS = {"miss": LookupResult.MISS, "hit": LookupResult.HIT, "hit_pending": LookupResult.HIT_PENDING}


class PoolOffloadingManager(OffloadingManager):
    def __init__(self, spec: "PoolOffloadingSpec") -> None:
        self._spec = spec
        self.client = spec.make_client()
        self.client.attach(rank=None)
        self._sends_events = (
            spec.kv_events_config.enable_kv_cache_events and spec.kv_events_config.self_describing_kv_events
        )

    # --- Loads ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return _LOOKUP_RESULTS[self.client.lookup(req_context.req_id, key)]

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        # vLLM cannot fail a load here, so the client raises when the server does
        # not prepare it.
        keys = list(keys)
        return CPULoadStoreSpec(self.client.prepare_load(req_context.req_id, keys, _positions(keys, req_context)))

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self.client.complete_load(list(keys))

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        # The step's last manager call.
        self.client.end_step()

    # --- Stores -----------------------------------------------------------------

    # vLLM versions with a StoreTrigger pass ``trigger`` to both store calls: whether
    # the store comes after the chunk was computed or before a state block is reused.
    # The server stores either kind the same way, so the manager ignores it. Other
    # versions pass none, hence the default.

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext, *, trigger: object = None
    ) -> PrepareStoreOutput | None:
        keys = list(keys)
        stored = self.client.prepare_store(req_context.req_id, keys, _positions(keys, req_context))
        if stored is None:
            return None
        keys_to_store, chunk_ids = stored
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=CPULoadStoreSpec(chunk_ids),
            # The server evicts for the whole node, and the connector does not
            # read this list.
            evicted_keys=[],
        )

    def complete_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True, *, trigger: object = None
    ) -> None:
        keys = list(keys)
        # A failed store publishes nothing: the server drops its chunks, so it
        # announces no BlockStored for them and needs no content.
        events = (
            build_stored_events(keys, req_context, self._spec.kv_events_config)
            if success and self._sends_events
            else None
        )
        self.client.complete_store(keys, success, events)

    # --- Requests and lifetime ----------------------------------------------------

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self.client.request_finished(req_context.req_id)

    def shutdown(self) -> None:
        # The connector calls shutdown() after the workers have stopped, so no
        # copy is still running when the client's Leave releases the engine's
        # pins and in-flight stores.
        self.client.close()

    def take_events(self) -> Iterable[OffloadingEvent]:
        # The server publishes the pool's events, so the connector's tracker
        # has none to translate here and nothing is published twice.
        return ()


def _positions(keys: list[OffloadKey], req_context: ReqContext) -> list[int | None]:
    """Each key's end-token position in the request: vLLM's cache policies order a request's keys by it."""
    return [req_context.get_offload_key_position(key) for key in keys]

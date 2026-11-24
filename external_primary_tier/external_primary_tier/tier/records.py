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

"""The pool's records, shared by every engine, and each engine's leases on them.

The records are vLLM's own ``CPUOffloadingManager`` over the region's chunks, so
chunk allocation, eviction and the cache policy are vLLM's. The manager sees pool
keys: a client's compatibility id followed by vLLM's key, which keeps the chunks
of different models apart while they share eviction. The records add who holds
what:

- A pin is one unit of a key's ``ref_cnt``, held by one client. ``lookup`` pins
  each hit. The pin carries the load that ``prepare_load`` starts, and
  ``complete_load`` releases it; ``release_hits`` releases the hits that were not
  loaded.
- A write is a key with ``ref_cnt`` -1 that one client fills, from
  ``prepare_store`` until ``complete_store``.
- Every pin and write is a lease that ends ``lease_seconds`` after it is granted;
  ``sweep`` releases the pins and fails the writes whose leases have ended.

Each request a client calls about gets a vLLM ``ReqContext``, which carries the
request's state in the manager and the cache policy, and the positions of its
keys. ``request_finished`` hands it to the manager's ``on_request_finished``.

KV events: the manager's own events say when a chunk becomes readable (a store
completes) and when it is evicted. After every call that can change that, the
records read those events and turn them into vLLM KV events, which
``take_events`` hands out: a ``BlockStored`` when a chunk becomes readable and a
``BlockRemoved`` when it is evicted, both with medium ``MEDIUM_CPU``. A chunk's ``BlockStored`` comes from
the engine that stored it and is kept until the chunk is evicted, because the
``BlockRemoved`` is built from it. A chunk whose engine sent none produces no
events. A store that fails produces none either: the chunk was never readable, so
it was never published.

The snapshot keeps each chunk's ``BlockStored``. ``restore`` puts them back, so a
restored chunk's eviction publishes a ``BlockRemoved``; it publishes nothing at
restore, because the indexer already holds those chunks. A chunk that is being written when the snapshot
is taken is not readable, so it has no ``BlockStored`` until its store completes.
"""

from collections.abc import Iterable, Mapping, Sequence

from vllm.distributed.kv_events import MEDIUM_CPU, BlockRemoved, BlockStored
from vllm.v1.kv_offload.base import LookupResult, ReqContext

try:
    from vllm.v1.kv_offload.base import StoreTrigger
except ImportError:  # a vLLM whose store calls take no trigger
    StoreTrigger = None
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

from external_primary_tier.protocol import OffloadKey
from external_primary_tier.tier.snapshot import ChunkEntry, ClientSnapshot, Snapshot
from external_primary_tier.tier.state import ClientContext

_NO_REQUEST = ReqContext(req_id="")
"""For the manager calls that take a request context and do not read it."""

_STORE_TRIGGER: dict[str, object] = {"trigger": StoreTrigger.ON_COMPUTE} if StoreTrigger is not None else {}
"""The keyword that vLLM's store calls require when its version has a store trigger.
The pool's manager does not read it, so every store passes the same one."""


class Records:
    def __init__(
        self,
        num_chunks: int,
        cache_policy: str = "lru",
        lease_seconds: float = 30.0,
        *,
        cache_policy_module_path: str | None = None,
    ) -> None:
        self.cache_policy = cache_policy
        self.lease_seconds = lease_seconds
        # The manager's events are read after every call that emits them (see
        # _collect_events), so the manager's own list stays empty. Their
        # req_context is _NO_REQUEST and is not used.
        self.manager = CPUOffloadingManager(
            num_chunks,
            cache_policy=cache_policy,
            cache_policy_module_path=cache_policy_module_path,
            enable_events=True,
        )
        self.clients: dict[str, ClientContext] = {}
        """Client id to its context."""
        self.fence_until = float("-inf")
        """Before this time every lookup misses and no chunk is handed out: after a
        crash, an engine may still be copying under the instance that is gone."""
        self.block_stored: dict[bytes, BlockStored] = {}
        """Pool key to the ``BlockStored`` of each readable chunk whose engine sent
        one, with its medium set. It is the event that was published, or, for a restored
        chunk, the one the previous instance published."""
        self._events: list[BlockStored | BlockRemoved] = []

    # --- Calls ----------------------------------------------------------------

    def lookup(
        self,
        client: ClientContext,
        req_id: str,
        key: OffloadKey,
        now: float,
    ) -> LookupResult:
        """vLLM's lookup; a hit is pinned for the client."""
        if now < self.fence_until:
            return LookupResult.MISS
        pool_key = client.pool_key(key)
        context = self._req_context(client, req_id)
        result = self.manager.lookup(pool_key, context)
        if result is LookupResult.HIT:
            # The pin vLLM's prepare_load takes, without recording an access:
            # prepare_load records it if the request loads the key.
            # We pin to avoid eviction by other clients.
            self.manager._prepare_load([pool_key], context, record_access=False)
            client.pins.setdefault(pool_key, []).append(now + self.lease_seconds)
        return result

    def prepare_load(
        self,
        client: ClientContext,
        req_id: str,
        keys: Sequence[OffloadKey],
        positions: Sequence[int | None],
    ) -> list[int] | None:
        """Return the keys' chunk ids, or None when the client holds no pin on one
        of them. The lookups' pins carry the load."""
        pool_keys = [client.pool_key(key) for key in keys]
        if not all(client.pins.get(pool_key) for pool_key in pool_keys):
            return None
        context = self._req_context(client, req_id, pool_keys, positions)
        # The access vLLM's prepare_load records.
        self.manager._record_request_cache_access(pool_keys, context, reused_keys=pool_keys)
        chunks = []
        for key in pool_keys:
            chunk = self.manager._policy.get(key)
            assert chunk is not None, f"Chunk {key!r} not found in cache"
            assert chunk.is_ready, f"Chunk {key!r} is not ready for reading"
            chunks.append(chunk.chunk_id)
        return chunks

    def complete_load(self, client: ClientContext, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            self._release_pins(client, client.pool_key(key), 1)

    def release_hits(self, client: ClientContext, hits: Iterable[tuple[OffloadKey, int]]) -> None:
        """Release the step's hits that were not loaded, ``count`` pins per key."""
        for key, count in hits:
            self._release_pins(client, client.pool_key(key), count)

    def prepare_store(
        self,
        client: ClientContext,
        req_id: str,
        keys: Sequence[OffloadKey],
        positions: Sequence[int | None],
        now: float,
    ) -> tuple[list[OffloadKey], list[int]] | None:
        """vLLM's prepare_store: the keys to store, which become the client's
        writes, and their chunk ids. None when no chunks are handed out."""
        if now < self.fence_until:
            return None
        pool_keys = [client.pool_key(key) for key in keys]
        output = self.manager.prepare_store(
            pool_keys, self._req_context(client, req_id, pool_keys, positions), **_STORE_TRIGGER
        )
        # The evictions made to free chunks for this store.
        self._collect_events()
        if output is None:
            return None
        for pool_key in output.keys_to_store:
            client.writes[pool_key] = now + self.lease_seconds
        assert isinstance(output.store_spec, CPULoadStoreSpec)
        keys_to_store = [pool_key[len(client.compat_id) :] for pool_key in output.keys_to_store]
        return keys_to_store, output.store_spec.chunk_ids.tolist()

    def complete_store(
        self,
        client: ClientContext,
        keys: Sequence[OffloadKey],
        success: bool,
        events: Sequence[BlockStored] | None = None,
    ) -> None:
        """vLLM's complete_store, for the keys among the client's writes.

        ``events`` is one ``BlockStored`` per key, or None when the engine has KV
        events off. On success each is kept with its chunk, its medium is set to
        ``MEDIUM_CPU``, and it is buffered for ``take_events``; on failure they are
        dropped."""
        if events is not None and len(events) != len(keys):
            raise ValueError(f"{len(events)} events for {len(keys)} keys")
        stored: dict[bytes, BlockStored] = {}
        pool_keys = []
        for i, key in enumerate(keys):
            pool_key = client.pool_key(key)
            if client.writes.pop(pool_key, None) is None:
                continue
            pool_keys.append(pool_key)
            if events is not None:
                stored[pool_key] = events[i]
        if pool_keys:
            self.manager.complete_store(pool_keys, _NO_REQUEST, success=success, **_STORE_TRIGGER)
            self._collect_events(stored)

    def take_events(self) -> list[BlockStored | BlockRemoved]:
        """The events since the last call, oldest first, ready to publish."""
        events, self._events = self._events, []
        return events

    def request_finished(self, client: ClientContext, req_id: str) -> None:
        context = client.requests.pop(req_id, None)
        if context is not None:
            self.manager.on_request_finished(context)

    def release_client(self, client: ClientContext) -> None:
        """Release everything the client holds and forget it: its requests finish,
        its pins are released and its writes fail."""
        for req_id in list(client.requests):
            self.request_finished(client, req_id)
        for pool_key, deadlines in list(client.pins.items()):
            self._release_pins(client, pool_key, len(deadlines))
        self._fail_writes(client, list(client.writes))
        del self.clients[client.client_id]

    def sweep(self, now: float) -> int:
        """Release the pins and fail the writes whose leases have ended; return how
        many."""
        # Only vLLM's get_stats() clears this, and the records never call it.
        self.manager.allocation_sizes_in_current_batch.clear()
        ended = 0
        for client in self.clients.values():
            for pool_key, deadlines in list(client.pins.items()):
                count = sum(deadline <= now for deadline in deadlines)
                self._release_pins(client, pool_key, count)
                ended += count
            writes = [pool_key for pool_key, deadline in client.writes.items() if deadline <= now]
            self._fail_writes(client, writes)
            ended += len(writes)
        return ended

    def stats(self, client: ClientContext) -> dict[str, int]:
        manager = self.manager
        num_free = manager._get_num_free_chunks()
        num_evictable = manager._num_evictable_cache_chunks
        num_write_pending = manager._num_write_pending_chunks
        return {
            "num_chunks": manager._num_chunks,
            "num_free_chunks": num_free,
            "num_evictable_chunks": num_evictable,
            "num_write_pending_chunks": num_write_pending,
            "num_pinned_chunks": manager._num_chunks - num_free - num_evictable - num_write_pending,
            "num_clients": len(self.clients),
            "client_pins": sum(len(deadlines) for deadlines in client.pins.values()),
            "client_writes": len(client.writes),
        }

    # --- Leases ---------------------------------------------------------------

    def _req_context(
        self,
        client: ClientContext,
        req_id: str,
        pool_keys: Sequence[bytes] = (),
        positions: Sequence[int | None] = (),
    ) -> ReqContext:
        """The client's context for the request, made on first use, with the
        given keys' positions recorded."""
        context = client.requests.get(req_id)
        if context is None:
            context = client.requests[req_id] = ReqContext(req_id=req_id)
        for pool_key, position in zip(pool_keys, positions, strict=True):
            if position is not None:
                context.set_offload_key_position(pool_key, position)
        return context

    def _release_pins(self, client: ClientContext, pool_key: bytes, count: int) -> None:
        """Release up to ``count`` of the client's pins on the key, earliest lease
        first."""
        deadlines = client.pins.get(pool_key)
        if not deadlines or count <= 0:
            return
        count = min(count, len(deadlines))
        del deadlines[:count]
        if not deadlines:
            del client.pins[pool_key]
        self.manager.complete_load([pool_key] * count, _NO_REQUEST)

    def _fail_writes(self, client: ClientContext, pool_keys: list[bytes]) -> None:
        if not pool_keys:
            return
        for pool_key in pool_keys:
            del client.writes[pool_key]
        self.manager.complete_store(pool_keys, _NO_REQUEST, success=False, **_STORE_TRIGGER)
        self._collect_events()

    def _collect_events(self, new_block_stored: Mapping[bytes, BlockStored] | None = None) -> None:
        """Turn the manager's events into ``BlockStored`` and ``BlockRemoved`` events.

        ``new_block_stored`` holds the ``BlockStored`` of the keys whose store
        just completed. A stored event is kept with ``medium`` set to
        ``MEDIUM_CPU``. A removal is built from the kept ``BlockStored``, which
        is then dropped: its hashes and group, with ``MEDIUM_CPU``. The manager's
        removals of failed stores are dropped. A key with no ``BlockStored``
        produces nothing."""
        new_block_stored = new_block_stored or {}
        for event in self.manager.take_events():
            if event.metadata_only:
                continue
            for pool_key in event.keys:
                if event.removed:
                    kept = self.block_stored.pop(pool_key, None)
                    if kept is not None:
                        self._events.append(
                            BlockRemoved(block_hashes=kept.block_hashes, medium=MEDIUM_CPU, group_idx=kept.group_idx)
                        )
                else:
                    kept = new_block_stored.get(pool_key)
                    if kept is not None:
                        kept.medium = MEDIUM_CPU
                        self.block_stored[pool_key] = kept
                        self._events.append(kept)

    # --- Snapshot -------------------------------------------------------------

    def snapshot(self, region_id: bytes) -> Snapshot | None:
        """The records, or None for a cache policy other than LRU."""
        policy = self.manager._policy
        if type(policy) is not LRUCachePolicy:
            return None
        # Ascending rank is LRU's eviction order.
        keys = sorted(policy.chunks, key=policy._ranks.__getitem__)
        return Snapshot(
            region_id=region_id,
            num_allocated_chunks=self.manager._num_allocated_chunks,
            free_list=list(self.manager._free_list),
            entries=[
                ChunkEntry(key=key, chunk_id=policy.chunks[key].chunk_id, ref_cnt=policy.chunks[key].ref_cnt)
                for key in keys
            ],
            clients=[
                ClientSnapshot(
                    client_id=client.client_id,
                    compat_id=client.compat_id,
                    pins=[(pool_key, len(deadlines)) for pool_key, deadlines in client.pins.items()],
                    writes=list(client.writes),
                )
                for client in self.clients.values()
            ],
            block_stored=list(self.block_stored.items()),
        )

    @classmethod
    def restore(cls, snapshot: Snapshot, num_chunks: int, lease_seconds: float, *, lease_until: float) -> "Records":
        """LRU records from a snapshot. Its pins and writes are leased until
        ``lease_until``. The readable chunks get their ``BlockStored`` from the
        snapshot; a chunk being written has none."""
        records = cls(num_chunks, "lru", lease_seconds)
        manager = records.manager
        # Inserting in eviction order gives each key a later rank than the last.
        for entry in snapshot.entries:
            chunk = ChunkStatus(entry.chunk_id)
            chunk.ref_cnt = entry.ref_cnt
            manager._policy.insert(entry.key, chunk, _NO_REQUEST)
        manager._num_allocated_chunks = snapshot.num_allocated_chunks
        manager._free_list = list(snapshot.free_list)
        manager._num_evictable_cache_chunks = sum(entry.ref_cnt == 0 for entry in snapshot.entries)
        manager._num_write_pending_chunks = sum(entry.ref_cnt == -1 for entry in snapshot.entries)
        for saved in snapshot.clients:
            client = ClientContext(client_id=saved.client_id, compat_id=saved.compat_id)
            client.pins = {pool_key: [lease_until] * count for pool_key, count in saved.pins}
            client.writes = dict.fromkeys(saved.writes, lease_until)
            records.clients[saved.client_id] = client
        readable = {entry.key for entry in snapshot.entries if entry.ref_cnt != -1}
        records.block_stored = {key: event for key, event in snapshot.block_stored if key in readable}
        return records

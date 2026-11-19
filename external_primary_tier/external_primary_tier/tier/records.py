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
"""

from collections.abc import Iterable, Sequence

from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

from external_primary_tier.protocol import OffloadKey
from external_primary_tier.tier.snapshot import ChunkEntry, ClientSnapshot, Snapshot
from external_primary_tier.tier.state import ClientContext

_NO_REQUEST = ReqContext(req_id="")
"""For the manager calls that take a request context and do not read it."""


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
        # Events stay disabled: nothing in the External Primary Tier drains
        # them, and an event's request context carries engine-side request
        # data the protocol never sends. With them off, the manager never
        # reads the req_context the records pass as _NO_REQUEST.
        self.manager = CPUOffloadingManager(
            num_chunks,
            cache_policy=cache_policy,
            cache_policy_module_path=cache_policy_module_path,
        )
        self.clients: dict[str, ClientContext] = {}
        """Client id to its context."""
        self.fence_until = float("-inf")
        """Before this time every lookup misses and no chunk is handed out: after a
        crash, an engine may still be copying under the instance that is gone."""

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
        output = self.manager.prepare_store(pool_keys, self._req_context(client, req_id, pool_keys, positions))
        if output is None:
            return None
        for pool_key in output.keys_to_store:
            client.writes[pool_key] = now + self.lease_seconds
        assert isinstance(output.store_spec, CPULoadStoreSpec)
        keys_to_store = [pool_key[len(client.compat_id) :] for pool_key in output.keys_to_store]
        return keys_to_store, output.store_spec.chunk_ids.tolist()

    def complete_store(self, client: ClientContext, keys: Iterable[OffloadKey], success: bool) -> None:
        """vLLM's complete_store, for the keys among the client's writes."""
        pool_keys = [pool_key for key in keys if client.writes.pop(pool_key := client.pool_key(key), None) is not None]
        if pool_keys:
            self.manager.complete_store(pool_keys, _NO_REQUEST, success=success)

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
        self.manager.complete_store(pool_keys, _NO_REQUEST, success=False)

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
        )

    @classmethod
    def restore(cls, snapshot: Snapshot, num_chunks: int, lease_seconds: float, *, lease_until: float) -> "Records":
        """LRU records from a snapshot. Its pins and writes are leased until
        ``lease_until``."""
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
        return records

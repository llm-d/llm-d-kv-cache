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

"""The pool server's request loop.

One thread serves every engine on the node through one ROUTER socket. Each
engine process has a DEALER to it. As in vLLM's CPU EC connector, a message
arrives as the sender's identity (routing id), an empty delimiter and one msgpack payload,
and a reply goes back the same way. Before each message, and between messages, the loop sweeps the leases
whose deadlines have passed.

The first accepted attach creates and pre-faults the region before it is
answered, so that reply can take as long as pre-faulting the pool; engines wait
for it.

Start and stop:

- A new instance finds the region's files through the spec file beside them.
  With a snapshot from a clean stop, it loads the records, and the snapshot's
  pins and writes are held until the fence ends. Without one, after a crash, it
  starts with every chunk free, and until the fence ends every lookup misses and
  no chunk is handed out, so no store overwrites a chunk an engine may still be
  copying.
- When ``serve`` is told to stop, it answers the requests already queued with
  ``Draining``, applies the queued notices, publishes their events, and writes
  the snapshot.

KV events: the server creates vLLM's event publisher from its ``KVEventsConfig`` with
``EventPublisherFactory`` and publishes the pool's events (see
``Pool.take_events``) as one ``KVEventBatch`` after it has handled each burst of
messages. The publisher has its own thread and queue, and its topic is fixed.
``close`` shuts it down, which sends what is still queued. A config that does not
enable KV cache events is refused.

An instance that restores a snapshot publishes nothing at start: the indexer
already holds exactly the pool's chunks, because every store and eviction was
published and the stop published the last ones. The restored chunks keep their
``BlockStored``, so evicting one publishes its ``BlockRemoved``. Any other start
publishes one ``AllBlocksCleared`` before the socket is bound, because the indexer
may hold entries the pool no longer has: a fresh start, a crash (no snapshot), a
snapshot that is unreadable, of another region or of a cache policy that is not
snapshotted, or damaged region files. A subscriber that connects after that event
does not get it unless it asks the replay endpoint for it (see the README).
"""

import logging
import threading
import time
from collections.abc import Callable, Sequence

import msgspec
import zmq
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisher,
    EventPublisherFactory,
    KVEventBatch,
    NullEventPublisher,
)
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory

from kvpoold.api.v1 import (
    COMPAT_ID_BYTES,
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
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
    PrepareLoad,
    PrepareStore,
    Refused,
    ReleaseHits,
    Request,
    RequestFinished,
    Stats,
    StoreSpec,
    ToServer,
    UnknownClient,
)
from kvpoold.server.clients import ClientContext
from kvpoold.server.config import ServerConfig
from kvpoold.server.pool import Pool
from kvpoold.server.region import (
    MemorySpec,
    create_region,
    region_files_match,
    remove_region_files,
)
from kvpoold.server.snapshot import (
    read_spec,
    remove_spec_and_snapshot,
    take_snapshot,
    write_snapshot,
    write_spec,
)

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, config: ServerConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Raises ``ValueError`` when ``config.kv_events_config`` does not enable KV
        cache events. The server shuts the events' publisher down in ``close``."""
        CachePolicyFactory.get_cache_policy_cls(
            config.cache_policy, config.cache_policy_module_path
        )  # fails fast on an unknown name
        self.config = config
        self._region_dir = config.region_dir
        self._pool_bytes = config.pool_bytes
        self._cache_policy = config.cache_policy
        self._cache_policy_module_path = config.cache_policy_module_path
        self._lease_seconds = config.lease_seconds
        self._fence_seconds = config.fence_seconds
        self._clock = clock
        self._sweep_interval = min(1.0, config.lease_seconds / 4)
        self._next_sweep = float("-inf")
        self._draining = False
        self._event_publisher: EventPublisher = EventPublisherFactory.create(config.kv_events_config)
        if isinstance(self._event_publisher, NullEventPublisher):  # the config has events off
            raise ValueError("kv_events_config must enable KV cache events: the server always publishes them")

        self.memory_spec: MemorySpec | None = None
        self.pool: Pool | None = None
        """None until the region exists."""
        self._client_id_by_identity: dict[bytes, str] = {}
        """Each attached scheduler's routing id (zmq identity), to its client id.
        A GPU worker sends nothing after its attach, so it has no entry."""
        if not self._restore():
            self._clear_indexer()

        self._zmq_ctx = zmq.Context()
        self._router = make_zmq_socket(self._zmq_ctx, config.address, zmq.ROUTER, bind=True, linger=0)
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(ToServer)

    @property
    def event_publisher(self) -> EventPublisher:
        """The publisher of the pool's KV events."""
        return self._event_publisher

    @property
    def clients(self) -> dict[str, ClientContext]:
        return self.pool.clients if self.pool is not None else {}

    # --- Loop ---------------------------------------------------------------

    def serve(self, stop: threading.Event, poll_ms: int = 100) -> None:
        """Serve until ``stop`` is set, then drain and write the snapshot."""
        logger.info("pool server serving")
        while not stop.is_set():
            self.poll(poll_ms)
            self.sweep_if_due()
        self.drain()
        self.save_snapshot()

    def poll(self, timeout_ms: int) -> int:
        """Wait up to ``timeout_ms`` for messages, handle every pending one, and
        return how many were handled."""
        if not self._router.poll(timeout_ms, zmq.POLLIN):
            return 0
        handled = 0
        while True:
            try:
                frames = self._router.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            self.sweep_if_due()
            if len(frames) != 3 or frames[1] != b"":
                logger.warning("Dropped a message with a malformed envelope (%d frames)", len(frames))
                continue
            identity, _, payload = frames
            self._handle(identity, payload)
            handled += 1
        self._publish_events()
        return handled

    def drain(self, timeout_ms: int = 0) -> int:
        """From now on, answer every request with ``Draining``; handle the messages
        already queued, waiting up to ``timeout_ms`` for the first."""
        self._draining = True
        return self.poll(timeout_ms)

    def sweep_if_due(self) -> None:
        if self.pool is None:
            return
        now = self._clock()
        if now < self._next_sweep:
            return
        self._next_sweep = now + self._sweep_interval
        ended = self.pool.sweep(now)
        if ended:
            logger.warning("Released %d pins and writes whose leases ended", ended)

    def save_snapshot(self) -> None:
        """Write the records beside the region, for the next instance."""
        if self.pool is None or self.memory_spec is None:
            return
        snapshot = self.pool.snapshot(self.memory_spec.region.region_id)
        if snapshot is None:
            logger.warning("Only LRU records are snapshotted; the next instance starts empty")
            return
        write_snapshot(self._region_dir, snapshot)
        logger.info("Wrote the snapshot: %d clients", len(snapshot.clients))

    def close(self) -> None:
        """Close the sockets and shut the event publisher down, which sends the batches
        still queued. The region's files stay, for the next instance."""
        self._event_publisher.shutdown()
        self._router.close(linger=0)
        self._zmq_ctx.destroy(linger=0)

    # --- Dispatch -----------------------------------------------------------

    def _handle(self, identity: bytes, payload: bytes) -> None:
        try:
            message = self._decoder.decode(payload)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("Dropped a message that does not decode", exc_info=True)
            return

        if self._draining and isinstance(message, Request):
            self._reply(identity, Draining(call_id=message.call_id))
            return

        if isinstance(message, Attach):
            self._reply(identity, self._on_attach(identity, message))
            return

        client = self._client(identity)
        if client is None:
            # Requests are answered so the sender attaches again; notices from a
            # process that has not attached are dropped.
            if isinstance(message, Request):
                self._reply(identity, UnknownClient(call_id=message.call_id))
            return

        try:
            reply = self._serve(client, message)
        except Exception:
            logger.exception("Failed to serve %s from %s", type(message).__name__, client.client_id)
            if not isinstance(message, Request):
                return
            reply = Refused(call_id=message.call_id, reason="the pool server failed to serve it")
        if reply is not None:
            self._reply(identity, reply)

    def _publish_events(self) -> None:
        """Publish the events the handled messages produced as one batch."""
        if self.pool is None:
            return
        self._publish(self.pool.take_events())

    def _publish(self, events: Sequence[BlockStored | BlockRemoved | AllBlocksCleared]) -> None:
        if not events:
            return
        try:
            self._event_publisher.publish(KVEventBatch(ts=time.time(), events=list(events)))
        except Exception:
            logger.exception("Failed to publish %d KV events", len(events))

    def _clear_indexer(self) -> None:
        """Publish ``AllBlocksCleared``: the indexer may hold entries for chunks the
        pool no longer has, and this start has no event that removes them."""
        self._publish([AllBlocksCleared()])

    def _client(self, identity: bytes) -> ClientContext | None:
        client_id = self._client_id_by_identity.get(identity)
        return self.clients.get(client_id) if client_id is not None else None

    def _reply(self, identity: bytes, reply: FromServer) -> None:
        self._router.send_multipart([identity, b"", self._encoder.encode(reply)])

    def _serve(self, client: ClientContext, message: ToServer) -> FromServer | None:
        """Serve one message from an attached client; return the reply to a
        request, or None for a notice."""
        pool = self.pool
        assert pool is not None
        now = self._clock()
        match message:
            case Lookup():
                result = pool.lookup(client, message.req_id, message.key, now)
                return LookupOk(call_id=message.call_id, result=result.name.lower())
            case PrepareLoad():
                chunk_ids = pool.prepare_load(client, message.req_id, message.keys, message.positions)
                if chunk_ids is None:
                    return Refused(call_id=message.call_id, reason="a key to load is not pinned for this client")
                return LoadSpec(call_id=message.call_id, chunk_ids=chunk_ids)
            case PrepareStore():
                stored = pool.prepare_store(client, message.req_id, message.keys, message.positions, now)
                if stored is None:
                    return NoSpace(call_id=message.call_id)
                keys_to_store, chunk_ids = stored
                return StoreSpec(call_id=message.call_id, keys_to_store=keys_to_store, chunk_ids=chunk_ids)
            case GetStats():
                return Stats(call_id=message.call_id, values=pool.stats(client))
            case CompleteLoad():
                pool.complete_load(client, message.keys)
            case CompleteStore():
                pool.complete_store(client, message.keys, message.success, message.events)
            case RequestFinished():
                pool.request_finished(client, message.req_id)
            case ReleaseHits():
                pool.release_hits(client, message.hits)
            case Leave():
                if client.scheduler is not None:
                    self._client_id_by_identity.pop(client.scheduler, None)
                pool.release_client(client)
                logger.info("%s left", client.client_id)
        return None

    # --- Attach -------------------------------------------------------------

    def _on_attach(self, identity: bytes, msg: Attach) -> FromServer:
        def refuse(reason: str) -> Refused:
            logger.warning("Refused attach from %s (rank %s): %s", msg.client_id, msg.rank, reason)
            return Refused(call_id=msg.call_id, reason=reason)

        if msg.version != PROTOCOL_VERSION:
            return refuse(f"protocol version {msg.version}; this pool server speaks {PROTOCOL_VERSION}")

        # A GPU worker attaches once and keeps no connection, and the region
        # outlives the server, so a scheduler is let in whenever the pool exists.
        if msg.rank is None and self.memory_spec is None:
            return refuse("no GPU worker has attached, so the pool does not exist yet")
        if len(msg.compat.namespace) != COMPAT_ID_BYTES:
            return refuse(f"compatibility id of {len(msg.compat.namespace)} bytes, not {COMPAT_ID_BYTES}")
        context = self.clients.get(msg.client_id)
        if context is not None and msg.compat.namespace != context.compat_id:
            return refuse("compatibility id differs from the engine's other processes")

        reason = self._check_layout(msg)
        if reason is not None:
            return refuse(reason)
        assert self.memory_spec is not None and self.pool is not None

        if context is None:
            context = ClientContext(client_id=msg.client_id, compat_id=msg.compat.namespace)
            self.pool.clients[msg.client_id] = context
        if msg.rank is None:
            if context.scheduler is not None:
                self._client_id_by_identity.pop(context.scheduler, None)
            context.scheduler = identity
            self._client_id_by_identity[identity] = msg.client_id

        logger.info("Attached %s (rank %s)", msg.client_id, msg.rank)
        return AttachOk(call_id=msg.call_id, region=self.memory_spec.region)

    def _check_layout(self, msg: Attach) -> str | None:
        """Fix the pool's layout on the first attach; return why a later attach
        does not match it, or None."""
        if self.memory_spec is None:
            try:
                descriptor = create_region(self._region_dir, self._pool_bytes, msg.compat.layout)
            except Exception as e:
                logger.exception("Could not create the region")
                return f"the region could not be created: {e}"
            spec = MemorySpec(
                region=descriptor,
                hash_function=msg.compat.hash_function,
                hash_seed=msg.compat.hash_seed,
            )
            # Written before any engine learns where the region is.
            write_spec(self._region_dir, spec)
            self.memory_spec = spec
            self.pool = self._new_pool(descriptor.num_chunks)
            logger.info(
                "Created the region: %d chunks of %d bytes in %d files",
                descriptor.num_chunks,
                descriptor.kv_bytes_per_chunk,
                len(descriptor.segments),
            )
            return None

        spec = self.memory_spec
        if msg.compat.layout != spec.kv_bytes_per_chunk:
            return f"kv_bytes_per_chunk {msg.compat.layout} differs from the pool's {spec.kv_bytes_per_chunk}"
        if (msg.compat.hash_function, msg.compat.hash_seed) != (spec.hash_function, spec.hash_seed):
            return "the block hash function or seed differs from the pool's"
        return None

    # --- Start --------------------------------------------------------------

    def _new_pool(self, num_chunks: int) -> Pool:
        return Pool(
            num_chunks,
            self._cache_policy,
            self._lease_seconds,
            cache_policy_module_path=self._cache_policy_module_path,
        )

    def _restore(self) -> bool:
        """Take over the region and records a previous instance left, if any. Return
        True when a snapshot was restored, so the indexer holds exactly the restored
        chunks."""
        spec = read_spec(self._region_dir)
        if spec is None or not region_files_match(spec.region):
            # Without a spec, the files were never handed to an engine.
            remove_region_files(self._region_dir)
            remove_spec_and_snapshot(self._region_dir)
            return False

        region = spec.region
        self.memory_spec = spec
        now = self._clock()
        snapshot = take_snapshot(self._region_dir)
        if snapshot is not None and snapshot.region_id == region.region_id and self._cache_policy == "lru":
            self.pool = Pool.restore(
                snapshot,
                region.num_chunks,
                self._lease_seconds,
                lease_until=now + self._fence_seconds,
            )
            logger.info("Loaded the snapshot; its leases end in %.0f s", self._fence_seconds)
            return True

        self.pool = self._new_pool(region.num_chunks)
        self.pool.fence_until = now + self._fence_seconds
        logger.warning(
            "No snapshot of the region's records: every chunk is free, and no chunk is handed out for %.0f s",
            self._fence_seconds,
        )
        return False

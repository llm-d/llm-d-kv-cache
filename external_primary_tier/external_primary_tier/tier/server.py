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

"""The External Primary Tier's request loop.

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
  ``Draining``, applies the queued notices, and writes the snapshot.
"""

import threading
import time
from collections.abc import Callable

import msgspec
import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

from external_primary_tier.protocol import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    CompleteLoad,
    CompleteStore,
    Draining,
    FromTier,
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
    ToTier,
    UnknownClient,
)
from external_primary_tier.tier.records import Records
from external_primary_tier.tier.region import (
    create_region,
    region_files_match,
    remove_region_files,
)
from external_primary_tier.tier.snapshot import (
    read_spec,
    remove_spec_and_snapshot,
    take_snapshot,
    write_snapshot,
    write_spec,
)
from external_primary_tier.tier.state import (
    ClientContext,
    MemorySpec,
    SchedulerSession,
    WorkerSession,
)

logger = init_logger(__name__)


class ExternalPrimaryTier:
    def __init__(
        self,
        address: str,
        region_dir: str,
        pool_bytes: int,
        *,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        lease_seconds: float = 30.0,
        fence_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        CachePolicyFactory.get_cache_policy_cls(cache_policy, cache_policy_module_path)  # fails fast on an unknown name
        self._region_dir = region_dir
        self._pool_bytes = pool_bytes
        self._cache_policy = cache_policy
        self._cache_policy_module_path = cache_policy_module_path
        self._lease_seconds = lease_seconds
        self._fence_seconds = fence_seconds
        self._clock = clock
        self._sweep_interval = min(1.0, lease_seconds / 4)
        self._next_sweep = float("-inf")
        self._draining = False

        self.memory_spec: MemorySpec | None = None
        self.records: Records | None = None
        """None until the region exists."""
        self._region: SharedOffloadRegion | None = None
        """Keeps the region this instance created mapped for the life of the process."""
        self._client_of: dict[bytes, str] = {}
        """Each attached process's routing id (zmq identity), to its client id."""
        self._restore()

        self._zmq_ctx = zmq.Context()
        self._router = make_zmq_socket(self._zmq_ctx, address, zmq.ROUTER, bind=True, linger=0)
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(ToTier)

    @property
    def clients(self) -> dict[str, ClientContext]:
        return self.records.clients if self.records is not None else {}

    # --- Loop ---------------------------------------------------------------

    def serve(self, stop: threading.Event, poll_ms: int = 100) -> None:
        """Serve until ``stop`` is set, then drain and write the snapshot."""
        logger.info("External Primary Tier serving")
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
                return handled
            self.sweep_if_due()
            if len(frames) != 3 or frames[1] != b"":
                logger.warning("Dropped a message with a malformed envelope (%d frames)", len(frames))
                continue
            identity, _, payload = frames
            self._handle(identity, payload)
            handled += 1

    def drain(self, timeout_ms: int = 0) -> int:
        """From now on, answer every request with ``Draining``; handle the messages
        already queued, waiting up to ``timeout_ms`` for the first."""
        self._draining = True
        return self.poll(timeout_ms)

    def sweep_if_due(self) -> None:
        if self.records is None:
            return
        now = self._clock()
        if now < self._next_sweep:
            return
        self._next_sweep = now + self._sweep_interval
        ended = self.records.sweep(now)
        if ended:
            logger.warning("Released %d pins and writes whose leases ended", ended)

    def save_snapshot(self) -> None:
        """Write the records beside the region, for the next instance."""
        if self.records is None or self.memory_spec is None:
            return
        snapshot = self.records.snapshot(self.memory_spec.region.region_id)
        if snapshot is None:
            logger.warning("Only LRU records are snapshotted; the next instance starts empty")
            return
        write_snapshot(self._region_dir, snapshot)
        logger.info("Wrote the snapshot: %d clients", len(snapshot.clients))

    def close(self) -> None:
        """Close the socket. The region's files stay, for the next instance."""
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
            reply = Refused(call_id=message.call_id, reason="the External Primary Tier failed to serve it")
        if reply is not None:
            self._reply(identity, reply)

    def _client(self, identity: bytes) -> ClientContext | None:
        client_id = self._client_of.get(identity)
        return self.clients.get(client_id) if client_id is not None else None

    def _reply(self, identity: bytes, reply: FromTier) -> None:
        self._router.send_multipart([identity, b"", self._encoder.encode(reply)])

    def _serve(self, client: ClientContext, message: ToTier) -> FromTier | None:
        """Serve one message from an attached client; return the reply to a
        request, or None for a notice."""
        records = self.records
        assert records is not None
        now = self._clock()
        match message:
            case Lookup():
                result = records.lookup(client, message.req_id, message.key, now)
                return LookupOk(call_id=message.call_id, result=result.name.lower())
            case PrepareLoad():
                chunk_ids = records.prepare_load(client, message.req_id, message.keys, message.positions)
                if chunk_ids is None:
                    return Refused(call_id=message.call_id, reason="a key to load is not pinned for this client")
                return LoadSpec(call_id=message.call_id, chunk_ids=chunk_ids)
            case PrepareStore():
                stored = records.prepare_store(client, message.req_id, message.keys, message.positions, now)
                if stored is None:
                    return NoSpace(call_id=message.call_id)
                keys_to_store, chunk_ids = stored
                return StoreSpec(call_id=message.call_id, keys_to_store=keys_to_store, chunk_ids=chunk_ids)
            case GetStats():
                return Stats(call_id=message.call_id, values=records.stats(client))
            case CompleteLoad():
                records.complete_load(client, message.keys)
            case CompleteStore():
                records.complete_store(client, message.keys, message.success)
            case RequestFinished():
                records.request_finished(client, message.req_id)
            case ReleaseHits():
                records.release_hits(client, message.hits)
            case Leave():
                for identity in client.identities():
                    self._client_of.pop(identity, None)
                records.release_client(client)
                logger.info("%s left", client.client_id)
        return None

    # --- Attach -------------------------------------------------------------

    def _on_attach(self, identity: bytes, msg: Attach) -> FromTier:
        def refuse(reason: str) -> Refused:
            logger.warning("Refused attach from %s (rank %s): %s", msg.client_id, msg.rank, reason)
            return Refused(call_id=msg.call_id, reason=reason)

        if msg.version != PROTOCOL_VERSION:
            return refuse(f"protocol version {msg.version}; this External Primary Tier speaks {PROTOCOL_VERSION}")

        context = self.clients.get(msg.client_id)
        if msg.rank is None and (context is None or not context.workers):
            return refuse("no GPU worker of this engine has attached")
        if context is not None and msg.compat_id != context.compat_id:
            return refuse("compatibility id differs from the engine's other processes")

        reason = self._check_layout(msg)
        if reason is not None:
            return refuse(reason)
        assert self.memory_spec is not None and self.records is not None

        if context is None:
            context = ClientContext(client_id=msg.client_id, compat_id=msg.compat_id)
            self.records.clients[msg.client_id] = context
        if msg.rank is None:
            if context.scheduler is not None:
                self._client_of.pop(context.scheduler.identity, None)
            context.scheduler = SchedulerSession(identity=identity)
        else:
            worker = context.worker(msg.rank)
            if worker is not None:
                self._client_of.pop(worker.identity, None)
                worker.identity = identity
            else:
                context.workers.append(WorkerSession(identity=identity, rank=msg.rank))
        self._client_of[identity] = msg.client_id

        logger.info("Attached %s (rank %s)", msg.client_id, msg.rank)
        return AttachOk(call_id=msg.call_id, region=self.memory_spec.region)

    def _check_layout(self, msg: Attach) -> str | None:
        """Fix the pool's layout on the first attach; return why a later attach
        does not match it, or None."""
        if self.memory_spec is None:
            try:
                self._region, descriptor = create_region(self._region_dir, self._pool_bytes, msg.kv_bytes_per_chunk)
            except Exception as e:
                logger.exception("Could not create the region")
                return f"the region could not be created: {e}"
            spec = MemorySpec(
                region=descriptor,
                hash_function=msg.hash_function,
                hash_seed=msg.hash_seed,
                protocol_version=PROTOCOL_VERSION,
            )
            # Written before any engine learns where the region is.
            write_spec(self._region_dir, spec)
            self.memory_spec = spec
            self.records = self._new_records(descriptor.num_chunks)
            logger.info(
                "Created the region: %d chunks of %d bytes in %d files",
                descriptor.num_chunks,
                descriptor.kv_bytes_per_chunk,
                len(descriptor.segments),
            )
            return None

        spec = self.memory_spec
        if msg.kv_bytes_per_chunk != spec.kv_bytes_per_chunk:
            return f"kv_bytes_per_chunk {msg.kv_bytes_per_chunk} differs from the pool's {spec.kv_bytes_per_chunk}"
        if (msg.hash_function, msg.hash_seed) != (spec.hash_function, spec.hash_seed):
            return "the block hash function or seed differs from the pool's"
        return None

    # --- Start --------------------------------------------------------------

    def _new_records(self, num_chunks: int) -> Records:
        return Records(
            num_chunks,
            self._cache_policy,
            self._lease_seconds,
            cache_policy_module_path=self._cache_policy_module_path,
        )

    def _restore(self) -> None:
        """Take over the region and records a previous instance left, if any."""
        spec = read_spec(self._region_dir)
        if spec is None or not region_files_match(spec.region):
            # Without a spec, the files were never handed to an engine.
            remove_region_files(self._region_dir)
            remove_spec_and_snapshot(self._region_dir)
            return

        region = spec.region
        self.memory_spec = spec
        now = self._clock()
        snapshot = take_snapshot(self._region_dir)
        if snapshot is not None and snapshot.region_id == region.region_id and self._cache_policy == "lru":
            self.records = Records.restore(
                snapshot, region.num_chunks, self._lease_seconds, lease_until=now + self._fence_seconds
            )
            logger.info("Loaded the snapshot; its leases end in %.0f s", self._fence_seconds)
            return

        self.records = self._new_records(region.num_chunks)
        self.records.fence_until = now + self._fence_seconds
        logger.warning(
            "No snapshot of the region's records: every chunk is free, and no chunk is handed out for %.0f s",
            self._fence_seconds,
        )

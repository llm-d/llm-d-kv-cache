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
engine process has a DEALER to it; a message arrives as the sender's routing id
followed by the frames of one encoded message, and a reply goes back to the same
routing id.

The first accepted attach creates and pre-faults the region before it is
answered, so that reply can take as long as pre-faulting the pool; engines wait
for it.
"""

import threading
from collections.abc import Sequence

import msgspec
import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

from external_primary_tier.protocol import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    FromTier,
    Refused,
    Request,
    UnknownClient,
    tier_codec,
)
from external_primary_tier.tier.region import create_region
from external_primary_tier.tier.state import (
    ClientContext,
    MemorySpec,
    SchedulerSession,
    WorkerSession,
)

logger = init_logger(__name__)


class ExternalPrimaryTier:
    def __init__(self, address: str, region_dir: str, pool_bytes: int) -> None:
        self._region_dir = region_dir
        self._pool_bytes = pool_bytes
        self._zmq_ctx = zmq.Context()
        self._router = make_zmq_socket(self._zmq_ctx, address, zmq.ROUTER, bind=True, linger=0)
        self._codec = tier_codec()

        self.memory_spec: MemorySpec | None = None
        self._region: SharedOffloadRegion | None = None
        """Keeps the region mapped for the life of the process."""
        self.clients: dict[str, ClientContext] = {}
        self._client_of: dict[bytes, str] = {}
        """Each attached process's routing id, to its client id."""

    # --- Loop ---------------------------------------------------------------

    def serve(self, stop: threading.Event, poll_ms: int = 100) -> None:
        """Serve until ``stop`` is set."""
        logger.info("External Primary Tier serving")
        while not stop.is_set():
            self.poll(poll_ms)

    def poll(self, timeout_ms: int) -> int:
        """Wait up to ``timeout_ms`` for messages, handle every pending one, and
        return how many were handled."""
        if not self._router.poll(timeout_ms, zmq.POLLIN):
            return 0
        handled = 0
        while True:
            try:
                frames = self._router.recv_multipart(zmq.NOBLOCK, copy=False)
            except zmq.Again:
                return handled
            self._handle(frames[0].bytes, frames[1:])
            handled += 1

    def close(self) -> None:
        """Close the socket. The region's files stay, for the next instance."""
        self._router.close(linger=0)
        self._zmq_ctx.destroy(linger=0)

    # --- Dispatch -----------------------------------------------------------

    def _handle(self, routing_id: bytes, frames: Sequence[zmq.Frame]) -> None:
        try:
            message = self._codec.decode(frames)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("Dropped a message that does not decode", exc_info=True)
            return

        if isinstance(message, Attach):
            self._reply(routing_id, self._on_attach(routing_id, message))
            return

        if routing_id not in self._client_of:
            # Requests are answered so the sender attaches again; notices from a
            # process that has not attached are dropped.
            if isinstance(message, Request):
                self._reply(routing_id, UnknownClient(call_id=message.call_id))
            return

        logger.warning("Not served: %s", type(message).__name__)

    def _reply(self, routing_id: bytes, reply: FromTier) -> None:
        self._router.send_multipart([routing_id, *self._codec.encode(reply)])

    # --- Attach -------------------------------------------------------------

    def _on_attach(self, routing_id: bytes, msg: Attach) -> FromTier:
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
        assert self.memory_spec is not None

        if context is None:
            context = ClientContext(client_id=msg.client_id, compat_id=msg.compat_id)
            self.clients[msg.client_id] = context
        if msg.rank is None:
            if context.scheduler is not None:
                self._client_of.pop(context.scheduler.routing_id, None)
            context.scheduler = SchedulerSession(routing_id=routing_id)
        else:
            worker = context.worker(msg.rank)
            if worker is not None:
                self._client_of.pop(worker.routing_id, None)
                worker.routing_id = routing_id
            else:
                context.workers.append(WorkerSession(routing_id=routing_id, rank=msg.rank))
        self._client_of[routing_id] = msg.client_id

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
            self.memory_spec = MemorySpec(
                region=descriptor,
                hash_function=msg.hash_function,
                hash_seed=msg.hash_seed,
                protocol_version=PROTOCOL_VERSION,
            )
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

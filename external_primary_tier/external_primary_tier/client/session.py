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

"""One engine process's connection to the External Primary Tier: a DEALER to its ROUTER.

The DEALER queues messages only while it is connected. While the tier is down,
``connected()`` is False, a request returns None at once, and a notice is
dropped: the engine serves without the pool rather than wait on a timeout. ZMQ
reconnects on its own when the tier is back.
"""

import itertools
import time
from typing import Any

import msgspec
import zmq
from vllm.logger import init_logger

from external_primary_tier.protocol import FromTier, ToTier

logger = init_logger(__name__)

# The heartbeat vLLM's CPU EC connector uses for its producers: a tier that stops
# answering is disconnected within the timeout, even if its socket stays open.
HEARTBEAT_IVL_MS = 2000
HEARTBEAT_TIMEOUT_MS = 4000

LEAVE_FLUSH_MS = 1000
"""How long ``close()`` waits for queued messages, such as ``Leave``, to reach the tier."""


class Session:
    def __init__(self, address: str) -> None:
        self._call_ids = itertools.count(1)
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(FromTier)
        self._context = zmq.Context()
        self._dealer = self._context.socket(zmq.DEALER)
        # Set before connect(): queue nothing while no tier is connected.
        self._dealer.setsockopt(zmq.IMMEDIATE, 1)
        self._dealer.setsockopt(zmq.LINGER, 0)
        self._dealer.setsockopt(zmq.HEARTBEAT_IVL, HEARTBEAT_IVL_MS)
        self._dealer.setsockopt(zmq.HEARTBEAT_TIMEOUT, HEARTBEAT_TIMEOUT_MS)
        self._dealer.connect(address)

    def connected(self, timeout_ms: int = 0) -> bool:
        """Whether a tier is connected, waiting up to ``timeout_ms`` for one."""
        return bool(self._dealer.poll(timeout_ms, zmq.POLLOUT))

    def call(self, request_type: type, timeout_ms: int, **fields: Any) -> FromTier | None:
        """Send a request and return its reply, or None if none arrives within ``timeout_ms``.

        Each request gets its own ``call_id``, so a late reply to an earlier
        request is dropped rather than taken for this one's.

        Known limitation: a call is synchronous, with one request outstanding
        and one reply to it. A reply with another ``call_id`` is consumed and
        discarded, so requests cannot be pipelined, and a request cannot be
        answered in several parts, without matching replies by ``call_id`` and
        buffering them instead.
        """
        call_id = next(self._call_ids)
        if not self.connected(timeout_ms):
            return None
        if not self._send(request_type(call_id=call_id, **fields)):
            return None
        deadline = time.monotonic() + timeout_ms / 1000
        while (remaining_ms := int((deadline - time.monotonic()) * 1000)) > 0:
            if not self._dealer.poll(remaining_ms, zmq.POLLIN):
                break
            reply = self._receive()
            if reply is not None and reply.call_id == call_id:
                return reply
        return None

    def notify(self, notice: ToTier) -> None:
        """Send a notice, which gets no reply. It is dropped while no tier is connected."""
        if not self._send(notice):
            logger.debug("Dropped %s: no External Primary Tier is connected", type(notice).__name__)

    def close(self) -> None:
        """Close the connection, after up to ``LEAVE_FLUSH_MS`` for queued messages to go out."""
        self._dealer.close(linger=LEAVE_FLUSH_MS)
        self._context.term()

    def _send(self, message: ToTier) -> bool:
        try:
            self._dealer.send_multipart([b"", self._encoder.encode(message)], flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

    def _receive(self) -> FromTier | None:
        frames = self._dealer.recv_multipart()
        if len(frames) != 2 or frames[0] != b"":
            logger.warning("Dropped a reply with a malformed envelope (%d frames)", len(frames))
            return None
        try:
            return self._decoder.decode(frames[1])
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("Dropped a reply that does not decode")
            return None

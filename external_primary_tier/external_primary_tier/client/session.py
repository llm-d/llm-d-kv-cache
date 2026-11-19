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

"""One engine process's connection to the External Primary Tier: a DEALER to its ROUTER."""

import itertools
import time
from typing import Any

import msgspec
import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

from external_primary_tier.protocol import FromTier

logger = init_logger(__name__)


class Session:
    def __init__(self, address: str) -> None:
        self._call_ids = itertools.count(1)
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(FromTier)
        self._context = zmq.Context()
        self._dealer = make_zmq_socket(self._context, address, zmq.DEALER, bind=False, linger=0)

    def call(self, request_type: type, timeout_ms: int, **fields: Any) -> FromTier | None:
        """Send a request and return its reply, or None if none arrives within ``timeout_ms``.

        Each request gets its own ``call_id``, so a late reply to an earlier
        request is dropped rather than taken for this one's.
        """
        call_id = next(self._call_ids)
        self._dealer.send_multipart([b"", self._encoder.encode(request_type(call_id=call_id, **fields))])
        deadline = time.monotonic() + timeout_ms / 1000
        while (remaining_ms := int((deadline - time.monotonic()) * 1000)) > 0:
            if not self._dealer.poll(remaining_ms, zmq.POLLIN):
                break
            reply = self._receive()
            if reply is not None and reply.call_id == call_id:
                return reply
        return None

    def close(self) -> None:
        self._context.destroy(linger=0)

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

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


"""A stand-in for one engine process: a DEALER that sends requests and notices
and waits for replies."""

import msgspec
import zmq
from vllm.utils.network_utils import make_zmq_socket

from kvpoold.api.v1 import PROTOCOL_VERSION, Attach, CompatDescriptor, FromServer

KV_BYTES_PER_CHUNK = 1 << 20
COMPAT = CompatDescriptor(layout=KV_BYTES_PER_CHUNK, namespace=b"\x11" * 8, hash_function="sha256", hash_seed=None)


def key(i: int, group: int = 0) -> bytes:
    """vLLM's OffloadKey: a 32-byte block hash and the group index."""
    return i.to_bytes(32, "big") + group.to_bytes(4, "big")


class FakeEngineProcess:
    def __init__(self, address: str, timeout_ms: int = 10_000) -> None:
        self._ctx = zmq.Context()
        self._dealer = make_zmq_socket(self._ctx, address, zmq.DEALER, bind=False, linger=0)
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(FromServer)
        self._timeout_ms = timeout_ms
        self._next_call_id = 1

    def next_call_id(self) -> int:
        call_id = self._next_call_id
        self._next_call_id += 1
        return call_id

    def send(self, message) -> None:
        """Send a message without waiting for a reply."""
        self._dealer.send_multipart([b"", self._encoder.encode(message)])

    def receive(self):
        if not self._dealer.poll(self._timeout_ms, zmq.POLLIN):
            raise TimeoutError("no reply")
        delimiter, payload = self._dealer.recv_multipart()
        assert delimiter == b""
        return self._decoder.decode(payload)

    def request(self, message):
        """Send a request and return its reply."""
        self.send(message)
        reply = self.receive()
        assert reply.call_id == message.call_id
        return reply

    def call(self, request_type, **fields):
        """Send a request of ``request_type`` with the next call id; return its reply."""
        return self.request(request_type(call_id=self.next_call_id(), **fields))

    def attach(self, client_id: str, rank: int | None, **fields):
        return self.request(self.attach_message(client_id, rank, **fields))

    def attach_message(self, client_id: str, rank: int | None, *, version: int = PROTOCOL_VERSION, **compat) -> Attach:
        """``compat`` replaces fields of ``COMPAT``."""
        return Attach(
            call_id=self.next_call_id(),
            version=version,
            client_id=client_id,
            rank=rank,
            compat=msgspec.structs.replace(COMPAT, **compat),
        )

    def close(self) -> None:
        self._dealer.close(linger=0)
        self._ctx.destroy(linger=0)

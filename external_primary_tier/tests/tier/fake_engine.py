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

"""A stand-in for one engine process: a DEALER that sends requests and waits
for their replies."""

import zmq
from external_primary_tier.protocol import PROTOCOL_VERSION, Attach, client_codec
from vllm.utils.network_utils import make_zmq_socket

KV_BYTES_PER_CHUNK = 1 << 20
COMPAT_ID = b"\x11" * 8


class FakeEngineProcess:
    def __init__(self, address: str, timeout_ms: int = 10_000) -> None:
        self._ctx = zmq.Context()
        self._dealer = make_zmq_socket(self._ctx, address, zmq.DEALER, bind=False, linger=0)
        self._codec = client_codec()
        self._timeout_ms = timeout_ms
        self._next_call_id = 1

    def next_call_id(self) -> int:
        call_id = self._next_call_id
        self._next_call_id += 1
        return call_id

    def request(self, message):
        """Send a request and return its reply."""
        self._dealer.send_multipart(self._codec.encode(message))
        if not self._dealer.poll(self._timeout_ms, zmq.POLLIN):
            raise TimeoutError(f"no reply to {type(message).__name__}")
        reply = self._codec.decode(self._dealer.recv_multipart())
        assert reply.call_id == message.call_id
        return reply

    def attach(self, client_id: str, rank: int | None, **overrides):
        fields = dict(
            call_id=self.next_call_id(),
            version=PROTOCOL_VERSION,
            client_id=client_id,
            rank=rank,
            kv_bytes_per_chunk=KV_BYTES_PER_CHUNK,
            hash_function="sha256",
            hash_seed=None,
            compat_id=COMPAT_ID,
        )
        fields.update(overrides)
        return self.request(Attach(**fields))

    def close(self) -> None:
        self._dealer.close(linger=0)
        self._ctx.destroy(linger=0)

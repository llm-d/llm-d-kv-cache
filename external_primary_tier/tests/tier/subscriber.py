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

"""A stand-in for the llm-d KV-cache indexer: a SUB socket that decodes batches
as vLLM's ``KVEventBatch``."""

import time

import msgspec
import zmq
from vllm.distributed.kv_events import EventPublisher, KVEventBatch


class Subscriber:
    def __init__(self, endpoint: str, topic: str, timeout_ms: int = 5_000) -> None:
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._sub.connect(endpoint)
        self._decoder = msgspec.msgpack.Decoder(KVEventBatch)
        self._timeout_ms = timeout_ms

    def wait_until_subscribed(self, publisher: EventPublisher) -> None:
        """Publish empty batches until one arrives, so the batches sent after this
        are not lost to a subscription that has not reached the publisher yet.
        ``receive`` skips empty batches, which the tier never publishes."""
        while not self._sub.poll(50):
            publisher.publish(KVEventBatch(ts=time.time(), events=[]))

    def receive(self) -> tuple[str, KVEventBatch]:
        """The next batch with events as ``(topic, batch)``; raises if none arrives in time."""
        while True:
            if not self._sub.poll(self._timeout_ms):
                raise TimeoutError("no batch")
            topic, _, payload = self._sub.recv_multipart()
            batch = self._decoder.decode(payload)
            if batch.events:
                return topic.decode(), batch

    def close(self) -> None:
        self._sub.close(linger=0)
        self._ctx.destroy(linger=0)

# Copyright 2026 The llm-d Authors.
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

import logging
import struct
import threading
import time
from collections.abc import Iterable

import msgpack
import zmq

logger = logging.getLogger(__name__)

_UINT64_MASK = (1 << 64) - 1


def _hash_to_uint64(block_hash: int | bytes) -> int:
    """Mask each hash to lower 64 bits, matching the FileMapper."""
    if isinstance(block_hash, bytes):
        return int.from_bytes(block_hash, "big") & _UINT64_MASK
    return int(block_hash) & _UINT64_MASK


class StorageEventPublisher:
    """Publishes storage-tier KV cache events via ZMQ PUB socket.

    Events use the same msgpack positional-array format as vLLM's GPU
    KV events so the Go vLLMAdapter can parse them without modification.
    ZMQ messages use the 3-frame format expected by zmq_subscriber.go:
    [topic, sequence, payload].
    """

    def __init__(
        self, endpoint: str, model_name: str, source_id: str = "SHARED_STORAGE"
    ):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 1000)
        self._socket.bind(endpoint)

        self._model_name = model_name
        self._source_id = source_id
        self._topic = f"kv@{self._source_id}@{self._model_name}"
        self._seq: int = 0
        self._closed = False
        self._send_lock = threading.Lock()
        logger.info(
            "StorageEventPublisher bound to %s (topic: %s)",
            endpoint,
            self._topic,
        )

    def publish_blocks_stored(self, block_hashes: Iterable[int | bytes]) -> None:
        hashes = [_hash_to_uint64(h) for h in block_hashes]
        if not hashes:
            return

        events: list[bytes] = []
        for h in hashes:
            event = [
                "BlockStored",  # [0] tag
                [h],  # [1] block_hashes (one per file)
                0,  # [2] parent_hash (unused)
                [],  # [3] token_ids (empty)
                0,  # [4] block_size (unused)
                None,  # [5] lora_id
                "SHARED_STORAGE",  # [6] medium / device tier
            ]
            events.append(msgpack.packb(event, use_bin_type=True))

        self._send_batch(events)

    def _send_batch(self, packed_events: list[bytes]) -> None:
        batch = [time.time(), packed_events]
        payload = msgpack.packb(batch, use_bin_type=True)

        with self._send_lock:
            if self._closed:
                return

            self._seq += 1
            self._socket.send_multipart(
                [
                    self._topic.encode("utf-8"),
                    struct.pack(">Q", self._seq),
                    payload,
                ]
            )

    def close(self) -> None:
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
            self._socket.close()
            self._ctx.term()

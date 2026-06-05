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

import enum
import logging
import struct
import threading
import time
from collections.abc import Iterable

import msgpack
import zmq

logger = logging.getLogger(__name__)

_UINT64_MASK = (1 << 64) - 1
DEFAULT_STORAGE_EVENTS_HWM = 100_000  # vLLM's default


class StorageMedium(enum.Enum):
    """Storage backend types used as the device-tier label in events."""

    SHARED_STORAGE = "SHARED_STORAGE"
    OBJECT_STORE = "OBJECT_STORE"


def _hash_to_uint64(block_hash: int | bytes) -> int:
    """Mask a block hash to its lower 64 bits to match the FileMapper truncation."""
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
        self,
        endpoint: str,
        model_name: str | None = None,
        sndhwm: int = DEFAULT_STORAGE_EVENTS_HWM,
        medium: StorageMedium = StorageMedium.SHARED_STORAGE,
    ) -> None:
        """Bind a ZMQ PUB socket on *endpoint* and configure the topic prefix.

        Args:
            endpoint: ZMQ bind address (e.g. ``tcp://*:5559``).
            model_name: Model identifier included in the topic string.
            medium: Storage backend type embedded in the topic and each event.
            sndhwm: ZMQ send high-water mark.
        """
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, sndhwm)
        self._socket.bind(endpoint)

        self._model_name = model_name
        self._medium = medium
        self._topic = (
            f"kv@{self._medium.value}@{self._model_name}" if self._model_name else None
        )
        self._seq: int = 0
        self._closed = False
        self._send_lock = threading.Lock()
        logger.info(
            "StorageEventPublisher bound to %s (topic: %s)",
            endpoint,
            self._topic,
        )

    def publish_blocks_stored(self, block_hashes: Iterable[int | bytes]) -> None:
        """Publish a ``BlockStored`` event for each hash in *block_hashes*.

        Each hash is masked to 64 bits and wrapped in a positional-array event
        matching vLLM's GPU ``BlockStored`` wire format.  All events are batched
        into a single ZMQ multipart message.
        """
        hashes = [_hash_to_uint64(h) for h in block_hashes]
        if not hashes:
            return

        event = [
            "BlockStored",  # [0] tag
            hashes,  # [1] block_hashes (all hashes from this complete_store call)
            0,  # [2] parent_hash (unknown at storage tier)
            [],  # [3] token_ids (empty)
            0,  # [4] block_size (unused)
            None,  # [5] lora_id
            self._medium.value,  # [6] medium / device tier
        ]
        self._send_batch([msgpack.packb(event, use_bin_type=True)])

    def publish_blocks_removed(
        self,
        block_hashes: Iterable[int | bytes],
        model_name: str | None = None,
    ) -> None:
        """Publish a ``BlockRemoved`` event for each hash in *block_hashes*.

        Each hash is masked to 64 bits and wrapped in a 3-field positional
        array matching the Go ``convertBlockRemovedEvent()`` wire format.

        Args:
            block_hashes: Block hashes to publish removal events for.
            model_name: Override the model name in the ZMQ topic. Useful
                when the publisher handles multiple models (e.g. PVC evictor).
        """
        hashes = [_hash_to_uint64(h) for h in block_hashes]
        if not hashes:
            return

        event = [
            "BlockRemoved",  # [0] tag
            hashes,  # [1] block_hashes
            self._medium.value,  # [2] medium / device tier
        ]

        topic = f"kv@{self._medium.value}@{model_name}" if model_name else None
        self._send_batch([msgpack.packb(event, use_bin_type=True)], topic=topic)

    def _send_batch(self, packed_events: list[bytes], topic: str | None = None) -> None:
        """Send a batch of pre-packed events as a 3-frame ZMQ message.

        Frames: ``[topic, sequence, payload]``.  Thread-safe; silently drops
        the message if the publisher has been closed or no topic is configured.
        """
        with self._send_lock:
            if self._closed:
                return

            effective_topic = topic or self._topic
            if not effective_topic:
                logger.warning(
                    "Cannot send event: no topic configured "
                    "(model_name not provided to constructor or publish call)"
                )
                return

            payload = msgpack.packb([time.time(), packed_events], use_bin_type=True)
            self._seq += 1
            self._socket.send_multipart(
                [
                    effective_topic.encode("utf-8"),
                    struct.pack(">Q", self._seq),
                    payload,
                ]
            )

    def close(self) -> None:
        """Close the ZMQ socket and terminate the context. Idempotent."""
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
            self._socket.close()
            self._ctx.term()

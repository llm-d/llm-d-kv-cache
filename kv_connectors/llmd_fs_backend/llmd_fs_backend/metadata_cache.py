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

import threading
import time
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash

from llmd_fs_backend import get_logger
from llmd_fs_backend.metrics import METADATA_CACHE_ENTRIES, METADATA_CACHE_EVICTIONS

logger = get_logger("metadata-cache")


class MetadataCache:
    """
    A thread-safe, in-memory positive LRU cache for KV block metadata.

    This cache acts as a fast, local index of entries known to exist. It
    maintains strict LRU ordering, enforces thread-safe single and
    batch primitives, and respects a configurable Time-To-Live (TTL) limit to
    prevent stale positive hits.

    API Primitives:
    - contains(key) / batch_contains(keys): Query presence. Hits move the
      corresponding keys to the end of the queue (most recently used),
      protecting them from upcoming eviction. Expired keys are discarded.
    - insert(key) / batch_insert(keys): Register keys with current timestamp.
      If the cache is at capacity, evicts the oldest key(s) (first in, first out)
      to make room. Existing keys are moved to the end of the queue (LRU priority)
      but retain their original insertion timestamp.
    - remove(key) / batch_remove(keys): Manually discard keys from the index.

    Design rational for a hard TTL:
    To guarantee eventual consistency with the external asynchronous `pvc_evictor`
    (which deletes files out-of-band), this cache enforces a strict Hard TTL.
    Subsequent write operations (insert / batch_insert) on pre-existing keys
    preserve their original insertion timestamp. This bounds the maximum window
    of inconsistency (stale positive hits pointing to evicted files) to a deterministic
    threshold (exactly 1x TTL), forcing hot keys to eventually expire and trigger
    a physical filesystem existence check.

    Thread Safety:
    All read and write operations (both single and batch) are synchronized
    via a reentrant lock (`threading.RLock`), ensuring consistency across
    concurrent scheduler lookup threads.
    """

    def __init__(self, max_entries: int, metadata_cache_ttl_secs: int = 300):
        self.max_entries = max_entries
        self.metadata_cache_ttl_secs = metadata_cache_ttl_secs
        self._lock = threading.RLock()
        self._cache: OrderedDict[BlockHash, float] = OrderedDict()
        logger.info(
            "Initialized metadata cache with max_entries=%d, "
            "metadata_cache_ttl_secs=%ds",
            max_entries,
            metadata_cache_ttl_secs,
        )
        # Initialize Prometheus capacity stats
        METADATA_CACHE_ENTRIES.set(0)

    def contains(self, block_hash: BlockHash) -> bool:
        with self._lock:
            if block_hash in self._cache:
                inserted_time = self._cache[block_hash]
                if (
                    self.metadata_cache_ttl_secs != -1
                    and time.monotonic() - inserted_time > self.metadata_cache_ttl_secs
                ):
                    # Expired entry - prune it
                    self._cache.pop(block_hash)
                    METADATA_CACHE_ENTRIES.set(len(self._cache))
                    METADATA_CACHE_EVICTIONS.labels(type="ttl").inc()
                    logger.debug("Cache entry for block %s expired", block_hash.hex())
                    return False

                self._cache.move_to_end(block_hash)
                logger.debug("Cache hit for block %s", block_hash.hex())
                return True
            logger.debug("Cache miss for block %s", block_hash.hex())
            return False

    def insert(self, block_hash: BlockHash):
        with self._lock:
            now = time.monotonic()
            if block_hash in self._cache:
                self._cache.move_to_end(block_hash)
                logger.debug("Cache hit (update/move) for block %s", block_hash.hex())
                return

            if len(self._cache) >= self.max_entries:
                # Evict the oldest entry (first key in OrderedDict)
                evicted, _ = self._cache.popitem(last=False)
                METADATA_CACHE_EVICTIONS.labels(type="lru").inc()
                logger.debug("Evicted block %s from cache to make room", evicted.hex())

            self._cache[block_hash] = now
            METADATA_CACHE_ENTRIES.set(len(self._cache))
            logger.debug(
                "Inserted block %s into cache (size: %d/%d)",
                block_hash.hex(),
                len(self._cache),
                self.max_entries,
            )

    def remove(self, block_hash: BlockHash):
        with self._lock:
            if self._cache.pop(block_hash, None) is not None:
                METADATA_CACHE_ENTRIES.set(len(self._cache))
                logger.debug("Removed block %s from cache", block_hash.hex())

    # ----------------------------------------------------------------------
    # Batch Operations
    # ----------------------------------------------------------------------

    def batch_contains(self, block_hashes: list[BlockHash]) -> list[bool]:
        """Check presence for a list of hashes in a single lock acquisition."""
        with self._lock:
            results = []
            hits = 0
            ttl_prunes = 0
            now = time.monotonic()
            for h in block_hashes:
                if h in self._cache:
                    inserted_time = self._cache[h]
                    if (
                        self.metadata_cache_ttl_secs != -1
                        and now - inserted_time > self.metadata_cache_ttl_secs
                    ):
                        self._cache.pop(h)
                        ttl_prunes += 1
                        results.append(False)
                    else:
                        self._cache.move_to_end(h)
                        results.append(True)
                        hits += 1
                else:
                    results.append(False)

            if ttl_prunes > 0:
                METADATA_CACHE_ENTRIES.set(len(self._cache))
                METADATA_CACHE_EVICTIONS.labels(type="ttl").inc(ttl_prunes)

            logger.debug(
                "batch_contains: %d hits, %d misses out of %d blocks",
                hits,
                len(results) - hits,
                len(results),
            )
            return results

    def batch_insert(self, block_hashes: Iterable[BlockHash]):
        """Insert multiple hashes in a single lock acquisition."""
        with self._lock:
            new_inserts = 0
            hits = 0
            lru_evictions = 0
            now = time.monotonic()
            for h in block_hashes:
                if h in self._cache:
                    self._cache.move_to_end(h)
                    hits += 1
                else:
                    if len(self._cache) >= self.max_entries:
                        evicted, _ = self._cache.popitem(last=False)
                        lru_evictions += 1
                        logger.debug(
                            "Evicted block %s from cache to make room",
                            evicted.hex(),
                        )
                    self._cache[h] = now
                    new_inserts += 1

            if lru_evictions > 0:
                METADATA_CACHE_EVICTIONS.labels(type="lru").inc(lru_evictions)
            if new_inserts > 0:
                METADATA_CACHE_ENTRIES.set(len(self._cache))

            logger.debug(
                "Batch insert: %d new inserts, %d hits (current size: %d/%d)",
                new_inserts,
                hits,
                len(self._cache),
                self.max_entries,
            )

    def batch_remove(self, block_hashes: Iterable[BlockHash]):
        """Remove multiple hashes in a single lock acquisition."""
        with self._lock:
            count = 0
            for h in block_hashes:
                if self._cache.pop(h, None) is not None:
                    count += 1
            if count > 0:
                METADATA_CACHE_ENTRIES.set(len(self._cache))
            logger.debug("Batch removed %d blocks", count)

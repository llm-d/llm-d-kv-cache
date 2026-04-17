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

import os
import time
from collections.abc import Collection

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    get_offload_block_hash,
)
from zmq import ZMQError

from llmd_fs_backend.event_publisher import StorageMedium
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.metadata_cache import MetadataCache
from llmd_fs_backend.metrics import (
    METADATA_CACHE_LOOKUP_BLOCKS,
    METADATA_CACHE_LOOKUP_DURATION,
)

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.

    Metadata Cache:
    - The metadata cache is OPTIONAL and disabled by default.
    - Enable it by setting env `VLLM_LLMD_FS_METADATA_CACHE_MAX_ENTRIES`
      or passing `metadata_cache_max_entries` in the extra config dict to > 0.

    High-Level Architecture & Metadata Workflow:
    - Positive Caching: The `MetadataCache` acts as a positive index of files
      known to reside on disk. The files on disk are the absolute ground truth.
    - Decoupled State Tracking (No Pinning): File-based storage is safe from
      in-place eviction corruption. We do not require pin/unpin reference
      counting. Presence in cache means "confirmed on disk". Absence means "not
      confirmed in cache" but may still be on disk (verified by FS checks).
    - Cache Eviction Safety & Stateless I/O: The load path (`prepare_load`) is
      stateless and reads files directly from disk without querying the cache.
      Because cache eviction only forgets metadata but never deletes files
      (which is handled by an external `pvc_evictor`), there is no correctness
      impact if a large lookup triggers LRU self-eviction of blocks scheduled
      to be loaded—the physical load will still succeed perfectly.

    Interface Function Workflows:
    - lookup(key, req_context):
        1. Queries `MetadataCache` (via `contains`) for a hit.
        2. On cache miss, falls back to a physical filesystem check (`os.path.exists`).
        3. If found on disk, back-fills the block hash into the cache (via `insert`)
           for subsequent lookups, then returns True.
    - prepare_load(keys, req_context) / complete_load(keys, req_context):
        - Stateless operations. `prepare_load` returns file-pointing load specs.
    - prepare_store(keys, req_context):
        - Stateless operation. Prepares state-free file store specs. It does not
          touch the cache, keeping in-flight writes invisible to concurrent lookups.
    - complete_store(keys, req_context, success):
        - On success, the block hashes for the newly offloaded keys are batch-inserted
          (via `batch_insert`) into the positive cache. On failure, no action is taken.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        extra_config: dict | None = None,
        event_publisher=None,
        metadata_cache_max_entries: int = 0,
        metadata_cache_ttl_secs: int = 300,
    ) -> None:
        self.file_mapper: FileMapper = file_mapper
        self._event_publisher = (
            event_publisher
            if event_publisher is not None
            else self._create_event_publisher(
                self.file_mapper.model_name, extra_config or {}
            )
        )
        self.cache: MetadataCache | None = (
            MetadataCache(
                max_entries=metadata_cache_max_entries,
                metadata_cache_ttl_secs=metadata_cache_ttl_secs,
            )
            if metadata_cache_max_entries > 0
            else None
        )

    @staticmethod
    def _create_event_publisher(model_name: str, extra_config: dict):
        """Create a StorageEventPublisher if events are enabled in *extra_config*."""
        if not extra_config.get("enable_events", False):
            return None

        endpoint = extra_config.get("storage_events_endpoint")
        if not endpoint:
            return None

        kwargs = {}
        if "storage_medium" in extra_config:
            kwargs["medium"] = StorageMedium(extra_config["storage_medium"])
        if "storage_events_hwm" in extra_config:
            kwargs["sndhwm"] = int(extra_config["storage_events_hwm"])

        try:
            from llmd_fs_backend.event_publisher import StorageEventPublisher

            return StorageEventPublisher(
                endpoint=endpoint,
                model_name=model_name,
                **kwargs,
            )
        except ZMQError:
            logger.warning(
                "failed to create storage event publisher for %s",
                endpoint,
                exc_info=True,
            )
            return None

    def _publish_blocks_stored(self, keys: Collection[OffloadKey]) -> None:
        if self._event_publisher is None:
            return
        try:
            block_hashes = [get_offload_block_hash(k) for k in keys]
            self._event_publisher.publish_blocks_stored(block_hashes)
        except Exception:
            logger.warning("failed to publish storage event", exc_info=True)

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a single block is offloaded and ready to be read.
        """
        start_time = time.monotonic()
        block_hash = get_offload_block_hash(key)

        is_hit = False
        memory_hit = False
        fs_hit = False

        if self.cache and self.cache.contains(block_hash):
            is_hit = True
            memory_hit = True

        if not is_hit:
            file_path = self.file_mapper.get_file_name(key)
            if os.path.exists(file_path):
                is_hit = True
                fs_hit = True
                if self.cache:
                    self.cache.insert(block_hash)

        duration = time.monotonic() - start_time

        METADATA_CACHE_LOOKUP_DURATION.observe(duration)

        if memory_hit:
            METADATA_CACHE_LOOKUP_BLOCKS.labels(result="mem_hit").inc()
        elif fs_hit:
            METADATA_CACHE_LOOKUP_BLOCKS.labels(result="fs_hit").inc()
        else:
            METADATA_CACHE_LOOKUP_BLOCKS.labels(result="fs_miss").inc()

        logger.debug(
            "Lookup finished: duration=%.6f [s] hit=%s (memory=%s, filesystem=%s)",
            duration,
            is_hit,
            memory_hit,
            fs_hit,
        )

        return is_hit

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        For shared storage, loading is stateless - return specs that point to files.
        """
        return SharedStorageLoadStoreSpec(keys)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Update access times if desired.
        Shared storage version does nothing here because updates are handled
        by the file thread for performance reasons.
        """
        pass

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """Stateless load - no post-load action needed."""
        pass

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare storing new blocks.
        Shared storage always accepts new blocks. Eviction is not needed.
        If a file already exists, the file thread handles it.
        """
        keys_to_store = list(keys)

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(keys_to_store)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=[],  # no eviction needed
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        For shared storage, storing is stateless but we emit events for stored blocks.
        """
        if success:
            if self.cache:
                block_hashes = [get_offload_block_hash(k) for k in keys]
                self.cache.batch_insert(block_hashes)
            self._publish_blocks_stored(keys)

    def shutdown(self) -> None:
        if self._event_publisher is not None:
            self._event_publisher.close()

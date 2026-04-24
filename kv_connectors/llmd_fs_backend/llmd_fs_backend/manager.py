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

from collections.abc import Iterable

import os

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    PrepareStoreOutput,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.
    """

    LOOKUP_MODE_FILE = "file"
    LOOKUP_MODE_NIXL_QUERY = "nixl_query"
    LOOKUP_MODE_DICT = "dict"

    def __init__(
        self,
        file_mapper: FileMapper,
        lookup_mode: str = "file",
        extra_config: dict | None = None,
    ) -> None:
        cfg = extra_config or {}
        self.file_mapper: FileMapper = file_mapper
        self.lookup_mode = lookup_mode
        self._stored_keys: set[str] = set()

        if lookup_mode == self.LOOKUP_MODE_NIXL_QUERY:
            from llmd_nixl.nixl_lookup import NixlLookup  # lazy: avoids nixl import
            self._nixl_lookup = NixlLookup(cfg)

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        """
        Return how many consecutive blocks from the start are already offloaded.
        """
        hit_count = 0
        for block_hash in block_hashes:
            key = self.file_mapper.get_file_name(block_hash)
            if self.lookup_mode == self.LOOKUP_MODE_DICT:
                # this is good only for local lookup 
                # or for identifying fastest possible lookup latency
                if key not in self._stored_keys:
                    break
            elif self.lookup_mode == self.LOOKUP_MODE_NIXL_QUERY:
                if not self._nixl_lookup.exists(key):
                    break
            elif self.lookup_mode == self.LOOKUP_MODE_FILE:
                if not os.path.exists(key):
                    break
            else:
                raise ValueError(f"Unknown lookup_mode: {self.lookup_mode!r}")
            hit_count += 1
        logger.debug("lookup: %d", hit_count)
        return hit_count

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        For shared storage, loading is stateless - return specs that point to files.
        """
        return SharedStorageLoadStoreSpec(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Update access times if desired.
        Shared storage version does nothing here because updates are handled
        by the file thread for performance reasons.
        """
        pass

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """Stateless load - no post-load action needed."""
        pass

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """
        Prepare storing new blocks.
        Shared storage always accepts new blocks. Eviction is not needed.
        If a file already exists, the file thread handles it.
        """
        block_hashes_to_store = list(block_hashes)

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(block_hashes_to_store)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=[],  # no eviction needed
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """
        For shared storage, storing is stateless - no action needed.
        In dict lookup mode, record successfully stored keys.
        """
        if success and self.lookup_mode == self.LOOKUP_MODE_DICT:
            for block_hash in block_hashes:
                self._stored_keys.add(self.file_mapper.get_file_name(block_hash))

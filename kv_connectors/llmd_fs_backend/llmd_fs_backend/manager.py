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
from collections.abc import Iterable

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

    def __init__(self, file_mapper: FileMapper | tuple[FileMapper, ...]) -> None:
        if isinstance(file_mapper, tuple):
            self.file_mappers = file_mapper
        else:
            self.file_mappers = (file_mapper,)
        self._pending_store: set[BlockHash] = set()

    def _block_exists(self, block_hash: BlockHash) -> bool:
        return all(
            os.path.exists(file_mapper.get_file_name(block_hash))
            for file_mapper in self.file_mappers
        )

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        """
        Return how many consecutive blocks from the start are already offloaded.
        """
        hit_count = 0
        for block_hash in block_hashes:
            if block_hash in self._pending_store:
                break
            if not self._block_exists(block_hash):
                break
            hit_count += 1
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
        block_hashes_to_store = [
            block_hash
            for block_hash in block_hashes
            if block_hash not in self._pending_store
            and not self._block_exists(block_hash)
        ]
        self._pending_store.update(block_hashes_to_store)

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
        """
        for block_hash in block_hashes:
            self._pending_store.discard(block_hash)

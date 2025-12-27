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
from pathlib import Path
from typing import Optional

import torch
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import StorageOffloadingHandlers
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (LoadStoreSpec, OffloadingManager,
                                         PrepareStoreOutput)

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.
    """

    def __init__(
        self,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        root_dir: str,
    ) -> None:

        # Basic metadata about the model and tensor parallelism
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype

        # Resolve base directory where KV files for this model and tp rank are stored
        self.base_path: Path = StorageOffloadingHandlers.get_kv_cache_base_path(
            dtype=dtype,
            model_name=model_name,
            tp_size=tp_size,
            tp_rank=tp_rank,
            root_dir=root_dir,
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
            file_path = StorageOffloadingHandlers.get_file_name(
                self.base_path, block_hash)
            if not os.path.exists(file_path):
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
            self,
            block_hashes: Iterable[BlockHash]) -> Optional[PrepareStoreOutput]:
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

    def complete_store(self,
                       block_hashes: Iterable[BlockHash],
                       success: bool = True):
        """
        For shared storage, storing is stateless - no action needed.
        """
        pass

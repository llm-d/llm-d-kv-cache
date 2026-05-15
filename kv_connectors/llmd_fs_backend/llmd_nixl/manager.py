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

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_nixl.nixl_lookup import NixlLookup

logger = init_logger(__name__)

LOOKUP_MODE_DICT = "dict"
LOOKUP_MODE_NIXL_QUERY = "nixl_query"


class NixlStorageOffloadingManager(SharedStorageOffloadingManager):
    """
    Extends SharedStorageOffloadingManager with dict and nixl_query lookup modes.

    Use this manager when the backend is OBJ or when an in-memory dict lookup
    is preferred over filesystem stat calls.
    """

    def __init__(
        self, file_mapper: FileMapper, extra_config: dict | None = None
    ) -> None:
        super().__init__(file_mapper)
        cfg = extra_config or {}
        self.lookup_mode = cfg.get("lookup_mode", LOOKUP_MODE_NIXL_QUERY)

        self._stored_keys: set[str] = set()
        self._nixl_lookup = None

        if self.lookup_mode == LOOKUP_MODE_NIXL_QUERY:
            self._nixl_lookup = NixlLookup(cfg)

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            key = self.file_mapper.get_file_name(block_hash)
            if self.lookup_mode == LOOKUP_MODE_NIXL_QUERY:
                if not self._nixl_lookup.exists(key):
                    break
            else:  # dict
                if key not in self._stored_keys:
                    break
            hit_count += 1
        logger.debug("lookup(%s): %d", self.lookup_mode, hit_count)
        return hit_count

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        if not success or self.lookup_mode != LOOKUP_MODE_DICT:
            return
        for block_hash in block_hashes:
            self._stored_keys.add(self.file_mapper.get_file_name(block_hash))

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

from collections.abc import Collection

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

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

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        file_name = self.file_mapper.get_file_name(key)
        if self.lookup_mode == LOOKUP_MODE_NIXL_QUERY:
            exists = self._nixl_lookup.exists(file_name)
        else:  # dict
            exists = file_name in self._stored_keys
        logger.debug("lookup(%s): %s", self.lookup_mode, exists)
        return exists

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        if not success or self.lookup_mode != LOOKUP_MODE_DICT:
            return
        for key in keys:
            self._stored_keys.add(self.file_mapper.get_file_name(key))

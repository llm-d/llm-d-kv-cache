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

from typing import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec


class SharedStorageLoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading and storing KV blocks on shared storage.
    Stores block hashes internally as a numpy array.
    """

    def __init__(self, block_hashes: Iterable[BlockHash]):
        self.block_hashes = list(block_hashes)

    def __repr__(self) -> str:
        return repr(self.block_hashes)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"

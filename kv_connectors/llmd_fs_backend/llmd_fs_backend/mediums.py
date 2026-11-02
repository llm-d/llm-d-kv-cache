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

from vllm.v1.kv_offload.base import LoadStoreSpec, OffloadKey


class SharedStorageLoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading and storing KV blocks on shared storage.
    Stores offload keys internally as a list. ``source_paths`` is an optional
    internal worker-path mapping for either transfer direction: loads can point
    to pinned FDs; stores can target a server-selected TTL namespace. It must
    never be populated from arbitrary client-supplied paths.
    """

    def __init__(
        self, keys: Iterable[OffloadKey], source_paths: dict[str, str] | None = None
    ):
        self.keys = list(keys)
        self.source_paths = source_paths

    def __repr__(self) -> str:
        return repr(self.keys)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"

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

"""In-memory set-backed block existence lookup."""


class DictLookup:
    """Tracks offloaded KV-cache block keys in an in-process set.

    Keys are the same file path / obj key strings produced by FileMapper.
    Presence of a key means the block has been successfully offloaded.
    State is not persisted across process restarts.
    """

    def __init__(self) -> None:
        self._keys: set[str] = set()

    def exists(self, key: str) -> bool:
        """Return True if the block identified by key has been recorded."""
        return key in self._keys

    def add(self, key: str) -> None:
        """Mark a block key as offloaded."""
        self._keys.add(key)

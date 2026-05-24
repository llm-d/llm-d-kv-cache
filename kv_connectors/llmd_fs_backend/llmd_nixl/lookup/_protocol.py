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

"""Shared protocol for block-existence lookup backends."""

from collections.abc import Iterable
from typing import Protocol


class LookupBackend(Protocol):
    """Structural interface shared by all lookup backends.

    Implementations: NixlLookup (S3 query_memory), RedisLookup, DictLookup.
    """

    def exists(self, key: str) -> bool:
        """Return True if the block identified by key has been offloaded."""
        ...

    def lookup(self, keys: Iterable[str]) -> int:
        """Return consecutive hit count from the start of keys."""
        ...

    def add(self, key: str) -> None:
        """Mark a block key as offloaded."""
        ...

    def add_all(self, keys: Iterable[str]) -> None:
        """Mark multiple block keys as offloaded."""
        ...

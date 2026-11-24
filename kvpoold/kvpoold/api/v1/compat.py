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

"""What an engine states about its KV data when it attaches.

Two things decide whether an engine can use the pool and whose chunks it reads:

- The layout: how many bytes one chunk takes across every rank of the engine.
  The first accepted attach fixes the region's row stride to it, and every later
  engine must match it.
- The namespace: which engines' keys are interchangeable. Engines in the same
  namespace read each other's chunks; the pool keeps other namespaces' chunks
  apart by prefixing every key with its namespace id, while they share eviction.

The pool server also requires one block hash function and seed across the pool.
"""

import msgspec

COMPAT_ID_BYTES = 8
"""The length of a ``NamespaceId``."""

LayoutId = int
"""kv_bytes_per_chunk: the region's row stride, fixed by the first accepted attach."""

NamespaceId = bytes
"""``COMPAT_ID_BYTES`` bytes naming whose keys are interchangeable. Prefixes every pool key."""


class CompatDescriptor(msgspec.Struct, kw_only=True, frozen=True, array_like=True):
    layout: LayoutId
    namespace: NamespaceId
    hash_function: str
    """The engine's prefix-caching hash algorithm, e.g. ``sha256``."""
    hash_seed: str | None
    """The engine's PYTHONHASHSEED, or None when it is not set."""

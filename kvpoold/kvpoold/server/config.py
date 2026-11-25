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

"""``ServerConfig``: everything a pool server is started with."""

from dataclasses import dataclass

from vllm.config.kv_events import KVEventsConfig


@dataclass(frozen=True, kw_only=True)
class ServerConfig:
    address: str
    """ZMQ endpoint the server binds, e.g. ``ipc:///host-shm/kvpoold.sock``."""
    region_dir: str
    """Directory for the region's files and the spec and snapshot beside them, e.g. ``/host-shm``."""
    pool_bytes: int
    """Size of the shared CPU pool."""
    kv_events_config: KVEventsConfig
    """The publisher of the pool's KV events. It must enable KV cache events: the
    server always publishes them."""
    cache_policy: str = "lru"
    """vLLM's cache policy for the pool, by name."""
    cache_policy_module_path: str | None = None
    """Module to load ``cache_policy`` from when it is not one of vLLM's built-in policies."""
    lease_seconds: float = 30.0
    """How long a pin or write lasts before it is released; far longer than the longest copy."""
    fence_seconds: float = 10.0
    """After a restart, how long the previous instance's leases stay, or, after a
    crash, how long no chunk is handed out."""

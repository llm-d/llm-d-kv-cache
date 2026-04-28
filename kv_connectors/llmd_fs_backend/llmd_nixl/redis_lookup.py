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

"""Redis-backed block existence lookup."""

from collections.abc import Iterable

import redis


class RedisLookup:
    """Tracks offloaded KV-cache block keys in Redis.

    Keys are the same file path / S3 key strings produced by FileMapper.
    Presence of a key in Redis means the block has been successfully offloaded.

    Configuration keys in extra_config:
        redis_url:  Redis connection URL (default: "redis://localhost:6379/0")
        redis_ttl:  Key TTL in seconds; 0 = no expiry (default: 0)
    """

    def __init__(self, extra_config: dict):
        cfg = extra_config or {}
        redis_url = cfg.get("redis_url")
        if not redis_url:
            raise ValueError(
                "RedisLookup requires 'redis_url' in extra_config "
                "(e.g. 'redis://localhost:6379/0')"
            )
        self._ttl = int(cfg.get("redis_ttl", 0))
        self._client = redis.from_url(redis_url)
        self._client.ping()

    def exists(self, key: str) -> bool:
        """Return True if the block identified by key has been recorded."""
        return bool(self._client.exists(key))

    def lookup(self, keys: Iterable[str]) -> int:
        """Return consecutive hit count from the start of keys.

        Checks the first key individually; on hit, batches the rest in
        a single pipelined round-trip.
        """
        it = iter(keys)
        try:
            first_key = next(it)
        except StopIteration:
            return 0
        if not self._client.exists(first_key):
            return 0
        rest = list(it)
        if not rest:
            return 1
        with self._client.pipeline(transaction=False) as pipe:
            for key in rest:
                pipe.exists(key)
            results = pipe.execute()
        for i, hit in enumerate(results, start=1):
            if not hit:
                return i
        return 1 + len(rest)

    def add(self, key: str) -> None:
        """Mark a block key as offloaded."""
        if self._ttl > 0:
            self._client.set(key, 1, ex=self._ttl)
        else:
            self._client.set(key, 1)

    def add_all(self, keys: Iterable[str]) -> None:
        """Mark multiple block keys as offloaded in a single round-trip."""
        with self._client.pipeline(transaction=False) as pipe:
            for key in keys:
                if self._ttl > 0:
                    pipe.set(key, 1, ex=self._ttl)
                else:
                    pipe.set(key, 1)
            pipe.execute()

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

"""Block existence lookup via NIXL query_memory."""

from collections.abc import Iterable

from nixl._api import nixl_agent, nixl_agent_config

from llmd_nixl.obj_backend import obj_key_to_dev_id


class NixlLookup:
    """Checks whether blocks exist using the NIXL query_memory interface.

    Works with any NIXL backend that supports query_memory (e.g. OBJ/S3).
    """

    def __init__(self, extra_config: dict):
        cfg = extra_config or {}
        if not cfg.get("bucket"):
            raise ValueError("NixlLookup requires 'bucket' in extra_config")
        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("NixlLookup", agent_config)
        backend_params = {
            "bucket": cfg.get("bucket", ""),
            "endpoint_override": cfg.get("endpoint_override", ""),
            "scheme": cfg.get("scheme", "http"),
            "access_key": cfg.get("access_key", ""),
            "secret_key": cfg.get("secret_key", ""),
        }
        if cfg.get("ca_bundle"):
            backend_params["ca_bundle"] = cfg["ca_bundle"]
        self._agent.create_backend("OBJ", backend_params)

    def exists(self, key: str) -> bool:
        """Return True if the S3 object identified by key exists."""
        results = self._agent.query_memory(
            [(0, 1, obj_key_to_dev_id(key), key)], "OBJ", "OBJ"
        )
        return results[0] is not None

    def lookup(self, keys: Iterable[str]) -> int:
        """Return consecutive hit count from the start of keys.

        Checks the first key individually; on hit, batches the rest in
        a single query_memory call.
        """
        it = iter(keys)
        try:
            first_key = next(it)
        except StopIteration:
            return 0
        first = self._agent.query_memory(
            [(0, 1, obj_key_to_dev_id(first_key), first_key)], "OBJ", "OBJ"
        )
        if first[0] is None:
            return 0
        rest = list(it)
        if not rest:
            return 1
        descs = [(0, 1, obj_key_to_dev_id(k), k) for k in rest]
        results = self._agent.query_memory(descs, "OBJ", "OBJ")
        for i, result in enumerate(results, start=1):
            if result is None:
                return i
        return 1 + len(rest)

    def add(self, _key: str) -> None:
        """No-op: S3 object existence is ground truth; no index needed."""

    def add_all(self, _keys: Iterable[str]) -> None:
        """No-op: S3 object existence is ground truth; no index needed."""

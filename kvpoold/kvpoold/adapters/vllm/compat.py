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

"""A vLLM engine's ``CompatDescriptor``, from its offloading spec."""

import hashlib
import json
import os

from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.file_mapper import FileMapper

from kvpoold.api.v1 import COMPAT_ID_BYTES, CompatDescriptor, NamespaceId

DEFAULT_HASH_FUNCTION = "sha256"
"""vLLM's default ``--prefix-caching-hash-algo``."""


def compat_descriptor(spec: CPUOffloadingSpec) -> CompatDescriptor:
    """The layout is the spec's ``kv_bytes_per_chunk``, and the namespace hashes
    its run config (see ``namespace_id``).

    vLLM does not pass its prefix-caching hash algorithm to offloading specs, so
    the engine states it in the ``hash_function`` extra-config key, which must
    match ``--prefix-caching-hash-algo``. The seed is the process's
    PYTHONHASHSEED.
    """
    return CompatDescriptor(
        layout=spec.kv_bytes_per_chunk,
        namespace=namespace_id(spec),
        hash_function=spec.extra_config.get("hash_function", DEFAULT_HASH_FUNCTION),
        hash_seed=os.getenv("PYTHONHASHSEED"),
    )


def namespace_id(spec: CPUOffloadingSpec) -> NamespaceId:
    """The first ``COMPAT_ID_BYTES`` bytes of SHA-256 over ``FileMapper``'s run config.

    The run config holds the fields vLLM's own secondary tiers compare to decide
    that two engines' blocks are interchangeable. Engines with the same model and
    layout share chunks; other engines never read them.
    """
    run_config = FileMapper.from_offloading_spec(
        root_dir="",  # only the run config is used
        offloading_spec=spec,
        # Leaves out the parallel sizes when the layout allows it, so engines
        # with different TP share chunks.
        parallel_agnostic=True,
    ).get_run_config()
    serialized = json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).digest()[:COMPAT_ID_BYTES]

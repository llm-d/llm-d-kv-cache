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

"""``ExternalOffloadingSpec``: vLLM's CPU offloading on the node's shared pool.

vLLM builds one instance in the scheduler process and one in each GPU worker.
The External Primary Tier owns the pool:

- ``create_worker()``, in each GPU worker, attaches to the tier, maps the
  region it describes, and builds vLLM's stock ``CPUOffloadingWorker`` on it.
- ``get_manager()``, in the scheduler, returns ``ExternalPrimaryTierClient``,
  which attaches through ``attach()`` and sends every manager call to the tier.

A worker that the tier refuses, or that cannot reach the tier or map its region,
serves without the pool.
"""

import hashlib
import json
import os
from dataclasses import replace

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    DevicePointers,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import REGION_SEGMENTS, SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from external_primary_tier.client.session import Session
from external_primary_tier.protocol import PROTOCOL_VERSION, Attach, AttachOk, Refused, ShmFileRegion

logger = init_logger(__name__)

DEFAULT_ATTACH_TIMEOUT_MS = 300_000
"""The first attach waits while the tier creates and pre-faults the whole pool."""

DEFAULT_HASH_FUNCTION = "sha256"
"""vLLM's default ``--prefix-caching-hash-algo``."""


class ExternalOffloadingSpec(CPUOffloadingSpec):
    # CPUOffloadingSpec does not validate the canonical layout, and
    # TieringOffloadingSpec's check reads only self.config, so it runs as is on
    # this spec. It records self.all_layers_portable.
    _validate_canonical_refs = TieringOffloadingSpec._validate_canonical_refs

    def __init__(self, config: OffloadingConfig) -> None:
        # CPUOffloadingSpec raises unless cpu_bytes_to_use is set, and derives only
        # num_chunks from it. The tier sets this pool's chunk count, so a one-byte
        # placeholder passes the check, and the num_chunks it yields (0) is unused:
        # the chunk count comes from the tier's attach reply.
        super().__init__(replace(config, extra_config={**config.extra_config, "cpu_bytes_to_use": 1}))
        self.tier_address: str = self.extra_config["master_address"]
        self._attach_timeout_ms = int(self.extra_config.get("attach_timeout_ms", DEFAULT_ATTACH_TIMEOUT_MS))
        # vLLM does not pass its prefix-caching hash algorithm to offloading
        # specs, so the engine states it. It must match --prefix-caching-hash-algo.
        self._hash_function: str = self.extra_config.get("hash_function", DEFAULT_HASH_FUNCTION)
        self._compat_id = _compat_id(self)

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            # Imported here: only the scheduler process builds a manager.
            from external_primary_tier.client.manager import ExternalPrimaryTierClient

            self._manager = ExternalPrimaryTierClient(self)
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self.config.canonical_layout:
            self._validate_canonical_refs(kv_caches)

        # The ExternalTier needs no connection after the attach: it gives every pin and
        # write a deadline instead of watching engines.
        session = Session(self.tier_address)
        try:
            descriptor = self.attach(session, rank=self.config.parallel.rank)
        finally:
            session.close()
        if descriptor is None:
            return _DisabledWorker()

        region = self._map(descriptor)
        if region is None:
            return _DisabledWorker()
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=descriptor.num_chunks,
                mmap_region=region,
                canonical_layout=self.config.canonical_layout,
            )
        except Exception:
            region.cleanup()
            raise

    def attach(self, session: Session, rank: int | None) -> ShmFileRegion | None:
        """Attach this process to the ExternalPrimaryTier: a GPU worker by its rank, the
        scheduler with no rank. Return the pool's region, or None when the tier
        refuses or does not answer."""
        reply = session.call(
            Attach,
            timeout_ms=self._attach_timeout_ms,
            version=PROTOCOL_VERSION,
            client_id=self.config.engine_id,
            rank=rank,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            hash_function=self._hash_function,
            hash_seed=os.getenv("PYTHONHASHSEED"),
            compat_id=self._compat_id,
        )
        if isinstance(reply, AttachOk):
            return reply.region
        if reply is None:
            _warn_without_pool(f"no answer from {self.tier_address} within {self._attach_timeout_ms} ms")
        elif isinstance(reply, Refused):
            _warn_without_pool(f"the External Primary Tier refused the attach: {reply.reason}")
        else:
            _warn_without_pool(f"the External Primary Tier answered the attach with {type(reply).__name__}")
        return None

    def _map(self, descriptor: ShmFileRegion) -> SharedOffloadRegion | None:
        """Map the region's files, or return None when they are not the region
        this vLLM would map."""
        # SharedOffloadRegion checks only the first file's size, and a copy into
        # a page past the end of a file kills the process with SIGBUS.
        for path, _, length in descriptor.segments:
            try:
                size = os.stat(path).st_size
            except FileNotFoundError:
                _warn_without_pool(f"the region's file {path} is missing")
                return None
            if size != length:
                _warn_without_pool(f"the region's file {path} has {size} bytes, not {length}")
                return None

        try:
            region = SharedOffloadRegion(
                engine_id="",  # unused: mmap_path names the files
                num_chunks=descriptor.num_chunks,
                rank=0 if self.replicated_layout else self.config.parallel.rank,
                kv_bytes_per_chunk=descriptor.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                num_segments=REGION_SEGMENTS,
                mmap_path=descriptor.segments[0][0],
                populate=False,  # the tier has pre-faulted every page
            )
        except Exception as e:
            _warn_without_pool(f"the region could not be mapped: {e}")
            return None

        # SharedOffloadRegion has no public accessors for these two.
        if region._creator:
            # The files vanished after the size check, so this process created
            # new, empty ones; cleanup() removes them.
            region.cleanup()
            _warn_without_pool("the region's files disappeared while they were being mapped")
            return None
        if region._segments != descriptor.segments:
            region.cleanup()
            _warn_without_pool("this vLLM splits the region into other files than the External Primary Tier did")
            return None
        return region


class _DisabledWorker(OffloadingWorker):
    """The worker of a process that serves without the pool.

    vLLM asserts that every submission is accepted, so each job is accepted and
    then reported failed, and the engine recomputes instead of copying.
    """

    def __init__(self) -> None:
        self._submitted: list[int] = []

    def submit_store(self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec) -> bool:
        self._submitted.append(job_id)
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, device_ptrs: DevicePointers, expedited: bool = False
    ) -> bool:
        self._submitted.append(job_id)
        return True

    def get_finished(self) -> list[TransferResult]:
        submitted, self._submitted = self._submitted, []
        return [TransferResult(job_id=job_id, success=False) for job_id in submitted]

    def wait(self, job_ids: set[int]) -> None:
        # Nothing is ever in flight. As with vLLM's own worker, the jobs waited
        # for are still reported by get_finished().
        pass


def _compat_id(spec: CPUOffloadingSpec) -> bytes:
    """The 8 bytes the tier puts in front of this engine's keys.

    They hash ``FileMapper``'s run config: the fields vLLM's own secondary tiers
    compare to decide that two engines' blocks are interchangeable. Engines with
    the same model and layout share chunks; other engines never read them.
    """
    run_config = FileMapper.from_offloading_spec(
        root_dir="",  # only the run config is used
        offloading_spec=spec,
        # Leaves out the parallel sizes when the layout allows it, so engines
        # with different TP share chunks.
        parallel_agnostic=True,
    ).get_run_config()
    serialized = json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).digest()[:8]


def _warn_without_pool(reason: str) -> None:
    logger.warning("Serving without the shared CPU pool: %s", reason)

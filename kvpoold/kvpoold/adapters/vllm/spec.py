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

"""``PoolOffloadingSpec``: vLLM's CPU offloading on the node's shared pool.

vLLM builds one instance in the scheduler process and one in each GPU worker.
The pool server owns the pool:

- ``create_worker()``, in each GPU worker, attaches to the server, maps the
  region it describes, and builds vLLM's stock ``CPUOffloadingWorker`` on it.
- ``get_manager()``, in the scheduler, returns ``PoolOffloadingManager``,
  which attaches and sends every manager call to the server.

Both talk to the server through a ``Client`` from ``make_client()``. A worker
that the server refuses, or that cannot reach the server or map its region,
serves without the pool.
"""

import logging
import os
from dataclasses import replace

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
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from kvpoold.adapters.vllm.compat import compat_descriptor
from kvpoold.api.v1 import ShmFileRegion
from kvpoold.client import Client
from kvpoold.client.client import DEFAULT_ATTACH_TIMEOUT_MS, DEFAULT_CALL_TIMEOUT_MS

logger = logging.getLogger(__name__)


class PoolOffloadingSpec(CPUOffloadingSpec):
    # CPUOffloadingSpec does not validate the canonical layout, and
    # TieringOffloadingSpec's check reads only self.config, so it runs as is on
    # this spec. It records self.all_layers_portable.
    _validate_canonical_refs = TieringOffloadingSpec._validate_canonical_refs

    def __init__(self, config: OffloadingConfig) -> None:
        # CPUOffloadingSpec raises unless cpu_bytes_to_use is set, and derives only
        # num_chunks from it. The server sets this pool's chunk count, so a one-byte
        # placeholder passes the check, and the num_chunks it yields (0) is unused:
        # the chunk count comes from the server's attach reply.
        super().__init__(replace(config, extra_config={**config.extra_config, "cpu_bytes_to_use": 1}))
        self.pool_address: str = self.extra_config["pool_address"]
        self._attach_timeout_ms = int(self.extra_config.get("attach_timeout_ms", DEFAULT_ATTACH_TIMEOUT_MS))
        self._call_timeout_ms = int(self.extra_config.get("call_timeout_ms", DEFAULT_CALL_TIMEOUT_MS))
        self.compat = compat_descriptor(self)

    def make_client(self) -> Client:
        """A client of the pool server for this engine, not yet attached."""
        return Client(
            self.pool_address,
            self.config.engine_id,
            self.compat,
            attach_timeout_ms=self._attach_timeout_ms,
            call_timeout_ms=self._call_timeout_ms,
        )

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            # Imported here: only the scheduler process builds a manager.
            from kvpoold.adapters.vllm.manager import PoolOffloadingManager

            self._manager = PoolOffloadingManager(self)
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self.config.canonical_layout:
            self._validate_canonical_refs(kv_caches)

        client = self.make_client()
        try:
            descriptor = client.attach(rank=self.config.parallel.rank)
        finally:
            client.close()
        if descriptor is None:
            return DisabledWorker()

        region = self._map(descriptor)
        if region is None:
            return DisabledWorker()
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
                populate=False,  # the server has pre-faulted every page
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
            _warn_without_pool("this vLLM splits the region into other files than the pool server did")
            return None
        return region


class DisabledWorker(OffloadingWorker):
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


def _warn_without_pool(reason: str) -> None:
    logger.warning("Serving without the shared CPU pool: %s", reason)

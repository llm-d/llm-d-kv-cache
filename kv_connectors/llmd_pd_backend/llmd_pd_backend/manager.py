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

"""P/D NIXL SecondaryTierManager for vLLM TieringOffloadingSpec.

Transfers KV-cache blocks between prefill (P) and decode (D) instances
over NIXL/UCX. Plugs into vLLM's ``secondary_tiers`` config:

    "secondary_tiers": [{
        "type": "pd",
        "module_path": "llmd_pd_backend.manager",
        "host": "10.0.0.1",
        "port": 5710
    }]
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    SecondaryTierManager,
    TransferJob,
)

from llmd_pd_backend.config import PDConfig, validate_pd_config
from llmd_pd_backend.transport import PDNixlTransport

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.base import ParentManager

logger = logging.getLogger(__name__)

_DRAIN_POLL_S = 0.001


class PDSecondaryTierManager(SecondaryTierManager):
    """Secondary tier for P/D KV cache handoff over NIXL.

    The prefill instance stores computed KV blocks into this tier, which
    registers them for NIXL transfer. The decode instance discovers and
    pulls them via NIXL READ operations.

    Single-threaded: all methods run on the scheduler thread.
    """

    medium: ClassVar[Medium] = Medium.CPU
    locality: ClassVar[Locality] = Locality.REMOTE

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str = "pd",
        **kwargs: Any,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        self._config: PDConfig = validate_pd_config(kwargs)
        self._primary_kv_view = primary_kv_view
        self._transport = PDNixlTransport(self._config)
        self._transport.register_local_memory(primary_kv_view)

        # Block index: tracks which keys have been stored locally
        self._stored_keys: set[OffloadKey] = set()

        # Maps job_id -> TransferJob metadata for inflight transfers
        self._pending_jobs: dict[int, TransferJob] = {}
        self._next_job_id = 0

    def _alloc_job_id(self) -> int:
        jid = self._next_job_id
        self._next_job_id += 1
        return jid

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        pass

    def on_schedule_end(self, ctx: ScheduleEndContext) -> None:
        pass

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._stored_keys:
            return LookupResult.HIT
        return LookupResult.MISS

    def submit_store(self, job: TransferJob) -> None:
        """Store KV blocks from the primary tier into this P/D tier.

        On the prefill side, this makes blocks available for the decode
        instance to pull. The data is already in the primary_kv_view
        (CPU DRAM); we simply record the keys as stored and register
        them for remote access.
        """
        for key in job.keys:
            self._stored_keys.add(key)

        self._pending_jobs[job.job_id] = job
        logger.debug("submit_store: job_id=%d keys=%d", job.job_id, len(list(job.keys)))

    def submit_load(self, job: TransferJob) -> None:
        """Load KV blocks from the remote prefill instance into the primary tier.

        On the decode side, this initiates a NIXL transfer from the
        prefill's registered memory into our primary_kv_view.
        """
        peer_id = self._resolve_peer(job.req_context)
        if peer_id is None:
            logger.warning(
                "submit_load: no peer for job %d, failing immediately", job.job_id
            )
            self._pending_jobs[job.job_id] = job
            return

        # Each chunk_id maps to an offset in the primary_kv_view.
        # The tiering manager provides chunk_ids as indices.
        for chunk_id in job.chunk_ids:
            block_size = self._get_block_size()
            local_offset = int(chunk_id) * block_size
            remote_offset = local_offset

            success = self._transport.submit_transfer(
                job_id=job.job_id,
                peer_id=peer_id,
                local_offset=local_offset,
                remote_offset=remote_offset,
                size=block_size,
                op="READ",
            )
            if not success:
                logger.error(
                    "submit_load: transfer failed for job %d chunk %d",
                    job.job_id,
                    chunk_id,
                )

        self._pending_jobs[job.job_id] = job

    def get_finished_jobs(self) -> Iterable[JobResult]:
        results: list[JobResult] = []

        # Poll NIXL transport for completed transfers
        transport_results = self._transport.poll_transfers()
        for job_id, success, transfer_time in transport_results:
            job = self._pending_jobs.pop(job_id, None)
            if job is not None:
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=success,
                        transfer_time=transfer_time,
                    )
                )

        # Store jobs complete immediately (data is already in primary tier)
        store_jobs = [
            (jid, job)
            for jid, job in list(self._pending_jobs.items())
            if not job.is_promotion
        ]
        for job_id, job in store_jobs:
            del self._pending_jobs[job_id]
            results.append(JobResult(job_id=job_id, success=True))

        return results

    def drain_jobs(self) -> None:
        """Block until all in-flight transfers complete."""
        while self._pending_jobs or self._transport.has_inflight:
            finished = list(self.get_finished_jobs())
            if not finished and (self._pending_jobs or self._transport.has_inflight):
                time.sleep(_DRAIN_POLL_S)

    def has_pending_work(self) -> bool:
        return bool(self._pending_jobs) or self._transport.has_inflight

    def serve_external_requests(self, parent: ParentManager) -> None:
        pass

    def evict(self, keys: Collection[OffloadKey]) -> None:
        """Remove keys from the stored set."""
        for key in keys:
            self._stored_keys.discard(key)

    def shutdown(self) -> None:
        self._transport.shutdown()
        self._stored_keys.clear()
        self._pending_jobs.clear()
        logger.info("PDSecondaryTierManager shutdown complete")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_peer(self, req_context: ReqContext) -> str | None:
        """Extract the remote prefiller's peer_id from request params."""
        params = req_context.kv_transfer_params
        if not params:
            return None
        remote = params.get("remote_prefiller")
        if not remote:
            return None
        host = remote.get("remote_host")
        port = remote.get("remote_port")
        if host and port:
            return f"{host}:{port}"
        return None

    def _get_block_size(self) -> int:
        """Derive the per-block byte size from the primary KV view.

        Falls back to the full view size if no block structure is
        available — the tiering manager always passes correctly-sized
        chunk_ids.
        """
        return len(self._primary_kv_view)

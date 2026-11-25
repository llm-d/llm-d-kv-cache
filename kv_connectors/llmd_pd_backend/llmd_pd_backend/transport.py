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

"""NIXL transport wrapper for P/D KV cache transfers over TCP/RDMA.

Encapsulates NIXL agent lifecycle, memory registration, and transfer
execution. Reuses patterns from llmd_nixl.nixl_offload but adapted for
the SecondaryTierManager memoryview-based interface (CPU↔CPU, no GPU).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from nixl._api import nixl_agent, nixl_agent_config

from llmd_pd_backend.config import PDConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.001


class NixlTransferHandle:
    """Tracks an in-flight NIXL transfer."""

    __slots__ = ("job_id", "xfer_handle", "remote_desc", "started_at")

    def __init__(
        self,
        job_id: int,
        xfer_handle: Any,
        remote_desc: Any,
        started_at: float,
    ):
        self.job_id = job_id
        self.xfer_handle = xfer_handle
        self.remote_desc = remote_desc
        self.started_at = started_at


class PDNixlTransport:
    """Manages NIXL agent and memory for P/D block transfers.

    The transport registers the primary tier's CPU memory region and
    provides async transfer primitives that operate on byte offsets
    within that region.
    """

    def __init__(self, config: PDConfig, agent_name: str = "PDTransport"):
        self._config = config
        agent_cfg = nixl_agent_config(backends=[])
        self._agent = nixl_agent(agent_name, agent_cfg)

        # UCX backend for TCP/RDMA transfers
        self._agent.create_backend("UCX", {"num_threads": "4"})

        self._local_desc: Any | None = None
        self._remote_agents: dict[str, Any] = {}
        self._inflight: dict[int, NixlTransferHandle] = {}

    @property
    def agent(self) -> nixl_agent:
        return self._agent

    def register_local_memory(self, buf: memoryview | bytearray) -> None:
        """Register the primary tier CPU buffer with NIXL."""
        if self._local_desc is not None:
            return
        desc_list = [(buf, len(buf), 0)]
        self._local_desc = self._agent.register_memory(desc_list, "DRAM")
        logger.info("Registered local memory: %d bytes", len(buf))

    def connect_to_peer(self, peer_id: str) -> None:
        """Establish a NIXL connection to a remote peer.

        Uses bounded retry with exponential backoff.
        """
        if peer_id in self._remote_agents:
            return

        host, port_str = peer_id.rsplit(":", 1)
        port = int(port_str)

        for attempt in range(1, self._config.max_retries + 1):
            try:
                remote_meta = self._agent.get_remote_metadata(host, port)
                self._remote_agents[peer_id] = remote_meta
                logger.info("Connected to peer %s on attempt %d", peer_id, attempt)
                return
            except Exception as err:
                if attempt == self._config.max_retries:
                    raise ConnectionError(
                        f"Failed to connect to {peer_id} after "
                        f"{self._config.max_retries} attempts"
                    ) from err
                backoff = self._config.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Connection attempt %d/%d to %s failed, retrying in %.1fs",
                    attempt,
                    self._config.max_retries,
                    peer_id,
                    backoff,
                )
                time.sleep(backoff)

    def submit_transfer(
        self,
        job_id: int,
        peer_id: str,
        local_offset: int,
        remote_offset: int,
        size: int,
        op: str,
    ) -> bool:
        """Submit an async NIXL transfer (READ or WRITE).

        Args:
            job_id: Caller-assigned transfer ID.
            peer_id: "host:port" of the remote NIXL agent.
            local_offset: Byte offset into the registered local buffer.
            remote_offset: Byte offset into the remote buffer.
            size: Number of bytes to transfer.
            op: "READ" (pull from remote) or "WRITE" (push to remote).

        Returns:
            True if the transfer was successfully submitted.
        """
        if peer_id not in self._remote_agents:
            self.connect_to_peer(peer_id)

        local_descs = self._agent.get_xfer_descs(
            [(self._local_desc, local_offset, size, 0)], "DRAM"
        )
        if not local_descs:
            logger.error("Failed to create local xfer descs for job %d", job_id)
            return False

        remote_meta = self._remote_agents[peer_id]
        remote_descs = self._agent.get_xfer_descs(
            [(remote_meta, remote_offset, size, 0)], "DRAM"
        )
        if not remote_descs:
            logger.error("Failed to create remote xfer descs for job %d", job_id)
            return False

        xfer_handle = self._agent.initialize_xfer(
            op, local_descs, remote_descs, "PDTransport"
        )
        if not xfer_handle:
            logger.error("initialize_xfer failed for job %d", job_id)
            return False

        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            logger.error("agent.transfer failed for job %d", job_id)
            self._agent.release_xfer_handle(xfer_handle)
            return False

        self._inflight[job_id] = NixlTransferHandle(
            job_id=job_id,
            xfer_handle=xfer_handle,
            remote_desc=remote_descs,
            started_at=time.monotonic(),
        )
        return True

    def poll_transfers(self) -> list[tuple[int, bool, float | None]]:
        """Poll in-flight transfers for completion.

        Returns:
            List of (job_id, success, transfer_time_seconds) tuples.
        """
        results: list[tuple[int, bool, float | None]] = []
        completed_ids: list[int] = []
        now = time.monotonic()

        for job_id, handle in self._inflight.items():
            state = self._agent.check_xfer_state(handle.xfer_handle)
            if state == "DONE":
                elapsed = now - handle.started_at
                self._agent.release_xfer_handle(handle.xfer_handle)
                results.append((job_id, True, elapsed))
                completed_ids.append(job_id)
            elif state == "PROC":
                if now - handle.started_at > self._config.transfer_timeout_s:
                    logger.error("Transfer timeout for job %d", job_id)
                    self._agent.release_xfer_handle(handle.xfer_handle)
                    results.append((job_id, False, None))
                    completed_ids.append(job_id)
            else:
                logger.error("Transfer error state=%s for job %d", state, job_id)
                self._agent.release_xfer_handle(handle.xfer_handle)
                results.append((job_id, False, None))
                completed_ids.append(job_id)

        for job_id in completed_ids:
            del self._inflight[job_id]

        return results

    def wait_all(self) -> list[tuple[int, bool, float | None]]:
        """Block until all in-flight transfers complete or timeout."""
        results: list[tuple[int, bool, float | None]] = []
        while self._inflight:
            batch = self.poll_transfers()
            results.extend(batch)
            if self._inflight:
                time.sleep(_POLL_INTERVAL_S)
        return results

    @property
    def has_inflight(self) -> bool:
        return bool(self._inflight)

    def shutdown(self) -> None:
        """Release all NIXL resources."""
        import contextlib

        for handle in list(self._inflight.values()):
            with contextlib.suppress(Exception):
                self._agent.release_xfer_handle(handle.xfer_handle)
        self._inflight.clear()

        if self._local_desc is not None:
            with contextlib.suppress(Exception):
                self._agent.deregister_memory(self._local_desc)
            self._local_desc = None

        self._remote_agents.clear()
        logger.info("PDNixlTransport shutdown complete")

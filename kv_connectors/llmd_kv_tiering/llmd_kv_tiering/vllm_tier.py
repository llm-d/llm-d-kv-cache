# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM secondary tier backed by the kvtierd daemon.

Load it out of tree through TieringOffloadingSpec. No vLLM patch is needed:

    --kv-transfer-config '{
      "kv_connector": "OffloadingConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 20000000000,
        "secondary_tiers": [{
          "type": "LLMDKVTierManager",
          "module_path": "llmd_kv_tiering.vllm_tier",
          "socket_path": "/var/run/llmd-kv-tiering/kvtierd.sock"
        }]
      }
    }'

The daemon owns the memory. This class is a thin control plane: it asks the
daemon for slots, and copies blocks between vLLM's primary CPU tier and the
daemon's shared-memory pool on worker threads. Because the daemon outlives
the engine, blocks stored before an engine restart are still loadable after
it, provided block hashes are stable (set PYTHONHASHSEED).
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    SecondaryTierManager,
    TransferJob,
)

from llmd_kv_tiering.client import (
    ALLOCATED,
    NO_SPACE,
    DaemonClient,
    DaemonError,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector

# Under "vllm." so vLLM's logging configuration applies to these records.
logger = logging.getLogger("vllm.llmd_kv_tiering")

DEFAULT_SOCKET = "/var/run/llmd-kv-tiering/kvtierd.sock"


def derive_namespace(offloading_spec: OffloadingSpec, block_size_bytes: int, salt: str = "") -> str:
    """Name the block layout, so engines share blocks only when their bytes
    are interchangeable.

    Equal byte sizes are not enough: two engines can lay out the same bytes
    differently. So the name also covers the KV cache groups and their
    layers, vLLM's KV cache memory layout, the canonical-layout flag and the
    vLLM version. The model revision is not in vLLM's offloading config;
    pass it, or anything else that changes the bytes, as salt.
    """
    import vllm

    cfg = offloading_spec.config
    par = cfg.parallel
    layout = {
        "engine": "vllm",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "salt": salt,
        "model": cfg.model.name,
        # The KV cache dtype, which can differ from the weights' dtype.
        "dtype": str(cfg.model.dtype),
        "groups": [
            {"tokens_per_block": g.tokens_per_block, "layers": list(g.layer_names)} for g in getattr(cfg, "groups", ())
        ],
        "kv_cache_layout": getattr(cfg, "kv_cache_layout", None),
        "canonical_layout": getattr(cfg, "canonical_layout", False),
        "tokens_per_hash": cfg.cache.tokens_per_hash,
        "blocks_per_chunk": cfg.cache.blocks_per_chunk,
        "block_size_bytes": block_size_bytes,
        "tp": par.tp_size,
        "pp": par.pp_size,
        "pcp": par.pcp_size,
        "dcp": par.dcp_size,
        "replicated_layout": cfg.replicated_layout,
    }
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    return f"{cfg.model.name}:{digest[:16]}"


class LLMDKVTierManager(SecondaryTierManager):
    """Secondary tier whose blocks live in the node's kvtierd pool."""

    medium: ClassVar[Medium] = Medium.CPU
    # Blocks in the daemon are shared with other engines on the node and
    # survive this engine, so a hit is not tied to this process.

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str,
        backpressure_detector: BackpressureDetector | None = None,
        socket_path: str = DEFAULT_SOCKET,
        namespace: str | None = None,
        namespace_salt: str = "",
        num_io_threads: int = 4,
        num_copy_threads: int = 8,
        timeout_s: float = 5.0,
        reconnect_interval_s: float = 1.0,
        locality: str | None = None,
        enable_kv_events: bool = True,
    ) -> None:
        super().__init__(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            backpressure_detector=backpressure_detector,
        )
        self.locality = Locality(locality) if locality is not None else None
        self._socket_path = socket_path
        self._namespace = namespace or derive_namespace(offloading_spec, self.block_size_bytes, namespace_salt)
        self._timeout = timeout_s
        self._reconnect_interval = reconnect_interval_s
        self._client_name = f"vllm/{socket.gethostname()}/{os.getpid()}"

        # Address of the primary tier's first byte. Keep the array alive.
        self._primary = np.frombuffer(primary_kv_view.cast("B"), dtype=np.uint8)
        self._primary_base: int = self._primary.ctypes.data
        self._primary_chunks = len(self._primary) // self.block_size_bytes

        self._events: list[OffloadingEvent] | None = None
        if enable_kv_events and offloading_spec.kv_events_config.enable_kv_cache_events:
            self._events = []
        self._events_lock = threading.Lock()

        # Keys loadable from the daemon, as far as this engine knows. Only
        # the daemon's events change it. They arrive in the order the pool
        # changed, this engine's own stores and evictions included, so a
        # late event never overrides a newer one. An entry can still be
        # stale for a moment; that costs one failed load, which vLLM
        # recomputes.
        self._mirror: set[OffloadKey] = set()
        # Events that arrive while a connection is being set up, applied
        # after its snapshot. None when no connection is being set up.
        self._connecting: list[dict[str, Any]] | None = None
        # The request each in-flight store belongs to, so its stored event
        # can carry the request context.
        self._store_ctx: dict[OffloadKey, ReqContext | None] = {}

        self._client: DaemonClient | None = None
        self._client_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._wake = threading.Event()

        self._executor = ThreadPoolExecutor(max_workers=num_io_threads, thread_name_prefix="llmd-kv-tiering-io")
        # Copies of one job are split across these threads; ctypes.memmove
        # releases the GIL, so they run in parallel.
        self._copy_pool = ThreadPoolExecutor(
            max_workers=max(1, num_copy_threads), thread_name_prefix="llmd-kv-tiering-copy"
        )
        self._copy_threads = max(1, num_copy_threads)
        self._inflight: dict[int, Future] = {}
        self._finished: deque[JobResult] = deque()

        self._try_connect()
        self._reconnector = threading.Thread(target=self._reconnect_loop, name="llmd-kv-tiering-connect", daemon=True)
        self._reconnector.start()
        logger.info(
            "llm-d KV tier: socket=%s namespace=%s block=%d bytes connected=%s",
            socket_path,
            self._namespace,
            self.block_size_bytes,
            self._client is not None,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _try_connect(self) -> bool:
        # The disconnect callback names its own client, so a late callback
        # from an old connection cannot clear a newer one.
        holder: list[DaemonClient] = []
        with self._client_lock:
            self._connecting = []
        try:
            client = DaemonClient(
                self._socket_path,
                self._namespace,
                self.block_size_bytes,
                self._client_name,
                on_event=self._on_event,
                on_disconnect=lambda: self._on_disconnect(holder[0] if holder else None),
                timeout_s=self._timeout,
            )
        except (OSError, DaemonError) as e:
            logger.debug("kvtierd not reachable at %s: %s", self._socket_path, e)
            with self._client_lock:
                self._connecting = None
            return False
        try:
            resp = client.call("snapshot")
        except DaemonError as e:
            logger.warning("kvtierd snapshot failed: %s", e)
            client.close()
            with self._client_lock:
                self._connecting = None
            return False
        snapshot = [OffloadKey(k) for k in resp.get("keys") or ()]
        with self._client_lock:
            holder.append(client)
            buffered, self._connecting = self._connecting or [], None
            # The connection can drop between the snapshot and here. Its
            # disconnect callback then found nothing to clear, so check now,
            # or a dead client would be installed and never replaced.
            if not client.connected:
                logger.warning("kvtierd connection dropped while connecting")
                return False
            self._client = client
            self._mirror.update(snapshot)
        self._emit(snapshot, removed=False)
        # The snapshot already reflects events received before its reply.
        for msg in buffered:
            if msg["_rx"] > resp["_rx"]:
                self._apply_event(msg)
        logger.info(
            "Connected to kvtierd (session %s, %d slots, %d blocks present)",
            client.session,
            client.num_slots,
            len(snapshot),
        )
        return True

    def _on_disconnect(self, client: DaemonClient | None) -> None:
        with self._client_lock:
            if client is None or self._client is not client:
                return
            self._client = None
            lost = list(self._mirror)
            self._mirror.clear()
        self._emit(lost, removed=True)
        if not self._shutdown.is_set():
            logger.warning("Lost connection to kvtierd; lookups miss until it returns")
        self._wake.set()

    def _reconnect_loop(self) -> None:
        while not self._shutdown.is_set():
            self._wake.wait(self._reconnect_interval)
            self._wake.clear()
            if self._shutdown.is_set():
                return
            if self._client is None:
                self._try_connect()

    def _current(self) -> DaemonClient | None:
        with self._client_lock:
            return self._client

    def _on_event(self, msg: dict[str, Any]) -> None:
        with self._client_lock:
            if self._connecting is not None:
                self._connecting.append(msg)
                return
        self._apply_event(msg)

    def _apply_event(self, msg: dict[str, Any]) -> None:
        stored = [OffloadKey(k) for k in msg.get("stored") or ()]
        removed = [OffloadKey(k) for k in msg.get("removed") or ()]
        self._mirror.update(stored)
        self._mirror.difference_update(removed)
        # Stored events of this engine's own stores carry their request.
        by_ctx: dict[int, tuple[ReqContext | None, list[OffloadKey]]] = {}
        for k in stored:
            ctx = self._store_ctx.pop(k, None)
            by_ctx.setdefault(id(ctx), (ctx, []))[1].append(k)
        for ctx, keys in by_ctx.values():
            self._emit(keys, removed=False, req_context=ctx)
        self._emit(removed, removed=True)

    def _emit(self, keys: list[OffloadKey], removed: bool, req_context: ReqContext | None = None) -> None:
        if self._events is None or not keys:
            return
        with self._events_lock:
            self._events.append(
                OffloadingEvent(
                    keys=keys,
                    medium=self.medium,
                    removed=removed,
                    locality=self.locality,
                    req_context=req_context,
                )
            )

    # ------------------------------------------------------------------
    # SecondaryTierManager
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._mirror:
            return LookupResult.HIT
        return LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        self._submit(job_metadata, self._run_store)

    def submit_load(self, job_metadata: TransferJob) -> None:
        self._submit(job_metadata, self._run_load)

    def _submit(self, job: TransferJob, fn) -> None:
        assert len(job.keys) == len(job.chunk_ids), "keys and chunk_ids differ"
        fut = self._executor.submit(self._run_job, job, fn)
        self._inflight[job.job_id] = fut

    def _run_job(self, job: TransferJob, fn) -> None:
        start = time.monotonic()
        try:
            result = fn(job)
        except Exception as e:  # noqa: BLE001 - every failure becomes a result
            logger.debug("job %d failed: %s", job.job_id, e)
            result = JobResult(job_id=job.job_id, success=False)
        result.transfer_time = time.monotonic() - start
        self._finished.append(result)

    def _copy_many(self, copies: list[tuple[int, int]]) -> None:
        """memmove each (dst, src) pair of block_size_bytes, in parallel."""
        if not copies:
            return
        size = self.block_size_bytes
        n = min(self._copy_threads, len(copies))
        if n == 1:
            for dst, src in copies:
                ctypes.memmove(dst, src, size)
            return

        def run(part: list[tuple[int, int]]) -> None:
            for dst, src in part:
                ctypes.memmove(dst, src, size)

        futures = [self._copy_pool.submit(run, copies[i::n]) for i in range(n)]
        for f in futures:
            f.result()

    def _primary_address(self, chunk_id: int) -> int:
        if not 0 <= chunk_id < self._primary_chunks:
            raise IndexError(f"chunk {chunk_id} outside the primary tier")
        return self._primary_base + chunk_id * self.block_size_bytes

    def _run_store(self, job: TransferJob) -> JobResult:
        client = self._current()
        if client is None:
            return JobResult(job_id=job.job_id, success=False)
        keys = list(job.keys)
        resp = client.call("alloc", keys=keys)
        slots, status = resp["slots"], resp["status"]

        size = self.block_size_bytes
        written: list[OffloadKey] = []
        ok: list[bool] = []
        copies: list[tuple[int, int]] = []
        for key, cid, slot, st in zip(keys, job.chunk_ids, slots, status):
            if st != ALLOCATED:
                continue
            written.append(key)
            try:
                copies.append((client.arena.slot_address(slot), self._primary_address(int(cid))))
                ok.append(True)
            except IndexError:
                logger.exception("bad primary chunk %d", int(cid))
                ok.append(False)
        try:
            self._copy_many(copies)
        except Exception:  # noqa: BLE001
            logger.exception("copy into the pool failed")
            ok = [False] * len(written)
        committed = [k for k, good in zip(written, ok) if good]
        if written:
            # The stored event arrives before the commit reply, and the
            # mirror and vLLM's KV events are updated from it.
            for k in committed:
                self._store_ctx[k] = job.req_context
            try:
                client.call("commit", keys=written, ok=ok)
            finally:
                for k in committed:
                    self._store_ctx.pop(k, None)
        success = all(st != NO_SPACE for st in status) and all(ok)
        return JobResult(
            job_id=job.job_id,
            success=success,
            transfer_bytes=len(committed) * size,
        )

    def _run_load(self, job: TransferJob) -> JobResult:
        client = self._current()
        if client is None:
            return JobResult(job_id=job.job_id, success=False)
        keys = list(job.keys)
        t1 = time.monotonic()
        slots = client.call("pin", keys=keys)["slots"]
        t2 = time.monotonic()
        loaded: list[OffloadKey] = []
        pinned: list[OffloadKey] = []
        copies: list[tuple[int, int]] = []
        try:
            for key, cid, slot in zip(keys, job.chunk_ids, slots):
                if slot < 0:
                    continue
                pinned.append(key)
                copies.append((self._primary_address(int(cid)), client.arena.slot_address(slot)))
            self._copy_many(copies)
            loaded = list(pinned)
        finally:
            if pinned:
                # The daemon drops a lost session's pins.
                with contextlib.suppress(DaemonError):
                    client.call("unpin", keys=pinned)
        t3 = time.monotonic()
        size = self.block_size_bytes
        logger.debug(
            "load job %d: %d/%d blocks, %.1f MB; pin %.1f ms, copy %.1f ms",
            job.job_id,
            len(loaded),
            len(keys),
            len(loaded) * size / 1e6,
            (t2 - t1) * 1e3,
            (t3 - t2) * 1e3,
        )
        success = len(loaded) == len(keys)
        return JobResult(
            job_id=job.job_id,
            success=success,
            successful_keys=None if success else tuple(loaded),
            transfer_bytes=len(loaded) * size,
        )

    def get_finished_jobs(self) -> Iterable[JobResult]:
        results = []
        while self._finished:
            result = self._finished.popleft()
            self._inflight.pop(result.job_id, None)
            results.append(result)
        return results

    def has_pending_work(self) -> bool:
        return bool(self._inflight) or bool(self._finished)

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self._events is None:
            return ()
        with self._events_lock:
            events, self._events = self._events, []
        return events

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def drain_jobs(self) -> None:
        pending = list(self._inflight.values())
        if pending:
            wait(pending)

    def shutdown(self) -> None:
        self._shutdown.set()
        self._wake.set()
        self._executor.shutdown(wait=True)
        self._copy_pool.shutdown(wait=True)
        client = self._current()
        if client is not None:
            client.close()
        self._reconnector.join(timeout=self._timeout)

    # Test and debugging helpers.
    @property
    def connected(self) -> bool:
        return self._current() is not None

    @property
    def namespace(self) -> str:
        return self._namespace

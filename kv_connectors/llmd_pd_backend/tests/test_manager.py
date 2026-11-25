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

"""Unit tests for PDSecondaryTierManager with mocked NIXL."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.conftest import (
    LookupResult,
    ReqContext,
    TransferJob,
    make_offload_key,
)


def _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec):
    """Create a PDSecondaryTierManager with mocked transport."""
    from llmd_pd_backend.manager import PDSecondaryTierManager

    mgr = PDSecondaryTierManager(
        offloading_spec=mock_offloading_spec,
        primary_kv_view=primary_kv_view,
        tier_type="pd",
        host="127.0.0.1",
        port=5710,
    )
    return mgr


def _make_job(
    job_id: int,
    keys: list | None = None,
    is_promotion: bool = True,
    req_context: ReqContext | None = None,
) -> TransferJob:
    if keys is None:
        keys = [make_offload_key(b"test", 0)]
    if req_context is None:
        req_context = ReqContext(req_id="test")
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array([0]),
        is_promotion=is_promotion,
        req_context=req_context,
    )


class TestLookup:
    def test_miss_when_key_not_stored(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        key = make_offload_key(b"missing", 0)
        assert mgr.lookup(key, req_context) == LookupResult.MISS

    def test_hit_after_store(
        self,
        mock_nixl_agent,
        primary_kv_view,
        mock_offloading_spec,
        req_context,
        sample_keys,
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)

        store_job = TransferJob(
            job_id=0,
            keys=sample_keys[:2],
            chunk_ids=np.array([0, 1]),
            is_promotion=False,
            req_context=req_context,
        )
        mgr.submit_store(store_job)

        assert mgr.lookup(sample_keys[0], req_context) == LookupResult.HIT
        assert mgr.lookup(sample_keys[1], req_context) == LookupResult.HIT
        assert mgr.lookup(sample_keys[2], req_context) == LookupResult.MISS


class TestSubmitStore:
    def test_creates_pending_job(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        job = _make_job(42, is_promotion=False, req_context=req_context)
        mgr.submit_store(job)
        assert mgr.has_pending_work()

    def test_store_completes_immediately_in_get_finished(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        job = _make_job(42, is_promotion=False, req_context=req_context)
        mgr.submit_store(job)

        results = list(mgr.get_finished_jobs())
        assert len(results) == 1
        assert results[0].job_id == 42
        assert results[0].success is True


class TestSubmitLoad:
    def test_load_with_peer_params(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, pd_req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        job = _make_job(10, is_promotion=True, req_context=pd_req_context)
        mgr.submit_load(job)
        assert mgr.has_pending_work()

    def test_load_without_peer_warns(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        job = _make_job(10, is_promotion=True, req_context=req_context)
        mgr.submit_load(job)
        assert mgr.has_pending_work()


class TestGetFinishedJobs:
    def test_returns_completed_transport_results(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, pd_req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        job = _make_job(99, is_promotion=True, req_context=pd_req_context)
        mgr.submit_load(job)

        # Simulate transport completing the transfer
        with patch.object(
            mgr._transport, "poll_transfers", return_value=[(99, True, 0.5)]
        ):
            results = list(mgr.get_finished_jobs())

        assert len(results) == 1
        assert results[0].job_id == 99
        assert results[0].success is True
        assert results[0].transfer_time == 0.5

    def test_returns_failure(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, pd_req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        job = _make_job(100, is_promotion=True, req_context=pd_req_context)
        mgr.submit_load(job)

        with patch.object(
            mgr._transport, "poll_transfers", return_value=[(100, False, None)]
        ):
            results = list(mgr.get_finished_jobs())

        assert len(results) == 1
        assert results[0].success is False


class TestDrainJobs:
    def test_drain_blocks_until_complete(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        job = _make_job(1, is_promotion=False, req_context=req_context)
        mgr.submit_store(job)
        mgr.drain_jobs()
        assert not mgr.has_pending_work()


class TestHasPendingWork:
    def test_false_when_idle(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        assert not mgr.has_pending_work()

    def test_true_with_pending_jobs(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        mgr.submit_store(_make_job(1, is_promotion=False, req_context=req_context))
        assert mgr.has_pending_work()


class TestEviction:
    def test_evicted_key_returns_miss(
        self,
        mock_nixl_agent,
        primary_kv_view,
        mock_offloading_spec,
        req_context,
        sample_keys,
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)

        store_job = TransferJob(
            job_id=0,
            keys=sample_keys,
            chunk_ids=np.array(range(len(sample_keys))),
            is_promotion=False,
            req_context=req_context,
        )
        mgr.submit_store(store_job)
        assert mgr.lookup(sample_keys[0], req_context) == LookupResult.HIT

        mgr.evict([sample_keys[0]])
        assert mgr.lookup(sample_keys[0], req_context) == LookupResult.MISS
        assert mgr.lookup(sample_keys[1], req_context) == LookupResult.HIT

    def test_evict_nonexistent_key_is_noop(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        mgr.evict([make_offload_key(b"nonexistent", 0)])


class TestDuplicateStore:
    def test_duplicate_store_is_idempotent(
        self, mock_nixl_agent, primary_kv_view, mock_offloading_spec, req_context
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)
        key = make_offload_key(b"dup", 0)

        for i in range(3):
            job = TransferJob(
                job_id=i,
                keys=[key],
                chunk_ids=np.array([0]),
                is_promotion=False,
                req_context=req_context,
            )
            mgr.submit_store(job)

        assert mgr.lookup(key, req_context) == LookupResult.HIT


class TestShutdown:
    def test_shutdown_clears_state(
        self,
        mock_nixl_agent,
        primary_kv_view,
        mock_offloading_spec,
        req_context,
        sample_keys,
    ):
        mgr = _make_manager(mock_nixl_agent, primary_kv_view, mock_offloading_spec)

        store_job = TransferJob(
            job_id=0,
            keys=sample_keys,
            chunk_ids=np.array(range(len(sample_keys))),
            is_promotion=False,
            req_context=req_context,
        )
        mgr.submit_store(store_job)
        mgr.shutdown()

        assert not mgr.has_pending_work()
        for key in sample_keys:
            assert mgr.lookup(key, req_context) == LookupResult.MISS

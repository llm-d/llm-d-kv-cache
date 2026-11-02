# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Actual vLLM manager contracts on real POSIX/GPFS files; no GPU simulation claims."""

import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.v1.kv_offload.base import ReqContext

from llmd_fs_backend.request_manager import RequestStorageTTLManager
from llmd_fs_backend.request_routing import validate_request_policies

pytestmark = pytest.mark.no_cuda_required


def make(tmp_path):
    maps = {}
    for seconds in (1, 30):
        parent = tmp_path / f".llmd-ttl-v1-{seconds}"
        parent.mkdir()
        maps[seconds] = SimpleNamespace(
            model_name="test",
            get_file_name=lambda k, p=parent: str(p / (k.hex() + ".bin")),
        )
    worker = SimpleNamespace(get_file_name=lambda k: str(tmp_path / "worker" / k.hex()))
    return RequestStorageTTLManager(worker, maps, {}, 30), maps


def ctx(name, ttl=30):
    return ReqContext(name, {"llmd_storage_ttl_seconds": ttl})


def publish(manager, context, keys):
    out = manager.prepare_store(keys, context)
    for path in out.store_spec.source_paths.values():
        Path(path).write_bytes(b"KV")
    manager.complete_store(keys, context)
    return out


def test_same_prefix_separate_lifetimes_and_mapper_unchanged(tmp_path):
    m, maps = make(tmp_path)
    for sec in (1, 30):
        c = ctx(str(sec), sec)
        m.on_new_request(c)
        publish(m, c, [b"key"])
        m.on_request_finished(c)
    assert maps[1].get_file_name(b"key") != maps[30].get_file_name(b"key")
    assert all(
        Path(x.get_file_name(b"key")).read_bytes() == b"KV" for x in maps.values()
    )
    assert "worker" in m.worker_mapper.get_file_name(b"key")
    m.shutdown()


def test_default_hint_and_invalid_decline(tmp_path):
    m, _ = make(tmp_path)
    c = ReqContext("default", None)
    m.on_new_request(c)
    assert m._tokens[c.req_id].seconds == 30
    m.on_request_finished(c)
    for i, hint in enumerate([None, True, 0, 2, "30", {}, [30]]):
        c = ctx(str(i), hint)
        m.on_new_request(c)
        assert not m.lookup(b"key", c) and m.prepare_store([b"key"], c) is None
        m.on_request_finished(c)
    assert m.routes.snapshot()["requests"] == 0
    m.shutdown()


def test_route_immutable_and_internal_spec_pickles(tmp_path):
    m, maps = make(tmp_path)
    c = ctx("r", 1)
    m.on_new_request(c)
    c.kv_transfer_params["llmd_storage_ttl_seconds"] = 30
    out = m.prepare_store([b"a"], c)
    assert pickle.loads(pickle.dumps(out.store_spec)).source_paths == {
        m.worker_mapper.get_file_name(b"a"): maps[1].get_file_name(b"a")
    }
    m.complete_store([b"a"], c, success=False)
    m.on_request_finished(c)
    m.shutdown()


def test_finish_before_partial_stores_and_reset_guard(tmp_path):
    m, _ = make(tmp_path)
    c = ctx("r")
    m.on_new_request(c)
    m.prepare_store([b"a", b"b"], c)
    m.on_request_finished(c)
    for fn in (m.reset_cache, m.shutdown):
        with pytest.raises(RuntimeError):
            fn()
    m.complete_store([b"b"], c)
    assert m.routes.snapshot()["jobs"] == 1
    m.complete_store([b"a"], c, success=False)
    assert m.routes.snapshot()["requests"] == 0
    m.shutdown()


def test_duplicate_key_jobs_counted_and_stale_context_cannot_complete_reused_id(
    tmp_path,
):
    m, _ = make(tmp_path)
    old = ctx("r")
    m.on_new_request(old)
    m.prepare_store([b"a"], old)
    m.prepare_store([b"a"], old)
    m.on_request_finished(old)
    m.complete_store([b"a"], old)
    assert m.routes.snapshot()["jobs"] == 1
    m.complete_store([b"a"], old)
    new = ctx("r", 1)
    m.on_new_request(new)
    m.prepare_store([b"a"], new)
    with pytest.raises(RuntimeError):
        m.complete_store([b"a"], old)
    assert m.routes.snapshot()["jobs"] == 1
    m.complete_store([b"a"], new)
    m.on_request_finished(new)
    m.shutdown()


def test_accepted_fd_survives_unlink_expiry_and_finished_request(tmp_path):
    m, maps = make(tmp_path)
    w = ctx("write", 1)
    m.on_new_request(w)
    publish(m, w, [b"a"])
    m.on_request_finished(w)
    c = ctx("read", 1)
    m.on_new_request(c)
    assert m.lookup(b"a", c)
    spec = m.prepare_load([b"a"], c)
    fd = spec.source_paths[m.worker_mapper.get_file_name(b"a")]
    m.on_request_finished(c)
    os.unlink(maps[1].get_file_name(b"a"))
    assert Path(fd).read_bytes() == b"KV"
    time.sleep(1.05)
    assert Path(fd).read_bytes() == b"KV" and m.routes.snapshot()["jobs"] == 1
    m.complete_load([b"a"], c)
    assert m.routes.snapshot()["requests"] == 0
    with pytest.raises(FileNotFoundError):
        Path(fd).read_bytes()
    m.shutdown()


def test_failed_load_prepare_does_not_leak_route_job(tmp_path):
    m, _ = make(tmp_path)
    c = ctx("r")
    m.on_new_request(c)
    # The upstream manager requires a successful lookup before prepare_load.
    # Its KeyError still must not strand the route job accepted by the wrapper.
    with pytest.raises(KeyError):
        m.prepare_load([b"absent"], c)
    assert m.routes.snapshot()["jobs"] == 0
    m.on_request_finished(c)
    m.shutdown()


def test_unmatched_store_batch_leaves_accepted_jobs_unchanged(tmp_path):
    m, _ = make(tmp_path)
    c = ctx("r")
    m.on_new_request(c)
    m.prepare_store([b"a"], c)
    with pytest.raises(RuntimeError):
        m.complete_store([b"a", b"absent"], c)
    assert m.routes.snapshot()["jobs"] == 1
    m.complete_store([b"a"], c)
    m.on_request_finished(c)
    m.shutdown()


def test_new_work_after_finish_is_rejected_before_store_delegation(tmp_path):
    m, _ = make(tmp_path)
    c = ctx("r")
    m.on_new_request(c)
    m.prepare_store([b"a"], c)
    m.on_request_finished(c)
    with pytest.raises(RuntimeError):
        m.prepare_store([b"b"], c)
    m.complete_store([b"a"], c)
    m.shutdown()


@pytest.mark.parametrize(
    "allowed,default",
    [
        ([], 1),
        ([1, 1], 1),
        ([True], True),
        ([1], None),
        ([0], 0),
        ([1], 2),
        (["1"], 1),
        ("1", 1),
        ([2147483648], 2147483648),
    ],
)
def test_server_allowlist_rejects_unsupported_configuration(allowed, default):
    with pytest.raises(ValueError):
        validate_request_policies(allowed, default)


def test_allowlist_validation_has_no_client_path_input():
    assert validate_request_policies([1, 30], 30) == (1, 30)

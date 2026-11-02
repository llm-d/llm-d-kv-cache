# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler lifecycle and namespace hint tests with the actual vLLM API."""

import os
from types import SimpleNamespace

import pytest
from vllm.v1.kv_offload.base import ReqContext

from llmd_fs_backend.manager import SharedStorageOffloadingManager

pytestmark = pytest.mark.no_cuda_required


def manager(tmp_path):
    path = tmp_path / "entry.bin"
    path.write_bytes(b"KV")
    mapper = SimpleNamespace(get_file_name=lambda _: str(path), model_name="test-model")
    return SharedStorageOffloadingManager(mapper, {"storage_ttl_seconds": 60}), path


def test_request_hint_matches_namespace_without_renewal(tmp_path):
    m, path = manager(tmp_path)
    context = ReqContext("r", {"llmd_storage_ttl_seconds": 60})
    m.on_new_request(context)
    before = path.stat().st_mtime_ns
    assert m.lookup(b"key", context)
    spec = m.prepare_load([b"key"], context)
    path.unlink()
    with open(spec.source_paths[str(path)], "rb") as pinned:
        assert pinned.read() == b"KV"
    assert os.stat(spec.source_paths[str(path)]).st_mtime_ns == before
    m.on_request_finished(context)
    assert m._ttl_pins.requests
    m.complete_load([b"key"], context)
    assert not m._ttl_pins.requests
    m.shutdown()


@pytest.mark.parametrize("hint", [30, 0, -1, True, "60"])
def test_conflicting_request_ttl_declines_storage_safely(tmp_path, hint):
    m, _ = manager(tmp_path)
    ctx = ReqContext("r", {"llmd_storage_ttl_seconds": hint})
    m.on_new_request(ctx)
    assert not m.lookup(b"key", ctx)
    assert m.prepare_store([b"key"], ctx) is None
    m.on_request_finished(ctx)
    assert not m._declined_ttl_requests
    m.shutdown()


def test_reset_refuses_inflight_descriptors_then_releases(tmp_path):
    m, path = manager(tmp_path)
    ctx = ReqContext("r")
    assert m.lookup(b"key", ctx)
    m.prepare_load([b"key"], ctx)
    with pytest.raises(RuntimeError, match="accepted TTL loads"):
        m.reset_cache()
    m.complete_load([b"key"], ctx)
    m.reset_cache()
    assert not m._ttl_pins.requests
    m.shutdown()


def test_ttl_configuration_creates_separate_namespace(tmp_path):
    from test_spec import make_spec

    normal = make_spec(tmp_path, {})
    fixed = make_spec(tmp_path, {"storage_ttl_seconds": 60})
    assert normal.file_mapper.base_path != fixed.file_mapper.base_path
    assert "/.llmd-ttl-v1-60/" in fixed.file_mapper.base_path


@pytest.mark.parametrize("setting", [{"backend": "OBJ"}, {"gds_mode": "read_write"}])
def test_unsupported_ttl_backend_rejected(tmp_path, setting):
    from test_spec import make_spec

    with pytest.raises(ValueError, match="POSIX"):
        make_spec(tmp_path, {"storage_ttl_seconds": 60, **setting})

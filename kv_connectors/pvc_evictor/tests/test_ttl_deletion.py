"""Exercise the packaged TTL protocol through the production deletion path."""

import importlib.util
import logging
import os
import sys
import time
import types
from pathlib import Path

import pytest
from test_deleter import _deleter


@pytest.fixture
def ttl_protocol(monkeypatch):
    path = Path(__file__).parents[2] / "llmd_fs_backend/llmd_fs_backend/storage_ttl.py"
    spec = importlib.util.spec_from_file_location("llmd_fs_backend.storage_ttl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The evictor image copies this filesystem-only module without vLLM's init.
    package = types.ModuleType("llmd_fs_backend")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "llmd_fs_backend", package)
    monkeypatch.setitem(sys.modules, "llmd_fs_backend.storage_ttl", module)
    return module


def candidate(root, *, ttl=True, age=0, idle=0):
    namespace = root / ".llmd-ttl-v1-30" if ttl else root
    path = namespace / "model_r0/abc/00_g0/abc000.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"KV bytes")
    now = time.time()
    os.utime(path, (now - idle, now - age))
    return path


@pytest.mark.parametrize("dry_run", [False, True])
def test_expiry_without_pressure_respects_dry_run(tmp_path, ttl_protocol, dry_run):
    expired = candidate(tmp_path, age=60)
    result = _deleter.delete_batch([str(expired)], dry_run, logging.getLogger("test"), pressure_enabled=False)
    assert result == ((1, 0, []) if dry_run else (1, 8, [str(expired)]))
    assert expired.exists() == dry_run


def test_pressure_off_keeps_unexpired_ttl_and_legacy_files(tmp_path, ttl_protocol):
    fresh = candidate(tmp_path, idle=7200)
    legacy = candidate(tmp_path, ttl=False, idle=7200)
    assert _deleter.delete_batch(
        [str(fresh), str(legacy)], False, logging.getLogger("test"), pressure_enabled=False
    ) == (0, 0, [])
    assert fresh.exists() and legacy.exists()


def test_pressure_can_remove_idle_unexpired_ttl(tmp_path, ttl_protocol):
    idle = candidate(tmp_path, idle=7200)
    assert _deleter.delete_batch([str(idle)], False, logging.getLogger("test")) == (1, 8, [str(idle)])
    assert not idle.exists()


def test_mixed_batch_preserves_hot_legacy_file(tmp_path, ttl_protocol):
    expired = candidate(tmp_path, age=60)
    hot = candidate(tmp_path, ttl=False)
    assert _deleter.delete_batch([str(expired), str(hot)], False, logging.getLogger("test")) == (1, 8, [str(expired)])
    assert hot.exists() and not expired.exists()


def test_vanished_ttl_candidate_is_not_counted(tmp_path, ttl_protocol):
    vanished = candidate(tmp_path, age=60)
    vanished.unlink()
    assert _deleter.delete_batch([str(vanished)], False, logging.getLogger("test")) == (0, 0, [])

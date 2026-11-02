# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Actual filesystem protocol checks; no model or GPU required."""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.no_cuda_required,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux procfs protocol"),
]
p = Path(__file__).parents[1] / "llmd_fs_backend/storage_ttl.py"
spec = importlib.util.spec_from_file_location("storage_ttl_protocol", p)
ttl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttl)


def entry(tmp_path, age=0):
    p = tmp_path / (ttl.PREFIX + "2") / "model_r0/abc/00_g0/abc000.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"original KV bytes")
    now = time.time_ns()
    os.utime(p, ns=(now, now - int(age * 1e9)))
    return p


def test_deadline_is_not_renewed_by_reads(tmp_path):
    p = entry(tmp_path, age=3)
    with p.open("rb") as f:
        assert f.read() == b"original KV bytes"
    assert ttl.try_delete(p, pressure=False, idle_seconds=3600) is not None
    assert not p.exists()


def test_accepted_read_survives_expiry_unlink_and_late_completion(tmp_path):
    p = entry(tmp_path)
    pins = ttl.InodePins(2)
    assert pins.lookup("r", b"key", str(p))
    read_path = pins.prepare_load("r", [(b"key", str(p))])[str(p)]
    # The pathname is removed after admission; the worker reads the pinned inode.
    p.unlink()
    pins.finish("r")
    code = 'import sys;sys.stdout.buffer.write(open(sys.argv[1],"rb").read())'
    assert (
        subprocess.check_output([sys.executable, "-c", code, read_path])
        == b"original KV bytes"
    )
    pins.complete_load("r")
    assert not Path(read_path).exists()
    assert not pins.requests


def test_replacement_does_not_change_admitted_generation(tmp_path):
    p = entry(tmp_path)
    pins = ttl.InodePins(2)
    assert pins.lookup("r", b"key", str(p))
    old = pins.prepare_load("r", [(b"key", str(p))])[str(p)]
    replacement = p.with_suffix(".tmp")
    replacement.write_bytes(b"new generation")
    replacement.replace(p)
    assert Path(old).read_bytes() == b"original KV bytes"
    assert p.read_bytes() == b"new generation"
    pins.complete_load("r")
    pins.finish("r")


def test_speculative_lookup_is_released_at_finish(tmp_path):
    p = entry(tmp_path)
    pins = ttl.InodePins(2)
    assert pins.lookup("r", b"key", str(p))
    fd = pins.requests["r"]["current"][b"key"][0]
    pins.finish("r")
    with pytest.raises(OSError):
        os.fstat(fd)


def test_new_lookup_after_expiry_misses(tmp_path):
    p = entry(tmp_path, age=3)
    pins = ttl.InodePins(2)
    assert not pins.lookup("r", b"key", str(p))
    pins.finish("r")


def test_pressure_off_does_not_delete_unexpired_file(tmp_path):
    p = entry(tmp_path)
    assert ttl.try_delete(p, pressure=False, idle_seconds=0) is None
    assert p.exists()


def test_publisher_guard_prevents_delete(tmp_path):
    p = entry(tmp_path, age=3)
    # POSIX locks are process-owned: a separate process is necessary to test
    # exclusion. Native writers additionally serialize threads per stripe.
    code = (
        "import fcntl,sys;f=open(sys.argv[1],'a+b');"
        "fcntl.lockf(f,fcntl.LOCK_EX);print('held',flush=True);sys.stdin.read()"
    )
    owner = subprocess.Popen(
        [sys.executable, "-u", "-c", code, str(ttl.lock_path(p))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert owner.stdout.readline().strip() == "held"
        assert ttl.try_delete(p, pressure=False, idle_seconds=0) is None
    finally:
        owner.stdin.close()
        owner.wait(10)
    assert ttl.try_delete(p, pressure=False, idle_seconds=0) is not None


def test_new_generation_is_rechecked_under_guard(tmp_path):
    p = entry(tmp_path, age=3)
    replacement = p.with_suffix(".tmp")
    replacement.write_bytes(b"new generation")
    replacement.replace(p)
    assert ttl.try_delete(p, pressure=False, idle_seconds=3600) is None
    assert p.read_bytes() == b"new generation"


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "60", 2147483648])
def test_invalid_policy_is_rejected(value):
    with pytest.raises(ValueError):
        ttl.validate_seconds(value)

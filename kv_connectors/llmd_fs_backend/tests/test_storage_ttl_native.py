# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native GPU persistence/expiry and descriptor handoff; tiny real KV buffers."""

import json
import os
import subprocess
import sys
import time

import torch
from test_access_time import wait_finished

from llmd_fs_backend import worker
from llmd_fs_backend.storage_ttl import InodePins


def engine_and_data():
    data = (
        torch.arange(256, dtype=torch.int32, device="cuda").view(4, 64).to(torch.int8)
    )
    engine = worker.storage_offload.StorageOffloadEngine(
        2, 2, [data], [[0]], [64], 1, "disabled", 0, True
    )
    return engine, data


def path_for(tmp_path, seconds=60):
    return (
        tmp_path
        / f".llmd-ttl-v1-{seconds}"
        / "model_r0"
        / "abc"
        / "00_g0"
        / "abc000.bin"
    )


def test_duplicate_native_store_does_not_renew_deadline(tmp_path):
    engine, data = engine_and_data()
    path = path_for(tmp_path)
    assert engine.async_store_gpu_blocks(1, [0], [str(path)], [[0]], [0])
    wait_finished(engine, 1)
    first = path.stat()
    for job in [2, 3, 4]:
        assert engine.async_store_gpu_blocks(job, [0], [str(path)], [[0]], [0])
    pending = {2, 3, 4}
    deadline = time.monotonic() + 10
    while pending and time.monotonic() < deadline:
        results = dict(engine.get_finished())
        assert all(results.values())
        pending.difference_update(results)
        time.sleep(0.01)
    assert not pending
    last = path.stat()
    assert last.st_ino == first.st_ino
    assert last.st_mtime_ns == first.st_mtime_ns
    assert path.read_bytes() == bytes(data[0].cpu().tolist())


def test_expired_native_store_replaces_generation(tmp_path):
    engine, data = engine_and_data()
    path = path_for(tmp_path, seconds=1)
    assert engine.async_store_gpu_blocks(1, [0], [str(path)], [[0]], [0])
    wait_finished(engine, 1)
    first = path.stat()
    time.sleep(1.2)
    pins = InodePins(1)
    assert not pins.lookup("expired", b"key", str(path))
    pins.finish("expired")
    assert engine.async_store_gpu_blocks(2, [0], [str(path)], [[0]], [0])
    wait_finished(engine, 2)
    last = path.stat()
    assert last.st_ino != first.st_ino
    assert last.st_mtime_ns > first.st_mtime_ns
    assert path.read_bytes() == bytes(data[0].cpu().tolist())


def test_native_read_of_pinned_unlinked_inode(tmp_path):
    engine, data = engine_and_data()
    path = path_for(tmp_path)
    assert engine.async_store_gpu_blocks(1, [0], [str(path)], [[0]], [0])
    wait_finished(engine, 1)
    pins = InodePins(60)
    assert pins.lookup("reader", b"key", str(path))
    source = pins.prepare_load("reader", [(b"key", str(path))])[str(path)]
    path.unlink()
    assert engine.async_load_gpu_blocks(2, [0], [source], [[1]], [0])
    wait_finished(engine, 2)
    assert torch.equal(data[1], data[0])
    pins.complete_load("reader")
    pins.finish("reader")


def test_gpu_worker_process_reads_scheduler_descriptor(tmp_path):
    engine, data = engine_and_data()
    path = path_for(tmp_path)
    assert engine.async_store_gpu_blocks(1, [0], [str(path)], [[0]], [0])
    wait_finished(engine, 1)
    pins = InodePins(60)
    assert pins.lookup("reader", b"key", str(path))
    source = pins.prepare_load("reader", [(b"key", str(path))])[str(path)]
    path.unlink()
    code = """import json,sys,time,torch
from llmd_fs_backend import worker
x=torch.zeros((2,64),dtype=torch.int8,device='cuda')
e=worker.storage_offload.StorageOffloadEngine(2,1,[x],[[0]],[64],1,'disabled',0,True)
assert e.async_load_gpu_blocks(1,[0],[sys.argv[1]],[[1]],[0])
deadline=time.monotonic()+10
while time.monotonic()<deadline:
 r=dict(e.get_finished())
 if 1 in r:assert r[1];break
 time.sleep(.01)
else:raise TimeoutError('native child read')
print('DATA '+json.dumps(x[1].cpu().tolist()),flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, source],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    returned = next(
        json.loads(line[5:])
        for line in result.stdout.splitlines()
        if line.startswith("DATA ")
    )
    assert returned == data[0].cpu().tolist()
    pins.complete_load("reader")
    pins.finish("reader")

# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Actual optional backend manager, native handlers, and CUDA byte equality."""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.v1.kv_offload.base import GPULoadStoreSpec, ReqContext, make_offload_key

from llmd_fs_backend.request_manager import RequestStorageTTLManager
from llmd_fs_backend.worker import (
    GPUToStorageHandler,
    StorageToGPUHandler,
    storage_offload,
)


def wait(handler, job):
    end = time.monotonic() + 30
    while time.monotonic() < end:
        for x in handler.get_finished():
            if x.job_id == job:
                assert x.success
                return
        time.sleep(0.01)
    raise TimeoutError("Native handler job " + str(job))


def test_actual_backend_mixed_namespace_bytes_late_read_and_duplicate_store(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    roots = {s: root / f".llmd-ttl-v1-{s}" for s in [1, 30]}
    for p in roots.values():
        p.mkdir()

    def mapper(base):
        return SimpleNamespace(
            model_name="backend-native",
            get_file_name=lambda k: str(
                base / "model_r0" / "abc" / "00_g0" / (k.hex() + ".bin")
            ),
        )

    worker = mapper(root / "worker-default")
    m = RequestStorageTTLManager(
        worker, {s: mapper(p) for s, p in roots.items()}, {}, 30
    )
    x = torch.arange(512, dtype=torch.int32, device="cuda").view(8, 64).to(torch.int8)
    x[4].fill_(42)
    engine = storage_offload.StorageOffloadEngine(
        2, 2, [x], [[0]], [64], 1, "disabled", 0, True
    )
    put = GPUToStorageHandler(1, worker, engine, ("GPU", "SHARED_STORAGE"), [64])
    get = StorageToGPUHandler(1, worker, engine, ("SHARED_STORAGE", "GPU"), [64])
    key = make_offload_key(bytes.fromhex("ab" * 16), 0)

    def ctx(name, s):
        return ReqContext(name, {"llmd_storage_ttl_seconds": s})

    def store(name, seconds, block, job):
        c = ctx(name, seconds)
        m.on_new_request(c)
        out = m.prepare_store([key], c)
        assert out and out.keys_to_store == [key]
        path = next(iter(out.store_spec.source_paths.values()))
        assert f"/.llmd-ttl-v1-{seconds}/" in path
        assert put.transfer_async(
            job, (GPULoadStoreSpec([block], [1], [0]), out.store_spec)
        )
        wait(put, job)
        m.complete_store([key], c)
        m.on_request_finished(c)
        return path

    short = store("short", 1, 0, 1)
    long = store("long", 30, 1, 2)
    assert short != long
    longstat = Path(long).stat()
    long_generation = (longstat.st_ino, longstat.st_mtime_ns, longstat.st_size)
    c = ctx("accepted-short", 1)
    m.on_new_request(c)
    assert m.lookup(key, c)
    oldspec = m.prepare_load([key], c)
    m.on_request_finished(c)
    time.sleep(1.1)
    expired = ctx("new-short", 1)
    m.on_new_request(expired)
    assert not m.lookup(key, expired)
    m.on_request_finished(expired)
    os.unlink(short)
    assert store("republish-short", 1, 4, 3) == short
    assert get.transfer_async(4, (oldspec, GPULoadStoreSpec([2], [1], [0])))
    wait(get, 4)
    assert torch.equal(x[2], x[0])
    m.complete_load([key], c)
    # An unexpired duplicate store must neither renew nor overwrite the long copy.
    assert store("duplicate-long", 30, 4, 5) == long
    after = Path(long).stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == long_generation
    for name, seconds, destination, source, job in [
        ("long-read", 30, 3, 1, 6),
        ("new-short-read", 1, 5, 4, 7),
    ]:
        c = ctx(name, seconds)
        m.on_new_request(c)
        assert m.lookup(key, c)
        spec = m.prepare_load([key], c)
        m.on_request_finished(c)
        assert get.transfer_async(
            job, (spec, GPULoadStoreSpec([destination], [1], [0]))
        )
        wait(get, job)
        assert torch.equal(x[destination], x[source])
        m.complete_load([key], c)
    assert (
        not m._tokens
        and m.routes.snapshot()["jobs"] == 0
        and not put._pending_jobs
        and not get._pending_jobs
    )
    assert all(not child._ttl_pins.requests for child in m._managers.values())
    m.shutdown()

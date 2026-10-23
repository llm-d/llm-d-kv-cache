# Copyright 2026 The llm-d Authors.
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

"""Exercise recency updates through the actual native storage engine."""

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from llmd_fs_backend import worker


@pytest.mark.no_cuda_required
@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, False),
        ({"track_file_access": False}, False),
        ({"track_file_access": True}, True),
    ],
)
def test_tracking_configuration_reaches_native_engine(monkeypatch, config, expected):
    constructor = Mock()
    monkeypatch.setattr(worker.storage_offload, "StorageOffloadEngine", constructor)
    caches = SimpleNamespace(
        tensors=[SimpleNamespace(tensor=object())],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=64)]],
    )
    worker.StorageOffloadingHandlers._create_engine(
        None, 2, 1, caches, 1, 0, config, "disabled"
    )
    assert constructor.call_args.args[-1] is expected


@pytest.mark.no_cuda_required
@pytest.mark.parametrize("value", ["true", 1, None])
def test_tracking_configuration_rejects_non_boolean(value):
    caches = SimpleNamespace(tensors=[], group_data_refs=[])
    with pytest.raises(ValueError, match="track_file_access must be a boolean"):
        worker.StorageOffloadingHandlers._create_engine(
            None, 2, 1, caches, 1, 0, {"track_file_access": value}, "disabled"
        )


def wait_finished(engine, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        results = dict(engine.get_finished())
        if job_id in results:
            assert results[job_id]
            return
        time.sleep(0.01)
    pytest.fail("Native storage job did not complete")


def test_successful_native_read_refreshes_atime_preserves_mtime(tmp_path):
    tensors = [
        torch.arange(256, dtype=torch.int32, device="cuda").view(4, 64).to(torch.int8)
    ]
    expected = tensors[0][:2].clone()
    engine = worker.storage_offload.StorageOffloadEngine(
        2, 1, tensors, [[0]], [64], 1, "disabled", 0, True
    )
    paths = [str(tmp_path / "first.bin"), str(tmp_path / "second.bin")]
    assert engine.async_store_gpu_blocks(1, [0, 0], paths, [[0], [1]], [0, 0])
    wait_finished(engine, 1)
    for path in paths:
        os.utime(path, ns=(1_000_000_000, 2_000_000_000))
    tensors[0][2:].zero_()
    before = time.time_ns()
    assert engine.async_load_gpu_blocks(2, [0, 0], paths, [[2], [3]], [0, 0])
    wait_finished(engine, 2)
    assert torch.equal(tensors[0][2:], expected)
    for path in paths:
        stat = os.stat(path)
        # UTIME_NOW uses the kernel filesystem clock, which can lag
        # userspace CLOCK_REALTIME by a clock tick. The aged value must
        # become recent; nanosecond ordering across those clocks is invalid.
        assert (
            before - 1_000_000_000 <= stat.st_atime_ns <= time.time_ns() + 1_000_000_000
        )
        assert stat.st_mtime_ns == 2_000_000_000

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
"""Fixtures: build kvtierd once, start daemons, and fake vLLM's offloading spec."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[3]
BLOCK = 4096  # bytes per primary-tier chunk in these tests


@pytest.fixture(scope="session")
def kvtierd_bin(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bin") / "kvtierd"
    go = shutil.which("go")
    if go is None:
        pytest.skip("go toolchain not found")
    subprocess.run([go, "build", "-o", str(out), "./services/kvtierd"], cwd=REPO, check=True)
    return out


class Daemon:
    def __init__(self, binary: Path, socket_path: Path, slots: int) -> None:
        self.binary = binary
        self.socket_path = socket_path
        self.slots = slots
        self.proc: subprocess.Popen | None = None

    def start(self) -> Daemon:
        self.proc = subprocess.Popen(
            [
                str(self.binary),
                "--socket",
                str(self.socket_path),
                "--capacity",
                str(self.slots * BLOCK),
                "--metrics-addr",
                "",
                "--log-level",
                "warn",
            ]
        )
        deadline = time.monotonic() + 10
        while not self.socket_path.exists():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("kvtierd did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        self.proc = None


@pytest.fixture
def daemon_factory(kvtierd_bin, tmp_path):
    daemons: list[Daemon] = []

    def make(slots: int = 16, start: bool = True) -> Daemon:
        # Unix socket paths are limited to 108 bytes; keep it short.
        sock = Path(f"/tmp/kvtierd-{os.getpid()}-{len(daemons)}-{time.monotonic_ns()}.sock")
        d = Daemon(kvtierd_bin, sock, slots)
        daemons.append(d)
        return d.start() if start else d

    yield make
    for d in daemons:
        d.stop()
        d.socket_path.unlink(missing_ok=True)


@pytest.fixture
def tiers():
    """Tiers a test made; shut down when it ends."""
    made: list = []
    yield made
    for t in made:
        t.shutdown()


def make_offloading_spec(enable_kv_cache_events: bool = True) -> MagicMock:
    """A spec carrying the fields the tier reads, as in vLLM's own tier tests."""
    from vllm.v1.kv_offload.base import OffloadingKVEventsConfig
    from vllm.v1.kv_offload.config import (
        OffloadingCacheConfig,
        OffloadingConfig,
        OffloadingModelConfig,
        OffloadingParallelConfig,
    )

    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
        replicated_layout=False,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec

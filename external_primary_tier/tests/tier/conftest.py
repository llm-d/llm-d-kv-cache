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


import threading

import pytest
from external_primary_tier.protocol import AttachOk
from external_primary_tier.tier.server import ExternalPrimaryTier

from .fake_engine import COMPAT_ID, KV_BYTES_PER_CHUNK, FakeEngineProcess

POOL_CHUNKS = 8
LEASE_SECONDS = 30.0
FENCE_SECONDS = 10.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RunningTier:
    """An External Primary Tier serving in a background thread."""

    def __init__(self, region_dir: str, clock: FakeClock, **kwargs) -> None:
        self.region_dir = region_dir
        self.address = f"ipc://{region_dir}/tier.sock"
        kwargs.setdefault("lease_seconds", LEASE_SECONDS)
        kwargs.setdefault("fence_seconds", FENCE_SECONDS)
        self.server = ExternalPrimaryTier(
            self.address,
            region_dir,
            pool_bytes=POOL_CHUNKS * KV_BYTES_PER_CHUNK,
            clock=clock,
            **kwargs,
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.server.serve, args=(self._stop, 20), daemon=True)
        self._thread.start()
        self._stopped = False

    def stop(self) -> None:
        """Stop cleanly: drain, write the snapshot, close the socket."""
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        self._thread.join(timeout=10)
        self.server.close()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def start_tier(tmp_path, clock):
    """Starts External Primary Tiers on one region directory, stopped after the test."""
    started: list[RunningTier] = []

    def start(**kwargs) -> RunningTier:
        tier = RunningTier(str(tmp_path), clock, **kwargs)
        started.append(tier)
        return tier

    yield start
    for tier in started:
        tier.stop()


@pytest.fixture
def tier(start_tier) -> RunningTier:
    return start_tier()


@pytest.fixture
def engine_process(tier):
    """Makes fake engine processes connected to the tier's address, closed after
    the test. They reach whichever instance is bound to the address."""
    made: list[FakeEngineProcess] = []

    def make() -> FakeEngineProcess:
        process = FakeEngineProcess(tier.address)
        made.append(process)
        return process

    yield make
    for process in made:
        process.close()


@pytest.fixture
def engine(engine_process):
    """Attaches an engine's GPU worker and then its scheduler; returns the
    scheduler's process."""

    def attach(client_id: str = "engine-a", compat_id: bytes = COMPAT_ID) -> FakeEngineProcess:
        assert isinstance(engine_process().attach(client_id, rank=0, compat_id=compat_id), AttachOk)
        scheduler = engine_process()
        assert isinstance(scheduler.attach(client_id, rank=None, compat_id=compat_id), AttachOk)
        return scheduler

    return attach

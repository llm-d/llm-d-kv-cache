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


import pytest

from kvpoold.server.config import ServerConfig

from ..serving import FakeClock, RunningTier
from .fake_engine import COMPAT, KV_BYTES_PER_CHUNK, FakeEngineProcess
from .helpers import attach_scheduler

POOL_CHUNKS = 8
LEASE_SECONDS = 30.0
FENCE_SECONDS = 10.0


def server_config(region_dir, **fields) -> ServerConfig:
    """The config of a pool server of ``POOL_CHUNKS`` chunks in ``region_dir``; ``fields`` override it."""
    defaults = dict(
        address=f"ipc://{region_dir}/tier.sock",
        region_dir=str(region_dir),
        pool_bytes=POOL_CHUNKS * KV_BYTES_PER_CHUNK,
        lease_seconds=LEASE_SECONDS,
        fence_seconds=FENCE_SECONDS,
    )
    return ServerConfig(**{**defaults, **fields})


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def start_tier(tmp_path, clock, recording_config):
    """Starts pool servers on one region directory, stopped after the test.
    Unless a test passes its own ``kv_events_config``, a server publishes to a
    ``RecordingPublisher``."""
    started: list[RunningTier] = []

    def start(**fields) -> RunningTier:
        fields.setdefault("kv_events_config", recording_config)
        tier = RunningTier(server_config(tmp_path, **fields), clock)
        started.append(tier)
        return tier

    yield start
    for tier in started:
        tier.stop()


@pytest.fixture
def tier(start_tier) -> RunningTier:
    return start_tier()


@pytest.fixture
def processes():
    """Makes fake engine processes connected to an address, closed after the test.
    They reach whichever instance is bound to the address."""
    made: list[FakeEngineProcess] = []

    def make(address: str) -> FakeEngineProcess:
        made.append(FakeEngineProcess(address))
        return made[-1]

    yield make
    for process in made:
        process.close()


@pytest.fixture
def engine_process(tier, processes):
    """Makes fake engine processes connected to the server's address."""
    return lambda: processes(tier.address)


@pytest.fixture
def engine(tier, processes):
    """Attaches an engine's GPU worker and then its scheduler; returns the
    scheduler's process."""

    def attach(client_id: str = "engine-a", namespace: bytes = COMPAT.namespace) -> FakeEngineProcess:
        return attach_scheduler(processes, tier.address, client_id, with_worker=True, namespace=namespace)

    return attach

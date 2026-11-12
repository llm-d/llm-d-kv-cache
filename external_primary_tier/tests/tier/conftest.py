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
from external_primary_tier.tier.server import ExternalPrimaryTier

from .fake_engine import KV_BYTES_PER_CHUNK, FakeEngineProcess

POOL_CHUNKS = 8


@pytest.fixture
def tier(tmp_path):
    """An External Primary Tier serving in a background thread."""
    address = f"ipc://{tmp_path}/tier.sock"
    server = ExternalPrimaryTier(address, str(tmp_path), pool_bytes=POOL_CHUNKS * KV_BYTES_PER_CHUNK)
    stop = threading.Event()
    thread = threading.Thread(target=server.serve, args=(stop, 20), daemon=True)
    thread.start()
    server.address = address
    yield server
    stop.set()
    thread.join(timeout=5)
    server.close()


@pytest.fixture
def engine_process(tier):
    """Makes fake engine processes connected to the tier, closed after the test."""
    made = []

    def make() -> FakeEngineProcess:
        process = FakeEngineProcess(tier.address)
        made.append(process)
        return process

    yield make
    for process in made:
        process.close()

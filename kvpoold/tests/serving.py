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

"""A pool server serving in a background thread, for the tests that talk to one."""

import threading
import time
from collections.abc import Callable

from kvpoold.server.app import Server
from kvpoold.server.config import ServerConfig

POLL_MS = 2
"""How often the serving thread checks for a stop; short, so a stop takes no time."""


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RunningTier:
    def __init__(self, config: ServerConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.address = config.address
        self.server = Server(config, clock=clock)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.server.serve, args=(self._stop, POLL_MS), daemon=True)
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

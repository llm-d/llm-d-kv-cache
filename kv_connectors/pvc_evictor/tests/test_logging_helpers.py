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

"""Disk-pressure logs use the same capacity as cleanup thresholds."""

import logging
import multiprocessing
import sys
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processes import activator  # noqa: E402
from utils import system  # noqa: E402
from utils.logging_helpers import log_aggregated_stats  # noqa: E402


def test_activator_and_status_logs_exclude_reserved_capacity(monkeypatch, caplog):
    """Show used/usable capacity consistently through activation and status logs."""
    gib = 1024**3
    readings = iter(
        [
            SimpleNamespace(f_frsize=gib, f_blocks=100, f_bfree=20, f_bavail=10),
            SimpleNamespace(f_frsize=gib, f_blocks=100, f_bfree=40, f_bavail=30),
        ]
    )
    monkeypatch.setattr(system.os, "statvfs", lambda _: next(readings))
    monkeypatch.setattr(activator, "setup_logging", lambda *args: None)
    deletion_event = multiprocessing.Event()
    shutdown_event = multiprocessing.Event()
    result_queue = SimpleQueue()
    deletion_states = []

    def stop_after_two_polls(interval):
        deletion_states.append(deletion_event.is_set())
        if len(deletion_states) == 2:
            shutdown_event.set()

    monkeypatch.setattr(activator.time, "sleep", stop_after_two_polls)
    caplog.set_level(logging.INFO)
    activator.activator_process(9, "/cache", 85.0, 70.0, 0.5, deletion_event, result_queue, shutdown_event, {})

    assert deletion_states == [True, False]
    start_message = next(record.message for record in caplog.records if record.message.startswith("DELETION_START:"))
    end_message = next(record.message for record in caplog.records if record.message.startswith("DELETION_END:"))
    assert "usage=88.89%, used=80.00GB, usable=90.00GB, total=100.00GB" in start_message
    assert "usage=66.67%, used=60.00GB, usable=90.00GB, total=100.00GB" in end_message

    stats_type, process_num, stats = result_queue.get_nowait()
    assert stats_type == "activator_stats"
    log_aggregated_stats(logging.getLogger("status"), {}, {process_num: stats}, {}, 85.0, 70.0)
    assert "PVC Usage: 88.9% (80.00GB / 90.00GB usable; total=100.00GB)" in caplog.text

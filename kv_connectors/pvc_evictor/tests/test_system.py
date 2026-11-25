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

"""Disk pressure follows space available to the cache writer."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PVC_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pvc_evictor_system", PVC_ROOT / "utils/system.py")
_system = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _system
spec.loader.exec_module(_system)


@pytest.mark.parametrize("block_size", [1024, 4096])
def test_disk_usage_excludes_reserved_blocks(monkeypatch, block_size):
    stat = SimpleNamespace(f_frsize=block_size, f_blocks=100, f_bfree=20, f_bavail=10)
    monkeypatch.setattr(_system.os, "statvfs", lambda path: stat)

    usage = _system.get_disk_usage_from_statvfs("/cache")

    assert usage.total_bytes == 100 * block_size
    assert usage.used_bytes == 80 * block_size
    assert usage.available_bytes == 10 * block_size
    assert usage.usage_percent == pytest.approx(80 / 90 * 100)


def test_disk_usage_reaches_full_when_writer_has_no_available_space(monkeypatch):
    stat = SimpleNamespace(f_frsize=4096, f_blocks=100, f_bfree=20, f_bavail=0)
    monkeypatch.setattr(_system.os, "statvfs", lambda path: stat)

    usage = _system.get_disk_usage_from_statvfs("/cache")

    assert usage.available_bytes == 0
    assert usage.usage_percent == 100


def test_disk_usage_without_reserved_blocks(monkeypatch):
    stat = SimpleNamespace(f_frsize=4096, f_blocks=100, f_bfree=20, f_bavail=20)
    monkeypatch.setattr(_system.os, "statvfs", lambda path: stat)

    assert _system.get_disk_usage_from_statvfs("/cache").usage_percent == 80


def test_empty_filesystem_does_not_divide_by_zero(monkeypatch):
    stat = SimpleNamespace(f_frsize=4096, f_blocks=0, f_bfree=0, f_bavail=0)
    monkeypatch.setattr(_system.os, "statvfs", lambda path: stat)

    assert _system.get_disk_usage_from_statvfs("/cache").usage_percent == 0


def test_activator_triggers_on_capacity_available_to_writer(monkeypatch):
    import multiprocessing

    sys.path.insert(0, str(PVC_ROOT))
    from processes import activator

    stat = SimpleNamespace(f_frsize=4096, f_blocks=100, f_bfree=20, f_bavail=10)
    monkeypatch.setattr(_system.os, "statvfs", lambda path: stat)
    monkeypatch.setattr(activator, "get_disk_usage_from_statvfs", _system.get_disk_usage_from_statvfs)
    monkeypatch.setattr(activator, "setup_logging", lambda *args: None)
    deletion_event = multiprocessing.Event()
    shutdown_event = multiprocessing.Event()
    observed = []

    def stop_after_poll(interval):
        observed.append(deletion_event.is_set())
        shutdown_event.set()

    monkeypatch.setattr(activator.time, "sleep", stop_after_poll)
    activator.activator_process(
        9, "/cache", 85.0, 70.0, 0.5, deletion_event, multiprocessing.Queue(), shutdown_event, {}
    )

    assert observed == [True]

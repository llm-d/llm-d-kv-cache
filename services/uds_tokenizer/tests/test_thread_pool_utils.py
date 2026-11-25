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

"""Unit tests for thread_pool_utils's CPU count and pool size calculations."""

from utils import thread_pool_utils


def _no_cgroup_files(path):
    return False


def test_get_cpu_count_uses_cgroup_v1_quota(monkeypatch, tmp_path):
    quota_path = tmp_path / "cpu.cfs_quota_us"
    period_path = tmp_path / "cpu.cfs_period_us"
    quota_path.write_text("200000")
    period_path.write_text("100000")
    real_open = open

    def fake_exists(path):
        return path in (
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        )

    def fake_open(path, mode="r"):
        if path == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
            return real_open(quota_path)
        return real_open(period_path)

    monkeypatch.setattr(thread_pool_utils.os.path, "exists", fake_exists)
    monkeypatch.setattr("builtins.open", fake_open)

    assert thread_pool_utils.get_cpu_count() == 2


def test_get_cpu_count_falls_back_to_multiprocessing(monkeypatch):
    monkeypatch.setattr(thread_pool_utils.os.path, "exists", _no_cgroup_files)
    monkeypatch.setattr("multiprocessing.cpu_count", lambda: 7, raising=False)

    assert thread_pool_utils.get_cpu_count() == 7


def test_get_thread_pool_size_uses_env_override(monkeypatch):
    monkeypatch.setattr(thread_pool_utils, "get_cpu_count", lambda: 4)
    monkeypatch.setenv("THREAD_POOL_SIZE", "3")

    assert thread_pool_utils.get_thread_pool_size() == 3


def test_get_thread_pool_size_computes_default_from_cpu_count(monkeypatch):
    monkeypatch.setattr(thread_pool_utils, "get_cpu_count", lambda: 4)
    monkeypatch.delenv("THREAD_POOL_SIZE", raising=False)

    assert thread_pool_utils.get_thread_pool_size(multiplier=2, max_workers=32) == 8


def test_get_thread_pool_size_caps_at_max_workers(monkeypatch):
    monkeypatch.setattr(thread_pool_utils, "get_cpu_count", lambda: 100)
    monkeypatch.delenv("THREAD_POOL_SIZE", raising=False)

    assert thread_pool_utils.get_thread_pool_size(multiplier=2, max_workers=32) == 32

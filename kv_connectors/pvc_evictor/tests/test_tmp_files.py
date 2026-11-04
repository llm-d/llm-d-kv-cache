"""Tests for stale vLLM temporary file recognition."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.tmp_files import (  # noqa: E402
    is_block_tmp_file,
    is_probe_tmp_file,
    is_stale_tmp_file,
)


class TestIsBlockTmpFile:
    @pytest.mark.parametrize(
        "name",
        [
            "abcdef0123456789.bin_0.tmp",
            "abcdef0123456789.bin_9223372036854775807.tmp",
            "ABCDEF0123456789.bin_42.tmp",
        ],
    )
    def test_matches_vllm_block_tmp_names(self, name):
        assert is_block_tmp_file(name)

    @pytest.mark.parametrize(
        "name",
        [
            "abcdef0123456789.bin",
            "abcdef0123456789.bin.tmp",
            "abcdef0123456789.bin_.tmp",
            "abcdef0123456789.bin_-1.tmp",
            "abcdef0123456789.bin_12.tmp.bak",
            "abcdef0123456789.bin_1a.tmp",
            "notahash.bin_1.tmp",
            ".bin_1.tmp",
            "foo.tmp",
            ".o_direct_probe_1.tmp",
            "config.json",
        ],
    )
    def test_rejects_other_names(self, name):
        assert not is_block_tmp_file(name)


class TestIsProbeTmpFile:
    def test_matches_probe_name(self):
        assert is_probe_tmp_file(".o_direct_probe_1234567890.tmp")

    @pytest.mark.parametrize(
        "name",
        [
            "o_direct_probe_1.tmp",
            ".o_direct_probe.tmp",
            ".o_direct_probe_1.tmp.bak",
            ".o_direct_probe_x.tmp",
            "abcdef0123456789.bin_1.tmp",
        ],
    )
    def test_rejects_other_names(self, name):
        assert not is_probe_tmp_file(name)


class TestIsStaleTmpFile:
    def _make(self, path: Path, age_seconds: float, now: float) -> Path:
        path.write_bytes(b"x")
        os.utime(path, (now - age_seconds, now - age_seconds))
        return path

    def test_old_file_is_stale(self, tmp_path: Path):
        now = 1_000_000.0
        f = self._make(tmp_path / "a.bin_1.tmp", age_seconds=3600, now=now)
        assert is_stale_tmp_file(str(f), min_age_seconds=3600, now=now)

    def test_recent_file_is_not_stale(self, tmp_path: Path):
        now = 1_000_000.0
        f = self._make(tmp_path / "a.bin_1.tmp", age_seconds=3599, now=now)
        assert not is_stale_tmp_file(str(f), min_age_seconds=3600, now=now)

    def test_uses_mtime_not_atime(self, tmp_path: Path):
        now = 1_000_000.0
        f = tmp_path / "a.bin_1.tmp"
        f.write_bytes(b"x")
        # Old access time, recent modification: still being written.
        os.utime(f, (now - 10_000, now - 10))
        assert not is_stale_tmp_file(str(f), min_age_seconds=3600, now=now)

    def test_non_positive_threshold_never_stale(self, tmp_path: Path):
        now = 1_000_000.0
        f = self._make(tmp_path / "a.bin_1.tmp", age_seconds=10_000, now=now)
        assert not is_stale_tmp_file(str(f), min_age_seconds=0, now=now)
        assert not is_stale_tmp_file(str(f), min_age_seconds=-1, now=now)

    def test_missing_file_is_not_stale(self, tmp_path: Path):
        assert not is_stale_tmp_file(str(tmp_path / "gone.bin_1.tmp"), min_age_seconds=1, now=1_000_000.0)

    def test_directory_is_not_stale(self, tmp_path: Path):
        d = tmp_path / "a.bin_1.tmp"
        d.mkdir()
        os.utime(d, (0, 0))
        assert not is_stale_tmp_file(str(d), min_age_seconds=1, now=1_000_000.0)

    def test_symlink_is_not_stale(self, tmp_path: Path):
        now = 1_000_000.0
        target = self._make(tmp_path / "target", age_seconds=10_000, now=now)
        link = tmp_path / "a.bin_1.tmp"
        link.symlink_to(target)
        assert not is_stale_tmp_file(str(link), min_age_seconds=1, now=now)

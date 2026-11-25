"""Tests for stale vLLM temp-file cleanup in the crawler."""

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from processes.crawler import (  # noqa: E402
    iter_probe_tmp_files,
    stream_cache_files_with_mapper,
    sweep_stale_tmp_files,
)

logger = logging.getLogger("test_tmp_cleanup")

NOW = 1_000_000.0
HASH = "abcdef0123456789"


def _aged(path: Path, age_seconds: float, now: float = NOW) -> Path:
    path.write_bytes(b"x")
    os.utime(path, (now - age_seconds, now - age_seconds))
    return path


@pytest.fixture
def cache_with_tmp(tmp_path: Path) -> Path:
    """A rank layout holding one real block, one stale temp file and one fresh one."""
    cache = tmp_path / "cache"
    bucket = cache / "model_abc123def456_r0" / "abc" / "de_g0"
    bucket.mkdir(parents=True)
    _aged(bucket / f"{HASH}.bin", age_seconds=99_999)
    _aged(bucket / f"{HASH}.bin_1.tmp", age_seconds=99_999)
    _aged(bucket / f"{HASH}.bin_2.tmp", age_seconds=5)
    # Must never be touched: not vLLM temp-file shapes.
    _aged(bucket / "foo.tmp", age_seconds=99_999)
    _aged(bucket / f"{HASH}.bin_3.tmp.bak", age_seconds=99_999)

    base = cache / "model_abc123def456"
    base.mkdir()
    (base / "config.json").write_text("{}")
    _aged(base / ".o_direct_probe_7.tmp", age_seconds=99_999)
    return cache


class TestStreamerReportsTmpFiles:
    def test_tmp_files_go_to_the_callback_not_the_queue(self, cache_with_tmp: Path):
        seen: list[str] = []
        yielded = list(stream_cache_files_with_mapper(cache_with_tmp, on_tmp_file_discovered=seen.append))

        assert [p.name for p in yielded] == [f"{HASH}.bin"]
        assert sorted(os.path.basename(p) for p in seen) == [f"{HASH}.bin_1.tmp", f"{HASH}.bin_2.tmp"]

    def test_without_a_callback_tmp_files_are_ignored(self, cache_with_tmp: Path):
        yielded = list(stream_cache_files_with_mapper(cache_with_tmp))
        assert [p.name for p in yielded] == [f"{HASH}.bin"]

    def test_unrelated_tmp_names_are_never_reported(self, cache_with_tmp: Path):
        seen: list[str] = []
        list(stream_cache_files_with_mapper(cache_with_tmp, on_tmp_file_discovered=seen.append))
        names = {os.path.basename(p) for p in seen}
        assert "foo.tmp" not in names
        assert f"{HASH}.bin_3.tmp.bak" not in names


class TestIterProbeTmpFiles:
    def test_finds_probe_file_beside_config_json(self, cache_with_tmp: Path):
        found = {os.path.basename(p) for p in iter_probe_tmp_files(cache_with_tmp)}
        assert found == {".o_direct_probe_7.tmp"}

    def test_finds_probe_file_at_the_cache_root(self, tmp_path: Path):
        cache = tmp_path / "cache"
        cache.mkdir()
        _aged(cache / ".o_direct_probe_1.tmp", age_seconds=10)
        assert [os.path.basename(p) for p in iter_probe_tmp_files(cache)] == [".o_direct_probe_1.tmp"]

    def test_ignores_non_probe_names(self, tmp_path: Path):
        cache = tmp_path / "cache"
        cache.mkdir()
        _aged(cache / "o_direct_probe_1.tmp", age_seconds=10)
        _aged(cache / ".o_direct_probe_x.tmp", age_seconds=10)
        assert list(iter_probe_tmp_files(cache)) == []

    def test_missing_cache_root_yields_nothing(self, tmp_path: Path):
        assert list(iter_probe_tmp_files(tmp_path / "absent")) == []


class TestSweepStaleTmpFiles:
    def _candidates(self, cache: Path) -> list[str]:
        out: list[str] = []
        list(stream_cache_files_with_mapper(cache, on_tmp_file_discovered=out.append))
        return out

    def test_deletes_stale_keeps_fresh(self, cache_with_tmp: Path):
        bucket = cache_with_tmp / "model_abc123def456_r0" / "abc" / "de_g0"
        deleted, skipped = sweep_stale_tmp_files(
            self._candidates(cache_with_tmp),
            cache_path=cache_with_tmp,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=False,
            logger=logger,
            now=NOW,
        )
        assert (deleted, skipped) == (1, 1)
        assert not (bucket / f"{HASH}.bin_1.tmp").exists()
        assert (bucket / f"{HASH}.bin_2.tmp").exists()
        # The real block and the unrelated files are untouched.
        assert (bucket / f"{HASH}.bin").exists()
        assert (bucket / "foo.tmp").exists()
        assert (bucket / f"{HASH}.bin_3.tmp.bak").exists()

    def test_runs_without_any_deletion_event(self, cache_with_tmp: Path):
        """The sweep takes no deletion event: temp files go whatever the disk usage."""
        deleted, _ = sweep_stale_tmp_files(
            self._candidates(cache_with_tmp),
            cache_path=cache_with_tmp,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=False,
            logger=logger,
            now=NOW,
        )
        assert deleted == 1

    def test_dry_run_counts_but_keeps_the_file(self, cache_with_tmp: Path):
        bucket = cache_with_tmp / "model_abc123def456_r0" / "abc" / "de_g0"
        deleted, _ = sweep_stale_tmp_files(
            self._candidates(cache_with_tmp),
            cache_path=cache_with_tmp,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=True,
            logger=logger,
            now=NOW,
        )
        assert deleted == 1
        assert (bucket / f"{HASH}.bin_1.tmp").exists()

    @pytest.mark.parametrize("min_age", [0, -1])
    def test_non_positive_age_is_a_no_op(self, cache_with_tmp: Path, min_age):
        bucket = cache_with_tmp / "model_abc123def456_r0" / "abc" / "de_g0"
        assert sweep_stale_tmp_files(
            self._candidates(cache_with_tmp),
            cache_path=cache_with_tmp,
            min_age_seconds=min_age,
            batch_size=100,
            dry_run=False,
            logger=logger,
            now=NOW,
        ) == (0, 0)
        assert (bucket / f"{HASH}.bin_1.tmp").exists()

    def test_age_boundary_is_inclusive(self, tmp_path: Path):
        cache = tmp_path / "cache"
        bucket = cache / "m_r0" / "abc" / "de_g0"
        bucket.mkdir(parents=True)
        exactly = _aged(bucket / f"{HASH}.bin_1.tmp", age_seconds=3600)
        deleted, skipped = sweep_stale_tmp_files(
            [str(exactly)],
            cache_path=cache,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=False,
            logger=logger,
            now=NOW,
        )
        assert (deleted, skipped) == (1, 0)

    def test_probe_files_only_for_the_owning_crawler(self, cache_with_tmp: Path):
        probe = cache_with_tmp / "model_abc123def456" / ".o_direct_probe_7.tmp"
        # Not the owner: the probe file is left for the crawler that owns it.
        sweep_stale_tmp_files(
            [],
            cache_path=cache_with_tmp,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=False,
            logger=logger,
            include_probe_files=False,
            now=NOW,
        )
        assert probe.exists()
        # The owner removes it.
        deleted, _ = sweep_stale_tmp_files(
            [],
            cache_path=cache_with_tmp,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=False,
            logger=logger,
            include_probe_files=True,
            now=NOW,
        )
        assert deleted == 1
        assert not probe.exists()

    def test_batches_larger_than_the_batch_size(self, tmp_path: Path):
        cache = tmp_path / "cache"
        bucket = cache / "m_r0" / "abc" / "de_g0"
        bucket.mkdir(parents=True)
        paths = [str(_aged(bucket / f"{HASH}.bin_{i}.tmp", age_seconds=99_999)) for i in range(7)]
        deleted, _ = sweep_stale_tmp_files(
            paths,
            cache_path=cache,
            min_age_seconds=3600,
            batch_size=2,
            dry_run=False,
            logger=logger,
            now=NOW,
        )
        assert deleted == 7
        assert not any(Path(p).exists() for p in paths)

    def test_bucket_left_empty_becomes_reapable(self, tmp_path: Path):
        """A bucket holding only a stale temp file can be rmdir'd once it is gone."""
        cache = tmp_path / "cache"
        bucket = cache / "m_r0" / "abc" / "de_g0"
        bucket.mkdir(parents=True)
        _aged(bucket / f"{HASH}.bin_1.tmp", age_seconds=99_999)
        assert list(bucket.iterdir())  # not empty, so the folder cleaner cannot reap it

        sweep_stale_tmp_files(
            [str(bucket / f"{HASH}.bin_1.tmp")],
            cache_path=cache,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=False,
            logger=logger,
            now=NOW,
        )
        assert list(bucket.iterdir()) == []
        bucket.rmdir()  # what the folder cleaner does on a later pass

    def test_vanished_candidate_is_not_counted(self, tmp_path: Path):
        cache = tmp_path / "cache"
        cache.mkdir()
        assert sweep_stale_tmp_files(
            [str(cache / f"{HASH}.bin_1.tmp")],
            cache_path=cache,
            min_age_seconds=3600,
            batch_size=100,
            dry_run=False,
            logger=logger,
            now=NOW,
        ) == (0, 1)


class TestConfig:
    def test_tmp_cleanup_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv("ENABLE_TMP_CLEANUP", raising=False)
        monkeypatch.delenv("TMP_FILE_MIN_AGE_MINUTES", raising=False)
        config = Config.from_env()
        assert config.enable_tmp_cleanup is False
        assert config.tmp_file_min_age_minutes == 1440.0
        config.validate()

    def test_env_enables_it(self, monkeypatch):
        monkeypatch.setenv("ENABLE_TMP_CLEANUP", "true")
        monkeypatch.setenv("TMP_FILE_MIN_AGE_MINUTES", "60")
        config = Config.from_env()
        assert config.enable_tmp_cleanup is True
        assert config.tmp_file_min_age_minutes == 60.0
        config.validate()

    def test_knobs_reach_the_worker_processes(self, monkeypatch):
        monkeypatch.setenv("ENABLE_TMP_CLEANUP", "true")
        monkeypatch.setenv("TMP_FILE_MIN_AGE_MINUTES", "30")
        as_dict = Config.from_env().to_dict()
        assert as_dict["enable_tmp_cleanup"] is True
        assert as_dict["tmp_file_min_age_minutes"] == 30.0

    @pytest.mark.parametrize("value", ["0", "-5"])
    def test_non_positive_age_is_rejected_when_enabled(self, monkeypatch, value):
        monkeypatch.setenv("ENABLE_TMP_CLEANUP", "true")
        monkeypatch.setenv("TMP_FILE_MIN_AGE_MINUTES", value)
        with pytest.raises(ValueError, match="TMP_FILE_MIN_AGE_MINUTES"):
            Config.from_env().validate()

    @pytest.mark.parametrize("value", ["0", "-5"])
    def test_non_positive_age_is_allowed_when_disabled(self, monkeypatch, value):
        monkeypatch.setenv("ENABLE_TMP_CLEANUP", "false")
        monkeypatch.setenv("TMP_FILE_MIN_AGE_MINUTES", value)
        Config.from_env().validate()

"""Unit tests for crawler path discovery (post-#611 layout walker)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processes.crawler import (  # noqa: E402
    get_hex_modulo_ranges,
    hex_to_int,
    stream_cache_files_with_mapper,
)


class TestHexHelpers:
    def test_hex_to_int_valid(self):
        assert hex_to_int("abc") == 0xABC
        assert hex_to_int("000") == 0

    def test_hex_to_int_invalid(self):
        assert hex_to_int("not_hex") is None


class TestStreamCacheFilesWithMapper:
    def test_stream_new_layout_yields_bins(self, new_layout_cache: Path):
        paths = sorted(stream_cache_files_with_mapper(new_layout_cache))
        names = {p.name for p in paths}
        assert names == {"block1.bin", "block2.bin", "other.bin", "rank1.bin"}
        assert "skip.bin" not in names

    def test_skips_base_dir_without_rank_suffix(self, new_layout_cache: Path):
        paths = list(stream_cache_files_with_mapper(new_layout_cache))
        assert all("config.json" not in str(p) for p in paths)
        assert all(p.suffix == ".bin" for p in paths)

    def test_hex_modulo_filter(self, new_layout_cache: Path):
        # abc: 0xabc % 16 == 12; def: 0xdef % 16 == 15; 000: % 16 == 0
        paths_mod_12 = list(stream_cache_files_with_mapper(new_layout_cache, (12, 12)))
        assert {p.name for p in paths_mod_12} == {"block1.bin", "block2.bin"}

        paths_mod_0 = list(stream_cache_files_with_mapper(new_layout_cache, (0, 0)))
        assert {p.name for p in paths_mod_0} == {"rank1.bin"}

    def test_multiple_ranks(self, new_layout_cache: Path):
        paths = list(stream_cache_files_with_mapper(new_layout_cache))
        rank_dirs = {p.parts[-4] for p in paths}
        assert "model_abc123def456_r0" in rank_dirs
        assert "model_abc123def456_r1" in rank_dirs

    def test_missing_cache_path(self, tmp_path: Path):
        missing = tmp_path / "missing"
        assert list(stream_cache_files_with_mapper(missing)) == []

    def test_nested_rank_dir_is_discovered(self, tmp_path: Path):
        """Rank dirs may sit under extra prefixes; _iter_rank_dirs walks recursively."""
        cache = tmp_path / "cache"
        rank = cache / "extra" / "nested" / "model_abc123def456_r0"
        (rank / "abc" / "de_g0").mkdir(parents=True)
        (rank / "abc" / "de_g0" / "nested.bin").write_bytes(b"n")

        paths = list(stream_cache_files_with_mapper(cache))
        assert {p.name for p in paths} == {"nested.bin"}

    def test_malformed_first_level_bucket_skipped(self, new_layout_cache: Path):
        paths = list(stream_cache_files_with_mapper(new_layout_cache))
        assert all(p.name != "skip.bin" for p in paths)


class TestHexModuloRanges:
    def test_eight_processes(self):
        ranges = get_hex_modulo_ranges(8)
        assert len(ranges) == 8
        assert ranges[0] == (0, 1)
        assert ranges[7] == (14, 15)

    def test_invalid_count_raises(self):
        with pytest.raises(ValueError):
            get_hex_modulo_ranges(3)

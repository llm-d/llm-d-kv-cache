"""Unit tests for crawler path discovery (flat fs_backend layout)."""

import sys
from pathlib import Path

import pytest

# Import crawler module from pvc_evictor root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processes.crawler import (  # noqa: E402
    stream_cache_files,
    is_rank_dir,
    is_hex3_dir,
    is_group_dir,
    hex_mod_in_range,
    get_hex_modulo_ranges,
)


class TestPathHelpers:
    def test_is_rank_dir(self):
        assert is_rank_dir("meta-llama_Model_abc123def456_r0")
        assert is_rank_dir("model_r12")
        assert not is_rank_dir("model_abc123def456")
        assert not is_rank_dir("block_size_16_blocks_per_file_256")

    def test_is_hex3_dir(self):
        assert is_hex3_dir("abc")
        assert is_hex3_dir("000")
        assert not is_hex3_dir("ab")
        assert not is_hex3_dir("not_hex")

    def test_is_group_dir(self):
        assert is_group_dir("de_g0")
        assert is_group_dir("01_g1")
        assert is_group_dir("AB_g99")
        assert not is_group_dir("de")
        assert not is_group_dir("bad_g0_extra")

    def test_hex_mod_in_range(self):
        # abc -> 0xabc = 2748, 2748 % 16 = 12
        assert hex_mod_in_range("abc", 12, 12)
        assert not hex_mod_in_range("abc", 0, 7)


class TestStreamCacheFiles:
    def test_stream_new_layout_yields_bins(self, new_layout_cache: Path):
        paths = sorted(stream_cache_files(new_layout_cache))
        names = {p.name for p in paths}
        assert names == {"block1.bin", "block2.bin", "other.bin", "rank1.bin"}
        assert "skip.bin" not in names

    def test_skips_config_json_and_base_dir(self, new_layout_cache: Path):
        paths = list(stream_cache_files(new_layout_cache))
        assert all("config.json" not in str(p) for p in paths)
        assert all("_r" in str(p) for p in paths)

    def test_hex_modulo_filter(self, new_layout_cache: Path):
        # abc: mod 12; def: 0xdef=3567, mod 15; 000: mod 0
        paths_mod_12 = list(stream_cache_files(new_layout_cache, (12, 12)))
        assert {p.name for p in paths_mod_12} == {"block1.bin", "block2.bin"}

        paths_mod_0 = list(stream_cache_files(new_layout_cache, (0, 0)))
        assert {p.name for p in paths_mod_0} == {"rank1.bin"}

    def test_multiple_ranks(self, new_layout_cache: Path):
        paths = list(stream_cache_files(new_layout_cache))
        ranks = {p.parts[-4] for p in paths}
        assert "model_abc123def456_r0" in ranks
        assert "model_abc123def456_r1" in ranks

    def test_empty_cache_path(self, tmp_path: Path):
        empty = tmp_path / "missing"
        assert list(stream_cache_files(empty)) == []

    def test_malformed_dirs_ignored(self, new_layout_cache: Path):
        paths = list(stream_cache_files(new_layout_cache))
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

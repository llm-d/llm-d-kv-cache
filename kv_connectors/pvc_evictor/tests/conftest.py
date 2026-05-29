"""Pytest fixtures for PVC evictor tests."""

import pytest
from pathlib import Path


@pytest.fixture
def new_layout_cache(tmp_path: Path) -> Path:
    """
    Minimal flat fs_backend layout:
      model_abc123def456/          (base + config.json only)
      model_abc123def456_r0/abc/de_g0/block1.bin
      model_abc123def456_r0/abc/de_g0/block2.bin
      model_abc123def456_r0/def/01_g1/other.bin
      model_abc123def456_r1/000/00_g0/rank1.bin
    """
    cache = tmp_path / "cache"
    cache.mkdir()

    base = cache / "model_abc123def456"
    base.mkdir()
    (base / "config.json").write_text("{}")

    rank0 = cache / "model_abc123def456_r0"
    (rank0 / "abc" / "de_g0").mkdir(parents=True)
    (rank0 / "abc" / "de_g0" / "block1.bin").write_bytes(b"x")
    (rank0 / "abc" / "de_g0" / "block2.bin").write_bytes(b"y")
    (rank0 / "def" / "01_g1").mkdir(parents=True)
    (rank0 / "def" / "01_g1" / "other.bin").write_bytes(b"z")

    rank1 = cache / "model_abc123def456_r1"
    (rank1 / "000" / "00_g0").mkdir(parents=True)
    (rank1 / "000" / "00_g0" / "rank1.bin").write_bytes(b"r")

    (rank0 / "not_hex" / "bad_g0").mkdir(parents=True)
    (rank0 / "not_hex" / "bad_g0" / "skip.bin").write_bytes(b"s")

    return cache

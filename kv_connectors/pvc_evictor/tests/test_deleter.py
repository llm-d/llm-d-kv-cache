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

import logging
import os
import sys
from pathlib import Path

import pytest

# Add pvc_evictor root to path so imports work without installing
PVC_EVICTOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PVC_EVICTOR_ROOT))

from processes.deleter import cleanup_empty_dirs, delete_batch


@pytest.fixture
def cache_tree(tmp_path):
    """Create a realistic FileMapper cache directory structure.

    Layout:
        <tmp>/cache/model_abc123_r0/0a1/0a_g0/block_0.bin
        <tmp>/cache/model_abc123_r0/0a1/0a_g0/block_1.bin
        <tmp>/cache/model_abc123_r0/0a1/0b_g1/block_2.bin
        <tmp>/cache/model_abc123_r0/ff0/ff_g0/block_3.bin
    """
    cache_path = tmp_path / "cache"
    dirs = [
        cache_path / "model_abc123_r0" / "0a1" / "0a_g0",
        cache_path / "model_abc123_r0" / "0a1" / "0b_g1",
        cache_path / "model_abc123_r0" / "ff0" / "ff_g0",
    ]
    for d in dirs:
        d.mkdir(parents=True)

    files = [
        dirs[0] / "block_0.bin",
        dirs[0] / "block_1.bin",
        dirs[1] / "block_2.bin",
        dirs[2] / "block_3.bin",
    ]
    for f in files:
        f.write_bytes(b"\x00" * 128)

    return cache_path, files


class TestCleanupEmptyDirs:
    def test_removes_leaf_and_parent_dirs(self, cache_tree):
        cache_path, files = cache_tree
        # Delete all files in 0a_g0
        for f in files[:2]:
            f.unlink()

        logger = logging.getLogger("test")
        removed = cleanup_empty_dirs(
            [str(f) for f in files[:2]], cache_path, logger
        )

        assert removed > 0
        # 0a_g0 should be gone (was the only directory with those files)
        assert not (cache_path / "model_abc123_r0" / "0a1" / "0a_g0").exists()
        # 0a1 should still exist because 0b_g1 still has a file
        assert (cache_path / "model_abc123_r0" / "0a1").exists()

    def test_removes_entire_branch_when_empty(self, cache_tree):
        cache_path, files = cache_tree
        # Delete the only file under ff0/ff_g0
        files[3].unlink()

        logger = logging.getLogger("test")
        removed = cleanup_empty_dirs([str(files[3])], cache_path, logger)

        # ff_g0, ff0 should both be removed (both empty after deletion)
        assert not (cache_path / "model_abc123_r0" / "ff0" / "ff_g0").exists()
        assert not (cache_path / "model_abc123_r0" / "ff0").exists()
        # model_abc123_r0 should still exist (0a1 branch still has files)
        assert (cache_path / "model_abc123_r0").exists()
        assert removed >= 2

    def test_stops_at_cache_root(self, cache_tree):
        cache_path, files = cache_tree
        # Delete all files
        for f in files:
            f.unlink()

        logger = logging.getLogger("test")
        cleanup_empty_dirs([str(f) for f in files], cache_path, logger)

        # cache_path itself must not be removed
        assert cache_path.exists()

    def test_no_error_on_nonempty_dirs(self, cache_tree):
        cache_path, files = cache_tree
        # Only delete one of two files in 0a_g0
        files[0].unlink()

        logger = logging.getLogger("test")
        removed = cleanup_empty_dirs([str(files[0])], cache_path, logger)

        # 0a_g0 still has block_1.bin so nothing should be removed
        assert removed == 0
        assert (cache_path / "model_abc123_r0" / "0a1" / "0a_g0").exists()

    def test_no_error_on_already_removed_dirs(self, cache_tree):
        cache_path, files = cache_tree
        # Delete files and manually remove directory before calling cleanup
        files[3].unlink()
        os.rmdir(str(cache_path / "model_abc123_r0" / "ff0" / "ff_g0"))

        logger = logging.getLogger("test")
        removed = cleanup_empty_dirs([str(files[3])], cache_path, logger)

        # ff_g0 was already gone, but ff0 should still be cleaned up
        assert not (cache_path / "model_abc123_r0" / "ff0").exists()
        assert removed >= 1


class TestDeleteBatchWithCleanup:
    def test_delete_batch_cleans_empty_dirs(self, cache_tree):
        cache_path, files = cache_tree
        # Delete the single file under ff0/ff_g0
        deleted, freed = delete_batch(
            [str(files[3])], dry_run=False,
            logger=logging.getLogger("test"), cache_path=cache_path,
        )

        assert deleted == 1
        assert freed > 0
        assert not (cache_path / "model_abc123_r0" / "ff0").exists()

    def test_delete_batch_dry_run_preserves_dirs(self, cache_tree):
        cache_path, files = cache_tree
        deleted, _ = delete_batch(
            [str(files[3])], dry_run=True,
            logger=logging.getLogger("test"), cache_path=cache_path,
        )

        assert deleted == 1
        # Dry run should not delete the file or directory
        assert files[3].exists()
        assert (cache_path / "model_abc123_r0" / "ff0" / "ff_g0").exists()

    def test_delete_batch_without_cache_path_skips_cleanup(self, cache_tree):
        cache_path, files = cache_tree
        deleted, freed = delete_batch(
            [str(files[3])], dry_run=False,
            logger=logging.getLogger("test"),
        )

        assert deleted == 1
        # Without cache_path, empty directories are not cleaned up
        assert (cache_path / "model_abc123_r0" / "ff0" / "ff_g0").exists()

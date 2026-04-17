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

from unittest.mock import MagicMock, patch

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.base import get_offload_block_hash

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.metadata_cache import MetadataCache


def test_metadata_cache_basic():
    """Test basic insert, contains, and remove logic."""
    cache = MetadataCache(max_entries=10)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")

    assert not cache.contains(h1)
    cache.insert(h1)
    assert cache.contains(h1)
    assert not cache.contains(h2)

    cache.remove(h1)
    assert not cache.contains(h1)


def test_metadata_cache_lru_eviction():
    """Test LRU eviction policy when cache is at capacity."""
    cache = MetadataCache(max_entries=2)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    cache.insert(h1)
    cache.insert(h2)
    assert cache.contains(h1)
    assert cache.contains(h2)

    # Insert h3, should evict h1 (since h2 was most recently accessed via contains)
    cache.insert(h3)
    assert not cache.contains(h1)
    assert cache.contains(h2)
    assert cache.contains(h3)


def test_metadata_cache_lru_update_on_contains():
    """Test that contains/reads update the LRU order."""
    cache = MetadataCache(max_entries=2)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    cache.insert(h1)
    cache.insert(h2)

    # Access h1, making it the most recently used
    assert cache.contains(h1)

    # Insert h3, should evict h2 (since h1 was recently accessed)
    cache.insert(h3)
    assert cache.contains(h1)
    assert not cache.contains(h2)
    assert cache.contains(h3)


def test_metadata_cache_batch_ops():
    """Test batch contains, insert, and remove."""
    cache = MetadataCache(max_entries=10)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    # Batch Insert
    cache.batch_insert([h1, h2])
    results = cache.batch_contains([h1, h2, h3])
    assert results == [True, True, False]

    # Batch Remove
    cache.batch_remove([h1])
    assert not cache.contains(h1)
    assert cache.contains(h2)


def test_metadata_cache_batch_ops_empty():
    """Test batch operations with empty inputs."""
    cache = MetadataCache(max_entries=10)

    assert cache.batch_contains([]) == []
    cache.batch_insert([])
    assert len(cache._cache) == 0
    cache.batch_remove([])


def test_metadata_cache_lru_update_on_batch_contains():
    """Test that batch_contains updates LRU order."""
    cache = MetadataCache(max_entries=2)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    cache.insert(h1)
    cache.insert(h2)

    # batch_contains for h1 should move it to the end (most recent)
    results = cache.batch_contains([h1])
    assert results == [True]

    # Insert h3, should evict h2 (since h1 was recently accessed via batch_contains)
    cache.insert(h3)
    assert cache.contains(h1)
    assert not cache.contains(h2)
    assert cache.contains(h3)


def test_manager_lookup_fallthrough():
    """Test manager lookup with cache hits and FS fallthrough."""
    file_mapper = MagicMock(spec=FileMapper, model_name="test-model")
    file_mapper.get_file_name.return_value = "/tmp/fake_path"
    file_mapper.model_name = "test-model"
    manager = SharedStorageOffloadingManager(file_mapper, metadata_cache_max_entries=10)

    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    req_context = MagicMock()

    patch_hash = patch(
        "llmd_fs_backend.manager.get_offload_block_hash",
        side_effect=lambda k: k,
    )
    with patch_hash, patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        # First lookup: H1 should hit FS and be cached
        assert manager.lookup(h1, req_context) is True
        assert mock_exists.call_count == 1
        assert manager.cache.contains(h1)

        # Second lookup: H1 should hit cache, FS should NOT be called
        assert manager.lookup(h1, req_context) is True
        assert mock_exists.call_count == 1

        # Lookup with H2 (not in cache)
        assert manager.lookup(h2, req_context) is True
        assert mock_exists.call_count == 2
        assert manager.cache.contains(h2)


def test_manager_complete_store_success():
    """Test that complete_store with success=True inserts into cache."""
    file_mapper = MagicMock(spec=FileMapper)
    file_mapper.model_name = "test-model"
    manager = SharedStorageOffloadingManager(file_mapper, metadata_cache_max_entries=10)
    h1 = BlockHash(b"hash1111")
    req_context = MagicMock()

    assert not manager.cache.contains(h1)
    patch_hash = patch(
        "llmd_fs_backend.manager.get_offload_block_hash",
        side_effect=lambda k: k,
    )
    with patch_hash:
        manager.complete_store([h1], req_context, success=True)
    assert manager.cache.contains(h1)


def test_manager_complete_store_failure():
    """Test that complete_store with success=False does not insert into cache."""
    file_mapper = MagicMock(spec=FileMapper)
    file_mapper.model_name = "test-model"
    manager = SharedStorageOffloadingManager(file_mapper, metadata_cache_max_entries=10)
    h1 = BlockHash(b"hash1111")
    req_context = MagicMock()

    assert not manager.cache.contains(h1)
    patch_hash = patch(
        "llmd_fs_backend.manager.get_offload_block_hash",
        side_effect=lambda k: k,
    )
    with patch_hash:
        manager.complete_store([h1], req_context, success=False)
    assert not manager.cache.contains(h1)


def test_manager_lookup_cache_disabled():
    """Test that the manager works correctly when the cache is disabled (None)."""
    file_mapper = MagicMock(spec=FileMapper)
    file_mapper.get_file_name.return_value = "/tmp/fake_path"
    file_mapper.model_name = "test-model"
    manager = SharedStorageOffloadingManager(file_mapper, metadata_cache_max_entries=0)

    assert manager.cache is None
    h1 = BlockHash(b"hash1111")
    req_context = MagicMock()

    patch_hash = patch(
        "llmd_fs_backend.manager.get_offload_block_hash",
        side_effect=lambda k: k,
    )
    with patch_hash, patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        # Lookup should work and call FS directly
        assert manager.lookup(h1, req_context) is True
        assert mock_exists.call_count == 1

        # Second lookup should STILL call FS directly (no caching)
        assert manager.lookup(h1, req_context) is True
        assert mock_exists.call_count == 2


def test_metadata_cache_ttl_expiration():
    """Test that positive cache hits expire correctly based on TTL."""
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")

    # Set TTL to 5 seconds
    cache = MetadataCache(max_entries=10, metadata_cache_ttl_secs=5.0)

    now = 100.0
    with patch("time.monotonic", return_value=now):
        cache.insert(h1)
        cache.insert(h2)

    # 2 seconds pass: still active
    with patch("time.monotonic", return_value=now + 2.0):
        assert cache.contains(h1)
        assert cache.contains(h2)

    # 6 seconds pass (from start): expired
    with patch("time.monotonic", return_value=now + 6.0):
        assert not cache.contains(h1)
        # Verify it was pruned
        assert (
            len(cache._cache) == 1
        )  # h1 got popped on contains lookup, h2 still there but will expire on query
        assert not cache.contains(h2)
        assert len(cache._cache) == 0


def test_metadata_cache_ttl_batch_expiration():
    """Test that batch_contains correctly processes TTL expiration."""
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    cache = MetadataCache(max_entries=10, metadata_cache_ttl_secs=5.0)

    now = 200.0
    with patch("time.monotonic", return_value=now):
        cache.batch_insert([h1, h2])

    # 3 seconds pass: insert h3
    with patch("time.monotonic", return_value=now + 3.0):
        cache.insert(h3)

    # 6 seconds pass (from start): h1 and h2 expired, h3 still active
    with patch("time.monotonic", return_value=now + 6.0):
        results = cache.batch_contains([h1, h2, h3])
        assert results == [False, False, True]
        # h1 and h2 should be pruned on lookup
        assert list(cache._cache.keys()) == [h3]


def test_manager_respects_metadata_cache_ttl():
    """Test that the manager instantiates the cache with the configured TTL."""
    file_mapper = MagicMock(spec=FileMapper)
    file_mapper.model_name = "test-model"
    manager = SharedStorageOffloadingManager(
        file_mapper,
        metadata_cache_max_entries=10,
        metadata_cache_ttl_secs=15,
    )
    assert manager.cache is not None
    assert manager.cache.metadata_cache_ttl_secs == 15


def test_metadata_cache_ttl_not_refreshed_on_insert():
    """Test that subsequent inserts of the same key do NOT reset positive TTL."""
    h1 = BlockHash(b"hash1111")
    cache = MetadataCache(max_entries=10, metadata_cache_ttl_secs=5)

    now = 300.0
    with patch("time.monotonic", return_value=now):
        cache.insert(h1)

    # 3 seconds pass: attempt to overwrite/insert again
    with patch("time.monotonic", return_value=now + 3.0):
        cache.insert(h1)

    # 6 seconds pass (from start): should still expire, because the
    # timestamp was NOT updated/reset
    with patch("time.monotonic", return_value=now + 6.0):
        assert not cache.contains(h1)


def test_metadata_cache_infinite_ttl():
    """Test that positive cache entries do not expire when TTL is set to -1."""
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")

    # Set TTL to -1 (infinite)
    cache = MetadataCache(max_entries=10, metadata_cache_ttl_secs=-1)

    now = 100.0
    with patch("time.monotonic", return_value=now):
        cache.insert(h1)
        cache.batch_insert([h2])

    # Advance clock by a day (86400 seconds)
    with patch("time.monotonic", return_value=now + 86400.0):
        assert cache.contains(h1)
        results = cache.batch_contains([h2])
        assert results == [True]
        assert len(cache._cache) == 2


def test_metadata_cache_metrics_instrumentation():
    """Test that MetadataCache and Manager operations record metrics correctly."""
    with (
        patch(
            "llmd_fs_backend.manager.METADATA_CACHE_LOOKUP_DURATION"
        ) as mock_duration,
        patch("llmd_fs_backend.manager.METADATA_CACHE_LOOKUP_BLOCKS") as mock_blocks,
        patch("llmd_fs_backend.metadata_cache.METADATA_CACHE_ENTRIES") as mock_entries,
        patch(
            "llmd_fs_backend.metadata_cache.METADATA_CACHE_EVICTIONS"
        ) as mock_evictions,
    ):
        # Setup mock blocks to cache child label mocks
        label_mocks = {}

        def get_label_mock(result=None, type=None):
            key = result or type
            if key not in label_mocks:
                label_mocks[key] = MagicMock()
            return label_mocks[key]

        mock_blocks.labels.side_effect = get_label_mock
        mock_evictions.labels.side_effect = get_label_mock

        # 1. Test MetadataCache metrics (insert/remove)
        cache = MetadataCache(max_entries=2)
        h1_raw = BlockHash(b"hash1111")
        h2_raw = BlockHash(b"hash2222")
        h3_raw = BlockHash(b"hash3333")

        cache.insert(h1_raw)
        mock_entries.set.assert_called_with(1)

        cache.insert(h2_raw)
        mock_entries.set.assert_called_with(2)

        # Trigger eviction
        cache.insert(h3_raw)
        label_mocks["lru"].inc.assert_called_once()
        mock_entries.set.assert_called_with(2)

        # Remove key
        cache.remove(h2_raw)
        mock_entries.set.assert_called_with(1)

        # Reset mock calls for manager test
        mock_entries.reset_mock()
        label_mocks["lru"].reset_mock()

        # 2. Test Manager lookup metrics (mem_hit, fs_hit, miss, latency)
        file_mapper = MagicMock(spec=FileMapper)
        file_mapper.model_name = "test-model"
        manager = SharedStorageOffloadingManager(
            file_mapper, metadata_cache_max_entries=10
        )
        req_context = MagicMock()

        # We create OffloadKeys (12 bytes) so get_offload_block_hash (key[:-4])
        # extracts the correct 8-byte BlockHash views:
        k1 = b"hash1111\x00\x00\x00\x00"
        k2 = b"hash2222\x00\x00\x00\x00"
        k3 = b"hash3333\x00\x00\x00\x00"

        h1 = BlockHash(b"hash1111")
        h2 = BlockHash(b"hash2222")

        # Pre-populate cache with h1
        manager.cache.insert(h1)

        file_mapper.get_file_name.side_effect = lambda k: (
            f"/tmp/{get_offload_block_hash(k).hex()}"
        )
        with patch("os.path.exists") as mock_exists:
            # H1 is in cache, H2 is in FS, H3 is NOT in FS
            def exists_side_effect(path):
                return h2.hex() in path

            mock_exists.side_effect = exists_side_effect

            # Perform single key lookups
            assert manager.lookup(k1, req_context) is True
            assert manager.lookup(k2, req_context) is True
            assert manager.lookup(k3, req_context) is False

            # Should observe lookup duration
            assert mock_duration.observe.call_count == 3

            # Verify metric counts
            label_mocks["mem_hit"].inc.assert_called_with()
            label_mocks["fs_hit"].inc.assert_called_with()
            label_mocks["fs_miss"].inc.assert_called_with()

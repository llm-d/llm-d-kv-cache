"""Tests for deleter process helper functions and event publishing integration."""

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

PVC_ROOT = Path(__file__).resolve().parents[1]


def _load_deleter():
    """Load deleter module with stubs for utils.system."""
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    utils_system = types.ModuleType("utils.system")
    utils_system.setup_logging = lambda *args, **kwargs: None

    sentinel = object()
    saved = {}
    for name in ["utils", "utils.system"]:
        saved[name] = sys.modules.get(name, sentinel)

    sys.modules["utils"] = utils_pkg
    sys.modules["utils.system"] = utils_system

    try:
        spec = importlib.util.spec_from_file_location(
            "processes.deleter",
            PVC_ROOT / "processes" / "deleter.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for name, previous in saved.items():
            if previous is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    return mod


_deleter = _load_deleter()
extract_block_hash = _deleter.extract_block_hash
extract_model_name = _deleter.extract_model_name
delete_file_batch = _deleter.delete_file_batch


# -- extract_block_hash tests --


def test_extract_block_hash_valid_path():
    path = "/cache/model/block_size_16/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin"
    assert extract_block_hash(path) == 0xABCDEF0123456789


def test_extract_block_hash_all_zeros():
    path = "/cache/0000000000000000.bin"
    assert extract_block_hash(path) == 0


def test_extract_block_hash_max_value():
    path = "/cache/ffffffffffffffff.bin"
    assert extract_block_hash(path) == 0xFFFFFFFFFFFFFFFF


def test_extract_block_hash_not_bin_extension():
    assert extract_block_hash("/cache/abcdef0123456789.txt") is None


def test_extract_block_hash_no_extension():
    assert extract_block_hash("/cache/abcdef0123456789") is None


def test_extract_block_hash_wrong_hex_length_short():
    assert extract_block_hash("/cache/abcdef.bin") is None


def test_extract_block_hash_wrong_hex_length_long():
    assert extract_block_hash("/cache/abcdef01234567890.bin") is None


def test_extract_block_hash_invalid_hex_chars():
    assert extract_block_hash("/cache/ghijklmnopqrstuv.bin") is None


def test_extract_block_hash_directory_path():
    assert extract_block_hash("/cache/abc/de/") is None


# -- extract_model_name tests --


def test_extract_model_name_simple():
    cache_path = "/kv-cache/models"
    file_path = f"{cache_path}/my-model/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin"
    assert extract_model_name(file_path, cache_path) == "my-model"


def test_extract_model_name_hf_style_with_slash():
    cache_path = "/kv-cache/models"
    file_path = f"{cache_path}/meta-llama/Llama-3.1-8B/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin"
    assert extract_model_name(file_path, cache_path) == os.sep.join(
        ["meta-llama", "Llama-3.1-8B"]
    )


def test_extract_model_name_deeply_nested_org():
    cache_path = "/kv-cache/models"
    file_path = f"{cache_path}/org/sub/model/block_size_32_blocks_per_file_2/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin"
    assert extract_model_name(file_path, cache_path) == os.sep.join(
        ["org", "sub", "model"]
    )


def test_extract_model_name_no_block_size_marker():
    cache_path = "/kv-cache/models"
    file_path = f"{cache_path}/my-model/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin"
    assert extract_model_name(file_path, cache_path) is None


def test_extract_model_name_block_size_at_root():
    cache_path = "/kv-cache/models"
    file_path = f"{cache_path}/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin"
    assert extract_model_name(file_path, cache_path) is None


def test_extract_model_name_file_outside_cache_path():
    assert extract_model_name("/other/path/file.bin", "/kv-cache/models") is None


# -- delete_file_batch event publishing integration --


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item, timeout=None):
        self.items.append(item)


class RecordingPublisher:
    def __init__(self):
        self.calls = []

    def publish_blocks_removed(self, block_hashes, model_name=None):
        self.calls.append((model_name, list(block_hashes)))


class ExplodingPublisher:
    def publish_blocks_removed(self, block_hashes, model_name=None):
        raise RuntimeError("publish failed")


def test_delete_file_batch_publishes_events_grouped_by_model(monkeypatch):
    cache_path = "/kv-cache/models"
    files = [
        f"{cache_path}/meta-llama/Llama-3.1-8B/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin",
        f"{cache_path}/meta-llama/Llama-3.1-8B/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/123/45/1234567890abcdef.bin",
        f"{cache_path}/other-model/block_size_32_blocks_per_file_1/tp_1/rank_0/float16/fed/cb/fedcba9876543210.bin",
    ]

    publisher = RecordingPublisher()
    logger = logging.getLogger("test_deleter")
    queue = FakeQueue()

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log: (len(paths), 1024))

    delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, publisher, cache_path
    )

    assert len(publisher.calls) == 2
    publisher.calls.sort(key=lambda x: x[0])

    model1, hashes1 = publisher.calls[0]
    assert model1 == os.sep.join(["meta-llama", "Llama-3.1-8B"])
    assert len(hashes1) == 2
    assert 0xABCDEF0123456789 in hashes1
    assert 0x1234567890ABCDEF in hashes1

    model2, hashes2 = publisher.calls[1]
    assert model2 == "other-model"
    assert hashes2 == [0xFEDCBA9876543210]


def test_delete_file_batch_no_events_when_publisher_is_none(monkeypatch):
    cache_path = "/kv-cache/models"
    files = [
        f"{cache_path}/model/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin",
    ]
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log: (1, 512))

    total_deleted, total_freed, _ = delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, None, None
    )

    assert total_deleted == 1
    assert total_freed == 512


def test_delete_file_batch_no_events_when_nothing_deleted(monkeypatch):
    cache_path = "/kv-cache/models"
    files = [
        f"{cache_path}/model/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin",
    ]

    publisher = RecordingPublisher()
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log: (0, 0))

    delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, publisher, cache_path
    )

    assert publisher.calls == []


def test_delete_file_batch_publish_failure_is_fail_open(monkeypatch):
    cache_path = "/kv-cache/models"
    files = [
        f"{cache_path}/model/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin",
    ]

    publisher = ExplodingPublisher()
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log: (1, 512))

    total_deleted, total_freed, _ = delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, publisher, cache_path
    )

    assert total_deleted == 1
    assert total_freed == 512


def test_delete_file_batch_skips_unparseable_paths(monkeypatch):
    cache_path = "/kv-cache/models"
    files = [
        f"{cache_path}/model/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin",
        "/some/random/file.txt",
        f"{cache_path}/model/block_size_16_blocks_per_file_1/tp_1/rank_0/float16/no-hash-dir/bad.bin",
    ]

    publisher = RecordingPublisher()
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log: (len(paths), 1024))

    delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, publisher, cache_path
    )

    assert len(publisher.calls) == 1
    model, hashes = publisher.calls[0]
    assert model == "model"
    assert hashes == [0xABCDEF0123456789]

"""Tests for deleter process helper functions and event publishing integration."""

import importlib.util
import json
import logging
import os
import sys
import threading
import types
from pathlib import Path

import pytest

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


@pytest.fixture(autouse=True)
def _clear_model_name_cache():
    _deleter._model_name_cache.clear()
    yield
    _deleter._model_name_cache.clear()


def _write_config(cache_path: Path, base_dir_name: str, model_name: str) -> None:
    base_dir = cache_path / base_dir_name
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "config.json").write_text(json.dumps({"model_name": model_name}))


# -- extract_block_hash tests --


def test_extract_block_hash_valid_path():
    path = "/cache/my-model_abc123def456_r0/abc/de_g0/abcdef0123456789.bin"
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


def test_extract_model_name_simple(tmp_path):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "my-model_abcdef012345", "my-model")

    file_path = str(cache_path / "my-model_abcdef012345_r0/abc/de_g0/abcdef0123456789.bin")
    assert extract_model_name(file_path, str(cache_path)) == "my-model"


def test_extract_model_name_hf_style(tmp_path):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "meta-llama_Llama-3.1-8B_fedcba987654", "meta-llama/Llama-3.1-8B")

    file_path = str(cache_path / "meta-llama_Llama-3.1-8B_fedcba987654_r0/abc/de_g0/abcdef0123456789.bin")
    assert extract_model_name(file_path, str(cache_path)) == "meta-llama/Llama-3.1-8B"


def test_extract_model_name_no_rank_suffix():
    assert extract_model_name("/cache/some_dir/abc/de_g0/hash.bin", "/cache") is None


def test_extract_model_name_no_config_json(tmp_path):
    cache_path = tmp_path / "models"
    file_path = str(cache_path / "no-config_aaaaaaaaaaaa_r0/abc/de_g0/hash.bin")
    assert extract_model_name(file_path, str(cache_path)) is None


def test_extract_model_name_too_few_path_components():
    assert extract_model_name("/cache/file.bin", "/cache") is None


def test_extract_model_name_caches_result(tmp_path):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "cached-model_abcdef012345", "cached-model")

    file_a = str(cache_path / "cached-model_abcdef012345_r0/abc/de_g0/aaaaaaaaaaaaaaaa.bin")
    file_b = str(cache_path / "cached-model_abcdef012345_r0/fff/ff_g0/ffffffffffffffff.bin")

    assert extract_model_name(file_a, str(cache_path)) == "cached-model"
    # Delete config.json — second call should still work from cache
    (cache_path / "cached-model_abcdef012345" / "config.json").unlink()
    assert extract_model_name(file_b, str(cache_path)) == "cached-model"


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


def test_delete_file_batch_publishes_events_grouped_by_model(tmp_path, monkeypatch):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "meta-llama_Llama-3.1-8B_aaaaaaaaaaaa", "meta-llama/Llama-3.1-8B")
    _write_config(cache_path, "other-model_bbbbbbbbbbbb", "other-model")

    files = [
        str(cache_path / "meta-llama_Llama-3.1-8B_aaaaaaaaaaaa_r0/abc/de_g0/abcdef0123456789.bin"),
        str(cache_path / "meta-llama_Llama-3.1-8B_aaaaaaaaaaaa_r0/123/45_g0/1234567890abcdef.bin"),
        str(cache_path / "other-model_bbbbbbbbbbbb_r0/fed/cb_g0/fedcba9876543210.bin"),
    ]

    publisher = RecordingPublisher()
    logger = logging.getLogger("test_deleter")
    queue = FakeQueue()

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log, **kwargs: (len(paths), 1024, list(paths)))

    delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, event_publisher=publisher, cache_path=str(cache_path)
    )

    assert len(publisher.calls) == 2
    publisher.calls.sort(key=lambda x: x[0])

    model1, hashes1 = publisher.calls[0]
    assert model1 == "meta-llama/Llama-3.1-8B"
    assert len(hashes1) == 2
    assert 0xABCDEF0123456789 in hashes1
    assert 0x1234567890ABCDEF in hashes1

    model2, hashes2 = publisher.calls[1]
    assert model2 == "other-model"
    assert hashes2 == [0xFEDCBA9876543210]


def test_delete_file_batch_no_events_when_publisher_is_none(monkeypatch):
    files = ["/cache/model_aaa_r0/abc/de_g0/abcdef0123456789.bin"]
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log, **kwargs: (1, 512, list(paths)))

    total_deleted, total_freed, _ = delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, event_publisher=None, cache_path=None
    )

    assert total_deleted == 1
    assert total_freed == 512


def test_delete_file_batch_no_events_when_nothing_deleted(tmp_path, monkeypatch):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "model_aaaaaaaaaaaa", "model")

    files = [str(cache_path / "model_aaaaaaaaaaaa_r0/abc/de_g0/abcdef0123456789.bin")]
    publisher = RecordingPublisher()
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log, **kwargs: (0, 0, []))

    delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, event_publisher=publisher, cache_path=str(cache_path)
    )

    assert publisher.calls == []


def test_delete_file_batch_publish_failure_is_fail_open(tmp_path, monkeypatch):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "model_aaaaaaaaaaaa", "model")

    files = [str(cache_path / "model_aaaaaaaaaaaa_r0/abc/de_g0/abcdef0123456789.bin")]
    publisher = ExplodingPublisher()
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log, **kwargs: (1, 512, list(paths)))

    total_deleted, total_freed, _ = delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, event_publisher=publisher, cache_path=str(cache_path)
    )

    assert total_deleted == 1
    assert total_freed == 512


def test_delete_file_batch_skips_unparsable_paths(tmp_path, monkeypatch):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "model_aaaaaaaaaaaa", "model")

    files = [
        str(cache_path / "model_aaaaaaaaaaaa_r0/abc/de_g0/abcdef0123456789.bin"),
        "/some/random/file.txt",
        str(cache_path / "model_aaaaaaaaaaaa_r0/no-hash-dir/bad.bin"),
    ]

    publisher = RecordingPublisher()
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log, **kwargs: (len(paths), 1024, list(paths)))

    delete_file_batch(
        files, False, logger, "P1", 0, 0, None, queue, event_publisher=publisher, cache_path=str(cache_path)
    )

    assert len(publisher.calls) == 1
    model, hashes = publisher.calls[0]
    assert model == "model"
    assert hashes == [0xABCDEF0123456789]


def test_delete_file_batch_no_events_on_dry_run(tmp_path, monkeypatch):
    cache_path = tmp_path / "models"
    _write_config(cache_path, "model_aaaaaaaaaaaa", "model")

    files = [str(cache_path / "model_aaaaaaaaaaaa_r0/abc/de_g0/abcdef0123456789.bin")]
    publisher = RecordingPublisher()
    queue = FakeQueue()
    logger = logging.getLogger("test_deleter")

    monkeypatch.setattr(_deleter, "delete_batch", lambda paths, dry, log, **kwargs: (len(paths), 0, []))

    total_deleted, total_freed, _ = delete_file_batch(
        files, True, logger, "P1", 0, 0, None, queue, event_publisher=publisher, cache_path=str(cache_path)
    )

    assert total_deleted == len(files)
    assert publisher.calls == []


# -- deletion-time access-age revalidation --


@pytest.mark.parametrize("dry_run", [False, True])
def test_delete_batch_skips_candidate_refreshed_after_queueing(tmp_path, monkeypatch, dry_run):
    now = 2_000_000_000.0
    monkeypatch.setattr(_deleter.time, "time", lambda: now)
    candidate = tmp_path / "queued.bin"
    candidate.write_bytes(b"warm cache")
    os.utime(candidate, (now - 7200, now - 7200))
    queued_paths = [str(candidate)]
    os.utime(candidate, (now - 1, now - 7200))

    def unexpected_deletion(*args, **kwargs):
        pytest.fail("A refreshed candidate must not reach rm")

    monkeypatch.setattr(_deleter.subprocess, "run", unexpected_deletion)
    assert _deleter.delete_batch(queued_paths, dry_run, logging.getLogger("test")) == (0, 0, [])
    assert candidate.read_bytes() == b"warm cache"


@pytest.mark.parametrize(
    "age_seconds,threshold_minutes,deleted", [(3599, 60, False), (3600, 60, True), (5400, 120, False), (1, 0, True)]
)
def test_delete_batch_honors_configured_age(tmp_path, monkeypatch, age_seconds, threshold_minutes, deleted):
    now = 2_000_000_000.0
    monkeypatch.setattr(_deleter.time, "time", lambda: now)
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"cache")
    os.utime(candidate, (now - age_seconds, now - age_seconds))

    result = _deleter.delete_batch(
        [str(candidate)],
        False,
        logging.getLogger("test"),
        file_access_time_threshold_minutes=threshold_minutes,
    )

    assert result == ((1, 5, [str(candidate)]) if deleted else (0, 0, []))
    assert candidate.exists() is not deleted


def test_mixed_batch_only_reports_cold_files_to_events_and_folder_cleaner(tmp_path, monkeypatch):
    now = 2_000_000_000.0
    monkeypatch.setattr(_deleter.time, "time", lambda: now)
    cache = tmp_path / "models"
    _write_config(cache, "model_aaaaaaaaaaaa", "model")
    directory = cache / "model_aaaaaaaaaaaa_r0" / "abc" / "de_g0"
    directory.mkdir(parents=True)
    cold = directory / "0000000000000001.bin"
    hot = directory / "0000000000000002.bin"
    for path, age in [(cold, 7200), (hot, 10)]:
        path.write_bytes(b"cache")
        os.utime(path, (now - age, now - age))
    publisher, results = RecordingPublisher(), FakeQueue()
    folders = types.SimpleNamespace(items=[])
    folders.put_nowait = folders.items.append

    deleted, freed, _ = delete_file_batch(
        [str(cold), str(hot), str(directory / "missing.bin")],
        False,
        logging.getLogger("test"),
        "P1",
        0,
        0,
        None,
        results,
        folder_queue=folders,
        event_publisher=publisher,
        cache_path=str(cache),
    )

    assert (deleted, freed) == (1, 5)
    assert hot.exists() and not cold.exists()
    assert publisher.calls == [("model", [1])]
    assert folders.items == [str(directory)]
    assert results.items == [("progress", 1, 5)]


@pytest.mark.parametrize("flush", ["full", "partial", "shutdown"])
def test_deleter_process_passes_custom_threshold_on_every_flush(tmp_path, monkeypatch, flush):
    now = 2_000_000_000.0
    monkeypatch.setattr(_deleter.time, "time", lambda: now)
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"cache")
    # Cold under the default 60 minutes, hot under the configured 120 minutes.
    os.utime(candidate, (now - 5400, now - 5400))
    shutdown, deletion = threading.Event(), threading.Event()
    deletion.set()

    class OneCandidateQueue:
        def get(self, timeout=None):
            shutdown.set()
            return str(candidate)

        def empty(self):
            return flush == "partial"

    results = FakeQueue()
    _deleter.deleter_process(
        1,
        tmp_path,
        {
            "deletion_batch_size": 1 if flush == "full" else 100,
            "dry_run": False,
            "file_access_time_threshold_minutes": 120,
        },
        deletion,
        OneCandidateQueue(),
        results,
        shutdown,
    )

    assert candidate.exists()
    assert results.items == [("progress", 0, 0), ("done", 0, 0)]

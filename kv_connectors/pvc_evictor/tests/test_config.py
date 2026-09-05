"""Unit tests for Config.from_env()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    DEFAULT_CLEANUP_THRESHOLD,
    DEFAULT_ENABLE_DIR_CLEANUP,
    DEFAULT_NUM_CRAWLER_PROCESSES,
    Config,
)

ENV_VARS = [
    "PVC_MOUNT_PATH", "CLEANUP_THRESHOLD", "TARGET_THRESHOLD", "CACHE_DIRECTORY",
    "DRY_RUN", "LOG_LEVEL", "NUM_CRAWLER_PROCESSES", "LOGGER_INTERVAL_SECONDS",
    "FILE_QUEUE_MAXSIZE", "FILE_QUEUE_MIN_SIZE", "DELETION_BATCH_SIZE",
    "LOG_FILE_PATH", "FILE_ACCESS_TIME_THRESHOLD_MINUTES", "HEX_BUCKET_LEN",
    "STORAGE_EVENTS_ENDPOINT", "ENABLE_DIR_CLEANUP", "DIR_CLEANUP_TTL_SECONDS",
]


def test_from_env_defaults(monkeypatch):
    """With no env vars set, from_env() must fall back to the module defaults."""
    for key in ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    config = Config.from_env()
    assert config.cleanup_threshold == DEFAULT_CLEANUP_THRESHOLD
    assert config.num_crawler_processes == DEFAULT_NUM_CRAWLER_PROCESSES
    assert config.dry_run is False
    assert config.enable_dir_cleanup == DEFAULT_ENABLE_DIR_CLEANUP
    assert config.log_file_path is None


def test_from_env_overrides_and_numeric_coercion(monkeypatch):
    """Env vars override defaults, and numeric fields coerce a float-looking string."""
    monkeypatch.setenv("CLEANUP_THRESHOLD", "90")
    monkeypatch.setenv("NUM_CRAWLER_PROCESSES", "4.0")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("ENABLE_DIR_CLEANUP", "false")

    config = Config.from_env()
    assert config.cleanup_threshold == 90.0
    assert config.num_crawler_processes == 4
    assert config.dry_run is True
    assert config.enable_dir_cleanup is False

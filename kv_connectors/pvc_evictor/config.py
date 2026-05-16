"""Configuration management for PVC Evictor."""

import os
from dataclasses import dataclass

# Default configuration values
DEFAULT_PVC_MOUNT_PATH = "/kv-cache"
DEFAULT_CLEANUP_THRESHOLD = 85.0
DEFAULT_TARGET_THRESHOLD = 70.0
DEFAULT_CACHE_DIRECTORY = "kv/model-cache/models"
DEFAULT_DRY_RUN = False
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_CRAWLER_PROCESSES = 8
DEFAULT_LOGGER_INTERVAL = 0.5
DEFAULT_FILE_QUEUE_MAXSIZE = 10000
DEFAULT_FILE_QUEUE_MIN_SIZE = 1000
DEFAULT_DELETION_BATCH_SIZE = 100
DEFAULT_FILE_ACCESS_TIME_THRESHOLD_MINUTES = 60.0
DEFAULT_HEX_BUCKET_LEN = 3
DEFAULT_STORAGE_EVENTS_ENDPOINT = ""


@dataclass
class Config:
    """Configuration loaded from environment variables.
    Environment variables are used instead of CLI arguments.
    """

    pvc_mount_path: str  # Mount path of PVC in pod (default: /kv-cache)
    cleanup_threshold: float  # Disk usage % to trigger deletion (default: 85.0)
    target_threshold: float  # Disk usage % to stop deletion (default: 70.0)
    cache_directory: str  # Subdirectory within PVC containing cache files (default: kv/model-cache/models)
    dry_run: bool  # If true, simulate deletion without actually deleting files (default: false)
    log_level: str  # Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)
    num_crawler_processes: int  # P1-PN (default: 8, valid: 1, 2, 4, 8, 16)
    logger_interval: float  # P9 monitoring interval (default: 0.5s)
    file_queue_maxsize: int  # Max items in file queue (default: 10000)
    file_queue_min_size: int  # Min queue size to maintain when deletion OFF (default: 1000)
    deletion_batch_size: int  # Files per deletion batch (default: 100)
    file_access_time_threshold_minutes: float  # Skip files accessed within this time (default: 60.0 minutes)
    hex_bucket_len: int  # Number of hex chars in the first-level bucket directory (default: 3)
    # log_file_path: Optional file logging for persistent log storage and debugging
    log_file_path: str | None = None  # Optional file path to write logs to (default: None, stdout only)
    storage_events_endpoint: str = ""  # Storage events publisher endpoint

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for multiprocessing."""
        return {
            "pvc_mount_path": self.pvc_mount_path,
            "cleanup_threshold": self.cleanup_threshold,
            "target_threshold": self.target_threshold,
            "cache_directory": self.cache_directory,
            "dry_run": self.dry_run,
            "log_level": self.log_level,
            "deletion_batch_size": self.deletion_batch_size,
            "file_queue_min_size": self.file_queue_min_size,
            "file_queue_maxsize": self.file_queue_maxsize,
            "log_file_path": self.log_file_path,
            "file_access_time_threshold_minutes": self.file_access_time_threshold_minutes,
            "hex_bucket_len": self.hex_bucket_len,
            "storage_events_endpoint": self.storage_events_endpoint,
        }

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            pvc_mount_path=os.getenv("PVC_MOUNT_PATH", DEFAULT_PVC_MOUNT_PATH),
            cleanup_threshold=float(os.getenv("CLEANUP_THRESHOLD", str(DEFAULT_CLEANUP_THRESHOLD))),
            target_threshold=float(os.getenv("TARGET_THRESHOLD", str(DEFAULT_TARGET_THRESHOLD))),
            cache_directory=os.getenv("CACHE_DIRECTORY", DEFAULT_CACHE_DIRECTORY),
            dry_run=os.getenv("DRY_RUN", str(DEFAULT_DRY_RUN).lower()).lower() == "true",
            log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
            num_crawler_processes=int(float(os.getenv("NUM_CRAWLER_PROCESSES", str(DEFAULT_NUM_CRAWLER_PROCESSES)))),
            logger_interval=float(os.getenv("LOGGER_INTERVAL_SECONDS", str(DEFAULT_LOGGER_INTERVAL))),
            file_queue_maxsize=int(float(os.getenv("FILE_QUEUE_MAXSIZE", str(DEFAULT_FILE_QUEUE_MAXSIZE)))),
            file_queue_min_size=int(float(os.getenv("FILE_QUEUE_MIN_SIZE", str(DEFAULT_FILE_QUEUE_MIN_SIZE)))),
            deletion_batch_size=int(float(os.getenv("DELETION_BATCH_SIZE", str(DEFAULT_DELETION_BATCH_SIZE)))),
            log_file_path=os.getenv("LOG_FILE_PATH", None),
            file_access_time_threshold_minutes=float(
                os.getenv(
                    "FILE_ACCESS_TIME_THRESHOLD_MINUTES",
                    str(DEFAULT_FILE_ACCESS_TIME_THRESHOLD_MINUTES),
                )
            ),
            hex_bucket_len=int(float(os.getenv("HEX_BUCKET_LEN", str(DEFAULT_HEX_BUCKET_LEN)))),
            storage_events_endpoint=os.getenv(
                "STORAGE_EVENTS_ENDPOINT", str(DEFAULT_STORAGE_EVENTS_ENDPOINT)
            ),
        )

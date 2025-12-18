"""Configuration management for PVC Evictor."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration loaded from environment variables.
    
    Environment variables are used instead of CLI arguments.
    - Standard Kubernetes practice: ConfigMaps/Secrets inject env vars into containers
    - Allows easy configuration changes via ConfigMap updates without rebuilding images
    """

    pvc_mount_path: str  # Mount path of PVC in pod (default: /kv-cache)
    cleanup_threshold: float  # Disk usage % to trigger deletion (default: 85.0)
    target_threshold: float  # Disk usage % to stop deletion (default: 70.0)
    cache_directory: str  # Subdirectory within PVC containing cache files (default: kv/model-cache/models)
    dry_run: bool  # If true, simulate deletion without actually deleting files (default: false)
    log_level: str  # Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)
    timing_file_path: str  # Path for timing analysis file (default: /tmp/timing_analysis.txt, reserved for future use)
    num_crawler_processes: int  # P1-PN (default: 8, valid: 1, 2, 4, 8, 16)
    logger_interval: float  # P9 monitoring interval (default: 0.5s)
    file_queue_maxsize: int  # Max items in file queue (default: 10000)
    file_queue_min_size: (
        int  # Min queue size to maintain when deletion OFF (default: 1000)
    )
    deletion_batch_size: int  # Files per deletion batch (default: 100)
    file_access_time_threshold_minutes: (
        float  # Skip files accessed within this time (default: 60.0 minutes)
    )
    log_file_path: Optional[str] = None  # Optional file path to write logs to

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            pvc_mount_path=os.getenv("PVC_MOUNT_PATH", "/kv-cache"),
            cleanup_threshold=float(os.getenv("CLEANUP_THRESHOLD", "85.0")),
            target_threshold=float(os.getenv("TARGET_THRESHOLD", "70.0")),
            cache_directory=os.getenv("CACHE_DIRECTORY", "kv/model-cache/models"),
            dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            timing_file_path=os.getenv("TIMING_FILE_PATH", "/tmp/timing_analysis.txt"),
            num_crawler_processes=int(os.getenv("NUM_CRAWLER_PROCESSES", "8")),
            logger_interval=float(os.getenv("LOGGER_INTERVAL_SECONDS", "0.5")),
            file_queue_maxsize=int(os.getenv("FILE_QUEUE_MAXSIZE", "10000")),
            file_queue_min_size=int(os.getenv("FILE_QUEUE_MIN_SIZE", "1000")),
            deletion_batch_size=int(os.getenv("DELETION_BATCH_SIZE", "100")),
            log_file_path=os.getenv("LOG_FILE_PATH", None),  # Optional log file path
            file_access_time_threshold_minutes=float(
                os.getenv("FILE_ACCESS_TIME_THRESHOLD_MINUTES", "60.0")
            ),
        )


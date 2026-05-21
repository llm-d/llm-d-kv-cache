"""Configuration management for PVC Evictor."""

import os
import re
from dataclasses import dataclass

# Default configuration values
DEFAULT_PVC_MOUNT_PATH = "/kv-cache"
DEFAULT_CLEANUP_THRESHOLD = 85.0
DEFAULT_TARGET_THRESHOLD = 70.0
DEFAULT_CACHE_DIRECTORY = "kv/model-cache/models"
DEFAULT_DRY_RUN = False
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_CRAWLER_PROCESSES = 8
DEFAULT_NUM_DELETER_PROCESSES = 1
DEFAULT_LOGGER_INTERVAL = 0.5
DEFAULT_FILE_QUEUE_MAXSIZE = 10000
DEFAULT_FILE_QUEUE_MIN_SIZE = 1000
DEFAULT_DELETION_BATCH_SIZE = 5000
DEFAULT_FILE_ACCESS_TIME_THRESHOLD_MINUTES = 60.0
DEFAULT_ENABLE_DIR_CLEANUP = True
DEFAULT_KV_EVENTS_ENABLED = False
DEFAULT_ZMQ_ENDPOINT = "tcp://localhost:5557"
DEFAULT_ZMQ_TOPIC = "kv@"
DEFAULT_USE_LRU_EVICTION = True
DEFAULT_CONTINUOUS_CLEANUP_HOURS = 24.0
DEFAULT_CONTINUOUS_CLEANUP_INTERVAL_MIN = 60.0
DEFAULT_MOCK_DISK_TOTAL_BYTES = 0


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
    num_deleter_processes: int  # Deleter process count (default: 1)
    logger_interval: float  # P9 monitoring interval (default: 0.5s)
    file_queue_maxsize: int  # Max items in file queue (default: 10000)
    file_queue_min_size: int  # Min queue size to maintain when deletion OFF (default: 1000)
    deletion_batch_size: int  # Files per deletion batch (default: 100)
    file_access_time_threshold_minutes: float  # Skip files accessed within this time (default: 60.0 minutes)
    shard_index: int = 0  # Index of this pod/shard for multi-pod sharding (default: 0)
    total_shards: int = 1  # Total number of pods/shards across deployment (default: 1)
    enable_dir_cleanup: bool = DEFAULT_ENABLE_DIR_CLEANUP  # Enable empty directory cleanup in background
    # log_file_path: Optional file logging for persistent log storage and debugging
    log_file_path: str | None = None  # Optional file path to write logs to (default: None, stdout only)
    kv_events_enabled: bool = DEFAULT_KV_EVENTS_ENABLED
    zmq_endpoint: str = DEFAULT_ZMQ_ENDPOINT
    zmq_topic: str = DEFAULT_ZMQ_TOPIC
    use_lru_eviction: bool = DEFAULT_USE_LRU_EVICTION
    continuous_cleanup_hours: float = DEFAULT_CONTINUOUS_CLEANUP_HOURS
    continuous_cleanup_interval_minutes: float = DEFAULT_CONTINUOUS_CLEANUP_INTERVAL_MIN
    mock_disk_total_bytes: int = DEFAULT_MOCK_DISK_TOTAL_BYTES

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for multiprocessing."""
        return {
            "pvc_mount_path": self.pvc_mount_path,
            "cleanup_threshold": self.cleanup_threshold,
            "target_threshold": self.target_threshold,
            "cache_directory": self.cache_directory,
            "dry_run": self.dry_run,
            "log_level": self.log_level,
            "num_crawler_processes": self.num_crawler_processes,
            "num_deleter_processes": self.num_deleter_processes,
            "deletion_batch_size": self.deletion_batch_size,
            "file_queue_min_size": self.file_queue_min_size,
            "file_queue_maxsize": self.file_queue_maxsize,
            "log_file_path": self.log_file_path,
            "file_access_time_threshold_minutes": self.file_access_time_threshold_minutes,
            "shard_index": self.shard_index,
            "total_shards": self.total_shards,
            "enable_dir_cleanup": self.enable_dir_cleanup,
            "kv_events_enabled": self.kv_events_enabled,
            "zmq_endpoint": self.zmq_endpoint,
            "zmq_topic": self.zmq_topic,
            "use_lru_eviction": self.use_lru_eviction,
            "continuous_cleanup_hours": self.continuous_cleanup_hours,
            "continuous_cleanup_interval_minutes": self.continuous_cleanup_interval_minutes,
            "mock_disk_total_bytes": self.mock_disk_total_bytes,
        }

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        shard_index_str = os.getenv("SHARD_INDEX")
        if shard_index_str is not None:
            shard_index = int(shard_index_str)
        else:
            pod_name = os.getenv("POD_NAME", os.getenv("HOSTNAME", ""))
            match = re.search(r"-(\d+)$", pod_name)
            if match:
                shard_index = int(match.group(1))
            else:
                shard_index = 0

        total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

        return cls(
            pvc_mount_path=os.getenv("PVC_MOUNT_PATH", DEFAULT_PVC_MOUNT_PATH),
            cleanup_threshold=float(os.getenv("CLEANUP_THRESHOLD", str(DEFAULT_CLEANUP_THRESHOLD))),
            target_threshold=float(os.getenv("TARGET_THRESHOLD", str(DEFAULT_TARGET_THRESHOLD))),
            cache_directory=os.getenv("CACHE_DIRECTORY", DEFAULT_CACHE_DIRECTORY),
            dry_run=os.getenv("DRY_RUN", str(DEFAULT_DRY_RUN).lower()).lower() == "true",
            log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
            num_crawler_processes=int(os.getenv("NUM_CRAWLER_PROCESSES", str(DEFAULT_NUM_CRAWLER_PROCESSES))),
            num_deleter_processes=int(os.getenv("NUM_DELETER_PROCESSES", str(DEFAULT_NUM_DELETER_PROCESSES))),
            logger_interval=float(os.getenv("LOGGER_INTERVAL_SECONDS", str(DEFAULT_LOGGER_INTERVAL))),
            file_queue_maxsize=int(os.getenv("FILE_QUEUE_MAXSIZE", str(DEFAULT_FILE_QUEUE_MAXSIZE))),
            file_queue_min_size=int(os.getenv("FILE_QUEUE_MIN_SIZE", str(DEFAULT_FILE_QUEUE_MIN_SIZE))),
            deletion_batch_size=int(os.getenv("DELETION_BATCH_SIZE", str(DEFAULT_DELETION_BATCH_SIZE))),
            log_file_path=os.getenv("LOG_FILE_PATH", None),
            file_access_time_threshold_minutes=float(
                os.getenv(
                    "FILE_ACCESS_TIME_THRESHOLD_MINUTES",
                    str(DEFAULT_FILE_ACCESS_TIME_THRESHOLD_MINUTES),
                )
            ),
            shard_index=shard_index,
            total_shards=total_shards,
            enable_dir_cleanup=os.getenv("ENABLE_DIR_CLEANUP", str(DEFAULT_ENABLE_DIR_CLEANUP)).lower() == "true",
            kv_events_enabled=os.getenv("KV_EVENTS_ENABLED", str(DEFAULT_KV_EVENTS_ENABLED)).lower() == "true",
            zmq_endpoint=os.getenv("ZMQ_ENDPOINT", DEFAULT_ZMQ_ENDPOINT),
            zmq_topic=os.getenv("ZMQ_TOPIC", DEFAULT_ZMQ_TOPIC),
            use_lru_eviction=os.getenv("USE_LRU_EVICTION", str(DEFAULT_USE_LRU_EVICTION)).lower() == "true",
            continuous_cleanup_hours=float(os.getenv("CONTINUOUS_CLEANUP_HOURS", str(DEFAULT_CONTINUOUS_CLEANUP_HOURS))),
            continuous_cleanup_interval_minutes=float(os.getenv("CONTINUOUS_CLEANUP_INTERVAL_MINUTES", str(DEFAULT_CONTINUOUS_CLEANUP_INTERVAL_MIN))),
            mock_disk_total_bytes=int(os.getenv("MOCK_DISK_TOTAL_BYTES", str(DEFAULT_MOCK_DISK_TOTAL_BYTES))),
        )


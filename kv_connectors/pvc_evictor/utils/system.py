"""System utilities for PVC Evictor: disk usage monitoring and logging setup."""

import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiskUsage:
    """Disk usage information."""

    total_bytes: int
    used_bytes: int
    available_bytes: int
    usage_percent: float


def get_disk_usage_from_statvfs(mount_path: str) -> Optional[DiskUsage]:
    """
    Get disk usage using statvfs() - O(1) operation, critical for multi-TB volumes.
    
    Trade-off: statvfs() provides instant disk usage statistics but is less accurate
    than `du` which would be O(n) and could take hours on large volumes.
    """
    try:
        stat = os.statvfs(mount_path)
        block_size = stat.f_frsize
        total_blocks = stat.f_blocks
        free_blocks = stat.f_bfree

        total_bytes = total_blocks * block_size
        free_bytes = free_blocks * block_size
        used_bytes = total_bytes - free_bytes
        usage_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0

        return DiskUsage(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            available_bytes=free_bytes,
            usage_percent=usage_percent,
        )
    except Exception:
        return None


def setup_logging(
    log_level: str = "INFO",
    process_id: Optional[int] = None,
    log_file: Optional[str] = None,
):
    """Configure logging with specified level and optional process ID prefix.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        process_id: Process ID (1-10) for prefix in logs
        log_file: Optional file path to also write logs to (in addition to stdout)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create format with process ID prefix
    if process_id is not None:
        format_str = (
            f"[P{process_id}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Force reconfiguration for child processes
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Always log to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Optionally also log to file
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception:
            # If file logging fails, continue with stdout only
            pass


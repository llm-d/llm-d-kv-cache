"""Activator process for monitoring disk usage and controlling deletion."""

import logging
import multiprocessing
import os
import time

from utils.logging_helpers import send_stats_to_queue
from utils.system import get_disk_usage_from_statvfs, setup_logging, DiskUsage

def get_mock_disk_usage(mount_path: str, cache_dir: str, mock_total_bytes: int) -> DiskUsage | None:
    """Helper to fake disk usage metrics by summing actual file sizes in the cache."""
    total_used = 0
    target_path = os.path.join(mount_path, cache_dir) if cache_dir else mount_path
    if os.path.exists(target_path):
        try:
            for root, _, files in os.walk(target_path):
                for f in files:
                    if f.endswith(".bin"):
                        try:
                            total_used += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
        except Exception:
            return None
    usage_percent = (total_used / mock_total_bytes) * 100 if mock_total_bytes > 0 else 0
    return DiskUsage(
        total_bytes=mock_total_bytes,
        used_bytes=total_used,
        available_bytes=max(0, mock_total_bytes - total_used),
        usage_percent=usage_percent,
    )


def activator_process(
    process_num: int,
    mount_path: str,
    cleanup_threshold: float,
    target_threshold: float,
    logger_interval: float,
    deletion_event: multiprocessing.Event,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
    config_dict: dict = None,
):
    """
    Activator process (P(N+1)): Monitors disk usage and controls deletion trigger.

    Monitors statvfs() every logger_interval seconds, sets deletion_event when
    usage > cleanup_threshold, and clears deletion_event when usage < target_threshold.

    Reports statistics to main process for aggregated logging.
    """
    log_file = os.getenv("LOG_FILE_PATH", None)
    setup_logging("INFO", process_num, log_file)
    logger = logging.getLogger("activator")

    # Log activator startup information
    logger.info(f"Activator P{process_num} started - monitoring every {logger_interval}s")
    logger.info(f"Activator P{process_num} thresholds: cleanup={cleanup_threshold}%, target={target_threshold}%")

    deletion_active = False
    last_stats_send_time = 0.0

    try:
        while not shutdown_event.is_set():
            mock_total = config_dict.get("mock_disk_total_bytes", 0) if config_dict else 0
            if mock_total > 0:
                cache_dir = config_dict.get("cache_directory", "")
                usage = get_mock_disk_usage(mount_path, cache_dir, mock_total)
            else:
                usage = get_disk_usage_from_statvfs(mount_path)

            if usage:
                current_time = time.time()

                # Send stats periodically to main process for aggregated logging
                last_stats_send_time = send_stats_to_queue(
                    result_queue,
                    "activator_stats",
                    process_num,
                    {
                        "usage_percent": usage.usage_percent,
                        "used_bytes": usage.used_bytes,
                        "total_bytes": usage.total_bytes,
                        "deletion_active": deletion_active,
                    },
                    last_stats_send_time,
                )

                # Control deletion based on thresholds (always log these critical events)
                # Trigger deletion when usage exceeds cleanup threshold (hysteresis: ON at cleanup%, OFF at target%)
                if usage.usage_percent >= cleanup_threshold and not deletion_active:
                    # Log deletion activation (always immediate, even with aggregated logging)
                    logger.warning(
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% >= "
                        f"{cleanup_threshold}% - Triggering deletion ON"
                    )
                    logger.info(
                        f"DELETION_START: timestamp={current_time:.3f}, usage={usage.usage_percent:.2f}%, "
                        f"used={usage.used_bytes / (1024**3):.2f}GB, total={usage.total_bytes / (1024**3):.2f}GB"
                    )
                    deletion_event.set()
                    deletion_active = True
                elif usage.usage_percent <= target_threshold and deletion_active:
                    # Log deletion deactivation (always immediate, even with aggregated logging)
                    logger.info(
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% <= "
                        f"{target_threshold}% - Triggering deletion OFF"
                    )
                    logger.info(
                        f"DELETION_END: timestamp={current_time:.3f}, usage={usage.usage_percent:.2f}%, "
                        f"used={usage.used_bytes / (1024**3):.2f}GB, total={usage.total_bytes / (1024**3):.2f}GB"
                    )
                    deletion_event.clear()
                    deletion_active = False

            time.sleep(logger_interval)

    except Exception as e:
        logger.exception(f"Activator P{process_num} error: {e}")
    finally:
        logger.info(f"Activator P{process_num} stopping")
        deletion_event.clear()

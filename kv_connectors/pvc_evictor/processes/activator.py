"""Activator process for monitoring disk usage and controlling deletion."""

import os
import time
import logging
import multiprocessing

from utils.system import setup_logging, get_disk_usage_from_statvfs
from utils.logging_helpers import send_stats_to_queue


def activator_process(
    process_num: int,
    mount_path: str,
    cleanup_threshold: float,
    target_threshold: float,
    logger_interval: float,
    deletion_event: multiprocessing.Event,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
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
    logger.info(
        f"Activator P{process_num} thresholds: cleanup={cleanup_threshold}%, target={target_threshold}%"
    )

    deletion_active = False
    last_stats_send_time = 0.0

    try:
        while not shutdown_event.is_set():
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
                    last_stats_send_time
                )

                # Control deletion based on thresholds (always log these critical events)
                # Trigger deletion when usage exceeds cleanup threshold (hysteresis: ON at cleanup%, OFF at target%)
                if usage.usage_percent >= cleanup_threshold and not deletion_active:
                    # Log deletion activation (always immediate, even with aggregated logging)
                    logger.warning(
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% >= {cleanup_threshold}% - Triggering deletion ON"
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
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% <= {target_threshold}% - Triggering deletion OFF"
                    )
                    logger.info(
                        f"DELETION_END: timestamp={current_time:.3f}, usage={usage.usage_percent:.2f}%, "
                        f"used={usage.used_bytes / (1024**3):.2f}GB, total={usage.total_bytes / (1024**3):.2f}GB"
                    )
                    deletion_event.clear()
                    deletion_active = False

            time.sleep(logger_interval)

    except Exception as e:
        logger.error(f"Activator P{process_num} error: {e}", exc_info=True)
    finally:
        logger.info(f"Activator P{process_num} stopping")
        deletion_event.clear()


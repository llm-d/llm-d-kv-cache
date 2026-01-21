"""Activator process for monitoring disk usage and controlling deletion."""

import os
import time
import logging
import multiprocessing

from utils.system import setup_logging, get_disk_usage_from_statvfs


def activator_process(
    process_num: int,
    mount_path: str,
    cleanup_threshold: float,
    target_threshold: float,
    logger_interval: float,
    deletion_event: multiprocessing.Event,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
    aggregated_logging: bool = True,
    aggregated_logging_interval: float = 30.0,
):
    """
    Activator process (P(N+1)): Monitors disk usage and controls deletion trigger.

    - Monitors statvfs() every logger_interval seconds
    - Sets deletion_event when usage > cleanup_threshold
    - Clears deletion_event when usage < target_threshold
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
    last_queue_log_time = 0.0
    last_stats_send_time = 0.0

    try:
        while not shutdown_event.is_set():
            usage = get_disk_usage_from_statvfs(mount_path)

            if usage:
                current_time = time.time()
                
                # Send stats to result_queue for aggregated logging or log locally
                if aggregated_logging:
                    # Send stats periodically to main process for aggregated logging
                    if current_time - last_stats_send_time >= aggregated_logging_interval:
                        try:
                            result_queue.put(
                                (
                                    "activator_stats",
                                    process_num,
                                    {
                                        "usage_percent": usage.usage_percent,
                                        "used_bytes": usage.used_bytes,
                                        "total_bytes": usage.total_bytes,
                                        "deletion_active": deletion_active,
                                    },
                                ),
                                timeout=0.1,
                            )
                        except Exception:
                            pass  # Queue full or timeout - skip stats update
                        last_stats_send_time = current_time
                else:
                    # Traditional local logging (when aggregated logging is OFF)
                    logger.info(
                        f"PVC Usage: {usage.usage_percent:.2f}% "
                        f"({usage.used_bytes / (1024**3):.2f}GB / {usage.total_bytes / (1024**3):.2f}GB) "
                        f"[statvfs]"
                    )

                    # Log queue status periodically (every 10 seconds)
                    if current_time - last_queue_log_time >= 10.0:
                        try:
                            # Try to get queue size (might fail if queue is in different process)
                            logger.debug(
                                f"Queue status check (deletion={'ON' if deletion_event.is_set() else 'OFF'})"
                            )
                            last_queue_log_time = current_time
                        except Exception as e:
                            # Log specific exception for diagnostics
                            logger.debug(f"Failed to check queue status: {e}")
                            last_queue_log_time = current_time

                # Control deletion based on thresholds (always log these critical events)
                if usage.usage_percent >= cleanup_threshold and not deletion_active:
                    # Log deletion activation (always immediate, even with aggregated logging)
                    logger.warning(
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% >= {cleanup_threshold}% - Triggering deletion ON"
                    )
                    logger.info(
                        f"DELETION_START:{current_time:.3f},{usage.usage_percent:.2f}"
                    )
                    deletion_event.set()
                    deletion_active = True
                elif usage.usage_percent <= target_threshold and deletion_active:
                    # Log deletion deactivation (always immediate, even with aggregated logging)
                    logger.info(
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% <= {target_threshold}% - Triggering deletion OFF"
                    )
                    logger.info(
                        f"DELETION_END:{current_time:.3f},{usage.usage_percent:.2f}"
                    )
                    deletion_event.clear()
                    deletion_active = False

            time.sleep(logger_interval)

    except Exception as e:
        logger.error(f"Activator P{process_num} error: {e}", exc_info=True)
    finally:
        logger.info(f"Activator P{process_num} stopping")
        deletion_event.clear()


"""Activator process for monitoring disk usage and controlling deletion."""

import os
import time
import logging
import multiprocessing

from utils.system import setup_logging, get_disk_usage_from_statvfs


def activator_process(
    mount_path: str,
    cleanup_threshold: float,
    target_threshold: float,
    logger_interval: float,
    deletion_event: multiprocessing.Event,
    shutdown_event: multiprocessing.Event,
):
    """
    Activator process (P9): Monitors disk usage and controls deletion trigger.

    - Monitors statvfs() every logger_interval seconds
    - Sets deletion_event when usage > cleanup_threshold
    - Clears deletion_event when usage < target_threshold
    """
    log_file = os.getenv("LOG_FILE_PATH", None)
    setup_logging("INFO", 9, log_file)
    logger = logging.getLogger("activator")

    # Log activator startup information
    logger.info(f"Activator P9 started - monitoring every {logger_interval}s")
    logger.info(
        f"Activator P9 thresholds: cleanup={cleanup_threshold}%, target={target_threshold}%"
    )

    deletion_active = False

    try:
        while not shutdown_event.is_set():
            usage = get_disk_usage_from_statvfs(mount_path)

            if usage:
                current_time = time.time()
                
                # Log disk usage information
                logger.info(
                    f"PVC Usage: {usage.usage_percent:.2f}% "
                    f"({usage.used_bytes / (1024**3):.2f}GB / {usage.total_bytes / (1024**3):.2f}GB) "
                    f"[statvfs]"
                )

                # Log queue status periodically
                if int(current_time) % 10 == 0:  # Every 10 seconds
                    try:
                        # Try to get queue size (might fail if queue is in different process)
                        logger.debug(
                            f"Queue status check (deletion={'ON' if deletion_event.is_set() else 'OFF'})"
                        )
                    except:
                        pass

                # Control deletion based on thresholds
                if usage.usage_percent >= cleanup_threshold and not deletion_active:
                    # Log deletion activation
                    logger.warning(
                        f"Activator P9: Usage {usage.usage_percent:.2f}% >= {cleanup_threshold}% - Triggering deletion ON"
                    )
                    logger.info(
                        f"DELETION_START:{current_time:.3f},{usage.usage_percent:.2f}"
                    )
                    deletion_event.set()
                    deletion_active = True
                elif usage.usage_percent <= target_threshold and deletion_active:
                    # Log deletion deactivation
                    logger.info(
                        f"Activator P9: Usage {usage.usage_percent:.2f}% <= {target_threshold}% - Triggering deletion OFF"
                    )
                    logger.info(
                        f"DELETION_END:{current_time:.3f},{usage.usage_percent:.2f}"
                    )
                    deletion_event.clear()
                    deletion_active = False

            time.sleep(logger_interval)

    except Exception as e:
        logger.error(f"Activator P9 error: {e}", exc_info=True)
    finally:
        logger.info("Activator P9 stopping")
        deletion_event.clear()


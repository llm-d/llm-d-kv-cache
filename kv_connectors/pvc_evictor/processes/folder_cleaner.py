"""Folder cleaner process for background empty directory deletion."""

import contextlib
import logging
import os
import time
from typing import Any

from utils.logging_helpers import send_stats_to_queue
from utils.system import setup_logging


def folder_cleaner_process(
    process_num: int,
    folder_queue: Any,
    result_queue: Any,
    shutdown_event: Any,
    config_dict: dict,
):
    """
    Background process that pulls directory paths from folder_queue
    and attempts to delete them using os.rmdir.

    Only deletes folders when they are empty; fails silently on non-empty folders.
    """
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict["log_level"], process_num, log_file)
    logger = logging.getLogger(f"folder_cleaner_{process_num}")

    logger.info(f"Folder Cleaner background process P{process_num} started")

    folders_deleted = 0
    last_report_time = time.time()

    try:
        while not shutdown_event.is_set():
            try:
                # Pull directory path from queue (block up to 1 second)
                try:
                    d_str = folder_queue.get(timeout=1.0)
                except Exception:
                    continue

                # Attempt to delete empty directory
                try:
                    os.rmdir(d_str)
                    folders_deleted += 1
                    logger.debug(f"Purged empty folder: {d_str}")
                except OSError:
                    # Directory not empty yet, skip silently
                    pass

                # Periodically report progress (every 30 seconds)
                last_report_time = send_stats_to_queue(
                    result_queue,
                    "folder_cleaner_stats",
                    process_num,
                    {"folders_purged": folders_deleted},
                    last_report_time,
                    interval=30.0,
                )

            except Exception as e:
                logger.exception(f"Folder Cleaner P{process_num} encountered error: {e}")
                time.sleep(1.0)

    except Exception as e:
        logger.exception(f"Folder Cleaner P{process_num} critical failure: {e}")
    finally:
        logger.info(f"Folder Cleaner P{process_num} stopping - total empty folders purged: {folders_deleted}")
        with contextlib.suppress(Exception):
            result_queue.put(
                ("folder_cleaner_stats", process_num, {"folders_purged": folders_deleted}),
                timeout=1.0,
            )

"""Folder cleaner process for background empty directory deletion."""

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from utils.logging_helpers import send_stats_to_queue
from utils.system import setup_logging


def cleanup_empty_dirs(paths: list[str], cache_path: Path, logger: logging.Logger) -> int:
    """
    Remove empty directories left behind after file deletion.

    Walks from each path (or its parent directory if it's a file) upward,
    removing empty directories until reaching cache_path or a non-empty directory.
    Uses os.rmdir which only succeeds on empty directories.

    Returns the number of directories removed.
    """
    dirs_removed = 0
    cache_path_str = str(cache_path)

    parents_seen: set[str] = set()
    for path_str in paths:
        path = Path(path_str)
        parent = str(path.parent) if path_str.endswith(".bin") or (path.exists() and path.is_file()) else path_str
        if parent not in parents_seen:
            parents_seen.add(parent)

    for dir_path_str in sorted(parents_seen, key=len, reverse=True):
        current = dir_path_str
        while current.startswith(cache_path_str) and current != cache_path_str:
            if not os.path.isdir(current):
                current = str(Path(current).parent)
                continue
            try:
                os.rmdir(current)
                dirs_removed += 1
            except OSError:
                break
            current = str(Path(current).parent)

    if dirs_removed > 0:
        logger.debug(f"Cleaned up {dirs_removed} empty directories")

    return dirs_removed


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

    cache_path = Path(config_dict["pvc_mount_path"]) / config_dict["cache_directory"]
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

                # Attempt to delete empty directory and its parents upward
                try:
                    removed = cleanup_empty_dirs([d_str], cache_path, logger)
                    if removed > 0:
                        folders_deleted += removed
                        logger.debug(f"Purged {removed} empty folders starting from: {d_str}")
                except Exception as e:
                    logger.error(f"Error cleaning up folder {d_str}: {e}")

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

"""Deleter process for batch file deletion."""

import contextlib
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Any

from utils.system import setup_logging

# Constants for timing
PARTIAL_BATCH_TIMEOUT_SECONDS = 5.0  # Process partial batch after N seconds of inactivity


def delete_batch(file_paths: list[str], dry_run: bool, logger: logging.Logger, drain_time: float, executor: Any, folder_queue: Any = None) -> tuple[int, int, int]:
    """
    Delete a batch of files using persistent ThreadPoolExecutor with zero stat overhead.

    Returns: (files_deleted, bytes_freed, folders_deleted)
    """
    if dry_run:
        logger.debug(f"[DRY RUN] Would delete {len(file_paths)} files")
        return len(file_paths), 0, 0

    if not file_paths:
        return 0, 0, 0

    import os

    def _unlink(p_str: str) -> int:
        try:
            os.unlink(p_str)
            return 1
        except Exception:
            return 0

    t0 = time.time()
    # 1. Unlink all files concurrently across 64 threads
    results = list(executor.map(_unlink, file_paths))
    deleted_count = sum(results)
    t1 = time.time()
    unlink_time = t1 - t0

    # 2. Bottom-up empty directory candidates pushed to folder queue instead of synchronous rmdir
    folders_deleted_count = 0
    if folder_queue and deleted_count > 0:
        unique_parents = set()
        for f_str in file_paths:
            unique_parents.add(Path(f_str).parent)
        for p in unique_parents:
            try:
                # Non-blocking push to ensure unlink pipeline is never blocked
                folder_queue.put_nowait(str(p))
            except Exception:
                pass  # Silent skip if queue is saturated

    rmdir_time = 0.0

    total_cycle = drain_time + unlink_time + rmdir_time
    logger.debug(
        f"[INSTRUMENTATION] Batch of {len(file_paths)} files: "
        f"queue_drain={drain_time:.3f}s, unlink_threads={unlink_time:.3f}s, rmdir_cleanup={rmdir_time:.3f}s, "
        f"total_cycle={total_cycle:.3f}s (deletion_rate={len(file_paths) / max(0.001, unlink_time + rmdir_time):.1f} files/sec)"
    )

    # Return 0 for bytes_freed to maintain tuple compatibility without doing slow stat()
    return deleted_count, 0, folders_deleted_count


def delete_file_batch(
    batch: list[str],
    dry_run: bool,
    logger: logging.Logger,
    process_id: str,
    total_files_deleted: int,
    total_bytes_freed: int,
    total_folders_deleted: int,
    prev_batch_time: float | None,
    result_queue: multiprocessing.Queue,
    drain_time: float,
    executor: Any,
    process_num: int,
    folder_queue: Any = None,
) -> tuple[int, int, int, float]:
    """
    Process a batch of files for deletion and report progress to main process.

    Returns: (updated_total_files_deleted, updated_total_bytes_freed, updated_total_folders_deleted, batch_start_time)
    """
    batch_start_time = time.time()
    deleted, freed, folders_deleted = delete_batch(batch, dry_run, logger, drain_time, executor, folder_queue)

    total_files_deleted += deleted
    total_bytes_freed += freed
    total_folders_deleted += folders_deleted

    logger.debug(
        f"Batch of {len(batch)} files: deleted_files={deleted}, deleted_folders={folders_deleted}, "
        f"total_files_deleted={total_files_deleted}, total_folders_deleted={total_folders_deleted}"
    )

    # Report progress to result_queue for aggregated logging in main process
    with contextlib.suppress(Exception):
        result_queue.put(
            ("progress", process_num, total_files_deleted, total_bytes_freed, total_folders_deleted),
            timeout=1.0,
        )

    return total_files_deleted, total_bytes_freed, total_folders_deleted, batch_start_time


def deleter_process(
    process_num: int,
    cache_path: Path,
    config_dict: dict,
    deletion_event: multiprocessing.Event,
    file_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
    folder_queue: Any = None,
):
    """
    Deleter process: Deletes files (when deletion_event is set) from queue in batches.
    """
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict["log_level"], process_num, log_file)
    logger = logging.getLogger(f"deleter_{process_num}")

    process_id = f"P{process_num}"

    batch_size = config_dict["deletion_batch_size"]
    dry_run = config_dict["dry_run"]

    logger.info(f"Deleter P{process_num} started - batch size: {batch_size}, dry_run: {dry_run}")

    total_files_deleted = 0
    total_bytes_freed = 0
    total_folders_deleted = 0
    current_batch = []
    prev_batch_time = None
    last_batch_check_time = time.time()
    partial_batch_timeout = PARTIAL_BATCH_TIMEOUT_SECONDS
    last_idle_log_time = 0.0
    drain_start_time = time.time()

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=64) as executor:
            while not shutdown_event.is_set():
                # Only process when deletion is ON
                if deletion_event.is_set():
                    try:
                        # Try to get file from queue
                        try:
                            # Pull as many items as possible up to batch_size without blocking
                            while len(current_batch) < batch_size:
                                try:
                                    file_path_str = file_queue.get_nowait()
                                    current_batch.append(file_path_str)
                                    last_batch_check_time = time.time()
                                except Exception:
                                    break

                            # If we have nothing, do a blocking get with timeout to avoid busy-waiting
                            if not current_batch:
                                file_path_str = file_queue.get(timeout=1.0)
                                current_batch.append(file_path_str)
                                last_batch_check_time = time.time()

                            # Delete batch when full
                            if len(current_batch) >= batch_size:
                                drain_time = time.time() - drain_start_time
                                total_files_deleted, total_bytes_freed, total_folders_deleted, prev_batch_time = delete_file_batch(
                                    current_batch,
                                    dry_run,
                                    logger,
                                    process_id,
                                    total_files_deleted,
                                    total_bytes_freed,
                                    total_folders_deleted,
                                    prev_batch_time,
                                    result_queue,
                                    drain_time,
                                    executor,
                                    process_num,
                                    folder_queue,
                                )
                                current_batch = []
                                drain_start_time = time.time()

                        except Exception:
                            # Queue empty or timeout - check if we should process partial batch
                            # Sleep only when queue is empty to prevent busy-waiting
                            time.sleep(0.1)

                        # Process partial batch if:
                        # 1. We have files in batch AND
                        # 2. (Batch is full - already handled above OR enough time has passed OR queue is empty)
                        current_time = time.time()
                        time_since_last_check = current_time - last_batch_check_time
                        queue_empty = False
                        try:
                            queue_empty = file_queue.empty()
                        except Exception as e:
                            # If we can't check queue status, assume it's not empty and continue
                            logger.debug(f"Failed to check queue empty status: {e}")
                            queue_empty = False

                        should_process_partial = current_batch and (
                            (time_since_last_check >= partial_batch_timeout) or queue_empty
                        )

                        if should_process_partial:
                            drain_time = time.time() - drain_start_time
                            total_files_deleted, total_bytes_freed, total_folders_deleted, prev_batch_time = delete_file_batch(
                                current_batch,
                                dry_run,
                                logger,
                                process_id,
                                total_files_deleted,
                                total_bytes_freed,
                                total_folders_deleted,
                                prev_batch_time,
                                result_queue,
                                drain_time,
                                executor,
                                process_num,
                                folder_queue,
                            )
                            current_batch = []
                            last_batch_check_time = current_time
                            drain_start_time = time.time()

                    except Exception as e:
                        logger.exception(f"Deleter P{process_num} error processing queue: {e}")
                        time.sleep(1.0)
                else:
                    # Deletion is OFF - clear any pending batch and wait
                    if current_batch:
                        logger.debug(f"Deletion OFF - clearing {len(current_batch)} pending files")
                        current_batch = []
                    # Log idle status periodically (every 30 seconds)
                    current_time = time.time()
                    if current_time - last_idle_log_time >= 30.0:
                        logger.debug("Deletion OFF - waiting for trigger")
                        last_idle_log_time = current_time
                    time.sleep(0.5)

            # Delete remaining batch on shutdown
            if current_batch:
                drain_time = time.time() - drain_start_time
                deleted, freed, folders_deleted = delete_batch(current_batch, dry_run, logger, drain_time, executor, folder_queue)
                total_files_deleted += deleted
                total_bytes_freed += freed
                total_folders_deleted += folders_deleted

    except Exception as e:
        logger.exception(f"Deleter P{process_num} error: {e}")
    finally:
        logger.info(
            f"Deleter P{process_num} stopping - deleted {total_files_deleted} files, "
            f"deleted {total_folders_deleted} folders, {total_bytes_freed / (1024**3):.2f}GB"
        )
        result_queue.put(("done", process_num, total_files_deleted, total_bytes_freed, total_folders_deleted), timeout=1.0)


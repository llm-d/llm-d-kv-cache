"""Deleter process for batch file deletion."""

import time
import logging
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Optional

from utils.system import setup_logging


def delete_batch(
    file_paths: List[str], dry_run: bool, logger: logging.Logger
) -> Tuple[int, int]:
    """
    Delete a batch of files using xargs rm -f (batch deletion).
    
    If xargs fails, logs error and skips the batch (files will be retried in next cycle).

    Returns: (files_deleted, bytes_freed)
    """
    if dry_run:
        logger.debug(f"[DRY RUN] Would delete {len(file_paths)} files")
        return len(file_paths), 0

    valid_paths = []
    total_bytes = 0

    # Validate paths and calculate total size
    for path_str in file_paths:
        try:
            file_path = Path(path_str)
            if file_path.exists():
                stat = file_path.stat()
                valid_paths.append(path_str)
                total_bytes += stat.st_size
        except Exception:
            continue

    if not valid_paths:
        # All files in batch don't exist (already deleted or invalid paths)
        # This is normal - files may have been deleted between queuing and processing
        # or the same files were queued multiple times
        return 0, 0

    # Use xargs rm -f for batch deletion
    # Use null-terminated input for xargs -0 (safe handling of file paths with special characters)
    try:
        input_data = "\0".join(valid_paths).encode("utf-8")
        result = subprocess.run(
            ["xargs", "-0", "rm", "-f"],
            input=input_data,
            capture_output=True,
            timeout=60,
            check=False,
        )

        if result.returncode == 0:
            return len(valid_paths), total_bytes
        else:
            # Log error and skip batch - files will be retried in next cycle
            logger.error(
                f"xargs rm failed (returncode={result.returncode}), skipping batch of {len(valid_paths)} files. "
                f"Files will be retried in next cycle."
            )
            if result.stderr:
                logger.debug(f"xargs stderr: {result.stderr.decode('utf-8', errors='ignore')}")
            return 0, 0
    except subprocess.TimeoutExpired:
        logger.error(
            f"xargs rm timed out, skipping batch of {len(valid_paths)} files. "
            f"Files will be retried in next cycle."
        )
        return 0, 0
    except Exception as e:
        logger.error(
            f"Batch deletion error: {e}, skipping batch of {len(valid_paths)} files. "
            f"Files will be retried in next cycle."
        )
        return 0, 0


def process_and_log_batch(
    batch: List[str],
    dry_run: bool,
    logger: logging.Logger,
    process_id: str,
    total_files_deleted: int,
    total_bytes_freed: int,
    prev_batch_time: Optional[float],
    result_queue: multiprocessing.Queue,
    aggregated_logging: bool = True,
) -> Tuple[int, int, float]:
    """
    Process a batch of files for deletion and log results.
    
    Returns: (updated_total_files_deleted, updated_total_bytes_freed, batch_start_time)
    """
    batch_start_time = time.time()
    deleted, freed = delete_batch(batch, dry_run, logger)
    batch_duration_ms = (time.time() - batch_start_time) * 1000

    total_files_deleted += deleted
    total_bytes_freed += freed

    # Track batch-to-batch timing
    if prev_batch_time is not None:
        batch_gap_ms = (batch_start_time - prev_batch_time) * 1000
        log_timing_event(
            logger,
            "BATCH_TO_BATCH",
            batch_gap_ms,
            process_id=process_id,
            batch_size=len(batch),
        )

    log_timing_event(
        logger,
        "BATCH_DELETE",
        batch_duration_ms,
        process_id=process_id,
        files_deleted=deleted,
        bytes_freed=freed,
    )

    # Log batch completion locally only if aggregated logging is OFF
    if not aggregated_logging:
        logger.info(
            f"Deleted batch: {deleted} files, {freed / (1024**3):.2f}GB freed "
            f"(total: {total_files_deleted} files, {total_bytes_freed / (1024**3):.2f}GB)"
        )

    # Report progress to result_queue (used by both aggregated and non-aggregated logging)
    try:
        result_queue.put(
            ("progress", total_files_deleted, total_bytes_freed),
            timeout=1.0,
        )
    except Exception:
        pass  # Queue full, skip progress update

    return total_files_deleted, total_bytes_freed, batch_start_time


def log_timing_event(logger: logging.Logger, event_type: str, duration_ms: float, **kwargs):
    """Log timing event."""
    try:
        unix_timestamp = time.time()
        extra_fields = ",".join(f"{k}={v}" for k, v in kwargs.items())
        log_line = f"TIMING_{event_type}:{unix_timestamp:.3f},{duration_ms:.3f}"
        if extra_fields:
            log_line += f",{extra_fields}"
        logger.debug(log_line)
    except Exception as e:
        # Never let timing/logging failures affect core deletion logic,
        # but emit a debug message so issues can be diagnosed if needed.
        logger.debug(f"Failed to log timing event '{event_type}': {e}", exc_info=True)


def deleter_process(
    process_num: int,
    cache_path: Path,
    config_dict: dict,
    deletion_event: multiprocessing.Event,
    file_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
):
    """
    Deleter process (P(N+2)): Deletes files (when deletion_event is set) from queue in batches.
    """
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict.get("log_level", "INFO"), process_num, log_file)
    logger = logging.getLogger("deleter")
    
    process_id = f"P{process_num}"

    batch_size = config_dict.get("deletion_batch_size", 100)
    dry_run = config_dict.get("dry_run", False)
    aggregated_logging = config_dict.get("aggregated_logging", True)

    logger.info(f"Deleter P{process_num} started - batch size: {batch_size}, dry_run: {dry_run}")

    total_files_deleted = 0
    total_bytes_freed = 0
    current_batch = []
    prev_batch_time = None
    last_batch_check_time = time.time()
    partial_batch_timeout = 5.0  # Process partial batch after 5 seconds
    last_idle_log_time = 0.0

    try:
        while not shutdown_event.is_set():
            # Only process when deletion is ON
            if deletion_event.is_set():
                try:
                    # Try to get file from queue
                    try:
                        file_path_str = file_queue.get(timeout=1.0)
                        current_batch.append(file_path_str)
                        last_batch_check_time = time.time()

                        # Delete batch when full
                        if len(current_batch) >= batch_size:
                            total_files_deleted, total_bytes_freed, prev_batch_time = process_and_log_batch(
                                current_batch,
                                dry_run,
                                logger,
                                process_id,
                                total_files_deleted,
                                total_bytes_freed,
                                prev_batch_time,
                                result_queue,
                                aggregated_logging,
                            )
                            current_batch = []

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
                    
                    should_process_partial = (
                        current_batch and (
                            (time_since_last_check >= partial_batch_timeout) or
                            queue_empty
                        )
                    )
                    
                    if should_process_partial:
                        total_files_deleted, total_bytes_freed, prev_batch_time = process_and_log_batch(
                            current_batch,
                            dry_run,
                            logger,
                            process_id,
                            total_files_deleted,
                            total_bytes_freed,
                            prev_batch_time,
                            result_queue,
                            aggregated_logging,
                        )
                        current_batch = []
                        last_batch_check_time = current_time

                except Exception as e:
                    logger.error(
                        f"Deleter P{process_num} error processing queue: {e}", exc_info=True
                    )
                    time.sleep(1.0)
            else:
                # Deletion is OFF - clear any pending batch and wait
                if current_batch:
                    logger.debug(
                        f"Deletion OFF - clearing {len(current_batch)} pending files"
                    )
                    current_batch = []
                # Log idle status periodically (every 30 seconds)
                current_time = time.time()
                if current_time - last_idle_log_time >= 30.0:
                    logger.debug("Deletion OFF - waiting for trigger")
                    last_idle_log_time = current_time
                time.sleep(0.5)

        # Delete remaining batch on shutdown
        if current_batch:
            deleted, freed = delete_batch(current_batch, dry_run, logger)
            total_files_deleted += deleted
            total_bytes_freed += freed

    except Exception as e:
        logger.error(f"Deleter P{process_num} error: {e}", exc_info=True)
    finally:
        logger.info(
            f"Deleter P{process_num} stopping - deleted {total_files_deleted} files, {total_bytes_freed / (1024**3):.2f}GB"
        )
        result_queue.put(("done", total_files_deleted, total_bytes_freed), timeout=1.0)


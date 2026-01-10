"""Deleter process for batch file deletion."""

import time
import logging
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Tuple

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


def deleter_process(
    cache_path: Path,
    config_dict: dict,
    deletion_event: multiprocessing.Event,
    file_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
):
    """
    Deleter process (P10): Deletes files from queue in batches.

    - Only deletes when deletion_event is set
    - Reads file paths from file_queue
    - Batches files for efficient deletion using xargs rm -f
    """
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict.get("log_level", "INFO"), 10, log_file)
    logger = logging.getLogger("deleter")

    batch_size = config_dict.get("deletion_batch_size", 100)
    dry_run = config_dict.get("dry_run", False)

    logger.info(f"Deleter P10 started - batch size: {batch_size}, dry_run: {dry_run}")

    total_files_deleted = 0
    total_bytes_freed = 0
    current_batch = []
    prev_batch_time = None
    last_batch_check_time = time.time()
    partial_batch_timeout = 5.0  # Process partial batch after 5 seconds

    def log_timing(event_type: str, duration_ms: float, **kwargs):
        """Log timing event (matching v3 format)."""
        try:
            unix_timestamp = time.time()
            extra_fields = ",".join(f"{k}={v}" for k, v in kwargs.items())
            log_line = f"TIMING_{event_type}:{unix_timestamp:.3f},{duration_ms:.3f}"
            if extra_fields:
                log_line += f",{extra_fields}"
            logger.debug(log_line)
        except Exception:
            pass

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
                            batch_start_time = time.time()
                            deleted, freed = delete_batch(
                                current_batch, dry_run, logger
                            )
                            batch_duration_ms = (time.time() - batch_start_time) * 1000

                            total_files_deleted += deleted
                            total_bytes_freed += freed

                            # Track batch-to-batch timing
                            if prev_batch_time is not None:
                                batch_gap_ms = (
                                    batch_start_time - prev_batch_time
                                ) * 1000
                                log_timing(
                                    "BATCH_TO_BATCH",
                                    batch_gap_ms,
                                    process_id="P10",
                                    batch_size=len(current_batch),
                                )

                            prev_batch_time = batch_start_time

                            log_timing(
                                "BATCH_DELETE",
                                batch_duration_ms,
                                process_id="P10",
                                files_deleted=deleted,
                                bytes_freed=freed,
                            )

                            logger.info(
                                f"Deleted batch: {deleted} files, {freed / (1024**3):.2f}GB freed "
                                f"(total: {total_files_deleted} files, {total_bytes_freed / (1024**3):.2f}GB)"
                            )

                            current_batch = []

                            # Report progress
                            result_queue.put(
                                ("progress", total_files_deleted, total_bytes_freed),
                                timeout=1.0,
                            )

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
                    except Exception:
                        pass
                    
                    should_process_partial = (
                        current_batch and (
                            (time_since_last_check >= partial_batch_timeout and len(current_batch) > 0) or
                            queue_empty
                        )
                    )
                    
                    if should_process_partial:
                        batch_start_time = time.time()
                        deleted, freed = delete_batch(
                            current_batch, dry_run, logger
                        )
                        batch_duration_ms = (time.time() - batch_start_time) * 1000

                        total_files_deleted += deleted
                        total_bytes_freed += freed

                        # Track batch-to-batch timing
                        if prev_batch_time is not None:
                            batch_gap_ms = (
                                batch_start_time - prev_batch_time
                            ) * 1000
                            log_timing(
                                "BATCH_TO_BATCH",
                                batch_gap_ms,
                                process_id="P10",
                                batch_size=len(current_batch),
                            )

                        prev_batch_time = batch_start_time

                        log_timing(
                            "BATCH_DELETE",
                            batch_duration_ms,
                            process_id="P10",
                            files_deleted=deleted,
                            bytes_freed=freed,
                        )

                        logger.info(
                            f"Deleted batch: {deleted} files, {freed / (1024**3):.2f}GB freed "
                            f"(total: {total_files_deleted} files, {total_bytes_freed / (1024**3):.2f}GB)"
                        )
                        current_batch = []
                        last_batch_check_time = current_time

                        # Report progress
                        result_queue.put(
                            ("progress", total_files_deleted, total_bytes_freed),
                            timeout=1.0,
                        )
                except Exception as e:
                    logger.error(
                        f"Deleter P10 error processing queue: {e}", exc_info=True
                    )
                    time.sleep(1.0)
            else:
                # Deletion is OFF - clear any pending batch and wait
                if current_batch:
                    logger.debug(
                        f"Deletion OFF - clearing {len(current_batch)} pending files"
                    )
                    current_batch = []
                # Log idle status periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    logger.debug("Deletion OFF - waiting for trigger")
                time.sleep(0.5)

        # Delete remaining batch on shutdown
        if current_batch:
            deleted, freed = delete_batch(current_batch, dry_run, logger)
            total_files_deleted += deleted
            total_bytes_freed += freed

    except Exception as e:
        logger.error(f"Deleter P10 error: {e}", exc_info=True)
    finally:
        logger.info(
            f"Deleter P10 stopping - deleted {total_files_deleted} files, {total_bytes_freed / (1024**3):.2f}GB"
        )
        result_queue.put(("done", total_files_deleted, total_bytes_freed), timeout=1.0)


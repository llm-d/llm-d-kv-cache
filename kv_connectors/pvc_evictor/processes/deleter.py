"""Deleter process for batch file deletion."""

import contextlib
import json
import logging
import multiprocessing
import os
import re
import subprocess
import time
from pathlib import Path

from utils.system import setup_logging

# Constants for timing
PARTIAL_BATCH_TIMEOUT_SECONDS = 5.0  # Process partial batch after N seconds of inactivity

_model_name_cache: dict[str, str | None] = {}


def extract_block_hash(file_path: str) -> int | None:
    """Extract block hash from a file path like .../abc/de_g0/abcdef0123456789.bin"""
    basename = os.path.basename(file_path)
    if not basename.endswith(".bin"):
        return None

    hex_str = basename[:-4]
    if len(hex_str) != 16:
        return None

    try:
        return int(hex_str, 16)
    except ValueError:
        return None


def extract_model_name(file_path: str, cache_path: str) -> str | None:
    """Extract the model name from the directory structure.

    The FileMapper layout is:
        <cache_path>/<safe_model_name>_<sha256[:12]>_r<rank>/hhh/hh_g<idx>/hash.bin

    The original model name is read from config.json at:
        <cache_path>/<safe_model_name>_<sha256[:12]>/config.json
    """
    try:
        relative = os.path.relpath(file_path, cache_path)
    except ValueError:
        return None

    parts = relative.split(os.sep)
    if len(parts) < 2:
        return None

    rank_dir = parts[0]
    match = re.match(r"^(.+)_r\d+$", rank_dir)
    if not match:
        return None

    base_dir_path = os.path.join(cache_path, match.group(1))

    if base_dir_path in _model_name_cache:
        return _model_name_cache[base_dir_path]

    config_path = os.path.join(base_dir_path, "config.json")
    try:
        with open(config_path) as f:
            model_name = json.load(f).get("model_name")
    except (OSError, json.JSONDecodeError):
        model_name = None

    _model_name_cache[base_dir_path] = model_name
    return model_name


def delete_batch(file_paths: list[str], dry_run: bool, logger: logging.Logger) -> tuple[int, int]:
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
            f"xargs rm timed out, skipping batch of {len(valid_paths)} files. Files will be retried in next cycle."
        )
        return 0, 0
    except Exception as e:
        logger.error(
            f"Batch deletion error: {e}, skipping batch of {len(valid_paths)} files. "
            f"Files will be retried in next cycle."
        )
        return 0, 0


def delete_file_batch(
    batch: list[str],
    dry_run: bool,
    logger: logging.Logger,
    process_id: str,
    total_files_deleted: int,
    total_bytes_freed: int,
    prev_batch_time: float | None,
    result_queue: multiprocessing.Queue,
    event_publisher=None,
    cache_path=None,
) -> tuple[int, int, float]:
    """
    Process a batch of files for deletion and report progress to main process.

    Returns: (updated_total_files_deleted, updated_total_bytes_freed, batch_start_time)
    """
    batch_start_time = time.time()
    deleted, freed = delete_batch(batch, dry_run, logger)

    if deleted > 0 and event_publisher is not None and cache_path is not None:
        model_hashes = {}
        for b in batch:
            h = extract_block_hash(b)
            model = extract_model_name(b, str(cache_path))
            if h is not None and model is not None:
                model_hashes.setdefault(model, []).append(h)

        for model, hashes in model_hashes.items():
            try:
                event_publisher.publish_blocks_removed(hashes, model_name=model)
            except Exception:
                logger.warning("Failed to publish deletion events", exc_info=True)

    total_files_deleted += deleted
    total_bytes_freed += freed

    # Report progress to result_queue for aggregated logging in main process
    with contextlib.suppress(Exception):
        result_queue.put(
            ("progress", total_files_deleted, total_bytes_freed),
            timeout=1.0,
        )

    return total_files_deleted, total_bytes_freed, batch_start_time


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
    setup_logging(config_dict["log_level"], process_num, log_file)
    logger = logging.getLogger("deleter")

    process_id = f"P{process_num}"

    batch_size = config_dict["deletion_batch_size"]
    dry_run = config_dict["dry_run"]

    logger.info(f"Deleter P{process_num} started - batch size: {batch_size}, dry_run: {dry_run}")

    total_files_deleted = 0
    total_bytes_freed = 0
    current_batch = []
    prev_batch_time = None
    last_batch_check_time = time.time()
    partial_batch_timeout = PARTIAL_BATCH_TIMEOUT_SECONDS
    last_idle_log_time = 0.0

    event_publisher = None
    endpoint = config_dict.get("storage_events_endpoint", "")

    if endpoint:
        try:
            from llmd_fs_backend.event_publisher import (
                StorageEventPublisher,
                StorageMedium,
            )

            event_publisher = StorageEventPublisher(
                endpoint=endpoint,
                medium=StorageMedium.SHARED_STORAGE,
            )
            logger.info(
                "Storage event publisher created: endpoint=%s",
                endpoint,
            )
        except Exception:
            logger.warning("Failed to create storage event publisher", exc_info=True)

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
                            total_files_deleted, total_bytes_freed, prev_batch_time = delete_file_batch(
                                current_batch,
                                dry_run,
                                logger,
                                process_id,
                                total_files_deleted,
                                total_bytes_freed,
                                prev_batch_time,
                                result_queue,
                                event_publisher,
                                cache_path,
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

                    should_process_partial = current_batch and (
                        (time_since_last_check >= partial_batch_timeout) or queue_empty
                    )

                    if should_process_partial:
                        total_files_deleted, total_bytes_freed, prev_batch_time = delete_file_batch(
                            current_batch,
                            dry_run,
                            logger,
                            process_id,
                            total_files_deleted,
                            total_bytes_freed,
                            prev_batch_time,
                            result_queue,
                            event_publisher,
                            cache_path,
                        )
                        current_batch = []
                        last_batch_check_time = current_time

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
            deleted, freed = delete_batch(current_batch, dry_run, logger)
            total_files_deleted += deleted
            total_bytes_freed += freed

    except Exception as e:
        logger.exception(f"Deleter P{process_num} error: {e}")
    finally:
        logger.info(
            f"Deleter P{process_num} stopping - deleted {total_files_deleted} files, "
            f"{total_bytes_freed / (1024**3):.2f}GB"
        )
        result_queue.put(("done", total_files_deleted, total_bytes_freed), timeout=1.0)

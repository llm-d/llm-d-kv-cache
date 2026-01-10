"""Crawler process for discovering and queuing cache files."""

import os
import time
import logging
import multiprocessing
import re
from pathlib import Path
from typing import Optional, Iterator, List, Tuple

from utils.system import setup_logging


def safe_scandir(path: str) -> Iterator[os.DirEntry]:
    """
    Safely scan a directory, handling filesystem errors.
    
    Returns an iterator of directory entries, or empty iterator on error.
    This reduces exception handling duplication while maintaining streaming behavior.
    """
    try:
        return os.scandir(path)
    except (OSError, PermissionError):
        return iter([])


def hex_to_int(hex_str: str) -> Optional[int]:
    """Convert hex string to integer."""
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def extract_hex_folder_from_path(path: Path, cache_path: Path) -> Optional[str]:
    """
    Extract hex folder name from KV cache path.

    KV cache structure: {model}/[optional {path}/]tp_{N}/rank_{M}/{X}/{hash1}/{hash2}/*.bin
    where {X} can be any folder name (auto, half, float16, bfloat16, float, float32, etc.).
    Returns the first hex folder ({hash1}) after the {X} folder. hash1 length may vary.
    """
    try:
        path = Path(path)
        cache_path = Path(cache_path)

        try:
            relative = path.relative_to(cache_path)
        except ValueError:
            return None

        parts = relative.parts

        # Find the first hex folder after rank_{M}/{X}/
        # Look for pattern: .../rank_{M}/{X}/{hash1}/...
        for i, part in enumerate(parts):
            # Check if this part looks like a rank directory (rank_ followed by number)
            if part.startswith("rank_") and i + 2 < len(parts):
                # The next part is {X} (any folder name), and the one after should be hash1 (hex, any length)
                hash1 = parts[i + 2]
                # Check if it's a valid hex string (any length >= 1)
                if re.match(r"^[0-9a-fA-F]+$", hash1):
                    return hash1.lower()

        return None
    except Exception:
        return None


def get_hex_modulo_ranges(num_processes: int = 8) -> List[Tuple[int, int]]:
    """
    Get hex modulo ranges for each crawler process.

    Valid num_processes: 1, 2, 4, 8, 16
    Divides the 16 possible hex modulo values (0-15) evenly across processes.

    Examples:
    - 1 process:  %16 in [0, 15] (all values)
    - 2 processes: %16 in [0, 7] and [8, 15]
    - 4 processes: %16 in [0, 3], [4, 7], [8, 11], [12, 15]
    - 8 processes: %16 in [0, 1], [2, 3], ..., [14, 15]
    - 16 processes: %16 in [0], [1], ..., [15] (one value each)
    """
    valid_counts = [1, 2, 4, 8, 16]
    if num_processes not in valid_counts:
        raise ValueError(
            f"NUM_CRAWLER_PROCESSES must be one of {valid_counts}, got {num_processes}"
        )

    ranges = []
    values_per_process = 16 // num_processes

    for i in range(num_processes):
        modulo_range_min = i * values_per_process
        modulo_range_max = modulo_range_min + values_per_process - 1
        ranges.append((modulo_range_min, modulo_range_max))

    return ranges


def stream_cache_files(
    cache_path: Path, hex_modulo_range: Optional[Tuple[int, int]] = None
) -> Iterator[Path]:
    """
    Stream cache files using os.scandir() with hex folder filtering.

    Yields only files from folders where hex_folder_name % 16 is in the specified range.
    """
    if not cache_path.exists():
        return

    modulo_range_min, modulo_range_max = hex_modulo_range if hex_modulo_range else (0, 15)

    # Walk through cache directory structure
    # Structure: {model}/[optional {path}/]tp_{N}/rank_{M}/{X}/{hash1}/{hash2}/*.bin
    # where {X} can be any folder name (auto, half, float16, bfloat16, float, float32, etc.)
    # hash1 length may vary, hash2 is any folder name
    for model_dir in safe_scandir(str(cache_path)):
        if not model_dir.is_dir():
            continue

        # Find all tp_{N} directories, handling nested optional {path} folders
        # Structure: {model}/[optional {path}/.../]tp_{N}/rank_{M}/...
        # {path} can be nested (multiple levels), so we need to recursively search for tp_{N}
        tp_dirs = []
        
        def find_tp_dirs(current_path):
            """Recursively find tp_{N} directories."""
            for item in safe_scandir(current_path):
                if not item.is_dir():
                    continue
                if item.name.startswith("tp_"):
                    tp_dirs.append(item.path)
                else:
                    # Recursively search in subdirectories (handles nested {path} folders)
                    find_tp_dirs(item.path)
        
        find_tp_dirs(model_dir.path)

        # Process all tp_{N} directories found
        for tp_path in tp_dirs:
            # Scan rank_{M} directories under tp_{N}
            for rank_dir in safe_scandir(tp_path):
                if not rank_dir.is_dir() or not rank_dir.name.startswith("rank_"):
                    continue

                # Scan all directories under rank_{M} to find {X} folder
                # {X} can be any name (auto, half, float16, etc.)
                for dtype_dir in safe_scandir(rank_dir.path):
                    if not dtype_dir.is_dir():
                        continue

                    # Scan hash1 folders under {X} (hex, any length)
                    for hash1_dir in safe_scandir(dtype_dir.path):
                        if not hash1_dir.is_dir():
                            continue

                        # Check if this hex folder matches our modulo range
                        hex_folder = hash1_dir.name
                        hex_int = hex_to_int(hex_folder)
                        if hex_int is None:
                            continue

                        hex_mod = hex_int % 16
                        if not modulo_range_min <= hex_mod <= modulo_range_max:
                            continue

                        # After finding hash1, recursively find all .bin files below it
                        # No need to check for specific hash2 structure - any structure below hash1 is valid
                        try:
                            for root, dirs, files in os.walk(hash1_dir.path):
                                for file_name in files:
                                    if file_name.endswith(".bin"):
                                        yield Path(root) / file_name
                        except (OSError, PermissionError):
                            continue


def crawler_process(
    process_id: int,
    hex_modulo_range: Tuple[int, int],
    cache_path: Path,
    config_dict: dict,
    deletion_event: multiprocessing.Event,
    file_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
):
    """
    Crawler process (P1-P8): Discovers files and queues them for deletion.

    Queueing strategy:
    - When deletion is OFF: Queue files until queue size >= MINQ (pre-fill for fast start)
    - When deletion is ON: Queue files until queue size >= MAXQ (maximize throughput)

    Uses streaming discovery to avoid memory accumulation.
    """
    process_num = process_id + 1  # P1-P8
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict.get("log_level", "INFO"), process_num, log_file)
    logger = logging.getLogger(f"crawler_{process_num}")

    modulo_range_min, modulo_range_max = hex_modulo_range
    min_queue_size = config_dict.get("file_queue_min_size", 1000)
    max_queue_size = config_dict.get("file_queue_maxsize", 10000)
    access_time_threshold_seconds = (
        config_dict.get("file_access_time_threshold_minutes", 15.0) * 60.0
    )

    # Convert decimal range to hex characters for clarity
    if modulo_range_min == modulo_range_max:
        hex_chars = f"'{format(modulo_range_min, 'x')}'"
    else:
        hex_chars = f"'{format(modulo_range_min, 'x')}'-'{format(modulo_range_max, 'x')}'"
    
    # Log crawler startup information
    logger.info(
        f"Crawler P{process_num} started - hex %16 in [{modulo_range_min}, {modulo_range_max}] (hex: {hex_chars})"
    )
    logger.info(
        f"Crawler P{process_num} queue limits: MINQ={min_queue_size} (when OFF), MAXQ={max_queue_size} (when ON)"
    )

    files_discovered = 0
    files_queued = 0
    files_skipped = 0
    last_heartbeat_time = time.time()
    heartbeat_interval = 30.0  # Log heartbeat every 30 seconds

    def get_queue_size() -> int:
        """Get approximate queue size (non-blocking)."""
        try:
            return file_queue.qsize()
        except Exception:
            return 0

    def log_timing(event_type: str, duration_ms: float, **kwargs):
        """Log timing event."""
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
            # Stream files from assigned hex range
            for file_path in stream_cache_files(cache_path, hex_modulo_range):
                files_discovered += 1
                current_time = time.time()

                # Check file access time - skip recently accessed files
                # Note: relatime filesystem may not update atime on every access
                # This can cause false positives (deleting "hot" files)
                try:
                    file_stat = file_path.stat()
                    file_atime = file_stat.st_atime  # Last access time
                    time_since_access = current_time - file_atime

                    if time_since_access < access_time_threshold_seconds:
                        # File was accessed recently - skip it
                        files_skipped += 1
                        continue
                except (OSError, AttributeError) as e:
                    # If we can't stat the file (deleted, permission error, etc.), skip it
                    continue

                # Determine target queue size based on deletion state
                if deletion_event.is_set():
                    # Deletion is ON: fill up to MAXQ
                    target_size = max_queue_size
                    queue_size = get_queue_size()

                    if queue_size >= target_size:
                        # Queue is full - slow down
                        time.sleep(0.1)
                        continue
                else:
                    # Deletion is OFF: pre-fill up to MINQ (for fast start when triggered)
                    target_size = min_queue_size
                    queue_size = get_queue_size()

                    if queue_size >= target_size:
                        # Queue is pre-filled - just discover, don't queue
                        if files_discovered % 10000 == 0:
                            logger.debug(
                                f"Crawler P{process_num} pre-fill complete: queue={queue_size}/{target_size}, discovered={files_discovered}"
                            )
                        continue

                # Queue the file
                try:
                    file_queue.put(str(file_path), timeout=1.0)
                    files_queued += 1

                    # Log progress periodically
                    if files_queued % 1000 == 0:
                        queue_size = get_queue_size()
                        deletion_state = "ON" if deletion_event.is_set() else "OFF"
                        logger.debug(
                            f"Queued {files_queued} files "
                            f"(discovered {files_discovered}, queue={queue_size}/{target_size}, deletion={deletion_state})"
                        )

                    # Log every 10000 files discovered (even if not queued)
                    if files_discovered % 10000 == 0 and files_discovered > 0:
                        queue_size = get_queue_size()
                        deletion_state = "ON" if deletion_event.is_set() else "OFF"
                        logger.debug(
                            f"Discovered {files_discovered} files total "
                            f"(queued {files_queued}, queue={queue_size}, deletion={deletion_state})"
                        )
                except Exception:
                    # Queue full or timeout - continue discovering
                    time.sleep(0.1)

            # If we've scanned everything, wait a bit before rescanning
            time.sleep(1.0)

            # Periodic heartbeat log (even when no files found)
            current_time = time.time()
            if current_time - last_heartbeat_time >= heartbeat_interval:
                queue_size = get_queue_size()
                deletion_state = "ON" if deletion_event.is_set() else "OFF"
                logger.debug(
                    f"Heartbeat: discovered={files_discovered}, queued={files_queued}, "
                    f"skipped={files_skipped}, queue={queue_size}, deletion={deletion_state}"
                )
                last_heartbeat_time = current_time

    except Exception as e:
        logger.error(f"Crawler P{process_num} error: {e}", exc_info=True)
    finally:
        logger.info(
            f"Crawler P{process_num} stopping - discovered {files_discovered}, queued {files_queued}, skipped {files_skipped}"
        )


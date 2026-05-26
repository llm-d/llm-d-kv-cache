"""Crawler process for discovering and queuing cache files."""

import logging
import multiprocessing
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

from utils.logging_helpers import send_stats_to_queue
from utils.system import setup_logging

# Module-level logger for functions
logger = logging.getLogger(__name__)

# Pattern for a rank-suffixed FileMapper base directory: <safe_model>_<sha>_r<rank>
_RANK_DIR_RE = re.compile(r"_r\d+$")

# Constants for hex modulo load balancing
HEX_MODULO_BASE = 16  # Number of possible hex modulo values (0-15)

# Constants for timing and intervals
MINUTES_TO_SECONDS = 60.0  # Conversion factor from minutes to seconds
QUEUE_FULL_SLEEP_SECONDS = 0.1  # Sleep duration when queue is full
DISCOVERY_LOG_INTERVAL = 10000  # Log every N files discovered
QUEUE_PUT_TIMEOUT_SECONDS = 0.1  # Timeout for non-blocking queue put


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


def hex_to_int(hex_str: str) -> int | None:
    """Convert hex string to integer."""
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def get_hex_modulo_ranges(num_processes: int = 8) -> list[tuple[int, int]]:
    """
    Get hex modulo ranges for each crawler process.

    Valid num_processes: Powers of 2 from 1 to HEX_MODULO_BASE (1, 2, 4, 8, 16)
    Divides the 16 possible hex modulo values (0-15) evenly across processes.

    Examples:
    - 1 process:  %16 in [0, 15] (all values)
    - 2 processes: %16 in [0, 7] and [8, 15]
    - 4 processes: %16 in [0, 3], [4, 7], [8, 11], [12, 15]
    - 8 processes: %16 in [0, 1], [2, 3], ..., [14, 15]
    - 16 processes: %16 in [0], [1], ..., [15] (one value each)
    """
    # Generate valid counts: all powers of 2 from 1 to HEX_MODULO_BASE
    import math

    valid_counts = [2**i for i in range(int(math.log2(HEX_MODULO_BASE)) + 1)]
    if num_processes not in valid_counts:
        raise ValueError(f"NUM_CRAWLER_PROCESSES must be a power of 2 from 1 to {HEX_MODULO_BASE}, got {num_processes}")

    ranges = []
    values_per_process = HEX_MODULO_BASE // num_processes

    for i in range(num_processes):
        modulo_range_min = i * values_per_process
        modulo_range_max = modulo_range_min + values_per_process - 1
        ranges.append((modulo_range_min, modulo_range_max))

    return ranges


def _iter_rank_dirs(cache_path: Path) -> Iterator[os.DirEntry]:
    """
    Recursively yield FileMapper rank directories under cache_path.

    Since #585 the layout is <root>/<safe_model_name>_<sha256>_r<rank>/, where
    <safe_model_name> contains '_'-flattened HuggingFace IDs like
    'Qwen_Qwen3-7B'. Rank directories are recognized purely by the trailing
    '_r<digits>' suffix on a directory name, regardless of how deep they sit.
    """
    stack: list[str] = [str(cache_path)]
    while stack:
        current = stack.pop()
        for entry in safe_scandir(current):
            # follow_symlinks=False: avoid unbounded recursion if a symlink
            # cycle is present under the cache root.
            if not entry.is_dir(follow_symlinks=False):
                continue
            if _RANK_DIR_RE.search(entry.name):
                yield entry
            else:
                stack.append(entry.path)


def stream_cache_files_with_mapper(cache_path: Path, hex_modulo_range: tuple[int, int] | None = None) -> Iterator[Path]:
    """
    Stream cache files under the collapsed FileMapper layout introduced in #585.

    On-disk layout:
        <root>/<safe_model_name>_<sha256-12>_r<rank>/<hhh>/<hh>_g<group_idx>/*.bin

    The walker recognizes rank directories by the '_r<digits>' suffix, then
    iterates the first-level hex bucket ({hhh}, three hex chars), filters by
    hex_modulo_range, and yields *.bin files from any second-level bucket
    underneath (typically {hh}_g{group_idx}, but kept agnostic so we don't
    depend on the group-index encoding).

    Yields Path objects.
    """
    if not cache_path.exists():
        logger.warning(f"cache_path does not exist: {cache_path}")
        return

    modulo_range_min, modulo_range_max = hex_modulo_range if hex_modulo_range else (0, HEX_MODULO_BASE - 1)

    for rank_dir in _iter_rank_dirs(cache_path):
        # Iterate first-level hex buckets ({hhh}, three hex chars).
        for hex3_dir in safe_scandir(rank_dir.path):
            if not hex3_dir.is_dir() or len(hex3_dir.name) != 3:
                continue

            # Apply hex modulo filtering for load balancing across crawlers.
            hex_int = hex_to_int(hex3_dir.name)
            if hex_int is None:
                continue
            hex_mod = hex_int % HEX_MODULO_BASE
            if not (modulo_range_min <= hex_mod <= modulo_range_max):
                continue

            # Iterate second-level buckets ({hh}_g{group_idx} or similar).
            for hex2_dir in safe_scandir(hex3_dir.path):
                if not hex2_dir.is_dir():
                    continue

                for bin_file_entry in safe_scandir(hex2_dir.path):
                    if bin_file_entry.is_file() and bin_file_entry.name.endswith(".bin"):
                        yield Path(bin_file_entry.path)


def crawler_process(
    process_id: int,
    hex_modulo_range: tuple[int, int],
    cache_path: Path,
    config_dict: dict,
    deletion_event: multiprocessing.Event,
    file_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
):
    """
    Crawler process (P1-PN): Discovers files and queues them for deletion.

    Uses streaming discovery to avoid memory accumulation.
    """
    process_num = process_id + 1  # P1-PN
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict["log_level"], process_num, log_file)
    logger = logging.getLogger(f"crawler_{process_num}")

    modulo_range_min, modulo_range_max = hex_modulo_range
    min_queue_size = config_dict["file_queue_min_size"]
    max_queue_size = config_dict["file_queue_maxsize"]
    access_time_threshold_seconds = config_dict["file_access_time_threshold_minutes"] * MINUTES_TO_SECONDS

    # Convert decimal range to hex characters for clarity
    if modulo_range_min == modulo_range_max:
        hex_chars = f"'{format(modulo_range_min, 'x')}'"
    else:
        hex_chars = f"'{format(modulo_range_min, 'x')}'-'{format(modulo_range_max, 'x')}'"

    # Log crawler startup information
    logger.info(
        f"Crawler P{process_num} started - hex %{HEX_MODULO_BASE} "
        f"in [{modulo_range_min}, {modulo_range_max}] (hex: {hex_chars})"
    )
    logger.info(
        f"Crawler P{process_num} queue limits: MINQ={min_queue_size} (when OFF), MAXQ={max_queue_size} (when ON)"
    )
    logger.info(
        f"Crawler P{process_num} hex_modulo_range: "
        f"{hex_modulo_range[0]}-{hex_modulo_range[1]} (hex mod {HEX_MODULO_BASE})"
    )

    logger.info(f"Crawler P{process_num} using FileMapper cache structure")

    files_discovered = 0
    files_queued = 0
    files_skipped = 0
    files_skipped_stat_error = 0
    stat_error_samples = []  # Store first few stat errors for logging
    max_stat_error_samples = 3
    last_stats_send_time = time.time()

    def get_queue_size() -> int:
        """Get approximate queue size (non-blocking)."""
        try:
            return file_queue.qsize()
        except Exception:
            return 0

    try:
        while not shutdown_event.is_set():
            # Stream files from assigned hex range using FileMapper
            file_stream = stream_cache_files_with_mapper(cache_path, hex_modulo_range)

            for file_path in file_stream:
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
                    files_skipped_stat_error += 1
                    # Log first few errors with details for diagnostics
                    if len(stat_error_samples) < max_stat_error_samples:
                        stat_error_samples.append(f"{file_path}: {type(e).__name__}: {e}")
                    continue

                # Determine target queue size based on deletion state
                if deletion_event.is_set():
                    # Deletion is ON: fill up to MAXQ
                    target_size = max_queue_size
                    queue_size = get_queue_size()

                    if queue_size >= target_size:
                        # Queue is full - slow down
                        time.sleep(QUEUE_FULL_SLEEP_SECONDS)
                        continue
                else:
                    # Deletion is OFF: pre-fill up to MINQ (for fast start when triggered)
                    target_size = min_queue_size
                    queue_size = get_queue_size()

                    if queue_size >= target_size:
                        # Queue is pre-filled - just discover, don't queue
                        if files_discovered % DISCOVERY_LOG_INTERVAL == 0:
                            logger.debug(
                                f"Crawler P{process_num} pre-fill complete: "
                                f"queue={queue_size}/{target_size}, "
                                f"discovered={files_discovered}"
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
                            f"(discovered {files_discovered}, "
                            f"queue={queue_size}/{target_size}, "
                            f"deletion={deletion_state})"
                        )

                    # Log every N files discovered (even if not queued)
                    if files_discovered % DISCOVERY_LOG_INTERVAL == 0 and files_discovered > 0:
                        queue_size = get_queue_size()
                        deletion_state = "ON" if deletion_event.is_set() else "OFF"
                        logger.debug(
                            f"Discovered {files_discovered} files total "
                            f"(queued {files_queued}, queue={queue_size}, deletion={deletion_state})"
                        )
                except Exception:
                    # Queue full or timeout - continue discovering
                    time.sleep(QUEUE_FULL_SLEEP_SECONDS)

            # If we've scanned everything, wait a bit before rescanning
            time.sleep(1.0)

            # Send stats to result_queue for aggregated logging
            queue_size = get_queue_size()
            last_stats_send_time = send_stats_to_queue(
                result_queue,
                "crawler_stats",
                process_num,
                {
                    "files_discovered": files_discovered,
                    "files_queued": files_queued,
                    "files_skipped": files_skipped,
                    "files_skipped_stat_error": files_skipped_stat_error,
                    "queue_size": queue_size,
                    "deletion_active": deletion_event.is_set(),
                },
                last_stats_send_time,
            )

    except Exception as e:
        logger.exception(f"Crawler P{process_num} error: {e}")
    finally:
        logger.info(
            f"Crawler P{process_num} stopping - discovered {files_discovered}, queued {files_queued}, "
            f"skipped {files_skipped} (access_time), skipped_stat_error {files_skipped_stat_error}"
        )

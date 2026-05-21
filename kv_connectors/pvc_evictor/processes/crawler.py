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

# Constants for hex modulo load balancing
HEX_MODULO_BASE = 16  # Number of possible hex modulo values (0-15)

# Constants for timing and intervals
MINUTES_TO_SECONDS = 60.0  # Conversion factor from minutes to seconds
QUEUE_FULL_SLEEP_SECONDS = 0.1  # Sleep duration when queue is full
DISCOVERY_LOG_INTERVAL = 10000  # Log every N files discovered

# llmd_fs_backend flat layout (post FileMapper v0.20+):
#   <cache_path>/<model>_<digest>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin
RANK_DIR_PATTERN = re.compile(r".*_r\d+$", re.IGNORECASE)
GROUP_DIR_PATTERN = re.compile(r"^[0-9a-f]{2}_g\d+$", re.IGNORECASE)


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


def is_rank_dir(name: str) -> bool:
    """True if directory name is a per-rank KV data folder (*_r<N>)."""
    return bool(RANK_DIR_PATTERN.match(name))


def is_hex3_dir(name: str) -> bool:
    """True if name is a 3-character hex prefix directory (hhh)."""
    if len(name) != 3:
        return False
    return hex_to_int(name) is not None


def is_group_dir(name: str) -> bool:
    """True if name matches <hh>_g<group_idx> (case-insensitive)."""
    return bool(GROUP_DIR_PATTERN.match(name))


def hex_mod_in_range(
    hex3_name: str,
    modulo_range_min: int,
    modulo_range_max: int,
) -> bool:
    """True if int(hex3, 16) % 16 falls within [modulo_range_min, modulo_range_max]."""
    hex_int = hex_to_int(hex3_name)
    if hex_int is None:
        return False
    hex_mod = hex_int % HEX_MODULO_BASE
    return modulo_range_min <= hex_mod <= modulo_range_max


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


def stream_cache_files(
    cache_path: Path, hex_modulo_range: tuple[int, int] | None = None
) -> Iterator[Path]:
    """
    Stream KV cache .bin files using the flat fs_backend on-disk layout.

    Layout (path-only, no FileMapper import):
        <cache_path>/<model>_<digest>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin

    Base fingerprint dirs (<model>_<digest>/ with config.json only) are skipped.
    """
    if not cache_path.exists():
        logger.warning(f"Crawler: cache_path does not exist: {cache_path}")
        return

    modulo_range_min, modulo_range_max = hex_modulo_range if hex_modulo_range else (0, 15)

    for entry in safe_scandir(str(cache_path)):
        if not entry.is_dir() or not is_rank_dir(entry.name):
            continue

        for hex3_dir in safe_scandir(entry.path):
            if not hex3_dir.is_dir() or not is_hex3_dir(hex3_dir.name):
                continue

            if not hex_mod_in_range(hex3_dir.name, modulo_range_min, modulo_range_max):
                continue

            for group_dir in safe_scandir(hex3_dir.path):
                if not group_dir.is_dir() or not is_group_dir(group_dir.name):
                    continue

                for bin_entry in safe_scandir(group_dir.path):
                    if bin_entry.is_file() and bin_entry.name.endswith(".bin"):
                        yield Path(bin_entry.path)


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

    if not cache_path.exists():
        logger.error(
            f"Crawler P{process_num} cache_path does not exist: {cache_path}"
        )
        return

    # Convert decimal range to hex characters for clarity
    if modulo_range_min == modulo_range_max:
        hex_chars = f"'{format(modulo_range_min, 'x')}'"
    else:
        hex_chars = f"'{format(modulo_range_min, 'x')}'-'{format(modulo_range_max, 'x')}'"

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
    logger.info(
        f"Crawler P{process_num} using flat fs_backend cache layout at {cache_path}"
    )

    files_discovered = 0
    files_queued = 0
    files_skipped = 0
    files_skipped_stat_error = 0
    stat_error_samples = []
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
            file_stream = stream_cache_files(cache_path, hex_modulo_range)

            for file_path in file_stream:
                files_discovered += 1
                current_time = time.time()

                try:
                    file_stat = file_path.stat()
                    file_atime = file_stat.st_atime
                    time_since_access = current_time - file_atime

                    if time_since_access < access_time_threshold_seconds:
                        files_skipped += 1
                        continue
                except (OSError, AttributeError) as e:
                    files_skipped_stat_error += 1
                    if len(stat_error_samples) < max_stat_error_samples:
                        stat_error_samples.append(f"{file_path}: {type(e).__name__}: {e}")
                    continue

                if deletion_event.is_set():
                    target_size = max_queue_size
                    queue_size = get_queue_size()

                    if queue_size >= target_size:
                        time.sleep(QUEUE_FULL_SLEEP_SECONDS)
                        continue
                else:
                    target_size = min_queue_size
                    queue_size = get_queue_size()

                    if queue_size >= target_size:
                        if files_discovered % DISCOVERY_LOG_INTERVAL == 0:
                            logger.debug(
                                f"Crawler P{process_num} pre-fill complete: "
                                f"queue={queue_size}/{target_size}, "
                                f"discovered={files_discovered}"
                            )
                        continue

                try:
                    file_queue.put(str(file_path), timeout=1.0)
                    files_queued += 1

                    if files_queued % 1000 == 0:
                        queue_size = get_queue_size()
                        deletion_state = "ON" if deletion_event.is_set() else "OFF"
                        logger.debug(
                            f"Queued {files_queued} files "
                            f"(discovered {files_discovered}, "
                            f"queue={queue_size}/{target_size}, "
                            f"deletion={deletion_state})"
                        )

                    if files_discovered % DISCOVERY_LOG_INTERVAL == 0 and files_discovered > 0:
                        queue_size = get_queue_size()
                        deletion_state = "ON" if deletion_event.is_set() else "OFF"
                        logger.debug(
                            f"Discovered {files_discovered} files total "
                            f"(queued {files_queued}, queue={queue_size}, deletion={deletion_state})"
                        )
                except Exception:
                    time.sleep(QUEUE_FULL_SLEEP_SECONDS)

            time.sleep(1.0)

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

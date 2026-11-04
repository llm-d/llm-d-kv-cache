"""Crawler process for discovering and queuing cache files."""

import contextlib
import logging
import multiprocessing
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from utils.logging_helpers import send_stats_to_queue
from utils.system import setup_logging
from utils.tmp_files import (
    is_block_tmp_file,
    is_probe_tmp_file,
    is_stale_tmp_file,
)

from processes.deleter import delete_batch

# Module-level logger for functions
logger = logging.getLogger(__name__)

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
    Recursively yield directories that directly contain first-level hex buckets.

    Supports both:
    - New layout: <root>/<safe_model_name>_<sha256>_r<rank>/<hhh>/
    - Old layout: <root>/<model>/.../rank_<rank>/<dtype>/<hhh>/
    """
    stack: list[str] = [str(cache_path)]
    while stack:
        current = stack.pop()
        for entry in safe_scandir(current):
            # follow_symlinks=False: avoid unbounded recursion if a symlink
            # cycle is present under the cache root.
            if not entry.is_dir(follow_symlinks=False):
                continue

            # Detect if this directory directly contains valid first-level hex buckets.
            # We check if any of its subdirectories are valid hex buckets (e.g., len 2, 3, or 4).
            is_hex_parent = False
            for sub_entry in safe_scandir(entry.path):
                if sub_entry.is_dir() and len(sub_entry.name) in (2, 3, 4) and hex_to_int(sub_entry.name) is not None:
                    is_hex_parent = True
                    break

            if is_hex_parent:
                yield entry
            else:
                stack.append(entry.path)


def iter_probe_tmp_files(cache_path: Path) -> Iterator[str]:
    """Yield O_DIRECT probe temp files left under the cache root.

    vLLM's ``probe_o_direct`` writes ``.o_direct_probe_<random>.tmp`` beside the
    model's ``config.json``, i.e. in the per-model base directory that sits
    alongside the ``_r<rank>`` directories rather than inside a hex bucket. Both
    the root and its immediate children are scanned, so the file is found
    whichever of the two the deployment uses.

    These files are outside the hex-bucket space the crawlers shard on, so only
    one crawler should call this (see ``owns_probe_cleanup``) to avoid several
    processes racing on the same path.
    """
    for entry in safe_scandir(str(cache_path)):
        if entry.is_file(follow_symlinks=False):
            if is_probe_tmp_file(entry.name):
                yield entry.path
        elif entry.is_dir(follow_symlinks=False):
            for sub_entry in safe_scandir(entry.path):
                if sub_entry.is_file(follow_symlinks=False) and is_probe_tmp_file(sub_entry.name):
                    yield sub_entry.path


def sweep_stale_tmp_files(
    candidates: list[str],
    *,
    cache_path: Path,
    min_age_seconds: float,
    batch_size: int,
    dry_run: bool,
    logger: logging.Logger,
    folder_queue: Any = None,
    include_probe_files: bool = False,
    now: float | None = None,
) -> tuple[int, int]:
    """Remove the stale temp files among ``candidates``; return (deleted, skipped_recent).

    Deliberately independent of the deletion event: an abandoned temp file is
    garbage whether or not the PVC is under pressure, and it keeps its bucket
    directory from being reaped. Staleness is re-checked here, immediately
    before removal, so a file a writer touched during the sweep is left alone.

    Deletion reuses the deleter's batch path, so DRY_RUN is honoured and
    failures are logged and non-fatal: anything left behind is retried on the
    next sweep. Temp files never carry a block hash, so no BlockRemoved event
    is published for them.
    """
    if min_age_seconds <= 0:
        return 0, 0

    paths = list(candidates)
    if include_probe_files:
        paths.extend(iter_probe_tmp_files(cache_path))
    if not paths:
        return 0, 0

    if now is None:
        now = time.time()
    stale = [p for p in paths if is_stale_tmp_file(p, min_age_seconds, now)]
    skipped_recent = len(paths) - len(stale)

    deleted = 0
    for start in range(0, len(stale), max(batch_size, 1)):
        batch = stale[start : start + max(batch_size, 1)]
        removed, _, _ = delete_batch(batch, dry_run, logger, folder_queue=folder_queue)
        deleted += removed

    if deleted or skipped_recent:
        logger.debug(
            f"Temp cleanup: deleted={deleted}, skipped_recent={skipped_recent}, "
            f"min_age={min_age_seconds / MINUTES_TO_SECONDS:.1f}min"
        )
    return deleted, skipped_recent


def is_dir_empty(dir_path: str) -> bool:
    """Check if a directory is completely empty."""
    try:
        with os.scandir(dir_path) as entries:
            for _ in entries:
                return False
        return True
    except (OSError, PermissionError):
        return False


def queue_folder(
    folder_path: str,
    folder_queue: Any,
    on_empty_folder_discovered: Any,
    min_age_seconds: float = 0.0,
):
    """Offer an empty folder to the background cleaner.

    Skips folders modified within ``min_age_seconds`` to avoid racing a writer
    that just created the directory and is about to populate it. The cleaner
    only rmdir's empty directories, so this age guard is defense-in-depth on
    top of that: it keeps freshly-created, about-to-be-written buckets out of
    the cleanup queue entirely. A folder filtered out here is simply
    re-evaluated on the next crawl sweep.
    """
    if min_age_seconds > 0.0:
        try:
            if time.time() - os.stat(folder_path).st_mtime < min_age_seconds:
                return
        except OSError:
            # Vanished or unreadable - nothing to clean up.
            return
    if on_empty_folder_discovered:
        on_empty_folder_discovered(folder_path)
    if folder_queue is not None:
        with contextlib.suppress(Exception):
            folder_queue.put_nowait(folder_path)


def stream_cache_files_with_mapper(
    cache_path: Path,
    hex_modulo_range: tuple[int, int] | None = None,
    hex_bucket_len: int = 3,
    on_empty_folder_discovered: Any = None,
    folder_queue: Any = None,
    dir_cleanup_ttl_seconds: float = 0.0,
    on_tmp_file_discovered: Any = None,
) -> Iterator[Path]:
    """
    Stream cache files under the collapsed FileMapper layout introduced in #585.

    On-disk layout:
        <root>/<safe_model_name>_<sha256-12>_r<rank>/<hex_bucket_len-chars>/<hh>_g<group_idx>/*.bin

    The walker recognizes rank directories by the '_r<digits>' suffix, then
    iterates the first-level hex bucket (hex_bucket_len hex chars), filters by
    hex_modulo_range, and yields *.bin files from any second-level bucket
    underneath (typically {hh}_g{group_idx}, but kept agnostic so we don't
    depend on the group-index encoding).

    Empty directories encountered along the way are offered to the folder
    cleaner via folder_queue, subject to the dir_cleanup_ttl_seconds age guard.

    If on_tmp_file_discovered is given, vLLM block temp files
    (``<hash>.bin_<random>.tmp``) seen in the same scandir pass are reported to
    it instead of being yielded, so they never enter the disk-pressure deletion
    queue. Staleness is decided by the caller.

    Yields Path objects.
    """
    if not cache_path.exists():
        logger.warning(f"cache_path does not exist: {cache_path}")
        return

    modulo_range_min, modulo_range_max = hex_modulo_range if hex_modulo_range else (0, HEX_MODULO_BASE - 1)

    for rank_dir in _iter_rank_dirs(cache_path):
        # If the rank directory itself is empty, queue it!
        if is_dir_empty(rank_dir.path):
            queue_folder(rank_dir.path, folder_queue, on_empty_folder_discovered, dir_cleanup_ttl_seconds)
            continue

        has_hex3_dirs = False
        # Iterate first-level hex buckets (hex_bucket_len hex chars).
        for hex3_dir in safe_scandir(rank_dir.path):
            if not hex3_dir.is_dir() or len(hex3_dir.name) != hex_bucket_len:
                continue
            has_hex3_dirs = True

            # If the hex3 directory itself is empty, queue it!
            if is_dir_empty(hex3_dir.path):
                queue_folder(hex3_dir.path, folder_queue, on_empty_folder_discovered, dir_cleanup_ttl_seconds)
                continue

            # Apply hex modulo filtering for load balancing across crawlers.
            hex_int = hex_to_int(hex3_dir.name)
            if hex_int is None:
                continue
            hex_mod = hex_int % HEX_MODULO_BASE
            if not (modulo_range_min <= hex_mod <= modulo_range_max):
                continue

            has_hex2_dirs = False
            # Iterate second-level buckets ({hh}_g{group_idx} or similar).
            for hex2_dir in safe_scandir(hex3_dir.path):
                if not hex2_dir.is_dir():
                    continue
                has_hex2_dirs = True

                has_bin_files = False
                for bin_file_entry in safe_scandir(hex2_dir.path):
                    if bin_file_entry.is_file() and bin_file_entry.name.endswith(".bin"):
                        has_bin_files = True
                        yield Path(bin_file_entry.path)
                    elif on_tmp_file_discovered is not None and is_block_tmp_file(bin_file_entry.name):
                        # Reported, not yielded: a temp file is not a cache block,
                        # so it must not reach the deleter's queue or its events.
                        on_tmp_file_discovered(bin_file_entry.path)

                # A bucket holding only a stale temp file still counts as having no
                # .bin files; it stays un-rmdir-able until temp cleanup removes the
                # file, after which a later sweep reaps the directory.
                if not has_bin_files:
                    queue_folder(hex2_dir.path, folder_queue, on_empty_folder_discovered, dir_cleanup_ttl_seconds)

            if not has_hex2_dirs:
                queue_folder(hex3_dir.path, folder_queue, on_empty_folder_discovered, dir_cleanup_ttl_seconds)

        if not has_hex3_dirs:
            queue_folder(rank_dir.path, folder_queue, on_empty_folder_discovered, dir_cleanup_ttl_seconds)


def crawler_process(
    process_id: int,
    hex_modulo_range: tuple[int, int],
    cache_path: Path,
    config_dict: dict,
    deletion_event: multiprocessing.Event,
    file_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
    folder_queue: Any = None,
):
    """
    Crawler process (P1-PN): Discovers files and queues them for deletion.

    Uses streaming discovery to avoid memory accumulation.
    """
    process_num = process_id + 1  # P1-PN
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict.get("log_level", "INFO"), process_num, log_file)
    logger = logging.getLogger(f"crawler_{process_num}")

    modulo_range_min, modulo_range_max = hex_modulo_range
    min_queue_size = config_dict["file_queue_min_size"]
    max_queue_size = config_dict["file_queue_maxsize"]
    batch_size = config_dict["deletion_batch_size"]
    dry_run = config_dict["dry_run"]
    access_time_threshold_seconds = config_dict["file_access_time_threshold_minutes"] * MINUTES_TO_SECONDS
    enable_tmp_cleanup = config_dict.get("enable_tmp_cleanup", False)
    tmp_min_age_seconds = config_dict.get("tmp_file_min_age_minutes", 0.0) * MINUTES_TO_SECONDS
    # Probe temp files live outside the hex-bucket space every crawler shards on,
    # so exactly one crawler owns them: the one holding hex modulo 0.
    owns_probe_cleanup = enable_tmp_cleanup and modulo_range_min == 0

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
    empty_folders_queued = 0
    tmp_files_deleted = 0
    tmp_files_skipped_recent = 0
    tmp_candidates: list[str] = []
    stat_error_samples = []  # Store first few stat errors for logging
    max_stat_error_samples = 3
    last_stats_send_time = time.time()

    def get_queue_size() -> int:
        """Get approximate queue size (non-blocking)."""
        try:
            return file_queue.qsize()
        except Exception:
            return 0

    def on_empty_folder(*args, **kwargs):
        # Counts empty dirs this crawler discovered and handed to the folder
        # cleaner; the cleaner performs the actual rmdir.
        nonlocal empty_folders_queued
        empty_folders_queued += 1

    def on_tmp_file(path: str):
        # Collected during the sweep and acted on once it finishes, so we never
        # unlink while a scandir over the same directory is still open.
        tmp_candidates.append(path)

    def sweep_tmp_files() -> tuple[int, int]:
        if not enable_tmp_cleanup:
            return 0, 0
        return sweep_stale_tmp_files(
            tmp_candidates,
            cache_path=cache_path,
            min_age_seconds=tmp_min_age_seconds,
            batch_size=batch_size,
            dry_run=dry_run,
            logger=logger,
            folder_queue=folder_queue,
            include_probe_files=owns_probe_cleanup,
        )

    try:
        while not shutdown_event.is_set():
            # Stream files from assigned hex range using FileMapper
            hex_bucket_len = config_dict.get("hex_bucket_len", 3)
            file_stream = stream_cache_files_with_mapper(
                cache_path,
                hex_modulo_range,
                hex_bucket_len=hex_bucket_len,
                on_empty_folder_discovered=on_empty_folder,
                folder_queue=folder_queue,
                dir_cleanup_ttl_seconds=config_dict.get("dir_cleanup_ttl_seconds", 0.0),
                on_tmp_file_discovered=on_tmp_file if enable_tmp_cleanup else None,
            )
            tmp_candidates.clear()

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

            # The sweep is finished, so no scandir is open on those directories.
            swept, swept_skipped = sweep_tmp_files()
            tmp_files_deleted += swept
            tmp_files_skipped_recent += swept_skipped

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
                    "empty_folders_queued": empty_folders_queued,
                    "tmp_files_deleted": tmp_files_deleted,
                    "tmp_files_skipped_recent": tmp_files_skipped_recent,
                    "queue_size": queue_size,
                    "deletion_active": deletion_event.is_set(),
                },
                last_stats_send_time,
            )

    except Exception as e:
        logger.exception(f"Crawler P{process_num} error: {e}")
    finally:
        logger.info(
            f"Crawler P{process_num} stopping - discovered {files_discovered}, "
            f"queued {files_queued}, skipped {files_skipped} (access_time), "
            f"skipped_stat_error {files_skipped_stat_error}, "
            f"queued {empty_folders_queued} empty folders for cleanup, "
            f"deleted {tmp_files_deleted} stale temp files"
        )

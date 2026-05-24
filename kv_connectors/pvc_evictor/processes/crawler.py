"""Crawler process for discovering and queuing cache files."""

import logging
import multiprocessing
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from utils.logging_helpers import send_stats_to_queue
from utils.system import setup_logging

# Mock vllm module to avoid heavyweight dependency in evictor container
import sys
from types import ModuleType
vllm = ModuleType("vllm")
vllm.__path__ = []  # Declare as package

vllm.logger = ModuleType("vllm.logger")
vllm.logger.init_logger = lambda name: logging.getLogger(name)

vllm.config = ModuleType("vllm.config")
vllm.config.VllmConfig = object

vllm.v1 = ModuleType("vllm.v1")
vllm.v1.__path__ = []  # Declare as package

vllm.v1.kv_cache_interface = ModuleType("vllm.v1.kv_cache_interface")
vllm.v1.kv_cache_interface.KVCacheConfig = object

vllm.v1.kv_offload = ModuleType("vllm.v1.kv_offload")
vllm.v1.kv_offload.__path__ = []  # Declare as package

vllm.v1.kv_offload.base = ModuleType("vllm.v1.kv_offload.base")
vllm.v1.kv_offload.base.OffloadKey = object
vllm.v1.kv_offload.base.get_offload_block_hash = lambda x: b""
vllm.v1.kv_offload.base.get_offload_group_idx = lambda x: 0

sys.modules["vllm"] = vllm
sys.modules["vllm.logger"] = vllm.logger
sys.modules["vllm.config"] = vllm.config
sys.modules["vllm.v1"] = vllm.v1
sys.modules["vllm.v1.kv_cache_interface"] = vllm.v1.kv_cache_interface
sys.modules["vllm.v1.kv_offload"] = vllm.v1.kv_offload
sys.modules["vllm.v1.kv_offload.base"] = vllm.v1.kv_offload.base

# FileMapper integration for canonical cache structure
try:
    from llmd_fs_backend.file_mapper import FileMapper

    FILEMAPPER_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import FileMapper: {e}", flush=True)
    FILEMAPPER_AVAILABLE = False
    FileMapper = None

# Module-level logger for functions
logger = logging.getLogger(__name__)

# Constants for hex modulo load balancing
HEX_MODULO_BASE = 4096  # Number of possible hex modulo values (0-4095)

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


def is_dir_empty(dir_path: str) -> bool:
    """Check if a directory is completely empty."""
    try:
        with os.scandir(dir_path) as entries:
            for _ in entries:
                return False
        return True
    except (OSError, PermissionError):
        return False


def queue_folder(folder_path: str, folder_queue: Any, on_empty_folder_discovered: Any, ttl_seconds: float = 120.0):
    """Safely push empty folder to queue in background if it is older than ttl_seconds."""
    try:
        folder_stat = os.stat(folder_path)
        folder_age = time.time() - folder_stat.st_mtime
        if folder_age < ttl_seconds:
            # Directory is newly created/modified, skip queueing it to prevent race conditions
            return
    except OSError:
        # If we can't stat the directory, it might be deleted already or permission error, skip it
        return

    if on_empty_folder_discovered:
        on_empty_folder_discovered(folder_path)
    if folder_queue:
        try:
            folder_queue.put_nowait(folder_path)
        except Exception:
            pass


def hex_to_int(hex_str: str) -> int | None:
    """Convert hex string to integer."""
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def parse_filemapper_params(dir_name: str, pattern: str) -> dict:
    """
    Parse FileMapper parameters from directory name.

    Examples:
        parse_filemapper_params("block_size_16_blocks_per_file_256",
                               "block_size_{gpu_block_size}_blocks_per_file_{gpu_blocks_per_file}")
        -> {"gpu_block_size": 16, "gpu_blocks_per_file": 256}
    """
    # Convert pattern to regex, replacing {X} with named capture groups
    regex_pattern = pattern
    param_names = re.findall(r"\{(\w+)\}", pattern)

    for param in param_names:
        regex_pattern = regex_pattern.replace(f"{{{param}}}", f"(?P<{param}>\\d+)")

    match = re.match(regex_pattern, dir_name)
    if not match:
        return {}

    # Convert matched values to integers
    result = {}
    for param in param_names:
        value = match.group(param)
        if value:
            result[param] = int(value)

    return result


def get_hex_modulo_ranges(num_processes: int, shard_index: int = 0, total_shards: int = 1) -> list[tuple[int, int]]:
    """
    Get hex modulo ranges for each crawler process in a multi-pod sharded environment.

    Calculates ranges across total_crawlers = total_shards * num_processes.
    Divides the 4096 possible hex modulo values (0-4095) evenly across all total crawlers.
    """
    total_crawlers = total_shards * num_processes
    if total_crawlers > HEX_MODULO_BASE:
        raise ValueError(f"Total crawlers ({total_crawlers}) cannot exceed HEX_MODULO_BASE ({HEX_MODULO_BASE})")
    if not (0 <= shard_index < total_shards):
        raise ValueError(f"Invalid shard_index {shard_index} for total_shards {total_shards}")

    ranges = []
    for i in range(num_processes):
        k = shard_index * num_processes + i
        start = (k * HEX_MODULO_BASE) // total_crawlers
        end = ((k + 1) * HEX_MODULO_BASE) // total_crawlers - 1
        ranges.append((start, end))

    return ranges


def stream_cache_files_with_mapper(cache_path: Path, hex_modulo_range: tuple[int, int] | None = None, on_empty_folder_discovered: Any = None, folder_queue: Any = None, ttl_seconds: float = 120.0) -> Iterator[os.DirEntry]:
    """
    Stream cache files using the collapsed FileMapper directory structure.

    Directory structure:
    Supports old layout: <root_dir>/<safe_model_name>_<digest>_r<rank>/{hhh}/{hh}_g{group_idx}/*.bin
    Supports new layout: <root_dir>/Qwen/Qwen3.5-27B/block_size_400_blocks_per_file_1/tp_8_pp_size_1_pcp_size_1/rank_7/auto/{hhh}/{hh}_g{group_idx}/*.bin
    
    Yields os.DirEntry objects for .bin files in this layout.
    """
    if not cache_path.exists():
        logger.warning(f"FileMapper: cache_path does not exist: {cache_path}")
        return

    if not FILEMAPPER_AVAILABLE:
        logger.warning("FileMapper: FILEMAPPER_AVAILABLE is False")
        return

    modulo_range_min, modulo_range_max = hex_modulo_range if hex_modulo_range else (0, HEX_MODULO_BASE - 1)

    # Find all rank-specific directories recursively under the cache path
    # A rank directory is identified if its name matches _r\d+ or rank_\d+
    rank_paths = []
    
    def find_rank_dirs(dir_path: str):
        try:
            for entry in os.scandir(dir_path):
                if not entry.is_dir():
                    continue
                # Match both old _r<rank> and new rank_<rank>
                if re.search(r"_r\d+$", entry.name) or re.search(r"rank_\d+$", entry.name):
                    rank_paths.append(entry.path)
                    # Check if there's an 'auto' subfolder inside this rank folder (new layout)
                    auto_subpath = os.path.join(entry.path, "auto")
                    if os.path.isdir(auto_subpath):
                        rank_paths.append(auto_subpath)
                else:
                    # Recursive search down the directory tree
                    find_rank_dirs(entry.path)
                    if is_dir_empty(entry.path):
                        queue_folder(entry.path, folder_queue, on_empty_folder_discovered, ttl_seconds)
        except (OSError, PermissionError):
            pass

    find_rank_dirs(str(cache_path))
    logger.debug(f"FileMapper: Discovered {len(rank_paths)} rank-level paths for crawler processing.")

    # Process each discovered rank path
    for rank_path in rank_paths:
        if is_dir_empty(rank_path):
            queue_folder(rank_path, folder_queue, on_empty_folder_discovered, ttl_seconds)
            continue

        # Traversal optimized: Directly target assigned hex3 folders
        if hex_modulo_range:
            for i in range(modulo_range_min, modulo_range_max + 1):
                hhh = format(i, '03x')
                hex3_path_str = os.path.join(rank_path, hhh)
                if not os.path.isdir(hex3_path_str):
                    continue

                has_subdirs = False
                for hex2_dir in safe_scandir(hex3_path_str):
                    if not hex2_dir.is_dir():
                        continue
                    has_subdirs = True

                    has_bin_files = False
                    for bin_file_entry in safe_scandir(hex2_dir.path):
                        if bin_file_entry.is_file() and bin_file_entry.name.endswith(".bin"):
                            has_bin_files = True
                            yield bin_file_entry

                    if not has_bin_files:
                        queue_folder(hex2_dir.path, folder_queue, on_empty_folder_discovered, ttl_seconds)

                if not has_subdirs:
                    queue_folder(hex3_path_str, folder_queue, on_empty_folder_discovered, ttl_seconds)
        else:
            # Fallback traversal of all hex3 folders
            for hex3_dir in safe_scandir(rank_path):
                if not hex3_dir.is_dir() or len(hex3_dir.name) != 3:
                    continue
                
                has_subdirs = False
                for hex2_dir in safe_scandir(hex3_dir.path):
                    if not hex2_dir.is_dir():
                        continue
                    has_subdirs = True
                    
                    has_bin_files = False
                    for bin_file_entry in safe_scandir(hex2_dir.path):
                        if bin_file_entry.is_file() and bin_file_entry.name.endswith(".bin"):
                            has_bin_files = True
                            yield bin_file_entry
                            
                    if not has_bin_files:
                        queue_folder(hex2_dir.path, folder_queue, on_empty_folder_discovered, ttl_seconds)
                        
                if not has_subdirs:
                    queue_folder(hex3_dir.path, folder_queue, on_empty_folder_discovered, ttl_seconds)


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
    import cProfile
    import io
    import pstats

    process_num = process_id + 1  # P1-PN
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict["log_level"], process_num, log_file)
    logger = logging.getLogger(f"crawler_{process_num}")

    modulo_range_min, modulo_range_max = hex_modulo_range
    min_queue_size = config_dict["file_queue_min_size"]
    max_queue_size = config_dict["file_queue_maxsize"]
    ttl_minutes = config_dict["file_access_time_threshold_minutes"]
    ttl_seconds = ttl_minutes * MINUTES_TO_SECONDS

    # Convert decimal range to hex characters for clarity
    if modulo_range_min == modulo_range_max:
        hex_chars = f"'{format(modulo_range_min, '03x')}'"
    else:
        hex_chars = f"'{format(modulo_range_min, '03x')}'-'{format(modulo_range_max, '03x')}'"

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
    logger.info(f"Crawler P{process_num} Access Time Threshold: {ttl_minutes} minutes ({ttl_seconds}s)")

    # Verify FileMapper is available
    if not FILEMAPPER_AVAILABLE:
        logger.error(f"Crawler P{process_num} FileMapper not available - cannot proceed")
        return

    logger.info(f"Crawler P{process_num} using FileMapper cache structure")


    files_discovered = 0
    files_queued = 0
    files_skipped = 0
    files_skipped_stat_error = 0
    folders_deleted = 0
    last_stats_send_time = time.time()
    buffer = []

    def get_queue_size() -> int:
        """Get approximate queue size (non-blocking)."""
        try:
            return file_queue.qsize()
        except Exception:
            return 0

    profiler = cProfile.Profile()
    profiler.enable()
 
    def on_empty_folder(*args, **kwargs):
        nonlocal folders_deleted
        folders_deleted += 1

    try:
        while not shutdown_event.is_set():
            # Stream files from assigned hex range using FileMapper
            file_stream = stream_cache_files_with_mapper(cache_path, hex_modulo_range, on_empty_folder, folder_queue, ttl_seconds)

            for file_entry in file_stream:
                files_discovered += 1

                # Send stats periodically inside discovery loop (every 30 seconds)
                current_time = time.time()
                if current_time - last_stats_send_time >= 30.0:
                    last_stats_send_time = send_stats_to_queue(
                        result_queue,
                        "crawler_stats",
                        process_num,
                        {
                            "files_discovered": files_discovered,
                            "files_queued": files_queued,
                            "files_skipped": files_skipped,
                            "files_skipped_stat_error": files_skipped_stat_error,
                            "folders_deleted": folders_deleted,
                            "queue_size": get_queue_size(),
                            "deletion_active": deletion_event.is_set(),
                        },
                        last_stats_send_time,
                    )
                    # Periodically dump cProfile stats during active execution
                    try:
                        profiler.dump_stats(f"/kv-cache/cprofile_crawler_p{process_num}.prof")
                    except Exception as e:
                        logger.debug(f"Failed to dump periodic crawler profile: {e}")

                # Determine target queue size or LRU buffering mode
                lru_mode = config_dict.get("use_lru_eviction", False)

                if lru_mode:
                    try:
                        file_stat = file_entry.stat(follow_symlinks=False)
                        stat_info = (file_stat.st_atime, file_stat.st_mtime)
                    except OSError:
                        stat_info = None
                    
                    buffer.append((file_entry.path, stat_info))
                    if len(buffer) >= 100:
                        result_queue.put(("discovered_files_batch", buffer))
                        files_queued += len(buffer)
                        buffer = []
                else:
                    # Protect file if accessed or modified within the threshold
                    if ttl_seconds > 0:
                        try:
                            file_stat = file_entry.stat(follow_symlinks=False)
                            file_age = current_time - max(file_stat.st_atime, file_stat.st_mtime)
                            if file_age < ttl_seconds:
                                files_skipped += 1
                                continue
                        except OSError:
                            files_skipped_stat_error += 1
                            continue

                    if deletion_event.is_set():
                        # Deletion is ON: fill up to MAXQ
                        target_size = max_queue_size
                        queue_size = get_queue_size()

                        if queue_size >= target_size:
                            # Queue is full - slow down
                            file_queue.put(file_entry.path, timeout=5.0)
                            files_queued += 1
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
                        file_queue.put(file_entry.path, timeout=1.0)
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

            # Flush remaining buffer at the end of discovery sweep
            if lru_mode and buffer:
                result_queue.put(("discovered_files_batch", buffer))
                files_queued += len(buffer)
                buffer = []

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
                    "folders_deleted": folders_deleted,
                    "queue_size": queue_size,
                    "deletion_active": deletion_event.is_set(),
                },
                last_stats_send_time,
            )

    except Exception as e:
        logger.exception(f"Crawler P{process_num} error: {e}")
    finally:
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        logger.info(f"=== CRAWLER P{process_num} CPROFILE STATS ===\n{s.getvalue()}")
        try:
            profiler.dump_stats(f"/kv-cache/cprofile_crawler_p{process_num}.prof")
        except Exception as e:
            logger.error(f"Failed to dump normal crawler cProfile stats: {e}")

        logger.info(
            f"Crawler P{process_num} stopping - discovered {files_discovered}, queued {files_queued}, "
            f"skipped {files_skipped} (access_time), skipped_stat_error {files_skipped_stat_error}, deleted {folders_deleted} empty folders"
        )

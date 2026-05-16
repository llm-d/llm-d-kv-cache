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

# Mock vllm module to avoid heavyweight dependency in evictor container
import sys
from types import ModuleType
vllm = ModuleType("vllm")
vllm.logger = ModuleType("vllm.logger")
vllm.logger.init_logger = lambda name: logging.getLogger(name)
vllm.v1 = ModuleType("vllm.v1")
vllm.v1.kv_offload = ModuleType("vllm.v1.kv_offload")
vllm.v1.kv_offload.abstract = ModuleType("vllm.v1.kv_offload.abstract")
vllm.v1.kv_offload.abstract.OffloadKey = object
vllm.v1.kv_offload.abstract.get_offload_block_hash = lambda x: b""
vllm.v1.kv_offload.abstract.get_offload_group_idx = lambda x: 0
sys.modules["vllm"] = vllm
sys.modules["vllm.logger"] = vllm.logger
sys.modules["vllm.v1"] = vllm.v1
sys.modules["vllm.v1.kv_offload"] = vllm.v1.kv_offload
sys.modules["vllm.v1.kv_offload.abstract"] = vllm.v1.kv_offload.abstract

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


def stream_cache_files_with_mapper(cache_path: Path, hex_modulo_range: tuple[int, int] | None = None) -> Iterator[os.DirEntry]:
    """
    Stream cache files using FileMapper structure for canonical traversal.

    This function streams through FileMapper configurations in the cache directory
    and uses FileMapper.base_path to traverse the canonical structure:

    {model}/block_size_{X}_blocks_per_file_{Y}/tp_{tp}_pp_size_{pp}_pcp_size_{pcp}/
    rank_{rank}/{dtype}/{hhh}/{hh}/*.bin

    Yields os.DirEntry objects for .bin files in FileMapper structure
    """
    if not cache_path.exists():
        logger.warning(f"FileMapper: cache_path does not exist: {cache_path}")
        return

    if not FILEMAPPER_AVAILABLE:
        # FileMapper not available - this should not happen if properly configured
        # Fall back to vLLM structure
        logger.warning("FileMapper: FILEMAPPER_AVAILABLE is False")
        return

    modulo_range_min, modulo_range_max = hex_modulo_range if hex_modulo_range else (0, HEX_MODULO_BASE - 1)

    # Iterate through models
    for model_dir in safe_scandir(str(cache_path)):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Iterate through block_size_*_blocks_per_file_* directories
        for block_config_dir in Path(model_dir.path).glob("block_size_*_blocks_per_file_*"):
            if not block_config_dir.is_dir():
                continue

            # Parse: gpu_block_size, gpu_blocks_per_file from dirname
            block_params = parse_filemapper_params(
                block_config_dir.name,
                "block_size_{gpu_block_size}_blocks_per_file_{gpu_blocks_per_file}",
            )
            if not block_params:
                continue  # Malformed directory name, skip

            gpu_block_size = block_params.get("gpu_block_size")
            gpu_blocks_per_file = block_params.get("gpu_blocks_per_file")

            # Iterate through tp_*_pp_size_*_pcp_size_* directories
            for parallel_config_dir in block_config_dir.glob("tp_*_pp_size_*_pcp_size_*"):
                if not parallel_config_dir.is_dir():
                    continue

                # Parse: tp_size, pp_size, pcp_size from dirname
                parallel_params = parse_filemapper_params(
                    parallel_config_dir.name,
                    "tp_{tp_size}_pp_size_{pp_size}_pcp_size_{pcp_size}",
                )
                if not parallel_params:
                    continue  # Malformed directory name, skip

                tp_size = parallel_params.get("tp_size")
                pp_size = parallel_params.get("pp_size")
                pcp_size = parallel_params.get("pcp_size")

                # Iterate through rank_* directories
                for rank_dir in parallel_config_dir.glob("rank_*"):
                    if not rank_dir.is_dir():
                        continue

                    # Parse: rank from dirname
                    rank_match = re.match(r"rank_(\d+)", rank_dir.name)
                    if not rank_match:
                        continue  # Malformed directory name, skip

                    rank = int(rank_match.group(1))

                    # Iterate through dtype directories
                    for dtype_dir in safe_scandir(str(rank_dir)):
                        if not dtype_dir.is_dir():
                            continue

                        dtype = dtype_dir.name

                        # Create FileMapper instance to get canonical base_path
                        try:
                            mapper = FileMapper(
                                root_dir=str(cache_path),
                                model_name=model_name,
                                gpu_block_size=gpu_block_size,
                                gpu_blocks_per_file=gpu_blocks_per_file,
                                tp_size=tp_size,
                                pp_size=pp_size,
                                pcp_size=pcp_size,
                                rank=rank,
                                dtype=dtype,
                            )

                            # FileMapper.base_path is a string, convert to Path
                            base_path = Path(mapper.base_path)
                            if not base_path.exists():
                                continue

                        except Exception as e:
                            # FileMapper initialization failed, skip this configuration
                            logger.warning(f"FileMapper: Failed to create FileMapper for {model_name}: {e}")
                            continue

                        # Iterate through hex folders (hhh) - first 3 hex digits
                        for hex3_dir in safe_scandir(str(base_path)):
                            if not hex3_dir.is_dir() or len(hex3_dir.name) != 3:
                                continue

                            # Apply hex modulo filtering for load balancing
                            hex_int = hex_to_int(hex3_dir.name)
                            if hex_int is not None and hex_modulo_range:
                                hex_mod = hex_int % HEX_MODULO_BASE
                                if not (modulo_range_min <= hex_mod <= modulo_range_max):
                                    continue

                            # Iterate through second hex level (hh) - next 2 hex digits
                            for hex2_dir in safe_scandir(hex3_dir.path):
                                if not hex2_dir.is_dir():
                                    continue

                                # Yield all .bin files
                                for bin_file_entry in safe_scandir(hex2_dir.path):
                                    if bin_file_entry.is_file() and bin_file_entry.name.endswith(".bin"):
                                        yield bin_file_entry


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
    last_stats_send_time = time.time()

    def get_queue_size() -> int:
        """Get approximate queue size (non-blocking)."""
        try:
            return file_queue.qsize()
        except Exception:
            return 0

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        while not shutdown_event.is_set():
            # Stream files from assigned hex range using FileMapper
            file_stream = stream_cache_files_with_mapper(cache_path, hex_modulo_range)

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

                # Determine target queue size based on deletion state
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
            f"skipped {files_skipped} (access_time), skipped_stat_error {files_skipped_stat_error}"
        )

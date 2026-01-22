"""Crawler process for discovering and queuing cache files."""

import os
import time
import logging
import multiprocessing
import re
from pathlib import Path
from typing import Optional, Iterator, List, Tuple

from utils.system import setup_logging

# FileMapper integration for canonical cache structure
try:
    from llmd_fs_backend.file_mapper import FileMapper
    FILEMAPPER_AVAILABLE = True
except ImportError:
    FILEMAPPER_AVAILABLE = False
    FileMapper = None

# Module-level logger for functions
logger = logging.getLogger(__name__)


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
    param_names = re.findall(r'\{(\w+)\}', pattern)
    
    for param in param_names:
        regex_pattern = regex_pattern.replace(f'{{{param}}}', f'(?P<{param}>\\d+)')
    
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


def stream_cache_files_with_mapper(
    cache_path: Path, hex_modulo_range: Optional[Tuple[int, int]] = None
) -> Iterator[Path]:
    """
    Stream cache files using FileMapper structure for canonical traversal.
    
    This function streams through FileMapper configurations in the cache directory
    and uses FileMapper.base_path to traverse the canonical structure:
    
    {model}/block_size_{X}_blocks_per_file_{Y}/tp_{tp}_pp_size_{pp}_pcp_size_{pcp}/
    rank_{rank}/{dtype}/{hhh}/{hh}/*.bin
    
    Yields path objects for .bin files in FileMapper structure
    """
    if not cache_path.exists():
        logger.warning(f"FileMapper: cache_path does not exist: {cache_path}")
        return
    
    if not FILEMAPPER_AVAILABLE:
        # FileMapper not available - this should not happen if properly configured
        # Fall back to vLLM structure
        logger.warning("FileMapper: FILEMAPPER_AVAILABLE is False")
        return
    
    modulo_range_min, modulo_range_max = hex_modulo_range if hex_modulo_range else (0, 15)
    
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
                "block_size_{gpu_block_size}_blocks_per_file_{gpu_blocks_per_file}"
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
                    "tp_{tp_size}_pp_size_{pp_size}_pcp_size_{pcp_size}"
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
                                dtype=dtype
                            )
                            
                            base_path = mapper.base_path
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
                                hex_mod = hex_int % 16
                                if not (modulo_range_min <= hex_mod <= modulo_range_max):
                                    continue
                            
                            # Iterate through second hex level (hh) - next 2 hex digits
                            for hex2_dir in safe_scandir(hex3_dir.path):
                                if not hex2_dir.is_dir():
                                    continue
                                
                                # Yield all .bin files
                                for bin_file_entry in safe_scandir(hex2_dir.path):
                                    if bin_file_entry.is_file() and bin_file_entry.name.endswith(".bin"):
                                        yield Path(bin_file_entry.path)


def crawler_process(
    process_id: int,
    hex_modulo_range: Tuple[int, int],
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
    setup_logging(config_dict.get("log_level", "INFO"), process_num, log_file)
    logger = logging.getLogger(f"crawler_{process_num}")

    modulo_range_min, modulo_range_max = hex_modulo_range
    min_queue_size = config_dict.get("file_queue_min_size", 1000)
    max_queue_size = config_dict.get("file_queue_maxsize", 10000)
    access_time_threshold_seconds = (
        config_dict.get("file_access_time_threshold_minutes", 15.0) * 60.0
    )
    aggregated_logging = config_dict.get("aggregated_logging", True)
    aggregated_logging_interval = config_dict.get("aggregated_logging_interval", 30.0)
    cache_structure_mode = config_dict.get("cache_structure_mode", "file_mapper")

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
    logger.info(
        f"Crawler P{process_num} hex_modulo_range: {hex_modulo_range[0]}-{hex_modulo_range[1]} (hex mod 16)"
    )
    
    # Log cache structure mode
    if cache_structure_mode == "file_mapper":
        if FILEMAPPER_AVAILABLE:
            logger.info(
                f"Crawler P{process_num} using FileMapper cache structure mode (canonical)"
            )
        else:
            logger.warning(
                f"Crawler P{process_num} FileMapper not available - falling back to vLLM mode"
            )
            cache_structure_mode = "vllm"
    else:
        logger.info(
            f"Crawler P{process_num} using vLLM cache structure mode (legacy)"
        )

    files_discovered = 0
    files_queued = 0
    files_skipped = 0
    files_skipped_stat_error = 0
    stat_error_samples = []  # Store first few stat errors for logging
    max_stat_error_samples = 3
    last_heartbeat_time = time.time()
    last_stats_send_time = time.time()
    heartbeat_interval = 30.0  # Log heartbeat every 30 seconds (when aggregated logging is OFF)

    def get_queue_size() -> int:
        """Get approximate queue size (non-blocking)."""
        try:
            return file_queue.qsize()
        except Exception:
            return 0

    try:
        while not shutdown_event.is_set():
            # Stream files from assigned hex range using appropriate mode
            if cache_structure_mode == "file_mapper" and FILEMAPPER_AVAILABLE:
                file_stream = stream_cache_files_with_mapper(cache_path, hex_modulo_range)
            else:
                # Fallback to legacy vLLM structure
                file_stream = stream_cache_files(cache_path, hex_modulo_range)
            
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

            # Send stats to result_queue for aggregated logging or log locally
            current_time = time.time()
            if aggregated_logging:
                # Send stats periodically to main process for aggregated logging
                if current_time - last_stats_send_time >= aggregated_logging_interval:
                    queue_size = get_queue_size()
                    try:
                        result_queue.put(
                            (
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
                            ),
                            timeout=0.1,
                        )
                    except Exception:
                        pass  # Queue full or timeout - skip stats update
                    last_stats_send_time = current_time
            else:
                # Traditional local logging (when aggregated logging is OFF)
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    queue_size = get_queue_size()
                    deletion_state = "ON" if deletion_event.is_set() else "OFF"
                    logger.debug(
                        f"Heartbeat: discovered={files_discovered}, queued={files_queued}, "
                        f"skipped={files_skipped} (access_time), skipped_stat_error={files_skipped_stat_error}, "
                        f"queue={queue_size}, deletion={deletion_state}"
                    )
                    # Log first few stat error samples if any
                    if stat_error_samples:
                        logger.debug(f"Sample stat errors: {stat_error_samples}")
                    last_heartbeat_time = current_time

    except Exception as e:
        logger.error(f"Crawler P{process_num} error: {e}", exc_info=True)
    finally:
        logger.info(
            f"Crawler P{process_num} stopping - discovered {files_discovered}, queued {files_queued}, "
            f"skipped {files_skipped} (access_time), skipped_stat_error {files_skipped_stat_error}"
        )


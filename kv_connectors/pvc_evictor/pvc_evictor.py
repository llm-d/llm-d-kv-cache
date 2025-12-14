#!/usr/bin/env python3
"""
PVC Evictor - Multi-Process Architecture

N+2 Process Architecture:
- P1-PN: Crawler processes that discover and tag files for deletion (N configurable: 1, 2, 4, 8, or 16)
- P(N+1): Logger process that monitors disk usage and controls deletion
- P(N+2): Deleter process that performs actual file deletions

Key Features:
- Bounded queue for file paths (prevents memory overflow)
- Event-based deletion control (Logger triggers Deleter ON/OFF)
- Streaming file discovery (no memory accumulation)
- Fast batch deletion using xargs rm -f
- Configurable crawler count for different PVC sizes
"""

import os
import sys
import time
import logging
import subprocess
import signal
import multiprocessing
from pathlib import Path
from typing import Optional, Iterator, List, Tuple
from dataclasses import dataclass
import re


# Configure logging
def setup_logging(log_level: str = "INFO", process_id: Optional[int] = None, log_file: Optional[str] = None):
    """Configure logging with specified level and optional process ID prefix.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        process_id: Process ID (1-10) for prefix in logs
        log_file: Optional file path to also write logs to (in addition to stdout)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create format with process ID prefix
    if process_id is not None:
        format_str = f'[P{process_id}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Force reconfiguration for child processes
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers
    root_logger.setLevel(numeric_level)
    
    formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Always log to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)
    
    # Optionally also log to file
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception:
            # If file logging fails, continue with stdout only
            pass


@dataclass
class Config:
    """Configuration loaded from environment variables."""
    pvc_mount_path: str
    cleanup_threshold: float
    target_threshold: float
    monitoring_interval: float
    cache_directory: str
    batch_size: int
    dry_run: bool
    log_level: str
    timing_file_path: str
    num_crawler_processes: int  # P1-PN (default: 8, valid: 1, 2, 4, 8, 16)
    logger_interval: float  # P9 monitoring interval (default: 0.5s)
    file_queue_maxsize: int  # Max items in file queue (default: 10000)
    file_queue_min_size: int  # Min queue size to maintain when deletion OFF (default: 1000)
    deletion_batch_size: int  # Files per deletion batch (default: 100)
    log_file_path: Optional[str] = None  # Optional file path to write logs to
    file_access_time_threshold_minutes: float = 15.0  # Skip files accessed within this time (default: 15.0)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls(
            pvc_mount_path=os.getenv('PVC_MOUNT_PATH', '/kv-cache'),
            cleanup_threshold=float(os.getenv('CLEANUP_THRESHOLD', '85.0')),
            target_threshold=float(os.getenv('TARGET_THRESHOLD', '70.0')),
            monitoring_interval=float(os.getenv('MONITORING_INTERVAL_SECONDS', '1.0')),
            cache_directory=os.getenv('CACHE_DIRECTORY', 'kv/model-cache/models'),
            batch_size=int(os.getenv('BATCH_SIZE', '5')),
            dry_run=os.getenv('DRY_RUN', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            timing_file_path=os.getenv('TIMING_FILE_PATH', '/tmp/timing_analysis.txt'),
            num_crawler_processes=int(os.getenv('NUM_CRAWLER_PROCESSES', '8')),
            logger_interval=float(os.getenv('LOGGER_INTERVAL_SECONDS', '0.5')),
            file_queue_maxsize=int(os.getenv('FILE_QUEUE_MAXSIZE', '10000')),
            file_queue_min_size=int(os.getenv('FILE_QUEUE_MIN_SIZE', '1000')),
            deletion_batch_size=int(os.getenv('DELETION_BATCH_SIZE', '100')),
            log_file_path=os.getenv('LOG_FILE_PATH', None),  # Optional log file path
            file_access_time_threshold_minutes=float(os.getenv('FILE_ACCESS_TIME_THRESHOLD_MINUTES', '60.0'))
        )


@dataclass
class DiskUsage:
    """Disk usage information."""
    total_bytes: int
    used_bytes: int
    available_bytes: int
    usage_percent: float


def hex_to_int(hex_str: str) -> Optional[int]:
    """Convert hex string to integer."""
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def extract_hex_folder_from_path(path: Path, cache_path: Path) -> Optional[str]:
    """
    Extract hex folder name from KV cache path.
    
    KV cache structure: /kv/{model}/{path}/tp_{N}/rank_{M}/auto/{hash1}/{hash2}/...
    Returns the first hex folder ({hash1}) after 'auto'.
    """
    try:
        path = Path(path)
        cache_path = Path(cache_path)
        
        try:
            relative = path.relative_to(cache_path)
        except ValueError:
            return None
        
        parts = relative.parts
        
        for i, part in enumerate(parts):
            if part == 'auto' and i + 1 < len(parts):
                hash1 = parts[i + 1]
                if len(hash1) == 8 and re.match(r'^[0-9a-fA-F]{8}$', hash1):
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
        raise ValueError(f"NUM_CRAWLER_PROCESSES must be one of {valid_counts}, got {num_processes}")
    
    ranges = []
    values_per_process = 16 // num_processes
    
    for i in range(num_processes):
        min_mod = i * values_per_process
        max_mod = min_mod + values_per_process - 1
        ranges.append((min_mod, max_mod))
    
    return ranges


def get_disk_usage_from_statvfs(mount_path: str) -> Optional[DiskUsage]:
    """Get disk usage using statvfs() - fastest (~1ms)."""
    try:
        stat = os.statvfs(mount_path)
        block_size = stat.f_frsize
        total_blocks = stat.f_blocks
        free_blocks = stat.f_bfree
        
        total_bytes = total_blocks * block_size
        free_bytes = free_blocks * block_size
        used_bytes = total_bytes - free_bytes
        usage_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
        
        return DiskUsage(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            available_bytes=free_bytes,
            usage_percent=usage_percent
        )
    except Exception:
        return None


# ============================================================================
# P1-P8: Crawler Processes
# ============================================================================

def crawler_process(process_id: int, hex_modulo_range: Tuple[int, int],
                    cache_path: Path, config_dict: dict,
                    deletion_event: multiprocessing.Event,
                    file_queue: multiprocessing.Queue,
                    shutdown_event: multiprocessing.Event):
    """
    Crawler process (P1-P8): Discovers files and queues them for deletion.
    
    Queueing strategy:
    - When deletion is OFF: Queue files until queue size >= MINQ (pre-fill for fast start)
    - When deletion is ON: Queue files until queue size >= MAXQ (maximize throughput)
    
    Uses streaming discovery to avoid memory accumulation.
    """
    process_num = process_id + 1  # P1-P8
    log_file = config_dict.get('log_file_path')
    setup_logging(config_dict.get('log_level', 'INFO'), process_num, log_file)
    logger = logging.getLogger(f"crawler_{process_num}")
    
    min_mod, max_mod = hex_modulo_range
    min_queue_size = config_dict.get('file_queue_min_size', 1000)
    max_queue_size = config_dict.get('file_queue_maxsize', 10000)
    access_time_threshold_seconds = config_dict.get('file_access_time_threshold_minutes', 15.0) * 60.0
    
    # Convert decimal range to hex characters for clarity (0-9, then a-f)
    if min_mod == max_mod:
        hex_chars = f"'{format(min_mod, 'x')}'"
    else:
        hex_chars = f"'{format(min_mod, 'x')}'-'{format(max_mod, 'x')}'"
    logger.info(f"Crawler P{process_num} started - hex %16 in [{min_mod}, {max_mod}] (hex: {hex_chars})")
    logger.info(f"Crawler P{process_num} queue limits: MINQ={min_queue_size} (when OFF), MAXQ={max_queue_size} (when ON)")
    
    files_discovered = 0
    files_queued = 0
    files_skipped = 0  # Track files skipped due to recent access
    prev_file_path = None
    prev_file_time = None
    last_heartbeat_time = time.time()
    heartbeat_interval = 30.0  # Log heartbeat every 30 seconds
    
    def get_queue_size() -> int:
        """Get approximate queue size (non-blocking)."""
        try:
            return file_queue.qsize()
        except Exception:
            return 0
    
    def log_timing(event_type: str, duration_ms: float, **kwargs):
        """Log timing event (matching v3 format)."""
        try:
            unix_timestamp = time.time()
            extra_fields = ','.join(f"{k}={v}" for k, v in kwargs.items())
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
                
                # Track file-to-file transition timing (matching v3)
                if prev_file_path is not None and prev_file_time is not None:
                    duration_ms = (current_time - prev_file_time) * 1000
                    log_timing(
                        "FILE_TO_FILE",
                        duration_ms,
                        process_id=f"P{process_num}",
                        from_file=str(prev_file_path),
                        to_file=str(file_path)
                    )
                
                prev_file_path = file_path
                prev_file_time = current_time
                
                # Check file access time - skip recently accessed files
                try:
                    file_stat = file_path.stat()
                    file_atime = file_stat.st_atime  # Last access time
                    time_since_access = current_time - file_atime
                    
                    if time_since_access < access_time_threshold_seconds:
                        # File was accessed recently - skip it
                        time_ago_minutes = time_since_access / 60.0
                        files_skipped += 1
                        
                        # Log skip periodically (every 10k skips to avoid spam)
                        if files_skipped % 10000 == 0:
                            logger.info(f"Crawler P{process_num} skipped {files_skipped} files (last: accessed {time_ago_minutes:.2f} min ago)")
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
                        # Queue is full - backpressure: slow down
                        time.sleep(0.1)
                        continue
                else:
                    # Deletion is OFF: pre-fill up to MINQ (for fast start when triggered)
                    target_size = min_queue_size
                    queue_size = get_queue_size()
                    
                    if queue_size >= target_size:
                        # Queue is pre-filled - just discover, don't queue
                        if files_discovered % 10000 == 0:
                            logger.debug(f"Crawler P{process_num} pre-fill complete: queue={queue_size}/{target_size}, discovered={files_discovered}")
                        continue
                
                # Queue the file
                try:
                    file_start_time = time.time()
                    file_queue.put(str(file_path), timeout=1.0)
                    files_queued += 1
                    queue_time_ms = (time.time() - file_start_time) * 1000
                    
                    log_timing(
                        "FILE_QUEUED_ON" if deletion_event.is_set() else "FILE_QUEUED_OFF",
                        queue_time_ms,
                        process_id=f"P{process_num}",
                        file_path=str(file_path),
                        queue_size=get_queue_size()
                    )
                    
                    # Log progress periodically
                    if files_queued % 1000 == 0:
                        queue_size = get_queue_size()
                        deletion_state = "ON" if deletion_event.is_set() else "OFF"
                        logger.info(
                            f"Queued {files_queued} files "
                            f"(discovered {files_discovered}, queue={queue_size}/{target_size}, deletion={deletion_state})"
                        )
                    
                    # Log every 10000 files discovered (even if not queued)
                    if files_discovered % 10000 == 0 and files_discovered > 0:
                        queue_size = get_queue_size()
                        deletion_state = "ON" if deletion_event.is_set() else "OFF"
                        logger.info(
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
                logger.info(
                    f"Heartbeat: discovered={files_discovered}, queued={files_queued}, "
                    f"skipped={files_skipped}, queue={queue_size}, deletion={deletion_state}"
                )
                last_heartbeat_time = current_time
            
    except Exception as e:
        logger.error(f"Crawler P{process_num} error: {e}", exc_info=True)
    finally:
        logger.info(f"Crawler P{process_num} stopping - discovered {files_discovered}, queued {files_queued}, skipped {files_skipped}")


def stream_cache_files(cache_path: Path, hex_modulo_range: Optional[Tuple[int, int]] = None) -> Iterator[Path]:
    """
    Stream cache files using os.scandir() with hex folder filtering.
    
    Yields only files from folders where hex_folder_name % 16 is in the specified range.
    """
    if not cache_path.exists():
        return
    
    min_mod, max_mod = hex_modulo_range if hex_modulo_range else (0, 15)
    
    try:
        # Walk through cache directory structure
        # Structure: /kv/{model}/{path}/tp_{N}/rank_{M}/auto/{hash1}/{hash2}/...
        for model_dir in os.scandir(cache_path):
            if not model_dir.is_dir():
                continue
            
            try:
                for path_dir in os.scandir(model_dir.path):
                    if not path_dir.is_dir():
                        continue
                    
                    try:
                        for tp_dir in os.scandir(path_dir.path):
                            if not tp_dir.is_dir():
                                continue
                            
                            try:
                                for rank_dir in os.scandir(tp_dir.path):
                                    if not rank_dir.is_dir():
                                        continue
                                    
                                    try:
                                        auto_path = Path(rank_dir.path) / 'auto'
                                        if not auto_path.exists():
                                            continue
                                        
                                        # Scan hex folders
                                        for hash1_dir in os.scandir(str(auto_path)):
                                            if not hash1_dir.is_dir():
                                                continue
                                            
                                            # Check if this hex folder matches our modulo range
                                            hex_folder = hash1_dir.name
                                            hex_int = hex_to_int(hex_folder)
                                            if hex_int is None:
                                                continue
                                            
                                            hex_mod = hex_int % 16
                                            if not (min_mod <= hex_mod <= max_mod):
                                                continue
                                            
                                            # Scan files in this hex folder
                                            try:
                                                for hash2_dir in os.scandir(hash1_dir.path):
                                                    if hash2_dir.is_dir():
                                                        for file_entry in os.scandir(hash2_dir.path):
                                                            if file_entry.is_file() and file_entry.name.endswith('.bin'):
                                                                yield Path(file_entry.path)
                                            except (OSError, PermissionError):
                                                continue
                                            
                                    except (OSError, PermissionError):
                                        continue
                            except (OSError, PermissionError):
                                continue
                    except (OSError, PermissionError):
                        continue
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError):
        return


# ============================================================================
# P9: Logger Process
# ============================================================================

def logger_process(mount_path: str, cleanup_threshold: float, target_threshold: float,
                   logger_interval: float, deletion_event: multiprocessing.Event,
                   shutdown_event: multiprocessing.Event):
    """
    Logger process (P9): Monitors disk usage and controls deletion trigger.
    
    - Monitors statvfs() every logger_interval seconds
    - Sets deletion_event when usage > cleanup_threshold
    - Clears deletion_event when usage < target_threshold
    """
    log_file = os.getenv('LOG_FILE_PATH', None)
    setup_logging('INFO', 9, log_file)
    logger = logging.getLogger("logger")
    
    logger.info(f"Logger P9 started - monitoring every {logger_interval}s")
    logger.info(f"Logger P9 thresholds: cleanup={cleanup_threshold}%, target={target_threshold}%")
    
    deletion_active = False
    
    try:
        while not shutdown_event.is_set():
            usage = get_disk_usage_from_statvfs(mount_path)
            
            if usage:
                current_time = time.time()
                # Match v3 logging format exactly
                logger.info(f"CACHE_PERCENT_LOG:{current_time:.3f},{usage.usage_percent:.2f},statvfs")
                logger.info(
                    f"PVC Usage: {usage.usage_percent:.2f}% "
                    f"({usage.used_bytes / (1024**3):.2f}GB / {usage.total_bytes / (1024**3):.2f}GB) "
                    f"[statvfs]"
                )
                
                # Log queue status periodically
                if int(current_time) % 10 == 0:  # Every 10 seconds
                    try:
                        # Try to get queue size (might fail if queue is in different process)
                        logger.debug(f"Queue status check (deletion={'ON' if deletion_event.is_set() else 'OFF'})")
                    except:
                        pass
                
                # Control deletion based on thresholds
                if usage.usage_percent >= cleanup_threshold and not deletion_active:
                    logger.warning(f"Logger P9: Usage {usage.usage_percent:.2f}% >= {cleanup_threshold}% - Triggering deletion ON")
                    logger.info(f"DELETION_START:{current_time:.3f},{usage.usage_percent:.2f}")
                    deletion_event.set()
                    deletion_active = True
                elif usage.usage_percent <= target_threshold and deletion_active:
                    logger.info(f"Logger P9: Usage {usage.usage_percent:.2f}% <= {target_threshold}% - Triggering deletion OFF")
                    logger.info(f"DELETION_END:{current_time:.3f},{usage.usage_percent:.2f}")
                    deletion_event.clear()
                    deletion_active = False
            
            time.sleep(logger_interval)
            
    except Exception as e:
        logger.error(f"Logger P9 error: {e}", exc_info=True)
    finally:
        logger.info("Logger P9 stopping")
        deletion_event.clear()


# ============================================================================
# P10: Deleter Process
# ============================================================================

def deleter_process(cache_path: Path, config_dict: dict,
                   deletion_event: multiprocessing.Event,
                   file_queue: multiprocessing.Queue,
                   result_queue: multiprocessing.Queue,
                   shutdown_event: multiprocessing.Event):
    """
    Deleter process (P10): Deletes files from queue in batches.
    
    - Only deletes when deletion_event is set
    - Reads file paths from file_queue
    - Batches files for efficient deletion using xargs rm -f
    """
    log_file = config_dict.get('log_file_path')
    setup_logging(config_dict.get('log_level', 'INFO'), 10, log_file)
    logger = logging.getLogger("deleter")
    
    batch_size = config_dict.get('deletion_batch_size', 100)
    dry_run = config_dict.get('dry_run', False)
    
    logger.info(f"Deleter P10 started - batch size: {batch_size}, dry_run: {dry_run}")
    
    total_files_deleted = 0
    total_bytes_freed = 0
    current_batch = []
    prev_batch_time = None
    
    def log_timing(event_type: str, duration_ms: float, **kwargs):
        """Log timing event (matching v3 format)."""
        try:
            unix_timestamp = time.time()
            extra_fields = ','.join(f"{k}={v}" for k, v in kwargs.items())
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
                    # Try to get file from queue (non-blocking with timeout)
                    try:
                        file_path_str = file_queue.get(timeout=1.0)
                        current_batch.append(file_path_str)
                        
                        # Delete batch when full
                        if len(current_batch) >= batch_size:
                            batch_start_time = time.time()
                            deleted, freed = delete_batch(current_batch, dry_run, logger)
                            batch_duration_ms = (time.time() - batch_start_time) * 1000
                            
                            total_files_deleted += deleted
                            total_bytes_freed += freed
                            
                            # Track batch-to-batch timing (matching v3)
                            if prev_batch_time is not None:
                                batch_gap_ms = (batch_start_time - prev_batch_time) * 1000
                                log_timing(
                                    "BATCH_TO_BATCH",
                                    batch_gap_ms,
                                    process_id="P10",
                                    batch_size=len(current_batch)
                                )
                            
                            prev_batch_time = batch_start_time
                            
                            log_timing(
                                "BATCH_DELETE",
                                batch_duration_ms,
                                process_id="P10",
                                files_deleted=deleted,
                                bytes_freed=freed
                            )
                            
                            logger.info(
                                f"Deleted batch: {deleted} files, {freed / (1024**3):.2f}GB freed "
                                f"(total: {total_files_deleted} files, {total_bytes_freed / (1024**3):.2f}GB)"
                            )
                            
                            current_batch = []
                            
                            # Report progress
                            result_queue.put(('progress', total_files_deleted, total_bytes_freed), timeout=1.0)
                    
                    except Exception:
                        # Queue empty or timeout - delete remaining batch if any
                        if current_batch:
                            batch_start_time = time.time()
                            deleted, freed = delete_batch(current_batch, dry_run, logger)
                            batch_duration_ms = (time.time() - batch_start_time) * 1000
                            
                            total_files_deleted += deleted
                            total_bytes_freed += freed
                            
                            log_timing(
                                "BATCH_DELETE",
                                batch_duration_ms,
                                process_id="P10",
                                files_deleted=deleted,
                                bytes_freed=freed
                            )
                            
                            logger.info(f"Deleted final batch: {deleted} files, {freed / (1024**3):.2f}GB freed")
                            current_batch = []
                            result_queue.put(('progress', total_files_deleted, total_bytes_freed), timeout=1.0)
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Deleter P10 error processing queue: {e}", exc_info=True)
                    time.sleep(1.0)
            else:
                # Deletion is OFF - clear any pending batch and wait
                if current_batch:
                    logger.debug(f"Deletion OFF - clearing {len(current_batch)} pending files")
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
        logger.info(f"Deleter P10 stopping - deleted {total_files_deleted} files, {total_bytes_freed / (1024**3):.2f}GB")
        result_queue.put(('done', total_files_deleted, total_bytes_freed), timeout=1.0)


def delete_batch(file_paths: List[str], dry_run: bool, logger: logging.Logger) -> Tuple[int, int]:
    """
    Delete a batch of files using xargs rm -f (fastest method).
    
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
        return 0, 0
    
    # Use xargs rm -f for batch deletion
    try:
        # Use null-terminated input for xargs -0
        input_data = '\0'.join(valid_paths).encode('utf-8')
        result = subprocess.run(
            ['xargs', '-0', 'rm', '-f'],
            input=input_data,
            capture_output=True,
            timeout=60,
            check=False
        )
        
        if result.returncode == 0:
            return len(valid_paths), total_bytes
        else:
            # Fallback to individual deletion
            logger.warning(f"xargs rm failed, falling back to individual deletion")
            return delete_individually(valid_paths)
    except subprocess.TimeoutExpired:
        logger.warning("xargs rm timed out, falling back to individual deletion")
        return delete_individually(valid_paths)
    except Exception as e:
        logger.error(f"Batch deletion error: {e}, falling back to individual deletion")
        return delete_individually(valid_paths)


def delete_individually(file_paths: List[str]) -> Tuple[int, int]:
    """Fallback: Delete files one by one."""
    files_deleted = 0
    bytes_freed = 0
    
    for path_str in file_paths:
        try:
            file_path = Path(path_str)
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_path.unlink()
                files_deleted += 1
                bytes_freed += file_size
        except Exception:
            continue
    
    return files_deleted, bytes_freed


# ============================================================================
# Main Process (Coordinator)
# ============================================================================

class PVCEvictor:
    """Main evictor controller coordinating N+2 processes (N crawlers + logger + deleter)."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Wait for PVC mount
        self._wait_for_mount()
        
        # Initialize shared objects for IPC
        self.deletion_event = multiprocessing.Event()  # P9 controls P10, P1-P8 check this
        self.file_queue = multiprocessing.Queue(maxsize=config.file_queue_maxsize)  # P1-P8 → P10
        self.result_queue = multiprocessing.Queue()  # P10 → Main
        self.shutdown_event = multiprocessing.Event()  # All processes check this
        
        self.logger.info("PVC Cleanup Service v4 (10-Process Architecture) initialized")
        self.logger.info(f"  Mount Path: {config.pvc_mount_path}")
        self.logger.info(f"  Cache Directory: {config.cache_directory}")
        self.logger.info(f"  Crawler Processes: {config.num_crawler_processes} (P1-P{config.num_crawler_processes})")
        logger_process_num = config.num_crawler_processes + 1
        deleter_process_num = config.num_crawler_processes + 2
        self.logger.info(f"  Logger Process: P{logger_process_num} (monitoring every {config.logger_interval}s)")
        self.logger.info(f"  Deleter Process: P{deleter_process_num} (batch size: {config.deletion_batch_size})")
        self.logger.info(f"  Cleanup Threshold: {config.cleanup_threshold}%")
        self.logger.info(f"  Target Threshold: {config.target_threshold}%")
        self.logger.info(f"  File Queue: MINQ={config.file_queue_min_size} (pre-fill when OFF), MAXQ={config.file_queue_maxsize} (max when ON)")
    
    def _wait_for_mount(self):
        """Wait for PVC mount to be ready."""
        max_wait = 60
        wait_interval = 2
        waited = 0
        
        while waited < max_wait:
            try:
                if os.path.exists(self.config.pvc_mount_path):
                    self.logger.info(f"PVC mount path is ready: {self.config.pvc_mount_path}")
                    return
            except Exception:
                pass
            
            time.sleep(wait_interval)
            waited += wait_interval
            self.logger.info(f"Still waiting for mount... ({waited}s/{max_wait}s)")
        
        self.logger.error(f"PVC mount path not available after {max_wait}s")
        sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.shutdown_event.set()
        self.deletion_event.clear()  # Stop deletion
    
    def run(self):
        """Main coordination loop - spawns and manages all processes."""
        total_processes = self.config.num_crawler_processes + 2
        self.logger.info(f"Starting {total_processes}-process evictor service...")
        
        cache_path = Path(self.config.pvc_mount_path) / self.config.cache_directory
        
        # Convert Config to dict for pickling
        config_dict = {
            'pvc_mount_path': self.config.pvc_mount_path,
            'cleanup_threshold': self.config.cleanup_threshold,
            'target_threshold': self.config.target_threshold,
            'monitoring_interval': self.config.monitoring_interval,
            'cache_directory': self.config.cache_directory,
            'batch_size': self.config.batch_size,
            'dry_run': self.config.dry_run,
            'log_level': self.config.log_level,
            'timing_file_path': self.config.timing_file_path,
            'deletion_batch_size': self.config.deletion_batch_size,
            'file_queue_min_size': self.config.file_queue_min_size,
            'file_queue_maxsize': self.config.file_queue_maxsize,
            'log_file_path': self.config.log_file_path,
            'file_access_time_threshold_minutes': self.config.file_access_time_threshold_minutes
        }
        
        # Get hex modulo ranges for crawlers
        hex_ranges = get_hex_modulo_ranges(self.config.num_crawler_processes)
        
        # Spawn P1-PN: Crawler processes (N = num_crawler_processes)
        crawler_processes = []
        for i in range(self.config.num_crawler_processes):
            hex_range = hex_ranges[i]
            process = multiprocessing.Process(
                target=crawler_process,
                args=(i, hex_range, cache_path, config_dict,
                      self.deletion_event, self.file_queue, self.shutdown_event),
                name=f"Crawler-P{i+1}"
            )
            process.start()
            crawler_processes.append(process)
            # Convert decimal range to hex characters for clarity (0-9, then a-f)
            min_mod, max_mod = hex_range[0], hex_range[1]
            if min_mod == max_mod:
                hex_chars = f"'{format(min_mod, 'x')}'"
            else:
                hex_chars = f"'{format(min_mod, 'x')}'-'{format(max_mod, 'x')}'"
            self.logger.info(f"Started crawler P{i+1} (hex %16 in [{min_mod}, {max_mod}], hex: {hex_chars})")
        
        # Spawn P(N+1): Logger process
        logger_process_num = self.config.num_crawler_processes + 1
        logger_process_obj = multiprocessing.Process(
            target=logger_process,
            args=(self.config.pvc_mount_path, self.config.cleanup_threshold,
                  self.config.target_threshold, self.config.logger_interval,
                  self.deletion_event, self.shutdown_event),
            name=f"Logger-P{logger_process_num}"
        )
        logger_process_obj.start()
        self.logger.info(f"Started logger P{logger_process_num}")
        
        # Spawn P(N+2): Deleter process
        deleter_process_num = self.config.num_crawler_processes + 2
        deleter_process_obj = multiprocessing.Process(
            target=deleter_process,
            args=(cache_path, config_dict, self.deletion_event,
                  self.file_queue, self.result_queue, self.shutdown_event),
            name=f"Deleter-P{deleter_process_num}"
        )
        deleter_process_obj.start()
        self.logger.info(f"Started deleter P{deleter_process_num}")
        
        # Monitor processes and handle results
        try:
            while self.running:
                try:
                    # Check for deletion results (non-blocking)
                    result = self.result_queue.get(timeout=5.0)
                    result_type, *data = result
                    
                    if result_type == 'progress':
                        files_deleted, bytes_freed = data
                        self.logger.info(
                            f"Deletion progress: {files_deleted} files, "
                            f"{bytes_freed / (1024**3):.2f}GB freed"
                        )
                    elif result_type == 'done':
                        files_deleted, bytes_freed = data
                        self.logger.info(
                            f"Deletion complete: {files_deleted} files, "
                            f"{bytes_freed / (1024**3):.2f}GB freed"
                        )
                
                except Exception:
                    # Timeout or queue empty - continue monitoring
                    # Check if processes are still alive
                    logger_process_num = self.config.num_crawler_processes + 1
                    if not logger_process_obj.is_alive():
                        self.logger.error(f"Logger P{logger_process_num} died, restarting...")
                        logger_process_obj = multiprocessing.Process(
                            target=logger_process,
                            args=(self.config.pvc_mount_path, self.config.cleanup_threshold,
                                  self.config.target_threshold, self.config.logger_interval,
                                  self.deletion_event, self.shutdown_event),
                            name=f"Logger-P{logger_process_num}"
                        )
                        logger_process_obj.start()
                    
                    deleter_process_num = self.config.num_crawler_processes + 2
                    if not deleter_process_obj.is_alive():
                        self.logger.error(f"Deleter P{deleter_process_num} died, restarting...")
                        deleter_process_obj = multiprocessing.Process(
                            target=deleter_process,
                            args=(cache_path, config_dict, self.deletion_event,
                                  self.file_queue, self.result_queue, self.shutdown_event),
                            name=f"Deleter-P{deleter_process_num}"
                        )
                        deleter_process_obj.start()
                    
                    time.sleep(1.0)
        
        except KeyboardInterrupt:
            self.logger.warning("Shutdown requested, stopping all processes...")
        
        finally:
            # Graceful shutdown
            self.logger.info("Shutting down all processes...")
            self.shutdown_event.set()
            self.deletion_event.clear()
            
            # Wait for processes to finish
            for process in crawler_processes:
                process.join(timeout=10)
                if process.is_alive():
                    self.logger.warning(f"Process {process.name} did not terminate, forcing...")
                    process.terminate()
                    process.join(timeout=5)
            
            logger_process_obj.join(timeout=10)
            if logger_process_obj.is_alive():
                logger_process_obj.terminate()
                logger_process_obj.join(timeout=5)
            
            deleter_process_num = self.config.num_crawler_processes + 2
            deleter_process_obj.join(timeout=30)  # Give deleter more time to finish batch
            if deleter_process_obj.is_alive():
                self.logger.warning(f"Deleter P{deleter_process_num} did not terminate, forcing...")
                deleter_process_obj.terminate()
                deleter_process_obj.join(timeout=5)
            
            self.logger.info("All processes stopped")


def main():
    """Main entry point."""
    print("PVC Evictor starting...", flush=True)
    
    try:
        config = Config.from_env()
        setup_logging(config.log_level, None, config.log_file_path)
        
        # Validate crawler count before creating evictor
        try:
            get_hex_modulo_ranges(config.num_crawler_processes)
        except ValueError as e:
            print(f"ERROR: {e}", flush=True)
            sys.exit(1)
        
        print(f"Configuration loaded: PVC={config.pvc_mount_path}, Crawlers={config.num_crawler_processes}, Total Processes={config.num_crawler_processes + 2}", flush=True)
        
        evictor = PVCEvictor(config)
        print("PVC Evictor initialized, starting coordination loop...", flush=True)
        evictor.run()
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


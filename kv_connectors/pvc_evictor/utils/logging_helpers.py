"""Logging helper functions for PVC Evictor."""

import logging
import time
import multiprocessing
from typing import Dict, Any, Optional


# Constants for aggregated logging
AGGREGATED_LOGGING_INTERVAL_SECONDS = 30.0  # Log aggregated stats every N seconds


def send_stats_to_queue(
    result_queue: multiprocessing.Queue,
    stats_type: str,
    process_num: int,
    stats: Dict[str, Any],
    last_send_time: float,
    interval: float = AGGREGATED_LOGGING_INTERVAL_SECONDS
) -> float:
    """
    Send process statistics to result queue for aggregated logging.
    
    Args:
        result_queue: Queue to send stats to
        stats_type: Type of stats ("crawler_stats", "activator_stats", "deleter_stats")
        process_num: Process number for identification
        stats: Dictionary of statistics to send
        last_send_time: Timestamp of last send
        interval: Minimum interval between sends (seconds)
    
    Returns:
        Updated last_send_time (current time if sent, unchanged if not)
    """
    current_time = time.time()
    if current_time - last_send_time >= interval:
        try:
            result_queue.put(
                (stats_type, process_num, stats),
                timeout=0.1
            )
            return current_time
        except Exception:
            # Queue full or timeout - skip stats update
            pass
    return last_send_time


def log_aggregated_stats(
    logger: logging.Logger,
    crawler_stats: Dict[int, Dict[str, Any]],
    activator_stats: Dict[int, Dict[str, Any]],
    deleter_stats: Dict[int, Dict[str, Any]],
    cleanup_threshold: float,
    target_threshold: float,
) -> None:
    """
    Log aggregated statistics from all processes in a unified format.
    
    Args:
        logger: Logger instance to use for output
        crawler_stats: Dictionary mapping process_num to crawler statistics
        activator_stats: Dictionary mapping process_num to activator statistics
        deleter_stats: Dictionary mapping process_num to deleter statistics
        cleanup_threshold: Cleanup threshold percentage for display
        target_threshold: Target threshold percentage for display
    """
    if not crawler_stats and not activator_stats and not deleter_stats:
        return
    
    # Build aggregated log message
    log_lines = ["=== System Status ==="]
    
    # Crawler stats
    if crawler_stats:
        total_files_discovered = sum(stats.get("files_discovered", 0) for stats in crawler_stats.values())
        total_files_queued = sum(stats.get("files_queued", 0) for stats in crawler_stats.values())
        total_files_skipped = sum(stats.get("files_skipped", 0) for stats in crawler_stats.values())
        log_lines.append(f"Crawlers: {len(crawler_stats)} active")
        log_lines.append(f"  Total files discovered: {total_files_discovered}")
        log_lines.append(f"  Total files queued: {total_files_queued}")
        log_lines.append(f"  Total files skipped (hot): {total_files_skipped}")
        for process_num in sorted(crawler_stats.keys()):
            stats = crawler_stats[process_num]
            log_lines.append(
                f"  P{process_num}: discovered={stats.get('files_discovered', 0)}, "
                f"queued={stats.get('files_queued', 0)}, "
                f"skipped={stats.get('files_skipped', 0)}"
            )
    
    # Activator stats
    if activator_stats:
        for process_num in sorted(activator_stats.keys()):
            stats = activator_stats[process_num]
            deletion_status = "ON" if stats.get("deletion_active", False) else "OFF"
            used_gb = stats.get('used_bytes', 0) / (1024**3)
            total_gb = stats.get('total_bytes', 0) / (1024**3)
            log_lines.append(f"Activator P{process_num}:")
            log_lines.append(
                f"  PVC Usage: {stats.get('usage_percent', 0):.1f}% "
                f"({used_gb:.2f}GB / {total_gb:.2f}GB)"
            )
            log_lines.append(f"  Deletion: {deletion_status}")
            log_lines.append(
                f"  Thresholds: cleanup={cleanup_threshold}%, "
                f"target={target_threshold}%"
            )
    
    # Deleter stats
    if deleter_stats:
        for process_num in sorted(deleter_stats.keys()):
            stats = deleter_stats[process_num]
            files_deleted = stats.get('files_deleted', 0)
            bytes_freed = stats.get('bytes_freed', 0)
            gb_freed = bytes_freed / (1024**3)
            log_lines.append(f"Deleter P{process_num}:")
            log_lines.append(f"  Files deleted: {files_deleted}")
            log_lines.append(f"  Space freed: {gb_freed:.2f}GB")
    
    log_lines.append("=" * 21)
    
    # Log as single multi-line message
    logger.info("\n".join(log_lines))

# Made with Bob

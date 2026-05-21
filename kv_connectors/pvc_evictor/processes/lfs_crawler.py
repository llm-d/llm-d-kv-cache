"""LFS Crawler process for discovering and queuing cache files using lfs find."""

import logging
import multiprocessing
import os
import select
import subprocess
import time
from pathlib import Path

from utils.logging_helpers import send_stats_to_queue
from utils.system import setup_logging

MINUTES_TO_SECONDS = 60.0
DISCOVERY_LOG_INTERVAL = 10000
QUEUE_FULL_SLEEP_SECONDS = 0.1


def lfs_crawler_process(
    process_id: int,
    cache_path: Path,
    config_dict: dict,
    deletion_event: multiprocessing.Event,
    file_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
):
    """
    LFS Crawler process: Discovers files using lfs find and queues them for deletion.
    """
    import cProfile
    import io
    import pstats

    process_num = process_id + 1
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict["log_level"], process_num, log_file)
    logger = logging.getLogger(f"lfs_crawler_{process_num}")

    min_queue_size = config_dict["file_queue_min_size"]
    max_queue_size = config_dict["file_queue_maxsize"]
    ttl_minutes = config_dict["file_access_time_threshold_minutes"]
    ttl_seconds = ttl_minutes * MINUTES_TO_SECONDS

    logger.info(f"LFS Crawler P{process_num} started - scanning {cache_path}")
    logger.info(f"LFS Crawler P{process_num} queue limits: MINQ={min_queue_size} (OFF), MAXQ={max_queue_size} (ON)")
    logger.info(f"LFS Crawler P{process_num} Access Time Threshold: {ttl_minutes} minutes ({ttl_seconds}s)")

    files_discovered = 0
    files_queued = 0
    files_skipped = 0
    files_skipped_stat_error = 0
    last_stats_send_time = time.time()
    buffer = []

    def get_queue_size() -> int:
        try:
            return file_queue.qsize()
        except Exception:
            return 0

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        lru_mode = config_dict.get("use_lru_eviction", False)
        while not shutdown_event.is_set():
            cmd = ["lfs", "find", str(cache_path), "-type", "f"]
            if not lru_mode and ttl_minutes > 0:
                cmd.extend(["-atime", f"+{int(ttl_minutes)}m"])

            logger.debug(f"Starting lfs find scan: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            try:
                while not shutdown_event.is_set() and process.poll() is None:
                    if not lru_mode:
                        # Determine target queue size based on deletion state
                        if deletion_event.is_set():
                            target_size = max_queue_size
                        else:
                            target_size = min_queue_size

                        queue_size = get_queue_size()
                        if queue_size >= target_size:
                            time.sleep(QUEUE_FULL_SLEEP_SECONDS)
                            continue

                    ready, _, _ = select.select([process.stdout], [], [], 0.5)
                    if ready:
                        line_bytes = process.stdout.readline()
                        if not line_bytes:
                            break

                        file_path_str = line_bytes.decode("utf-8", errors="ignore").strip()
                        if not file_path_str:
                            continue

                        files_discovered += 1
                        current_time = time.time()

                        # Send stats periodically
                        last_stats_send_time = send_stats_to_queue(
                            result_queue,
                            "crawler_stats",
                            process_num,
                            {
                                "files_discovered": files_discovered,
                                "files_queued": files_queued,
                                "files_skipped": files_skipped,
                                "files_skipped_stat_error": files_skipped_stat_error,
                                "queue_size": get_queue_size() if not lru_mode else 0,
                                "deletion_active": deletion_event.is_set(),
                            },
                            last_stats_send_time,
                        )

                        if lru_mode:
                            try:
                                stat = os.stat(file_path_str)
                                stat_info = (stat.st_atime, stat.st_mtime)
                            except OSError:
                                stat_info = None
                            
                            buffer.append((file_path_str, stat_info))
                            if len(buffer) >= 100:
                                result_queue.put(("discovered_files_batch", buffer))
                                files_queued += len(buffer)
                                buffer = []
                        else:
                            try:
                                file_queue.put(file_path_str, timeout=1.0)
                                files_queued += 1

                                if files_queued % 1000 == 0:
                                    q_sz = get_queue_size()
                                    del_state = "ON" if deletion_event.is_set() else "OFF"
                                    logger.debug(
                                        f"Queued {files_queued} files (discovered {files_discovered}, "
                                        f"queue={q_sz}/{target_size}, deletion={del_state})"
                                    )
                            except Exception:
                                time.sleep(QUEUE_FULL_SLEEP_SECONDS)

            finally:
                if process and process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=5.0)
                    except Exception:
                        pass

            # Flush remaining buffer at the end of LFS sweep
            if lru_mode and buffer:
                result_queue.put(("discovered_files_batch", buffer))
                files_queued += len(buffer)
                buffer = []

            # If scan finished, sleep briefly before restarting
            time.sleep(5.0)

            # Send final stats for the cycle
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

    except Exception as e:
        logger.exception(f"LFS Crawler P{process_num} error: {e}")
    finally:
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        logger.info(f"=== LFS CRAWLER P{process_num} CPROFILE STATS ===\n{s.getvalue()}")
        try:
            profiler.dump_stats(f"/kv-cache/cprofile_lfs_crawler_p{process_num}.prof")
        except Exception as e:
            logger.error(f"Failed to dump crawler cProfile stats: {e}")

        logger.info(
            f"LFS Crawler P{process_num} stopping - discovered {files_discovered}, queued {files_queued}, "
            f"skipped {files_skipped} (TTL), skipped_stat_error {files_skipped_stat_error}"
        )

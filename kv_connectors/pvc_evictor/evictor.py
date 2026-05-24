#!/usr/bin/env python3
"""
PVC Evictor - Multi-Process Architecture

N+1+M Process Architecture:
- P1-PN: Crawler processes that discover and tag files for deletion (N configurable: 1, 2, 4, 8, or 16)
- P(N+1): Activator process that monitors disk usage and controls deletion
- P(N+2) - P(N+1+M): Deleter processes that perform actual file deletions (M configurable)
"""

import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from config import Config
from processes.activator import activator_process
from processes.crawler import crawler_process, get_hex_modulo_ranges
from processes.deleter import deleter_process
from processes.folder_cleaner import folder_cleaner_process
from utils.logging_helpers import (
    AGGREGATED_LOGGING_INTERVAL_SECONDS,
    log_aggregated_stats,
)
from utils.system import setup_logging
from strategies import LRUPolicy, AgeTTLPolicy, DiscoveryCoordinator, ContinuousCleanup
import threading
import queue


class PVCEvictor:
    """Main evictor controller coordinating N+1+M processes (N crawlers + activator + M deleters)."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = True

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Wait for PVC mount
        self._wait_for_mount()

        # Inter-Process Communication:
        # - shutdown_event: multiprocessing.Event - shared boolean flag, all processes check this in their loops
        # - deletion_event: multiprocessing.Event - shared boolean flag, activator controls deleter via this
        # - deletion_queue: multiprocessing.Queue - FIFO queue, crawlers put files, deleter gets files
        # - result_queue: multiprocessing.Queue - FIFO queue, deleter reports progress to main
        # Events use shared memory, queues use pipes with pickling

        # Initialize shared objects for IPC
        self.deletion_event = multiprocessing.Event()  # Activator controls Deleter, Crawlers check this
        self.deletion_queue = multiprocessing.Queue(maxsize=config.file_queue_maxsize)  # Crawlers → Deleter
        self.folder_queue = multiprocessing.Queue(maxsize=50000) if config.enable_dir_cleanup else None  # Deleter → Folder Cleaner
        self.result_queue = multiprocessing.Queue()  # Deleter → Main
        self.shutdown_event = multiprocessing.Event()  # All processes check this

        # Convert Config to dict for pickling (needed for multiprocessing)
        self.config_dict = self.config.to_dict()

        self.logger.info(
            f"PVC Cleanup Service (Multi-Process Architecture: "
            f"{config.num_crawler_processes + 1 + config.num_deleter_processes} total processes) initialized"
        )
        self.logger.info(f"  Mount Path: {config.pvc_mount_path}")
        self.logger.info(f"  Cache Directory: {config.cache_directory}")
        self.logger.info(f"  Shard Index: {config.shard_index} / {config.total_shards} (Total Shards)")
        self.logger.info(f"  Crawler Processes: {config.num_crawler_processes} (P1-P{config.num_crawler_processes})")
        activator_process_num = config.num_crawler_processes + 1
        self.logger.info(f"  Activator Process: P{activator_process_num} (monitoring every {config.logger_interval}s)")
        self.logger.info(f"  Deleter Processes: {config.num_deleter_processes} (batch size: {config.deletion_batch_size})")
        self.logger.info(f"  Cleanup Threshold: {config.cleanup_threshold}%")
        self.logger.info(f"  Target Threshold: {config.target_threshold}%")
        self.logger.info(
            f"  File Queue: MINQ={config.file_queue_min_size} (pre-fill when OFF), "
            f"MAXQ={config.file_queue_maxsize} (max when ON)"
        )
        self.logger.info(f"  Folder Cleanup: {'ENABLED' if config.enable_dir_cleanup else 'DISABLED'}")

        # 1. Compose Pluggable Eviction Policy
        if self.config.use_lru_eviction:
            self.eviction_policy = LRUPolicy(self.config.file_access_time_threshold_minutes)
            self.logger.info("Initialized Eviction Policy: Least-Recently-Used (LRU) Cache Policy.")
        else:
            self.eviction_policy = AgeTTLPolicy(self.config.file_access_time_threshold_minutes)
            self.logger.info("Initialized Eviction Policy: Legacy Age/TTL Policy.")

        # 2. Compose Discovery Coordinator
        cache_path = Path(self.config.pvc_mount_path) / self.config.cache_directory
        self.discovery_coordinator = DiscoveryCoordinator(
            self.config,
            self.eviction_policy,
            cache_path,
            spawn_scanner_fn=self._spawn_crawlers
        )

        # 3. Compose Continuous Background Purge
        self.continuous_cleanup = ContinuousCleanup(self.config, cache_path, self.eviction_policy)

        # Crawlers pool tracking
        self.crawler_processes = []
        self.feeder_thread = None

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
            except OSError as exc:
                # Continue retrying, but log the error to aid diagnostics.
                self.logger.warning(
                    "Error while checking PVC mount path '%s': %s",
                    self.config.pvc_mount_path,
                    exc,
                )

            time.sleep(wait_interval)
            waited += wait_interval
            self.logger.info(f"Still waiting for mount... ({waited}s/{max_wait}s)")

        self.logger.error(f"PVC mount path not available after {max_wait}s")
        sys.exit(1)

    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully.

        This handler is called when Kubernetes/kubelet sends SIGTERM (or user sends SIGINT).
        It coordinates graceful shutdown across all processes:

        1. Sets shutdown_event - All child processes check this in their loops and exit
        2. Clears deletion_event - Immediately stops any ongoing deletion operations
        3. Sets running = False - Causes main loop to exit
        4. Main process then waits for all child processes in run() finally block
        """
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.shutdown_event.set()  # Signal all processes to shutdown
        self.deletion_event.clear()  # Stop deletion immediately
        self.continuous_cleanup.stop()
        self.discovery_coordinator.stop()

    def _spawn_crawlers(self):
        """Spawn P1-PN Crawler processes when FS scanning is required."""
        cache_path = Path(self.config.pvc_mount_path) / self.config.cache_directory
        hex_ranges = get_hex_modulo_ranges(self.config.num_crawler_processes, self.config.shard_index, self.config.total_shards)

        for i in range(self.config.num_crawler_processes):
            hex_range = hex_ranges[i]
            process = multiprocessing.Process(
                target=crawler_process,
                args=(
                    i,
                    hex_range,
                    cache_path,
                    self.config_dict,
                    self.deletion_event,
                    self.deletion_queue,
                    self.result_queue,
                    self.shutdown_event,
                    self.folder_queue,
                ),
                name=f"Crawler-P{i + 1}",
            )
            process.start()
            self.crawler_processes.append(process)
            modulo_range_min, modulo_range_max = hex_range[0], hex_range[1]
            hex_chars = f"'{format(modulo_range_min, '03x')}'" if modulo_range_min == modulo_range_max else f"'{format(modulo_range_min, '03x')}'-'{format(modulo_range_max, '03x')}'"
            self.logger.info(
                f"Started crawler P{i + 1} (hex %4096 in [{modulo_range_min}, {modulo_range_max}], hex: {hex_chars})"
            )

    def run(self):
        """Main coordination loop - spawns and manages all processes."""
        total_processes = self.config.num_crawler_processes + 1 + self.config.num_deleter_processes
        self.logger.info(f"Starting {total_processes}-process evictor service...")

        cache_path = Path(self.config.pvc_mount_path) / self.config.cache_directory

        # 1. Start Continuous Absolute age purge
        self.continuous_cleanup.start()

        # 2. Start pluggable discovery (scanner and/or events coordinator)
        self.discovery_coordinator.start(self.result_queue)

        # Spawn P(N+1): Activator process
        activator_process_num = self.config.num_crawler_processes + 1
        activator_process_obj = multiprocessing.Process(
            target=activator_process,
            args=(
                activator_process_num,
                self.config.pvc_mount_path,
                self.config.cleanup_threshold,
                self.config.target_threshold,
                self.config.logger_interval,
                self.deletion_event,
                self.result_queue,
                self.shutdown_event,
                self.config_dict,
            ),
            name=f"Activator-P{activator_process_num}",
        )
        activator_process_obj.start()
        self.logger.info(f"Started activator P{activator_process_num}")

        # 3. If in LRU mode, spawn background deletion queue feeder thread
        if self.config.use_lru_eviction:
            self.feeder_thread = threading.Thread(target=self._feed_deletion_queue_loop, daemon=True)
            self.feeder_thread.start()
            self.logger.info("Started in-process queue feeder thread for LRU eviction.")

        # Spawn P(N+2) - P(N+1+M): Deleter processes
        deleter_processes = []
        for j in range(self.config.num_deleter_processes):
            deleter_process_num = self.config.num_crawler_processes + 2 + j
            process = multiprocessing.Process(
                target=deleter_process,
                args=(
                    deleter_process_num,
                    cache_path,
                    self.config_dict,
                    self.deletion_event,
                    self.deletion_queue,
                    self.result_queue,
                    self.shutdown_event,
                    self.folder_queue,
                ),
                name=f"Deleter-P{deleter_process_num}",
            )
            process.start()
            deleter_processes.append(process)
            self.logger.info(f"Started deleter P{deleter_process_num}")

        # Spawn P(N+2+M): Folder Cleaner process (NEW)
        self.folder_cleaner_process_obj = None
        if self.config.enable_dir_cleanup:
            cleaner_process_num = self.config.num_crawler_processes + 2 + self.config.num_deleter_processes
            self.folder_cleaner_process_obj = multiprocessing.Process(
                target=folder_cleaner_process,
                args=(
                    cleaner_process_num,
                    self.folder_queue,
                    self.result_queue,
                    self.shutdown_event,
                    self.config_dict,
                ),
                name=f"FolderCleaner-P{cleaner_process_num}",
            )
            self.folder_cleaner_process_obj.start()
            self.logger.info(f"Started background folder cleaner P{cleaner_process_num}")

        # Monitor processes and handle results
        # Aggregated logging state
        crawler_stats = {}  # {process_num: {stats_dict}}
        activator_stats = {}  # {process_num: {stats_dict}}
        deleter_stats = {}  # {process_num: {stats_dict}}
        folder_cleaner_stats = {}  # {process_num: {stats_dict}}
        last_aggregated_log_time = time.time()

        try:
            while self.running:
                try:
                    # Check for deletion results
                    result = self.result_queue.get(timeout=5.0)
                    result_type, *data = result

                    if result_type == "discovered_files_batch":
                        batch = data[0]
                        for file_path, stat_info in batch:
                            self.eviction_policy.record_file_discovery(file_path, stat_info)

                    elif result_type == "zmq_active":
                        self.logger.warning("Dynamic connection to ZMQ established! Minimizing filesystem crawling.")

                    elif result_type == "progress":
                        process_num, files_deleted, bytes_freed, folders_deleted = data
                        # Update deleter stats for aggregated logging
                        deleter_stats[process_num] = {
                            "files_deleted": files_deleted,
                            "bytes_freed": bytes_freed,
                            "folders_deleted": folders_deleted,
                        }
                    elif result_type == "done":
                        process_num, files_deleted, bytes_freed, folders_deleted = data
                        self.logger.info(
                            f"Deletion P{process_num} complete: {files_deleted} files, {folders_deleted} folders, {bytes_freed / (1024**3):.2f}GB freed"
                        )
                    elif result_type == "crawler_stats":
                        process_num, stats = data
                        crawler_stats[process_num] = stats
                    elif result_type == "activator_stats":
                        process_num, stats = data
                        activator_stats[process_num] = stats
                    elif result_type == "folder_cleaner_stats":
                        process_num, stats = data
                        folder_cleaner_stats[process_num] = stats

                    # Periodically log aggregated stats
                    current_time = time.time()
                    if current_time - last_aggregated_log_time >= AGGREGATED_LOGGING_INTERVAL_SECONDS:
                        log_aggregated_stats(
                            self.logger,
                            crawler_stats,
                            activator_stats,
                            deleter_stats,
                            self.config.cleanup_threshold,
                            self.config.target_threshold,
                            folder_cleaner_stats,
                        )
                        last_aggregated_log_time = current_time

                except Exception:
                    # Timeout or queue empty - continue monitoring
                    # Check if processes are still alive
                    activator_process_num = self.config.num_crawler_processes + 1
                    if not activator_process_obj.is_alive():
                        self.logger.error(f"Activator P{activator_process_num} died, restarting...")
                        activator_process_obj = multiprocessing.Process(
                            target=activator_process,
                            args=(
                                activator_process_num,
                                self.config.pvc_mount_path,
                                self.config.cleanup_threshold,
                                self.config.target_threshold,
                                self.config.logger_interval,
                                self.deletion_event,
                                self.result_queue,
                                self.shutdown_event,
                                self.config_dict,
                            ),
                            name=f"Activator-P{activator_process_num}",
                        )
                        activator_process_obj.start()

                    for j in range(self.config.num_deleter_processes):
                        deleter_process_num = self.config.num_crawler_processes + 2 + j
                        proc = deleter_processes[j]
                        if not proc.is_alive():
                            self.logger.error(f"Deleter P{deleter_process_num} died, restarting...")
                            new_proc = multiprocessing.Process(
                                target=deleter_process,
                                args=(
                                    deleter_process_num,
                                    cache_path,
                                    self.config_dict,
                                    self.deletion_event,
                                    self.deletion_queue,
                                    self.result_queue,
                                    self.shutdown_event,
                                    self.folder_queue,
                                ),
                                name=f"Deleter-P{deleter_process_num}",
                            )
                            new_proc.start()
                            deleter_processes[j] = new_proc

                    if self.config.enable_dir_cleanup and self.folder_cleaner_process_obj and not self.folder_cleaner_process_obj.is_alive():
                        cleaner_process_num = self.config.num_crawler_processes + 2 + self.config.num_deleter_processes
                        self.logger.error(f"Folder Cleaner P{cleaner_process_num} died, restarting...")
                        self.folder_cleaner_process_obj = multiprocessing.Process(
                            target=folder_cleaner_process,
                            args=(
                                cleaner_process_num,
                                self.folder_queue,
                                self.result_queue,
                                self.shutdown_event,
                                self.config_dict,
                            ),
                            name=f"FolderCleaner-P{cleaner_process_num}",
                        )
                        self.folder_cleaner_process_obj.start()

                    time.sleep(1.0)

        except KeyboardInterrupt:
            self.logger.warning("Shutdown requested, stopping all processes...")

        finally:
            # Graceful shutdown
            self.logger.info("Shutting down all processes...")
            self.shutdown_event.set()
            self.deletion_event.clear()

            # Wait for processes to finish
            for process in self.crawler_processes:
                process.join(timeout=10)
                if process.is_alive():
                    self.logger.warning(f"Process {process.name} did not terminate, forcing...")
                    process.terminate()
                    process.join(timeout=5)

            activator_process_obj.join(timeout=10)
            if activator_process_obj.is_alive():
                activator_process_obj.terminate()
                activator_process_obj.join(timeout=5)

            for proc in deleter_processes:
                proc.join(timeout=30)
                if proc.is_alive():
                    self.logger.warning(f"Deleter {proc.name} did not terminate, forcing...")
                    proc.terminate()
                    proc.join(timeout=5)

            if self.config.enable_dir_cleanup and self.folder_cleaner_process_obj:
                cleaner_process_num = self.config.num_crawler_processes + 2 + self.config.num_deleter_processes
                self.folder_cleaner_process_obj.join(timeout=10)
                if self.folder_cleaner_process_obj.is_alive():
                    self.logger.warning(f"Folder Cleaner P{cleaner_process_num} did not terminate, forcing...")
                    self.folder_cleaner_process_obj.terminate()
                    self.folder_cleaner_process_obj.join(timeout=5)

            self.logger.info("All processes stopped")

    def _feed_deletion_queue_loop(self):
        """Background thread feeding deletion_queue with candidate paths when deletion is active."""
        batch_size = self.config.deletion_batch_size
        while self.running:
            if self.deletion_event.is_set():
                candidates = self.eviction_policy.get_eviction_candidates(batch_size)
                if candidates:
                    for p in candidates:
                        queued = False
                        while self.running and not queued:
                            try:
                                # Blocking put with timeout to check shutdown flag occasionally
                                self.deletion_queue.put(p, timeout=1.0)
                                queued = True
                            except queue.Full:
                                continue
                else:
                    time.sleep(1.0)
            else:
                time.sleep(0.5)


def main():
    """Main entry point."""
    print("PVC Evictor starting...", flush=True)

    try:
        config = Config.from_env()
        setup_logging(config.log_level, None, config.log_file_path)

        # Validate crawler count before creating evictor
        try:
            get_hex_modulo_ranges(config.num_crawler_processes, config.shard_index, config.total_shards)
        except ValueError as e:
            print(f"ERROR: {e}", flush=True)
            sys.exit(1)

        print(
            f"Configuration loaded: PVC={config.pvc_mount_path}, "
            f"Shard={config.shard_index}/{config.total_shards}, "
            f"Crawlers={config.num_crawler_processes}, "
            f"Deleters={config.num_deleter_processes}, "
            f"Total Processes={config.num_crawler_processes + 1 + config.num_deleter_processes}",
            flush=True,
        )

        evictor = PVCEvictor(config)
        print("PVC Evictor initialized, starting coordination loop...", flush=True)
        evictor.run()
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


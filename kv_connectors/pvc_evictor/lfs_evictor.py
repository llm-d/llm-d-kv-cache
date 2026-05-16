#!/usr/bin/env python3
"""
PVC Evictor - LFS-Based Multi-Process Architecture

3-Process Architecture:
- P1: LFS Crawler process that discovers files using lfs find and queues them
- P2: Activator process that monitors disk usage and controls deletion
- P3: Deleter process that performs actual file deletions
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
from processes.deleter import deleter_process
from processes.lfs_crawler import lfs_crawler_process
from utils.logging_helpers import (
    AGGREGATED_LOGGING_INTERVAL_SECONDS,
    log_aggregated_stats,
)
from utils.system import setup_logging


class LFSPVCEvictor:
    """Main evictor controller coordinating 3 processes (LFS crawler + activator + deleter)."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = True

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self._wait_for_mount()

        self.deletion_event = multiprocessing.Event()
        self.deletion_queue = multiprocessing.Queue(maxsize=config.file_queue_maxsize)
        self.result_queue = multiprocessing.Queue()
        self.shutdown_event = multiprocessing.Event()

        self.config_dict = self.config.to_dict()

        self.logger.info(
            f"LFS PVC Cleanup Service (Multi-Process Architecture: "
            f"1 + 1 + {config.num_deleter_processes} total processes) initialized"
        )
        self.logger.info(f"  Mount Path: {config.pvc_mount_path}")
        self.logger.info(f"  Cache Directory: {config.cache_directory}")
        self.logger.info("  Crawler Process: P1 (LFS find streaming)")
        self.logger.info(f"  Activator Process: P2 (monitoring every {config.logger_interval}s)")
        self.logger.info(f"  Deleter Processes: {config.num_deleter_processes} (batch size: {config.deletion_batch_size})")
        self.logger.info(f"  Cleanup Threshold: {config.cleanup_threshold}%")
        self.logger.info(f"  Target Threshold: {config.target_threshold}%")
        self.logger.info(
            f"  File Queue: MINQ={config.file_queue_min_size} (pre-fill when OFF), "
            f"MAXQ={config.file_queue_maxsize} (max when ON)"
        )

    def _wait_for_mount(self):
        max_wait = 60
        wait_interval = 2
        waited = 0

        while waited < max_wait:
            try:
                if os.path.exists(self.config.pvc_mount_path):
                    self.logger.info(f"PVC mount path is ready: {self.config.pvc_mount_path}")
                    return
            except OSError as exc:
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
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.shutdown_event.set()
        self.deletion_event.clear()

    def run(self):
        total_processes = 1 + 1 + self.config.num_deleter_processes
        self.logger.info(f"Starting {total_processes}-process LFS evictor service...")
        cache_path = Path(self.config.pvc_mount_path) / self.config.cache_directory

        # Spawn P1: LFS Crawler
        crawler_process_obj = multiprocessing.Process(
            target=lfs_crawler_process,
            args=(
                0,  # process_id = 0 -> P1
                cache_path,
                self.config_dict,
                self.deletion_event,
                self.deletion_queue,
                self.result_queue,
                self.shutdown_event,
            ),
            name="LFS-Crawler-P1",
        )
        crawler_process_obj.start()
        self.logger.info("Started LFS crawler P1")

        # Spawn P2: Activator process
        activator_process_num = 2
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
            ),
            name=f"Activator-P{activator_process_num}",
        )
        activator_process_obj.start()
        self.logger.info(f"Started activator P{activator_process_num}")

        # Spawn P3 - P(2+M): Deleter processes
        deleter_processes = []
        for j in range(self.config.num_deleter_processes):
            deleter_process_num = 3 + j
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
                ),
                name=f"Deleter-P{deleter_process_num}",
            )
            process.start()
            deleter_processes.append(process)
            self.logger.info(f"Started deleter P{deleter_process_num}")

        crawler_stats = {}
        activator_stats = {}
        deleter_stats = {}
        last_aggregated_log_time = time.time()

        try:
            while self.running:
                try:
                    if os.path.exists("/pod-shared/simulation.complete"):
                        self.logger.info("Simulation complete signal detected at /pod-shared/simulation.complete.")
                        if self.deletion_queue.empty():
                            self.logger.info("Local deletion queue is empty. Shutting down evictor sidecar gracefully...")
                            self.running = False
                            break

                    result = self.result_queue.get(timeout=5.0)
                    result_type, *data = result

                    if result_type == "progress":
                        process_num, files_deleted, bytes_freed = data
                        deleter_stats[process_num] = {
                            "files_deleted": files_deleted,
                            "bytes_freed": bytes_freed,
                        }
                    elif result_type == "done":
                        process_num, files_deleted, bytes_freed = data
                        self.logger.info(
                            f"Deletion P{process_num} complete: {files_deleted} files, {bytes_freed / (1024**3):.2f}GB freed"
                        )
                    elif result_type == "crawler_stats":
                        process_num, stats = data
                        crawler_stats[process_num] = stats
                    elif result_type == "activator_stats":
                        process_num, stats = data
                        activator_stats[process_num] = stats

                    current_time = time.time()
                    if current_time - last_aggregated_log_time >= AGGREGATED_LOGGING_INTERVAL_SECONDS:
                        log_aggregated_stats(
                            self.logger,
                            crawler_stats,
                            activator_stats,
                            deleter_stats,
                            self.config.cleanup_threshold,
                            self.config.target_threshold,
                        )
                        last_aggregated_log_time = current_time

                except Exception:
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
                            ),
                            name=f"Activator-P{activator_process_num}",
                        )
                        activator_process_obj.start()

                    for j in range(self.config.num_deleter_processes):
                        deleter_process_num = 3 + j
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
                                ),
                                name=f"Deleter-P{deleter_process_num}",
                            )
                            new_proc.start()
                            deleter_processes[j] = new_proc

                    if not crawler_process_obj.is_alive():
                        self.logger.error("LFS Crawler P1 died, restarting...")
                        crawler_process_obj = multiprocessing.Process(
                            target=lfs_crawler_process,
                            args=(
                                0,
                                cache_path,
                                self.config_dict,
                                self.deletion_event,
                                self.deletion_queue,
                                self.result_queue,
                                self.shutdown_event,
                            ),
                            name="LFS-Crawler-P1",
                        )
                        crawler_process_obj.start()

                    time.sleep(1.0)

        except KeyboardInterrupt:
            self.logger.warning("Shutdown requested, stopping all processes...")

        finally:
            self.logger.info("Shutting down all processes...")
            self.shutdown_event.set()
            self.deletion_event.clear()

            crawler_process_obj.join(timeout=10)
            if crawler_process_obj.is_alive():
                self.logger.warning("Process LFS-Crawler-P1 did not terminate, forcing...")
                crawler_process_obj.terminate()
                crawler_process_obj.join(timeout=5)

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

            self.logger.info("All processes stopped")


def main():
    """Main entry point."""
    print("LFS PVC Evictor starting...", flush=True)

    try:
        config = Config.from_env()
        setup_logging(config.log_level, None, config.log_file_path)

        print(
            f"Configuration loaded: PVC={config.pvc_mount_path}, "
            "Crawlers=1 (LFS streaming), "
            f"Deleters={config.num_deleter_processes}, "
            f"Total Processes={1 + 1 + config.num_deleter_processes}",
            flush=True,
        )

        evictor = LFSPVCEvictor(config)
        print("LFS PVC Evictor initialized, starting coordination loop...", flush=True)
        evictor.run()
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

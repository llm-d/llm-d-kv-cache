#!/usr/bin/env python3
"""
PVC Evictor - Multi-Process Architecture

N+2 Process Architecture:
- P1-PN: Crawler processes that discover and tag files for deletion (N configurable: 1, 2, 4, 8, or 16)
- P(N+1): Activator process that monitors disk usage and controls deletion
- P(N+2): Deleter process that performs actual file deletions
"""

import os
import sys
import time
import logging
import signal
import multiprocessing
import traceback
from pathlib import Path

from config import Config
from utils.system import setup_logging
from processes.crawler import crawler_process, get_hex_modulo_ranges
from processes.activator import activator_process
from processes.deleter import deleter_process


class PVCEvictor:
    """Main evictor controller coordinating N+2 processes (N crawlers + activator + deleter)."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = True

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Wait for PVC mount
        self._wait_for_mount()

        # Inter-Process Communication (IPC):
        # - shutdown_event: multiprocessing.Event - shared boolean flag, all processes check this in their loops
        # - deletion_event: multiprocessing.Event - shared boolean flag, activator controls deleter via this
        # - file_queue: multiprocessing.Queue - FIFO queue, crawlers put files, deleter gets files
        # - result_queue: multiprocessing.Queue - FIFO queue, deleter reports progress to main
        # Events use shared memory, queues use pipes with pickling

        # Initialize shared objects for IPC
        self.deletion_event = (
            multiprocessing.Event()
        )  # Activator controls Deleter, Crawlers check this
        self.file_queue = multiprocessing.Queue(
            maxsize=config.file_queue_maxsize
        )  # Crawlers → Deleter
        self.result_queue = multiprocessing.Queue()  # Deleter → Main
        self.shutdown_event = multiprocessing.Event()  # All processes check this

        # Convert Config to dict for pickling (needed for multiprocessing)
        self.config_dict = {
            "pvc_mount_path": self.config.pvc_mount_path,
            "cleanup_threshold": self.config.cleanup_threshold,
            "target_threshold": self.config.target_threshold,
            "cache_directory": self.config.cache_directory,
            "dry_run": self.config.dry_run,
            "log_level": self.config.log_level,
            "timing_file_path": self.config.timing_file_path,
            "deletion_batch_size": self.config.deletion_batch_size,
            "file_queue_min_size": self.config.file_queue_min_size,
            "file_queue_maxsize": self.config.file_queue_maxsize,
            "log_file_path": self.config.log_file_path,
            "file_access_time_threshold_minutes": self.config.file_access_time_threshold_minutes,
        }

        self.logger.info("PVC Cleanup Service v4 (10-Process Architecture) initialized")
        self.logger.info(f"  Mount Path: {config.pvc_mount_path}")
        self.logger.info(f"  Cache Directory: {config.cache_directory}")
        self.logger.info(
            f"  Crawler Processes: {config.num_crawler_processes} (P1-P{config.num_crawler_processes})"
        )
        activator_process_num = config.num_crawler_processes + 1
        deleter_process_num = config.num_crawler_processes + 2
        self.logger.info(
            f"  Activator Process: P{activator_process_num} (monitoring every {config.logger_interval}s)"
        )
        self.logger.info(
            f"  Deleter Process: P{deleter_process_num} (batch size: {config.deletion_batch_size})"
        )
        self.logger.info(f"  Cleanup Threshold: {config.cleanup_threshold}%")
        self.logger.info(f"  Target Threshold: {config.target_threshold}%")
        self.logger.info(
            f"  File Queue: MINQ={config.file_queue_min_size} (pre-fill when OFF), MAXQ={config.file_queue_maxsize} (max when ON)"
        )

    def _wait_for_mount(self):
        """Wait for PVC mount to be ready."""
        max_wait = 60
        wait_interval = 2
        waited = 0

        while waited < max_wait:
            try:
                if os.path.exists(self.config.pvc_mount_path):
                    self.logger.info(
                        f"PVC mount path is ready: {self.config.pvc_mount_path}"
                    )
                    return
            except Exception:
                pass

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
        
        (NOT hardcoded control - it's a description of how we use the event)
        """
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.shutdown_event.set()  # Signal all processes to shutdown
        self.deletion_event.clear()  # Stop deletion immediately

    def run(self):
        """Main coordination loop - spawns and manages all processes."""
        total_processes = self.config.num_crawler_processes + 2
        self.logger.info(f"Starting {total_processes}-process evictor service...")

        cache_path = Path(self.config.pvc_mount_path) / self.config.cache_directory

        # Get hex modulo ranges for crawlers
        hex_ranges = get_hex_modulo_ranges(self.config.num_crawler_processes)

        # Spawn P1-PN: Crawler processes (N = num_crawler_processes)
        crawler_processes = []
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
                    self.file_queue,
                    self.shutdown_event,
                ),
                name=f"Crawler-P{i+1}",
            )
            process.start()
            crawler_processes.append(process)
            # Convert decimal range to hex characters for clarity
            modulo_range_min, modulo_range_max = hex_range[0], hex_range[1]
            if modulo_range_min == modulo_range_max:
                hex_chars = f"'{format(modulo_range_min, 'x')}'"
            else:
                hex_chars = f"'{format(modulo_range_min, 'x')}'-'{format(modulo_range_max, 'x')}'"
            self.logger.info(
                f"Started crawler P{i+1} (hex %16 in [{modulo_range_min}, {modulo_range_max}], hex: {hex_chars})"
            )

        # Spawn P(N+1): Activator process
        activator_process_num = self.config.num_crawler_processes + 1
        activator_process_obj = multiprocessing.Process(
            target=activator_process,
            args=(
                self.config.pvc_mount_path,
                self.config.cleanup_threshold,
                self.config.target_threshold,
                self.config.logger_interval,
                self.deletion_event,
                self.shutdown_event,
            ),
            name=f"Activator-P{activator_process_num}",
        )
        activator_process_obj.start()
        self.logger.info(f"Started activator P{activator_process_num}")

        # Spawn P(N+2): Deleter process
        deleter_process_num = self.config.num_crawler_processes + 2
        deleter_process_obj = multiprocessing.Process(
            target=deleter_process,
            args=(
                cache_path,
                self.config_dict,
                self.deletion_event,
                self.file_queue,
                self.result_queue,
                self.shutdown_event,
            ),
            name=f"Deleter-P{deleter_process_num}",
        )
        deleter_process_obj.start()
        self.logger.info(f"Started deleter P{deleter_process_num}")

        # Monitor processes and handle results
        try:
            while self.running:
                try:
                    # Check for deletion results
                    result = self.result_queue.get(timeout=5.0)
                    result_type, *data = result

                    if result_type == "progress":
                        files_deleted, bytes_freed = data
                        self.logger.debug(
                            f"Deletion progress: {files_deleted} files, "
                            f"{bytes_freed / (1024**3):.2f}GB freed"
                        )
                    elif result_type == "done":
                        files_deleted, bytes_freed = data
                        self.logger.info(
                            f"Deletion complete: {files_deleted} files, "
                            f"{bytes_freed / (1024**3):.2f}GB freed"
                        )

                except Exception:
                    # Timeout or queue empty - continue monitoring
                    # Check if processes are still alive
                    activator_process_num = self.config.num_crawler_processes + 1
                    if not activator_process_obj.is_alive():
                        self.logger.error(
                            f"Activator P{activator_process_num} died, restarting..."
                        )
                        activator_process_obj = multiprocessing.Process(
                            target=activator_process,
                            args=(
                                self.config.pvc_mount_path,
                                self.config.cleanup_threshold,
                                self.config.target_threshold,
                                self.config.logger_interval,
                                self.deletion_event,
                                self.shutdown_event,
                            ),
                            name=f"Activator-P{activator_process_num}",
                        )
                        activator_process_obj.start()

                    deleter_process_num = self.config.num_crawler_processes + 2
                    if not deleter_process_obj.is_alive():
                        self.logger.error(
                            f"Deleter P{deleter_process_num} died, restarting..."
                        )
                        deleter_process_obj = multiprocessing.Process(
                            target=deleter_process,
                            args=(
                                cache_path,
                                self.config_dict,
                                self.deletion_event,
                                self.file_queue,
                                self.result_queue,
                                self.shutdown_event,
                            ),
                            name=f"Deleter-P{deleter_process_num}",
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
                    self.logger.warning(
                        f"Process {process.name} did not terminate, forcing..."
                    )
                    process.terminate()
                    process.join(timeout=5)

            activator_process_obj.join(timeout=10)
            if activator_process_obj.is_alive():
                activator_process_obj.terminate()
                activator_process_obj.join(timeout=5)

            deleter_process_num = self.config.num_crawler_processes + 2
            deleter_process_obj.join(
                timeout=30
            )  # Give deleter more time to finish batch
            if deleter_process_obj.is_alive():
                self.logger.warning(
                    f"Deleter P{deleter_process_num} did not terminate, forcing..."
                )
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

        print(
            f"Configuration loaded: PVC={config.pvc_mount_path}, Crawlers={config.num_crawler_processes}, Total Processes={config.num_crawler_processes + 2}",
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


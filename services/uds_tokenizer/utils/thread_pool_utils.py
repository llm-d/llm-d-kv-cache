# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Thread pool utilities for managing shared thread pools in containerized environments.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor
import threading


# Global thread pool for handling CPU-intensive operations
_thread_pool = None
_thread_pool_lock = threading.Lock()


def get_cpu_count() -> int:
    """
    Get the actual CPU count available to the container, considering cgroup limits.

    This function is designed to work properly in both regular environments
    and containerized environments like Docker and Kubernetes.

    Returns:
        int: The number of CPUs available to the process
    """
    # First check for cgroup v1 CPU quota and period
    try:
        # For cgroup v1
        cpu_quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        cpu_period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"

        if os.path.exists(cpu_quota_path) and os.path.exists(cpu_period_path):
            with open(cpu_quota_path, 'r') as f:
                quota = int(f.read().strip())
            with open(cpu_period_path, 'r') as f:
                period = int(f.read().strip())

            # If quota is -1, there's no limit
            if quota != -1 and period > 0:
                cpu_count = max(1, int(quota / period))
                logging.info(f"Using CPU count from cgroup v1: {cpu_count}")
                return cpu_count
    except Exception as e:
        logging.debug(f"Could not read cgroup v1 CPU limits: {e}")

    # Check for cgroup v2 CPU weight or quota
    try:
        # For cgroup v2
        cpu_max_path = "/sys/fs/cgroup/cpu.max"

        if os.path.exists(cpu_max_path):
            with open(cpu_max_path, 'r') as f:
                cpu_max_content = f.read().strip()

            if cpu_max_content != "max":  # "max" means no limit
                # Format: quota period (e.g., "100000 100000" for 1 CPU)
                parts = cpu_max_content.split()
                if len(parts) == 2:
                    quota = int(parts[0])
                    period = int(parts[1])
                    if period > 0:
                        cpu_count = max(1, int(quota / period))
                        logging.info(f"Using CPU count from cgroup v2: {cpu_count}")
                        return cpu_count
    except Exception as e:
        logging.debug(f"Could not read cgroup v2 CPU limits: {e}")

    # Finally, fall back to multiprocessing.cpu_count()
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"Using CPU count from multiprocessing.cpu_count(): {cpu_count}")
    return cpu_count


def get_thread_pool_size(multiplier=2, max_workers=32) -> int:
    """
    Get an appropriate thread pool size based on available CPUs.

    Args:
        multiplier (int): Multiplier for CPU count to determine thread pool size
        max_workers (int): Maximum number of workers allowed

    Returns:
        int: The calculated thread pool size
    """
    cpu_count = get_cpu_count()
    default_thread_pool_size = min(cpu_count * multiplier, max_workers)
    thread_pool_size = int(os.getenv("THREAD_POOL_SIZE", default_thread_pool_size))
    logging.info(f"Calculated thread pool size: {thread_pool_size} (CPU count: {cpu_count}, multiplier: {multiplier}, max: {max_workers})")
    return thread_pool_size


def get_thread_pool():
    """
    Get a shared thread pool for handling CPU-intensive operations.

    Returns:
        ThreadPoolExecutor: Shared thread pool instance
    """
    global _thread_pool
    if _thread_pool is None:
        with _thread_pool_lock:
            if _thread_pool is None:
                # Get thread pool size from environment variable or calculate based on CPU count
                pool_size = get_thread_pool_size()
                _thread_pool = ThreadPoolExecutor(max_workers=pool_size)
                logging.info(f"Created shared thread pool with {pool_size} workers")
    return _thread_pool
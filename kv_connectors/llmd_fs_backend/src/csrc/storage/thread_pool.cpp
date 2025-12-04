/*
 * Copyright 2025 The llm-d Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <sys/syscall.h>
#include <unistd.h>
#include <numa.h>

#include "thread_pool.hpp"
#include "buffer.hpp"
#include "debug_utils.hpp"

// Thread-local index for CUDA streams
extern thread_local size_t thread_stream_idx;

// ThreadPool constructor
ThreadPool::ThreadPool(size_t threads,
                       size_t staging_buffer_mb,
                       int tp_rank,
                       int device_id)
    : m_device_id(device_id) {
    // Initialize PyTorch threading globally (main thread only)
    // at::init_num_threads();
    // at::set_num_threads(1);

    // Get GPU NUMA node ONCE outside the thread loop
    int gpu_numa = get_gpu_numa_node(device_id);
    std::cout << "[INFO] GPU " << device_id << " mapped to NUMA node "
              << gpu_numa << "\n";

    // Get all CPUs in that NUMA node
    auto local_cpus = get_cpus_in_numa_node(gpu_numa);

    if (local_cpus.empty()) {
        std::cerr << "[WARN] No CPUs found for NUMA node " << gpu_numa
                  << ". System may not be NUMA-aware. Using all CPUs.\n";
        // Populate with all available CPUs as fallback
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        for (int i = 0; i < num_cpus; ++i) {
            local_cpus.push_back(i);
        }
    }

    // Log available CPUs
    std::cout << "CPUs available for GPU " << device_id << " (NUMA " << gpu_numa
              << "): ";
    for (int cpu : local_cpus) std::cout << cpu << " ";
    std::cout << "\n";

    // Create all worker threads
    for (size_t i = 0; i < threads; ++i) {
        // Launch a new worker thread with a lambda that initializes thread
        // resources and processes queued tasks.
        workers.emplace_back([this,
                              i,
                              threads,
                              staging_buffer_mb,
                              tp_rank,
                              device_id,
                              gpu_numa,
                              local_cpus] {
            cudaSetDevice(device_id);

            // Round-robin CPUs within the NUMA node
            // TODO: Re-evaluate whether strict NUMA-based round-robin CPU
            // assignment is optimal for performance.
            int cpu_id = local_cpus[i % local_cpus.size()];

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_id, &cpuset);

            if (pthread_setaffinity_np(pthread_self(),
                                       sizeof(cpuset),
                                       &cpuset) != 0) {
                std::cerr << "[ERROR] Failed to set affinity for thread " << i
                          << " to CPU " << cpu_id << "\n";
            }

            DEBUG_PRINT("IO thread " << i << " set CUDA device to " << device_id
                                     << ", tp_rank=" << tp_rank
                                     << ") pinned to CPU " << cpu_id);

            // Attach preallocated staging buffer for this thread
            if (i < g_staging_buffers.size() &&
                g_staging_buffers[i].ptr != nullptr) {
                t_staging_buffer.ptr = g_staging_buffers[i].ptr;
                t_staging_buffer.size = g_staging_buffers[i].size;
                DEBUG_PRINT("IO thread "
                            << i << " attached to preallocated staging buffer "
                            << (t_staging_buffer.size / (1024 * 1024))
                            << " MB");
            } else {
                std::cerr << "[WARN] IO thread " << i
                          << " has no preallocated staging buffer; it will "
                             "allocate one on first use.\n";
            }

            // Each thread gets its own CUDA stream index
            thread_stream_idx = i;

            // Worker loop
            while (true) {
                std::function<void()> task;
                {
                    // Lock the task queue before checking it
                    std::unique_lock<std::mutex> lock(queue_mutex);

                    // Wait until either a new task arrives or the pool is
                    // stopping. (wait() unlocks the mutex while sleeping and
                    // re-locks it when waking)
                    condition.wait(lock,
                                   [this] { return stop || !tasks.empty(); });

                    // Exit thread if pool is stopping and no tasks remain
                    if (stop && tasks.empty()) return;

                    // Fetch next task from the queue
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                try {
                    // Execute the task
                    task();
                } catch (const std::exception& e) {
                    std::cerr
                        << "[ERROR] Exception in worker thread: " << e.what()
                        << "\n";
                } catch (...) {
                    std::cerr << "[ERROR] Unknown exception in worker thread\n";
                }
            }
        });
    }

    std::cout << "[INFO] All " << threads
              << " I/O threads initialized with staging buffers\n";
}

// ThreadPool destructor
ThreadPool::~ThreadPool() {
    stop = true;
    condition.notify_all();
    // Wait for all worker threads to exit
    for (std::thread& worker : workers) {
        worker.join();
    }
}

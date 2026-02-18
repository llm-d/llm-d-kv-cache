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
#include "numa_utils.hpp"
#include "logger.hpp"

// Minimum staging buffer size: 16 MB
const size_t MIN_STAGING_BUFFER_SIZE = 16 * 1024 * 1024;

// Define static thread-local members
thread_local StagingBufferInfo ThreadPool::m_staging_buffer{};
thread_local at::cuda::CUDAStream ThreadPool::m_thread_stream =
    at::cuda::getStreamFromPool();

// ThreadPool constructor
ThreadPool::ThreadPool(size_t threads,
                       size_t staging_buffer_bytes,
                       int device_id,
                       size_t read_preferring_workers)
    : m_device_id(device_id) {
  // Calculate read preferrig count
  size_t num_first_read = std::min(read_preferring_workers, threads);

  // Initialize preferences
  m_worker_preferences.reserve(threads);
  for (size_t i = 0; i < threads; i++) {
    WorkerPreferenceType::Type pref = (i < num_read_first)
                                          ? WorkerPreferenceType::READ_FIRST
                                          : WorkerPreferenceType::WRITE_FIRST;
    m_worker_preferences.push_back(pref);
  }

  FS_LOG_INFO("ThreadPool: "
              << num_read_first << " reading-preferring workers, "
              << (threads - num_read_first) << " write-preferring workers");

  // Enable GPU access to mapped host memory (needed only for
  // cudaHostAllocMapped before any CUDA context)
  cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (flags_err != cudaSuccess) {
    FS_LOG_WARN("cudaSetDeviceFlags(cudaDeviceMapHost) failed: "
                << cudaGetErrorString(flags_err));
  }

  int gpu_numa = get_gpu_numa_node(device_id);
  std::vector<int> local_cpus;
  if (gpu_numa >= 0) {
    FS_LOG_INFO("GPU " << device_id << " mapped to NUMA node " << gpu_numa);
    // Get all CPUs in that NUMA node
    local_cpus = get_cpus_in_numa_node(gpu_numa);
    // Bind memory allocations in this thread to the NUMA node local to the GPU.
    numa_set_preferred(gpu_numa);
  }

  if (local_cpus.empty()) {
    FS_LOG_WARN("No CPUs found for NUMA node "
                << gpu_numa
                << ". System may not be NUMA-aware. Using all CPUs.");
    // Populate with all available CPUs as fallback
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < num_cpus; ++i) {
      local_cpus.push_back(i);
    }
  }

  // Log available CPUs
  {
    std::ostringstream cpu_list;
    for (int cpu : local_cpus) cpu_list << cpu << " ";
    FS_LOG_INFO("CPUs available for GPU " << device_id << " (NUMA " << gpu_numa
                                          << "): " << cpu_list.str());
  }

  // Create all worker threads
  for (size_t i = 0; i < threads; ++i) {
    // Launch a new worker thread with a lambda that initializes thread
    // resources and processes queued tasks.
    m_workers.emplace_back([this,
                            i,
                            threads,
                            staging_buffer_bytes,
                            device_id,
                            gpu_numa,
                            local_cpus] {
      cudaError_t err = cudaSetDevice(device_id);
      if (err != cudaSuccess) {
        FS_LOG_ERROR("cudaSetDevice failed for device "
                     << device_id << ": " << cudaGetErrorString(err));
      }

      // Round-robin CPUs within the NUMA node
      // TODO: Re-evaluate whether strict NUMA-based round-robin CPU
      // assignment is optimal for performance.
      int cpu_id = local_cpus[i % local_cpus.size()];

      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(cpu_id, &cpuset);

      if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) !=
          0) {
        FS_LOG_ERROR("Failed to set affinity for thread " << i << " to CPU "
                                                          << cpu_id);
      }

      FS_LOG_DEBUG("IO thread " << i << " set CUDA device to " << device_id
                                << " pinned to CPU " << cpu_id);

      // Allocate thread-local staging buffer for this IO thread
      auto ok = ThreadPool::allocate_staging_buffer(staging_buffer_bytes);

      if (!ok) {
        FS_LOG_ERROR("Failed to allocate staging buffer for IO thread " << i);
        return;
      }

      FS_LOG_DEBUG("IO thread " << i << " allocated staging buffer "
                                << (m_staging_buffer.size / (1024 * 1024))
                                << " MB");

      // Set thread to a dedicated CUDA stream for async task.
      at::cuda::setCurrentCUDAStream(ThreadPool::m_thread_stream);

      WorkerPreference::Type preference = m_worker_preferences[i];

      // Worker loop
      while (true) {
        std::function<void()> task;
        {
          // Lock the task queue before checking it
          std::unique_lock<std::mutex> lock(m_queue_mutex);

          // Wait until either a new task arrives or the pool is
          // stopping. (wait() unlocks the mutex while sleeping and
          // re-locks it when waking)
          m_condition.wait(lock,
                           [this] { return m_stop || has_pending_tasks(); });

          // Exit thread if pool is stopping
          if (m_stop) {
            // Free thread-local staging buffer owned by this worker thread
            auto& buf = ThreadPool::get_staging_buffer();
            if (buf.ptr) {
              cudaFreeHost(buf.ptr);
              buf.ptr = nullptr;
              buf.size = 0;
            }
            return;
          }

          // Prioritize high-priority tasks (reads) over normal tasks (writes)
          // based on preference
          if (preference == WorkerPreference::READ_FIRST) {
            if (!m_high_tasks.empty()) {
              task = std::move(m_high_tasks.front());
              m_high_tasks.pop();
            } else if (!m_normal_tasks.empty()) {
              task = std::move(m_normal_tasks.front());
              m_normal_tasks.pop();
            }
          } else {
            if (!m_normal_tasks.empty()) {
              task = std::move(m_normal_tasks.front());
              m_normal_tasks.pop();
            } else if (!m_high_tasks.empty()) {
              task = std::move(m_high_tasks.front());
              m_high_tasks.pop();
            }
          }
        }
        try {
          // Execute the task
          task();
        } catch (const std::exception& e) {
          FS_LOG_ERROR("Exception in worker thread: " << e.what());
        } catch (...) {
          FS_LOG_ERROR("Unknown exception in worker thread");
        }
      }
    });
  }

  FS_LOG_INFO("All " << threads
                     << " I/O threads initialized with staging buffers");
}

// ThreadPool destructor
ThreadPool::~ThreadPool() {
  m_stop = true;
  m_condition.notify_all();
  // Wait for all worker threads to exit
  for (std::thread& worker : m_workers) {
    worker.join();
  }
}

// Get thread-local CUDA stream
at::cuda::CUDAStream& ThreadPool::get_tls_stream() { return m_thread_stream; }

// Allocate the thread-local staging buffer to at least required_bytes
bool ThreadPool::allocate_staging_buffer(size_t required_bytes) {
  size_t alloc_size = std::max(required_bytes, MIN_STAGING_BUFFER_SIZE);
  cudaError_t err = cudaHostAlloc(&m_staging_buffer.ptr,
                                  alloc_size,
                                  cudaHostAllocMapped | cudaHostAllocPortable);

  if (err != cudaSuccess) {
    FS_LOG_ERROR("cudaHostAlloc failed: " << cudaGetErrorString(err));
    m_staging_buffer.ptr = nullptr;
    m_staging_buffer.size = 0;
    return false;
  }

  m_staging_buffer.size = alloc_size;
  FS_LOG_DEBUG("Thread " << std::this_thread::get_id()
                         << " allocated staging buffer "
                         << (alloc_size / (1024 * 1024)) << " MB");
  return true;
}

// Return the thread-local staging buffer
StagingBufferInfo& ThreadPool::get_staging_buffer() { return m_staging_buffer; }

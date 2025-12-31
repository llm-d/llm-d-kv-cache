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
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <future>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <optional>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <filesystem>
#include <numa.h>

#include "storage_offload.hpp"
#include "storage_types.hpp"
#include "file_io.hpp"
#include "numa_utils.hpp"
#include "thread_pool.hpp"
#include "debug_utils.hpp"
#include "tensor_copy.hpp"

// Initialize IO threads, CUDA streams, and staging memory pool
StorageOffloadEngine::StorageOffloadEngine(
    int io_threads,
    int gpu_blocks_per_file,
    std::vector<torch::Tensor>& tensors) {
  // Get current device (should be set by vLLM before calling this)
  int device_id;
  cudaGetDevice(&device_id);

  // Calculate staging buffer size
  auto block_size_in_bytes = 0;
  for (const auto& tensor : tensors) {
    block_size_in_bytes += tensor.stride(0) * tensor.element_size();
  }
  auto staging_buffer_bytes = block_size_in_bytes * gpu_blocks_per_file;
  std::cout << "[INFO] Initializing ThreadPool with " << io_threads
            << " threads on device " << device_id << ", "
            << staging_buffer_bytes / (1024 * 1024)
            << " MB staging buffer per thread\n";

  // Initialize TensorCopy class
  m_tensor_copy = std::make_unique<TensorCopy>(tensors, gpu_blocks_per_file);
  // Enable GPU access to mapped host memory (needed only for
  // cudaHostAllocMapped before any CUDA context)
  cudaSetDeviceFlags(cudaDeviceMapHost);
  int gpu_numa = get_gpu_numa_node(device_id);
  numa_set_preferred(gpu_numa);

  // Pass device_id to thread pool
  m_thread_pool =
      std::make_unique<ThreadPool>(io_threads, staging_buffer_bytes, device_id);
}

// Destructor: stops worker threads and releases CUDA and memory resources.
StorageOffloadEngine::~StorageOffloadEngine() { m_thread_pool.reset(); }

// -------------------------------
// Status and job management
// -------------------------------
// Return finished jobs and their success status
std::vector<std::pair<int, bool>> StorageOffloadEngine::get_finished() {
  std::lock_guard<std::mutex> lock(m_jobs_mutex);

  std::vector<std::pair<int, bool>> results;
  std::vector<int> to_erase;

  // Iterate over all active jobs.
  for (auto& kv : m_jobs) {
    int job_id = kv.first;
    auto& job_state = kv.second;

    // Check if the job has completed all its tasks.
    if (job_state->completed_tasks.load() == job_state->total_tasks.load()) {
      bool all_ok = job_state->all_success.load();
      results.emplace_back(job_id, all_ok);
      to_erase.push_back(job_id);
    }
  }

  // Remove all finished jobs from the map.
  for (int jid : to_erase) {
    m_jobs.erase(jid);
  }
  return results;
}

// Wait for all tasks in the specified job to complete
void StorageOffloadEngine::wait_job(int job_id) {
  std::vector<std::shared_future<bool>> futures;

  {
    std::lock_guard<std::mutex> lock(m_jobs_mutex);
    auto it = m_jobs.find(job_id);
    if (it == m_jobs.end()) return;
    futures = it->second->futures;
  }

  for (auto& fut : futures) {
    fut.wait();
  }
}
// -------------------------------
// Put and Get operations
// -------------------------------
// Async GPU -> Storage transfer (PUT)
bool StorageOffloadEngine::transfer_async_put(
    int job_id,
    std::vector<std::string> dst_files,
    std::vector<std::vector<int64_t>> all_block_ids) {
  // Create job state object that will track progress and futures for this
  // job.
  auto job_state = std::make_unique<JobState>();
  job_state->total_tasks = dst_files.size();

  // For each dst_file file, enqueue one async task in the I/O thread pool.
  for (size_t i = 0; i < dst_files.size(); i++) {
    std::string dst_file = dst_files[i];
    auto block_ids = all_block_ids[i];

    auto future = m_thread_pool->enqueue(
        [this, dst_file, block_ids, job_state = job_state.get()]() -> bool {
          // Check if dst_file file already exists - skip write if it does
          if (std::ifstream(dst_file).good()) {
            update_atime(dst_file);
            job_state->completed_tasks.fetch_add(1);
            return true;  // File exists
          }
          // Ensure correct device is set (thread-local)
          int device_id;
          cudaGetDevice(&device_id);

          // Set thread to a dedicated CUDA stream for async task.
          auto& tls_stream = ThreadPool::tls_stream();
          at::cuda::setCurrentCUDAStream(tls_stream);

          StagingBufferInfo& buf = ThreadPool::tls_staging_buffer();
          auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
          bool is_put = true;
          bool success = false;
          // Execute the copy operation
          try {
            // Stage 1: copy tensors from GPU to staging CPU tensor.
            TIME_EXPR("write phase 1: copy_blocks ",
                      m_tensor_copy->copy_blocks(cpu_base,
                                                 block_ids,
                                                 tls_stream,
                                                 is_put),
                      "file: " + dst_file);

            cudaError_t err = cudaStreamSynchronize(tls_stream.stream());
            if (err != cudaSuccess) {
              std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                        << cudaGetErrorString(err) << std::endl;
            }
            // Stage 2: Write the cpu tensor to disk.
            success = TIME_EXPR(
                "write phase 2: write_tensor_to_file",
                write_tensor_to_file(buf, dst_file),
                ("file:" + dst_file + " size:" + std::to_string(buf.size)));
            if (!success)
              std::cerr << "[ERROR] PUT failed during file write: " << dst_file
                        << "\n";

          } catch (const std::exception& e) {
            std::cerr << "[ERROR] PUT failed for " << dst_file << ": "
                      << e.what() << std::endl;
            success = false;
          } catch (...) {
            std::cerr << "[ERROR] PUT failed for " << dst_file << "\n";
            success = false;
          }

          // Mark task completion.
          job_state->completed_tasks.fetch_add(1);
          // if (!success) job_state->all_success = false; // TODO- silent
          // ignore write failures for now offloading connector not
          // able to handle failures
          return success;
        });
    // Convert std::future -> std::shared_future, which is copyable and can
    // be waited on by multiple threads.
    job_state->futures.push_back(future.share());
  }

  std::lock_guard<std::mutex> lock(m_jobs_mutex);  // protect jobs map
  m_jobs[job_id] = std::move(job_state);

  return true;
}

// Async Storage -> GPU transfer (GET)
bool StorageOffloadEngine::transfer_async_get(
    int job_id,
    std::vector<std::string> src_files,
    std::vector<std::vector<int64_t>> all_block_ids) {
  // Create job state object to track progress and futures for this job.
  auto job_state = std::make_unique<JobState>();
  job_state->total_tasks = src_files.size();

  // For each source file, enqueue one async task in the I/O thread pool.
  for (size_t i = 0; i < src_files.size(); i++) {
    std::string src_file = src_files[i];
    auto block_ids = all_block_ids[i];
    auto future = m_thread_pool->enqueue(
        [this, src_file, block_ids, job_state = job_state.get()]() -> bool {
          // Set thread to a dedicated CUDA stream for async task.
          auto& tls_stream = ThreadPool::tls_stream();
          at::cuda::setCurrentCUDAStream(tls_stream);

          StagingBufferInfo& buf = ThreadPool::tls_staging_buffer();
          bool success = false;
          try {
            // Stage 1: Read file to staging CPU tensor.
            // Read data from disk into a tensor.
            success = TIME_EXPR("read phase 1: read_tensor_from_file",
                                read_tensor_from_file(src_file, buf),
                                ("file:" + src_file));
            if (!success) {
              std::cerr << "[ERROR] Stage1 read_tensor_from_file failed for "
                        << src_file << std::endl;
            } else {
              // Stage 2:  copy tensors from staging CPU tensor to GPU.
              // Perform asynchronous GPU copy and tensor swap.
              auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
              bool is_put = false;
              // Execute the copy operation
              success =
                  TIME_EXPR("read phase 2: copy_cpu_tensor_to_gpu_tensors",
                            m_tensor_copy->copy_blocks(cpu_base,
                                                       block_ids,
                                                       tls_stream,
                                                       is_put),
                            "file: " + src_file);
              cudaError_t err = cudaStreamSynchronize(tls_stream.stream());
              if (err != cudaSuccess) {
                std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                          << cudaGetErrorString(err) << std::endl;
                success = false;
              }
            }
          } catch (const std::exception& e) {
            std::cerr << "[ERROR] GET failed for " << src_file << ": "
                      << e.what() << std::endl;
            success = false;
          } catch (...) {
            std::cerr << "[ERROR] GET unknown failure for " << src_file
                      << std::endl;
            success = false;
          }

          // Final cleanup & accounting
          job_state->completed_tasks.fetch_add(1);
          if (!success) job_state->all_success = false;
          return success;
        });

    // Convert std::future -> std::shared_future - is copyable and can be
    // waited on by multiple threads.
    job_state->futures.push_back(future.share());
  }

  std::lock_guard<std::mutex> lock(m_jobs_mutex);
  m_jobs[job_id] = std::move(job_state);
  return true;
}

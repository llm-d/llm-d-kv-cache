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
#include "file_io.hpp"
#include "numa_utils.hpp"
#include "thread_pool.hpp"
#include "debug_utils.hpp"
#include "tensor_copy.hpp"
#include "cfg.hpp"

// Initialize IO threads, CUDA streams, and staging memory pool
StorageOffloadEngine::StorageOffloadEngine(int io_threads,
                                           size_t staging_buffer_size_mb,
                                           size_t max_staging_memory_gb,
                                           int tp_rank,
                                           int gpu_blocks_per_file,
                                           std::vector<torch::Tensor>& tensors,
                                           bool kv_before_blocks,
                                           bool layers_before_blocks,
                                           int num_blocks_dimension) {
  // Build connector configuration
  CacheLayout layout(tensors,
                     num_blocks_dimension,
                     kv_before_blocks,
                     layers_before_blocks);
  connector_config = std::make_unique<ConnectorConfig>(gpu_blocks_per_file,
                                                       staging_buffer_size_mb,
                                                       layout);

  // Initialize IO thread pool if not already done
  if (!thread_pool) {
    if (io_threads == 0) {
      io_threads = std::max(4u, std::thread::hardware_concurrency() / 2);
    }

    // Get current device (should be set by vLLM before calling this)
    int device_id;
    cudaGetDevice(&device_id);

    std::cout << "[INFO] Initializing ThreadPool with " << io_threads
              << " threads on device " << device_id << ", "
              << staging_buffer_size_mb << " MB staging buffer per thread, "
              << max_staging_memory_gb << " GB max staging memory\n";

    // Enable GPU access to mapped host memory (needed only for
    // cudaHostAllocMapped before any CUDA context)
    cudaSetDeviceFlags(cudaDeviceMapHost);
    int gpu_numa = get_gpu_numa_node(device_id);
    numa_set_preferred(gpu_numa);

    // Pass device_id to thread pool
    thread_pool = std::make_unique<ThreadPool>(io_threads,
                                               staging_buffer_size_mb,
                                               tp_rank,
                                               device_id);
  }
}

// Release IO threads, CUDA streams, and staging buffer
StorageOffloadEngine::~StorageOffloadEngine() {
  thread_pool.reset();
  auto& buf = ThreadPool::tls_staging_buffer();
  if (buf.ptr) {
    cudaFreeHost(buf.ptr);
    buf.ptr = nullptr;
    buf.size = 0;
  }
}

// -------------------------------
// Status and job management
// -------------------------------
// Return finished jobs and their success status
std::vector<std::pair<int, bool>> StorageOffloadEngine::get_finished() {
  std::lock_guard<std::mutex> lock(jobs_mutex);

  std::vector<std::pair<int, bool>> results;
  std::vector<int> to_erase;

  // Iterate over all active jobs.
  for (auto& kv : jobs) {
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
    jobs.erase(jid);
  }
  return results;
}

// Wait for all tasks in the specified job to complete
void StorageOffloadEngine::wait_job(int job_id) {
  std::vector<std::shared_future<bool>> futures;

  {
    std::lock_guard<std::mutex> lock(jobs_mutex);
    auto it = jobs.find(job_id);
    if (it == jobs.end()) return;
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
    std::vector<torch::Tensor> src_tensors,
    std::vector<std::vector<int64_t>> all_block_ids) {
  // Create job state object that will track progress and futures for this
  // job.
  auto job_state = std::make_unique<JobState>();
  job_state->total_tasks = dst_files.size();

  // Store shared_ptr to tensors to avoid repeated refcount changes
  auto shared_src_tensors =
      std::make_shared<std::vector<torch::Tensor>>(std::move(src_tensors));

  // For each dst_file file, enqueue one async task in the I/O thread pool.
  for (size_t i = 0; i < dst_files.size(); i++) {
    std::string dst_file = dst_files[i];
    auto bids = all_block_ids[i];

    auto future = thread_pool->enqueue([this,
                                        dst_file,
                                        bids,
                                        shared_src_tensors,
                                        job_state = job_state.get()]() -> bool {
      // Check if dst_file file already exists - skip write if it does
      if (std::ifstream(dst_file).good()) {
        update_atime(dst_file);
        job_state->completed_tasks.fetch_add(1);
        return true;  // File exists
      }
      // Ensure correct device is set (thread-local)
      int device_id;
      cudaGetDevice(&device_id);

      // Each thread gets a dedicated CUDA stream for async GPU ops.
      auto& tls_stream = ThreadPool::tls_stream();
      // Save current CUDA stream so we can restore it later.
      auto current_stream = at::cuda::getCurrentCUDAStream();
      at::cuda::setCurrentCUDAStream(tls_stream);

      bool success = false;
      try {
        // Use reference to avoid copy - dereference shared_ptr
        const auto& src = *shared_src_tensors;
        torch::Tensor cpu_tensor;
        // Stage 1: copy tensors from GPU to staging CPU tensor.
        success = TIME_EXPR("write phase 1: copy_gpu_tensors_to_cpu_tensor",
                            copy_gpu_tensors_to_cpu_tensor(src,
                                                           bids,
                                                           cpu_tensor,
                                                           tls_stream,
                                                           *connector_config),
                            "file: " + dst_file);

        cudaError_t err = cudaStreamSynchronize(tls_stream.stream());
        if (err != cudaSuccess) {
          std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                    << cudaGetErrorString(err) << std::endl;
        }

        if (!success) {
          std::cerr << "[ERROR] PUT failed during GPU->CPU staging: "
                    << dst_file << "\n";
        } else {
          // Stage 2: Write the cpu tensor to disk.
          success = TIME_EXPR("write phase 2: write_tensor_to_file",
                              write_tensor_to_file(cpu_tensor, dst_file),
                              ("file:" + dst_file +
                               " size:" + std::to_string(cpu_tensor.nbytes())));

          if (!success)
            std::cerr << "[ERROR] PUT failed during file write: " << dst_file
                      << "\n";
        }
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] PUT failed for " << dst_file << ": " << e.what()
                  << std::endl;
        success = false;
      } catch (...) {
        std::cerr << "[ERROR] PUT failed for " << dst_file << "\n";
        success = false;
      }

      // Restore original CUDA stream for safety.
      at::cuda::setCurrentCUDAStream(current_stream);

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

  std::lock_guard<std::mutex> lock(jobs_mutex);  // protect jobs map
  jobs[job_id] = std::move(job_state);

  return true;
}

// Async Storage -> GPU transfer (GET)
bool StorageOffloadEngine::transfer_async_get(
    int job_id,
    std::vector<std::string> src_files,
    std::vector<std::vector<int64_t>> all_block_ids,
    std::vector<torch::Tensor> dst_tensors) {
  // Create job state object to track progress and futures for this job.
  auto job_state = std::make_unique<JobState>();
  job_state->total_tasks = src_files.size();

  // Store shared_ptr to tensors to avoid repeated refcount changes
  auto shared_dst_tensors =
      std::make_shared<std::vector<torch::Tensor>>(std::move(dst_tensors));
  // For each source file, enqueue one async task in the I/O thread pool.
  for (size_t i = 0; i < src_files.size(); i++) {
    std::string src_file = src_files[i];
    auto block_ids = all_block_ids[i];
    auto future = thread_pool->enqueue([this,
                                        src_file,
                                        block_ids,
                                        shared_dst_tensors,
                                        job_state = job_state.get()]() -> bool {
      // Save current CUDA stream so we can restore it later.
      auto current_stream = at::cuda::getCurrentCUDAStream();
      auto& tls_stream = ThreadPool::tls_stream();
      at::cuda::setCurrentCUDAStream(tls_stream);

      bool success = false;
      try {
        // Stage 1: Read file to staging CPU tensor.
        torch::Tensor cpu_tensor;
        // Read data from disk into a tensor.
        success = TIME_EXPR("read phase 1: read_tensor_from_file",
                            read_tensor_from_file(src_file, cpu_tensor),
                            ("file:" + src_file));
        if (!success) {
          std::cerr << "[ERROR] Stage1 read_tensor_from_file failed for "
                    << src_file << std::endl;
        } else {
          // Stage 2:  copy tensors from staging CPU tensor to GPU.
          // Perform asynchronous GPU copy and tensor swap.
          const auto& dst = *shared_dst_tensors;
          success = TIME_EXPR("read phase 2: copy_cpu_tensor_to_gpu_tensors",
                              copy_cpu_tensor_to_gpu_tensors(cpu_tensor,
                                                             block_ids,
                                                             dst,
                                                             tls_stream,
                                                             *connector_config),
                              "file: " + src_file);
          cudaError_t err = cudaStreamSynchronize(tls_stream.stream());
          if (err != cudaSuccess) {
            std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                      << cudaGetErrorString(err) << std::endl;
            success = false;
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] GET failed for " << src_file << ": " << e.what()
                  << std::endl;
        success = false;
      } catch (...) {
        std::cerr << "[ERROR] GET unknown failure for " << src_file
                  << std::endl;
        success = false;
      }

      // Final cleanup & accounting
      // Synchronize only this thread's CUDA stream.
      at::cuda::setCurrentCUDAStream(current_stream);
      job_state->completed_tasks.fetch_add(1);
      if (!success) job_state->all_success = false;
      return success;
    });

    // Convert std::future -> std::shared_future - is copyable and can be
    // waited on by multiple threads.
    job_state->futures.push_back(future.share());
  }

  std::lock_guard<std::mutex> lock(jobs_mutex);
  jobs[job_id] = std::move(job_state);
  return true;
}

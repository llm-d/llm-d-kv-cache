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
#include "gds_file_io.hpp"
#include "tensor_copier.hpp"
#include "logger.hpp"
#include "storage_handler.hpp"

// Initialize IO threads, CUDA streams, and staging memory pool
StorageOffloadEngine::StorageOffloadEngine(int io_threads,
                                           int gpu_blocks_per_file,
                                           std::vector<torch::Tensor>& tensors,
                                           int read_preferring_workers,
                                           const std::string& gds_mode_str)
    : m_tensor_copier(tensors, gpu_blocks_per_file),
      m_gds_mode(parse_gds_mode(gds_mode_str)),
      m_thread_pool(
          io_threads,
          calc_staging_bytes(gpu_blocks_per_file, tensors, m_gds_mode),
          get_device_id(),
          read_preferring_workers),
      m_gpu_blocks_per_file(gpu_blocks_per_file) {
  init_handlers(m_gds_mode, tensors);
}

// Initialize read/write handlers based on GDS mode.
// Creates a GdsFileIO if GDS is requested and available; falls back to FileIO
// otherwise.
void StorageOffloadEngine::init_handlers(
    GdsMode gds_mode,
    const std::vector<torch::Tensor>& tensors) {
  std::shared_ptr<GdsFileIO> gds_io;

  if (gds_mode != GdsMode::DISABLED) {
    std::vector<std::pair<void*, size_t>> gpu_buffers;
    for (const auto& tensor : tensors)
      gpu_buffers.emplace_back(tensor.data_ptr(),
                               tensor.numel() * tensor.element_size());

    gds_io = std::make_shared<GdsFileIO>(gpu_buffers,
                                         m_tensor_copier.get_block_size(),
                                         gds_mode,
                                         m_tensor_copier);

    if (!gds_io->is_gds_available()) {
      FS_LOG_WARN(
          "StorageOffloadEngine: GDS initialization failed, "
          "falling back to CPU_BUFFER_STAGE for both READ and WRITE");
      gds_io = nullptr;
    }
  }

  m_read_handler = (gds_io && gds_io->use_for_read())
                       ? std::shared_ptr<StorageHandler>(gds_io)
                       : std::make_shared<FileIO>(m_tensor_copier);
  m_write_handler = (gds_io && gds_io->use_for_write())
                        ? std::shared_ptr<StorageHandler>(gds_io)
                        : std::make_shared<FileIO>(m_tensor_copier);

  FS_LOG_INFO(
      "StorageOffloadEngine: READ="
      << (m_read_handler->get_mode() == StorageMode::GDS_DIRECT ? "GDS" : "CPU")
      << " WRITE="
      << (m_write_handler->get_mode() == StorageMode::GDS_DIRECT ? "GDS"
                                                                 : "CPU"));
}

// Get current device (should be set by vLLM before calling this)
int StorageOffloadEngine::get_device_id() {
  int device_id = 0;
  cudaError_t err = cudaGetDevice(&device_id);
  if (err != cudaSuccess) {
    FS_LOG_ERROR("cudaGetDevice failed: " << cudaGetErrorString(err));
  }
  return device_id;
}
// Calculate staging buffer size in bytes; returns 0 for full-GDS modes.
size_t StorageOffloadEngine::calc_staging_bytes(
    int gpu_blocks_per_file,
    const std::vector<torch::Tensor>& tensors,
    GdsMode gds_mode) {
  if (gds_mode == GdsMode::READ_WRITE || gds_mode == GdsMode::BB_READ_WRITE) {
    return 0;
  }
  size_t block_size_in_bytes = 0;
  for (const auto& tensor : tensors) {
    block_size_in_bytes += static_cast<size_t>(tensor.stride(0)) *
                           static_cast<size_t>(tensor.element_size());
  }
  return block_size_in_bytes * static_cast<size_t>(gpu_blocks_per_file);
}

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
    if (job_state->completed_tasks.load() == job_state->total_tasks) {
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
// Store and Load operations
// -------------------------------
// ScopeGuard: executes stored lambda on destruction.
class ScopeGuard {
  std::function<void()> func_;

 public:
  explicit ScopeGuard(std::function<void()> f) : func_(std::move(f)) {}
  ~ScopeGuard() {
    if (func_) func_();
  }
};

// Async GPU -> Storage transfer
bool StorageOffloadEngine::async_store_gpu_blocks(
    int job_id,
    std::vector<std::string> dst_files,
    std::vector<std::vector<int64_t>> all_block_ids) {
  // Create job state object that will track progress and futures for this
  // job.
  auto job_state = std::make_shared<JobState>();
  job_state->total_tasks = dst_files.size();

  // Record an event on the default stream to ensure all prior work is done
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, at::cuda::getCurrentCUDAStream().stream());
  // Wrap default_done event in shared_ptr with custom deleter
  auto gpu_kvs_ready_event = std::shared_ptr<CUevent_st>(
      event,
      [](CUevent_st* evt) { cudaEventDestroy(evt); });

  // For each dst_file file, enqueue one async task in the I/O thread pool.
  for (size_t i = 0; i < dst_files.size(); i++) {
    std::string dst_file = dst_files[i];
    auto block_ids = all_block_ids[i];

    auto future = m_thread_pool.enqueue(
        [this, dst_file, block_ids, job_state, gpu_kvs_ready_event]() -> bool {
          // Check if dst_file file already exists - skip write if it does
          if (std::ifstream(dst_file).good()) {
            FileIO::update_atime(dst_file);
            job_state->completed_tasks.fetch_add(1);
            return true;  // File exists
          }

          // Wait for default stream to complete before starting the copy
          auto& tls_stream = ThreadPool::get_tls_stream();
          cudaStreamWaitEvent(tls_stream.stream(),
                              gpu_kvs_ready_event.get(),
                              0);
          bool success = false;
          // Execute the write operation using polymorphic storage handler
          try {
            size_t total_size =
                block_ids.size() * m_tensor_copier.get_block_size();
            success = TIME_EXPR_THROUGHPUT(
                "write: storage handler",
                m_write_handler->write_blocks_to_file(dst_file,
                                                      block_ids,
                                                      tls_stream.stream()),
                total_size,
                "file:",
                dst_file,
                " blocks:",
                block_ids.size());
            job_state->completed_tasks.fetch_add(1);
            if (!success) {
              FS_LOG_ERROR("Store failed during file write: " << dst_file);
            }
            return success;

          } catch (const std::exception& e) {
            FS_LOG_ERROR("Store failed for " << dst_file << ": " << e.what());
            job_state->completed_tasks.fetch_add(1);
            return false;
          } catch (...) {
            FS_LOG_ERROR("Store failed for " << dst_file
                                             << " (unknown exception)");
            job_state->completed_tasks.fetch_add(1);
            return false;
          }
        },
        TaskPriority::kNormal);
    // Convert std::future -> std::shared_future, which is copyable and can
    // be waited on by multiple threads.
    job_state->futures.push_back(future.share());
  }

  std::lock_guard<std::mutex> lock(m_jobs_mutex);  // protect jobs map
  m_jobs[job_id] = std::move(job_state);

  return true;
}

// Async Storage -> GPU transfer
bool StorageOffloadEngine::async_load_gpu_blocks(
    int job_id,
    std::vector<std::string> src_files,
    std::vector<std::vector<int64_t>> all_block_ids) {
  // Create job state object to track progress and futures for this job.
  auto job_state = std::make_shared<JobState>();
  job_state->total_tasks = src_files.size();

  // For each source file, enqueue one async task in the I/O thread pool.
  for (size_t i = 0; i < src_files.size(); i++) {
    std::string src_file = src_files[i];
    auto block_ids = all_block_ids[i];
    auto future = m_thread_pool.enqueue(
        [this, src_file, block_ids, job_state]() -> bool {
          auto& tls_stream = ThreadPool::get_tls_stream();
          bool success = false;

          ScopeGuard completion([&]() {
            job_state->completed_tasks.fetch_add(1);
            // if (!success) job_state->all_success = false; // TODO- silent
            // ignore read failures for now offloading connector not able to
            // handle failures
          });

          // Execute the read operation using polymorphic storage handler
          try {
            size_t total_size =
                block_ids.size() * m_tensor_copier.get_block_size();
            success = TIME_EXPR_THROUGHPUT(
                "read: storage handler",
                m_read_handler->read_blocks_from_file(src_file,
                                                      block_ids,
                                                      tls_stream.stream()),
                total_size,
                "file:",
                src_file,
                " blocks:",
                block_ids.size());
            if (!success) {
              FS_LOG_ERROR("Load failed for " << src_file);
            }
            return success;

          } catch (const std::exception& e) {
            FS_LOG_ERROR("Load failed for " << src_file << ": " << e.what());
            return false;
          } catch (...) {
            FS_LOG_ERROR("Load unknown failure for " << src_file);
            return false;
          }
        },
        TaskPriority::kHigh);

    // Convert std::future -> std::shared_future - is copyable and can be
    // waited on by multiple threads.
    job_state->futures.push_back(future.share());
  }

  std::lock_guard<std::mutex> lock(m_jobs_mutex);
  m_jobs[job_id] = std::move(job_state);
  return true;
}

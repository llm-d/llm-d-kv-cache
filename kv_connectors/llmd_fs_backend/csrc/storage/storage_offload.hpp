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

// storage_offload.hpp
#pragma once

#include <torch/extension.h>
#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "thread_pool.hpp"
#include "tensor_copier.hpp"
#include "storage_handler.hpp"
#include "storage_types.hpp"

// Tracks progress and results for a multi-file async PUT/GET job
struct JobState {
  // Futures for each async task in the job
  std::vector<std::shared_future<bool>> futures;
  // Number of tasks completed so far
  std::atomic<int> completed_tasks{0};
  // Total number of tasks scheduled for this job
  int total_tasks{0};
  // Flag indicating if all tasks succeeded
  std::atomic<bool> all_success{true};
  // Flag to signal cancellation (e.g. on preemption) — queued tasks bail early
  std::atomic<bool> cancelled{false};
};

// StorageOffloadEngine class manages asynchronous storage offload operations
class StorageOffloadEngine {
 private:
  // Mutex protecting access to the jobs map
  std::mutex m_jobs_mutex;
  // Global map of job_id to JobState, tracking async job progress
  std::map<int, std::shared_ptr<JobState>> m_jobs;
  // Handles GPU <-> CPU tensor copy operations
  TensorCopier m_tensor_copier;
  // GDS operation mode parsed from constructor argument (must precede
  // m_thread_pool)
  GdsMode m_gds_mode;
  // Thread pool for scheduling async PUT/GET tasks
  ThreadPool m_thread_pool;
  // Storage handler for read operations
  std::shared_ptr<StorageHandler> m_read_handler;
  // Storage handler for write operations
  std::shared_ptr<StorageHandler> m_write_handler;
  // GPU blocks per file (needed for operations)
  int m_gpu_blocks_per_file;
  // Dynamic write queue limit: Exponential Moving Average (EMA) of
  // per-file write duration (microseconds).
  std::atomic<uint64_t> m_avg_write_duration_us{0};
  // Max seconds of queued writes before dropping (0 = disabled)
  float m_max_write_queued_seconds;
  // Counter of dropped writes (for rate-limited logging)
  size_t m_dropped_writes{0};
  // Calculate staging buffer size in bytes.
  // Sized for the largest group so one buffer fits any group's transfer.
  // Uses canonical per-group block bytes from CanonicalKVCacheRef rather
  // than introspecting torch tensors.
  static size_t calc_staging_bytes(
      int gpu_blocks_per_file,
      const std::vector<int64_t>& per_group_block_bytes);
  // Initialize read/write handlers: GdsFileIO if available, FileIO otherwise
  void init_handlers(GdsMode gds_mode,
                     const std::vector<torch::Tensor>& tensors);
  // Get current device
  static int get_device_id();

 public:
  // Initialize IO threads, CUDA streams, and staging memory pool.
  // group_tensor_indices[i] = list of tensor indices into `tensors` used by
  // KV cache group i. For single-group (non-HMA) models, pass a single list
  // containing indices for all tensors.
  // per_group_block_bytes[i] = bytes per block for group i, summed across
  // that group's layers (sourced from CanonicalKVCacheRef.page_size_bytes
  // on the Python side).
  StorageOffloadEngine(int io_threads,
                       int gpu_blocks_per_file,
                       std::vector<torch::Tensor>& tensors,
                       std::vector<std::vector<int64_t>> group_tensor_indices,
                       std::vector<int64_t> per_group_block_bytes,
                       int read_preferring_workers,
                       const std::string& gds_mode,
                       float max_write_queued_seconds = 10.0);
  // Return finished jobs and their success status
  std::vector<std::pair<int, bool>> get_finished();
  // Update EMA of per-file write duration (called by write workers)
  void update_write_duration(uint64_t duration_us);
  // Compute dynamic write queue limit based on avg write duration
  size_t get_dynamic_write_queue_limit() const;
  // Wait for all tasks in the specified job to complete
  void wait_job(int job_id);
  // Async GPU -> Storage transfer (PUT)
  bool async_store_gpu_blocks(int job_id,
                              std::vector<int> group_indices,
                              std::vector<std::string> dst_files,
                              std::vector<std::vector<int64_t>> all_block_ids);
  // Async Storage -> GPU transfer (GET)
  bool async_load_gpu_blocks(int job_id,
                             std::vector<int> group_indices,
                             std::vector<std::string> src_files,
                             std::vector<std::vector<int64_t>> all_block_ids);
};

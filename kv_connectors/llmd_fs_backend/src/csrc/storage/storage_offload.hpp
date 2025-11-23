/*
 * Copyright 2025 The llm-d Authors.
 *
 * Licensed under the Apache License,
 * Version 2.0 (the "License");
 * you may not use this file except in
 * compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by
 * applicable law or agreed to in writing, software
 * distributed under the
 * License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the
 * specific language governing permissions and
 * limitations under the
 * License.
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
#include "cfg.hpp"

// Tracks progress and results for a multi-file async PUT/GET job
struct JobState {
  // Futures for each async task in the job
  std::vector<std::shared_future<bool>> futures;
  // Number of tasks completed so far
  std::atomic<int> completed_tasks{0};
  // Total number of tasks scheduled for this job
  std::atomic<int> total_tasks{0};
  // Flag indicating if all tasks succeeded
  std::atomic<bool> all_success{true};
};

// StorageOffloadEngine class manages asynchronous storage offload operations
class StorageOffloadEngine {
 private:
  // Mutex protecting access to the jobs map
  std::mutex jobs_mutex;
  // Global map of job_id to JobState, tracking async job progress
  std::map<int, std::unique_ptr<JobState>> jobs;

  // Thread pool for scheduling async PUT/GET tasks
  std::unique_ptr<ThreadPool> thread_pool;
  // connector config instance
  std::unique_ptr<ConnectorConfig> connector_config;

 public:
  // Initialize IO threads, CUDA streams, and staging memory pool
  StorageOffloadEngine(int io_threads,
                       size_t staging_buffer_size_mb,
                       size_t max_staging_memory_gb,
                       int tp_rank,
                       int gpu_blocks_per_file,
                       std::vector<torch::Tensor>& tensors,
                       bool kv_before_blocks,
                       bool layers_before_blocks,
                       int num_blocks_dimension);

  ~StorageOffloadEngine();
  // Return finished jobs and their success status
  std::vector<std::pair<int, bool>> get_finished();
  // Wait for all tasks in the specified job to complete
  void wait_job(int job_id);
  // Async GPU -> Storage transfer (PUT)
  bool transfer_async_put(int job_id,
                          std::vector<std::string> dst_files,
                          std::vector<torch::Tensor> src_tensors,
                          std::vector<std::vector<int64_t>> all_block_ids);
  // Async Storage -> GPU transfer (GET)
  bool transfer_async_get(int job_id,
                          std::vector<std::string> src_files,
                          std::vector<std::vector<int64_t>> all_block_ids,
                          std::vector<torch::Tensor> dst_tensors);
};

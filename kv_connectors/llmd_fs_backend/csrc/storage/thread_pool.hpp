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

#pragma once
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <sys/syscall.h>
#include <unistd.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "logger.hpp"
#include "storage_types.hpp"

// ThreadPool class is a thread pool used for parallel file offloading. Each
// worker thread handles one file end-to-end: reading or writing the file,
// staging data through its own thread-local staging buffer, and launching the
// GPU copy on a dedicated CUDA stream. This enables many files to be processed
// concurrently with full I/O–GPU overlap.
class ThreadPool {
 public:
  ThreadPool(size_t threads, size_t staging_buffer_bytes, int device_id);

  ~ThreadPool();

  template <class F>
  auto enqueue(F&& f) -> std::future<std::invoke_result_t<F>>;

  // Get thread-local storage(tls) CUDA stream
  static at::cuda::CUDAStream& get_tls_stream();

  // Return the thread-local staging buffer
  static StagingBufferInfo& get_staging_buffer();

 private:
  std::vector<std::thread> m_workers;         // All worker threads
  std::queue<std::function<void()>> m_tasks;  // Queue of pending tasks

  std::mutex m_queue_mutex;  // Protects access to the task queue
  std::condition_variable
      m_condition;  // Signals workers when tasks are available

  std::atomic<bool> m_stop{false};  // Tells workers to stop and exit
  int m_device_id;                  // CUDA device this thread pool is bound to
  // Thread-local CUDA stream bound to this worker thread
  static thread_local at::cuda::CUDAStream m_thread_stream;
  // Thread-local buffer used by each IO thread
  static thread_local StagingBufferInfo m_staging_buffer;

  // Allocate the thread-local staging buffer to at least required_bytes
  static bool allocate_staging_buffer(size_t required_bytes);
};

// enqueue: submit a task to the thread pool
template <class F>
auto ThreadPool::enqueue(F&& f) -> std::future<std::invoke_result_t<F>> {
  // Get the return type of the submitted task
  using return_type = std::invoke_result_t<F>;

  // Wrap the callable into a packaged_task so we can return a future
  auto task =
      std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));

  // Future for the caller to wait on
  std::future<return_type> res = task->get_future();

  {
    std::unique_lock<std::mutex> lock(m_queue_mutex);

    // Reject new tasks if the pool is shutting down
    if (m_stop) {
      FS_LOG_WARN("ThreadPool is stopping. Rejecting new task.");
      return std::future<return_type>();  // empty future
    }

    // Push the task wrapper into the queue
    m_tasks.emplace([task]() { (*task)(); });
  }

  // Wake one worker thread to process the task
  m_condition.notify_one();

  return res;
}

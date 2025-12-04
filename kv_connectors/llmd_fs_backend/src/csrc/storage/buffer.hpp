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

#include <cstddef>
#include <vector>

struct StagingBufferInfo {
    void* ptr = nullptr;
    size_t size = 0;
};

// Thread-local staging pointers defined in buffer.cpp
extern thread_local StagingBufferInfo t_staging_buffer;

// Global preallocated pool per IO thread
extern std::vector<StagingBufferInfo> g_staging_buffers;

// Preallocate staging buffers for IO threads
void preallocate_staging_buffers(size_t io_threads, size_t buffer_size_mb);

// Return thread-local staging buffer, allocating or reallocating if needed
StagingBufferInfo get_thread_local_staging_buffer(size_t required_bytes);

// Return NUMA node that a GPU belongs to
int get_gpu_numa_node(int device_id);

// Return list of CPU cores in the given NUMA node
std::vector<int> get_cpus_in_numa_node(int node);

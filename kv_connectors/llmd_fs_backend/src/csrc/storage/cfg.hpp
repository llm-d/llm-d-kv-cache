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

// cfg.h
#pragma once

#include <torch/extension.h>
#include <cstddef>

// cfg.hpp
#pragma once

#include <torch/extension.h>
#include <cstddef>
#include <cstdint>
#include <ostream>

// Layout mode enum - defines common layout configurations
enum class LayoutMode : uint8_t {
  DefaultKVFirst,     // kv_before_blocks=true,  layers_before_blocks=true
  DefaultBlockFirst,  // kv_before_blocks=false, layers_before_blocks=true
  CrossLayer  // layers_before_blocks=false (kv_before_blocks must be false)
};

inline std::ostream& operator<<(std::ostream& os, LayoutMode m) {
  switch (m) {
    case LayoutMode::DefaultKVFirst:
      return os << "DefaultKVFirst";
    case LayoutMode::DefaultBlockFirst:
      return os << "DefaultBlockFirst";
    case LayoutMode::CrossLayer:
      return os << "CrossLayer";
    default:
      return os << "Unknown";
  }
}

// Encapsulates KV cache layout and dimensions
class CacheLayout {
 public:
  // Number of blocks, from num_blocks_dimension
  int64_t num_blocks;
  // Data type of the tensors
  torch::Dtype dtype;
  // Bytes per tensor element
  size_t elem_size;
  // Total Bytes for each block
  size_t bytes_per_block;
  // Bytes per K or V plane use only when kv_before_blocks is true
  size_t kv_bytes_per_plane;
  // Which axis is the num_block index
  int num_blocks_dimension;
  // Reference tensor for layout info
  int num_layers;
  // Cache layout mode
  LayoutMode mode;

  CacheLayout(std::vector<torch::Tensor>& tensors,
              int num_blocks_dimension,
              bool kv_before_blocks,
              bool layers_before_blocks);
};

// ------------------------------------------------------------
// ConnectorConfig (pure config + derived state)
// ------------------------------------------------------------
class ConnectorConfig {
 public:
  // Static configuration
  int gpu_blocks_per_file;

  // KV-cache layout
  CacheLayout layout;
  // Staging buffer size in bytes
  int64_t staging_buffer_size_bytes;
  // Staging buffer size in number of tensor elements
  int64_t staging_buffer_total_elements;

  // Env flags
  // Verbosity Debug flag
  bool debug;
  // Use kernel-based copy for put operations
  bool use_kernel_copy_write;
  // Use kernel-based copy for get operations
  bool use_kernel_copy_read;

  ConnectorConfig(int gpu_blocks_per_file,
                  size_t staging_buffer_size_mb,
                  CacheLayout layout);

  // Helper for reading environment variable flags
  static bool get_env_flag(const char* name, bool default_value = false);
};

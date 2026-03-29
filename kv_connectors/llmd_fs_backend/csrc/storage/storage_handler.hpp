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

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

#include "storage_types.hpp"
#include "tensor_copier.hpp"

// Abstract base class for storage operations
class StorageHandler {
 public:
  virtual ~StorageHandler() = default;

  // Write blocks to file
  // Returns true on success, false on failure
  virtual bool write_blocks_to_file(const std::string& dst_file,
                                    const std::vector<int64_t>& block_ids,
                                    cudaStream_t stream) = 0;

  // Read blocks from file
  // Returns true on success, false on failure
  virtual bool read_blocks_from_file(const std::string& src_file,
                                     const std::vector<int64_t>& block_ids,
                                     cudaStream_t stream) = 0;

  // Get the storage mode type
  virtual StorageMode get_mode() const = 0;
};
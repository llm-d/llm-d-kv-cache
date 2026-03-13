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
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "storage_types.hpp"
#include "tensor_copier.hpp"
#include "storage_handler.hpp"

// CPU File I/O class - uses CPU buffer for staging
class FileIO : public StorageHandler {
 public:
  FileIO(TensorCopier& tensor_copier) : m_tensor_copier(tensor_copier) {}
  ~FileIO() override = default;

  // Write blocks to file using CPU staging
  bool write_blocks_to_file(const std::string& dst_file,
                            const std::vector<int64_t>& block_ids,
                            cudaStream_t stream) override;

  // Read blocks from file using CPU staging
  bool read_blocks_from_file(const std::string& src_file,
                             const std::vector<int64_t>& block_ids,
                             cudaStream_t stream) override;

  StorageMode get_mode() const override {
    return StorageMode::CPU_BUFFER_STAGE;
  }

  // Update only the atime of a file without changing mtime
  static void update_atime(const std::string& path);

 private:
  TensorCopier& m_tensor_copier;

  // Write a buffer to disk using a temporary file and atomic rename
  static bool write_buffer_to_file(const StagingBufferInfo& buf,
                                   const std::string& target_path);

  // Read a file into a thread-local staging buffer
  static bool read_buffer_from_file(const std::string& path,
                                    StagingBufferInfo& buf);
};

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
                            int group_idx,
                            cudaStream_t stream) override;

  // Read blocks from file using CPU staging
  bool read_blocks_from_file(const std::string& src_file,
                             const std::vector<int64_t>& block_ids,
                             int group_idx,
                             cudaStream_t stream) override;

  StorageMode get_mode() const override {
    return StorageMode::CPU_BUFFER_STAGE;
  }

  // Update only the atime of a file without changing mtime
  static void update_atime(const std::string& path);

 private:
  TensorCopier& m_tensor_copier;

  // Write the staging buffer to file (via temp file + atomic rename),
  // with partial-write support: persists only `write_size` bytes starting
  // at buf.ptr + write_offset — the back-of-buffer slice that holds the
  // transferred blocks, not the whole buffer.
  static bool write_buffer_to_file(const StagingBufferInfo& buf,
                                   const std::string& target_path,
                                   size_t write_offset,
                                   size_t write_size);

  // Read from file into the staging buffer, with partial-read support:
  // reads the last `blocks_in_file` blocks (each `bytes_per_block` bytes)
  // into `buf` at `buf_offset`, seeking to the file tail when the file
  // holds more. Mirrors write_buffer_to_file's back-offset convention.
  static bool read_buffer_from_file(const std::string& path,
                                    StagingBufferInfo& buf,
                                    size_t buf_offset,
                                    size_t bytes_per_block,
                                    size_t blocks_in_file);
};

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

#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "storage_types.hpp"
#include "tensor_copier.hpp"
#include "storage_handler.hpp"

#include <stdexcept>
#include <string>

// -------------------------------------------------------------------
// RAII guard for temporary file cleanup
// -------------------------------------------------------------------
// Ensures temporary files are automatically cleaned up on scope exit
// unless explicitly released after successful operations (e.g., rename)
class TmpFile {
  public:
  // Constructor for with UNIX open flags
  explicit TmpFile(const std::string &path, int oflags, int mode);

  ~TmpFile();

  // Prevent copying
  TmpFile(const TmpFile&) = delete;
  TmpFile & operator=(const TmpFile &) = delete;

  // Allow moving
  TmpFile(TmpFile&& other) noexcept
      : m_path(std::move(other.m_path)),
        m_fp(std::exchange(other.m_fp, nullptr)),
        m_o_tmpfile(other.m_o_tmpfile) {
  }

  TmpFile& operator=(TmpFile&& other) noexcept {
    if (this != &other) {
      cleanup();
      m_path = std::move(other.m_path);
      m_fp = std::exchange(other.m_fp, nullptr);
      m_o_tmpfile = std::move(other.m_o_tmpfile);
    }
    return *this;
  }

  // Allow if (!tmp_file) to see if file is open
  explicit operator bool() const;

  // Write buffer directly to the file desciptor.
  // If you have previous data using the C-style API, you should call flush() first.
  // The return value is the total number of bytes written.
  // if there is an error, returns -1 and sets errno
  ssize_t write_unbuffered(const void *data, size_t size);

  // Flush buffer to disk
  bool flush();

  // Get raw fd
  // NOTE: Call flush() before accessing fd to ensure buffered data is written
  int fd() const;

  // Get FILE* for stdio operations
  // NOTE: do not call close() on this file descriptor
  FILE* file_ptr() const;

  // Rename/link the temporary file to a new path
  // If m_o_tmpfile is true, uses linkat to link the O_TMPFILE to the new path
  // Otherwise, uses standard rename
  // Note: linkat does not close the file descriptor, it just creates a link
  bool rename(const std::string& new_path);


  private:
    void cleanup();
    
    std::filesystem::path m_path; // NOTE: if O_TEMPFILE is used; then this will be a directory instead
    FILE* m_fp;
    bool m_o_tmpfile;
};

// CPU File I/O class - uses CPU buffer for staging
class FileIO : public StorageHandler {
 public:
  FileIO(TensorCopier& tensor_copier, bool o_tmpfile)
      : m_tensor_copier(tensor_copier), m_o_tmpfile(o_tmpfile) {}
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
  bool m_o_tmpfile;

  // Write a buffer to disk using a temporary file and atomic rename
  bool write_buffer_to_file(const StagingBufferInfo& buf,
                                   const std::string& target_path);

  // Read a file into a thread-local staging buffer
  bool read_buffer_from_file(const std::string& path,
                                    StagingBufferInfo& buf);
};

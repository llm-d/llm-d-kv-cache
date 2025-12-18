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

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <random>

#include "file_io.hpp"
#include "thread_pool.hpp"

namespace fs = std::filesystem;

// -------------------------------------------------------------------
// Constants and thread-local buffers
// -------------------------------------------------------------------
// Define a larger buffer (1MB) to reduce syscall overhead and speed up I/O
const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024;  // 1MB buffer
// Allocate custom I/O buffer for this thread (replaces small default buffer)
thread_local std::vector<char> thread_write_buffer(WRITE_BUFFER_SIZE);
// Thread-local unique id for temporary files
thread_local uint64_t thread_unique_id = std::random_device{}();
// -------------------------------------------------------------------
// file-IO Functions
// -------------------------------------------------------------------
// Write a tensor to disk using a temporary file and atomic rename
bool write_tensor_to_file(const torch::Tensor& cpu_tensor,
                          const std::string& target_path) {
  TORCH_CHECK(cpu_tensor.is_cpu(), "Tensor must be on CPU");
  TORCH_CHECK(cpu_tensor.is_contiguous(), "Tensor must be contiguous");
  // Pointer and size of data to write
  const void* data_ptr = cpu_tensor.data_ptr();
  size_t nbytes = cpu_tensor.nbytes();

  // Create parent directory if needed
  fs::path file_path(target_path);
  fs::path parent_dir = file_path.parent_path();
  try {
    fs::create_directories(parent_dir);
  } catch (const fs::filesystem_error& e) {
    std::cerr << "[ERROR] Failed to create directories: " << e.what() << "\n";
    return false;
  }

  // Write to a temporary file to ensure atomic replace on rename
  // Include thread_unique_id so each thread uses a unique temporary file
  std::string tmp_path =
      target_path + std::to_string(thread_unique_id) + ".tmp.";

  std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
  if (!ofs) {
    std::cerr << "[ERROR] Failed to open temporary file for writing: "
              << tmp_path << " - " << std::strerror(errno) << "\n";
    return false;
  }

  // Apply the custom buffer to the file stream
  ofs.rdbuf()->pubsetbuf(thread_write_buffer.data(), WRITE_BUFFER_SIZE);

  // Write file contents
  ofs.write(reinterpret_cast<const char*>(data_ptr), nbytes);
  if (!ofs) {
    std::cerr << "[ERROR] Failed to write to temporary file: " << tmp_path
              << " - " << std::strerror(errno) << "\n";
    std::remove(tmp_path.c_str());  // Clean up temp file
    return false;
  }

  // Atomically rename temp file to final target name after a successful write
  if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
    std::cerr << "[ERROR] "
              << "Failed to rename " + tmp_path + " to " + target_path + " - " +
                     std::strerror(errno)
              << "\n";
    std::remove(tmp_path.c_str());
    return false;
  }

  return true;
}

// Read a file into a CPU tensor using the thread-local staging buffer
bool read_tensor_from_file(const std::string& path, torch::Tensor& cpu_tensor) {
  // Open file
  std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs) {
    std::cerr << "[ERROR] Failed to open file: " << path << "\n";
    return false;
  }

  // Determine file size
  ifs.seekg(0, std::ios::end);  // Move read pointer to END
  std::ifstream::pos_type end_pos = ifs.tellg();
  if (end_pos == std::streampos(-1)) {
    std::cerr << "[ERROR] Failed to determine file size: " << path << "\n";
    return false;
  }
  size_t file_size = static_cast<size_t>(end_pos);
  ifs.seekg(0, std::ios::beg);  // Move read pointer to start for reading

  // Acquire staging buffer of the required size
  StagingBufferInfo& buf = ThreadPool::tls_staging_buffer(file_size);
  if (!buf.ptr || buf.size < file_size) {
    std::cerr << "[ERROR] Staging buffer too small for file: " << path << "\n"
              << "[INFO] Required size: " << file_size
              << " bytes, Available size: " << buf.size << " bytes\n"
              << "ptr: " << buf.ptr << "\n";
    return false;
  }

  // Read file into Staging buffer
  ifs.read(reinterpret_cast<char*>(buf.ptr),
           static_cast<std::streamsize>(file_size));
  std::streamsize bytes_read = ifs.gcount();
  if (bytes_read != static_cast<std::streamsize>(file_size) || !ifs.good()) {
    std::cerr << "[ERROR] Failed to read full file: " << path << " (read "
              << bytes_read << "/" << file_size << " bytes)\n";
    return false;
  }

  // Wrap buffer into a Torch tensor (CPU, pinned)
  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)  // raw bytes
                     .device(torch::kCPU)
                     .pinned_memory(true);

  // Wrap staging buffer into a tensor without copying the data
  cpu_tensor = torch::from_blob(
      buf.ptr,
      {static_cast<long>(file_size)},
      [p = buf.ptr](void* /*unused*/) {},
      options);
  return true;
}

// update_atime update only the atime of a file without changing mtime
void update_atime(const std::string& path) {
  struct timespec times[2];
  times[1].tv_nsec = UTIME_NOW;   // update atime to now
  times[0].tv_nsec = UTIME_OMIT;  // keep mtime unchanged
  utimensat(AT_FDCWD, path.c_str(), times, 0);
}

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
#include <cuda_runtime.h>
#include <random>

#include "tensor_copier.hpp"
#include "file_io.hpp"
#include "thread_pool.hpp"
#include "logger.hpp"

namespace fs = std::filesystem;

// -------------------------------------------------------------------
// Constants and thread-local buffers
// -------------------------------------------------------------------
// Define a larger buffer (1MB) to reduce syscall overhead and speed up I/O
const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024;  // 1MB buffer

// Allocate custom I/O buffer for this thread (replaces small default buffer)
thread_local std::vector<char> thread_write_buffer(WRITE_BUFFER_SIZE);

// Thread-local unique suffix for temporary files
thread_local std::string tmp_file_suffix =
    "_" + std::to_string(std::random_device{}()) + ".tmp";
// -------------------------------------------------------------------
// file-IO Functions
// -------------------------------------------------------------------
// Partial-write of a back-of-buffer slice via temp file + atomic rename.
bool FileIO::write_buffer_to_file(const StagingBufferInfo& buf,
                                  const std::string& target_path,
                                  size_t write_offset,
                                  size_t write_size) {
  if (!buf.ptr || write_offset + write_size > buf.size) {
    FS_LOG_ERROR("write_buffer_to_file: bad range for "
                 << target_path << " (offset=" << write_offset
                 << " size=" << write_size << " buf.size=" << buf.size << ")");
    return false;
  }
  // Create parent directory if needed
  fs::path file_path(target_path);
  fs::path parent_dir = file_path.parent_path();
  try {
    fs::create_directories(parent_dir);
  } catch (const fs::filesystem_error& e) {
    FS_LOG_ERROR("Failed to create directories: " << e.what());
    return false;
  }

  // Write to a temporary file to ensure atomic replace on rename
  // Include tmp_file_suffix so each thread uses a unique temporary file
  std::string tmp_path = target_path + tmp_file_suffix;

  std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
  if (!ofs) {
    FS_LOG_ERROR("Failed to open temporary file for writing: "
                 << tmp_path << " - " << std::strerror(errno));
    return false;
  }

  // Apply the custom buffer to the file stream
  ofs.rdbuf()->pubsetbuf(thread_write_buffer.data(), WRITE_BUFFER_SIZE);

  // Write only the actual data region of the staging buffer.
  ofs.write(reinterpret_cast<const char*>(buf.ptr) + write_offset, write_size);
  if (!ofs) {
    FS_LOG_ERROR("Failed to write to temporary file: " << tmp_path << " - "
                                                       << std::strerror(errno));
    std::remove(tmp_path.c_str());  // Clean up temp file
    return false;
  }

  ofs.flush();
  if (!ofs) {
    FS_LOG_ERROR("Failed to flush data to temporary file: "
                 << tmp_path << " - " << std::strerror(errno));
    return false;
  }

  // Atomically rename temp file to final target name after a successful write
  if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
    FS_LOG_ERROR("Failed to rename " << tmp_path << " to " << target_path
                                     << " - " << std::strerror(errno));
    std::remove(tmp_path.c_str());
    return false;
  }

  return true;
}

// Partial-read into a back-of-buffer slice; seeks to file tail if needed.
bool FileIO::read_buffer_from_file(const std::string& path,
                                   StagingBufferInfo& buf,
                                   size_t buf_offset,
                                   size_t bytes_per_block,
                                   size_t blocks_in_file) {
  // Open file and grab its size in one pass (ios::ate).
  std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs) {
    FS_LOG_ERROR("Failed to open file: " << path);
    return false;
  }
  std::ifstream::pos_type end_pos = ifs.tellg();
  if (end_pos == std::streampos(-1)) {
    FS_LOG_ERROR("Failed to determine file size: " << path);
    return false;
  }
  size_t file_size = static_cast<size_t>(end_pos);

  // File must hold at least the blocks the caller is asking for.
  size_t read_size = blocks_in_file * bytes_per_block;
  if (file_size < read_size) {
    FS_LOG_ERROR("File too small: " << path << " (file_size=" << file_size
                                    << " required=" << read_size << ")");
    return false;
  }

  size_t file_offset = file_size - read_size;
  ifs.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);

  // Bounds check destination buffer.
  if (!buf.ptr || buf.size < buf_offset + read_size) {
    FS_LOG_ERROR("Staging buffer too small for file: "
                 << path << " (buf_offset=" << buf_offset
                 << " required=" << read_size << " available=" << buf.size
                 << " ptr=" << buf.ptr << ")");
    return false;
  }

  // Read file into Staging buffer
  ifs.read(reinterpret_cast<char*>(buf.ptr) + buf_offset,
           static_cast<std::streamsize>(read_size));
  std::streamsize bytes_read = ifs.gcount();
  if (bytes_read != static_cast<std::streamsize>(read_size) || !ifs.good()) {
    FS_LOG_ERROR("Failed to read file: "
                 << path << " (read " << bytes_read << "/" << read_size
                 << " bytes from offset " << file_offset << ")");
    return false;
  }

  return true;
}

// update_atime update only the atime of a file without changing mtime
void FileIO::update_atime(const std::string& path) {
  struct timespec times[2];
  times[0].tv_nsec = UTIME_NOW;   // atime → now
  times[1].tv_nsec = UTIME_OMIT;  // mtime → unchanged
  utimensat(AT_FDCWD, path.c_str(), times, 0);
}

// Write via CPU staging - wraps copy_blocks + write_buffer_to_file
bool FileIO::write_blocks_to_file(const std::string& dst_file,
                                  const std::vector<int64_t>& block_ids,
                                  int group_idx,
                                  cudaStream_t stream) {
  // Get thread-local staging buffer
  StagingBufferInfo& buf = ThreadPool::get_staging_buffer();
  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_store = true;

  // Stage 1: copy tensors from GPU to staging CPU tensor
  TIME_EXPR(
      "write phase 1: copy_blocks ",
      m_tensor_copier.copy_blocks(cpu_base, block_ids, group_idx, is_store),
      "file: ",
      dst_file);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    FS_LOG_ERROR("write_blocks_to_file: cudaStreamSynchronize failed: "
                 << cudaGetErrorString(err));
    return false;
  }

  // Stage 2: Persist only the back-of-buffer slice that copy_blocks
  // populated. The staging buffer is sized worst-case (so usually larger
  // than this group's payload), and writing only the slice keeps the
  // on-disk file sized to the actual blocks — same back-offset convention
  // as the read path.
  size_t bytes_per_block = m_tensor_copier.bytes_per_block_for_group(group_idx);
  size_t blocks_in_file = block_ids.size();
  size_t write_offset =
      (m_tensor_copier.gpu_blocks_per_file() - blocks_in_file) *
      bytes_per_block;
  size_t write_size = blocks_in_file * bytes_per_block;
  bool success =
      TIME_EXPR("write phase 2: write_buffer_to_file",
                write_buffer_to_file(buf, dst_file, write_offset, write_size),
                "file:",
                dst_file,
                " size:",
                write_size);

  if (!success) {
    FS_LOG_ERROR(
        "write_blocks_to_file: Store failed during file write: " << dst_file);
  }

  return success;
}

// Read via CPU staging - wraps read_buffer_from_file + copy_blocks
bool FileIO::read_blocks_from_file(const std::string& src_file,
                                   const std::vector<int64_t>& block_ids,
                                   int group_idx,
                                   cudaStream_t stream) {
  // Get thread-local staging buffer
  StagingBufferInfo& buf = ThreadPool::get_staging_buffer();

  // Stage 1: Read the requested suffix into the back-of-buffer slot
  // copy_blocks() reads from. The staging buffer is sized worst-case
  // (so usually larger than this group's payload), and the read seeks
  // to the file tail — same back-offset convention as the write path.
  size_t bytes_per_block = m_tensor_copier.bytes_per_block_for_group(group_idx);
  size_t blocks_in_file = block_ids.size();
  size_t buf_offset = (m_tensor_copier.gpu_blocks_per_file() - blocks_in_file) *
                      bytes_per_block;
  bool success = TIME_EXPR("read phase 1: read_buffer_from_file",
                           read_buffer_from_file(src_file,
                                                 buf,
                                                 buf_offset,
                                                 bytes_per_block,
                                                 blocks_in_file),
                           "file:",
                           src_file);
  if (!success) {
    FS_LOG_ERROR("read_blocks_from_file: read_buffer_from_file failed for "
                 << src_file);
    return false;
  }

  // Stage 2: copy tensors from staging CPU tensor to GPU
  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_store = false;

  success = TIME_EXPR(
      "read phase 2: copy_cpu_tensor_to_gpu_tensors",
      m_tensor_copier.copy_blocks(cpu_base, block_ids, group_idx, is_store),
      "file: ",
      src_file);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    FS_LOG_ERROR("read_blocks_from_file: cudaStreamSynchronize failed: "
                 << cudaGetErrorString(err));
    return false;
  }

  return success;
}

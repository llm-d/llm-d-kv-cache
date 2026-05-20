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
// TmpFile Implementation
// -------------------------------------------------------------------
TmpFile::TmpFile(const std::string &path, int oflags, int mode) :
  m_path(path),
  m_fp(nullptr),
  m_o_tmpfile(oflags & O_TMPFILE) {
    auto o_path = m_path;
    if (m_o_tmpfile) {
      o_path = m_path.parent_path();
    }

    int fd = open(o_path.c_str(), oflags, mode);
    if (fd >= 0) {
      // Create FILE* wrapper for buffered I/O
      const char* mode = (oflags & O_ACCMODE) == O_RDONLY ? "rb" : "wb";
      m_fp = fdopen(fd, mode);

      if (!m_fp) {
        int saved_errno = errno;  // Save errno before close() potentially overwrites it
        close(fd);
        errno = saved_errno;
      }
    }
  }

TmpFile::operator bool() const {
  return m_fp != nullptr;
}

ssize_t TmpFile::write_unbuffered(const void* data, size_t size) {
  if (!m_fp) return -1;
  int fd = this->fd();
  const char *p = reinterpret_cast<const char*>(data);
  size_t total = 0;
  constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024; // 4MiB

  while (total < size) {
    size_t remaining = size - total;
    size_t to_write = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
    ssize_t n = ::write(fd, p + total, to_write);
    if (n > 0 ) {
      total += reinterpret_cast<ssize_t>(n);
      continue;
    }
    if (n == -1 && errno == EINTR) {
      // Special failure where the syscall can be interrupted
      continue;
    }
    // Some other errno that needs to be checked by the caller
    return -1;
  }
  return total;
}

bool TmpFile::flush() {
  return m_fp && fflush(m_fp) == 0;
}

int TmpFile::fd() const {
  return m_fp ? fileno(m_fp) : -1;
}

FILE* TmpFile::file_ptr() const {
  return m_fp;
}


bool TmpFile::rename(const std::string& new_path) {
  if (!m_fp) return false;

  // Flush any buffered data before rename/link
  if (!flush()) return false;

  if (m_o_tmpfile) {
    // For O_TMPFILE, use linkat with AT_EMPTY_PATH to link the fd to the filesystem
    int file_fd = fd();
    if (linkat(file_fd, "", AT_FDCWD, new_path.c_str(), AT_EMPTY_PATH) != 0) {
      return false;
    }
  } else {
    // Standard rename for regular temporary files
    if (std::rename(m_path.c_str(), new_path.c_str()) != 0) {
      return false;
    }
  }

  return true;
}

TmpFile::~TmpFile() {
  cleanup();
}

void TmpFile::cleanup() {
  if (m_fp) {
    fclose(m_fp);
    m_fp = nullptr;
  }

  if (!m_o_tmpfile && !m_path.empty()) {
    std::remove(m_path.c_str());
  }
  m_path.clear();
}

// -------------------------------------------------------------------
// file-IO Functions
// -------------------------------------------------------------------
// Write a buffer to disk using a temporary file and atomic rename
bool FileIO::write_buffer_to_file(const StagingBufferInfo& buf,
                                  const std::string& target_path) {
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
  int open_flags = O_RDWR | O_CREAT;
  if (m_o_tmpfile) {
    open_flags = O_RDWR | O_TMPFILE;
  }
  TmpFile tmp_file(tmp_path, open_flags, 0664);
  FILE *fp = tmp_file.file_ptr();
  if (!fp) {
    FS_LOG_ERROR("Failed to open temporary file for writing: "
                 << tmp_path << " - " << std::strerror(errno));
    return false;
  }

  // Write data using unbuffered I/O for better performance
  ssize_t amount_writen = tmp_file.write_unbuffered(buf.ptr, buf.size);
  if (amount_writen == -1) {
    FS_LOG_ERROR("Failed to write data to temporary file:" << tmp_path << " - "
                                                           << std::strerror(errno));
    return false;

  }
  else if (amount_writen < buf.size) {
    FS_LOG_ERROR("Failed to write data to temporary file:" << tmp_path
                                                           << "Wanted " << buf.size << "byte(s) but found "
                                                           << amount_writen);
  }


  // Atomically rename temp file to final target name after a successful write
  if (! tmp_file.rename(target_path)) {
    FS_LOG_ERROR("Failed to rename " << tmp_path << " to " << target_path
                                     << " - " << std::strerror(errno));
    return false;
  }

  return true;
}

// Read a file into a thread-local staging buffer
bool FileIO::read_buffer_from_file(const std::string& path,
                                   StagingBufferInfo& buf) {
  // Open file
  std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs) {
    FS_LOG_ERROR("Failed to open file: " << path);
    return false;
  }

  // Determine file size
  std::ifstream::pos_type end_pos = ifs.tellg();
  if (end_pos == std::streampos(-1)) {
    FS_LOG_ERROR("Failed to determine file size: " << path);
    return false;
  }
  size_t file_size = static_cast<size_t>(end_pos);
  ifs.seekg(0, std::ios::beg);  // Move read pointer to start for reading

  // Acquire staging buffer of the required size
  if (!buf.ptr || buf.size < file_size) {
    FS_LOG_ERROR("Staging buffer too small for file: "
                 << path << " (required=" << file_size
                 << " available=" << buf.size << " ptr=" << buf.ptr << ")");
    return false;
  }

  // Read file into Staging buffer
  ifs.read(reinterpret_cast<char*>(buf.ptr),
           static_cast<std::streamsize>(file_size));
  std::streamsize bytes_read = ifs.gcount();
  if (bytes_read != static_cast<std::streamsize>(file_size) || !ifs.good()) {
    FS_LOG_ERROR("Failed to read full file: " << path << " (read " << bytes_read
                                              << "/" << file_size << " bytes)");
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
                                  cudaStream_t stream) {
  // Get thread-local staging buffer
  StagingBufferInfo& buf = ThreadPool::get_staging_buffer();
  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_store = true;

  // Stage 1: copy tensors from GPU to staging CPU tensor
  TIME_EXPR("write phase 1: copy_blocks ",
            m_tensor_copier.copy_blocks(cpu_base, block_ids, is_store),
            "file: ",
            dst_file);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    FS_LOG_ERROR("write_blocks_to_file: cudaStreamSynchronize failed: "
                 << cudaGetErrorString(err));
    return false;
  }

  // Stage 2: Write the cpu tensor to disk
  bool success = TIME_EXPR("write phase 2: write_buffer_to_file",
                           write_buffer_to_file(buf, dst_file),
                           "file:",
                           dst_file,
                           " size:",
                           buf.size);

  if (!success) {
    FS_LOG_ERROR(
        "write_blocks_to_file: Store failed during file write: " << dst_file);
  }

  return success;
}

// Read via CPU staging - wraps read_buffer_from_file + copy_blocks
bool FileIO::read_blocks_from_file(const std::string& src_file,
                                   const std::vector<int64_t>& block_ids,
                                   cudaStream_t stream) {
  // Get thread-local staging buffer
  StagingBufferInfo& buf = ThreadPool::get_staging_buffer();

  // Stage 1: Read file to staging CPU tensor
  bool success = TIME_EXPR("read phase 1: read_buffer_from_file",
                           read_buffer_from_file(src_file, buf),
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

  success =
      TIME_EXPR("read phase 2: copy_cpu_tensor_to_gpu_tensors",
                m_tensor_copier.copy_blocks(cpu_base, block_ids, is_store),
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

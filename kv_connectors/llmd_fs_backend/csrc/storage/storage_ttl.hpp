// Copyright 2026 The llm-d Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cerrno>
#include <array>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

// Versioned namespace policy, shared with storage_ttl.py. No deadline renewal
// on reads or duplicate stores. Stable stripe locks serialize writers and
// deleters.
class StorageTTLLock {
 public:
  explicit StorageTTLLock(const std::string& path) {
    constexpr const char* prefix = ".llmd-ttl-v1-";
    if (path.find("/.llmd-ttl-v1-") == std::string::npos) return;
    std::filesystem::path root;
    for (const auto& component : std::filesystem::path(path)) {
      root /= component;
      const auto name = component.string();
      if (name.rfind(prefix, 0) != 0) continue;
      const auto suffix = name.substr(std::char_traits<char>::length(prefix));
      if (suffix.empty() ||
          suffix.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid TTL namespace");
      seconds_ = std::stoll(suffix);
      if (seconds_ < 1 || seconds_ > 2147483647LL)
        throw std::runtime_error("Invalid TTL namespace duration");
      const auto bucket =
          std::filesystem::path(path).stem().string().substr(0, 3);
      if (bucket.size() != 3 ||
          bucket.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("Invalid TTL block filename");
      std::filesystem::create_directories(root / ".locks");
      const auto lock = root / ".locks" / (bucket + ".lock");
      // POSIX record locks are process-owned. A process-local mutex also
      // serializes threads and prevents one FD close releasing a sibling lock.
      local_lock_ = std::unique_lock<std::mutex>(
          stripes()[std::stoul(bucket, nullptr, 16)]);
      fd_ = open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
      if (fd_ < 0)
        throw std::system_error(errno,
                                std::generic_category(),
                                "TTL lock open");
      struct flock request = {};
      request.l_type = F_WRLCK;
      request.l_whence = SEEK_SET;
      request.l_start = 0;
      request.l_len = 0;
      int result;
      do {
        result = fcntl(fd_, F_SETLKW, &request);
      } while (result < 0 && errno == EINTR);
      if (result < 0) {
        const int error = errno;
        close(fd_);
        fd_ = -1;
        throw std::system_error(error,
                                std::generic_category(),
                                "TTL publication lock");
      }
      return;
    }
  }
  StorageTTLLock(const StorageTTLLock&) = delete;
  StorageTTLLock& operator=(const StorageTTLLock&) = delete;
  ~StorageTTLLock() {
    if (fd_ >= 0) close(fd_);
  }
  bool expired(const std::string& path) const {
    if (seconds_ == 0) return false;
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return true;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    const auto modified =
        static_cast<int64_t>(info.st_mtim.tv_sec) * 1000000000LL +
        info.st_mtim.tv_nsec;
    return now - modified >= seconds_ * 1000000000LL;
  }

 private:
  static std::array<std::mutex, 4096>& stripes() {
    static std::array<std::mutex, 4096> locks;
    return locks;
  }
  std::unique_lock<std::mutex> local_lock_;
  int fd_ = -1;
  int64_t seconds_ = 0;
};

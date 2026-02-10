#pragma once

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

enum class LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4 };

class FSLogger {
 public:
  static LogLevel level() {
    static LogLevel lvl = init_level();
    return lvl;
  }

  static void log(LogLevel lvl, const std::string& msg) {
    if (lvl < level()) return;

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count() << " "
        << level_str(lvl) << " "
        << "[thread:" << std::this_thread::get_id() << "] " << msg;

    // Thread safe write to stderr
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cerr << oss.str() << std::endl;
  }

 private:
  static const char* level_str(LogLevel lvl) {
    switch (lvl) {
      case LogLevel::TRACE:
        return "[TRACE]";
      case LogLevel::DEBUG:
        return "[DEBUG]";
      case LogLevel::INFO:
        return "[INFO]";
      case LogLevel::WARN:
        return "[WARN]";
      case LogLevel::ERROR:
        return "[ERROR]";
      default:
        return "[INFO]";  // XXX: Default should be INFO?
    }
  }

  static LogLevel init_level() {
    const char* level_env = std::getenv("STORAGE_LOG_LEVEL");
    if (level_env) {
      std::string v(level_env);
      if (v == "trace" || v == "TRACE") return LogLevel::TRACE;
      if (v == "debug" || v == "DEBUG") return LogLevel::DEBUG;
      if (v == "info" || v == "INFO") return LogLevel::INFO;
      if (v == "warn" || v == "WARN") return LogLevel::WARN;
      if (v == "error" || v == "EROR") return LogLevel::ERROR;
    }

    // Backward compatibility: STORAGE_CONNECTOR_DEBUG=1 -> DEBUG level
    const char* debug_env = std::getenv("STORAGE_CONNECTOR_DEBUG");
    if (debug_env) {
      std::string v(debug_env);
      if (v == "1" || v == "true" || v == "TRUE") return LogLevel::DEBUG;
    }

    return LogLevel::INFO;  // Default
  }
};

// Convenience macros
#define FS_LOG(lvl, msg)               \
  do {                                 \
    if (lvl >= FSLogger::level()) {    \
      std::ostringstream __oss;        \
      __oss << msg;                    \
      FSLogger::log(lvl, __oss.str()); \
    }                                  \
  } while (0)

#define FS_LOG_ERROR(msg) FS_LOG(LogLevel::ERROR, msg)
#define FS_LOG_WARN(msg) FS_LOG(LogLevel::WARN, msg)
#define FS_LOG_INFO(msg) FS_LOG(LogLevel::INFO, msg)
#define FS_LOG_DEBUG(msg) FS_LOG(LogLevel::DEBUG, msg)
#define FS_LOG_TRACE(msg) FS_LOG(LogLevel::TRACE, msg)

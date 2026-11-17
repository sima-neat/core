#pragma once

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <sys/resource.h>
#include <system_error>

namespace sima_perf {

// Process-wide CPU includes all application/Runtime threads, not board-wide
// IRQ/kworker time, other processes or accelerator utilization. CPU/frame is
// work per result; busy cores is CPU/wall time and may legitimately exceed one.
struct ProcessCpuMetrics {
  std::uint64_t frames = 0;
  double wall_seconds = 0.0;
  double user_seconds = 0.0;
  double system_seconds = 0.0;

  double fps() const {
    return wall_seconds > 0.0 ? static_cast<double>(frames) / wall_seconds : 0.0;
  }
  double user_us_per_frame() const {
    return frames > 0 ? user_seconds * 1e6 / static_cast<double>(frames) : 0.0;
  }
  double system_us_per_frame() const {
    return frames > 0 ? system_seconds * 1e6 / static_cast<double>(frames) : 0.0;
  }
  double cpu_us_per_frame() const {
    return user_us_per_frame() + system_us_per_frame();
  }
  double average_busy_cores() const {
    return wall_seconds > 0.0 ? (user_seconds + system_seconds) / wall_seconds : 0.0;
  }
};

// Two snapshots per measured run, never per frame. Start immediately before
// the first measured submission and finish after the final output loan release,
// before reporting/teardown. Warmup and fixture allocation belong outside.
// CPU and wall samples are adjacent syscalls, not an atomic hardware snapshot.
class ProcessCpuWindow {
public:
  ProcessCpuWindow() : begin_(snapshot()) {}

  ProcessCpuMetrics finish(std::uint64_t completed_frames) const {
    const Snapshot end = snapshot();
    return {completed_frames, std::chrono::duration<double>(end.wall - begin_.wall).count(),
            seconds(end.usage.ru_utime) - seconds(begin_.usage.ru_utime),
            seconds(end.usage.ru_stime) - seconds(begin_.usage.ru_stime)};
  }

private:
  struct Snapshot {
    rusage usage{};
    std::chrono::steady_clock::time_point wall;
  };

  static Snapshot snapshot() {
    Snapshot result;
    if (::getrusage(RUSAGE_SELF, &result.usage) != 0) {
      throw std::system_error(errno, std::generic_category(), "measure process CPU");
    }
    result.wall = std::chrono::steady_clock::now();
    return result;
  }

  static double seconds(const timeval& value) {
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) * 1e-6;
  }

  Snapshot begin_;
};

} // namespace sima_perf

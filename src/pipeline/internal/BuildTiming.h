#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/EnvUtil.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>

namespace simaai::neat::pipeline_internal {

using BuildTimingClock = std::chrono::steady_clock;
using BuildTimingPoint = BuildTimingClock::time_point;

inline bool build_timing_enabled() {
  return env_bool("SIMA_GRAPH_BUILD_TIMING", false) || env_bool("SIMA_NEAT_BUILD_TIMING", false);
}

inline BuildTimingPoint build_timing_now() {
  return BuildTimingClock::now();
}

inline std::int64_t build_timing_us(BuildTimingPoint start,
                                    BuildTimingPoint end = build_timing_now()) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

inline void emit_build_timing(const char* scope,
                              std::initializer_list<std::pair<const char*, std::int64_t>> fields,
                              const std::string& detail = {}) {
  if (!build_timing_enabled()) {
    return;
  }
  std::fprintf(stderr, "[GRAPH-BUILD-TIMING] %s", scope ? scope : "<unknown>");
  if (!detail.empty()) {
    std::fprintf(stderr, " %s", detail.c_str());
  }
  for (const auto& field : fields) {
    const char* name = field.first ? field.first : "elapsed";
    std::fprintf(stderr, " %s=%.3fms", name, static_cast<double>(field.second) / 1000.0);
  }
  std::fprintf(stderr, "\n");
}

class ScopedBuildTiming {
public:
  ScopedBuildTiming(const char* scope, std::string detail = {})
      : scope_(scope), detail_(std::move(detail)), start_(build_timing_now()),
        enabled_(build_timing_enabled()) {}

  ScopedBuildTiming(const ScopedBuildTiming&) = delete;
  ScopedBuildTiming& operator=(const ScopedBuildTiming&) = delete;

  ~ScopedBuildTiming() {
    if (!enabled_) {
      return;
    }
    emit_build_timing(scope_, {{"total", build_timing_us(start_)}}, detail_);
  }

private:
  const char* scope_ = nullptr;
  std::string detail_;
  BuildTimingPoint start_{};
  bool enabled_ = false;
};

} // namespace simaai::neat::pipeline_internal

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace simaai::neat::pipeline_internal {

// rtspsrc waits for its RTSP TEARDOWN command during PAUSED -> READY. A fused
// live pipeline can contain dozens of rtspsrc instances, so the framework's
// synchronous NULL-state budget must include their aggregate configured wait.
// Keep the calculation independent of GStreamer objects so its overflow and
// cap behavior can be covered by a small unit test.
inline int synchronous_live_teardown_budget_ms(int base_timeout_ms,
                                               std::uint64_t rtsp_timeout_total_ns,
                                               std::size_t rtsp_source_count) {
  constexpr int kSchedulingMarginMs = 250;
  constexpr int kMaximumBudgetMs = 30'000;
  constexpr std::uint64_t kNanosecondsPerMillisecond = 1'000'000ULL;

  const std::uint64_t bounded_base_ms = static_cast<std::uint64_t>(
      std::clamp(base_timeout_ms, 0, kMaximumBudgetMs));
  const std::uint64_t rtsp_timeout_ms =
      rtsp_timeout_total_ns / kNanosecondsPerMillisecond +
      (rtsp_timeout_total_ns % kNanosecondsPerMillisecond != 0 ? 1ULL : 0ULL);
  const std::uint64_t scheduling_margin_ms =
      rtsp_source_count == 0 ? 0ULL : static_cast<std::uint64_t>(kSchedulingMarginMs);

  const std::uint64_t requested_ms =
      bounded_base_ms + rtsp_timeout_ms + scheduling_margin_ms;
  return static_cast<int>(
      std::min<std::uint64_t>(requested_ms, static_cast<std::uint64_t>(kMaximumBudgetMs)));
}

} // namespace simaai::neat::pipeline_internal

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstdint>

namespace simaai::neat::runtime {

// Run's deprecated forms (Run.h, [[deprecated]]): one runtime warning per form and process, on the
// application's first call of it. Core's own calls go through run_internal:: and never warn.
enum class DeprecatedRunApi : std::uint8_t {
  PushMats,          // push(const std::vector<cv::Mat>&)
  PushMatsNamed,     // push(std::string_view, const std::vector<cv::Mat>&)
  TryPushMats,       // try_push(const std::vector<cv::Mat>&)
  TryPushMatsNamed,  // try_push(std::string_view, const std::vector<cv::Mat>&)
  PullOptional,      // pull(int) -> std::optional<Sample>
  PullOptionalNamed, // pull(std::string_view, int) -> std::optional<Sample>
  PullTensors,       // pull_tensors(int)
  PullTensorsNamed,  // pull_tensors(std::string_view, int)
  PullSamples,       // pull_samples(int)
  PullSamplesNamed,  // pull_samples(std::string_view, int)
  Count,
};

// Writes "[WARN] Run::<form> is deprecated ...; use <replacement>." to stderr the first time it is
// called for `api` in this process (any thread), and nothing afterwards. Never throws.
void warn_deprecated_once(DeprecatedRunApi api) noexcept;

} // namespace simaai::neat::runtime

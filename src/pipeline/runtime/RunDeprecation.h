#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstdint>
#include <string_view>

namespace simaai::neat::runtime {

// Run's deprecated forms (Run.h, [[deprecated]]). On the GStreamer backend a form works, with one
// runtime warning per form and process on the application's first call of it; on the Dataflow
// backend the call fails before anything is pushed or pulled. Core's own calls go through
// run_internal:: and do neither.
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

// The application's call of `api`. On the Dataflow backend (the process's: a Run is on the backend
// the process runs on) it throws deprecated_on_dataflow's error for the form. Otherwise it writes
// "[WARN] Run::<form> is deprecated ...; use <replacement>. It does not work on the Dataflow
// backend." to stderr the first time it is called for `api` in this process (any thread), and
// nothing afterwards.
void deprecated_call(DeprecatedRunApi api);

// The NeatError (misconfig.pipeline_shape) of a deprecated form called on the Dataflow backend:
// "<form> is deprecated and is not carried by the Dataflow backend: use <replacement>." and, as
// its hint, that the form works on the GStreamer backend and the call that selects it. The Python
// bindings throw it with the Python names.
[[noreturn]] void throw_deprecated_on_dataflow(std::string_view form, std::string_view replacement);

// The sentence a deprecation warning ends with.
constexpr const char* kDeprecatedNotOnDataflow = "It does not work on the Dataflow backend.";

} // namespace simaai::neat::runtime

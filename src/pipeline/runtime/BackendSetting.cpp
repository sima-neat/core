// The process's backend (neat/runtime.h; pipeline/internal/BackendSetting.h).

#include "pipeline/internal/BackendSetting.h"

#include "gst/GstInit.h"
#include "neat/runtime.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/UxLogging.h"

#if defined(SIMANEAT_WITH_DATAFLOW)
#include "pipeline/runtime/DataflowDeviceMemory.h"
#endif

#include <atomic>
#include <cstdint>
#include <future>
#include <string>

namespace simaai::neat {

namespace {

// The setting and whether it is fixed, in one word, so that reading it for a build and fixing it
// are one step, and a set_backend() on another thread either comes before that step or is
// refused by it.
constexpr std::uint8_t kDataflow = 1U; ///< the value: set = Dataflow, clear = GStreamer
constexpr std::uint8_t kFixed = 2U;    ///< a Model was constructed or a Graph built

// Constant-initialised: a set_backend() or a Model in a static initialiser finds it ready.
constinit std::atomic<std::uint8_t> g_backend{0U};

constexpr RunBackend backend_of(std::uint8_t state) noexcept {
  return (state & kDataflow) != 0U ? RunBackend::Dataflow : RunBackend::GStreamer;
}

constexpr const char* name_of(RunBackend backend) noexcept {
  return backend == RunBackend::Dataflow ? "Dataflow" : "GStreamer";
}

} // namespace

void set_backend(RunBackend backend) {
  if (backend != RunBackend::Dataflow && backend != RunBackend::GStreamer) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kOptionOutOfRange,
        "simaai::neat::set_backend: not a backend (RunBackend::Dataflow, RunBackend::GStreamer)");
  }
  if (backend == RunBackend::Dataflow && !pipeline_internal::has_dataflow_backend()) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kOptionOutOfRange,
        "simaai::neat::set_backend(RunBackend::Dataflow): this libsima_neat was built without "
        "the Dataflow backend",
        {}, "This library runs every Model and Graph on the GStreamer backend.");
  }
  const std::uint8_t wanted = backend == RunBackend::Dataflow ? kDataflow : std::uint8_t{0U};
  std::uint8_t state = g_backend.load(std::memory_order_acquire);
  for (;;) {
    if ((state & kFixed) != 0U) {
      if (backend_of(state) == backend) {
        return;
      }
      pipeline_internal::error_util::throw_session_error(
          error_codes::kOptionOutOfRange,
          std::string("simaai::neat::set_backend(RunBackend::") + name_of(backend) +
              "): a process runs on one backend, and this one runs on the " +
              name_of(backend_of(state)) +
              " backend since its first Model was constructed or its first Graph built",
          {},
          "Call simaai::neat::set_backend (Python: pyneat.set_backend) before the program "
          "constructs its first Model or builds its first Graph. A Model or a Run that was "
          "destroyed, or whose build failed, fixed the backend all the same.");
    }
    if (g_backend.compare_exchange_weak(state, wanted, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return;
    }
  }
}

RunBackend backend() {
  return backend_of(g_backend.load(std::memory_order_acquire));
}

// The process's backend's one-time setup, which its first build would do otherwise; the setting is
// read, not fixed. GStreamer: GStreamer and its plugins (and Core's stdio filter). Dataflow: the
// device memory's and Core's stdio filter (GStreamer is never started in such a process).
void prewarm_runtime() {
  if (backend() == RunBackend::Dataflow) {
    pipeline_internal::ux::install_device_log_filters();
#if defined(SIMANEAT_WITH_DATAFLOW)
    runtime::dataflow_device::prewarm_device_memory();
#endif
    return;
  }
  gst_init_once();
}

std::future<void> prewarm_runtime_async() {
  return std::async(std::launch::async, []() { prewarm_runtime(); });
}

namespace pipeline_internal {

RunBackend use_backend() noexcept {
  return backend_of(g_backend.fetch_or(kFixed, std::memory_order_acq_rel));
}

void require_gstreamer_backend(const char* where, const DataflowNotCarried& what) {
  if (use_backend() == RunBackend::Dataflow) {
    throw_dataflow_not_carried(where, what);
  }
}

bool has_dataflow_backend() noexcept {
#if defined(SIMANEAT_WITH_DATAFLOW)
  return true;
#else
  return false;
#endif
}

} // namespace pipeline_internal

} // namespace simaai::neat

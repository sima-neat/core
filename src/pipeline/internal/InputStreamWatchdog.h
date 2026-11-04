#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/InputStreamWork.h"

#include <gst/gst.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace simaai::neat::pipeline_internal {

// A dropping policy terminates one processing interval and starts another for
// each surviving buffer. Never match appsrc counts against post-drop outputs.
class InputStreamWatchdog {
public:
  InputStreamWatchdog();
  ~InputStreamWatchdog();
  void observe(GstElement* source, GstElement* sink);
  void begin() {
    spans_.front()->begin();
  }
  void cancel() {
    spans_.front()->cancel();
  }
  std::int64_t pending() const;
  bool timed_out(int timeout_ms);

private:
  struct QueueProgress;
  std::vector<std::shared_ptr<InputStreamWork>> spans_;
  std::vector<std::unique_ptr<QueueProgress>> queues_;
};

} // namespace simaai::neat::pipeline_internal

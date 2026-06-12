#pragma once

#include "pipeline/GraphOptions.h"

#include <chrono>
#include <memory>
#include <string>

namespace simaai::neat::pipeline_internal::ux {

enum class VerboseTopic {
  Planner,
  Graph,
  Pipeline,
  InputStream,
  Tensor,
  Plugins,
  GStreamer,
};

class ScopedVerboseContext {
public:
  explicit ScopedVerboseContext(const VerboseOptions& opt);
  ~ScopedVerboseContext();

  ScopedVerboseContext(const ScopedVerboseContext&) = delete;
  ScopedVerboseContext& operator=(const ScopedVerboseContext&) = delete;

private:
  const VerboseOptions* prev_ = nullptr;
};

std::shared_ptr<void> acquire_runtime_verbosity(const VerboseOptions& opt);

const VerboseOptions& current_verbose_options();
VerboseOptions current_effective_verbose_options();

bool should_emit_progress(const VerboseOptions& opt);
bool should_emit_verbose_details(const VerboseOptions& opt);
bool should_emit_any_details(const VerboseOptions& opt);
bool should_emit_topic(const VerboseOptions& opt, VerboseTopic topic);
bool should_emit_topic_for_current_context(VerboseTopic topic);
bool should_emit_gstreamer(const VerboseOptions& opt);
bool should_emit_gstreamer_for_current_context();

void emit_line(const VerboseOptions& opt, const std::string& message);

class ProgressReporter {
public:
  ProgressReporter(const VerboseOptions& opt, int total_steps);

  void step(const std::string& message);
  void detail(const std::string& message);
  void done(const std::string& message);

private:
  const VerboseOptions& opt_;
  int total_steps_ = 0;
  int current_step_ = 0;
  std::chrono::steady_clock::time_point start_;
};

} // namespace simaai::neat::pipeline_internal::ux

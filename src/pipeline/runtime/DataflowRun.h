#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The Dataflow backend's Run state (H0 plan 3.7). RunCore holds one in `dataflow_` for a Run that
// Graph::build() put on the Neat Dataflow runtime (RunOptions::backend), and every public Run entry
// branches to it before it reads any GStreamer state. The interface uses Core types only; the
// implementation (DataflowRun.cpp) is the only place that holds runtime objects, and it exists only
// in builds configured with SIMANEAT_DATAFLOW_SOURCE_DIR.

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/Run.h"
#include "pipeline/internal/InputStreamStats.h"
#include "policy/OverflowPolicy.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat::runtime {

struct RunCore;

// Default visibility on purpose: RunCore (default visibility) holds it by value of a unique_ptr,
// and a hidden field type there is a -Wattributes warning. It declares no runtime type.
class DataflowRunState {
public:
  virtual ~DataflowRunState() = default;

  // `input` is the named endpoint of push(name, ...); nullopt for the unnamed push. `block`: push
  // (waits for room, never drops) or try_push (applies the overflow policy). Return as Run's.
  virtual bool push(std::optional<std::string_view> input, const TensorList& inputs, bool block,
                    const char* where) = 0;
  virtual bool push(std::optional<std::string_view> input, const Sample& inputs, bool block,
                    const char* where) = 0;
  virtual bool push(std::optional<std::string_view> input, const std::vector<cv::Mat>& inputs,
                    bool block, const char* where) = 0;
  // GStreamer holders do not exist on this backend: always throws.
  virtual bool push_holder(const std::shared_ptr<void>& holder, bool block, const char* where) = 0;

  virtual PullStatus pull(std::optional<std::string_view> output, int timeout_ms, Sample& out,
                          PullError* err) = 0;

  virtual void close_input() = 0;
  virtual void stop() = 0;

  virtual bool running() const = 0;
  virtual std::vector<std::string> input_names() const = 0;
  virtual std::vector<std::string> output_names() const = 0;
  virtual std::string last_error() const = 0;
  virtual std::optional<PullError> last_error_detail() const = 0;
  virtual InputStreamStats input_stats() const = 0;
};

#pragma GCC visibility push(hidden)

// What DataflowGraphCompiler decided for a qualifying Input -> Output graph (H0 plan 3.4, 3.6).
struct DataflowRunConfig {
  InputOptions input;   ///< the public Input's options, resolved as compile_public_graph does
  OutputOptions output; ///< the public Output's options
  RunOptions run;       ///< RunOptions with the preset's queue defaults resolved
  std::uint32_t link_depth = 1;
  OverflowPolicy link_policy = OverflowPolicy::Block; ///< KeepLatest is the link's DropOldest
  /// OutputOptions::drop: the terminal queue drops its oldest message for every push, blocking
  /// or not (appsink drop=true); otherwise link_policy applies to try_push only.
  bool terminal_drop = false;
  std::vector<std::string> input_names;  ///< Graph::inputs(), sorted
  std::vector<std::string> output_names; ///< Graph::outputs(), sorted
  std::string default_stream;            ///< Sample::stream_id when a push carries none
  std::string label;                     ///< the graph, for messages
};

// Builds the Run state. A seed binds the input's spec now; without one, a declared image input
// binds now and anything else binds at the first push. Throws NeatError on a compile error.
std::unique_ptr<DataflowRunState> make_dataflow_run_state(RunCore& core, DataflowRunConfig config,
                                                          const Tensor* seed);

#pragma GCC visibility pop

} // namespace simaai::neat::runtime

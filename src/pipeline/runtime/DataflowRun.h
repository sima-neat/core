#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The Dataflow backend's Run state. RunCore holds one in `dataflow_` for a Run that
// Graph::build() put on the Neat Dataflow runtime (set_backend), and every public Run entry
// branches to it before it reads any GStreamer state. The interface uses Core types only; the
// implementation (DataflowRun.cpp and its DataflowInputPort) holds the runtime objects, and it
// exists only in builds configured with SIMANEAT_DATAFLOW_SOURCE_DIR.

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/Run.h"
#include "pipeline/internal/InputStreamStats.h"
#include "policy/OverflowPolicy.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
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
  // A TensorList is one item (a leaf, or a positional record); its first tensor's stamp proposes
  // the message's stream, frame_id, source_time and flags; a list of one positional record pushes
  // the record's fields. With `block`, a list of one Tensor from this Run's alloc() publishes its
  // slot without a copy (RunAllocation.h); try_push copies it like any CPU tensor.
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
  // Run::pull(milliseconds): the item's tensors (one for a leaf, one per field of a record), each
  // carrying the message's stamp in its envelope.
  virtual PullStatus pull(std::optional<std::string_view> output, int timeout_ms, TensorList& out,
                          PullError* err) = 0;
  // Run::alloc(name): a writable Tensor over a slot of the input's pool (RunAllocation.h).
  virtual Tensor alloc(std::string_view input, const char* where) = 0;

  virtual void close_input() = 0;
  virtual void stop() = 0;

  virtual bool running() const = 0;
  // A default input (output) exists: the graph has exactly one public Input (Output), so the
  // unnamed push (pull) reaches it. Run::can_push and Run::can_pull, as the graph path answers
  // them.
  virtual bool can_push() const = 0;
  virtual bool can_pull() const = 0;
  virtual std::vector<std::string> input_names() const = 0;
  virtual std::vector<std::string> output_names() const = 0;
  virtual std::string last_error() const = 0;
  virtual std::optional<PullError> last_error_detail() const = 0;
  // Every input's Feed and links summed. push_count counts the pushes that returned true, as
  // InputStreamStats states it.
  virtual InputStreamStats input_stats() const = 0;
  // RunStats' latency, an input's admission to an output's take in ms, over the Run's outputs,
  // when its Sessions time their frames (SessionOptions::timing, which Core leaves off: two clock
  // reads and a histogram increment a frame); false, with nothing written, otherwise.
  virtual bool latency(double& avg_ms, double& min_ms, double& max_ms) const = 0;
  // When the last pull returned a message; the clock's epoch before the first.
  virtual std::chrono::steady_clock::time_point last_pull_at() const = 0;
};

#pragma GCC visibility push(hidden)

struct DataflowModelRoute; // DataflowModelRoute.h

// One public Output of the graph: one Fetch, fed by one link from its Input's Feed, with the
// Output's own depth and drop policy.
struct DataflowOutputConfig {
  OutputOptions output;             ///< the public Output's options
  std::vector<std::string> names;   ///< its public names (Graph::outputs()), sorted
  std::uint32_t ingress_depth = 1;  ///< RunOptions::queue_depth; 4 when it is <= 0
  std::uint32_t terminal_depth = 1; ///< OutputOptions::max_buffers; ingress_depth when it is <= 0
  std::uint32_t link_depth = 1;     ///< max(ingress_depth, terminal_depth)
  /// OutputOptions::drop: this output's queue drops its oldest message for every push, blocking
  /// or not (appsink drop=true); otherwise DataflowRunConfig::link_policy applies to try_push only.
  bool terminal_drop = false;
  std::size_t input = 0; ///< the index of the Input that feeds it (DataflowRunConfig::inputs)
};

// One public Input of the graph: one Feed with one link per Output it feeds (a fan-out when there
// are several), one Session of its own (each input binds by its own first push). On a model
// route the Feed's one link goes to the first stage's operator, and the last one's output
// link feeds the route's one Fetch.
struct DataflowInputConfig {
  InputOptions input; ///< the public Input's options, resolved as compile_public_graph does
  std::vector<std::string> names; ///< its public names (Graph::inputs()), sorted
  std::string default_stream;     ///< Sample::stream_id when a push carries none
  std::uint32_t feed_loans = 1;   ///< Feed::Options::loans: 2 when Run::alloc can hold a slot
  std::vector<std::size_t>
      outputs; ///< the Outputs it feeds (DataflowRunConfig::outputs), link order
  /// A model route: the model's stages between this Input and its one Output (Feed -> one
  /// operator per stage -> Fetch). Null: the Input feeds its Outputs directly.
  std::shared_ptr<const DataflowModelRoute> model;
};

// What DataflowGraphCompiler decided for a qualifying graph of public Inputs and Outputs, each
// Output fed by exactly one Input.
struct DataflowRunConfig {
  RunOptions run; ///< RunOptions with the preset's queue defaults resolved
  OverflowPolicy link_policy = OverflowPolicy::Block; ///< KeepLatest is the link's DropOldest
  std::vector<DataflowInputConfig> inputs;            ///< vertex order
  std::vector<DataflowOutputConfig> outputs;          ///< vertex order
  std::vector<std::string> input_names;  ///< every input's names (Run::input_names), sorted
  std::vector<std::string> output_names; ///< every output's names (Run::output_names), sorted
  std::string label;                     ///< the graph, for messages
};

// "[a, b]": endpoint names in messages.
inline std::string dataflow_names_text(const std::vector<std::string>& names) {
  std::string out = "[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    out += (i == 0 ? "" : ", ") + names[i];
  }
  return out + "]";
}

// Builds the Run state. A seed binds the (one) input's spec now (one tensor: a leaf; several: a
// record); without one, each input whose declaration (an image, or a declared shape) gives its
// spec binds now and every other input binds at its own first push. Throws NeatError on a compile
// error.
std::unique_ptr<DataflowRunState> make_dataflow_run_state(RunCore& core, DataflowRunConfig config,
                                                          std::span<const Tensor> seed);

#pragma GCC visibility pop

} // namespace simaai::neat::runtime

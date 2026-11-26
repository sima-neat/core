// The Dataflow backend behind Core's Run (H0 C3), on the public API only: the process runs on the
// Dataflow backend (--backend=dataflow, set_backend), and no runtime type is named
// (libsima_neat.so hides the runtime's symbols). Registered a second time with
// --backend=gstreamer, which runs the rows that hold on both backends.
//
// Covers: a seedless, undeclared linear Run refused at the build with GStreamer's error; an
// undeclared input of a graph Run bound by its first push and enforced; a declared input with a
// dynamic axis; a push outside the declared shape rejected at push; the build errors, a fan-out
// building (unit_dataflow_endpoints_test covers several endpoints) and an Input feeding no Output;
// a Run Dataflow does not carry (the not-carried error at Graph::build, never a Run on
// GStreamer); the Session report Run::stop prints;
// the holder push error; Graph::run (a synchronous Run); a record bound by its first push (C2);
// the link depth (the larger of queue_depth and max_buffers, each after its default);
// try_push accounting
// under KeepLatest, DropIncoming and Block; a blocking push that never drops under DropIncoming and
// KeepLatest; close_input cancelling a parked push; stop unblocking a pending pull; pull timeouts
// -1/0/t; identity (frame_id, stream_id, pts_ns, input_seq) round trip; NV12 planes; DMA-heap
// tensors; Owned output; a pulled tensor outliving its Run; named endpoints; the push-time checks
// of a tensor that repeats an accepted geometry, back for any change they read (C7a); a try_push
// refused by a full link reading none of its tensor's bytes (C7c; with copy_input, so that the
// accepted ones copy: W2 wraps them otherwise, unit_run_push_wrap_test); the deprecated Run forms'
// one-time warnings; the options without a counterpart here (an appsrc's own: accepted, silently;
// caps_override, OutputOptions::sync, an unbounded queue: refused at the build in the one form);
// RunPreset::Realtime with an explicit
// Block on a connected graph (kept as given) and on a linear Run (kept too: only
// OverflowPolicy::Auto takes the preset's KeepLatest, on both backends); concurrent stops;
// state and measurement reads racing a stop; device inputs (H0b: InputOptions::memory_policy Ev74
// and Dms0): a CPU tensor copied into the input's DMA-BUF pool (with the compatibility copy
// allowed: without it the push throws, unit_run_backend_differential_test), Run::alloc on the
// pool, a pulled device tensor pushed
// back (copied, and copied again once its geometry is one the input has validated), an Owned pull,
// also of an EV74 tensor that was wrapped.

// This file calls the deprecated forms on purpose.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "dmabuf_identity.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include <unistd.h>

#include <cstdio>
#include <exception>
#include <sstream>

namespace {

using namespace simaai::neat;
using Clock = std::chrono::steady_clock;

// The Run options of most rows (the process runs on Dataflow).
RunOptions dataflow(int queue_depth = 4, OverflowPolicy policy = OverflowPolicy::Block) {
  RunOptions opt;
  opt.queue_depth = queue_depth;
  opt.overflow_policy = policy;
  return opt;
}

std::int64_t ms_since(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
}

// An RGB tensor whose bytes encode `tag` and their position, so a pull can be checked exactly.
Tensor rgb(int w, int h, std::uint8_t tag) {
  Tensor t = make_color_tensor(w, h, ImageSpec::PixelFormat::RGB, tag);
  auto map = t.storage->map(MapMode::Write);
  auto* bytes = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < map.size_bytes; ++i) {
    bytes[i] = static_cast<std::uint8_t>(tag + i * 7U);
  }
  return t;
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

// The next Sample of the Run, or of its output `output`, through the structured pull (the
// deprecated std::optional<Sample> forms fail on Dataflow); nullopt unless the pull is Ok.
std::optional<Sample> pull_sample(Run& run, int timeout_ms, const std::string& output = {}) {
  Sample s;
  PullError e;
  const PullStatus st =
      output.empty() ? run.pull(timeout_ms, s, &e) : run.pull(output, timeout_ms, s, &e);
  if (st != PullStatus::Ok) {
    return std::nullopt;
  }
  return s;
}

Sample tagged(int frame_id, const std::string& stream, std::int64_t pts, std::uint8_t tag) {
  Sample s;
  s.kind = SampleKind::Tensor;
  s.tensor = rgb(32, 24, tag);
  s.frame_id = frame_id;
  s.stream_id = stream;
  s.pts_ns = pts;
  return s;
}

Graph input_output(InputOptions in = {}, OutputOptions out = {}) {
  Graph g;
  g.add(nodes::Input(in));
  g.add(nodes::Output(out));
  return g;
}

// The same with named endpoints: a graph Run, which GStreamer builds without a seed too (a plain
// linear Run it refuses to build without one).
Graph named_input_output(InputOptions in = {}) {
  Graph g;
  g.add(nodes::Input("in", in));
  g.add(nodes::Output("out"));
  g.connect("in", "out");
  return g;
}

Tensor pulled_tensor(Run& run, int timeout_ms, const std::string& what) {
  Sample s;
  PullError err;
  const PullStatus st = run.pull(timeout_ms, s, &err);
  require(st == PullStatus::Ok,
          what + ": pull status " + std::to_string(static_cast<int>(st)) + " " + err.message);
  const TensorList tensors = tensors_from_sample(s);
  require(tensors.size() == 1U, what + ": one tensor expected");
  return tensors.front();
}

MeasureReport measured(Run& run, const std::function<void()>& fn) {
  MeasureScope scope = run.start_measurement();
  fn();
  return scope.stop();
}

std::uint64_t dropped_while(Run& run, const std::function<void()>& fn) {
  return measured(run, fn).counters.inputs_dropped;
}

// fd 2 to a temporary file while alive; text() restores it and returns what was written.
class StderrCapture {
public:
  StderrCapture() : file_(std::tmpfile()) {
    require(file_ != nullptr, "could not open the stderr capture");
    std::fflush(stderr);
    saved_ = dup(STDERR_FILENO);
    require(saved_ >= 0 && dup2(fileno(file_), STDERR_FILENO) >= 0, "could not capture stderr");
  }
  ~StderrCapture() {
    (void)text();
    std::fclose(file_);
  }
  StderrCapture(const StderrCapture&) = delete;
  StderrCapture& operator=(const StderrCapture&) = delete;

  std::string text() {
    std::fflush(stderr);
    if (saved_ >= 0) {
      (void)dup2(saved_, STDERR_FILENO);
      close(saved_);
      saved_ = -1;
    }
    std::rewind(file_);
    std::string out;
    char chunk[4096];
    while (const auto n = std::fread(chunk, 1U, sizeof(chunk), file_)) {
      out.append(chunk, n);
    }
    return out;
  }

private:
  std::FILE* file_;
  int saved_ = -1;
};

int lines_starting_with(const std::string& text, const std::string& prefix) {
  int n = 0;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);) {
    n += line.rfind(prefix, 0) == 0 ? 1 : 0;
  }
  return n;
}

constexpr const char* kDeprecatedForms[] = {
    "Run::push(const std::vector<cv::Mat>&)",
    "Run::push(std::string_view, const std::vector<cv::Mat>&)",
    "Run::try_push(const std::vector<cv::Mat>&)",
    "Run::try_push(std::string_view, const std::vector<cv::Mat>&)",
    "Run::pull(int)",
    "Run::pull(std::string_view, int)",
    "Run::pull_tensors(int)",
    "Run::pull_tensors(std::string_view, int)",
    "Run::pull_samples(int)",
    "Run::pull_samples(std::string_view, int)"};

// Run's ten deprecated forms, each called twice (H0 C5). On GStreamer each works and prints one
// "[WARN] Run::<form> is deprecated ...; use Run::...  It does not work on the Dataflow backend."
// line however often it is called (the warnings are once per process: the first case of the
// process). On Dataflow each fails at the call, before anything is pushed or pulled, with the
// not-carried error that names the replacement and the GStreamer backend.
void deprecated_forms() {
  InputOptions image;
  image.payload_type = PayloadType::Image;
  image.format = FormatTag::BGR;
  image.width = 32;
  image.height = 24;
  Graph g;
  g.add(nodes::Input("image", image));
  g.add(nodes::Output("out"));
  const std::vector<cv::Mat> mats{cv::Mat(24, 32, CV_8UC3, cv::Scalar(1, 2, 3))};
  if (sima_test::on_dataflow()) {
    Run run = g.build(dataflow());
    const std::function<void()> calls[] = {
        [&] { (void)run.push(mats); },         [&] { (void)run.push("image", mats); },
        [&] { (void)run.try_push(mats); },     [&] { (void)run.try_push("image", mats); },
        [&] { (void)run.pull(2000); },         [&] { (void)run.pull("out", 2000); },
        [&] { (void)run.pull_tensors(2000); }, [&] { (void)run.pull_tensors("out", 2000); },
        [&] { (void)run.pull_samples(0); },    [&] { (void)run.pull_samples("out", 0); }};
    for (int i = 0; i < 2; ++i) {
      for (std::size_t f = 0; f < std::size(calls); ++f) {
        require_neat_error(
            calls[f], error_codes::kPipelineShape,
            std::string(kDeprecatedForms[f]) +
                " is deprecated and is not carried by the Dataflow backend: use Run::",
            ".\nHint: The deprecated form works on the GStreamer backend: call "
            "simaai::neat::set_backend(simaai::neat::RunBackend::GStreamer) (Python: "
            "pyneat.set_backend(pyneat.RunBackend.GStreamer)) before the program constructs its "
            "first Model or builds its first Graph.");
      }
    }
    require(run.pull(std::chrono::milliseconds(50)).status() == PullStatus::Timeout,
            "a refused deprecated push pushed nothing");
    require(run.last_error().empty(), "a refused call is no error of the Run: " + run.last_error());
    run.close();
    return;
  }
  std::string text;
  {
    StderrCapture capture;
    Run run = g.build(dataflow());
    for (int i = 0; i < 2; ++i) {
      require(run.push(mats) && run.push("image", mats) && run.try_push(mats) &&
                  run.try_push("image", mats),
              "the cv::Mat pushes");
      require(run.pull(2000).has_value() && run.pull("out", 2000).has_value() &&
                  run.pull_tensors(2000).size() == 1U && run.pull_tensors("out", 2000).size() == 1U,
              "the deprecated pulls return what was pushed");
      require(run.pull_samples(0).empty() && run.pull_samples("out", 0).empty(),
              "pull_samples on an empty output");
    }
    run.close();
    text = capture.text();
  }
  for (const char* form : kDeprecatedForms) {
    const int n = lines_starting_with(
        text, std::string("[WARN] ") + form +
                  " is deprecated and will be removed in the next release; use Run::");
    require(n == 1, std::string(form) + " warns exactly once, naming its replacement (it warned " +
                        std::to_string(n) + " times):\n" + text);
  }
  require(lines_starting_with(text, "[WARN] Run::") == 10, "no other Run warning:\n" + text);
  require(text.find(". It does not work on the Dataflow backend.\n") != std::string::npos,
          "the warning says the form does not work on the Dataflow backend:\n" + text);
}

// The options without a counterpart on this backend. An appsrc's own (is_live, do_timestamp,
// stream_type, max_bytes, block, fps) describe a GStreamer element this backend does not have:
// the Run builds and round trips, and nothing is printed but the build's progress lines (which
// GStreamer's build prints too). The ones that change what the application observes are refused at
// the build in the one form: caps_override (the caps a GStreamer pipeline negotiates),
// OutputOptions::sync (pacing by GStreamer's clock), and a queue GStreamer leaves unbounded
// (queue_depth or max_buffers <= 0).
void options_without_a_counterpart() {
  std::string text;
  {
    StderrCapture capture;
    InputOptions in;
    in.is_live = false;
    in.do_timestamp = false;
    in.stream_type = 1;
    in.max_bytes = 1U << 20U;
    in.block = false;
    in.fps_n = 30;
    in.fps_d = 1;
    Graph g;
    g.add(nodes::Input("x", in));
    g.add(nodes::Output("y", OutputOptions{}));
    Run run = g.build(dataflow());
    require(run.push(TensorList{rgb(16, 8, 40)}) &&
                run.pull("y", std::chrono::milliseconds(5000)).ok(),
            "a Run with an appsrc's options round trips");
    run.close();
    text = capture.text();
  }
  std::string other;
  std::istringstream lines(text);
  for (std::string line; std::getline(lines, line);) {
    const bool progress =
        line.size() > 3U && line[0] == '[' && line[1] >= '0' && line[1] <= '9' && line[2] == '/';
    if (!progress) {
      other += line + "\n";
    }
  }
  require(other.empty(), "an appsrc's options print nothing but the build's progress:\n" + text);

  const auto refused = [](const InputOptions& in, const OutputOptions& out, const RunOptions& opt,
                          const std::string& subject) {
    require_neat_error(
        [&] {
          Graph g;
          g.add(nodes::Input("x", in));
          g.add(nodes::Output("y", out));
          (void)g.build(TensorList{rgb(8, 8, 1)}, opt);
        },
        error_codes::kOptionOutOfRange,
        "Graph::build(inputs): " + subject + " is not carried by the Dataflow backend: ",
        "\nHint: To run this program on the GStreamer backend, ");
  };
  InputOptions caps;
  caps.caps_override = "application/vnd.simaai.tensor";
  refused(caps, OutputOptions{}, dataflow(), "option InputOptions::caps_override of Input [x]");
  OutputOptions sync;
  sync.sync = true;
  refused(InputOptions{}, sync, dataflow(), "option OutputOptions::sync");
  OutputOptions unbounded;
  unbounded.max_buffers = 0;
  refused(InputOptions{}, unbounded, dataflow(), "option OutputOptions::max_buffers = 0");
  refused(InputOptions{}, OutputOptions{}, dataflow(0), "option RunOptions::queue_depth = 0");
}

void undeclared_input_bound_by_first_push() {
  // A plain linear Run without a seed: GStreamer's refusal and text.
  {
    Graph linear = input_output();
    bool refused = false;
    try {
      (void)linear.build(dataflow());
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()) ==
                "RunCore::start_pipeline_segment: push-style segment requires an input seed";
    }
    require(refused, "a seedless, undeclared linear Run is refused at the build");
  }
  Graph g = named_input_output();
  Run run = g.build(dataflow()); // seedless: the first push binds the input
  {
    Sample s;
    PullError err;
    require(run.pull(30, s, &err) == PullStatus::Timeout, "pull before any push times out");
  }
  const Tensor a = rgb(32, 24, 0x10);
  require(run.push(TensorList{a}), "first push binds and publishes");
  const Tensor out = pulled_tensor(run, 2000, "first push");
  require(out.shape == a.shape && out.dtype == TensorDType::UInt8, "shape and dtype kept");
  require(bytes_of(out) == bytes_of(a), "bytes kept");
  // `a` is wrapped (W2: CPU storage, 64-byte aligned, a whole slot): the pulled tensor is its
  // storage itself (plan D14), read-only. A copied push's is a CpuExternal lease on the slot
  // (unit_run_push_wrap_test).
  require(out.storage == a.storage && out.read_only,
          "a pulled tensor is the pushed storage, read-only");
  require(out.semantic.image && out.semantic.image->format == ImageSpec::PixelFormat::RGB,
          "image format kept");
  // The binding is enforced from now on.
  require_neat_error([&] { (void)run.push(TensorList{rgb(40, 24, 1)}); }, error_codes::kInputShape,
                     "out of bounds");
  require_neat_error(
      [&] {
        (void)run.push(TensorList{
            Tensor::from_vector(std::vector<float>(6, 1.0F), {2, 3}, TensorMemory::CPU)});
      },
      error_codes::kInputShape);
  require(run.push(TensorList{rgb(32, 24, 0x20)}), "a fitting push still passes");
  require(bytes_of(pulled_tensor(run, 2000, "second push")) == bytes_of(rgb(32, 24, 0x20)),
          "second push bytes");
  run.close();
}

void declared_input_with_dynamic_axis() {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.memory_policy = InputMemoryPolicy::SystemMemory;
  in.max_width = 96;
  in.max_height = 64;
  Graph g = input_output(in);
  Run run = g.build(TensorList{rgb(64, 48, 1)}, dataflow());
  for (const auto& [w, h] : std::vector<std::pair<int, int>>{{64, 48}, {32, 16}, {96, 64}}) {
    const Tensor t = rgb(w, h, static_cast<std::uint8_t>(w));
    require(run.push(TensorList{t}), "dynamic-axis push " + std::to_string(w));
    const Tensor out = pulled_tensor(run, 2000, "dynamic axis");
    require(out.shape == t.shape && bytes_of(out) == bytes_of(t), "dynamic axis round trip");
  }
  // Above the declared bound: rejected at push, nothing published.
  bool rejected = false;
  try {
    (void)run.push(TensorList{rgb(100, 48, 2)});
  } catch (const NeatError& e) {
    rejected = e.report().error_code == error_codes::kInputCapacity ||
               e.report().error_code == error_codes::kInputShape;
  }
  require(rejected, "a push above max_width is rejected with a structured error");
  Sample s;
  PullError err;
  require(run.pull(0, s, &err) == PullStatus::Timeout, "the rejected push published nothing");
  run.close();
}

void push_outside_declared_shape_rejected() {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.width = 64;
  in.height = 48;
  in.depth = 3;
  Graph g = input_output(in);
  Run run = g.build(dataflow()); // declared: bound at build, no seed needed
  require(run.push(TensorList{rgb(64, 48, 3)}), "declared shape passes");
  (void)pulled_tensor(run, 2000, "declared");
  bool rejected = false;
  try {
    (void)run.push(TensorList{rgb(32, 24, 3)});
  } catch (const NeatError& e) {
    rejected = e.report().error_code == error_codes::kInputShape ||
               e.report().error_code == error_codes::kInputCapacity;
  }
  require(rejected, "a push outside the declared shape is rejected at push");
  run.close();
}

void build_errors() {
  // H1: any node besides Input and Output.
  {
    Graph g;
    g.add(nodes::Input());
    g.add(nodes::Queue());
    g.add(nodes::Output());
    require_neat_error([&] { (void)g.build(dataflow()); }, error_codes::kPipelineShape,
                       "Graph::build: node Queue is not carried by the Dataflow backend: the "
                       "backend runs Inputs, Outputs and the stages of a Model between them.\n",
                       "Hint: To run this program on the GStreamer backend, ");
  }
  // A fan-out to two outputs builds since H0c (unit_dataflow_endpoints_test).
  {
    Graph g;
    g.add(nodes::Input("image"));
    g.add(nodes::Output("a"));
    g.add(nodes::Output("b"));
    g.connect("image", "a");
    g.connect("image", "b");
    Run run = g.build(dataflow());
    require(run.output_names() == std::vector<std::string>({"a", "b"}), "a fan-out builds");
    run.close();
  }
  // Two inputs, one of them connected to nothing: no Feed without a link.
  {
    Graph g;
    g.add(nodes::Input("left"));
    g.add(nodes::Input("right"));
    g.add(nodes::Output("out"));
    g.connect("left", "out");
    require_neat_error([&] { (void)g.build(dataflow()); }, error_codes::kPipelineShape,
                       "connected to no Output");
  }
  // A graph with no Input.
  {
    Graph g;
    g.add(nodes::Output());
    require_neat_error([&] { (void)g.build(dataflow()); }, error_codes::kPipelineShape,
                       "this graph is not carried by the Dataflow backend: it has no public "
                       "Input (a source graph).");
  }
  // A synchronous run (Graph::run): a Run driven once per call, its output the caller's own.
  {
    Graph g = input_output();
    for (std::uint8_t k = 1; k <= 2; ++k) {
      const TensorList out = g.run(TensorList{rgb(8, 8, k)}, dataflow());
      require(out.size() == 1U && bytes_of(out.front()) == bytes_of(rgb(8, 8, k)) &&
                  out.front().storage->kind == StorageKind::CpuOwned,
              "Graph::run gives the frame, owned");
    }
  }
  // Records (an item of several tensors) bind a record port since C2; the binding is enforced.
  {
    Graph g = named_input_output();
    Run run = g.build(dataflow());
    require(run.push(TensorList{rgb(8, 8, 1), rgb(8, 8, 2)}), "a record push");
    const auto s = pull_sample(run, 2000);
    require(s && s->tensors.size() == 2U && bytes_of(s->tensors[1]) == bytes_of(rgb(8, 8, 2)),
            "the record's tensors");
    require_neat_error([&] { (void)run.push(TensorList{rgb(8, 8, 1)}); }, error_codes::kInputShape,
                       "record");
    run.close();
  }
}

// No fallback: a Run that Dataflow does not carry fails at Graph::build with the not-carried
// error, whether the input binds at the build (a seed) or would bind at its first push (no seed).
// Nothing is built on GStreamer in its place. Here: a link deeper than the backend's (2^30 items;
// RunOptions::queue_depth above that).
void no_fallback_to_gstreamer() {
  constexpr int kDeeperThanALink = (1 << 30) + 1;
  const RunOptions options = dataflow(kDeeperThanALink);
  for (const bool seeded : {false, true}) {
    require_neat_error(
        [&] {
          Graph g = input_output();
          (void)(seeded ? g.build(TensorList{rgb(8, 8, 1)}, options) : g.build(options));
        },
        error_codes::kOptionOutOfRange,
        "option RunOptions::queue_depth = 1073741825 with OutputOptions::max_buffers = ",
        " is not carried by the Dataflow backend: the link from an Input to its Output is as "
        "deep as the larger of the two, 1073741825, and a link holds at most 1073741824 "
        "items.\nHint: To run this program on the GStreamer backend, ");
  }
}

// With SIMA_GRAPH_DEBUG set, Run::stop prints each input's Session report (SessionReport::to_text)
// under a [NEAT] line naming the input and its Outputs. The Session counts every item it carried:
// the seed the build pushed and pulled once, then the two frames.
void stop_prints_the_session_report() {
  std::string text;
  ::setenv("SIMA_GRAPH_DEBUG", "1", 1);
  {
    StderrCapture capture;
    Graph g = input_output(InputOptions{}, OutputOptions{});
    Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow());
    for (int i = 0; i < 2; ++i) {
      require(run.push(TensorList{rgb(8, 8, static_cast<std::uint8_t>(i))}), "report: push");
      (void)pulled_tensor(run, 2000, "report");
    }
    run.close();
    text = capture.text();
  }
  ::unsetenv("SIMA_GRAPH_DEBUG");
  for (const char* line :
       {"[NEAT] Run::stop: input [", ", outputs [", "[NEAT]   session  dropped_frames 0",
        "[NEAT]   feed     input", "pushes 3", "[NEAT]   link     input.0 -> output.0",
        "[NEAT]   fetch    output", "pulled 3", "[NEAT]   pool     input.0"}) {
    require(text.find(line) != std::string::npos,
            std::string("the stop's report has \"") + line + "\":\n" + text);
  }
}

void holder_push_is_an_error() {
  Graph g = input_output();
  Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow());
  const auto holder = std::make_shared<int>(1);
  require_neat_error([&] { (void)run.push_holder(holder); }, error_codes::kPipelineShape,
                     "a buffer holder is not carried by the Dataflow backend");
  require_neat_error([&] { (void)run.try_push_holder(holder); }, error_codes::kPipelineShape,
                     "a buffer holder is not carried by the Dataflow backend");
  run.close();
}

// The link holds max(RunOptions::queue_depth, OutputOptions::max_buffers) messages (the deeper of
// GStreamer's input queue and its appsink), each after its default: counted with try_push
// on a Block link, which refuses only when the link holds the whole depth.
int accepted_until_full(Run& run, int limit) {
  int accepted = 0;
  while (accepted < limit &&
         run.try_push(TensorList{rgb(8, 8, static_cast<std::uint8_t>(accepted))}))
    ++accepted;
  return accepted;
}

void link_depth_is_max_of_queue_depth_and_max_buffers() {
  struct Case {
    const char* what;
    RunOptions opt;
    int max_buffers;
    int depth;
  };
  RunOptions reliable = dataflow();
  reliable.preset = RunPreset::Reliable; // queue_depth left at the default: the preset's 8
  const Case cases[] = {
      {"defaults (queue_depth 4, max_buffers 4)", dataflow(), OutputOptions{}.max_buffers, 4},
      {"queue_depth 2, max_buffers 5", dataflow(2), 5, 5},
      {"Reliable preset (queue_depth 8), max_buffers 4", reliable, 4, 8},
  };
  for (const Case& c : cases) {
    OutputOptions out;
    out.max_buffers = c.max_buffers;
    Graph g = input_output({}, out);
    Run run = g.build(TensorList{rgb(8, 8, 1)}, c.opt);
    const int accepted = accepted_until_full(run, c.depth + 4);
    require(accepted == c.depth, std::string(c.what) + ": the link holds " +
                                     std::to_string(c.depth) + ", try_push accepted " +
                                     std::to_string(accepted));
    for (int i = 0; i < c.depth; ++i) {
      require(bytes_of(pulled_tensor(run, 1000, c.what)) ==
                  bytes_of(rgb(8, 8, static_cast<std::uint8_t>(i))),
              std::string(c.what) + ": in order, frame " + std::to_string(i));
    }
    run.close();
  }
  // GStreamer's pattern (async_stream_test): queue_depth blocking pushes before the first pull all
  // return; the next one parks until a pull makes room.
  OutputOptions out;
  out.max_buffers = 2;
  Graph g = input_output({}, out);
  Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow(3));
  auto pushes = std::async(std::launch::async, [&] {
    for (int i = 0; i < 3; ++i)
      if (!run.push(TensorList{rgb(8, 8, static_cast<std::uint8_t>(i))}))
        return false;
    return true;
  });
  require(pushes.wait_for(std::chrono::seconds(2)) == std::future_status::ready && pushes.get(),
          "queue_depth blocking pushes return without a puller");
  auto fourth = std::async(std::launch::async, [&] { return run.push(TensorList{rgb(8, 8, 3)}); });
  require(fourth.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
          "one more parks on the full link");
  require(bytes_of(pulled_tensor(run, 1000, "first")) == bytes_of(rgb(8, 8, 0)),
          "a pull takes the oldest");
  require(fourth.wait_for(std::chrono::seconds(1)) == std::future_status::ready && fourth.get(),
          "and lets the parked push in");
  run.close();
}

// The input statistics count what the counters do: push_count the pushes that returned true,
// dropped_frames what a link dropped (MeasureReport::input).
void try_push_accounting() {
  // The link holds max(queue_depth 1, max_buffers 2) = 2.
  OutputOptions out;
  out.max_buffers = 2;
  const auto counted = [](const MeasureReport& r, std::uint64_t pushes, std::uint64_t dropped,
                          std::uint64_t failures, const std::string& what) {
    require(r.input.push_count == pushes && r.counters.inputs_pushed == pushes,
            what + ": push_count " + std::to_string(r.input.push_count) + ", accepted " +
                std::to_string(pushes));
    require(r.input.dropped_frames == dropped,
            what + ": dropped_frames " + std::to_string(r.input.dropped_frames));
    require(r.input.push_failures == failures,
            what + ": push_failures " + std::to_string(r.input.push_failures));
  };
  // DropIncoming: the first two are queued, the rest are dropped and counted.
  {
    Graph g = input_output({}, out);
    Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow(1, OverflowPolicy::DropIncoming));
    std::vector<bool> accepted;
    const MeasureReport report = measured(run, [&] {
      for (int i = 0; i < 5; ++i)
        accepted.push_back(run.try_push(TensorList{rgb(8, 8, static_cast<std::uint8_t>(i))}));
    });
    const std::uint64_t dropped = report.counters.inputs_dropped;
    require(accepted == std::vector<bool>{true, true, false, false, false},
            "DropIncoming: the first two try_push are accepted, the rest refused");
    require(dropped == 3U, "DropIncoming: three drops counted, got " + std::to_string(dropped));
    counted(report, 2U, 3U, 0U, "DropIncoming");
    require(bytes_of(pulled_tensor(run, 1000, "DropIncoming")) == bytes_of(rgb(8, 8, 0)) &&
                bytes_of(pulled_tensor(run, 1000, "DropIncoming")) == bytes_of(rgb(8, 8, 1)),
            "DropIncoming keeps the oldest");
    run.close();
  }
  // KeepLatest: every try_push is accepted; each one past the second evicts and counts a drop.
  {
    Graph g = input_output({}, out);
    Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow(1, OverflowPolicy::KeepLatest));
    bool all = true;
    const MeasureReport report = measured(run, [&] {
      for (int i = 0; i < 5; ++i)
        all = run.try_push(TensorList{rgb(8, 8, static_cast<std::uint8_t>(i))}) && all;
    });
    const std::uint64_t dropped = report.counters.inputs_dropped;
    require(all, "KeepLatest: try_push never refuses");
    require(dropped == 3U, "KeepLatest: three drops counted, got " + std::to_string(dropped));
    counted(report, 5U, 3U, 0U, "KeepLatest");
    require(bytes_of(pulled_tensor(run, 1000, "KeepLatest")) == bytes_of(rgb(8, 8, 3)) &&
                bytes_of(pulled_tensor(run, 1000, "KeepLatest")) == bytes_of(rgb(8, 8, 4)),
            "KeepLatest keeps the newest");
    run.close();
  }
  // Block: refused when full, nothing dropped.
  {
    Graph g = input_output({}, out);
    Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow(1, OverflowPolicy::Block));
    std::vector<bool> accepted;
    const MeasureReport report = measured(run, [&] {
      for (int i = 0; i < 4; ++i)
        accepted.push_back(run.try_push(TensorList{rgb(8, 8, static_cast<std::uint8_t>(i))}));
    });
    require(accepted == std::vector<bool>{true, true, false, false}, "Block: refused when full");
    require(report.counters.inputs_dropped == 0U, "Block: no drop counted");
    counted(report, 2U, 0U, 2U, "Block");
    run.close();
  }
}

// A try_push that the full link refuses is refused before its tensor is read: the runtime's alloc
// answers a full link itself (scheduler-dataflow A3), so no slot is taken and nothing is copied.
// The tensor's storage counts its maps: each accepted try_push maps it for the copy, a refused one
// does not, and the refusals' outcome and drop count are try_push's as before
// (try_push_accounting).
void refused_try_push_reads_no_byte() {
  OutputOptions out;
  out.max_buffers = 2; // the link holds max(queue_depth 1, max_buffers 2) = 2
  for (const OverflowPolicy policy : {OverflowPolicy::DropIncoming, OverflowPolicy::Block}) {
    Graph g = input_output({}, out);
    // copy_input: the accepted pushes copy (and so map) the tensor, which shows the counter
    // works; without it they wrap its storage and read no byte either (W2,
    // unit_run_push_wrap_test).
    RunOptions opt = dataflow(1, policy);
    opt.advanced.copy_input = true;
    Run run = g.build(TensorList{rgb(8, 8, 1)}, opt);
    Tensor t = rgb(8, 8, 9);
    const auto maps = std::make_shared<std::atomic<int>>(0);
    auto counted = std::make_shared<TensorBuffer>(*t.storage);
    counted->map_fn = [plain = t.storage, maps](MapMode mode) {
      maps->fetch_add(1);
      return plain->map(mode);
    };
    t.storage = counted;
    require(run.try_push(TensorList{t}) && run.try_push(TensorList{t}),
            "two accepted try_push fill the link");
    const int accepted_maps = maps->load();
    require(accepted_maps >= 2, "an accepted try_push reads its tensor for the copy");
    const auto dropped = dropped_while(run, [&] {
      for (int i = 0; i < 3; ++i) {
        require(!run.try_push(TensorList{t}), "refused on the full link");
      }
    });
    require(maps->load() == accepted_maps, "a refused try_push reads no byte of its tensor, got " +
                                               std::to_string(maps->load() - accepted_maps) +
                                               " maps");
    require(dropped == (policy == OverflowPolicy::DropIncoming ? 3U : 0U),
            "the drops counted as before, got " + std::to_string(dropped));
    require(bytes_of(pulled_tensor(run, 1000, "refused")) == bytes_of(rgb(8, 8, 9)),
            "what was accepted is intact");
    run.close();
  }
}

void blocking_push_never_drops() {
  for (const OverflowPolicy policy : {OverflowPolicy::DropIncoming, OverflowPolicy::KeepLatest}) {
    OutputOptions out;
    out.max_buffers = 1;
    Graph g = input_output({}, out);
    Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow(1, policy));
    constexpr int kFrames = 40;
    std::atomic<bool> push_failed{false};
    std::uint64_t dropped = 0;
    {
      MeasureScope scope = run.start_measurement();
      std::thread pusher([&] {
        for (int i = 0; i < kFrames; ++i) {
          Sample s = tagged(i, "", -1, static_cast<std::uint8_t>(i));
          s.tensor = rgb(8, 8, static_cast<std::uint8_t>(i));
          if (!run.push(Sample{s}))
            push_failed = true;
        }
      });
      for (int i = 0; i < kFrames; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(i % 4 == 0 ? 5 : 0));
        Sample s;
        PullError err;
        require(run.pull(5000, s, &err) == PullStatus::Ok, "blocking push: every frame arrives");
        require(s.frame_id == i, "blocking push: in order, frame " + std::to_string(i));
      }
      pusher.join();
      dropped = scope.stop().counters.inputs_dropped;
    }
    require(!push_failed, "blocking push never returns false on a live run");
    require(dropped == 0U, "blocking push never drops, whatever the policy");
    run.close();
  }
}

// RunPreset::Realtime with queue_depth 1 and an explicit OverflowPolicy::Block, on a 32x24 RGB
// input into Output EveryFrame(1). True when try_push refuses 10 times in a row, 10 ms apart,
// within 5 s (the queue is full and stays full); `dropped`: the input drops meanwhile.
struct RealtimeBlock {
  InputOptions in;
  RunOptions opt = dataflow(1);
  RealtimeBlock() {
    in.payload_type = PayloadType::Image;
    in.format = FormatTag::RGB;
    in.width = in.max_width = 32;
    in.height = in.max_height = 24;
    in.depth = in.max_depth = 3;
    opt.preset = RunPreset::Realtime;
    opt.output_memory = OutputMemory::Owned;
  }
  static bool refuses_when_full(Run& run, const char* input, std::uint64_t& dropped) {
    MeasureScope scope = run.start_measurement();
    bool refused = false;
    int accepted = 0;
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    for (int consecutive = 0; !refused && Clock::now() < deadline;) {
      const TensorList item{rgb(32, 24, static_cast<std::uint8_t>(accepted))};
      if (input ? run.try_push(input, item) : run.try_push(item)) {
        ++accepted;
        consecutive = 0;
      } else if (++consecutive == 10) {
        refused = true;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    dropped = scope.stop().counters.inputs_dropped;
    require(accepted > 0, "Realtime + Block: nothing accepted");
    return refused;
  }
};

Graph connected_input_output(const InputOptions& in) {
  Graph source("input");
  source.add(nodes::Input("input", in));
  Graph sink("output");
  sink.add(nodes::Output("output", OutputOptions::EveryFrame(1)));
  Graph g("preset_overflow");
  g.connect(source, sink);
  return g;
}

// A connected (or named) graph is a graph Run, which resolves the Run preset as a plain linear Run
// does: under Realtime an explicit Block refuses a try_push when the queue is full and drops
// nothing, and a blocking push waits for a pull (pyneat's test_blocking_push_gil builds this
// graph); OverflowPolicy::Auto takes the preset's KeepLatest at the preset's queue depth (try_push
// never refuses and evicts the oldest). The same on both backends (each process runs it on its
// own).
void connected_realtime_resolves_the_preset() {
  const std::string on = std::string(" (") + sima_test::backend_name() + ")";
  {
    RealtimeBlock c;
    c.opt.overflow_policy = OverflowPolicy::Auto;
    c.opt.queue_depth = RunOptions{}.queue_depth;
    Graph g = connected_input_output(c.in);
    Run run = g.build(c.opt);
    std::uint64_t dropped = 0;
    require(!RealtimeBlock::refuses_when_full(run, "input", dropped),
            "connected Realtime + Auto: KeepLatest, try_push must never refuse" + on);
    require(dropped > 0U, "connected Realtime + Auto: KeepLatest evicts past the queue depth" + on);
    run.close();
  }
  const RealtimeBlock c;
  Graph g = connected_input_output(c.in);
  Run run = g.build(c.opt);
  std::uint64_t dropped = 0;
  require(RealtimeBlock::refuses_when_full(run, "input", dropped),
          "connected Realtime + explicit Block: try_push must refuse when full" + on);
  require(dropped == 0U,
          "connected Realtime + explicit Block: try_push dropped " + std::to_string(dropped) + on);
  auto consumer = std::async(std::launch::async, [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return pull_sample(run, 5000, "output");
  });
  const auto t0 = Clock::now();
  const bool pushed = run.push("input", TensorList{rgb(32, 24, 0xEE)});
  const std::int64_t waited = ms_since(t0);
  require(consumer.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              consumer.get().has_value(),
          "the consumer pulled a frame");
  require(pushed, "the blocking push is accepted once a pull makes room" + on);
  require(waited >= 100,
          "the blocking push waited for room (" + std::to_string(waited) + " ms)" + on);
  run.close();
}

// Run::can_push, can_pull and last_error follow close_input and the drain, as GStreamer's Run
// answers them, on a plain linear Run and on a connected graph Run: open, both true; after
// close_input no push is taken, a refused push's reason is last_error ("input is closed"), and
// what was pushed stays pullable; once a pull found the output closed and drained, can_pull is
// false. The same on both backends (each process runs it on its own).
void readiness_follows_close_and_drain() {
  const std::string on = std::string(" (") + sima_test::backend_name() + ")";
  for (const bool connected : {false, true}) {
    const std::string what = std::string(connected ? "connected" : "linear") + on;
    const RealtimeBlock c;
    RunOptions opt;
    opt.queue_depth = 4;
    opt.startup_preflight = false;
    Graph g = connected ? connected_input_output(c.in) : input_output(c.in);
    Run run = connected ? g.build(opt) : g.build(TensorList{rgb(32, 24, 1)}, opt);
    require(run.can_push() && run.can_pull(), "an open Run of one Input and one Output: " + what);
    const auto push = [&](std::uint8_t tag) {
      const TensorList item{rgb(32, 24, tag)};
      return connected ? run.push("input", item) : run.push(item);
    };
    const auto pull = [&](int timeout_ms) {
      return connected ? run.pull("output", std::chrono::milliseconds(timeout_ms))
                       : run.pull(std::chrono::milliseconds(timeout_ms));
    };
    require(push(2) && push(3), "two pushes: " + what);
    run.close_input();
    require(!run.can_push(), "close_input ends can_push: " + what);
    require(run.can_pull(), "what was pushed stays pullable: " + what);
    require(!push(4), "a push after close_input is refused: " + what);
    require_contains(run.last_error(), "input is closed",
                     "last_error says why the push was refused: " + what);
    for (int i = 0; i < 2; ++i) {
      const Result<TensorList> r = pull(5000);
      require(r.ok() && r->size() == 1U, "a queued item is pulled after close_input: " + what);
    }
    const Result<TensorList> end = pull(5000);
    require(end.status() == PullStatus::Closed, "then the output is closed: " + what);
    require(!run.can_pull() && !run.can_push(), "a drained Run advertises nothing: " + what);
    run.close();
  }
}

// A plain linear Run resolves its options as GStreamer's one-segment Run does, by the same code: an
// explicit Block survives the Realtime preset (try_push refuses when the queue is full and drops
// nothing), and only OverflowPolicy::Auto takes the preset's KeepLatest (try_push never refuses
// and evicts the oldest). The same on both backends (each process runs it on its own).
void linear_realtime_keeps_an_explicit_block() {
  {
    const std::string on = std::string(" (") + sima_test::backend_name() + ")";
    for (const OverflowPolicy policy : {OverflowPolicy::Block, OverflowPolicy::Auto}) {
      RealtimeBlock c;
      c.opt.overflow_policy = policy;
      Graph g = input_output(c.in, OutputOptions::EveryFrame(1));
      Run run = g.build(TensorList{rgb(32, 24, 1)}, c.opt);
      std::uint64_t dropped = 0;
      const bool refused = RealtimeBlock::refuses_when_full(run, nullptr, dropped);
      if (policy == OverflowPolicy::Block) {
        require(refused, "linear Realtime + explicit Block: try_push must refuse when full" + on);
        require(dropped == 0U, "linear Realtime + explicit Block: try_push dropped " +
                                   std::to_string(dropped) + on);
      } else {
        require(!refused, "linear Realtime + Auto: KeepLatest, try_push must never refuse" + on);
        require(dropped > 0U,
                "linear Realtime + Auto: KeepLatest evicts past the queue depth" + on);
      }
      run.close();
    }
  }
}

// Several threads stop the same Run at once while a push is parked: every call returns, the parked
// push is released.
void concurrent_stops_with_a_parked_push() {
  OutputOptions out;
  out.max_buffers = 2;
  Graph g = input_output({}, out);
  Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow(1));
  require(run.push(TensorList{rgb(8, 8, 1)}) && run.push(TensorList{rgb(8, 8, 2)}),
          "two pushes fill the link");
  auto parked = std::async(std::launch::async, [&] { return run.push(TensorList{rgb(8, 8, 9)}); });
  require(parked.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
          "the push parks on the full link");
  std::atomic<bool> go{false};
  std::vector<std::thread> stoppers;
  const auto t0 = Clock::now();
  for (int i = 0; i < 4; ++i) {
    stoppers.emplace_back([&] {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      run.stop();
    });
  }
  go.store(true, std::memory_order_release);
  for (auto& t : stoppers) {
    t.join();
  }
  require(ms_since(t0) < 3000, "four concurrent stops return within 3 s");
  require(parked.wait_for(std::chrono::seconds(1)) == std::future_status::ready && !parked.get(),
          "the parked push is released with false");
  require(!run.running(), "stopped");
  run.close();
  run.close(); // idempotent
}

// State and measurement reads from other threads while stop runs.
void reads_race_stop() {
  Graph g = input_output();
  Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow());
  for (std::uint8_t i = 0; i < 4; ++i) {
    require(run.push(TensorList{rgb(8, 8, i)}), "push");
  }
  MeasureScope measurement = run.start_measurement();
  std::atomic<bool> start{false};
  std::atomic<bool> keep_reading{true};
  std::atomic<std::uint64_t> reads{0};
  const auto wait_for_start = [&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  };
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      wait_for_start();
      while (keep_reading.load(std::memory_order_acquire)) {
        (void)run.running();
        (void)run.can_push();
        (void)run.can_pull();
        (void)run.last_error();
        (void)run.input_names();
        reads.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  std::exception_ptr measurement_error;
  std::thread stop_measurement([&] {
    wait_for_start();
    try {
      (void)measurement.stop();
    } catch (...) {
      measurement_error = std::current_exception();
    }
  });
  start.store(true, std::memory_order_release);
  run.stop();
  stop_measurement.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  keep_reading.store(false, std::memory_order_release);
  for (auto& r : readers) {
    r.join();
  }
  if (measurement_error) {
    std::rethrow_exception(measurement_error);
  }
  require(reads.load() > 0, "the readers ran");
  require(!run.running(), "stopped");
  run.close();
}

void close_input_cancels_a_parked_push() {
  OutputOptions out;
  out.max_buffers = 2;
  Graph g = input_output({}, out);
  Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow(1));
  require(run.push(TensorList{rgb(8, 8, 1)}) && run.push(TensorList{rgb(8, 8, 4)}),
          "two pushes fill the link (max of queue_depth 1, max_buffers 2)");
  auto parked = std::async(std::launch::async, [&] { return run.push(TensorList{rgb(8, 8, 2)}); });
  require(parked.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
          "the third push parks on the full link");
  const auto t0 = Clock::now();
  run.close_input();
  require(parked.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "close_input wakes the parked push");
  require(!parked.get(), "the cancelled push returns false");
  require(ms_since(t0) < 1000, "close_input does not wait behind the parked push");
  require(bytes_of(pulled_tensor(run, 1000, "queued")) == bytes_of(rgb(8, 8, 1)) &&
              bytes_of(pulled_tensor(run, 1000, "queued")) == bytes_of(rgb(8, 8, 4)),
          "what was queued stays pullable");
  Sample s;
  PullError err;
  require(run.pull(1000, s, &err) == PullStatus::Closed, "then the output is closed");
  require(!run.push(TensorList{rgb(8, 8, 3)}), "a push after close_input is refused");
  run.close();
}

void stop_unblocks_pull_and_refuses_push() {
  Graph g = input_output();
  Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow());
  auto pending = std::async(std::launch::async, [&] {
    Sample s;
    PullError err;
    return run.pull(5000, s, &err);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  run.stop();
  require(pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "a pending pull unblocks within 1 s of stop");
  require(pending.get() == PullStatus::Closed, "the pending pull reports Closed");
  require(!run.running(), "not running after stop");
  bool refused = false;
  try {
    refused = !run.push(TensorList{rgb(8, 8, 2)});
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("stopped") != std::string::npos;
  }
  require(refused, "a push after stop is refused");
  run.close();
  run.close();
}

void pull_timeouts() {
  Graph g = input_output();
  Run run = g.build(TensorList{rgb(8, 8, 1)}, dataflow());
  Sample s;
  PullError err;
  auto t0 = Clock::now();
  require(run.pull(0, s, &err) == PullStatus::Timeout, "pull(0) polls");
  require(ms_since(t0) < 50, "pull(0) does not wait");
  t0 = Clock::now();
  require(run.pull(120, s, &err) == PullStatus::Timeout, "pull(120) times out");
  const auto waited = ms_since(t0);
  require(waited >= 110 && waited < 700,
          "pull(120) waits about 120 ms, got " + std::to_string(waited));
  require(err.code == error_codes::kRuntimePull, "a timeout carries runtime.pull");
  require(run.last_error().empty(), "a timeout is not latched");
  auto forever = std::async(std::launch::async, [&] {
    Sample out;
    PullError e;
    return run.pull(-1, out, &e);
  });
  require(forever.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout,
          "pull(-1) waits");
  require(run.push(TensorList{rgb(8, 8, 9)}), "push wakes pull(-1)");
  require(forever.wait_for(std::chrono::seconds(2)) == std::future_status::ready &&
              forever.get() == PullStatus::Ok,
          "pull(-1) returns the pushed frame");
  run.close();
}

void identity_round_trip_and_names() {
  Graph in_graph;
  in_graph.add(nodes::Input("image"));
  Graph out_graph;
  out_graph.add(nodes::Output("classes"));
  Graph g;
  g.connect(in_graph, out_graph);
  Run run = g.build(dataflow());
  require(run.input_names() == std::vector<std::string>{"image"} &&
              run.output_names() == std::vector<std::string>{"classes"},
          "endpoint names");
  require(run.push("image", Sample{tagged(7, "cam-a", 1234, 1)}), "named push");
  require(run.push("image", Sample{tagged(8, "", -1, 2)}), "named push, no identity");
  require(run.push(Sample{tagged(-1, "cam-b", 99, 3)}), "default push");
  const auto first = pull_sample(run, 2000, "classes");
  const auto second = pull_sample(run, 2000, "classes");
  const auto third = pull_sample(run, 2000);
  require(first && second && third, "three outputs");
  require(first->frame_id == 7 && first->stream_id == "cam-a" && first->pts_ns == 1234,
          "frame_id, stream_id and pts_ns round trip");
  require(second->frame_id == 8 && second->stream_id == "image" && second->pts_ns == -1,
          "an empty stream_id becomes the endpoint's name on a connected graph");
  require(third->frame_id == -1 && third->stream_id == "cam-b", "frame_id -1 stays unset");
  require(first->input_seq == 0 && second->input_seq == 1 && third->input_seq == 2,
          "input_seq is contiguous per Run");
  require(first->kind == SampleKind::TensorSet && first->payload_tag == "RGB" &&
              first->media_type == "video/x-raw",
          "the pulled Sample's shape and tags");
  bool unknown = false;
  try {
    (void)run.push("nope", TensorList{rgb(32, 24, 1)});
  } catch (const std::exception& e) {
    unknown = std::string(e.what()).find("unknown input") != std::string::npos;
  }
  require(unknown, "an unknown input name is refused with Core's text");
  run.close();
}

void nv12_planes_round_trip() {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::NV12;
  in.width = 64;
  in.height = 48;
  Graph g = input_output(in);
  Run run = g.build(dataflow());
  Tensor t = make_nv12_tensor(64, 48, 0x22);
  {
    auto map = t.storage->map(MapMode::Write);
    auto* bytes = static_cast<std::uint8_t*>(map.data);
    for (std::size_t i = 0; i < map.size_bytes; ++i)
      bytes[i] = static_cast<std::uint8_t>(i * 13U);
  }
  require(run.push(TensorList{t}), "NV12 push");
  const Tensor out = pulled_tensor(run, 2000, "NV12");
  require(out.is_nv12() && out.planes.size() == 2U, "NV12 planes restored");
  require(out.copy_nv12_contiguous() == t.copy_nv12_contiguous(), "NV12 bytes kept");
  run.close();
}

// A push whose tensor repeats the geometry of an accepted one may skip the push-time checks (C7a);
// a change in anything they read brings them back, with the same refusal as on any push.
void repeated_geometry_is_rechecked_on_change() {
  Graph g = named_input_output();
  Run run = g.build(dataflow()); // the first push binds
  for (int i = 0; i < 3; ++i) {
    const Tensor t = rgb(32, 24, static_cast<std::uint8_t>(0x30 + i));
    require(run.push(TensorList{t}), "repeated geometry push " + std::to_string(i));
    require(bytes_of(pulled_tensor(run, 2000, "repeated geometry")) == bytes_of(t),
            "repeated geometry bytes " + std::to_string(i));
  }
  const auto refused = [&](const Tensor& t, const std::string& fragment) {
    require_neat_error([&] { (void)run.push(TensorList{t}); }, error_codes::kInputShape, fragment);
    Sample s;
    PullError err;
    require(run.pull(0, s, &err) == PullStatus::Timeout, "a refused push publishes nothing");
  };
  {
    // The same metadata over a storage one byte short.
    Tensor t = rgb(32, 24, 0x40);
    t.storage = make_cpu_owned_storage(t.storage->size_bytes - 1U);
    refused(t, "storage too small");
  }
  {
    // A byte offset into a storage large enough for it.
    Tensor t = rgb(32, 24, 0x41);
    t.storage = make_cpu_owned_storage(t.storage->size_bytes + 3U);
    t.byte_offset = 3;
    refused(t, "byte_offset");
  }
  {
    // A row stride below the row's bytes.
    Tensor t = rgb(32, 24, 0x42);
    t.strides_bytes = {32 * 3 - 3, 3, 1};
    refused(t, "stride");
  }
  {
    // Another image format.
    Tensor t = rgb(32, 24, 0x43);
    t.semantic.image->format = ImageSpec::PixelFormat::BGR;
    require_neat_error([&] { (void)run.push(TensorList{t}); }, error_codes::kInputShape);
  }
  const Tensor t = rgb(32, 24, 0x44);
  require(run.push(TensorList{t}), "the accepted geometry still passes after the refusals");
  require(bytes_of(pulled_tensor(run, 2000, "after refusals")) == bytes_of(t),
          "bytes after the refusals");
  run.close();
}

void dma_heap_tensor_is_copied() {
  // Core's default tensor: a DMA-BUF from the heap. Mappable, so the push copies it.
  const Tensor t =
      Tensor::from_vector(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, {4}, TensorMemory::EV74);
  Graph g = named_input_output();
  Run run = g.build(dataflow());
  require(run.push(TensorList{t}), "DMA-heap push");
  const Tensor out = pulled_tensor(run, 2000, "DMA-heap");
  require(out.dtype == TensorDType::Float32 && out.shape == std::vector<std::int64_t>{4},
          "dtype and shape");
  require(bytes_of(out) == bytes_of(t), "DMA-heap bytes copied");
  run.close();
}

// A CPU tensor goes into a device input only with the compatibility copy allowed, as on GStreamer
// (SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY, InputStreamPush.cpp): allowed for the scope.
struct CompatCopy {
  CompatCopy() {
    setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 1);
  }
  ~CompatCopy() {
    unsetenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY");
  }
};

void owned_output_and_lease_lifetime() {
  // From a host input and from a device input (H0b) alike.
  const CompatCopy copy;
  for (const InputMemoryPolicy memory :
       {InputMemoryPolicy::SystemMemory, InputMemoryPolicy::Ev74}) {
    RunOptions opt = dataflow();
    opt.output_memory = OutputMemory::Owned;
    InputOptions in;
    in.memory_policy = memory;
    Graph g = input_output(in);
    Run run = g.build(TensorList{rgb(8, 8, 1)}, opt);
    require(run.push(TensorList{rgb(8, 8, 5)}), "push");
    const Tensor out = pulled_tensor(run, 2000, "Owned");
    require(out.storage->kind == StorageKind::CpuOwned, "Owned output is a CPU copy");
    require(bytes_of(out) == bytes_of(rgb(8, 8, 5)), "Owned bytes");
    run.close();
  }
  Tensor kept;
  {
    Graph g = input_output();
    Run run = g.build(TensorList{rgb(16, 8, 1)}, dataflow());
    require(run.push(TensorList{rgb(16, 8, 6)}), "push");
    kept = pulled_tensor(run, 2000, "lease");
    run.close();
  }
  require(bytes_of(kept) == bytes_of(rgb(16, 8, 6)), "a pulled tensor outlives its Run");
}

// ---- H0b: device inputs
// --------------------------------------------------------------------------

InputOptions device_input(InputMemoryPolicy memory) {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.memory_policy = memory;
  return in;
}

// A CPU tensor pushed into a device input is copied into the input's pool: the pulled tensor is
// DMA-BUF memory whose storage is placed on the input's device, its CPU read map gives the pushed
// bytes, and the pushed storage is not kept. (Ev74; Dms0 once.)
void cpu_tensor_into_device_input(InputMemoryPolicy memory, DeviceType device) {
  const CompatCopy copy;
  Graph g = named_input_output(device_input(memory));
  Run run = g.build(dataflow());
  const Tensor pushed = rgb(64, 48, 0x21);
  require(run.push(TensorList{pushed}), "a CPU tensor into a device input");
  const Tensor out = pulled_tensor(run, 2000, "device input");
  require(out.storage->kind == StorageKind::DeviceHandle && sima_test::dmabuf_inode(out) != 0U,
          "the pulled tensor is DMA-BUF memory, the runtime's own");
  require(out.storage->device.type == device, "its storage is placed on the input's device");
  require(pushed.storage.use_count() == 1, "the pushed CPU storage is not kept");
  const std::vector<std::uint8_t> expected = bytes_of(pushed);
  const Mapping map = out.map_read();
  require(map.data != nullptr && map.size_bytes >= expected.size() &&
              std::memcmp(map.data, expected.data(), expected.size()) == 0,
          "its CPU read map gives the pushed bytes");
  run.close();
}

// Run::alloc on a device input lends a slot of the device pool: what the application writes there
// is what the pull hands out, as DMA-BUF memory.
void alloc_on_device_input() {
  InputOptions in;
  in.shape = {4, 8};
  in.dtype = TensorDType::Float32;
  in.memory_policy = InputMemoryPolicy::Ev74;
  Graph g;
  g.add(nodes::Input("x", in));
  g.add(nodes::Output("y"));
  Run run = g.build(dataflow());
  std::vector<float> values(32);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = 100.0F + static_cast<float>(i);
  }
  Tensor t = run.alloc("x");
  {
    Mapping map = t.map_write();
    require(map.data != nullptr && map.size_bytes >= values.size() * sizeof(float),
            "the allocated slot maps for write");
    std::memcpy(map.data, values.data(), values.size() * sizeof(float));
  }
  require(run.push("x", {t}), "push the allocated tensor");
  const auto out = pull_sample(run, 2000, "y");
  require(out && out->tensors.size() == 1U && sima_test::dmabuf_inode(out->tensors[0]) != 0U,
          "the pulled tensor is DMA-BUF memory");
  const std::vector<std::uint8_t> bytes = bytes_of(out->tensors[0]);
  require(bytes.size() == values.size() * sizeof(float) &&
              std::memcmp(bytes.data(), values.data(), bytes.size()) == 0,
          "the pulled bytes are the ones written into the allocation");
  run.close();
}

// A tensor pulled from a device input leases a slot of that input's pool, so pushing it into a
// device input copies it (D15): the next pull is another DMA-BUF with the same bytes. And again
// for the tensor that pull gave: its geometry has passed the input's checks by then, and what a
// repeated geometry skips is not the wrap's own question, whose memory it is.
void pulled_device_tensor_is_copied_back() {
  const CompatCopy copy;
  Graph g = named_input_output(device_input(InputMemoryPolicy::Ev74));
  Run run = g.build(dataflow());
  require(run.push(TensorList{rgb(64, 48, 0x51)}), "push");
  const Tensor first = pulled_tensor(run, 2000, "first");
  require(run.push(TensorList{first}), "push the pulled tensor back");
  const Tensor second = pulled_tensor(run, 2000, "second");
  require(sima_test::dmabuf_inode(second) != 0U &&
              sima_test::dmabuf_inode(second) != sima_test::dmabuf_inode(first),
          "the pulled tensor was copied into another slot");
  require(bytes_of(second) == bytes_of(rgb(64, 48, 0x51)), "bytes of the copy");
  require(run.push(TensorList{second}), "push a pulled tensor of the validated geometry");
  const Tensor third = pulled_tensor(run, 2000, "third");
  require(sima_test::dmabuf_inode(third) != 0U &&
              sima_test::dmabuf_inode(third) != sima_test::dmabuf_inode(second) &&
              sima_test::dmabuf_inode(third) != sima_test::dmabuf_inode(first),
          "a pulled tensor of a validated geometry was copied into another slot");
  require(bytes_of(third) == bytes_of(rgb(64, 48, 0x51)), "bytes of the second copy");
  run.close();
}

// An EV74 tensor pushed into an Ev74 input backs the slot with its own DMA-BUF, which the push
// does not map. OutputMemory::Owned needs the bytes: its pull is a CPU copy of them, read through
// the pushed storage, which is let go with the message.
void owned_pull_of_wrapped_device_tensor() {
  RunOptions opt = dataflow();
  opt.output_memory = OutputMemory::Owned;
  Graph g = named_input_output(device_input(InputMemoryPolicy::Ev74));
  Run run = g.build(opt);
  for (int i = 0; i < 3; ++i) {
    const Tensor pushed = rgb(64, 48, static_cast<std::uint8_t>(0x61 + i)).cvu();
    require(sima_test::dmabuf_inode(pushed) != 0U, "the pushed tensor is DMA-BUF memory");
    require(run.push(TensorList{pushed}), "an EV74 tensor into an Ev74 input");
    const Tensor out = pulled_tensor(run, 2000, "Owned, device tensor");
    require(out.storage->kind == StorageKind::CpuOwned && sima_test::dmabuf_inode(out) == 0U,
            "Owned output of a wrapped device tensor is a CPU copy");
    require(bytes_of(out) == bytes_of(rgb(64, 48, static_cast<std::uint8_t>(0x61 + i))),
            "Owned bytes of a wrapped device tensor");
    require(pushed.storage.use_count() == 1, "the pushed storage is let go with the message");
  }
  run.close();
}

void mat_and_bundle_pushes() {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::BGR;
  Graph g = input_output(in);
  const cv::Mat mat(24, 32, CV_8UC3, cv::Scalar(3, 5, 7));
  Run run = g.build(std::vector<cv::Mat>{mat}, dataflow());
  // The deprecated cv::Mat push fails on Dataflow (deprecated_forms); its replacement takes the
  // Mat.
  require(run.push(
              TensorList{Tensor::from_cv_mat(mat, ImageSpec::PixelFormat::BGR, TensorMemory::CPU)}),
          "cv::Mat push");
  const Tensor out = pulled_tensor(run, 2000, "cv::Mat");
  require(out.shape == std::vector<std::int64_t>{24, 32, 3}, "cv::Mat shape");
  Sample bundle;
  for (int i = 0; i < 3; ++i) {
    Sample field;
    field.kind = SampleKind::Tensor;
    field.tensor = Tensor::from_cv_mat_view(mat, ImageSpec::PixelFormat::BGR);
    field.frame_id = 100 + i;
    bundle.push_back(std::move(field));
  }
  require(run.push(bundle), "a Bundle is pushed one item per field");
  for (int i = 0; i < 3; ++i) {
    const auto s = pull_sample(run, 2000);
    require(s && s->frame_id == 100 + i, "bundle field " + std::to_string(i));
  }
  run.close();
}

} // namespace

RUN_BACKEND_TEST("unit_dataflow_backend_test", ([] {
                   deprecated_forms();
                   // The rows that hold on both backends.
                   linear_realtime_keeps_an_explicit_block();
                   connected_realtime_resolves_the_preset();
                   readiness_follows_close_and_drain();
                   if (!sima_test::on_dataflow()) {
                     return;
                   }
                   options_without_a_counterpart();
                   undeclared_input_bound_by_first_push();
                   declared_input_with_dynamic_axis();
                   push_outside_declared_shape_rejected();
                   build_errors();
                   no_fallback_to_gstreamer();
                   stop_prints_the_session_report();
                   holder_push_is_an_error();
                   link_depth_is_max_of_queue_depth_and_max_buffers();
                   try_push_accounting();
                   refused_try_push_reads_no_byte();
                   blocking_push_never_drops();
                   close_input_cancels_a_parked_push();
                   stop_unblocks_pull_and_refuses_push();
                   concurrent_stops_with_a_parked_push();
                   reads_race_stop();
                   pull_timeouts();
                   identity_round_trip_and_names();
                   nv12_planes_round_trip();
                   repeated_geometry_is_rechecked_on_change();
                   dma_heap_tensor_is_copied();
                   owned_output_and_lease_lifetime();
                   cpu_tensor_into_device_input(InputMemoryPolicy::Ev74, DeviceType::SIMA_CVU);
                   cpu_tensor_into_device_input(InputMemoryPolicy::Dms0, DeviceType::SIMA_MLA);
                   alloc_on_device_input();
                   pulled_device_tensor_is_copied_back();
                   owned_pull_of_wrapped_device_tensor();
                   mat_and_bundle_pushes();
                 }));

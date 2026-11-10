// The Dataflow backend's graphs of several public Inputs and Outputs (H0 step 10, H0c), on the
// public API only: every Run here is built with RunOptions::backend = RunBackend::Dataflow, so the
// test does not depend on SIMA_NEAT_BACKEND, and no runtime type is named.
//
// Covers: two inputs feeding two outputs (named verbs, each input bound by its own first push and
// enforced, the unnamed verbs' "no unambiguous default" errors, unknown names, can_push/can_pull,
// order per output, input_seq per input and orig_input_seq the Run's push counter, close_input of
// every input); fan-out by handle reuse (one push, one slot leased
// by every output: the same bytes and stamp, no copy per output); each Output's own depth and drop
// policy on its link (blocking push waits for room on every link, try_push refused only when every
// link refuses, KeepLatest never refused, Output drop beside a blocking output under Block, the
// mix the runtime cannot serve refused at build); stop with a push blocked on one link of a
// fan-out; per-endpoint pull timeouts; on_output per output; the counters across endpoints; public
// boundaries elided as GStreamer elides them (a Branch helper, connected fragments); seeds; alloc
// on a named input; the errors that name H1 (fan-in, Combine) and a shared endpoint name.

#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

RunOptions dataflow(int queue_depth = 4, OverflowPolicy policy = OverflowPolicy::Block) {
  RunOptions opt;
  opt.backend = RunBackend::Dataflow;
  opt.queue_depth = queue_depth;
  opt.overflow_policy = policy;
  return opt;
}

std::int64_t ms_since(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
}

// An RGB tensor whose bytes encode `tag` and their position.
Tensor rgb(std::uint8_t tag, int w = 32, int h = 24) {
  Tensor t = make_color_tensor(w, h, ImageSpec::PixelFormat::RGB, tag);
  auto map = t.storage->map(MapMode::Write);
  auto* bytes = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < map.size_bytes; ++i) {
    bytes[i] = static_cast<std::uint8_t>(tag + i * 7U);
  }
  return t;
}

Tensor floats(float value) {
  return Tensor::from_vector(std::vector<float>(6, value), {2, 3}, TensorMemory::CPU);
}

Tensor framed(const Tensor& t, std::int64_t frame_id) {
  Stamp proposal;
  proposal.frame_id = frame_id;
  return t.with_stamp(proposal);
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

// One item of `output`, which must be there within `timeout`.
Tensor pull_one(Run& run, const std::string& output, std::chrono::milliseconds timeout,
                const std::string& what) {
  Result<TensorList> r = run.pull(output, timeout);
  require(r.ok(), what + ": pull(\"" + output + "\") status " +
                      std::to_string(static_cast<int>(r.status())) + " " + r.error().message);
  require(r->size() == 1U, what + ": one tensor expected");
  return r->front();
}

// One item of `output` through the Sample surface (pull(name, int, Sample&, PullError*)).
Sample pull_sample(Run& run, const std::string& output, int timeout_ms, const std::string& what) {
  Sample s;
  PullError err;
  const PullStatus st = run.pull(output, timeout_ms, s, &err);
  require(st == PullStatus::Ok, what + ": pull(\"" + output + "\") status " +
                                    std::to_string(static_cast<int>(st)) + " " + err.message);
  return s;
}

PullStatus poll(Run& run, const std::string& output) {
  return run.pull(output, 0ms).status();
}

// Every item queued for `output` now, by frame_id.
std::vector<std::int64_t> drain(Run& run, const std::string& output) {
  std::vector<std::int64_t> frames;
  for (;;) {
    Result<TensorList> r = run.pull(output, 0ms);
    if (!r.ok()) {
      return frames;
    }
    frames.push_back(r->front().stamp().frame_id);
  }
}

std::string frames_text(const std::vector<std::int64_t>& frames) {
  std::string out = "[";
  for (std::size_t i = 0; i < frames.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(frames[i]);
  }
  return out + "]";
}

void require_runtime_error(const std::function<void()>& fn, const std::vector<std::string>& needles,
                           const std::string& what) {
  try {
    fn();
  } catch (const std::exception& e) {
    for (const std::string& needle : needles) {
      require_contains(std::string(e.what()), needle, what);
    }
    return;
  }
  throw std::runtime_error(what + ": expected an exception");
}

// A fan-out of one Input "image" to Outputs "a" and "b", each with its own options.
Graph fan_out(OutputOptions a, OutputOptions b) {
  Graph g;
  g.add(nodes::Input("image"));
  g.add(nodes::Output("a", a));
  g.add(nodes::Output("b", b));
  g.connect("image", "a");
  g.connect("image", "b");
  return g;
}

OutputOptions out_options(int max_buffers, bool drop = false) {
  OutputOptions out;
  out.max_buffers = max_buffers;
  out.drop = drop;
  return out;
}

// Run `fn` on its own thread; past `budget` the process reports and exits (a hang is a failure,
// not a ctest timeout).
template <class T> struct Async {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  T value{};
  std::string error;
};

template <class T> std::shared_ptr<Async<T>> start(std::function<T()> fn) {
  auto state = std::make_shared<Async<T>>();
  std::thread([state, fn = std::move(fn)] {
    T value{};
    std::string error;
    try {
      value = fn();
    } catch (const std::exception& e) {
      error = e.what();
    }
    std::lock_guard<std::mutex> lock(state->mu);
    state->value = value;
    state->error = error;
    state->done = true;
    state->cv.notify_all();
  }).detach();
  return state;
}

template <class T>
bool finished_within(const std::shared_ptr<Async<T>>& state, std::chrono::milliseconds budget) {
  std::unique_lock<std::mutex> lock(state->mu);
  return state->cv.wait_for(lock, budget, [&] { return state->done; });
}

template <class T>
void await(const std::shared_ptr<Async<T>>& state, std::chrono::milliseconds budget,
           const std::string& what) {
  if (!finished_within(state, budget)) {
    std::cerr << "[FAIL] unit_dataflow_endpoints_test: " << what << " did not return within "
              << budget.count() << " ms\n";
    std::_Exit(1);
  }
}

void two_inputs_two_outputs() {
  Graph g;
  const Graph::Handle left = g.input("left");
  const Graph::Handle right = g.input("right");
  g.output("left_out", left);
  g.output("right_out", right);
  Run run = g.build(dataflow());
  require(run.input_names() == std::vector<std::string>({"left", "right"}), "input names");
  require(run.output_names() == std::vector<std::string>({"left_out", "right_out"}),
          "output names");
  require(!run.can_push() && !run.can_pull(), "no default input or output");

  // A pull waits for its own input's first push.
  require(run.pull("left_out", 30ms).status() == PullStatus::Timeout, "pull before any push");
  require(run.push("left", {framed(rgb(1), 10)}), "first push into left binds left");
  require(poll(run, "right_out") == PullStatus::Timeout, "right is not bound by left's push");
  const Tensor l = pull_one(run, "left_out", 2000ms, "left round trip");
  require(bytes_of(l) == bytes_of(rgb(1)) && l.stamp().frame_id == 10, "left bytes and frame_id");
  require(l.stamp().stream_name == "left", "left's default stream is its name");
  require(run.push("right", {floats(2.0F)}), "first push into right binds right");
  const Tensor r = pull_one(run, "right_out", 2000ms, "right round trip");
  require(r.dtype == TensorDType::Float32 && r.shape == std::vector<std::int64_t>({2, 3}),
          "right keeps its own spec");
  require(r.stamp().stream_name == "right", "right's default stream is its name");

  // Each binding is enforced on its own input.
  require_neat_error([&] { (void)run.push("left", {floats(1.0F)}); }, error_codes::kInputShape);
  require_neat_error([&] { (void)run.push("right", {rgb(2)}); }, error_codes::kInputShape);

  // The unnamed verbs and unknown names, with GStreamer's texts.
  require_runtime_error([&] { (void)run.push({rgb(3)}); },
                        {"no unambiguous default input", "Available inputs: [left, right]"},
                        "unnamed push");
  {
    Sample s;
    PullError err;
    require(run.pull(10, s, &err) == PullStatus::Error, "unnamed pull is an error");
    require_contains(err.message, "no unambiguous default output", "unnamed pull");
    require_contains(err.message, "Available outputs: [left_out, right_out]", "unnamed pull");
  }
  require_runtime_error([&] { (void)run.push("middle", {rgb(3)}); },
                        {"unknown input", "Available inputs: [left, right]"}, "unknown input");
  require(run.pull("middle", 10ms).status() == PullStatus::Error, "unknown output");
  require_contains(run.pull("middle", 10ms).error().message, "Available outputs",
                   "unknown output text");

  // Interleaved pushes keep each output's order; input_seq counts per input, orig_input_seq the
  // Run's pushes (GStreamer's graph ingress counter): left 10 was 0, right 1, then 2..7 below.
  for (int i = 0; i < 3; ++i) {
    require(run.push("left", {framed(rgb(static_cast<std::uint8_t>(20 + i)), 20 + i)}),
            "left push " + std::to_string(i));
    require(run.push("right", {framed(floats(static_cast<float>(i)), 30 + i)}),
            "right push " + std::to_string(i));
  }
  require(drain(run, "right_out") == std::vector<std::int64_t>({30, 31, 32}), "right order");
  {
    const Sample s = pull_sample(run, "left_out", 2000, "left Sample");
    require(s.frame_id == 20 && s.input_seq == 1 && s.orig_input_seq == 2 && s.stream_id == "left",
            "left Sample identity: frame_id " + std::to_string(s.frame_id) + ", input_seq " +
                std::to_string(s.input_seq) + ", orig_input_seq " +
                std::to_string(s.orig_input_seq) + ", stream_id " + s.stream_id);
  }
  require(drain(run, "left_out") == std::vector<std::int64_t>({21, 22}), "left order");

  // A pushed orig_input_seq is kept and not counted (GStreamer's rule); the next push counts on.
  {
    Sample s;
    s.kind = SampleKind::Tensor;
    s.tensor = rgb(50);
    s.orig_input_seq = 500;
    require(run.push("left", Sample{s}), "a Sample with its own orig_input_seq");
    require(run.push("right", {floats(5.0F)}), "a counted push");
    require(pull_sample(run, "left_out", 2000, "pushed orig").orig_input_seq == 500,
            "the pushed orig_input_seq is kept");
    require(pull_sample(run, "right_out", 2000, "counted").orig_input_seq == 8,
            "the counter skipped the push that carried its own orig_input_seq");
  }
  // close_input closes every input: what is queued stays, then each output is Closed.
  require(run.push("left", {framed(rgb(40), 40)}), "queued before close_input");
  run.close_input();
  require(!run.push("right", {floats(9.0F)}), "push after close_input");
  require(pull_one(run, "left_out", 2000ms, "queued").stamp().frame_id == 40, "queued item kept");
  require(run.pull("left_out", 2000ms).status() == PullStatus::Closed, "left Closed");
  require(run.pull("right_out", 2000ms).status() == PullStatus::Closed, "right Closed");
  require(run.running(), "a Run of several outputs does not stop itself at a closed output");
  run.close();
}

void fan_out_by_handle_reuse() {
  Graph g;
  const Graph::Handle image = g.input("image");
  g.output("preview", image);
  g.output("model", image);
  Run run = g.build(dataflow());
  require(run.can_push() && !run.can_pull(), "one input, two outputs");
  require(run.output_names() == std::vector<std::string>({"model", "preview"}), "output names");
  const Tensor t = framed(rgb(7), 77);
  require(run.push({t}), "the unnamed push reaches the only input");
  const Tensor a = pull_one(run, "preview", 2000ms, "preview");
  const Tensor b = pull_one(run, "model", 2000ms, "model");
  require(bytes_of(a) == bytes_of(t) && bytes_of(b) == bytes_of(t), "both outputs get the bytes");
  require(a.stamp().frame_id == 77 && b.stamp().frame_id == 77, "both carry the frame_id");
  require(a.stamp().seq == b.stamp().seq, "one message: one seq");
  require(a.storage && b.storage && a.storage->data == b.storage->data,
          "one slot leased by both outputs: no copy per output");
  // The old surface, the same message.
  require(run.push(Sample{[] {
    Sample s;
    s.kind = SampleKind::Tensor;
    s.tensor = rgb(8);
    s.frame_id = 88;
    return s;
  }()}),
          "Sample push");
  const Sample sa = pull_sample(run, "preview", 2000, "preview Sample");
  const Sample sb = pull_sample(run, "model", 2000, "model Sample");
  require(sa.frame_id == 88 && sb.frame_id == 88, "Sample identity on both");
  require(sa.stream_id == "image" && sb.stream_id == "image", "stream_id is the input's name");
  require(sa.input_seq == 1 && sb.input_seq == 1, "input_seq per input");
  run.close();
}

// Blocking push into a fan-out waits for room on every link; each output keeps its own depth.
void fan_out_blocking_push_waits_for_every_link() {
  // queue_depth 1: link a = 1 + 1, link b = 1 + 3.
  Graph g = fan_out(out_options(1), out_options(3));
  Run run = g.build(dataflow(1));
  require(run.push({framed(rgb(1), 1)}) && run.push({framed(rgb(2), 2)}), "two fit both links");
  auto blocked = start<bool>([&run] { return run.push({framed(rgb(3), 3)}); });
  require(!finished_within(blocked, 300ms), "the third push waits: link a is full, b has room");
  require(pull_one(run, "a", 2000ms, "free a").stamp().frame_id == 1, "a's oldest");
  await(blocked, 2000ms, "the push after a pull on a");
  require(blocked->error.empty() && blocked->value, "the waiting push published");
  require(drain(run, "a") == std::vector<std::int64_t>({2, 3}), "a: nothing dropped");
  require(drain(run, "b") == std::vector<std::int64_t>({1, 2, 3}), "b: everything, in order");
  run.close();
}

// try_push into a fan-out: refused only when every link refuses (DropIncoming); a link that took
// it makes it accepted and the other's refusal is that output's drop. Under Block a full link
// refuses the push for all (it never drops). KeepLatest never refuses.
void fan_out_try_push_policies() {
  {
    Graph g = fan_out(out_options(1), out_options(3));
    Run run = g.build(dataflow(1, OverflowPolicy::DropIncoming));
    MeasureScope scope = run.start_measurement();
    std::vector<bool> accepted;
    for (int i = 1; i <= 5; ++i) {
      accepted.push_back(run.try_push({framed(rgb(static_cast<std::uint8_t>(i)), i)}));
    }
    const MeasureCounters c = scope.stop().counters;
    require(accepted == std::vector<bool>({true, true, true, true, false}),
            "DropIncoming: refused only when a and b are both full");
    require(c.inputs_dropped == 1U,
            "one input drop (the fifth), got " + std::to_string(c.inputs_dropped));
    require(c.outputs_dropped == 2U, "a refused the third and fourth while b took them, got " +
                                         std::to_string(c.outputs_dropped));
    require(c.inputs_pushed == 4U && c.outputs_ready == 6U,
            "four items published, six deliveries; got " + std::to_string(c.inputs_pushed) + " / " +
                std::to_string(c.outputs_ready));
    require(drain(run, "a") == std::vector<std::int64_t>({1, 2}), "a keeps the first two");
    require(drain(run, "b") == std::vector<std::int64_t>({1, 2, 3, 4}), "b keeps four");
    run.close();
  }
  {
    Graph g = fan_out(out_options(1), out_options(3));
    Run run = g.build(dataflow(1, OverflowPolicy::Block));
    require(run.try_push({framed(rgb(1), 1)}) && run.try_push({framed(rgb(2), 2)}), "two fit");
    require(!run.try_push({framed(rgb(3), 3)}), "Block: a full link refuses the push for all");
    require(drain(run, "b") == std::vector<std::int64_t>({1, 2}), "b got nothing more");
    run.close();
  }
  {
    Graph g = fan_out(out_options(1), out_options(3));
    Run run = g.build(dataflow(1, OverflowPolicy::KeepLatest));
    for (int i = 1; i <= 6; ++i) {
      require(run.try_push({framed(rgb(static_cast<std::uint8_t>(i)), i)}),
              "KeepLatest never refuses: " + std::to_string(i));
    }
    require(drain(run, "a") == std::vector<std::int64_t>({5, 6}), "a keeps the newest two");
    require(drain(run, "b") == std::vector<std::int64_t>({3, 4, 5, 6}), "b keeps the newest four");
    run.close();
  }
}

// Output drop = true beside a blocking output, under Block: the blocking push evicts on the lossy
// output and waits for the other. Under a dropping try_push policy that mix is refused at build.
void fan_out_mixed_drop() {
  {
    Graph g = fan_out(out_options(1, /*drop=*/true), out_options(2));
    Run run = g.build(dataflow(1));
    for (int i = 1; i <= 3; ++i) {
      require(run.push({framed(rgb(static_cast<std::uint8_t>(i)), i)}),
              "push " + std::to_string(i) + " never waits on the lossy output");
    }
    auto blocked = start<bool>([&run] { return run.push({framed(rgb(4), 4)}); });
    require(!finished_within(blocked, 300ms), "the fourth push waits for b");
    require(pull_one(run, "b", 2000ms, "free b").stamp().frame_id == 1, "b's oldest");
    await(blocked, 2000ms, "the push after a pull on b");
    require(blocked->value, "published");
    require(drain(run, "a") == std::vector<std::int64_t>({3, 4}), "a keeps the newest two");
    require(drain(run, "b") == std::vector<std::int64_t>({2, 3, 4}), "b lost nothing");
    run.close();
  }
  {
    Graph g = fan_out(out_options(1, /*drop=*/true), out_options(2));
    require_neat_error([&] { (void)g.build(dataflow(1, OverflowPolicy::DropIncoming)); },
                       error_codes::kPipelineShape, "OutputOptions::drop");
  }
}

// A push blocked on one link of a fan-out (the other has room) ends with stop, within 1 s.
void stop_with_push_blocked_on_one_link() {
  Graph g = fan_out(out_options(1), out_options(8));
  Run run = g.build(dataflow(1));
  require(run.push({rgb(1)}) && run.push({rgb(2)}), "link a full");
  auto blocked = start<bool>([&run] { return run.push({rgb(3)}); });
  require(!finished_within(blocked, 300ms), "blocked on a while b has room");
  const auto t0 = Clock::now();
  run.stop();
  await(blocked, 1000ms, "the blocked push after stop");
  require(ms_since(t0) < 1000, "stop is bounded");
  require(!blocked->value, "the blocked push is not accepted after stop: " + blocked->error);
  require(!run.push({rgb(4)}), "a push after stop is refused");
  require(run.pull("b", 1000ms).status() == PullStatus::Closed, "b is Closed after stop");
  run.close();
}

// One output's pull times out while another output of the Run has data.
void per_endpoint_pull_timeouts() {
  {
    Graph g = fan_out(out_options(4), out_options(4));
    Run run = g.build(dataflow());
    require(run.push({framed(rgb(1), 1)}), "push");
    (void)pull_one(run, "a", 2000ms, "a");
    const auto t0 = Clock::now();
    const Result<TensorList> r = run.pull("a", 120ms);
    const auto waited = ms_since(t0);
    require(r.status() == PullStatus::Timeout, "a is empty");
    require(waited >= 100 && waited < 1000, "a's timeout is its own: " + std::to_string(waited));
    require_contains(r.error().message, "Run::pull(\"a\")", "the timeout names its output");
    require(poll(run, "b") == PullStatus::Ok, "b still had the message");
    run.close();
  }
  {
    Graph g;
    g.output("left_out", g.input("left"));
    g.output("right_out", g.input("right"));
    Run run = g.build(dataflow());
    require(run.push("left", {rgb(1)}), "left push");
    // right was never pushed: its pull waits for its first push, and times out on its own.
    const auto t0 = Clock::now();
    require(run.pull("right_out", 120ms).status() == PullStatus::Timeout, "right times out");
    require(ms_since(t0) >= 100, "right waited its timeout");
    require(poll(run, "left_out") == PullStatus::Ok, "left was not affected");
    run.close();
  }
}

void on_output_per_endpoint() {
  Graph g = fan_out(out_options(8), out_options(8));
  Run run = g.build(dataflow());
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::int64_t> got_a;
  std::vector<std::int64_t> got_b;
  run.on_output("a", [&](TensorList t) {
    std::lock_guard<std::mutex> lock(mu);
    got_a.push_back(t.front().stamp().frame_id);
    cv.notify_all();
  });
  run.on_output("b", [&](TensorList t) {
    std::lock_guard<std::mutex> lock(mu);
    got_b.push_back(t.front().stamp().frame_id);
    cv.notify_all();
  });
  require(run.pull("a", 10ms).status() == PullStatus::Error, "a belongs to its callback");
  for (int i = 1; i <= 5; ++i) {
    require(run.push({framed(rgb(static_cast<std::uint8_t>(i)), i)}), "push " + std::to_string(i));
  }
  {
    std::unique_lock<std::mutex> lock(mu);
    require(cv.wait_for(lock, 3s, [&] { return got_a.size() == 5U && got_b.size() == 5U; }),
            "each callback receives every item");
    require(got_a == std::vector<std::int64_t>({1, 2, 3, 4, 5}) && got_b == got_a,
            "in order on both: " + frames_text(got_a) + " " + frames_text(got_b));
  }
  run.stop();
  run.close();
}

// Public boundaries inside the graph are elided as GStreamer elides them: a Branch helper's and a
// connected route fragment's Input and Output declarations are not endpoints.
void elided_boundaries() {
  {
    Graph source("image");
    source.add(nodes::Input("image"));
    Graph branch = graphs::Branch("image", {"preview", "model"});
    Graph preview("preview");
    preview.add(nodes::Output("preview"));
    Graph model("model");
    model.add(nodes::Output("model"));
    Graph app;
    app.connect(source, branch);
    app.connect(branch, preview);
    app.connect(branch, model);
    Run run = app.build(dataflow());
    require(run.input_names() == std::vector<std::string>({"image"}), "Branch: one input");
    require(run.output_names() == std::vector<std::string>({"model", "preview"}),
            "Branch: two outputs");
    require(run.push("image", {framed(rgb(5), 5)}), "Branch push");
    const Tensor a = pull_one(run, "preview", 2000ms, "Branch preview");
    const Tensor b = pull_one(run, "model", 2000ms, "Branch model");
    require(a.storage->data == b.storage->data && a.stamp().frame_id == 5, "one message, shared");
    run.close();
  }
  {
    Graph source("image");
    source.add(nodes::Input("raw"));
    Graph route("route");
    route.add(nodes::Input("image"));
    route.add(nodes::Output("classes"));
    route.connect("image", "classes");
    Graph sink("classes");
    sink.add(nodes::Output("classes"));
    Graph app;
    app.connect(source, route);
    app.connect(route, sink);
    Run run = app.build(dataflow());
    require(run.can_push() && run.can_pull(), "a linear chain through elided boundaries");
    require(run.push("raw", {rgb(6)}), "connected fragments push");
    require(bytes_of(pull_one(run, "classes", 2000ms, "connected fragments")) == bytes_of(rgb(6)),
            "connected fragments round trip");
    run.close();
  }
}

void seeds() {
  {
    Graph g = fan_out(out_options(4), out_options(4));
    Run run = g.build(TensorList{rgb(1, 16, 8)}, dataflow());
    require_neat_error([&] { (void)run.push({rgb(1)}); }, error_codes::kInputShape,
                       "out of bounds");
    require(run.push({rgb(2, 16, 8)}), "the seed bound the one input");
    require(poll(run, "a") == PullStatus::Ok && poll(run, "b") == PullStatus::Ok, "both outputs");
    run.close();
  }
  {
    Graph g;
    g.output("left_out", g.input("left"));
    g.output("right_out", g.input("right"));
    require_neat_error([&] { (void)g.build(TensorList{rgb(1)}, dataflow()); },
                       error_codes::kInputShape, "no unambiguous default input");
  }
}

void alloc_on_named_inputs() {
  Graph g;
  g.output("a_out", g.input("a", {2, 4}, TensorDType::UInt8));
  g.output("b_out", g.input("b", {3}, TensorDType::Float32));
  Run run = g.build(dataflow());
  Tensor ta = run.alloc("a");
  Tensor tb = run.alloc("b");
  require(ta.shape == std::vector<std::int64_t>({2, 4}) && tb.dtype == TensorDType::Float32,
          "each input's declaration");
  {
    auto map = ta.storage->map(MapMode::Write);
    std::memset(map.data, 0x5A, map.size_bytes);
  }
  require(run.push("a", {ta}), "commit a's slot");
  const Tensor out = pull_one(run, "a_out", 2000ms, "a's slot");
  require(bytes_of(out) == std::vector<std::uint8_t>(8, 0x5A), "a's bytes");
  require(poll(run, "b_out") == PullStatus::Timeout, "nothing on b");
  run.close();
}

void errors_that_stay_on_gstreamer() {
  // Fan-in: two Inputs into one Output.
  {
    Graph g;
    g.add(nodes::Input("left"));
    g.add(nodes::Input("right"));
    g.add(nodes::Output("combined"));
    g.connect("left", "combined");
    g.connect("right", "combined");
    require_neat_error([&] { (void)g.build(dataflow()); }, error_codes::kPipelineShape,
                       "fan-in is not supported before H1");
  }
  // Combine (a fan-in with a CombinePolicy).
  {
    Graph join = graphs::Combine({"left", "right"}, "combined", CombinePolicy::ByFrame);
    require_neat_error([&] { (void)join.build(dataflow()); }, error_codes::kPipelineShape,
                       "before H1");
  }
  // An Input that feeds nothing.
  {
    Graph g;
    g.add(nodes::Input("left"));
    g.add(nodes::Input("right"));
    g.add(nodes::Output("out"));
    g.connect("left", "out");
    require_neat_error([&] { (void)g.build(dataflow()); }, error_codes::kPipelineShape,
                       "connected to no Output");
  }
  // Two Outputs sharing one public name.
  {
    Graph input("image");
    input.add(nodes::Input());
    Graph a("classes");
    a.add(nodes::Output());
    Graph b("classes");
    b.add(nodes::Output());
    Graph app;
    app.connect(input, a);
    app.connect(input, b);
    require_neat_error([&] { (void)app.build(dataflow()); }, error_codes::kPipelineShape,
                       "used by more than one output");
  }
}

} // namespace

RUN_TEST("unit_dataflow_endpoints_test", ([] {
           two_inputs_two_outputs();
           fan_out_by_handle_reuse();
           fan_out_blocking_push_waits_for_every_link();
           fan_out_try_push_policies();
           fan_out_mixed_drop();
           stop_with_push_blocked_on_one_link();
           per_endpoint_pull_timeouts();
           on_output_per_endpoint();
           elided_boundaries();
           seeds();
           alloc_on_named_inputs();
           errors_that_stay_on_gstreamer();
         }));

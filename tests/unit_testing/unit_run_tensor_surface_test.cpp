// The TensorList surface of H0 (parts C2, C2b: Core's Tensor surface is TensorList-only) on the
// public API, on the process's backend (--backend=gstreamer|dataflow, set_backend). The same
// executable is registered once per backend (tests/CMakeLists.txt).
//
// Covers: Graph::input/output graphs (named endpoints, a Handle from another Graph); stamp fields
// after pull (proposed by a list's first tensor, push and try_push), and the projection of a
// pulled stamp onto the old Sample (orig_input_seq as develop gives it); positional records (a list
// of several tensors, a list of one Tensor::record) pushed and pulled through both surfaces; the
// four Result outcomes and operator(); on_output (order, exclusive output, end at close, a throwing
// callback); a declared input enforced at push; alloc() and its misuse (pushed twice, still mapped,
// pushed into another Run, read like a CPU tensor by try_push, a list of two and a Sample, dropped
// unpushed, pushed after stop, undeclared input, a bounded dynamic axis); counters through
// MeasureScope (H0's stats()).

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;

using sima_test::on_dataflow;

RunOptions options() {
  return RunOptions{};
}

// The old surface's pull of a Sample: the structured form, Run::pull(int, Sample&, PullError*)
// (the deprecated std::optional<Sample> forms fail on Dataflow); nullopt unless Ok.
std::optional<Sample> pull_sample(Run& run, int timeout_ms, const std::string& output = {}) {
  Sample s;
  PullError e;
  const PullStatus st =
      output.empty() ? run.pull(timeout_ms, s, &e) : run.pull(output, timeout_ms, s, &e);
  return st == PullStatus::Ok ? std::optional<Sample>(std::move(s)) : std::nullopt;
}

// An RGB tensor whose bytes encode `tag` and their position.
Tensor rgb(int w, int h, std::uint8_t tag) {
  Tensor t = make_color_tensor(w, h, ImageSpec::PixelFormat::RGB, tag);
  auto map = t.storage->map(MapMode::Write);
  auto* bytes = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < map.size_bytes; ++i) {
    bytes[i] = static_cast<std::uint8_t>(tag + i * 7U);
  }
  return t;
}

Tensor floats(std::vector<float> values, std::vector<std::int64_t> shape) {
  return Tensor::from_vector(values, std::move(shape), TensorMemory::CPU);
}

std::vector<float> iota(std::size_t n, float start) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = start + static_cast<float>(i);
  }
  return out;
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

Stamp proposal(std::int64_t frame_id, std::string stream = {}, std::int64_t source_ns = -1) {
  Stamp s;
  s.frame_id = frame_id;
  s.stream_name = std::move(stream);
  if (source_ns >= 0) {
    s.source_time = TimePoint{source_ns, ClockId::SensorLocal};
  }
  return s;
}

Graph named_graph(Graph::Handle* input = nullptr) {
  Graph g;
  const Graph::Handle in = g.input("image");
  g.output("out", in);
  if (input) {
    *input = in;
  }
  return g;
}

// A tensor-media input of FP32 tensors: what both backends carry as one item of several tensors.
Graph tensor_media_graph() {
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::FP32;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  Graph g;
  g.add(nodes::Input(input));
  g.add(nodes::Output());
  return g;
}

void graph_input_output() {
  Graph::Handle in;
  Graph g = named_graph(&in);
  require(g.inputs() == std::vector<std::string>{"image"} &&
              g.outputs() == std::vector<std::string>{"out"},
          "Graph::input/output declare the endpoints");
  Run run = g.build(options());
  require(run.input_names() == std::vector<std::string>{"image"} &&
              run.output_names() == std::vector<std::string>{"out"},
          "the Run's endpoint names");
  require(run.push("image", {rgb(32, 24, 1)}), "named push");
  require(run.push({rgb(32, 24, 2)}), "unnamed push into the only input");
  const Result<TensorList> a = run.pull("out", 2000ms);
  const Result<TensorList> b = run.pull(2000ms);
  require(a.ok() && b.ok(), "two outputs");
  require(a->size() == 1U && b->size() == 1U, "a leaf is pulled as a list of one");
  require(bytes_of(a->front()) == bytes_of(rgb(32, 24, 1)) &&
              bytes_of(b->front()) == bytes_of(rgb(32, 24, 2)),
          "payloads round trip in order");
  require(a->front().shape == std::vector<std::int64_t>{24, 32, 3} && !a->front().is_record(),
          "a leaf");
  run.close();

  Graph other;
  require_neat_error([&] { other.output("o", in); }, error_codes::kPipelineShape, "another Graph");
  require_neat_error([&] { other.output("o", Graph::Handle{}); }, error_codes::kPipelineShape,
                     "empty");
  require_neat_error([&] { (void)other.input(""); }, error_codes::kPipelineShape, "empty");
  require_neat_error([&] { (void)other.input("v", {0, 4}, TensorDType::Float32); },
                     error_codes::kInputShape, "axis 0");
  require_neat_error([&] { (void)other.input("v", {-1, 4}, TensorDType::Float32, {8}); },
                     error_codes::kInputShape, "rank");
}

void stamp_after_pull() {
  Graph g = named_graph();
  Run run = g.build(options());
  Stamp flagged = proposal(7, "cam-a", 1234);
  flagged.flags = StampFlags::Discontinuity;
  require(run.push("image", {rgb(16, 8, 1).with_stamp(flagged)}), "stamped push");
  require(run.push("image", {rgb(16, 8, 2)}), "unstamped push");
  Sample old;
  old.kind = SampleKind::Tensor;
  old.tensor = rgb(16, 8, 3);
  old.frame_id = 9;
  old.stream_id = "cam-b";
  require(run.push("image", old), "a Sample push");

  const Result<TensorList> first = run.pull("out", 2000ms);
  const Result<TensorList> second = run.pull("out", 2000ms);
  const Result<TensorList> third = run.pull("out", 2000ms);
  require(first.ok() && second.ok() && third.ok(), "three outputs");
  const Stamp& s1 = first->front().stamp();
  const Stamp& s2 = second->front().stamp();
  const Stamp& s3 = third->front().stamp();
  require(s1.frame_id == 7 && s1.stream_name == "cam-a", "frame_id and stream_name proposed");
  require(s2.frame_id == -1 && s2.stream_name == "image",
          "an unstamped push: no frame_id; the endpoint's stream");
  require(s3.frame_id == 9 && s3.stream_name == "cam-b", "a Sample push's identity");
  require(s1.seq == 1 && s2.seq == 2 && s3.seq == 3, "seq: 1-based, contiguous, runtime-written");
  require(s1.stream != 0 && s1.stream != s2.stream && s2.stream != s3.stream &&
              s1.stream != s3.stream,
          "per-Run stream ids, one per name");
  require(s1.source_time.valid(), "source_time carried");
  if (on_dataflow()) {
    require(s1.source_time == (TimePoint{1234, ClockId::SensorLocal}), "source_time exact");
    require(s1.flags == StampFlags::Discontinuity && s2.flags == StampFlags::None, "flags carried");
    require(s1.publish_time.valid() && s1.publish_time.clock == ClockId::Monotonic &&
                s2.publish_time.ns >= s1.publish_time.ns,
            "publish_time written at publish");
  } else {
    require(!s1.publish_time.valid(), "GStreamer records no publish_time");
  }
  require(s1.version == Stamp::kVersion, "stamp version");

  // The old surface's view of the same messages (to_sample, and a pulled Sample): develop's
  // meanings, orig_input_seq = input_seq when no orig_input_seq was pushed, frame_id or not.
  const Sample projected = to_sample(first->front());
  require(projected.frame_id == 7 && projected.stream_id == "cam-a" && projected.input_seq == 0 &&
              projected.orig_input_seq == 0,
          "to_sample: identity");
  require(run.push("image", {rgb(16, 8, 4).with_stamp(proposal(11))}), "push for the old pull");
  const std::optional<Sample> pulled = pull_sample(run, 2000, "out");
  require(pulled && pulled->frame_id == 11 && pulled->input_seq == 3 &&
              pulled->orig_input_seq == pulled->input_seq,
          "the old pull: frame_id, input_seq, and orig_input_seq = input_seq (develop)");
  const Tensor back = to_tensor(*pulled);
  require(back.stamp().frame_id == 11 && back.stamp().seq == 4 && back.stamp().stream == 0,
          "to_tensor: the stamp of a Sample outside a Run");
  // try_push proposes the list's stamp too.
  require(run.try_push("image", {rgb(16, 8, 5).with_stamp(proposal(12, "cam-c"))}),
          "a stamped try_push");
  const Result<TensorList> tried = run.pull("out", 2000ms);
  require(tried.ok() && tried->front().stamp().frame_id == 12 &&
              tried->front().stamp().stream_name == "cam-c",
          "try_push carried frame_id and stream_name");
  run.close();
}

void positional_records() {
  // Two FP32 tensors of the same shape: the item GStreamer's tensor media carries
  // (unit_run_overflow_policy_parity_test's pair); the Dataflow backend also carries fields of
  // different shapes (below).
  const Tensor a = floats(iota(2, 1), {2});
  const Tensor b = floats(iota(2, 10), {2});
  Graph g = tensor_media_graph();
  Run run = g.build(TensorList{a, b}, options());

  const Tensor record = Tensor::record({floats(iota(2, 5), {2}), floats(iota(2, 20), {2})});
  require(record.is_record() && record.fields().size() == 2U &&
              record.field_names()[1] == std::string("1"),
          "Tensor::record is positional");
  require(bytes_of(record["1"]) == bytes_of(floats(iota(2, 20), {2})) &&
              bytes_of(record[FieldId{0}]) == bytes_of(floats(iota(2, 5), {2})),
          "fields by name and by position");
  require(record.leaves().size() == 2U && record.leaves()[1].shape == std::vector<std::int64_t>{2},
          "leaves()");
  bool missing = false;
  try {
    (void)record["2"];
  } catch (const std::out_of_range&) {
    missing = true;
  }
  require(missing, "an unknown field is out_of_range");

  require(run.push({record.with_stamp(proposal(3))}), "a list of one record pushes its fields");
  require(run.push({floats(iota(2, 7), {2}).with_stamp(proposal(4)), floats(iota(2, 30), {2})}),
          "a list of two tensors is a record; its first tensor's stamp");
  require(run.push(TensorList{floats(iota(2, 8), {2}), floats(iota(2, 40), {2})}),
          "a record without a stamp");

  const Result<TensorList> r = run.pull(2000ms);
  require(r.ok() && r->size() == 2U, "a record is pulled as a list of its fields");
  require(bytes_of((*r)[1]) == bytes_of(floats(iota(2, 20), {2})) &&
              bytes_of((*r)[0]) == bytes_of(floats(iota(2, 5), {2})),
          "the fields in order");
  require((*r)[0].stamp().frame_id == 3 && (*r)[1].stamp().frame_id == 3 && !(*r)[0].is_record() &&
              (*r)[1].shape == std::vector<std::int64_t>{2},
          "every field carries the stamp of the record");
  const Result<TensorList> listed = run.pull(2000ms);
  require(listed.ok() && listed->size() == 2U && (*listed)[0].stamp().frame_id == 4 &&
              (*listed)[1].stamp().frame_id == 4 &&
              bytes_of((*listed)[1]) == bytes_of(floats(iota(2, 30), {2})),
          "the list's first tensor proposed the record's stamp");

  const std::optional<Sample> old = pull_sample(run, 2000);
  require(old && old->kind == SampleKind::TensorSet && old->tensors.size() == 2U &&
              bytes_of(old->tensors[1]) == bytes_of(floats(iota(2, 40), {2})),
          "the old pull: a TensorSet of the record's tensors");
  // An unstamped list proposes no identity: its frame_id is whatever the backend gives a push that
  // carries none, the same as for a Sample push without identity (the control below; on an image
  // graph both are -1, unit_run_verb_resolution_test). Printed to stderr (GStreamer's stdio filter
  // reorders stdout).
  {
    Sample control;
    control.kind = SampleKind::TensorSet;
    control.tensors = {floats(iota(2, 9), {2}), floats(iota(2, 50), {2})};
    require(run.push(control), "a Sample push without identity (the control)");
    const std::optional<Sample> pulled = pull_sample(run, 2000);
    require(pulled && pulled->tensors.size() == 2U, "the control's output");
    std::cerr << "[INFO] old pull frame_id: unstamped list push " << old->frame_id
              << ", Sample push without identity " << pulled->frame_id << "\n";
    require((old->frame_id < 0) == (pulled->frame_id < 0),
            "an unstamped list push gets the frame_id of a push without identity");
  }

  auto envelope = std::make_shared<TensorEnvelope>();
  envelope->fields = {a, b};
  envelope->names = {"boxes", "scores"};
  Tensor named;
  named.envelope = envelope;
  require_neat_error([&] { (void)run.push({named}); }, error_codes::kInputShape, "named fields");
  require_neat_error([&] { (void)run.push({record, a}); }, error_codes::kInputShape,
                     "inside a list");
  run.close();

  if (on_dataflow()) {
    // Fields of different shapes and dtypes, bound by the first push (no seed: a graph Run, as a
    // plain linear Run is not built without one).
    Graph h;
    {
      InputOptions input;
      input.payload_type = PayloadType::Tensor;
      input.format = FormatTag::FP32;
      input.memory_policy = InputMemoryPolicy::SystemMemory;
      h.add(nodes::Input("in", input));
      h.add(nodes::Output("out"));
      h.connect("in", "out");
    }
    Run mixed = h.build(options());
    const Tensor u8 =
        Tensor::from_vector(std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}, {2, 3}, TensorMemory::CPU);
    require(mixed.push({floats(iota(4, 0), {4}), u8}), "mixed record push");
    const Result<TensorList> m = mixed.pull(2000ms);
    require(m.ok() && m->size() == 2U && (*m)[1].dtype == TensorDType::UInt8 &&
                (*m)[1].shape == std::vector<std::int64_t>{2, 3} &&
                bytes_of((*m)[1]) == bytes_of(u8) &&
                bytes_of((*m)[0]) == bytes_of(floats(iota(4, 0), {4})),
            "fields of different shapes and dtypes");
    require_neat_error([&] { (void)mixed.push({floats(iota(4, 0), {4})}); },
                       error_codes::kInputShape, "record");
    mixed.close();
  }
}

void result_outcomes() {
  Graph g = named_graph();
  Run run = g.build(options());
  const Result<TensorList> none;
  require(!none.ok() && none.status() == PullStatus::Error, "a default Result is an Error");

  const auto t0 = std::chrono::steady_clock::now();
  const Result<TensorList> timeout = run.pull("out", 30ms);
  require(timeout.status() == PullStatus::Timeout && !timeout,
          "Timeout before any push, status " + std::to_string(static_cast<int>(timeout.status())));
  const auto waited =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
  std::cout << "[INFO] pull(\"out\", 30ms) before any push returned Timeout after "
            << waited.count() << " ms\n";
  if (on_dataflow()) {
    // GStreamer's connected-graph pull before its first push is not bounded by the timeout
    // (develop; c2.md, contract differences); the Dataflow backend's is.
    require(waited < 1000ms, "the timeout is bounded");
  }
  require_neat_error([&] { (void)timeout.value(); }, error_codes::kRuntimePull, "Timeout");

  const Result<TensorList> error = run.pull("nope", 30ms);
  require(error.status() == PullStatus::Error && error.error().code == error_codes::kRuntimePull &&
              error.error().message.find("unknown output") != std::string::npos,
          "Error for an unknown output");

  const TensorList y = run({rgb(8, 8, 5)});
  require(y.size() == 1U && bytes_of(y.front()) == bytes_of(rgb(8, 8, 5)),
          "operator(): push, then the output");

  require(run.push({rgb(8, 8, 6)}), "push before close_input");
  run.close_input();
  const Result<TensorList> last = run.pull("out", 2000ms);
  require(last.ok() && bytes_of(last->front()) == bytes_of(rgb(8, 8, 6)),
          "queued output after close_input");
  const Result<TensorList> closed = run.pull("out", 2000ms);
  require(closed.status() == PullStatus::Closed,
          "Closed after the last output, status " +
              std::to_string(static_cast<int>(closed.status())));
  run.close();
  require(run.pull(10ms).status() == PullStatus::Closed, "a closed Run pulls Closed");
}

void on_output_callbacks() {
  Graph g = named_graph();
  Run run = g.build(options());
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::int64_t> seen;
  run.on_output("out", [&](TensorList t) {
    std::lock_guard<std::mutex> lock(mu);
    seen.push_back(t.size() == 1U ? t.front().stamp().frame_id : -2);
    cv.notify_all();
  });
  require_neat_error([&] { run.on_output("out", [](TensorList) {}); }, error_codes::kPipelineShape,
                     "already");
  require_neat_error([&] { run.on_output("nope", [](TensorList) {}); }, error_codes::kRuntimePull,
                     "unknown output");
  for (int i = 0; i < 5; ++i) {
    require(run.push({rgb(8, 8, static_cast<std::uint8_t>(i)).with_stamp(proposal(100 + i))}),
            "push " + std::to_string(i));
  }
  {
    std::unique_lock<std::mutex> lock(mu);
    require(cv.wait_for(lock, 5s, [&] { return seen.size() == 5U; }), "five callbacks");
    for (int i = 0; i < 5; ++i) {
      require(seen[static_cast<std::size_t>(i)] == 100 + i, "delivered in order");
    }
  }
  require(run.pull("out", 10ms).status() == PullStatus::Error,
          "an output with a callback is not pulled (new verb)");
  PullError err;
  Sample sample;
  require(run.pull("out", 10, sample, &err) == PullStatus::Error &&
              err.message.find("on_output") != std::string::npos,
          "nor by the old verbs");
  run.close_input();
  const auto t0 = std::chrono::steady_clock::now();
  run.stop();
  require(std::chrono::steady_clock::now() - t0 < 3s, "stop joins the delivery promptly");
  run.close();

  // A callback that throws ends the delivery and is reported by last_error().
  Graph g2 = named_graph();
  Run run2 = g2.build(options());
  std::atomic<int> calls{0};
  run2.on_output([&](TensorList) {
    ++calls;
    throw std::runtime_error("boom");
  });
  require(run2.push({rgb(8, 8, 1)}) && run2.push({rgb(8, 8, 2)}), "pushes");
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (run2.last_error().find("boom") == std::string::npos &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  require(run2.last_error().find("callback threw: boom") != std::string::npos,
          "last_error reports the callback's exception: " + run2.last_error());
  std::this_thread::sleep_for(50ms);
  require(calls.load() == 1, "the delivery ended at the exception");
  run2.close();
}

void declared_input_enforced_and_alloc() {
  Graph g;
  const auto x = g.input("x", {4, 8}, TensorDType::Float32);
  g.output("y", x);
  Run run = g.build(options());

  // Enforced at push, on both backends.
  require_neat_error([&] { (void)run.push("x", {floats(iota(36, 0), {4, 9})}); },
                     error_codes::kInputShape, "not the declared size");
  require_neat_error([&] { (void)run.push("x", {floats(iota(32, 0), {32})}); },
                     error_codes::kInputShape, "rank");
  require_neat_error(
      [&] {
        (void)run.push("x", {Tensor::from_vector(std::vector<std::int32_t>(32, 1), {4, 8},
                                                 TensorMemory::CPU)});
      },
      error_codes::kInputShape, "dtype");
  require(run.push("x", {floats(iota(32, 0), {4, 8})}), "a fitting push");
  {
    const Result<TensorList> fitting = run.pull("y", 2000ms);
    require(fitting.ok() && bytes_of(fitting->front()) == bytes_of(floats(iota(32, 0), {4, 8})),
            "the fitting push round trips");
  }

  // alloc: the declared shape, writable; pushing it alone in a list is its commit.
  Tensor t = run.alloc("x");
  require(t.dtype == TensorDType::Float32 && t.shape == std::vector<std::int64_t>{4, 8} &&
              !t.read_only,
          "alloc gives the declared spec, writable");
  {
    Mapping map = t.map_write();
    require(map.data != nullptr && map.size_bytes >= 32U * sizeof(float), "mapped for write");
    const std::vector<float> values = iota(32, 100);
    std::memcpy(map.data, values.data(), 32U * sizeof(float));
    require_neat_error([&] { (void)run.push("x", {t}); }, error_codes::kInputShape, "still mapped");
  }
  require(run.push("x", {t}), "push the allocated tensor");
  require(t.map_read().data == nullptr, "after the push the tensor is empty");
  require_neat_error([&] { (void)run.push("x", {t}); }, error_codes::kInputShape, "pushed already");
  {
    const Result<TensorList> out = run.pull("y", 2000ms);
    require(out.ok() && bytes_of(out->front()) == bytes_of(floats(iota(32, 100), {4, 8})),
            "the allocated tensor's bytes arrive");
  }

  // Dropped unpushed, many times over: every slot comes back (on Dataflow the pool is a few
  // slots; alloc would wait forever otherwise).
  auto drops = std::async(std::launch::async, [&] {
    for (int i = 0; i < 40; ++i) {
      Tensor dropped = run.alloc("x");
      (void)dropped;
    }
  });
  if (drops.wait_for(10s) != std::future_status::ready) {
    run.stop(); // ends an alloc waiting for a slot, so the future can be joined
    drops.wait();
    throw std::runtime_error("allocations dropped unpushed did not return their slots");
  }
  drops.get();

  // Pushed into another Run: that Run reads it like any CPU tensor; it stays this Run's to push.
  {
    Graph g2;
    const auto x2 = g2.input("x", {4, 8}, TensorDType::Float32);
    g2.output("y", x2);
    Run other = g2.build(options());
    Tensor shared = run.alloc("x");
    {
      Mapping map = shared.map_write();
      const std::vector<float> values = iota(32, 500);
      std::memcpy(map.data, values.data(), 32U * sizeof(float));
    }
    require(other.push("x", {shared}), "push into another Run");
    {
      const Result<TensorList> copied = other.pull("y", 2000ms);
      require(copied.ok() && bytes_of(copied->front()) == bytes_of(floats(iota(32, 500), {4, 8})),
              "the other Run's output");
    }
    require(run.push("x", {shared}), "then into its own Run");
    const Result<TensorList> own = run.pull("y", 2000ms);
    require(own.ok() && bytes_of(own->front()) == bytes_of(floats(iota(32, 500), {4, 8})),
            "its own output");
    other.close();
  }

  // Every other push reads it like any CPU tensor and leaves it writable: try_push, a Sample, and a
  // list of two (refused here by the declaration before anything is read); then push({t}) commits.
  {
    Tensor listed = run.alloc("x");
    {
      Mapping map = listed.map_write();
      const std::vector<float> values = iota(32, 900);
      std::memcpy(map.data, values.data(), 32U * sizeof(float));
    }
    require_neat_error([&] { (void)run.push("x", {listed, listed}); }, error_codes::kInputShape,
                       "tensors");
    require(listed.map_read().data != nullptr, "a refused list of two leaves it writable");
    require(run.try_push("x", {listed}), "try_push pushes a copy");
    {
      const Result<TensorList> tried = run.pull("y", 2000ms);
      require(tried.ok() && bytes_of(tried->front()) == bytes_of(floats(iota(32, 900), {4, 8})),
              "try_push's output");
    }
    require(listed.map_read().data != nullptr, "try_push leaves it writable");
    Sample sample;
    sample.kind = SampleKind::Tensor;
    sample.tensor = listed;
    require(run.push("x", sample), "a Sample push pushes a copy");
    {
      const Result<TensorList> sampled = run.pull("y", 2000ms);
      require(sampled.ok() && bytes_of(sampled->front()) == bytes_of(floats(iota(32, 900), {4, 8})),
              "the Sample push's output");
    }
    sample = Sample{};
    require(listed.map_read().data != nullptr, "a Sample push leaves it writable");
    require(run.push("x", {listed}), "then push({t}) commits it");
    const Result<TensorList> committed = run.pull("y", 2000ms);
    require(committed.ok() &&
                bytes_of(committed->front()) == bytes_of(floats(iota(32, 900), {4, 8})),
            "the committed output");
    require(listed.map_read().data == nullptr, "committed: empty");
  }

  // Errors: an undeclared input, an unknown one.
  Graph g3 = named_graph();
  Run undeclared = g3.build(options());
  require_neat_error([&] { (void)undeclared.alloc("image"); }, error_codes::kInputShape,
                     "declares no shape");
  bool unknown = false;
  try {
    (void)run.alloc("nope");
  } catch (const std::exception& e) {
    unknown = std::string(e.what()).find("unknown input") != std::string::npos;
  }
  require(unknown, "alloc of an unknown input");
  undeclared.close();

  // Pushed after stop: refused, and the tensor stays writable; dropped after its Run is gone.
  Tensor late = run.alloc("x");
  run.stop();
  require(!run.push("x", {late}), "a push after stop is refused");
  require(late.map_read().data != nullptr, "a refused push leaves it writable");
  run.close();
  late = Tensor{};
}

void bounded_dynamic_axis() {
  Graph g;
  const auto v = g.input("v", {-1, 4}, TensorDType::Float32, {10, 4});
  g.output("w", v);
  Run run = g.build(options());
  Tensor t = run.alloc("v");
  require(t.shape == std::vector<std::int64_t>{10, 4}, "alloc gives the bound");
  t.shape = {3, 4};
  {
    Mapping map = t.map_write();
    const std::vector<float> values = iota(12, 1);
    std::memcpy(map.data, values.data(), 12U * sizeof(float));
  }
  require(run.push({t}), "push 3 of 10 rows (unnamed, the only input: commits it too)");
  require(t.map_read().data == nullptr, "committed by the unnamed push: empty");
  const Result<TensorList> out = run.pull("w", 2000ms);
  require(out.ok() && out->front().shape == std::vector<std::int64_t>{3, 4} &&
              bytes_of(out->front()) == bytes_of(floats(iota(12, 1), {3, 4})),
          "the pushed extent arrives");
  require_neat_error([&] { (void)run.push("v", {floats(iota(44, 0), {11, 4})}); },
                     error_codes::kInputShape, "outside its bound");
  run.close();

  // A dynamic axis without a bound: GStreamer accepts it (alloc does not); Dataflow cannot size
  // its pool and refuses the build.
  Graph open;
  const auto u = open.input("u", {-1, 4}, TensorDType::Float32);
  open.output("o", u);
  if (on_dataflow()) {
    require_neat_error([&] { (void)open.build(options()); }, error_codes::kInputShape,
                       "without a bound");
  } else {
    Run unbounded = open.build(options());
    require_neat_error([&] { (void)unbounded.alloc("u"); }, error_codes::kInputShape,
                       "without a bound");
    unbounded.close();
  }
}

// H0's stats(): MeasureScope's counters (no public Run::stats(), plan 3.3), on both backends.
void statistics() {
  Graph g = named_graph();
  Run run = g.build(options());
  MeasureOptions measure;
  measure.duration_ms = 1000; // MeasureScope requires a window; stop() ends it early
  measure.warmup_ms = 0;
  MeasureScope scope = run.start_measurement(measure);
  for (int i = 0; i < 3; ++i) {
    require(run.push({rgb(8, 8, static_cast<std::uint8_t>(i))}), "push");
  }
  for (int i = 0; i < 3; ++i) {
    require(run.pull("out", 2000ms).ok(), "pull");
  }
  const MeasureReport report = scope.stop();
  require(report.counters.outputs_pulled == 3U,
          "outputs_pulled " + std::to_string(report.counters.outputs_pulled));
  require(report.counters.inputs_dropped == 0U && report.counters.outputs_dropped == 0U,
          "nothing dropped");
  run.close();
}

} // namespace

RUN_BACKEND_TEST("unit_run_tensor_surface_test", ([] {
                   std::cout << "[INFO] backend under test: "
                             << (on_dataflow() ? "dataflow" : "gstreamer") << "\n";
                   graph_input_output();
                   stamp_after_pull();
                   positional_records();
                   result_outcomes();
                   on_output_callbacks();
                   declared_input_enforced_and_alloc();
                   bounded_dynamic_axis();
                   statistics();
                 }));

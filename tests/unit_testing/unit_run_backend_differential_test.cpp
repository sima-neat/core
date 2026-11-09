// Differential test (H0 3.10, part C4): one fixed input sequence through the same graph on both
// backends (RunBackend::GStreamer and RunBackend::Dataflow, requested explicitly, in one process),
// with and without frame_id / stream_id / pts_ns set and with copy_input on and off; the pulled
// items are compared field by field and byte by byte. SIMA_NEAT_BACKEND does not matter here.
//
// Compared (APPLICATION_MODEL.md 1.2: what applications read, plus what they echo):
//   the pull status; Sample::kind and its payload slots (tensor / tensors / fields, and how many);
//   per tensor: dtype, shape, image format and every byte (also against what was pushed);
//   frame_id (C0: -1 for push(TensorList) on GStreamer, so input_seq is the sequence key);
//   stream_id; input_seq; pts_ns when InputOptions::do_timestamp is false; orig_input_seq for the
//   items that push none; media_type and payload_tag (C3 shapes the Dataflow Sample as
//   GStreamer's); on the new surface (Run::pull(milliseconds) -> Result<Tensor>): is_record, the
//   leaf's dtype, shape, format and bytes, Stamp::frame_id, Stamp::stream_name, Stamp::seq, and
//   Stamp::source_time (ns and clock) when do_timestamp is false; the Closed status after
//   close_input and the drained queue.
// Excluded explicitly, each with its reason (never silently); the Dataflow side is still checked
// to hold the documented "not carried" value, and both values are printed:
//   - pts_ns / Stamp::source_time when do_timestamp is true: GStreamer's appsrc stamps running time
//     over the pushed value; Dataflow keeps the pushed value (c3.md section 4 row 4, c2.md
//     section 4 row 2).
//   - stream_label, dts_ns, duration_ns, caps_string, attributes: GStreamer fills them from caps
//     and buffer meta; Dataflow has no caps and no meta map (c3.md section 4 row 5,
//     APPLICATION_MODEL 2.1). Dataflow: empty / -1.
//   - orig_input_seq of an item pushed with orig_input_seq set: GStreamer keeps the pushed value;
//     the runtime's Stamp has no field for it, so Dataflow gives input_seq (c2.md section 3.6,
//     section 4 row 5). On GStreamer's graph path (the connected variants) the items after it lag
//     by one as well: its ingress counter does not advance for an item that carried its own
//     orig_input_seq (RunPush.cpp:278-287).
//   - Stamp::publish_time (GStreamer: unset; Dataflow: the publish instant, Monotonic);
//     Stamp::flags (not carried by GStreamer's Sample); the numeric Stamp::stream (per Run,
//     assigned in a different order; the name is the identity): c2.md section 4 rows 1, 3, 4.
// Shapes vary only under RunPreset::Realtime: on Balanced (the default) and Reliable, GStreamer's
// linear ingress holds a frame of a new shape until stability_frames (2) consecutive frames agree
// before it renegotiates (InputStreamCaps.cpp:257, preset_default_stability_frames), so a lockstep
// pull of one changed frame times out there, while Dataflow has nothing to renegotiate and delivers
// it; Realtime's stability is 1. The Balanced variants push one shape throughout.
// The seeded linear graph builds with RunOptions::startup_preflight = false on both backends: with
// the preflight, GStreamer pushes the seed as input_seq 0 and the first pushed item is 1, while
// Dataflow only binds the seed (c2.md section 4 row 6). No pull happens before the first push
// (GStreamer's connected graph takes 5000 ms there for any timeout, c2.md section 4 row 7).

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;

constexpr int kMaxW = 96;
constexpr int kMaxH = 64;

enum class Verb { List, Sample, Tensor };

// One item of the fixed sequence.
struct Item {
  Verb verb;
  int w;
  int h;
  std::int64_t frame_id; // -1: not set
  std::string stream;    // empty: not set
  std::int64_t pts_ns;   // -1: not set
  std::int64_t orig_seq; // -1: not set (Sample verb only)
};

std::vector<Item> sequence(bool vary_shapes) {
  std::vector<Item> items = {
      {Verb::List, 64, 48, -1, "", -1, -1},             // no identity at all
      {Verb::Sample, 32, 16, 11, "cam-a", 1001000, -1}, // all three set
      {Verb::Sample, 96, 64, -1, "", -1, -1},           // a Sample with none set
      {Verb::Tensor, 64, 48, 13, "cam-b", 3003000, -1}, // the new verb, with a stamp proposal
      {Verb::Sample, 48, 32, 14, "", -1, -1},           // frame_id only
      {Verb::Sample, 96, 64, -1, "cam-a", -1, 500},     // stream only, and an orig_input_seq
      {Verb::Tensor, 16, 8, -1, "", -1, -1},            // the new verb, no stamp
      {Verb::Sample, 64, 48, -1, "", 7007000, -1},      // pts only
  };
  if (!vary_shapes) {
    for (Item& item : items) {
      item.w = 64;
      item.h = 48;
    }
  }
  return items;
}

Tensor rgb(int w, int h, std::uint8_t tag) {
  Tensor t = make_color_tensor(w, h, ImageSpec::PixelFormat::RGB, tag);
  auto map = t.storage->map(MapMode::Write);
  auto* bytes = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < map.size_bytes; ++i) {
    bytes[i] = static_cast<std::uint8_t>(tag * 31U + i * 7U);
  }
  return t;
}

struct Variant {
  std::string name;
  RunPreset preset;
  bool connected;   // Graph::connect of named Input/Output fragments (seedless), else linear seeded
  bool vary_shapes; // shapes within the declared bounds, else 64x48 throughout
  bool copy_input;
  bool do_timestamp;
  bool new_pull; // Run::pull(milliseconds) -> Result<Tensor>, else Run::pull(int, Sample&)
};

struct TensorObs {
  TensorDType dtype = TensorDType::UInt8;
  std::vector<std::int64_t> shape;
  int format = -1;
  std::vector<std::uint8_t> bytes;
};

struct Observed {
  PullStatus status = PullStatus::Error;
  std::string error;
  // Old surface.
  SampleKind kind = SampleKind::Unknown;
  bool has_tensor = false;
  std::size_t tensors = 0;
  std::size_t fields = 0;
  std::int64_t frame_id = -1, pts_ns = -1, dts_ns = -1, duration_ns = -1;
  std::int64_t input_seq = -1, orig_input_seq = -1;
  std::string stream_id, stream_label, media_type, payload_tag, caps_string;
  std::size_t attributes = 0;
  // New surface.
  bool is_record = false;
  Stamp stamp;
  // Both.
  std::vector<TensorObs> leaves;
};

TensorObs observe(const Tensor& t) {
  TensorObs o;
  o.dtype = t.dtype;
  o.shape = t.shape;
  o.format = t.semantic.image ? static_cast<int>(t.semantic.image->format) : -1;
  o.bytes = t.copy_dense_bytes_tight();
  return o;
}

Run build(const Variant& v, RunBackend backend) {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.memory_policy = InputMemoryPolicy::SystemMemory;
  in.max_width = kMaxW;
  in.max_height = kMaxH;
  in.max_depth = 3;
  in.do_timestamp = v.do_timestamp;
  OutputOptions out;
  out.max_buffers = 16;
  out.drop = false;
  out.sync = false;
  RunOptions opt;
  opt.preset = v.preset;
  opt.backend = backend;
  opt.queue_depth = 16;
  opt.overflow_policy = OverflowPolicy::Block;
  opt.startup_preflight = false;
  opt.advanced.copy_input = v.copy_input;
  if (v.connected) {
    Graph in_graph;
    in_graph.add(nodes::Input("image", in));
    Graph out_graph;
    out_graph.add(nodes::Output("out", out));
    Graph g;
    g.connect(in_graph, out_graph);
    return g.build(opt);
  }
  Graph g;
  g.add(nodes::Input(in));
  g.add(nodes::Output(out));
  return g.build(TensorList{rgb(64, 48, 0xEE)}, opt);
}

bool push_item(Run& run, const Variant& v, const Item& item, const Tensor& t) {
  switch (item.verb) {
  case Verb::List:
    return v.connected ? run.push("image", TensorList{t}) : run.push(TensorList{t});
  case Verb::Sample: {
    Sample s;
    s.kind = SampleKind::Tensor;
    s.tensor = t;
    s.frame_id = item.frame_id;
    s.stream_id = item.stream;
    s.pts_ns = item.pts_ns;
    s.orig_input_seq = item.orig_seq;
    return v.connected ? run.push("image", Sample{s}) : run.push(Sample{s});
  }
  case Verb::Tensor: {
    Stamp proposal;
    proposal.frame_id = item.frame_id;
    proposal.stream_name = item.stream;
    if (item.pts_ns >= 0) {
      proposal.source_time = TimePoint{item.pts_ns, ClockId::SensorLocal};
    }
    const Tensor stamped =
        item.frame_id >= 0 || !item.stream.empty() || item.pts_ns >= 0 ? t.with_stamp(proposal) : t;
    return v.connected ? run.push("image", stamped) : run.push(stamped);
  }
  }
  return false;
}

Observed pull_one(Run& run, const Variant& v) {
  Observed o;
  if (v.new_pull) {
    Result<Tensor> r = v.connected ? run.pull("out", 5000ms) : run.pull(5000ms);
    o.status = r.status();
    if (!r.ok()) {
      o.error = r.error().message;
      return o;
    }
    const Tensor& t = *r;
    o.is_record = t.is_record();
    o.stamp = t.stamp();
    for (const Tensor& leaf : t.leaves()) {
      o.leaves.push_back(observe(leaf));
    }
    return o;
  }
  Sample s;
  PullError err;
  o.status = v.connected ? run.pull("out", 5000, s, &err) : run.pull(5000, s, &err);
  o.error = err.message;
  if (o.status != PullStatus::Ok) {
    return o;
  }
  o.kind = s.kind;
  o.has_tensor = s.tensor.has_value();
  o.tensors = s.tensors.size();
  o.fields = s.fields.size();
  o.frame_id = s.frame_id;
  o.pts_ns = s.pts_ns;
  o.dts_ns = s.dts_ns;
  o.duration_ns = s.duration_ns;
  o.input_seq = s.input_seq;
  o.orig_input_seq = s.orig_input_seq;
  o.stream_id = s.stream_id;
  o.stream_label = s.stream_label;
  o.media_type = s.media_type;
  o.payload_tag = s.payload_tag;
  o.caps_string = s.caps_string;
  o.attributes = s.attributes.size();
  for (const Tensor& t : tensors_from_sample(s, true)) {
    o.leaves.push_back(observe(t));
  }
  return o;
}

struct Outcome {
  std::vector<Observed> items;
  std::vector<std::vector<std::uint8_t>> pushed;
  PullStatus final_status = PullStatus::Error;
  std::string final_error;
  std::string build_error;
};

Outcome run_variant(const Variant& v, RunBackend backend) {
  Outcome out;
  try {
    Run run = build(v, backend);
    const std::vector<Item> items = sequence(v.vary_shapes);
    std::vector<Tensor> sources; // kept unchanged and alive until pulled (copy_input off)
    for (std::size_t i = 0; i < items.size(); ++i) {
      sources.push_back(rgb(items[i].w, items[i].h, static_cast<std::uint8_t>(0x10 + i)));
      out.pushed.push_back(sources.back().copy_dense_bytes_tight());
      require(push_item(run, v, items[i], sources.back()),
              "push of item " + std::to_string(i) + " refused");
      out.items.push_back(pull_one(run, v));
    }
    run.close_input();
    if (v.new_pull) {
      Result<Tensor> r = v.connected ? run.pull("out", 5000ms) : run.pull(5000ms);
      out.final_status = r.status();
      out.final_error = r.ok() ? std::string{} : r.error().message;
    } else {
      Sample s;
      PullError err;
      out.final_status = v.connected ? run.pull("out", 5000, s, &err) : run.pull(5000, s, &err);
      out.final_error = err.message;
    }
    run.close();
  } catch (const std::exception& e) {
    out.build_error = e.what();
  }
  return out;
}

const char* status_name(PullStatus s) {
  switch (s) {
  case PullStatus::Ok:
    return "Ok";
  case PullStatus::Timeout:
    return "Timeout";
  case PullStatus::Closed:
    return "Closed";
  case PullStatus::Error:
    return "Error";
  }
  return "?";
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
  std::string s;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    s += (i ? "x" : "") + std::to_string(shape[i]);
  }
  return s;
}

class Comparison {
public:
  explicit Comparison(std::string variant) : variant_(std::move(variant)) {}

  template <class T> void equal(std::size_t item, const char* field, const T& gst, const T& df) {
    ++compared_;
    if (!(gst == df)) {
      std::ostringstream os;
      os << variant_ << " item " << item << ": " << field << " differs: gstreamer=" << gst
         << " dataflow=" << df;
      failures_.push_back(os.str());
    }
  }
  void check(std::size_t item, const char* what, bool ok, const std::string& detail = {}) {
    ++compared_;
    if (!ok) {
      failures_.push_back(variant_ + " item " + std::to_string(item) + ": " + what +
                          (detail.empty() ? "" : " (" + detail + ")"));
    }
  }
  void excluded(std::size_t item, const char* field, const std::string& gst, const std::string& df,
                const char* reason) {
    std::cout << "  [EXCLUDED] " << variant_ << " item " << item << " " << field
              << ": gstreamer=" << gst << " dataflow=" << df << " (" << reason << ")\n";
  }
  const std::vector<std::string>& failures() const {
    return failures_;
  }
  std::size_t compared() const {
    return compared_;
  }

private:
  std::string variant_;
  std::vector<std::string> failures_;
  std::size_t compared_ = 0;
};

void print_row(const char* backend, std::size_t i, const Observed& o, bool new_pull) {
  std::cout << "  " << backend << " item " << i << ": status=" << status_name(o.status);
  if (o.status != PullStatus::Ok) {
    std::cout << " error=" << o.error << "\n";
    return;
  }
  if (new_pull) {
    std::cout << " record=" << o.is_record << " frame_id=" << o.stamp.frame_id << " stream_name='"
              << o.stamp.stream_name << "' stream=" << o.stamp.stream << " seq=" << o.stamp.seq
              << " source_time=" << o.stamp.source_time.ns << "/"
              << static_cast<int>(o.stamp.source_time.clock)
              << " publish_time_valid=" << o.stamp.publish_time.valid()
              << " flags=" << static_cast<int>(o.stamp.flags);
  } else {
    std::cout << " kind=" << static_cast<int>(o.kind) << " tensor=" << o.has_tensor
              << " tensors=" << o.tensors << " fields=" << o.fields << " frame_id=" << o.frame_id
              << " stream_id='" << o.stream_id << "' pts=" << o.pts_ns
              << " input_seq=" << o.input_seq << " orig_input_seq=" << o.orig_input_seq
              << " media_type='" << o.media_type << "' payload_tag='" << o.payload_tag
              << "' stream_label='" << o.stream_label << "' dts=" << o.dts_ns
              << " duration=" << o.duration_ns << " caps='" << o.caps_string
              << "' attributes=" << o.attributes;
  }
  for (const TensorObs& t : o.leaves) {
    std::cout << " [" << shape_text(t.shape) << " dtype=" << static_cast<int>(t.dtype)
              << " fmt=" << t.format << " " << t.bytes.size() << "B]";
  }
  std::cout << "\n";
}

void compare(const Variant& v, const Outcome& gst, const Outcome& df, Comparison& c) {
  c.check(0, "GStreamer run completes", gst.build_error.empty(), gst.build_error);
  c.check(0, "Dataflow run completes", df.build_error.empty(), df.build_error);
  if (!gst.build_error.empty() || !df.build_error.empty()) {
    return;
  }
  const std::vector<Item> items = sequence(v.vary_shapes);
  bool pushed_orig_seen = false;
  c.equal<std::size_t>(0, "items pulled", gst.items.size(), df.items.size());
  for (std::size_t i = 0; i < items.size() && i < gst.items.size() && i < df.items.size(); ++i) {
    const Item& item = items[i];
    const Observed& g = gst.items[i];
    const Observed& d = df.items[i];
    print_row("gstreamer", i, g, v.new_pull);
    print_row("dataflow ", i, d, v.new_pull);
    c.equal<std::string>(i, "status", status_name(g.status), status_name(d.status));
    if (g.status != PullStatus::Ok || d.status != PullStatus::Ok) {
      continue;
    }
    // Payload: the same leaves, each equal to what was pushed.
    c.equal<std::size_t>(i, "leaf count", g.leaves.size(), d.leaves.size());
    for (std::size_t k = 0; k < g.leaves.size() && k < d.leaves.size(); ++k) {
      c.equal<int>(i, "dtype", static_cast<int>(g.leaves[k].dtype),
                   static_cast<int>(d.leaves[k].dtype));
      c.equal<std::string>(i, "shape", shape_text(g.leaves[k].shape),
                           shape_text(d.leaves[k].shape));
      c.equal<int>(i, "image format", g.leaves[k].format, d.leaves[k].format);
      c.check(i, "bytes equal between backends", g.leaves[k].bytes == d.leaves[k].bytes);
      c.check(i, "GStreamer bytes equal the pushed bytes", g.leaves[k].bytes == gst.pushed[i]);
      c.check(i, "Dataflow bytes equal the pushed bytes", d.leaves[k].bytes == df.pushed[i]);
    }
    const bool proposes_time = item.pts_ns >= 0;
    if (v.new_pull) {
      c.equal<int>(i, "is_record", g.is_record, d.is_record);
      c.equal<std::int64_t>(i, "Stamp::frame_id", g.stamp.frame_id, d.stamp.frame_id);
      c.equal<std::int64_t>(i, "Stamp::frame_id vs pushed", item.frame_id, d.stamp.frame_id);
      c.equal<std::string>(i, "Stamp::stream_name", g.stamp.stream_name, d.stamp.stream_name);
      c.equal<std::uint64_t>(i, "Stamp::seq", g.stamp.seq, d.stamp.seq);
      if (!v.do_timestamp) {
        c.equal<std::int64_t>(i, "Stamp::source_time.ns", g.stamp.source_time.ns,
                              d.stamp.source_time.ns);
        c.equal<int>(i, "Stamp::source_time.clock", static_cast<int>(g.stamp.source_time.clock),
                     static_cast<int>(d.stamp.source_time.clock));
        c.equal<std::int64_t>(i, "Stamp::source_time.ns vs pushed",
                              proposes_time ? item.pts_ns : -1,
                              d.stamp.source_time.valid() ? d.stamp.source_time.ns : -1);
      } else {
        c.excluded(i, "Stamp::source_time", std::to_string(g.stamp.source_time.ns),
                   std::to_string(d.stamp.source_time.ns),
                   "do_timestamp: GStreamer stamps running time, c3.md 4 row 4");
      }
      c.excluded(i, "Stamp::publish_time", std::to_string(g.stamp.publish_time.ns),
                 std::to_string(d.stamp.publish_time.ns),
                 "GStreamer has no publish time, c2.md 4 row 1");
      c.check(i, "Dataflow Stamp::publish_time is set (Monotonic)",
              d.stamp.publish_time.valid() && d.stamp.publish_time.clock == ClockId::Monotonic);
      c.excluded(i, "Stamp::flags", std::to_string(static_cast<int>(g.stamp.flags)),
                 std::to_string(static_cast<int>(d.stamp.flags)),
                 "not carried by GStreamer's Sample, c2.md 4 row 3");
      c.excluded(i, "Stamp::stream (numeric)", std::to_string(g.stamp.stream),
                 std::to_string(d.stamp.stream),
                 "per-Run id, assigned in another order; the name is compared, c2.md 4 row 4");
      continue;
    }
    // Old surface: APPLICATION_MODEL 1.2's fields.
    c.equal<int>(i, "kind", static_cast<int>(g.kind), static_cast<int>(d.kind));
    c.equal<int>(i, "tensor slot", g.has_tensor, d.has_tensor);
    c.equal<std::size_t>(i, "tensors slot", g.tensors, d.tensors);
    c.equal<std::size_t>(i, "fields slot", g.fields, d.fields);
    c.equal<std::int64_t>(i, "frame_id", g.frame_id, d.frame_id);
    c.equal<std::int64_t>(i, "frame_id vs pushed", item.frame_id, d.frame_id);
    c.equal<std::string>(i, "stream_id", g.stream_id, d.stream_id);
    c.equal<std::int64_t>(i, "input_seq", g.input_seq, d.input_seq);
    c.equal<std::int64_t>(i, "input_seq vs push order", static_cast<std::int64_t>(i), d.input_seq);
    if (!v.do_timestamp) {
      c.equal<std::int64_t>(i, "pts_ns", g.pts_ns, d.pts_ns);
      c.equal<std::int64_t>(i, "pts_ns vs pushed", proposes_time ? item.pts_ns : -1, d.pts_ns);
    } else {
      c.excluded(i, "pts_ns", std::to_string(g.pts_ns), std::to_string(d.pts_ns),
                 "do_timestamp: GStreamer stamps running time, c3.md 4 row 4");
    }
    if (item.orig_seq >= 0) {
      pushed_orig_seen = true;
    }
    if (item.orig_seq < 0 && !(v.connected && pushed_orig_seen)) {
      c.equal<std::int64_t>(i, "orig_input_seq", g.orig_input_seq, d.orig_input_seq);
    } else if (item.orig_seq < 0) {
      c.excluded(i, "orig_input_seq (after a pushed one, graph path)",
                 std::to_string(g.orig_input_seq), std::to_string(d.orig_input_seq),
                 "GStreamer's graph ingress counter skips an item that carried its own "
                 "orig_input_seq (RunPush.cpp:278-287), so later items lag; c2.md 3.6");
      c.equal<std::int64_t>(i, "Dataflow orig_input_seq = input_seq", d.input_seq,
                            d.orig_input_seq);
    } else {
      c.excluded(i, "orig_input_seq (pushed)", std::to_string(g.orig_input_seq),
                 std::to_string(d.orig_input_seq),
                 "no Stamp field for a pushed orig_input_seq, c2.md 3.6");
      c.equal<std::int64_t>(i, "Dataflow orig_input_seq = input_seq (documented)", d.input_seq,
                            d.orig_input_seq);
    }
    c.equal<std::string>(i, "media_type", g.media_type, d.media_type);
    c.equal<std::string>(i, "payload_tag", g.payload_tag, d.payload_tag);
    c.excluded(i, "stream_label/dts_ns/duration_ns/caps_string/attributes",
               "'" + g.stream_label + "'/" + std::to_string(g.dts_ns) + "/" +
                   std::to_string(g.duration_ns) + "/caps:" + std::to_string(g.caps_string.size()) +
                   "B/" + std::to_string(g.attributes),
               "'" + d.stream_label + "'/" + std::to_string(d.dts_ns) + "/" +
                   std::to_string(d.duration_ns) + "/caps:" + std::to_string(d.caps_string.size()) +
                   "B/" + std::to_string(d.attributes),
               "no caps, no meta map on Dataflow, c3.md 4 row 5");
    c.check(i, "Dataflow carries no stream_label, dts_ns, duration_ns, caps or attributes",
            d.stream_label.empty() && d.dts_ns == -1 && d.duration_ns == -1 &&
                d.caps_string.empty() && d.attributes == 0);
  }
  c.equal<std::string>(items.size(), "status after close_input and the drained queue",
                       status_name(gst.final_status), status_name(df.final_status));
  c.equal<std::string>(items.size(), "Dataflow status after close_input", "Closed",
                       status_name(df.final_status));
}

} // namespace

RUN_TEST("unit_run_backend_differential_test", ([] {
           constexpr RunPreset kRealtime = RunPreset::Realtime;
           constexpr RunPreset kBalanced = RunPreset::Balanced;
           // name, preset, connected, vary_shapes, copy_input, do_timestamp, new_pull
           const std::vector<Variant> variants = {
               {"linear/realtime/old-pull/copy_input=off/do_timestamp=off", kRealtime, false, true,
                false, false, false},
               {"linear/realtime/old-pull/copy_input=on/do_timestamp=off", kRealtime, false, true,
                true, false, false},
               {"linear/realtime/new-pull/copy_input=off/do_timestamp=off", kRealtime, false, true,
                false, false, true},
               {"linear/realtime/new-pull/copy_input=on/do_timestamp=off", kRealtime, false, true,
                true, false, true},
               {"linear/realtime/old-pull/copy_input=off/do_timestamp=on", kRealtime, false, true,
                false, true, false},
               {"linear/balanced/old-pull/copy_input=off/do_timestamp=off", kBalanced, false, false,
                false, false, false},
               {"linear/balanced/new-pull/copy_input=on/do_timestamp=on", kBalanced, false, false,
                true, true, true},
               {"connected/balanced/old-pull/copy_input=off/do_timestamp=off", kBalanced, true,
                false, false, false, false},
               {"connected/balanced/new-pull/copy_input=on/do_timestamp=off", kBalanced, true,
                false, true, false, true},
           };
           std::vector<std::string> failures;
           std::size_t compared = 0;
           for (const Variant& v : variants) {
             std::cout << "== " << v.name << "\n";
             const Outcome gst = run_variant(v, RunBackend::GStreamer);
             const Outcome df = run_variant(v, RunBackend::Dataflow);
             Comparison c(v.name);
             compare(v, gst, df, c);
             std::cout << "  final: gstreamer=" << status_name(gst.final_status) << " '"
                       << gst.final_error << "' dataflow=" << status_name(df.final_status) << " '"
                       << df.final_error << "'\n";
             std::cout << "  compared " << c.compared() << " values, " << c.failures().size()
                       << " differences\n";
             compared += c.compared();
             failures.insert(failures.end(), c.failures().begin(), c.failures().end());
           }
           for (const std::string& f : failures) {
             std::cerr << "[DIFF] " << f << "\n";
           }
           std::cout << "[INFO] differential: " << variants.size() << " variants, " << compared
                     << " values compared, " << failures.size() << " differences\n";
           require(failures.empty(), std::to_string(failures.size()) +
                                         " differences between the backends (see [DIFF] lines)");
         }));

// Differential test (H0 3.10, parts C4, C2b): one fixed input sequence through the same graph on
// both backends (RunBackend::GStreamer and RunBackend::Dataflow, requested explicitly, in one
// process), with and without frame_id / stream_id / pts_ns set and with copy_input on and off; the
// pulled items are compared field by field and byte by byte. SIMA_NEAT_BACKEND does not matter
// here. W2 (zero-copy push): on Dataflow each single-input item is also checked to be wrapped (the
// pulled tensor's first byte is the pushed one) exactly when copy_input is off and the frame fills
// a slot (a fixed port's frames, a bounded port's frames at the bound), and copied otherwise; the
// fixed-port variants are wrapped throughout.
// H0b (device memory): two variants with InputOptions::memory_policy = Ev74, one pushing
// TensorMemory::EV74 tensors and one CPU tensors. Compared besides the above: whether the pulled
// tensor is DMA-BUF memory, and whether it is the pushed DMA-BUF itself (EV74 tensors: yes on both
// backends, GStreamer forwarding the buffer and Dataflow wrapping it; CPU tensors: no, both copy
// into the input's pool). GStreamer refuses a CPU tensor on such an input unless
// SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY is set (InputStreamPush.cpp), so that variant's
// GStreamer run sets it; Dataflow copies by default (h0b plan 3.2). Tensor::device is compared too
// (both backends label the pulled tensor CPU; the placement is the storage's). Excluded there,
// with both values printed:
//   - Stamp::frame_id and Stamp::source_time of a pushed EV74 tensor that proposes none: GStreamer
//     forwards the tensor's GstBuffer, whose metadata from the tensor's creation carries a frame id
//     and a timestamp; Dataflow carries what the push proposes (none).
//
// Compared (APPLICATION_MODEL.md 1.2: what applications read, plus what they echo):
//   the pull status; Sample::kind and its payload slots (tensor / tensors / fields, and how many);
//   per tensor: dtype, shape, image format and every byte (also against what was pushed);
//   frame_id (C0: -1 for push(TensorList) on GStreamer, so input_seq is the sequence key);
//   stream_id; input_seq; pts_ns when InputOptions::do_timestamp is false; orig_input_seq for the
//   items that push none; media_type and payload_tag (C3 shapes the Dataflow Sample as
//   GStreamer's); on the new surface (Run::pull(milliseconds) -> Result<TensorList>): the list's
//   size, each tensor's dtype, shape, format and bytes, Stamp::frame_id, Stamp::stream_name,
//   Stamp::seq, and Stamp::source_time (ns and clock) when do_timestamp is false; the Closed status
//   after close_input and the drained queue.
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
#include "dmabuf_identity.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;

constexpr int kMaxW = 96;
constexpr int kMaxH = 64;

// List: push(TensorList{t}); Stamped: push({t.with_stamp(...)}) (the same list verb, its tensor
// carrying a stamp proposal when the item sets one); Sample: push(Sample).
enum class Verb { List, Sample, Stamped };

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
      {Verb::List, 64, 48, -1, "", -1, -1},              // no identity at all
      {Verb::Sample, 32, 16, 11, "cam-a", 1001000, -1},  // all three set
      {Verb::Sample, 96, 64, -1, "", -1, -1},            // a Sample with none set
      {Verb::Stamped, 64, 48, 13, "cam-b", 3003000, -1}, // a list whose tensor proposes a stamp
      {Verb::Sample, 48, 32, 14, "", -1, -1},            // frame_id only
      {Verb::Sample, 96, 64, -1, "cam-a", -1, 500},      // stream only, and an orig_input_seq
      {Verb::Stamped, 16, 8, -1, "", -1, -1},            // push({t}), no stamp
      {Verb::Sample, 64, 48, -1, "", 7007000, -1},       // pts only
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
  bool new_pull; // Run::pull(milliseconds) -> Result<TensorList>, else Run::pull(int, Sample&)
  // W2: no max_* bounds, so the seed (64x48) fixes the port and a slot holds exactly one frame:
  // with copy_input off every item is wrapped on Dataflow (a linear seeded variant only).
  bool fixed_port = false;
  // H0b: the input's memory, and whether the pushed tensors are TensorMemory::EV74 ones.
  InputMemoryPolicy memory = InputMemoryPolicy::SystemMemory;
  bool ev74_source = false;
};

struct TensorObs {
  TensorDType dtype = TensorDType::UInt8;
  std::vector<std::int64_t> shape;
  int format = -1;
  std::vector<std::uint8_t> bytes;
  const std::byte* first = nullptr; // the tensor's first byte, when its storage has an address
  int device = 0;                   // Tensor::device.type
  std::uint64_t dmabuf = 0;         // its DMA-BUF's inode; 0 for CPU memory
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
  std::size_t list_size = 0;
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
  if (t.storage && t.storage->data) {
    o.first = static_cast<const std::byte*>(t.storage->data) + t.byte_offset;
  }
  o.device = static_cast<int>(t.device.type);
  o.dmabuf = sima_test::dmabuf_inode(t);
  return o;
}

Run build(const Variant& v, RunBackend backend) {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.memory_policy = v.memory;
  if (!v.fixed_port) {
    in.max_width = kMaxW;
    in.max_height = kMaxH;
    in.max_depth = 3;
  }
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
  case Verb::Stamped: {
    Stamp proposal;
    proposal.frame_id = item.frame_id;
    proposal.stream_name = item.stream;
    if (item.pts_ns >= 0) {
      proposal.source_time = TimePoint{item.pts_ns, ClockId::SensorLocal};
    }
    const Tensor stamped =
        item.frame_id >= 0 || !item.stream.empty() || item.pts_ns >= 0 ? t.with_stamp(proposal) : t;
    return v.connected ? run.push("image", {stamped}) : run.push({stamped});
  }
  }
  return false;
}

Observed pull_one(Run& run, const Variant& v) {
  Observed o;
  if (v.new_pull) {
    Result<TensorList> r = v.connected ? run.pull("out", 5000ms) : run.pull(5000ms);
    o.status = r.status();
    if (!r.ok()) {
      o.error = r.error().message;
      return o;
    }
    o.list_size = r->size();
    if (!r->empty()) {
      o.stamp = r->front().stamp();
    }
    for (const Tensor& t : *r) {
      o.leaves.push_back(observe(t));
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
  std::vector<const std::byte*> sources;     // each pushed tensor's first byte
  std::vector<std::uint64_t> source_dmabufs; // each pushed tensor's DMA-BUF (0: CPU memory)
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
      const Tensor source = rgb(items[i].w, items[i].h, static_cast<std::uint8_t>(0x10 + i));
      sources.push_back(v.ev74_source ? source.cvu() : source);
      out.source_dmabufs.push_back(sima_test::dmabuf_inode(sources.back()));
      out.pushed.push_back(sources.back().copy_dense_bytes_tight());
      out.sources.push_back(static_cast<const std::byte*>(sources.back().storage->data) +
                            sources.back().byte_offset);
      require(push_item(run, v, items[i], sources.back()),
              "push of item " + std::to_string(i) + " refused");
      out.items.push_back(pull_one(run, v));
    }
    run.close_input();
    if (v.new_pull) {
      Result<TensorList> r = v.connected ? run.pull("out", 5000ms) : run.pull(5000ms);
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
    std::cout << " list_size=" << o.list_size << " frame_id=" << o.stamp.frame_id
              << " stream_name='" << o.stamp.stream_name << "' stream=" << o.stamp.stream
              << " seq=" << o.stamp.seq << " source_time=" << o.stamp.source_time.ns << "/"
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
  std::size_t wrapped_items = 0;
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
    // W2: Dataflow wraps the pushed storage (the pulled tensor's first byte is the pushed one)
    // exactly when copy_input is off and the frame fills a slot: every frame of a fixed port, a
    // frame at the bound of a bounded one; every other item is copied.
    const bool device = v.memory != InputMemoryPolicy::SystemMemory;
    if (device && !g.leaves.empty() && !d.leaves.empty()) {
      // H0b: both backends hand out device memory, and the pushed DMA-BUF itself exactly when
      // the pushed tensor was one.
      const TensorObs& gl = g.leaves.front();
      const TensorObs& dl = d.leaves.front();
      c.equal<bool>(i, "the pulled tensor is DMA-BUF memory", gl.dmabuf != 0U, dl.dmabuf != 0U);
      c.check(i, "the Dataflow tensor is DMA-BUF memory", dl.dmabuf != 0U);
      c.equal<int>(i, "Tensor::device", gl.device, dl.device);
      c.equal<bool>(i, "the pulled tensor is the pushed DMA-BUF",
                    gl.dmabuf != 0U && gl.dmabuf == gst.source_dmabufs[i],
                    dl.dmabuf != 0U && dl.dmabuf == df.source_dmabufs[i]);
      c.equal<bool>(i, "Dataflow wraps a pushed EV74 tensor and copies a CPU one (H0b)",
                    v.ev74_source, dl.dmabuf != 0U && dl.dmabuf == df.source_dmabufs[i]);
    }
    if (!device && !d.leaves.empty() && i < df.sources.size()) {
      const bool fills_slot = v.fixed_port || (item.w == kMaxW && item.h == kMaxH);
      const bool wrapped = d.leaves.front().first == df.sources[i];
      c.equal<bool>(i, "Dataflow wraps the pushed storage (W2)", !v.copy_input && fills_slot,
                    wrapped);
      wrapped_items += wrapped ? 1U : 0U;
    }
    const bool proposes_time = item.pts_ns >= 0;
    // H0b: GStreamer forwards a pushed EV74 tensor's own GstBuffer and reads that buffer's
    // metadata for what the push does not propose.
    const char* const holder_meta =
        "GStreamer reads the forwarded EV74 buffer's creation metadata; Dataflow carries the "
        "push's proposal only";
    if (v.new_pull) {
      c.equal<std::size_t>(i, "list size", g.list_size, d.list_size);
      if (v.ev74_source && item.frame_id < 0) {
        c.excluded(i, "Stamp::frame_id", std::to_string(g.stamp.frame_id),
                   std::to_string(d.stamp.frame_id), holder_meta);
      } else {
        c.equal<std::int64_t>(i, "Stamp::frame_id", g.stamp.frame_id, d.stamp.frame_id);
      }
      c.equal<std::int64_t>(i, "Stamp::frame_id vs pushed", item.frame_id, d.stamp.frame_id);
      c.equal<std::string>(i, "Stamp::stream_name", g.stamp.stream_name, d.stamp.stream_name);
      c.equal<std::uint64_t>(i, "Stamp::seq", g.stamp.seq, d.stamp.seq);
      if (!v.do_timestamp && v.ev74_source && !proposes_time) {
        c.excluded(i, "Stamp::source_time", std::to_string(g.stamp.source_time.ns),
                   std::to_string(d.stamp.source_time.ns), holder_meta);
      } else if (!v.do_timestamp) {
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
  std::cout << "  W2: Dataflow wrapped " << wrapped_items << " of " << df.items.size()
            << " items (copy_input " << (v.copy_input ? "on" : "off")
            << (v.fixed_port ? ", fixed port" : ", bounded port") << ")\n";
}

// ---- H0c: graphs of several endpoints -------------------------------------------------------
// Two variants, each on both backends:
//   two-by-two: Inputs "left" and "right", each feeding its own Output ("left_out", "right_out");
//     the items alternate between the inputs, each pulled from its output right after its push.
//   fan-out: Input "image" feeding Outputs "a" and "b" (one push, two outputs), every item pulled
//     from both.
// Compared as the single-endpoint variants compare (status, payload, frame_id, stream_id, pts_ns,
// the new surface's stamp), per output, plus the status of every output after close_input.
// orig_input_seq across two inputs (two-by-two): GStreamer's graph ingress numbers the pushes of
// the whole Run in one counter and writes it there, keeping a pushed orig_input_seq without
// counting that push (next_public_graph_input_seq, RunPush.cpp:276-287); Dataflow does the same in
// a Run of several inputs. Compared for every item, and the Dataflow side against that rule.
// input_seq and Stamp::seq count each input's pushes on both (each GStreamer segment numbers its
// own): compared, and the Dataflow side checked to be contiguous per input from 0. Excluded
// explicitly, besides the single-endpoint exclusions (c3.md 4 row 5):
//   - fan-out, items pushed as a Sample: Sample::kind, the tensor/tensors slots, media_type and
//     payload_tag. GStreamer lowers a fan-out to its FanOut node with no pipeline segment, so a
//     pull returns the pushed Sample itself; Dataflow returns the Sample projection of the message
//     (a leaf is a TensorSet of one tensor, H0 plan 3.5) as on every other graph. An item pushed as
//     a TensorList is a TensorSet on both, compared; the payload is compared for every item.
//   - stream_label, dts_ns, duration_ns, caps_string, attributes, as for one endpoint.
//   - fan-out: input_seq, orig_input_seq and Stamp::seq from the item that pushes its own
//     orig_input_seq on: GStreamer's ingress takes it as input_seq and its counter does not advance
//     for that item, so later items lag (RunPush.cpp:276-287); the runtime's Stamp carries neither
//     (c2.md 3.6). Before that item they are compared; the Dataflow side is checked throughout.
enum class Topology { TwoByTwo, FanOut };

struct MultiVariant {
  std::string name;
  Topology topology;
  bool new_pull;
};

// One pull: the output it read and what it saw.
struct MultiObserved {
  std::string output;
  std::size_t item = 0;
  Observed seen;
};

struct MultiOutcome {
  std::vector<MultiObserved> pulls;
  std::vector<std::vector<std::uint8_t>> pushed;
  std::vector<std::pair<std::string, PullStatus>> final_status;
  std::string build_error;
};

std::vector<std::string> multi_outputs(Topology t) {
  return t == Topology::TwoByTwo ? std::vector<std::string>{"left_out", "right_out"}
                                 : std::vector<std::string>{"a", "b"};
}

Run build_multi(const MultiVariant& v, RunBackend backend) {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.memory_policy = InputMemoryPolicy::SystemMemory;
  in.max_width = kMaxW;
  in.max_height = kMaxH;
  in.max_depth = 3;
  in.do_timestamp = false;
  OutputOptions out;
  out.max_buffers = 16;
  out.drop = false;
  out.sync = false;
  RunOptions opt;
  opt.backend = backend;
  opt.queue_depth = 16;
  opt.overflow_policy = OverflowPolicy::Block;
  opt.startup_preflight = false;
  Graph g;
  if (v.topology == Topology::TwoByTwo) {
    g.add(nodes::Input("left", in));
    g.add(nodes::Output("left_out", out));
    g.add(nodes::Input("right", in));
    g.add(nodes::Output("right_out", out));
    g.connect("left", "left_out");
    g.connect("right", "right_out");
  } else {
    g.add(nodes::Input("image", in));
    g.add(nodes::Output("a", out));
    g.add(nodes::Output("b", out));
    g.connect("image", "a");
    g.connect("image", "b");
  }
  return g.build(opt);
}

Observed pull_named(Run& run, bool new_pull, const std::string& output) {
  Observed o;
  if (new_pull) {
    Result<TensorList> r = run.pull(output, 5000ms);
    o.status = r.status();
    if (!r.ok()) {
      o.error = r.error().message;
      return o;
    }
    o.list_size = r->size();
    if (!r->empty()) {
      o.stamp = r->front().stamp();
    }
    for (const Tensor& t : *r) {
      o.leaves.push_back(observe(t));
    }
    return o;
  }
  Sample s;
  PullError err;
  o.status = run.pull(output, 5000, s, &err);
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

MultiOutcome run_multi(const MultiVariant& v, RunBackend backend) {
  MultiOutcome out;
  try {
    Run run = build_multi(v, backend);
    const std::vector<Item> items = sequence(false);
    std::vector<Tensor> sources;
    for (std::size_t i = 0; i < items.size(); ++i) {
      sources.push_back(rgb(items[i].w, items[i].h, static_cast<std::uint8_t>(0x50 + i)));
      out.pushed.push_back(sources.back().copy_dense_bytes_tight());
      std::string input = "image";
      std::vector<std::string> outputs = {"a", "b"};
      if (v.topology == Topology::TwoByTwo) {
        const bool left = (i % 2U) == 0U;
        input = left ? "left" : "right";
        outputs = {left ? "left_out" : "right_out"};
      }
      const Item& item = items[i];
      bool pushed = false;
      switch (item.verb) {
      case Verb::List:
        pushed = run.push(input, TensorList{sources.back()});
        break;
      case Verb::Sample: {
        Sample s;
        s.kind = SampleKind::Tensor;
        s.tensor = sources.back();
        s.frame_id = item.frame_id;
        s.stream_id = item.stream;
        s.pts_ns = item.pts_ns;
        s.orig_input_seq = item.orig_seq;
        pushed = run.push(input, Sample{s});
        break;
      }
      case Verb::Stamped: {
        Stamp proposal;
        proposal.frame_id = item.frame_id;
        proposal.stream_name = item.stream;
        if (item.pts_ns >= 0) {
          proposal.source_time = TimePoint{item.pts_ns, ClockId::SensorLocal};
        }
        const bool stamped = item.frame_id >= 0 || !item.stream.empty() || item.pts_ns >= 0;
        pushed = run.push(input, {stamped ? sources.back().with_stamp(proposal) : sources.back()});
        break;
      }
      }
      require(pushed, "push of item " + std::to_string(i) + " into " + input + " refused");
      for (const std::string& output : outputs) {
        out.pulls.push_back(MultiObserved{output, i, pull_named(run, v.new_pull, output)});
      }
    }
    run.close_input();
    for (const std::string& output : multi_outputs(v.topology)) {
      out.final_status.emplace_back(output, pull_named(run, v.new_pull, output).status);
    }
    run.close();
  } catch (const std::exception& e) {
    out.build_error = e.what();
  }
  return out;
}

void compare_multi(const MultiVariant& v, const MultiOutcome& gst, const MultiOutcome& df,
                   Comparison& c) {
  c.check(0, "GStreamer run completes", gst.build_error.empty(), gst.build_error);
  c.check(0, "Dataflow run completes", df.build_error.empty(), df.build_error);
  if (!gst.build_error.empty() || !df.build_error.empty()) {
    return;
  }
  const std::vector<Item> items = sequence(false);
  const bool fan_out = v.topology == Topology::FanOut;
  c.equal<std::size_t>(0, "pulls", gst.pulls.size(), df.pulls.size());
  std::map<std::string, std::int64_t> next_seq; // Dataflow: per input, from 0
  std::int64_t run_counter = 0;                 // two-by-two: the Run's push counter
  std::int64_t expected_orig = -1;
  bool pushed_orig_seen = false;
  for (std::size_t k = 0; k < gst.pulls.size() && k < df.pulls.size(); ++k) {
    const std::size_t i = gst.pulls[k].item;
    const Item& item = items[i];
    pushed_orig_seen = pushed_orig_seen || item.orig_seq >= 0;
    // GStreamer's sequence fields: in the fan-out (one input, no pipeline segment) comparable
    // before any pushed orig_input_seq; in the two-by-two variant always (each segment numbers
    // input_seq itself, orig_input_seq is the Run's counter on both backends).
    const bool seq_comparable = !fan_out || !pushed_orig_seen;
    const bool orig_comparable = !fan_out || !pushed_orig_seen;
    const char* seq_reason =
        "GStreamer's ingress takes a pushed orig_input_seq as input_seq and "
        "skips it in its counter (RunPush.cpp:276-287); the runtime's Stamp has "
        "no field for it";
    const char* orig_reason = seq_reason;
    const Observed& g = gst.pulls[k].seen;
    const Observed& d = df.pulls[k].seen;
    const std::string& output = gst.pulls[k].output;
    std::cout << "  [" << output << "]";
    print_row("gstreamer", i, g, v.new_pull);
    std::cout << "  [" << output << "]";
    print_row("dataflow ", i, d, v.new_pull);
    c.equal<std::string>(i, "output", output, df.pulls[k].output);
    c.equal<std::string>(i, "status", status_name(g.status), status_name(d.status));
    if (g.status != PullStatus::Ok || d.status != PullStatus::Ok) {
      continue;
    }
    c.equal<std::size_t>(i, "leaf count", g.leaves.size(), d.leaves.size());
    for (std::size_t n = 0; n < g.leaves.size() && n < d.leaves.size(); ++n) {
      c.equal<int>(i, "dtype", static_cast<int>(g.leaves[n].dtype),
                   static_cast<int>(d.leaves[n].dtype));
      c.equal<std::string>(i, "shape", shape_text(g.leaves[n].shape),
                           shape_text(d.leaves[n].shape));
      c.check(i, "bytes equal between backends", g.leaves[n].bytes == d.leaves[n].bytes);
      c.check(i, "Dataflow bytes equal the pushed bytes", d.leaves[n].bytes == df.pushed[i]);
    }
    // The input this item went to, for the per-input sequence of the two-by-two variant.
    const std::string input = fan_out ? "image" : ((i % 2U) == 0U ? "left" : "right");
    const bool first_output_of_item = k == 0 || gst.pulls[k - 1].item != i;
    std::int64_t expected_seq = next_seq[input];
    if (first_output_of_item) {
      ++next_seq[input];
      // The Run's counter (two-by-two): a pushed orig_input_seq is kept and not counted.
      expected_orig = item.orig_seq >= 0 ? item.orig_seq : run_counter++;
    } else {
      --expected_seq; // the fan-out's second output: the same message
    }
    const bool proposes_time = item.pts_ns >= 0;
    if (v.new_pull) {
      c.equal<std::size_t>(i, "list size", g.list_size, d.list_size);
      c.equal<std::int64_t>(i, "Stamp::frame_id", g.stamp.frame_id, d.stamp.frame_id);
      c.equal<std::string>(i, "Stamp::stream_name", g.stamp.stream_name, d.stamp.stream_name);
      c.equal<std::int64_t>(i, "Stamp::source_time.ns vs pushed", proposes_time ? item.pts_ns : -1,
                            d.stamp.source_time.valid() ? d.stamp.source_time.ns : -1);
      c.equal<std::int64_t>(i, "Stamp::source_time.ns", g.stamp.source_time.ns,
                            d.stamp.source_time.ns);
      if (seq_comparable) {
        c.equal<std::uint64_t>(i, "Stamp::seq", g.stamp.seq, d.stamp.seq);
      } else {
        c.excluded(i, "Stamp::seq", std::to_string(g.stamp.seq), std::to_string(d.stamp.seq),
                   seq_reason);
      }
      c.equal<std::uint64_t>(i, "Dataflow Stamp::seq per input",
                             static_cast<std::uint64_t>(expected_seq + 1), d.stamp.seq);
      continue;
    }
    c.equal<std::int64_t>(i, "frame_id", g.frame_id, d.frame_id);
    c.equal<std::int64_t>(i, "frame_id vs pushed", item.frame_id, d.frame_id);
    c.equal<std::string>(i, "stream_id", g.stream_id, d.stream_id);
    c.equal<std::int64_t>(i, "pts_ns", g.pts_ns, d.pts_ns);
    c.equal<std::int64_t>(i, "pts_ns vs pushed", proposes_time ? item.pts_ns : -1, d.pts_ns);
    if (seq_comparable) {
      c.equal<std::int64_t>(i, "input_seq", g.input_seq, d.input_seq);
    } else {
      c.excluded(i, "input_seq", std::to_string(g.input_seq), std::to_string(d.input_seq),
                 seq_reason);
    }
    if (orig_comparable) {
      c.equal<std::int64_t>(i, "orig_input_seq", g.orig_input_seq, d.orig_input_seq);
    } else {
      c.excluded(i, "orig_input_seq", std::to_string(g.orig_input_seq),
                 std::to_string(d.orig_input_seq), orig_reason);
    }
    c.equal<std::int64_t>(i, "Dataflow input_seq per input", expected_seq, d.input_seq);
    if (fan_out) {
      c.equal<std::int64_t>(i, "Dataflow orig_input_seq = input_seq (one input)", d.input_seq,
                            d.orig_input_seq);
    } else {
      c.equal<std::int64_t>(i, "Dataflow orig_input_seq = the Run's push counter", expected_orig,
                            d.orig_input_seq);
    }
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
    if (fan_out && item.verb == Verb::Sample) {
      c.excluded(
          i, "kind/tensor/tensors/media_type/payload_tag",
          std::to_string(static_cast<int>(g.kind)) + "/" + std::to_string(g.has_tensor) + "/" +
              std::to_string(g.tensors) + "/'" + g.media_type + "'/'" + g.payload_tag + "'",
          std::to_string(static_cast<int>(d.kind)) + "/" + std::to_string(d.has_tensor) + "/" +
              std::to_string(d.tensors) + "/'" + d.media_type + "'/'" + d.payload_tag + "'",
          "GStreamer's FanOut returns the pushed Sample; Dataflow its projection (H0 3.5)");
    } else {
      c.equal<int>(i, "kind", static_cast<int>(g.kind), static_cast<int>(d.kind));
      c.equal<int>(i, "tensor slot", g.has_tensor, d.has_tensor);
      c.equal<std::size_t>(i, "tensors slot", g.tensors, d.tensors);
      c.equal<std::string>(i, "media_type", g.media_type, d.media_type);
      c.equal<std::string>(i, "payload_tag", g.payload_tag, d.payload_tag);
    }
  }
  c.equal<std::size_t>(items.size(), "outputs after close_input", gst.final_status.size(),
                       df.final_status.size());
  for (std::size_t k = 0; k < gst.final_status.size() && k < df.final_status.size(); ++k) {
    std::cout << "  final " << gst.final_status[k].first
              << ": gstreamer=" << status_name(gst.final_status[k].second)
              << " dataflow=" << status_name(df.final_status[k].second) << "\n";
    c.equal<std::string>(items.size(), "status after close_input, per output",
                         status_name(gst.final_status[k].second),
                         status_name(df.final_status[k].second));
    c.equal<std::string>(items.size(), "Dataflow status after close_input", "Closed",
                         status_name(df.final_status[k].second));
  }
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
               // W2: a fixed port, every item wrapped on Dataflow (copy_input off) or copied (on).
               {"linear/balanced/fixed/old-pull/copy_input=off/do_timestamp=off", kBalanced, false,
                false, false, false, false, true},
               {"linear/balanced/fixed/new-pull/copy_input=off/do_timestamp=off", kBalanced, false,
                false, false, false, true, true},
               {"linear/balanced/fixed/new-pull/copy_input=on/do_timestamp=off", kBalanced, false,
                false, true, false, true, true},
               // H0b: an Ev74 input; pushed EV74 tensors (the pulled tensor is the pushed
               // DMA-BUF) and pushed CPU tensors (copied into the input's device pool).
               {"linear/balanced/fixed/ev74-input/ev74-tensors/new-pull", kBalanced, false, false,
                false, false, true, true, InputMemoryPolicy::Ev74, true},
               {"linear/balanced/fixed/ev74-input/cpu-tensors/old-pull", kBalanced, false, false,
                false, false, false, true, InputMemoryPolicy::Ev74, false},
           };
           std::vector<std::string> failures;
           std::size_t compared = 0;
           for (const Variant& v : variants) {
             std::cout << "== " << v.name << "\n";
             // GStreamer takes a CPU tensor on a device input only with its compatibility copy.
             const bool compat_copy = v.memory != InputMemoryPolicy::SystemMemory && !v.ev74_source;
             if (compat_copy) {
               setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 1);
             }
             const Outcome gst = run_variant(v, RunBackend::GStreamer);
             if (compat_copy) {
               unsetenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY");
             }
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
           // H0c: several endpoints.
           const std::vector<MultiVariant> multi = {
               {"two-by-two/old-pull", Topology::TwoByTwo, false},
               {"two-by-two/new-pull", Topology::TwoByTwo, true},
               {"fan-out/old-pull", Topology::FanOut, false},
               {"fan-out/new-pull", Topology::FanOut, true},
           };
           for (const MultiVariant& v : multi) {
             std::cout << "== " << v.name << "\n";
             const MultiOutcome gst = run_multi(v, RunBackend::GStreamer);
             const MultiOutcome df = run_multi(v, RunBackend::Dataflow);
             Comparison c(v.name);
             compare_multi(v, gst, df, c);
             std::cout << "  compared " << c.compared() << " values, " << c.failures().size()
                       << " differences\n";
             compared += c.compared();
             failures.insert(failures.end(), c.failures().begin(), c.failures().end());
           }
           for (const std::string& f : failures) {
             std::cerr << "[DIFF] " << f << "\n";
           }
           std::cout << "[INFO] differential: " << variants.size() + multi.size() << " variants, "
                     << compared << " values compared, " << failures.size() << " differences\n";
           // A process that started GStreamer may lose its stdout tail at exit: flush first.
           std::cout << std::flush;
           require(failures.empty(), std::to_string(failures.size()) +
                                         " differences between the backends (see [DIFF] lines)");
         }));

// neat_qualification_compare: one process of the qualification harness's comparator (README.md).
//
// A process runs on one backend: --backend selects it with simaai::neat::set_backend before
// anything is built. For one component package and one option view it drives the customer's entry
// forms with fixed inputs and writes one JSON line per event: each build's outcome, every pulled
// Sample's fields, every output tensor's metadata and the XXH64 of its dense bytes, each run's
// MeasureReport counters, the execution plan, the benchmark report's fields. It judges nothing:
// qualify.py compares the records of two processes. Core's public API only.
//
// Forms (--forms, comma separated):
//   graph      Input() -> model.preprocess() -> inference() -> postprocess() -> Output() in a Graph
//              with the Model's placement options, built seeded with the first input; scenarios
//              lock (push one, pull one), sizes (image views: frames of eight sizes on the same
//              Run, last on it: a frame size a backend does not take ends the scenario, and on
//              GStreamer the Run) and free (a pushing and a pulling thread, on a Run of its own)
//   model      Model::build(seed) + Runner: lock and sizes on the default RunOptions; free on the
//              catalogue's own {queue_depth 4, Block, ZeroCopy}
//   model_run  Model::run(inputs, timeout) per frame
//   plan       Model::execution_plan().to_json()
//   bench      Model::benchmark with a short count (the report's fields, not its numbers)
// Inputs: an image view pushes img0..img3.bgr (1280x720, BGR bytes) of --images in the view's frame
// format; a tensor view pushes four deterministic synthetic tensor lists from the model's
// input_specs(). Both in EV74 memory (Tensor::cvu()): GStreamer refuses a CPU tensor on a route
// whose first stage is an EV74 kernel. Every pushed tensor carries a Stamp whose frame_id is the
// frame's index. The last record (done) also says what the process holds of GStreamer at its end
// (tests/support/gstreamer_footprint.h): a process on Dataflow must hold nothing.
//
//   neat_qualification_compare --backend gstreamer|dataflow --package <tar.gz> --opts <file|->
//       --images <dir> --out <file.jsonl> [--forms graph] [--frames 200] [--timeout-ms 20000]
//       [--route full|pre] [--hash-bench] [--idle-pull]
// Scenarios of the graph form: lock, lock_tl (16 more frames on the same Run through
// Run::pull(std::chrono::milliseconds), which returns the tensors with their Stamp), sizes, idle
// (with --idle-pull: one 200 ms pull before the first push of the free Run) and free.

#include "model/Model.h"
#include "neat/runtime.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Run.h"
#include "pipeline/TensorCore.h"

#include "qualification_common.h"
#include "../support/gstreamer_footprint.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace simaai::neat;
namespace q = simaai::neat::qualification;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::ofstream g_out;
void emit(const std::string& line) {
  g_out << line << "\n";
}

std::string error_text(const std::exception& e) {
  return e.what();
}
std::string error_code(const std::exception& e) {
  if (const auto* n = dynamic_cast<const NeatError*>(&e)) {
    return n->report().error_code;
  }
  return {};
}

// ---- names
// ------------------------------------------------------------------------------------------
std::string dtype_name(TensorDType d) {
  switch (d) {
  case TensorDType::Int8:
    return "Int8";
  case TensorDType::UInt8:
    return "UInt8";
  case TensorDType::Float32:
    return "Float32";
  case TensorDType::Float64:
    return "Float64";
  case TensorDType::Int32:
    return "Int32";
  case TensorDType::Int16:
    return "Int16";
  case TensorDType::UInt16:
    return "UInt16";
  case TensorDType::BFloat16:
    return "BFloat16";
  }
  return "dtype#" + std::to_string(static_cast<int>(d));
}
std::string layout_name(TensorLayout l) {
  switch (l) {
  case TensorLayout::HWC:
    return "HWC";
  case TensorLayout::CHW:
    return "CHW";
  case TensorLayout::HW:
    return "HW";
  default:
    return "layout#" + std::to_string(static_cast<int>(l));
  }
}
std::string device_name(const Device& d) {
  std::string n;
  switch (d.type) {
  case DeviceType::CPU:
    n = "CPU";
    break;
  case DeviceType::SIMA_APU:
    n = "SIMA_APU";
    break;
  case DeviceType::SIMA_CVU:
    n = "SIMA_CVU";
    break;
  case DeviceType::SIMA_MLA:
    n = "SIMA_MLA";
    break;
  default:
    n = "device#" + std::to_string(static_cast<int>(d.type));
  }
  return n + ":" + std::to_string(d.id);
}
std::string storage_name(StorageKind k) {
  switch (k) {
  case StorageKind::CpuOwned:
    return "CpuOwned";
  case StorageKind::CpuExternal:
    return "CpuExternal";
  case StorageKind::GstSample:
    return "GstSample";
  case StorageKind::DeviceHandle:
    return "DeviceHandle";
  default:
    return "storage#" + std::to_string(static_cast<int>(k));
  }
}
std::string sample_kind_name(SampleKind k) {
  switch (k) {
  case SampleKind::Tensor:
    return "Tensor";
  case SampleKind::TensorSet:
    return "TensorSet";
  case SampleKind::Bundle:
    return "Bundle";
  default:
    return "kind#" + std::to_string(static_cast<int>(k));
  }
}
std::string payload_name(PayloadType p) {
  switch (p) {
  case PayloadType::Auto:
    return "Auto";
  case PayloadType::Image:
    return "Image";
  case PayloadType::Tensor:
    return "Tensor";
  case PayloadType::Encoded:
    return "Encoded";
  }
  return "payload#" + std::to_string(static_cast<int>(p));
}
std::string status_name(PullStatus s) {
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
  return "status#" + std::to_string(static_cast<int>(s));
}
std::string pixel_name(ImageSpec::PixelFormat f) {
  switch (f) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::NV16:
    return "NV16";
  default:
    return "pixel#" + std::to_string(static_cast<int>(f));
  }
}

// ---- output hashing
// ----------------------------------------------------------------------------------
struct HashStats {
  std::uint64_t ns = 0, bytes = 0, tensors = 0;
};

struct Hashed {
  std::string hash, how;
  std::size_t bytes = 0;
};

// XXH64 of the tensor's dense bytes, read in place when they are one contiguous range of the
// mapping, else through Tensor::copy_dense_bytes_tight_to; a tensor without a dense size (tiled
// storage) hashes the whole mapping. `check`: also hash the copy and say whether both agree.
Hashed hash_tensor(const Tensor& t, HashStats& stats, bool check) {
  static std::vector<std::uint8_t> scratch;
  Hashed h;
  const auto t0 = Clock::now();
  std::size_t dense = 0;
  try {
    dense = t.dense_bytes_tight();
  } catch (const std::exception&) {
    dense = 0;
  }
  try {
    const Mapping m = t.map(MapMode::Read);
    if (dense > 0 && t.is_contiguous() && m.data && m.size_bytes >= dense) {
      h.hash = q::hex64(q::xxh64(m.data, dense));
      h.how = "in_place";
      h.bytes = dense;
    } else if (dense > 0) {
      if (scratch.size() < dense) {
        scratch.resize(dense);
      }
      if (t.copy_dense_bytes_tight_to(scratch.data(), dense)) {
        h.hash = q::hex64(q::xxh64(scratch.data(), dense));
        h.how = "copy";
        h.bytes = dense;
      }
    }
    if (h.hash.empty() && m.data && m.size_bytes > 0) {
      h.hash = q::hex64(q::xxh64(m.data, m.size_bytes));
      h.how = "mapping";
      h.bytes = m.size_bytes;
    }
  } catch (const std::exception& e) {
    h.how = std::string("unreadable: ") + e.what();
  }
  if (h.hash.empty()) {
    h.hash = "unreadable";
  }
  stats.ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
  stats.bytes += h.bytes;
  ++stats.tensors;
  if (check && h.how == "in_place") {
    if (scratch.size() < dense) {
      scratch.resize(dense);
    }
    const bool copied = t.copy_dense_bytes_tight_to(scratch.data(), dense);
    h.how += copied ? (q::hex64(q::xxh64(scratch.data(), dense)) == h.hash ? ",checked"
                                                                           : ",CHECK-MISMATCH")
                    : ",copy-unavailable";
  }
  return h;
}

// ---- records
// -------------------------------------------------------------------------------------------
std::string preprocess_record(const PreprocessRuntimeMeta& p) {
  q::Obj o;
  o.raw("original", q::json_integers(std::vector<long long>{p.original_width, p.original_height}));
  o.raw("resized", q::json_integers(std::vector<long long>{p.resized_width, p.resized_height}));
  o.raw("scaled", q::json_integers(std::vector<long long>{p.scaled_width, p.scaled_height}));
  o.raw("pad_lrtb",
        q::json_integers(std::vector<long long>{p.pad_left, p.pad_right, p.pad_top, p.pad_bottom}));
  o.str("resize_mode", p.resize_mode).str("color_in", p.color_in).str("color_out", p.color_out);
  o.boolean("normalize", p.normalize).boolean("quantize", p.quantize);
  o.boolean("tessellate", p.tessellate);
  o.raw("affine", q::json_array({q::json_number(p.affine_m00), q::json_number(p.affine_m01),
                                 q::json_number(p.affine_m02), q::json_number(p.affine_m10),
                                 q::json_number(p.affine_m11), q::json_number(p.affine_m12)}));
  o.raw("affine_scale",
        q::json_array({q::json_number(p.affine_scale_x), q::json_number(p.affine_scale_y)}));
  o.raw("affine_offset",
        q::json_array({q::json_number(p.affine_offset_x), q::json_number(p.affine_offset_y)}));
  o.boolean("roi_list_enabled", p.roi_list_enabled);
  o.integer("rois", static_cast<long long>(p.rois.size()));
  o.integer("roi_valid_count", p.roi_valid_count).integer("roi_capacity", p.roi_capacity);
  return o.text();
}

std::string stamp_record(const Stamp& s) {
  q::Obj o;
  o.integer("stream", s.stream).str("stream_name", s.stream_name);
  o.integer("seq", static_cast<long long>(s.seq)).integer("frame_id", s.frame_id);
  // Instants differ run to run: only whether each is set, and on which clock.
  o.str("source_time", s.source_time.valid()
                           ? "set:" + std::to_string(static_cast<int>(s.source_time.clock))
                           : "unset");
  o.str("publish_time", s.publish_time.valid()
                            ? "set:" + std::to_string(static_cast<int>(s.publish_time.clock))
                            : "unset");
  o.integer("flags", static_cast<long long>(s.flags)).integer("version", s.version);
  return o.text();
}

// One leaf tensor. `holders`: small integers for storage holders, by first appearance in the Sample
// (which tensors share one allocation); `check`: verify the in-place hash against a copy.
std::string tensor_record(const Tensor& t, const std::string& slot,
                          std::map<const void*, int>& holders, HashStats& stats, bool check) {
  q::Obj o;
  o.str("slot", slot);
  o.str("dtype", dtype_name(t.dtype)).str("layout", layout_name(t.layout));
  o.raw("shape", q::json_integers(t.shape)).raw("strides_bytes", q::json_integers(t.strides_bytes));
  o.integer("byte_offset", t.byte_offset);
  std::vector<int> axes;
  for (const auto a : t.axis_semantics) {
    axes.push_back(static_cast<int>(a));
  }
  o.raw("axis_semantics", q::json_integers(axes));
  o.str("device", device_name(t.device)).boolean("read_only", t.read_only);
  std::vector<int> roles;
  for (const auto& p : t.planes) {
    roles.push_back(static_cast<int>(p.role));
  }
  o.raw("planes", q::json_integers(roles));
  bool contiguous = false;
  try {
    contiguous = t.is_contiguous();
  } catch (const std::exception&) {
  }
  o.boolean("contiguous", contiguous);
  q::Obj route;
  route.str("stage_key", t.route.stage_key).integer("logical_index", t.route.logical_index);
  route.integer("backend_output_index", t.route.backend_output_index);
  route.integer("route_slot", t.route.route_slot).integer("physical_index", t.route.physical_index);
  route.integer("memory_index", t.route.memory_index);
  route.integer("physical_byte_offset", t.route.physical_byte_offset);
  route.str("name", t.route.name).str("backend_name", t.route.backend_name);
  route.str("segment_name", t.route.segment_name);
  o.raw("route", route.text());
  q::Obj sem;
  if (t.semantic.image) {
    sem.str("image", pixel_name(t.semantic.image->format));
  }
  if (t.semantic.tess) {
    q::Obj tess;
    tess.str("format", t.semantic.tess->format);
    tess.raw("slice_shape", q::json_integers(t.semantic.tess->slice_shape));
    tess.integer("storage_bytes", static_cast<long long>(t.semantic.tess->storage_bytes));
    sem.raw("tess", tess.text());
  }
  if (t.semantic.quant) {
    q::Obj qu;
    qu.num("scale", t.semantic.quant->scale).integer("zero_point", t.semantic.quant->zero_point);
    qu.integer("axis", t.semantic.quant->axis);
    qu.integer("scales", static_cast<long long>(t.semantic.quant->scales.size()));
    sem.raw("quant", qu.text());
  }
  if (t.semantic.detection) {
    sem.str("detection", t.semantic.detection->format);
  }
  if (t.semantic.feature) {
    sem.str("feature", t.semantic.feature->format);
  }
  if (t.semantic.preprocess) {
    sem.raw("preprocess", preprocess_record(*t.semantic.preprocess));
  }
  o.raw("semantic", sem.text());
  if (t.storage) {
    const TensorBuffer& s = *t.storage;
    q::Obj st;
    st.str("kind", storage_name(s.kind)).str("device", device_name(s.device));
    st.integer("size_bytes", static_cast<long long>(s.size_bytes));
    std::vector<std::string> segs;
    for (const auto& g : s.sima_segments) {
      segs.push_back(q::json_string(g.name + ":" + std::to_string(g.size_bytes)));
    }
    st.raw("segments", q::json_array(segs));
    const void* holder = s.holder ? s.holder.get() : static_cast<const void*>(&s);
    const auto it = holders.emplace(holder, static_cast<int>(holders.size())).first;
    st.integer("holder", it->second);
    o.raw("storage", st.text());
  } else {
    o.raw("storage", "null");
  }
  o.raw("stamp", stamp_record(t.stamp())).boolean("envelope", static_cast<bool>(t.envelope));
  const Hashed h = hash_tensor(t, stats, check);
  o.str("xxh64", h.hash).str("hashed", h.how).integer("bytes", static_cast<long long>(h.bytes));
  return o.text();
}

void leaves(const Tensor& t, const std::string& slot, std::vector<std::string>& out,
            std::map<const void*, int>& holders, HashStats& stats, bool check) {
  if (t.is_record()) {
    const auto names = t.field_names();
    std::size_t i = 0;
    for (const Tensor& f : t.fields()) {
      leaves(f, slot + "." + (i < names.size() ? names[i] : std::to_string(i)), out, holders, stats,
             check);
      ++i;
    }
    return;
  }
  out.push_back(tensor_record(t, slot, holders, stats, check));
}

std::string sample_fields(const Sample& s) {
  q::Obj o;
  o.str("kind", sample_kind_name(s.kind)).boolean("owned", s.owned);
  o.str("caps_string", s.caps_string).str("payload_type", payload_name(s.payload_type));
  o.str("media_type", s.media_type).str("payload_tag", s.payload_tag).str("format", s.format);
  o.integer("frame_id", s.frame_id).str("stream_id", s.stream_id);
  o.str("stream_label", s.stream_label).str("port_name", s.port_name);
  o.integer("output_index", s.output_index);
  o.integer("logical_output_index", s.logical_output_index);
  o.integer("memory_index", s.memory_index).integer("route_slot", s.route_slot);
  o.str("segment_name", s.segment_name);
  o.integer("input_seq", s.input_seq).integer("orig_input_seq", s.orig_input_seq);
  o.integer("pts_ns", s.pts_ns).integer("dts_ns", s.dts_ns).integer("duration_ns", s.duration_ns);
  q::Obj at;
  for (const auto& kv : s.attributes) {
    at.str(kv.first, kv.second);
  }
  o.raw("attributes", at.text());
  o.boolean("slot_tensor", s.tensor.has_value());
  o.integer("slot_tensors", static_cast<long long>(s.tensors.size()));
  o.integer("slot_fields", static_cast<long long>(s.fields.size()));
  return o.text();
}

void sample_leaves(const Sample& s, const std::string& prefix, std::vector<std::string>& out,
                   std::map<const void*, int>& holders, HashStats& stats, bool check) {
  if (s.tensor) {
    leaves(*s.tensor, prefix + "tensor", out, holders, stats, check);
  }
  for (std::size_t i = 0; i < s.tensors.size(); ++i) {
    leaves(s.tensors[i], prefix + "tensors[" + std::to_string(i) + "]", out, holders, stats, check);
  }
  for (std::size_t i = 0; i < s.fields.size(); ++i) {
    sample_leaves(s.fields[i], prefix + "fields[" + std::to_string(i) + "].", out, holders, stats,
                  check);
  }
}

// ---- inputs
// --------------------------------------------------------------------------------------------
struct Frame {
  int width = 0, height = 0;
  std::vector<std::uint8_t> bgr; // packed BGR
};

Frame load_bgr(const std::string& path, int width, int height) {
  Frame f{width, height, {}};
  f.bgr.resize(static_cast<std::size_t>(width) * height * 3U);
  std::ifstream in(path, std::ios::binary);
  if (!in ||
      !in.read(reinterpret_cast<char*>(f.bgr.data()), static_cast<std::streamsize>(f.bgr.size())))
    throw std::runtime_error("image " + path + " is not " + std::to_string(f.bgr.size()) +
                             " bytes");
  return f;
}

// Nearest-neighbour resampling: any fixed function serves, both backends receive the same bytes.
Frame resized(const Frame& src, int width, int height) {
  Frame f{width, height, {}};
  f.bgr.resize(static_cast<std::size_t>(width) * height * 3U);
  for (int y = 0; y < height; ++y) {
    const int sy = static_cast<int>((static_cast<long long>(y) * src.height) / height);
    for (int x = 0; x < width; ++x) {
      const int sx = static_cast<int>((static_cast<long long>(x) * src.width) / width);
      std::memcpy(&f.bgr[(static_cast<std::size_t>(y) * width + x) * 3U],
                  &src.bgr[(static_cast<std::size_t>(sy) * src.width + sx) * 3U], 3);
    }
  }
  return f;
}

Tensor frame_tensor(const Frame& f, const std::string& format) {
  const int w = f.width, h = f.height;
  if (format == "NV12") {
    const int ew = w & ~1, eh = h & ~1;
    std::vector<std::uint8_t> v(static_cast<std::size_t>(ew) * eh * 3U / 2U);
    const auto u8 = [](double x) {
      return static_cast<std::uint8_t>(std::clamp(std::lround(x), 0L, 255L));
    };
    const auto px = [&](int x, int y) {
      return &f.bgr[(static_cast<std::size_t>(y) * w + x) * 3U];
    };
    for (int y = 0; y < eh; ++y)
      for (int x = 0; x < ew; ++x) {
        const std::uint8_t* p = px(x, y);
        v[static_cast<std::size_t>(y) * ew + x] = u8(0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2]);
      }
    std::uint8_t* uv = v.data() + static_cast<std::size_t>(ew) * eh;
    for (int y = 0; y < eh; y += 2)
      for (int x = 0; x < ew; x += 2) {
        double b = 0, g = 0, r = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            const std::uint8_t* p = px(x + dx, y + dy);
            b += p[0] / 4.0, g += p[1] / 4.0, r += p[2] / 4.0;
          }
        uv[static_cast<std::size_t>(y / 2) * ew + x] =
            u8(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
        uv[static_cast<std::size_t>(y / 2) * ew + x + 1] =
            u8(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);
      }
    Tensor t = Tensor::from_vector(v, {static_cast<std::int64_t>(v.size())}, TensorMemory::CPU);
    t.layout = TensorLayout::HW;
    t.shape = {eh, ew};
    t.strides_bytes = {ew, 1};
    t.semantic.image = ImageSpec{ImageSpec::PixelFormat::NV12, ""};
    Plane luma;
    luma.role = PlaneRole::Y;
    luma.shape = {eh, ew};
    luma.strides_bytes = {ew, 1};
    Plane chroma;
    chroma.role = PlaneRole::UV;
    chroma.shape = {eh / 2, ew};
    chroma.strides_bytes = {ew, 1};
    chroma.byte_offset = static_cast<std::int64_t>(ew) * eh;
    t.planes = {luma, chroma};
    return t.cvu();
  }
  if (format == "GRAY8") {
    std::vector<std::uint8_t> g(static_cast<std::size_t>(w) * h);
    for (std::size_t i = 0; i < g.size(); ++i) {
      const std::uint8_t* p = &f.bgr[i * 3U];
      g[i] = static_cast<std::uint8_t>((114 * p[0] + 587 * p[1] + 299 * p[2] + 500) / 1000);
    }
    Tensor t = Tensor::from_vector(g, {h, w, 1}, TensorMemory::CPU);
    t.layout = TensorLayout::HWC;
    t.semantic.image = ImageSpec{ImageSpec::PixelFormat::GRAY8, ""};
    return t.cvu();
  }
  std::vector<std::uint8_t> bytes = f.bgr;
  if (format != "BGR") {
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
      std::swap(bytes[i], bytes[i + 2]);
    }
  }
  Tensor t = Tensor::from_vector(bytes, {h, w, 3}, TensorMemory::CPU);
  t.layout = TensorLayout::HWC;
  t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
  t.semantic.image =
      ImageSpec{format == "BGR" ? ImageSpec::PixelFormat::BGR : ImageSpec::PixelFormat::RGB, ""};
  return t.cvu();
}

// A tensor input: deterministic bytes in the spec's first dtype, in EV74 memory.
Tensor synthetic_tensor(const TensorSpec& spec, std::size_t index, int salt) {
  if (spec.shape.empty() || spec.dtypes.empty()) {
    throw std::runtime_error("input spec without shape or dtype");
  }
  const TensorDType dtype = spec.dtypes.front();
  const std::size_t elem = dtype == TensorDType::UInt8 || dtype == TensorDType::Int8      ? 1
                           : dtype == TensorDType::Float64                                ? 8
                           : dtype == TensorDType::Int32 || dtype == TensorDType::Float32 ? 4
                                                                                          : 2;
  std::vector<std::int64_t> shape;
  std::size_t count = 1;
  for (const std::int64_t d : spec.shape) {
    shape.push_back(d > 0 ? d : 1);
    count *= static_cast<std::size_t>(d > 0 ? d : 1);
  }
  std::vector<std::uint8_t> bytes(count * elem);
  std::uint32_t lcg = 12345U + static_cast<std::uint32_t>(index) * 7919U +
                      static_cast<std::uint32_t>(salt) * 104729U;
  const auto next = [&] {
    lcg = lcg * 1664525U + 1013904223U;
    return lcg >> 8;
  };
  for (std::size_t i = 0; i < count; ++i) {
    const float v = static_cast<float>(next() & 0xFFFFU) / 32768.0F - 1.0F;
    if (dtype == TensorDType::Float32) {
      std::memcpy(&bytes[4 * i], &v, 4);
    } else if (dtype == TensorDType::BFloat16) {
      std::uint32_t bits;
      std::memcpy(&bits, &v, 4);
      const auto b = static_cast<std::uint16_t>(bits >> 16);
      std::memcpy(&bytes[2 * i], &b, 2);
    } else {
      for (std::size_t k = 0; k < elem; ++k) {
        bytes[i * elem + k] = static_cast<std::uint8_t>(next() & 0xFFU);
      }
    }
  }
  Tensor t =
      Tensor::from_vector(bytes, {static_cast<std::int64_t>(bytes.size())}, TensorMemory::CPU);
  t.dtype = dtype;
  t.shape = shape;
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t step = static_cast<std::int64_t>(elem);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = step;
    step *= shape[static_cast<std::size_t>(i)];
  }
  t.strides_bytes = strides;
  if (spec.image_format.has_value()) {
    t.semantic.image = ImageSpec{*spec.image_format, ""};
    if (shape.size() == 3U) {
      t.layout = TensorLayout::HWC;
    }
  }
  return t.cvu();
}

TensorList stamped(const TensorList& in, long frame) {
  TensorList out;
  for (const Tensor& t : in) {
    Stamp s;
    s.frame_id = frame;
    out.push_back(t.with_stamp(s));
  }
  return out;
}

// ---- the run of a scenario
// --------------------------------------------------------------------------- What a scenario
// pushes into and pulls from: a Graph's Run or a Model's Runner.
struct Port {
  std::function<bool(const TensorList&)> push;
  // Ok with the Sample, or the status and the error's text.
  std::function<std::string(int, Sample&, std::string&)> pull;
  // Run::pull(std::chrono::milliseconds) -> Result<TensorList>: a Graph's Run only.
  std::function<std::string(int, TensorList&, std::string&)> pull_tl;
  std::function<std::string()> state; // JSON
  std::function<std::optional<MeasureScope>()> measure;
};

Port port_of(Run& run) {
  Port p;
  p.push = [&run](const TensorList& in) { return run.push(in); };
  p.pull = [&run](int timeout, Sample& s, std::string& err) {
    PullError e;
    const PullStatus st = run.pull(timeout, s, &e);
    if (st != PullStatus::Ok) {
      err = e.message;
    }
    return status_name(st);
  };
  p.pull_tl = [&run](int timeout, TensorList& out, std::string& err) {
    Result<TensorList> r = run.pull(std::chrono::milliseconds(timeout));
    if (!r.ok()) {
      err = r.error().message;
      return status_name(r.status());
    }
    out = std::move(r.value());
    return std::string("Ok");
  };
  p.state = [&run] {
    q::Obj o;
    o.boolean("can_push", run.can_push()).boolean("can_pull", run.can_pull());
    o.boolean("running", run.running()).str("last_error", run.last_error());
    return o.text();
  };
  p.measure = [&run] { return std::optional<MeasureScope>(run.start_measurement()); };
  return p;
}

Port port_of(Model::Runner& r) {
  Port p;
  p.push = [&r](const TensorList& in) { return r.push(in); };
  p.pull = [&r](int timeout, Sample& s, std::string& err) -> std::string {
    try {
      s = r.pull(timeout);
    } catch (const std::exception& e) {
      err = e.what();
      return "Threw";
    }
    return s.empty() ? std::string("Empty") : std::string("Ok");
  };
  p.state = [] { return std::string("{}"); };
  p.measure = [&r] { return std::optional<MeasureScope>(r.start_measurement()); };
  return p;
}

struct Ctx {
  q::Args args;
  q::OptionSet set;
  std::vector<TensorList> inputs;  // the frames or tensor lists, used in rotation
  std::vector<std::string> labels; // what each input is
  std::vector<TensorList> sized;   // image views: the size scenario's frames
  std::vector<std::string> sized_labels;
  int timeout_ms = 20000;
  long frames = 200;
  HashStats hash;
};

// `s` null: no output; `fields` false: a TensorList pull (no Sample fields exist).
std::string frame_record(const std::string& form, const std::string& sc, long i,
                         const std::string& label, const std::string& push, const std::string& pull,
                         const std::string& err, const Sample* s, HashStats& stats, bool check,
                         bool fields = true) {
  q::Obj o;
  o.str("record", "frame").str("form", form).str("sc", sc).integer("i", i).str("input", label);
  o.str("push", push).str("pull", pull);
  if (!err.empty()) {
    o.str("error", err);
  }
  if (s) {
    if (fields) {
      o.raw("sample", sample_fields(*s));
    }
    std::vector<std::string> ts;
    std::map<const void*, int> holders;
    sample_leaves(*s, "", ts, holders, stats, check);
    o.raw("tensors", q::json_array(ts));
  }
  return o.text();
}

std::string counters_record(const MeasureReport& r) {
  q::Obj o;
  o.integer("inputs_enqueued", static_cast<long long>(r.counters.inputs_enqueued));
  o.integer("inputs_pushed", static_cast<long long>(r.counters.inputs_pushed));
  o.integer("outputs_ready", static_cast<long long>(r.counters.outputs_ready));
  o.integer("outputs_pulled", static_cast<long long>(r.counters.outputs_pulled));
  o.integer("inputs_dropped", static_cast<long long>(r.counters.inputs_dropped));
  o.integer("outputs_dropped", static_cast<long long>(r.counters.outputs_dropped));
  o.integer("outputs", static_cast<long long>(r.outputs));
  o.integer("end_to_end.count", static_cast<long long>(r.end_to_end.count));
  o.boolean("latency_samples_collected", r.latency_samples_collected);
  return o.text();
}

// Push one, pull one, `inputs` in rotation for `count` frames.
void lockstep(Ctx& c, Port& p, const std::string& form, const std::string& sc,
              const std::vector<TensorList>& inputs, const std::vector<std::string>& labels,
              long count) {
  const auto t0 = Clock::now();
  std::optional<MeasureScope> scope;
  try {
    scope = p.measure();
  } catch (const std::exception&) {
  }
  long ok = 0;
  std::string first_error;
  for (long i = 0; i < count; ++i) {
    const std::size_t k = static_cast<std::size_t>(i) % inputs.size();
    std::string push = "ok";
    try {
      if (!p.push(stamped(inputs[k], i))) {
        push = "false";
      }
    } catch (const std::exception& e) {
      push = "threw";
      emit(frame_record(form, sc, i, labels[k], push, "", e.what(), nullptr, c.hash, false));
      first_error = e.what();
      break;
    }
    Sample s;
    std::string err;
    std::string st;
    const bool tl = sc == "lock_tl";
    if (push == "ok") {
      if (tl) {
        TensorList out;
        st = p.pull_tl(c.timeout_ms, out, err);
        s.kind = SampleKind::TensorSet;
        s.tensors = std::move(out);
      } else {
        st = p.pull(c.timeout_ms, s, err);
      }
    }
    const bool good = st == "Ok";
    emit(frame_record(form, sc, i, labels[k], push, st, err, good ? &s : nullptr, c.hash, i < 4,
                      !tl));
    if (!good) {
      first_error = push != "ok" ? "push returned false" : st + ": " + err;
      break;
    }
    ++ok;
  }
  q::Obj o;
  o.str("record", "end").str("form", form).str("sc", sc).integer("frames", count);
  o.integer("frames_ok", ok).str("first_error", first_error).num("seconds", seconds_since(t0));
  if (scope) {
    try {
      o.raw("counters", counters_record(scope->stop()));
    } catch (const std::exception& e) {
      o.str("counters_error", e.what());
    }
  }
  try {
    o.raw("state", p.state());
  } catch (const std::exception& e) {
    o.str("state_error", e.what());
  }
  emit(o.text());
}

// A pushing thread and a pulling thread: `count` frames; the records in pull order.
void free_running(Ctx& c, Port& p, const std::string& form, long count) {
  const auto t0 = Clock::now();
  std::atomic<bool> push_failed{false};
  std::string push_error;
  long pushed = 0;
  std::thread pusher([&] {
    for (long i = 0; i < count; ++i) {
      try {
        if (!p.push(stamped(c.inputs[static_cast<std::size_t>(i) % c.inputs.size()], i))) {
          push_error = "push returned false at frame " + std::to_string(i);
          push_failed = true;
          return;
        }
      } catch (const std::exception& e) {
        push_error = std::string("push threw at frame ") + std::to_string(i) + ": " + e.what();
        push_failed = true;
        return;
      }
      ++pushed;
    }
  });
  std::vector<std::string> records;
  long ok = 0;
  std::string first_error;
  for (long i = 0; i < count; ++i) {
    Sample s;
    std::string err;
    const std::string st = p.pull(c.timeout_ms, s, err);
    if (st != "Ok") {
      records.push_back(frame_record(form, "free", i, "", "", st, err, nullptr, c.hash, false));
      first_error = st + ": " + err;
      break;
    }
    const long frame = s.frame_id;
    const std::string label =
        frame >= 0 ? c.labels[static_cast<std::size_t>(frame) % c.labels.size()] : std::string("?");
    records.push_back(frame_record(form, "free", i, label, "", st, err, &s, c.hash, false));
    ++ok;
  }
  pusher.join();
  for (const auto& r : records) {
    emit(r);
  }
  q::Obj o;
  o.str("record", "end").str("form", form).str("sc", "free").integer("frames", count);
  o.integer("frames_ok", ok).integer("pushed", pushed);
  o.str("first_error", !first_error.empty() ? first_error : push_error);
  o.num("seconds", seconds_since(t0));
  try {
    o.raw("state", p.state());
  } catch (const std::exception& e) {
    o.str("state_error", e.what());
  }
  emit(o.text());
}

void build_record(const std::string& form, const std::string& group, bool ok, double seconds,
                  const std::string& error = {}, const std::string& code = {}) {
  q::Obj o;
  o.str("record", "build").str("form", form).str("group", group).boolean("ok", ok);
  o.num("seconds", seconds);
  if (!ok) {
    o.str("error", error).str("error_code", code);
  }
  emit(o.text());
}

// The close of a scenario group's Run, as a record: whether it returned. A Run that a refused
// frame ended (GStreamer's, after a frame its decode does not take) may refuse its close; the
// forms behind it still run.
template <class Close>
void close_record(const std::string& form, const std::string& group, Close&& close) {
  q::Obj o;
  o.str("record", "close").str("form", form).str("group", group);
  try {
    close();
    o.boolean("ok", true);
  } catch (const std::exception& e) {
    o.boolean("ok", false).str("error", e.what());
  }
  emit(o.text());
}

// The Graph's options are the Model's placement options, as Model::build gives them to its own
// Graph (and as the census asks the planner): without them the fragments' kernels take the
// default placement.
Graph route_graph(const Model& model, const Model::Options& options, const std::string& route) {
  GraphOptions go;
  go.processcvu = options.processcvu;
  go.processmla = options.processmla;
  go.advanced_execution = options.advanced_execution;
  go.resolve_advanced_execution();
  Graph g(go);
  g.add(nodes::Input());
  g.add(model.preprocess());
  if (route != "pre") {
    g.add(model.inference());
    g.add(model.postprocess());
  }
  g.add(nodes::Output());
  return g;
}

void form_graph(Ctx& c, Model& model) {
  const std::string route = c.args.get("route", "full");
  {
    const auto t0 = Clock::now();
    std::optional<Run> run;
    try {
      run.emplace(route_graph(model, c.set.options, route).build(c.inputs.front(), RunOptions{}));
    } catch (const std::exception& e) {
      build_record("graph", "lock", false, seconds_since(t0), error_text(e), error_code(e));
      return;
    }
    build_record("graph", "lock", true, seconds_since(t0));
    Port p = port_of(*run);
    lockstep(c, p, "graph", "lock", c.inputs, c.labels, c.frames);
    // The same Run through the pull that returns the tensors with their Stamp.
    lockstep(c, p, "graph", "lock_tl", c.inputs, c.labels, 16);
    if (!c.sized.empty()) {
      lockstep(c, p, "graph", "sizes", c.sized, c.sized_labels, static_cast<long>(c.sized.size()));
    }
    close_record("graph", "lock", [&run] {
      run->close_input();
      run->close();
    });
  }
  {
    const auto t0 = Clock::now();
    std::optional<Run> run;
    try {
      run.emplace(route_graph(model, c.set.options, route).build(c.inputs.front(), RunOptions{}));
    } catch (const std::exception& e) {
      build_record("graph", "free", false, seconds_since(t0), error_text(e), error_code(e));
      return;
    }
    build_record("graph", "free", true, seconds_since(t0));
    if (c.args.has("idle-pull")) {
      // A timed pull before the first push: when it returns, as a class (the instant varies).
      Sample s;
      PullError e;
      const auto t1 = Clock::now();
      const PullStatus st = run->pull(200, s, &e);
      const double waited = seconds_since(t1);
      q::Obj o;
      o.str("record", "frame").str("form", "graph").str("sc", "idle").integer("i", 0);
      o.str("input", "none").str("pull", status_name(st));
      o.str("returned", waited < 1.0   ? "within 1 s"
                        : waited < 4.0 ? "within 4 s"
                                       : "after 4 s or more");
      emit(o.text());
    }
    Port p = port_of(*run);
    free_running(c, p, "graph", c.frames);
    run->close_input();
    run->close();
  }
}

void form_model(Ctx& c, Model& model) {
  {
    const auto t0 = Clock::now();
    std::optional<Model::Runner> runner;
    try {
      runner.emplace(model.build(c.inputs.front(), Model::RouteOptions{}, RunOptions{}));
    } catch (const std::exception& e) {
      build_record("model", "lock", false, seconds_since(t0), error_text(e), error_code(e));
      return;
    }
    build_record("model", "lock", true, seconds_since(t0));
    Port p = port_of(*runner);
    lockstep(c, p, "model", "lock", c.inputs, c.labels, c.frames);
    if (!c.sized.empty()) {
      lockstep(c, p, "model", "sizes", c.sized, c.sized_labels, static_cast<long>(c.sized.size()));
    }
    close_record("model", "lock", [&runner] { runner->close(); });
  }
  {
    // The catalogue's multi-component scripts' RunOptions.
    RunOptions ro;
    ro.queue_depth = 4;
    ro.overflow_policy = OverflowPolicy::Block;
    ro.output_memory = OutputMemory::ZeroCopy;
    const auto t0 = Clock::now();
    std::optional<Model::Runner> runner;
    try {
      runner.emplace(model.build(c.inputs.front(), Model::RouteOptions{}, ro));
    } catch (const std::exception& e) {
      build_record("model", "free", false, seconds_since(t0), error_text(e), error_code(e));
      return;
    }
    build_record("model", "free", true, seconds_since(t0));
    Port p = port_of(*runner);
    free_running(c, p, "model", c.frames);
    runner->close();
  }
}

void form_model_run(Ctx& c, Model& model) {
  const auto t0 = Clock::now();
  long ok = 0;
  std::string first_error, code;
  for (long i = 0; i < c.frames; ++i) {
    const std::size_t k = static_cast<std::size_t>(i) % c.inputs.size();
    TensorList out;
    try {
      out = model.run(stamped(c.inputs[k], i), c.timeout_ms);
    } catch (const std::exception& e) {
      first_error = e.what();
      code = error_code(e);
      if (i == 0) {
        build_record("model_run", "run", false, seconds_since(t0), first_error, code);
        return;
      }
      break;
    }
    if (i == 0) {
      build_record("model_run", "run", true, seconds_since(t0));
    }
    Sample s;
    s.kind = SampleKind::TensorSet;
    s.tensors = std::move(out);
    q::Obj o;
    o.str("record", "frame").str("form", "model_run").str("sc", "run").integer("i", i);
    o.str("input", c.labels[k]);
    std::vector<std::string> ts;
    std::map<const void*, int> holders;
    sample_leaves(s, "", ts, holders, c.hash, i < 4);
    o.raw("tensors", q::json_array(ts));
    emit(o.text());
    ++ok;
  }
  q::Obj o;
  o.str("record", "end").str("form", "model_run").str("sc", "run").integer("frames", c.frames);
  o.integer("frames_ok", ok).str("first_error", first_error).num("seconds", seconds_since(t0));
  emit(o.text());
}

void form_plan(Model& model) {
  const auto t0 = Clock::now();
  try {
    const std::string json = model.execution_plan().to_json();
    build_record("plan", "plan", true, seconds_since(t0));
    q::Obj o;
    o.str("record", "plan").str("xxh64", q::hex64(q::xxh64(json.data(), json.size())));
    o.str("json", json);
    emit(o.text());
  } catch (const std::exception& e) {
    build_record("plan", "plan", false, seconds_since(t0), error_text(e), error_code(e));
  }
}

void form_bench(Ctx& c, Model& model) {
  const auto t0 = Clock::now();
  BenchmarkOptions bo;
  bo.num_samples = static_cast<int>(c.args.num("bench-samples", 50));
  bo.min_duration_ms = 0;
  if (c.set.image) {
    bo.inputs = c.inputs.front();
  }
  try {
    const BenchmarkReport r = model.benchmark(bo);
    build_record("bench", "bench", true, seconds_since(t0));
    // Numbers differ between runs: the record says which fields are filled, and their values for
    // the reader.
    q::Obj o;
    o.str("record", "bench").str("input", r.input);
    q::Obj filled, values;
    const std::pair<const char*, double> fields[] = {
        {"latency_ms", r.latency_ms},
        {"latency_p50_ms", r.latency_p50_ms},
        {"latency_p95_ms", r.latency_p95_ms},
        {"latency_p99_ms", r.latency_p99_ms},
        {"fps", r.fps},
        {"throughput_duration_ms", r.throughput_duration_ms},
        {"avg_power_watts", r.avg_power_watts}};
    for (const auto& [name, v] : fields) {
      filled.boolean(name, v > 0.0);
      values.num(name, v);
    }
    filled.boolean("throughput_samples", r.throughput_samples > 0);
    values.integer("throughput_samples", r.throughput_samples);
    o.raw("filled", filled.text()).raw("values", values.text());
    emit(o.text());
  } catch (const std::exception& e) {
    build_record("bench", "bench", false, seconds_since(t0), error_text(e), error_code(e));
  }
}

// Per output of the first frame: the cost of the byte-wise FNV-1a over a copy (the earlier tools'
// method), of XXH64 in place, and of XXH64 over a copy, each the median of 21 repetitions.
void hash_bench(Ctx& c, Model& model) {
  std::optional<Run> run;
  try {
    run.emplace(route_graph(model, c.set.options, c.args.get("route", "full"))
                    .build(c.inputs.front(), RunOptions{}));
  } catch (const std::exception& e) {
    build_record("hashbench", "hashbench", false, 0, e.what(), error_code(e));
    return;
  }
  if (!run->push(c.inputs.front())) {
    return;
  }
  Sample s;
  if (run->pull(c.timeout_ms, s, nullptr) != PullStatus::Ok) {
    return;
  }
  std::vector<const Tensor*> ts;
  if (s.tensor) {
    ts.push_back(&*s.tensor);
  }
  for (const Tensor& t : s.tensors) {
    ts.push_back(&t);
  }
  std::vector<std::uint8_t> scratch;
  for (const Tensor* t : ts) {
    const std::size_t n = t->dense_bytes_tight();
    if (n == 0) {
      continue;
    }
    scratch.resize(n);
    const auto median_us = [](const std::function<void()>& f) {
      std::vector<double> v;
      for (int r = 0; r < 21; ++r) {
        const auto t0 = Clock::now();
        f();
        v.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
      }
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    volatile std::uint64_t sink = 0;
    const double fnv_copy = median_us([&] {
      t->copy_dense_bytes_tight_to(scratch.data(), n);
      sink = q::fnv1a64(scratch.data(), n);
    });
    const double xxh_in_place = median_us([&] {
      const Mapping m = t->map(MapMode::Read);
      sink = q::xxh64(m.data, n);
    });
    const double xxh_copy = median_us([&] {
      t->copy_dense_bytes_tight_to(scratch.data(), n);
      sink = q::xxh64(scratch.data(), n);
    });
    const double copy_only = median_us([&] { t->copy_dense_bytes_tight_to(scratch.data(), n); });
    (void)sink;
    q::Obj o;
    o.str("record", "hashcost").integer("bytes", static_cast<long long>(n));
    o.boolean("contiguous", t->is_contiguous());
    o.num("fnv1a_bytewise_copy_us", fnv_copy).num("xxh64_in_place_us", xxh_in_place);
    o.num("xxh64_copy_us", xxh_copy).num("copy_only_us", copy_only);
    emit(o.text());
  }
  run->close_input();
  run->close();
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (const char ch : s + std::string(1, sep)) {
    if (ch == sep) {
      if (!cur.empty()) {
        out.push_back(cur);
      }
      cur.clear();
    } else {
      cur += ch;
    }
  }
  return out;
}

} // namespace

int main(int argc, char** argv) {
  Ctx c;
  c.args = q::Args::parse(argc, argv);
  const q::Args& a = c.args;
  if (!a.has("backend") || !a.has("package") || !a.has("out")) {
    std::cerr
        << "usage: neat_qualification_compare --backend gstreamer|dataflow --package <tar.gz> "
           "--opts <file|-> --images <dir> --out <file.jsonl> [--forms graph,...]\n";
    return 2;
  }
  // The process's backend, before anything is built.
  set_backend(a.get("backend") == "dataflow" ? RunBackend::Dataflow : RunBackend::GStreamer);
  g_out.open(a.get("out"), std::ios::binary | std::ios::trunc);
  if (!g_out) {
    std::cerr << "cannot write " << a.get("out") << "\n";
    return 2;
  }
  const auto t_start = Clock::now();
  c.timeout_ms = static_cast<int>(a.num("timeout-ms", 20000));
  c.frames = a.num("frames", 200);
  const std::vector<std::string> forms = split(a.get("forms", "graph"), ',');
  try {
    c.set = q::read_option_set(a.get("opts", "-"));
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
  {
    q::Obj o;
    o.str("record", "process").str("backend", a.get("backend")).str("package", a.get("package"));
    o.raw("forms", q::json_strings(forms)).integer("frames", c.frames);
    o.boolean("image", c.set.image).str("frame_format", c.set.image ? c.set.frame_format : "");
    o.raw("options_not_applied", q::json_strings(c.set.not_applied)).str("hash", "xxh64");
    emit(o.text());
  }
  std::optional<Model> model;
  const auto t_model = Clock::now();
  try {
    model.emplace(a.get("package"), c.set.options);
  } catch (const std::exception& e) {
    build_record("model", "construct", false, seconds_since(t_model), error_text(e), error_code(e));
    g_out.flush();
    return 0;
  }
  {
    q::Obj o;
    o.str("record", "model").num("seconds", seconds_since(t_model));
    std::vector<std::string> specs;
    for (const auto& sp : model->input_specs()) {
      std::vector<std::string> dt;
      for (const auto d : sp.dtypes) {
        dt.push_back(dtype_name(d));
      }
      q::Obj s;
      s.raw("shape", q::json_integers(sp.shape)).raw("dtypes", q::json_strings(dt));
      specs.push_back(s.text());
    }
    o.raw("input_specs", q::json_array(specs));
    emit(o.text());
  }
  try {
    if (c.set.image) {
      const std::string dir = a.get("images", ".");
      std::vector<Frame> frames;
      for (int k = 0; k < 4; ++k) {
        frames.push_back(load_bgr(dir + "/img" + std::to_string(k) + ".bgr", 1280, 720));
        c.inputs.push_back(TensorList{frame_tensor(frames.back(), c.set.frame_format)});
        c.labels.push_back("img" + std::to_string(k) + "@1280x720");
      }
      // The 16:9 sizes first: their letterbox has one scale on both axes on every canvas of the
      // catalogue, which is what an oriented-box decode takes (expected_differences.json,
      // oriented_boxes_unequal_scales); the scenario ends at the first frame a backend refuses.
      const std::pair<int, int> sizes[] = {{1280, 720}, {640, 360}, {1024, 576}, {960, 540},
                                           {1120, 630}, {854, 480}, {640, 427},  {720, 480}};
      for (int round = 0; round < 2; ++round) {
        int n = 0;
        for (const auto& [w, h] : sizes) {
          const int k = (n++ + round) % 4;
          c.sized.push_back(TensorList{frame_tensor(
              resized(frames[static_cast<std::size_t>(k)], w, h), c.set.frame_format)});
          c.sized_labels.push_back("img" + std::to_string(k) + "@" + std::to_string(w) + "x" +
                                   std::to_string(h));
        }
      }
    } else {
      const auto specs = model->input_specs();
      for (int salt = 0; salt < 4; ++salt) {
        TensorList in;
        for (std::size_t i = 0; i < specs.size(); ++i) {
          in.push_back(synthetic_tensor(specs[i], i, salt));
        }
        c.inputs.push_back(in);
        c.labels.push_back("synthetic" + std::to_string(salt));
      }
    }
  } catch (const std::exception& e) {
    build_record("inputs", "inputs", false, 0, e.what());
    g_out.flush();
    return 0;
  }
  for (const std::string& form : forms) {
    try {
      if (form == "graph") {
        form_graph(c, *model);
      } else if (form == "model") {
        form_model(c, *model);
      } else if (form == "model_run") {
        form_model_run(c, *model);
      } else if (form == "plan") {
        form_plan(*model);
      } else if (form == "bench") {
        form_bench(c, *model);
      } else {
        build_record(form, "unknown", false, 0, "unknown form");
      }
    } catch (const std::exception& e) {
      build_record(form, "uncaught", false, 0, error_text(e), error_code(e));
    }
    g_out.flush();
  }
  if (a.has("hash-bench")) {
    hash_bench(c, *model);
  }
  q::Obj o;
  o.str("record", "done").num("seconds", seconds_since(t_start));
  o.integer("hashed_tensors", static_cast<long long>(c.hash.tensors));
  o.integer("hashed_bytes", static_cast<long long>(c.hash.bytes));
  o.num("hash_ms", static_cast<double>(c.hash.ns) / 1e6);
  // What the process holds of GStreamer at its end: nothing on Dataflow (qualify.py judges it).
  const sima_test::GStreamerFootprint gst = sima_test::gstreamer_footprint();
  q::Obj g;
  g.boolean("initialized", gst.initialized).raw("types", q::json_strings(gst.types));
  o.raw("gstreamer", g.text());
  emit(o.text());
  g_out.flush();
  return 0;
}

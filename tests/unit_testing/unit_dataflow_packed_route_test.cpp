// An MLA stage whose one output buffer holds several tensors, on the Dataflow backend (H1, step
// Y2), on the public API only. A detection model's MLA writes its heads into one buffer, each at
// its byte offset; when Model::inference() is the route's last stage a pull returns them as a
// TensorList, every tensor a view of the one pulled output. Two graphs, each on the Dataflow
// backend and, for the comparison, on GStreamer. A process runs on one backend (set_backend): the
// test runs twice, --backend=gstreamer --reference=<file> recording what the comparison rows pull,
// then --backend=dataflow --reference=<file> running every row and comparing. No runtime type is
// named.
//   inference:  Input(the MLA's tessellated INT8 tensor) -> Model::inference() -> Output
//               (Model: preprocess off, inference_terminal.mla_only): Feed -> MlaOp -> Fetch
//   preprocess: Input(BGR frame, EV74 memory) -> Model::preprocess() -> Model::inference() ->
//               Output: Feed -> KernelOp(preproc, EV74) -> MlaOp -> Fetch. It needs the EV74 on
//               the Kernels firmware (the stage has no A65 implementation, on either backend).
// It needs a compiled model package whose MLA stage publishes several tensors:
// --model-package <X_mpk.tar.gz>, or SIMA_TEST_PACKED_MODEL_PACKAGE, SIMA_YOLO_TAR. Without one
// the test is skipped (it downloads nothing). YOLOv8s (six heads in one buffer of 1,209,600
// bytes) is the one it was written with.
// Rows:
//   inference        four inputs in rotation on each backend, pulled as TensorList and as
//                    Sample: the number of tensors; every tensor's dtype, layout, shape, strides,
//                    byte offset, route, Semantic; the Sample's tags and labels; every byte of
//                    every tensor. Excluded, printed with both values: the stage's name (GStreamer
//                    numbers its element), the storage's size (GStreamer: the allocation;
//                    Dataflow: the MLA's output buffer), frame_id / sequence numbers / timestamps /
//                    caps_string (as on every Dataflow Run).
//   shared           the tensors of one pull share one storage; one of them kept after the others
//                    and the list are gone keeps that output alive and unchanged while the Run
//                    produces more frames than it has output buffers, and after the Run is closed;
//                    dropping it releases the storage.
//   owned            OutputMemory::Owned: the same tensors over one CPU copy.
//   preprocess       the second graph on each backend, compared as `inference` is, with
//                    every field of Semantic::preprocess on every tensor. On a board whose EV74
//                    runs the stock firmware: the build is refused, naming the firmware.
// (The model's whole route, BoxDecode behind this MLA stage, is
// unit_dataflow_boxdecode_route_test's.)

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "cli_utils.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr int kPatterns = 4;

std::vector<std::int8_t> pattern(std::size_t bytes, unsigned index) {
  std::vector<std::int8_t> out(bytes);
  std::uint32_t x = 0x9e3779b9U * (index + 1U) + 12345U;
  for (auto& value : out) {
    x = x * 1664525U + 1013904223U;
    value = static_cast<std::int8_t>(static_cast<unsigned char>(x >> 24));
  }
  return out;
}

// A BGR "image": smooth gradients with texture, different for every index.
Tensor frame(int width, int height, unsigned index) {
  cv::Mat mat(height, width, CV_8UC3);
  std::uint32_t x = 0x9e3779b9U * (index + 1U) + 977U;
  for (int r = 0; r < height; ++r) {
    std::uint8_t* row = mat.ptr<std::uint8_t>(r);
    for (int c = 0; c < width; ++c) {
      x = x * 1664525U + 1013904223U;
      const unsigned noise = x >> 28;
      row[3 * c + 0] = static_cast<std::uint8_t>((c * 255 / width + noise + 31U * index) & 0xffU);
      row[3 * c + 1] = static_cast<std::uint8_t>((r * 255 / height + noise * 2U) & 0xffU);
      row[3 * c + 2] =
          static_cast<std::uint8_t>(((r + c) * 255 / (width + height) + 17U * index) & 0xffU);
    }
  }
  return Tensor::from_cv_mat(mat, ImageSpec::PixelFormat::BGR, TensorMemory::EV74);
}

bool ev74_runs_kernels_firmware() {
  const auto read = [](const std::string& path) {
    std::ifstream file(path);
    std::string text;
    std::getline(file, text);
    return text;
  };
  for (int i = 0; i < 16; ++i) {
    const std::string dir = "/sys/class/remoteproc/remoteproc" + std::to_string(i) + "/";
    if (read(dir + "name") == "ev74") {
      return read(dir + "state") == "running" &&
             read(dir + "firmware") == "sima-neat-native-cvu-fw";
    }
  }
  return false;
}

Model::Options mla_only_options() {
  Model::Options o;
  o.preprocess.kind = InputKind::Tensor;
  o.preprocess.enable = AutoFlag::Off;
  o.inference_terminal.mla_only = true;
  return o;
}

Model::Options image_options() {
  Model::Options o;
  o.preprocess.kind = InputKind::Image;
  o.preprocess.enable = AutoFlag::On;
  o.preprocess.normalize.enable = AutoFlag::On;
  o.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
  o.decode_type = BoxDecodeType::YoloV8;
  o.score_threshold = 0.49F;
  o.nms_iou_threshold = 0.5F;
  o.top_k = 100;
  return o;
}

RunOptions on() {
  RunOptions o;
  o.queue_depth = 4;
  o.overflow_policy = OverflowPolicy::Block;
  return o;
}

auto output() {
  OutputOptions out;
  out.max_buffers = 4;
  out.sync = false;
  return nodes::Output(out);
}

// The first graph: the MLA alone, its input the tessellated INT8 tensor.
struct Inference {
  Model model;
  std::vector<std::int64_t> shape;
  std::size_t bytes = 1;

  explicit Inference(const std::string& archive) : model(archive, mla_only_options()) {
    const auto specs = model.input_specs();
    require(specs.size() == 1U, "the test model has one input");
    shape = specs.front().shape;
    for (const auto dim : shape) {
      require(dim > 0, "the test model's MLA input shape is fixed");
      bytes *= static_cast<std::size_t>(dim);
    }
  }

  Graph graph() const {
    InputOptions in = model.input_appsrc_options(true);
    in.payload_type = PayloadType::Tensor;
    in.format = FormatTag::INT8;
    in.dtype = TensorDType::Int8;
    Graph g;
    g.add(nodes::Input(in));
    g.add(model.inference());
    g.add(output());
    return g;
  }

  std::vector<TensorList> inputs() const {
    std::vector<TensorList> out;
    for (int p = 0; p < kPatterns; ++p) {
      out.push_back({Tensor::from_vector(pattern(bytes, static_cast<unsigned>(p)), shape,
                                         TensorMemory::MLA)});
    }
    return out;
  }
};

// The second graph: the image preproc in front of the MLA.
Graph image_graph(const Model& model) {
  Graph g;
  g.add(nodes::Input());
  g.add(model.preprocess());
  g.add(model.inference());
  g.add(output());
  return g;
}

std::vector<TensorList> frames_of(int width, int height) {
  std::vector<TensorList> out;
  for (int p = 0; p < kPatterns; ++p) {
    out.push_back({frame(width, height, static_cast<unsigned>(p))});
  }
  return out;
}

std::string ints(const std::vector<std::int64_t>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    out += (i == 0 ? "" : ",") + std::to_string(values[i]);
  }
  return out + "]";
}

// Every field a user reads of a pulled tensor besides its bytes, its storage and the stage's
// name, one "name=value" each: compared as text between the two backends.
std::vector<std::string> fields(const Tensor& t) {
  std::vector<std::string> out;
  const auto add = [&out](const std::string& name, const std::string& value) {
    out.push_back(name + "=" + value);
  };
  const auto real = [](double value) {
    char text[40];
    std::snprintf(text, sizeof text, "%.17g", value);
    return std::string(text);
  };
  add("dtype", std::to_string(static_cast<int>(t.dtype)));
  add("layout", std::to_string(static_cast<int>(t.layout)));
  add("shape", ints(t.shape));
  add("strides_bytes", ints(t.strides_bytes));
  add("byte_offset", std::to_string(t.byte_offset));
  add("axis_semantics", std::to_string(t.axis_semantics.size()));
  add("device",
      std::to_string(static_cast<int>(t.device.type)) + ":" + std::to_string(t.device.id));
  add("read_only", t.read_only ? "1" : "0");
  add("planes", std::to_string(t.planes.size()));
  add("contiguous", t.is_contiguous() ? "1" : "0");
  add("dense_bytes_tight", std::to_string(t.dense_bytes_tight()));
  add("route.logical_index", std::to_string(t.route.logical_index));
  add("route.backend_output_index", std::to_string(t.route.backend_output_index));
  add("route.route_slot", std::to_string(t.route.route_slot));
  add("route.physical_index", std::to_string(t.route.physical_index));
  add("route.memory_index", std::to_string(t.route.memory_index));
  add("route.physical_byte_offset", std::to_string(t.route.physical_byte_offset));
  add("route.name", t.route.name);
  add("route.backend_name", t.route.backend_name);
  add("route.segment_name", t.route.segment_name);
  add("semantic.image", t.semantic.image ? "set" : "unset");
  add("semantic.detection", t.semantic.detection ? t.semantic.detection->format : "unset");
  add("semantic.encoded", t.semantic.encoded ? "set" : "unset");
  add("semantic.byte_stream", t.semantic.byte_stream ? "set" : "unset");
  add("semantic.quant", t.semantic.quant ? "set" : "unset");
  if (t.semantic.quant) {
    const QuantSpec& q = *t.semantic.quant;
    add("quant.scale", real(q.scale));
    add("quant.zero_point", std::to_string(q.zero_point));
    add("quant.axis", std::to_string(q.axis));
    add("quant.scales", std::to_string(q.scales.size()));
    add("quant.zero_points", std::to_string(q.zero_points.size()));
  }
  add("semantic.tess", t.semantic.tess ? "set" : "unset");
  if (t.semantic.tess) {
    add("tess.format", t.semantic.tess->format);
    add("tess.slice_shape", ints(t.semantic.tess->slice_shape));
    add("tess.axes", ints(t.semantic.tess->axes));
    add("tess.storage_bytes", std::to_string(t.semantic.tess->storage_bytes));
  }
  add("semantic.preprocess", t.semantic.preprocess ? "set" : "unset");
  if (t.semantic.preprocess) {
    const PreprocessRuntimeMeta& p = *t.semantic.preprocess;
    add("preprocess.original_width", std::to_string(p.original_width));
    add("preprocess.original_height", std::to_string(p.original_height));
    add("preprocess.resized_width", std::to_string(p.resized_width));
    add("preprocess.resized_height", std::to_string(p.resized_height));
    add("preprocess.scaled_width", std::to_string(p.scaled_width));
    add("preprocess.scaled_height", std::to_string(p.scaled_height));
    add("preprocess.pad_left", std::to_string(p.pad_left));
    add("preprocess.pad_right", std::to_string(p.pad_right));
    add("preprocess.pad_top", std::to_string(p.pad_top));
    add("preprocess.pad_bottom", std::to_string(p.pad_bottom));
    add("preprocess.resize_mode", p.resize_mode);
    add("preprocess.color_in", p.color_in);
    add("preprocess.color_out", p.color_out);
    add("preprocess.axis_perm", std::to_string(p.axis_perm.size()));
    add("preprocess.normalize", p.normalize ? "1" : "0");
    add("preprocess.quantize", p.quantize ? "1" : "0");
    add("preprocess.tessellate", p.tessellate ? "1" : "0");
    add("preprocess.affine_m00", real(p.affine_m00));
    add("preprocess.affine_m01", real(p.affine_m01));
    add("preprocess.affine_m02", real(p.affine_m02));
    add("preprocess.affine_m10", real(p.affine_m10));
    add("preprocess.affine_m11", real(p.affine_m11));
    add("preprocess.affine_m12", real(p.affine_m12));
    add("preprocess.affine_scale_x", real(p.affine_scale_x));
    add("preprocess.affine_scale_y", real(p.affine_scale_y));
    add("preprocess.affine_offset_x", real(p.affine_offset_x));
    add("preprocess.affine_offset_y", real(p.affine_offset_y));
    add("preprocess.roi_list_enabled", p.roi_list_enabled ? "1" : "0");
    add("preprocess.rois", std::to_string(p.rois.size()));
    add("preprocess.roi_input_batch_size", std::to_string(p.roi_input_batch_size));
    add("preprocess.roi_source_width", std::to_string(p.roi_source_width));
    add("preprocess.roi_source_height", std::to_string(p.roi_source_height));
    add("preprocess.roi_source_stride_bytes", std::to_string(p.roi_source_stride_bytes));
    add("preprocess.roi_pad_value", std::to_string(p.roi_pad_value));
    add("preprocess.roi_capacity", std::to_string(p.roi_capacity));
    add("preprocess.roi_valid_count", std::to_string(p.roi_valid_count));
    add("preprocess.roi_input_count", std::to_string(p.roi_input_count));
    add("preprocess.roi_dropped_invalid", std::to_string(p.roi_dropped_invalid));
    add("preprocess.roi_dropped_overflow", std::to_string(p.roi_dropped_overflow));
    add("preprocess.roi_affines", std::to_string(p.roi_affines.size()));
  }
  return out;
}

// The Sample's own fields that are the same on both backends.
std::vector<std::string> sample_fields(const Sample& s) {
  return {"kind=" + std::to_string(static_cast<int>(s.kind)),
          "owned=" + std::to_string(s.owned ? 1 : 0),
          "tensors=" + std::to_string(s.tensors.size()),
          "media_type=" + s.media_type,
          "payload_type=" + std::to_string(static_cast<int>(s.payload_type)),
          "payload_tag=" + s.payload_tag,
          "format=" + s.format,
          "stream_id=" + s.stream_id,
          "port_name=" + s.port_name,
          "output_index=" + std::to_string(s.output_index),
          "logical_output_index=" + std::to_string(s.logical_output_index),
          "route_slot=" + std::to_string(s.route_slot),
          "memory_index=" + std::to_string(s.memory_index)};
}

// One pulled frame, kept without its storage (a pulled tensor holds an output buffer of the Run).
struct Frame {
  std::vector<std::vector<std::string>> tensors; // fields() of each tensor, in the list's order
  std::vector<std::vector<std::uint8_t>> bytes;  // each tensor's dense bytes
  std::vector<std::string> sample;               // empty: pulled as a TensorList
  std::string stage_key, stream_label, segment_name;
  std::size_t storage_bytes = 0;
  std::size_t storage_segments = 0;
  bool one_storage = true;
  bool enveloped = true;
};

Frame keep(const TensorList& tensors) {
  Frame out;
  require(!tensors.empty(), "a pull returns tensors");
  for (const Tensor& tensor : tensors) {
    require(static_cast<bool>(tensor.storage), "a pulled tensor has storage");
    out.tensors.push_back(fields(tensor));
    out.bytes.push_back(tensor.copy_dense_bytes_tight());
    out.one_storage = out.one_storage && tensor.storage == tensors.front().storage;
    out.enveloped = out.enveloped && static_cast<bool>(tensor.envelope) &&
                    tensor.envelope == tensors.front().envelope;
  }
  out.stage_key = tensors.front().route.stage_key;
  out.storage_bytes = tensors.front().storage->size_bytes;
  out.storage_segments = tensors.front().storage->sima_segments.size();
  return out;
}

Frame pull_frame(Run& run, bool as_sample, const std::string& what) {
  if (!as_sample) {
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out), what + ": pull failed: " + (out ? "" : out.error().message));
    Frame kept = keep(*out);
    require(kept.enveloped, what + ": the tensors of one pull carry one envelope");
    return kept;
  }
  Sample sample;
  PullError error;
  require(run.pull(30000, sample, &error) == PullStatus::Ok,
          what + ": Sample pull: " + error.message);
  Frame out = keep(sample.tensors);
  out.sample = sample_fields(sample);
  out.stream_label = sample.stream_label;
  out.segment_name = sample.segment_name;
  return out;
}

std::vector<Frame> run_frames(Graph g, const std::vector<TensorList>& inputs, int frames,
                              const std::string& row) {
  Run run = g.build(inputs[0], on());
  std::vector<Frame> seen;
  for (int i = 0; i < frames; ++i) {
    require(run.push(inputs[static_cast<std::size_t>(i) % inputs.size()]), row + ": push");
    seen.push_back(pull_frame(run, (i % 2) == 1, row));
  }
  run.close_input();
  Sample none;
  PullError closed;
  require(run.pull(5000, none, &closed) == PullStatus::Closed,
          row + ": closed after the drain: " + closed.message);
  run.stop();
  return seen;
}

// A backend's frames, field by field and byte by byte: recorded by the GStreamer run and compared
// by the Dataflow run (`row` names the record's keys). Returns the number of tensors a frame.
std::size_t compare(sima_test::Reference& ref, const std::vector<Frame>& frames,
                    const std::string& row) {
  require(!frames.empty(), row + ": every frame was produced");
  ref.equal(row + ": frames", frames.size());
  const std::size_t count = frames[0].tensors.size();
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const std::string at = row + " frame " + std::to_string(i) + ": ";
    const auto same = [&](const std::vector<std::string>& fields, const std::string& what) {
      ref.equal(at + what + " fields", fields.size());
      for (std::size_t k = 0; k < fields.size(); ++k) {
        ref.equal(at + what + " field " + std::to_string(k), fields[k]);
      }
    };
    ref.equal(at + "tensors", frames[i].tensors.size());
    require(frames[i].tensors.size() == count, at + "every frame has the first frame's tensors");
    for (std::size_t t = 0; t < count; ++t) {
      const std::string which = "tensor " + std::to_string(t);
      same(frames[i].tensors[t], which);
      require(!frames[i].bytes[t].empty(), at + which + ": bytes");
      ref.equal(at + which + " bytes", sima_test::bytes_digest(frames[i].bytes[t]));
      bytes += i == 0 ? frames[i].bytes[t].size() : 0U;
    }
    same(frames[i].sample, "Sample");
    // Dataflow's tensors hold one TensorBuffer (the pulled message); GStreamer's each hold their
    // own over the one sample, so only the bytes and the offsets say that they share a buffer.
    if (sima_test::on_dataflow()) {
      require(frames[i].one_storage, at + "Dataflow's tensors share one storage");
    }
    // GStreamer names the element <stage>_<n>, n counting the elements this process built.
    require(!frames[i].stage_key.empty(), at + "the stage's name is set");
    ref.prefix(at + "route.stage_key", frames[i].stage_key);
    ref.prefix(at + "Sample stream_label", frames[i].stream_label);
    ref.prefix(at + "Sample segment_name", frames[i].segment_name);
  }
  for (int p = 1; p < kPatterns && static_cast<std::size_t>(p) < frames.size(); ++p) {
    require(frames[0].bytes != frames[static_cast<std::size_t>(p)].bytes,
            row + ": the outputs of different inputs differ");
  }
  std::cout << "[" << row << "] " << frames.size() << " frames of " << count << " tensors, "
            << bytes << " bytes a frame, " << frames[0].tensors[0].size() << " fields a tensor\n";
  ref.note(row + " excluded: stage_key", frames[0].stage_key);
  ref.note(row + " excluded: storage", sima_test::text(frames[0].storage_bytes) + " bytes in " +
                                           sima_test::text(frames[0].storage_segments) +
                                           " segment(s)");
  ref.note(row + " excluded: one TensorBuffer for the list", frames[0].one_storage);
  for (std::size_t t = 0; t < count; ++t) {
    std::cout << "  tensor " << t << ":";
    for (const std::string& field : frames[0].tensors[t]) {
      if (field.rfind("shape=", 0) == 0 || field.rfind("byte_offset=", 0) == 0 ||
          field.rfind("route.name=", 0) == 0 || field.rfind("semantic.", 0) == 0) {
        std::cout << " " << field;
      }
    }
    std::cout << "\n";
  }
  return count;
}

void inference(const Inference& route, sima_test::Reference& ref) {
  constexpr int kFrames = 32;
  const std::vector<TensorList> inputs = route.inputs();
  const std::size_t count =
      compare(ref, run_frames(route.graph(), inputs, kFrames, "inference"), "inference");
  require(count > 1U, "the test model's MLA stage publishes several tensors (" +
                          std::to_string(count) + " pulled)");
}

void shared(const Inference& route) {
  constexpr int kMore = 24; // more frames than a Run has output buffers
  const std::vector<TensorList> inputs = route.inputs();
  Graph g = route.graph();
  Run run = g.build(inputs[0], on());
  require(run.push(inputs[0]), "shared: push");
  Tensor kept;
  std::weak_ptr<TensorBuffer> storage;
  std::vector<std::uint8_t> expected;
  std::vector<std::vector<std::uint8_t>> first;
  {
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out) && out->size() > 1U, "shared: a list of several tensors");
    for (const Tensor& tensor : *out) {
      require(tensor.storage == out->front().storage, "shared: one storage for the whole list");
      first.push_back(tensor.copy_dense_bytes_tight());
    }
    const std::size_t pick = out->size() / 2U;
    kept = (*out)[pick];
    storage = kept.storage;
    expected = first[pick];
  } // the list and every other tensor are gone
  require(!storage.expired(), "shared: one kept tensor keeps the output alive");
  for (int i = 1; i <= kMore; ++i) {
    require(run.push(inputs[static_cast<std::size_t>(i % kPatterns)]), "shared: push more");
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out), "shared: pull more");
    require(out->front().storage != kept.storage, "shared: a new pull is another output");
    if ((i % kPatterns) == 0) {
      for (std::size_t t = 0; t < out->size(); ++t) {
        require((*out)[t].copy_dense_bytes_tight() == first[t],
                "shared: the same input gives the same tensors again");
      }
    }
    require(kept.copy_dense_bytes_tight() == expected,
            "shared: the kept tensor's bytes do not change while the Run goes on (frame " +
                std::to_string(i) + ")");
  }
  run.close();
  require(kept.copy_dense_bytes_tight() == expected, "shared: the kept tensor outlives its Run");
  kept = Tensor{};
  require(storage.expired(), "shared: dropping the last tensor releases the output");
  std::cout << "[shared] one of " << first.size() << " tensors kept over " << kMore
            << " further frames and the Run's close: " << expected.size() << " bytes unchanged\n";
}

void owned(const Inference& route) {
  const std::vector<TensorList> inputs = route.inputs();
  std::vector<std::vector<std::uint8_t>> expected;
  std::vector<std::vector<std::string>> described;
  {
    Graph g = route.graph();
    Run run = g.build(inputs[0], on());
    require(run.push(inputs[1]), "owned: push");
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out) &&
                out->front().storage->kind ==
                    (sima_test::on_dataflow() ? StorageKind::DeviceHandle : StorageKind::GstSample),
            "owned: the default output is the MLA's buffer");
    for (const Tensor& tensor : *out) {
      expected.push_back(tensor.copy_dense_bytes_tight());
      described.push_back(fields(tensor));
    }
    run.close();
  }
  Graph g = route.graph();
  RunOptions o = on();
  o.output_memory = OutputMemory::Owned;
  Run run = g.build(inputs[0], o);
  TensorList kept;
  for (int i = 0; i < 3; ++i) {
    require(run.push(inputs[1]), "owned: push");
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out) && out->size() == expected.size(), "owned: the same list");
    kept = std::move(*out);
    for (std::size_t t = 0; t < kept.size(); ++t) {
      require(kept[t].storage && kept[t].storage->kind == StorageKind::CpuOwned &&
                  kept[t].storage == kept.front().storage,
              "owned: every tensor is a view of one CPU copy");
      require(fields(kept[t]) == described[t],
              "owned: tensor " + std::to_string(t) + " is described as the zero-copy one");
      require(kept[t].copy_dense_bytes_tight() == expected[t], "owned: the copy holds the output");
    }
  }
  run.close();
  require(kept.back().copy_dense_bytes_tight() == expected.back(),
          "owned: the copy outlives its Run");
}

void preprocess(const Model& model, sima_test::Reference& ref) {
  constexpr int kFrames = 32;
  const std::vector<TensorList> inputs = frames_of(1280, 720);
  if (!ev74_runs_kernels_firmware()) {
    if (!sima_test::on_dataflow()) {
      return;
    }
    try {
      Graph g = image_graph(model);
      (void)g.build(inputs[0], on());
    } catch (const NeatError& error) {
      require_contains(error.what(), "sima-neat-native-cvu-fw",
                       std::string("preprocess: ") + error.what());
      std::cout << "[preprocess] the EV74 runs the stock firmware: " << error.what() << "\n";
      return;
    }
    throw std::runtime_error("preprocess: the route built on Dataflow on the stock firmware");
  }
  const std::vector<Frame> df = run_frames(image_graph(model), inputs, kFrames, "preprocess");
  const std::size_t count = compare(ref, df, "preprocess");
  require(count > 1U, "preprocess: several tensors");
  for (const std::vector<std::string>& tensor : df[0].tensors) {
    bool has = false;
    for (const std::string& field : tensor) {
      has = has || field == "semantic.preprocess=set";
    }
    require(has, "preprocess: every pulled tensor carries Semantic::preprocess");
  }
  for (const std::string& field : df[0].tensors.back()) {
    if (field.rfind("preprocess.", 0) == 0) {
      std::cout << "  " << field << "\n";
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string archive;
  sima_test::get_arg(argc, argv, "--model-package", archive);
  for (const char* name : {"SIMA_TEST_PACKED_MODEL_PACKAGE", "SIMA_YOLO_TAR"}) {
    if (const char* staged = std::getenv(name); archive.empty() && staged) {
      archive = staged;
    }
  }
  if (archive.empty() || !file_exists(archive)) {
    return skip_test("unit_dataflow_packed_route_test needs a model package whose MLA stage "
                     "publishes several tensors: --model-package <X_mpk.tar.gz>, "
                     "SIMA_TEST_PACKED_MODEL_PACKAGE or SIMA_YOLO_TAR");
  }
  std::string only;
  sima_test::get_arg(argc, argv, "--row", only);
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return sima_test::run_test("unit_dataflow_packed_route_test", [&] {
    sima_test::Reference ref = sima_test::Reference::from_args(argc, argv);
    // GStreamer: the comparison rows only (their record); Dataflow: every row.
    const bool every_row = sima_test::on_dataflow();
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    {
      const Inference route(archive);
      row("inference", [&] { inference(route, ref); });
      if (every_row) {
        row("shared", [&] { shared(route); });
        row("owned", [&] { owned(route); });
      }
    }
    const Model model(archive, image_options());
    row("preprocess", [&] { preprocess(model, ref); });
    if (only.empty()) {
      ref.finish();
    }
  });
}

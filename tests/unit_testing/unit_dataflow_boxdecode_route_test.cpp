// BoxDecode behind the MLA on the Dataflow backend (H1, step Y3), on the public API only: the
// whole route of a detection model,
//   Input(BGR frame, EV74 memory) -> Model::preprocess() -> Model::inference() ->
//   Model::postprocess() -> Output,
// on the Dataflow backend and, for the comparison, on GStreamer. A process runs on one backend
// (set_backend): the test runs twice, --backend=gstreamer --reference=<file> recording what the
// comparison rows pull, then --backend=dataflow --reference=<file> running every row and
// comparing. On Dataflow it is Feed -> KernelOp(preproc, EV74) ->
// MlaOp -> KernelOp(boxdecode, A65: the heads as views of the MLA's one output buffer) -> Core's
// wire serializer -> Fetch. No runtime type is named. It needs the EV74 on the Kernels firmware
// (the image preproc has no A65 implementation, on either backend) and a compiled YOLOv8
// detection package: --model-package <X_mpk.tar.gz>, or SIMA_TEST_BOXDECODE_MODEL_PACKAGE,
// SIMA_YOLO_TAR. Without one the test is skipped (it downloads nothing). YOLOv8s is the one it
// was written with. Frames: generated ones, and in front of them the real images given with
// --frames-dir <dir> (or SIMA_TEST_BOXDECODE_FRAMES): raw 1280x720 BGR files named img<N>.bgr,
// which are what gives boxes.
// Rows:
//   differential     the frames in rotation on each backend, pulled as TensorList and as
//                    Sample: one UInt8 tensor [4 + 24 x top_k]; its dtype, layout, shape, strides,
//                    route, Semantic (detection, the legacy tess tag, every field of preprocess);
//                    the Sample's tags and labels; every byte of the wire; the wire's count and
//                    its zero fill. Excluded, printed with both values: where the wire lies in
//                    its storage (GStreamer: at 2,560 of the stage's 8,192-byte arena; Dataflow:
//                    the whole of its own buffer), the stage's name (GStreamer numbers its
//                    element), frame_id / sequence numbers / timestamps / caps_string (as on
//                    every Dataflow Run).
//   capacity         another top_k: another wire size, the same bytes on both backends.
//   owned            OutputMemory::Owned: the same tensor over a CPU copy.
//   held             a pulled wire kept while the Run produces more frames than it has output
//                    buffers, and after the Run is closed, does not change.
//   refused          with an Input that declares the frame's size, a frame of another size is
//                    refused at the push with a text that says so, and the Run goes on.
//   declined         what stays off this backend at the build: the route without the image
//                    preproc in front (the decode's image geometry would come with each frame),
//                    and a standalone SimaBoxDecode node in the place of Model::postprocess().
// On a board whose EV74 runs the stock firmware every row but `declined` is one check: the build
// is refused, naming the firmware.

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
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
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr int kPatterns = 4;
constexpr int kWidth = 1280;
constexpr int kHeight = 720;

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

Model::Options image_options(int top_k = 100) {
  Model::Options o;
  o.preprocess.kind = InputKind::Image;
  o.preprocess.enable = AutoFlag::On;
  o.preprocess.normalize.enable = AutoFlag::On;
  o.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
  o.decode_type = BoxDecodeType::YoloV8;
  o.score_threshold = 0.49F;
  o.nms_iou_threshold = 0.5F;
  o.top_k = top_k;
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

Graph route_graph(const Model& model) {
  Graph g;
  g.add(nodes::Input());
  g.add(model.preprocess());
  g.add(model.inference());
  g.add(model.postprocess());
  g.add(output());
  return g;
}

std::string ints(const std::vector<std::int64_t>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    out += (i == 0 ? "" : ",") + std::to_string(values[i]);
  }
  return out + "]";
}

// Every field a user reads of a pulled tensor besides its bytes, its storage, where it lies in
// that storage and the stage's name, one "name=value" each: compared as text between the two
// backends.
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
  std::int64_t byte_offset = 0, physical_byte_offset = 0;
  bool enveloped = true;
};

Frame keep(const TensorList& tensors) {
  Frame out;
  require(!tensors.empty(), "a pull returns tensors");
  for (const Tensor& tensor : tensors) {
    require(static_cast<bool>(tensor.storage), "a pulled tensor has storage");
    out.tensors.push_back(fields(tensor));
    out.bytes.push_back(tensor.copy_dense_bytes_tight());
    out.enveloped = out.enveloped && static_cast<bool>(tensor.envelope) &&
                    tensor.envelope == tensors.front().envelope;
  }
  out.stage_key = tensors.front().route.stage_key;
  out.storage_bytes = tensors.front().storage->size_bytes;
  out.byte_offset = tensors.front().byte_offset;
  out.physical_byte_offset = tensors.front().route.physical_byte_offset;
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

std::vector<Frame> run_frames(const Model& model, const std::vector<TensorList>& inputs,
                              RunOptions options, int frames, const std::string& row) {
  Graph g = route_graph(model);
  Run run = g.build(inputs[0], options);
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

std::int32_t wire_count(const std::vector<std::uint8_t>& wire) {
  std::int32_t count = -1;
  if (wire.size() >= sizeof count) {
    std::memcpy(&count, wire.data(), sizeof count);
  }
  return count;
}

// The wire: an int32 count, then top_k rows of 24 bytes (x, y, w, h int32, score float, class
// int32), every byte behind the last box zero.
void require_wire(const std::vector<std::uint8_t>& wire, int top_k, const std::string& at) {
  require(wire.size() == 4U + 24U * static_cast<std::size_t>(top_k),
          at + "the wire is 4 + 24 x top_k bytes (" + std::to_string(wire.size()) + ")");
  const std::int32_t count = wire_count(wire);
  require(count >= 0 && count <= top_k, at + "the wire's count is 0..top_k");
  for (std::size_t i = 4U + 24U * static_cast<std::size_t>(count); i < wire.size(); ++i) {
    require(wire[i] == 0U, at + "the wire is zero behind its last box");
  }
}

std::string boxes_text(const std::vector<std::uint8_t>& wire) {
  std::string out;
  const std::int32_t count = wire_count(wire);
  for (std::int32_t i = 0; i < count; ++i) {
    std::int32_t box[4];
    float score = 0;
    std::int32_t cls = 0;
    const std::uint8_t* row = wire.data() + 4 + 24 * i;
    std::memcpy(box, row, sizeof box);
    std::memcpy(&score, row + 16, sizeof score);
    std::memcpy(&cls, row + 20, sizeof cls);
    char text[96];
    std::snprintf(text, sizeof text, " [x=%d y=%d w=%d h=%d score=%.6f cls=%d]", box[0], box[1],
                  box[2], box[3], static_cast<double>(score), cls);
    out += text;
  }
  return out;
}

// A backend's frames, field by field and byte by byte: recorded by the GStreamer run and compared
// by the Dataflow run (`row` names the record's keys).
void compare(sima_test::Reference& ref, const std::vector<Frame>& frames, int top_k,
             const std::string& row) {
  require(!frames.empty(), row + ": every frame was produced");
  ref.equal(row + ": frames", frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const std::string at = row + " frame " + std::to_string(i) + ": ";
    const auto same = [&](const std::vector<std::string>& fields, const std::string& what) {
      ref.equal(at + what + " fields", fields.size());
      for (std::size_t k = 0; k < fields.size(); ++k) {
        ref.equal(at + what + " field " + std::to_string(k), fields[k]);
      }
    };
    require(frames[i].tensors.size() == 1U,
            at + "one tensor, not " + std::to_string(frames[i].tensors.size()));
    same(frames[i].tensors[0], "tensor");
    require_wire(frames[i].bytes[0], top_k, at);
    ref.equal(at + "the wire's bytes",
              sima_test::bytes_digest(frames[i].bytes[0]) + boxes_text(frames[i].bytes[0]));
    same(frames[i].sample, "Sample");
    // GStreamer names the element <stage>_<n>, n counting the elements this process built.
    require(!frames[i].stage_key.empty(), at + "the stage's name is set");
    ref.prefix(at + "route.stage_key", frames[i].stage_key);
    ref.prefix(at + "Sample stream_label", frames[i].stream_label);
    ref.prefix(at + "Sample segment_name", frames[i].segment_name);
  }
  std::cout << "[" << row << "] " << frames.size() << " frames, " << frames[0].bytes[0].size()
            << " bytes a wire, " << frames[0].tensors[0].size() << " fields a tensor\n";
  ref.note(row + " excluded: stage_key", frames[0].stage_key);
  ref.note(row + " excluded: byte_offset", frames[0].byte_offset);
  ref.note(row + " excluded: route.physical_byte_offset", frames[0].physical_byte_offset);
  ref.note(row + " excluded: storage bytes", frames[0].storage_bytes);
}

// The frames of the test: the real images given, then generated ones.
std::vector<TensorList> frames_of(const std::string& dir, std::size_t* real) {
  std::vector<TensorList> out;
  for (int n = 0; !dir.empty() && n < 64; ++n) {
    std::ifstream file(dir + "/img" + std::to_string(n) + ".bgr", std::ios::binary);
    if (!file) {
      break;
    }
    cv::Mat mat(kHeight, kWidth, CV_8UC3);
    file.read(reinterpret_cast<char*>(mat.data),
              static_cast<std::streamsize>(kHeight) * kWidth * 3);
    require(file.gcount() == static_cast<std::streamsize>(kHeight) * kWidth * 3,
            "a frame file holds one 1280x720 BGR image");
    out.push_back({Tensor::from_cv_mat(mat, ImageSpec::PixelFormat::BGR, TensorMemory::EV74)});
  }
  *real = out.size();
  for (int p = 0; p < kPatterns; ++p) {
    out.push_back({frame(kWidth, kHeight, static_cast<unsigned>(p))});
  }
  return out;
}

void differential(sima_test::Reference& ref, const Model& model,
                  const std::vector<TensorList>& inputs, std::size_t real) {
  const int frames = static_cast<int>(4U * inputs.size());
  const std::vector<Frame> df = run_frames(model, inputs, on(), frames, "differential");
  compare(ref, df, 100, "differential");
  const auto has = [&](const char* field) {
    for (const std::string& value : df[0].tensors[0]) {
      if (value == field) {
        return true;
      }
    }
    return false;
  };
  require(has("semantic.detection=BBOX") && has("tess.format=BBOX") &&
              has("semantic.preprocess=set") && has("shape=[2404]"),
          "differential: the pulled tensor is the BBOX wire with its preprocess facts");
  int boxes = 0;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const std::int32_t count = wire_count(df[i].bytes[0]);
    boxes += count;
    std::cout << "  frame " << i << (i < real ? " (image)" : " (generated)") << ": " << count
              << " boxes" << boxes_text(df[i].bytes[0]) << "\n";
    for (std::size_t again = i + inputs.size(); again < df.size(); again += inputs.size()) {
      require(df[again].bytes[0] == df[i].bytes[0],
              "differential: the same frame gives the same wire again");
    }
  }
  require(real == 0U || boxes > 0, "differential: the real images give boxes");
  for (const std::string& field : df[0].tensors[0]) {
    if (field.rfind("semantic.", 0) == 0 || field.rfind("tess.", 0) == 0 ||
        field.rfind("preprocess.", 0) == 0 || field.rfind("route.", 0) == 0 ||
        field.rfind("shape=", 0) == 0) {
      std::cout << "  " << field << "\n";
    }
  }
}

void capacity(sima_test::Reference& ref, const std::string& archive,
              const std::vector<TensorList>& inputs) {
  constexpr int kTopK = 10;
  const Model model(archive, image_options(kTopK));
  const int frames = static_cast<int>(inputs.size());
  compare(ref, run_frames(model, inputs, on(), frames, "capacity"), kTopK, "capacity");
}

void owned(const Model& model, const std::vector<TensorList>& inputs) {
  const std::vector<Frame> expected = run_frames(model, inputs, on(), 2, "owned");
  Graph g = route_graph(model);
  RunOptions o = on();
  o.output_memory = OutputMemory::Owned;
  Run run = g.build(inputs[0], o);
  Tensor kept;
  for (int i = 0; i < 3; ++i) {
    require(run.push(inputs[0]), "owned: push");
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out) && out->size() == 1U, "owned: one tensor");
    kept = out->front();
    require(kept.storage && kept.storage->kind == StorageKind::CpuOwned,
            "owned: the tensor is a view of a CPU copy");
    require(fields(kept) == expected[0].tensors[0],
            "owned: the tensor is described as the zero-copy one");
    require(kept.copy_dense_bytes_tight() == expected[0].bytes[0],
            "owned: the copy holds the wire");
  }
  run.close();
  require(kept.copy_dense_bytes_tight() == expected[0].bytes[0],
          "owned: the copy outlives its Run");
}

void held(const Model& model, const std::vector<TensorList>& inputs) {
  constexpr int kMore = 24; // more frames than a Run has output buffers
  Graph g = route_graph(model);
  Run run = g.build(inputs[0], on());
  require(run.push(inputs[0]), "held: push");
  Tensor kept;
  std::weak_ptr<TensorBuffer> storage;
  std::vector<std::uint8_t> expected;
  {
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out) && out->size() == 1U, "held: one tensor");
    kept = out->front();
    storage = kept.storage;
    expected = kept.copy_dense_bytes_tight();
  }
  for (int i = 1; i <= kMore; ++i) {
    require(run.push(inputs[static_cast<std::size_t>(i) % inputs.size()]), "held: push more");
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out) && out->front().storage != kept.storage,
            "held: a new pull is another output");
    require(kept.copy_dense_bytes_tight() == expected,
            "held: the kept wire does not change while the Run goes on (frame " +
                std::to_string(i) + ")");
  }
  run.close();
  require(kept.copy_dense_bytes_tight() == expected, "held: the kept wire outlives its Run");
  kept = Tensor{};
  require(storage.expired(), "held: dropping the tensor releases the output");
  std::cout << "[held] one wire kept over " << kMore << " further frames and the Run's close\n";
}

void refused(const Model& model, const std::vector<TensorList>& inputs) {
  const std::vector<Frame> expected = run_frames(model, inputs, on(), 1, "refused");
  // The Input declares the frame's size: the route's kernels are made once, for that size (a
  // seed alone fixes none since step Y4: unit_dataflow_frame_route_test).
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::BGR;
  in.width = kWidth;
  in.height = kHeight;
  Graph g;
  g.add(nodes::Input(in));
  g.add(model.preprocess());
  g.add(model.inference());
  g.add(model.postprocess());
  g.add(output());
  Run run = g.build(inputs[0], on());
  bool threw = false;
  try {
    (void)run.push(TensorList{frame(kWidth / 2, kHeight / 2, 0)});
  } catch (const NeatError& error) {
    threw = true;
    require_contains(error.what(), "where the Input declares",
                     std::string("refused: ") + error.what());
    std::cout << "[refused] a frame of half the size: " << error.what() << "\n";
  }
  require(threw, "refused: a frame of another size is refused at the push");
  require(run.push(inputs[0]), "refused: the Run goes on");
  Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
  require(static_cast<bool>(out) && out->size() == 1U &&
              out->front().copy_dense_bytes_tight() == expected[0].bytes[0],
          "refused: the next frame of the built size gives its wire");
  run.close();
}

void declined(const std::string& archive, const Model& model,
              const std::vector<TensorList>& inputs) {
  // The MLA and BoxDecode without the image preproc: no stage fixes the image the boxes are
  // decoded for. The input is the MLA's tessellated INT8 tensor, as the MLA-only model takes it.
  Model::Options alone;
  alone.preprocess.kind = InputKind::Tensor;
  alone.preprocess.enable = AutoFlag::Off;
  alone.inference_terminal.mla_only = true;
  const Model mla(archive, alone);
  const auto specs = mla.input_specs();
  require(specs.size() == 1U, "declined: the model has one input");
  std::size_t bytes = 1;
  for (const auto dim : specs.front().shape) {
    require(dim > 0, "declined: the model's MLA input shape is fixed");
    bytes *= static_cast<std::size_t>(dim);
  }
  Model::Options o = image_options();
  o.preprocess.kind = InputKind::Tensor;
  o.preprocess.enable = AutoFlag::Off;
  const Model heads(archive, o);
  InputOptions in = mla.input_appsrc_options(true);
  in.payload_type = PayloadType::Tensor;
  in.format = FormatTag::INT8;
  in.dtype = TensorDType::Int8;
  Graph g;
  g.add(nodes::Input(in));
  g.add(heads.inference());
  g.add(heads.postprocess());
  g.add(output());
  const TensorList seed{Tensor::from_vector(std::vector<std::int8_t>(bytes, 0), specs.front().shape,
                                            TensorMemory::MLA)};
  bool built = true;
  try {
    (void)g.build(seed, on());
  } catch (const NeatError& error) {
    require(error.report().error_code == error_codes::kPipelineShape,
            std::string("declined: a build error: ") + error.what());
    require_contains(error.what(), "no image preproc stage in front",
                     std::string("declined: ") + error.what());
    std::cout << "[declined] the MLA and BoxDecode without the image preproc: " << error.what()
              << "\n";
    built = false;
  }
  require(!built, "declined: BoxDecode without the image preproc does not build on Dataflow");
  // A standalone BoxDecode node in the model's place is not a Model stage.
  try {
    Graph standalone;
    standalone.add(nodes::Input());
    standalone.add(model.preprocess());
    standalone.add(model.inference());
    standalone.add(nodes::SimaBoxDecode(model, BoxDecodeType::YoloV8, 0.49, 0.5, 100));
    standalone.add(output());
    (void)standalone.build(inputs[0], on());
  } catch (const NeatError& error) {
    require_contains(error.what(), "a standalone BoxDecode node",
                     std::string("declined: ") + error.what());
    std::cout << "[declined] a standalone SimaBoxDecode node: " << error.what() << "\n";
    return;
  }
  throw std::runtime_error("declined: a standalone SimaBoxDecode node built on Dataflow");
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string archive;
  sima_test::get_arg(argc, argv, "--model-package", archive);
  for (const char* name : {"SIMA_TEST_BOXDECODE_MODEL_PACKAGE", "SIMA_YOLO_TAR"}) {
    if (const char* staged = std::getenv(name); archive.empty() && staged) {
      archive = staged;
    }
  }
  if (archive.empty() || !file_exists(archive)) {
    return skip_test("unit_dataflow_boxdecode_route_test needs a YOLOv8 detection model "
                     "package: --model-package <X_mpk.tar.gz>, "
                     "SIMA_TEST_BOXDECODE_MODEL_PACKAGE or SIMA_YOLO_TAR");
  }
  std::string dir;
  sima_test::get_arg(argc, argv, "--frames-dir", dir);
  if (const char* staged = std::getenv("SIMA_TEST_BOXDECODE_FRAMES"); dir.empty() && staged) {
    dir = staged;
  }
  std::string only;
  sima_test::get_arg(argc, argv, "--row", only);
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return sima_test::run_test("unit_dataflow_boxdecode_route_test", [&] {
    sima_test::Reference ref = sima_test::Reference::from_args(argc, argv);
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    const Model model(archive, image_options());
    std::size_t real = 0;
    const std::vector<TensorList> inputs = frames_of(dir, &real);
    if (!ev74_runs_kernels_firmware()) {
      if (!sima_test::on_dataflow()) {
        return;
      }
      try {
        Graph g = route_graph(model);
        (void)g.build(inputs[0], on());
      } catch (const NeatError& error) {
        require_contains(error.what(), "sima-neat-native-cvu-fw",
                         std::string("stock firmware: ") + error.what());
        std::cout << "[stock firmware] " << error.what() << "\n";
        return;
      }
      throw std::runtime_error("the route built on Dataflow on the stock firmware");
    }
    row("differential", [&] { differential(ref, model, inputs, real); });
    row("capacity", [&] { capacity(ref, archive, inputs); });
    if (!sima_test::on_dataflow()) {
      // GStreamer: the comparison rows' records only.
      ref.finish();
      return;
    }
    row("owned", [&] { owned(model, inputs); });
    row("held", [&] { held(model, inputs); });
    row("refused", [&] { refused(model, inputs); });
    row("declined", [&] { declined(archive, model, inputs); });
    if (only.empty()) {
      ref.finish();
    }
  });
}

// Frames of a size that changes between pushes, and NV12 frames, on the Dataflow backend (H1,
// step Y4), on the public API only: the routes of a detection model,
//   Input(frame, EV74 memory) -> Model::preprocess() [-> Model::inference() [->
//   Model::postprocess()]] -> Output,
// on the Dataflow backend and, for the comparison, on GStreamer, pushed the same frames. A process
// runs on one backend (set_backend): the test runs twice, --backend=gstreamer --reference=<file>
// recording what the comparison rows pull, then --backend=dataflow --reference=<file> running
// every row and comparing. A build whose Input
// declares no frame size (a seed is an example) takes every frame up to the image stage's
// capacity: on Dataflow the preproc and BoxDecode operators keep a kernel per frame size, the
// frame's geometry travels in its stamp's user word, and the pull publishes the frame's own
// preprocess facts. No runtime type is named. It needs the EV74 on the Kernels firmware and a
// compiled YOLOv8 detection package: --model-package <X_mpk.tar.gz>, or
// SIMA_TEST_BOXDECODE_MODEL_PACKAGE, SIMA_YOLO_TAR. Without one the test is skipped (it
// downloads nothing). YOLOv8s is the one it was written with. Another decode family's package
// runs with --opts=<file>, the package's option set in the qualification harness's form: the
// wire then has that task's size and layout (support/boxdecode_wire.h) and the frames say the
// set's format. Frames: generated ones, and in
// front of them the real images given with --frames-dir <dir> (or SIMA_TEST_BOXDECODE_FRAMES):
// raw 1280x720 BGR files named img<N>.bgr, which are what gives boxes.
// Rows (each compares Dataflow with GStreamer for the same pushes unless said: every field a
// user reads of every pulled tensor, Semantic::preprocess among them, the Sample's tags, every
// byte):
//   sizes            the whole route: each image whole, at half its size, whole again (the shape
//                    of yolov8_dynamic_boxdecode_geometry_test), then crops and an enlargement;
//                    the frame's own original size in Semantic::preprocess, and every box inside
//                    the frame it was decoded for.
//   black            Dataflow: 32 black frames give a count of 0 and no error, and the Run goes
//                    on with the frames before.
//   heads            Model::preprocess() and inference(): two frame sizes in turn, six head
//                    tensors a frame (the shape of yolov8_preprocess_meta_async_test).
//   preprocess       Model::preprocess() alone: the tessellated tensor of each frame size.
//   many             eighteen frame sizes in rotation: more than the table and the operators
//                    keep.
//   memory           Dataflow: a frame with the kernel's batch axis, a frame in MLA memory
//                    (copied), a new tensor for every push, sizes in turn: the bytes of `sizes`.
//   free_running     Dataflow: several frames of different sizes in flight at once come out in
//                    order with their own bytes and facts; a Run closed with frames in flight.
//   declared         an Input that declares the frame's size runs the route at that size and
//                    refuses another; one that declares a bound (max_width, max_height) takes
//                    frames up to it.
//   refused          Dataflow: a frame above the Run's capacity (the seed's size: GStreamer's
//                    error), one above the stage's capacity, one with ROI metadata, one of
//                    another channel count and a CPU frame are each refused at the push, and the
//                    Run goes on. A frame with an upstream stage's preprocess metadata without an
//                    ROI is taken, its metadata not read: the bare frame's wire, as on GStreamer.
//   nv12             a model that reads NV12 frames: the frame's two planes as two views of the
//                    input's one buffer; a frame of the declared bound's size (wrapped) and a
//                    smaller one (copied to the port's offsets).
// On a board whose EV74 runs the stock firmware the test is one check: the build is refused,
// naming the firmware.

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "cli_utils.h"
#include "support/backend_selection.h"
#include "support/boxdecode_wire.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kTopK = 100;

// The Model's options and the frames' format: the YOLOv8 ones below, or --opts' (main).
sima_test::DecodeOptions g_decode;

// A BGR "image": smooth gradients with texture, different for every index.
cv::Mat pattern(int width, int height, unsigned index) {
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
  return mat;
}

// A packed frame of the image, in the format the Model's options read (BGR without --opts).
Tensor bgr(const cv::Mat& mat, TensorMemory memory = TensorMemory::EV74) {
  return Tensor::from_cv_mat(mat, g_decode.frame_format, memory);
}

// The NV12 frame of a BGR image (even sizes), in EV74 memory: the luma plane, then the
// interleaved chroma plane, rows not padded.
Tensor nv12(const cv::Mat& image) {
  const int w = image.cols, h = image.rows;
  cv::Mat i420;
  cv::cvtColor(image, i420, cv::COLOR_BGR2YUV_I420);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(w) * h * 3U / 2U);
  std::memcpy(bytes.data(), i420.data, static_cast<std::size_t>(w) * h);
  const std::uint8_t* u = i420.data + static_cast<std::size_t>(w) * h;
  const std::uint8_t* v = u + static_cast<std::size_t>(w / 2) * (h / 2);
  std::uint8_t* uv = bytes.data() + static_cast<std::size_t>(w) * h;
  for (std::size_t i = 0; i < static_cast<std::size_t>(w / 2) * (h / 2); ++i) {
    uv[2U * i] = u[i];
    uv[2U * i + 1U] = v[i];
  }
  Tensor t =
      Tensor::from_vector(bytes, {static_cast<std::int64_t>(bytes.size())}, TensorMemory::EV74);
  t.layout = TensorLayout::HW;
  t.shape = {h, w};
  t.strides_bytes = {w, 1};
  t.semantic.image = ImageSpec{ImageSpec::PixelFormat::NV12, ""};
  Plane luma;
  luma.role = PlaneRole::Y;
  luma.shape = {h, w};
  luma.strides_bytes = {w, 1};
  luma.byte_offset = 0;
  Plane chroma;
  chroma.role = PlaneRole::UV;
  chroma.shape = {h / 2, w};
  chroma.strides_bytes = {w, 1};
  chroma.byte_offset = static_cast<std::int64_t>(w) * h;
  t.planes = {luma, chroma};
  return t;
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

// The options the test was written with: YOLOv8 detection on BGR frames.
Model::Options yolov8_options() {
  Model::Options o;
  o.preprocess.kind = InputKind::Image;
  o.preprocess.enable = AutoFlag::On;
  o.preprocess.normalize.enable = AutoFlag::On;
  o.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
  o.decode_type = BoxDecodeType::YoloV8;
  o.score_threshold = 0.49F;
  o.nms_iou_threshold = 0.5F;
  return o;
}

// The Model's options; `format` set: for frames of that format in place of the packed ones.
Model::Options image_options(std::optional<PreprocessColorFormat> format = std::nullopt) {
  Model::Options o = g_decode.options;
  if (format) {
    o.preprocess.color_convert.input_format = *format;
  }
  o.top_k = kTopK;
  return o;
}

RunOptions on() {
  RunOptions o;
  o.queue_depth = 4;
  o.overflow_policy = OverflowPolicy::Block;
  return o;
}

// The route up to `stages` of the model's stages (1: preprocess, 2: and inference, 3: and
// postprocess), behind an Input with `in` (null: one that declares nothing).
Graph route_graph(const Model& model, int stages = 3, const InputOptions* in = nullptr) {
  OutputOptions out;
  out.max_buffers = 4;
  out.sync = false;
  Graph g;
  g.add(in ? nodes::Input(*in) : nodes::Input());
  g.add(model.preprocess());
  if (stages >= 2) {
    g.add(model.inference());
  }
  if (stages >= 3) {
    g.add(model.postprocess());
  }
  g.add(nodes::Output(out));
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
  std::vector<std::string> parsed; // a wire: what the task's public parser makes of it
  std::vector<std::string> sample;               // empty: pulled as a TensorList
  std::string stage_key;
  int original_width = 0, original_height = 0; // of the first tensor's Semantic::preprocess
};

Frame keep(const TensorList& tensors) {
  Frame out;
  require(!tensors.empty(), "a pull returns tensors");
  for (const Tensor& tensor : tensors) {
    require(static_cast<bool>(tensor.storage), "a pulled tensor has storage");
    out.tensors.push_back(fields(tensor));
    out.bytes.push_back(tensor.has_tiled_storage() || !tensor.is_dense()
                            ? std::vector<std::uint8_t>()
                            : tensor.copy_dense_bytes_tight());
    if (out.bytes.back().empty()) {
      // A tiled tensor is read through its storage.
      const Mapping mapping = tensor.storage->map(MapMode::Read);
      require(mapping.data != nullptr, "a pulled tensor's storage maps for reading");
      const auto* first = static_cast<const std::uint8_t*>(mapping.data);
      out.bytes.back().assign(first, first + mapping.size_bytes);
    }
    out.parsed.push_back(tensor.semantic.detection
                             ? sima_test::parsed_text(g_decode.task, tensor, 0)
                             : std::string());
  }
  out.stage_key = tensors.front().route.stage_key;
  if (tensors.front().semantic.preprocess) {
    out.original_width = tensors.front().semantic.preprocess->original_width;
    out.original_height = tensors.front().semantic.preprocess->original_height;
  }
  return out;
}

Frame pull_frame(Run& run, bool as_sample, const std::string& what) {
  if (!as_sample) {
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out), what + ": pull failed: " + (out ? "" : out.error().message));
    return keep(*out);
  }
  Sample sample;
  PullError error;
  require(run.pull(30000, sample, &error) == PullStatus::Ok,
          what + ": Sample pull: " + error.message);
  Frame out = keep(sample.tensors);
  out.sample = sample_fields(sample);
  return out;
}

// `frames` pushed one after the other, each pulled before the next (odd ones as a Sample), on a
// Run seeded with the first.
std::vector<Frame> run_frames(Graph g, const std::vector<TensorList>& frames, RunOptions options,
                              const std::string& row) {
  Run run = g.build(frames[0], options);
  std::vector<Frame> seen;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    require(run.push(frames[i]), row + ": push " + std::to_string(i));
    seen.push_back(pull_frame(run, (i % 2U) == 1U, row + " frame " + std::to_string(i)));
  }
  run.close_input();
  Sample none;
  PullError closed;
  require(run.pull(5000, none, &closed) == PullStatus::Closed,
          row + ": closed after the drain: " + closed.message);
  require(run.last_error().empty(), row + ": no error on the Run: " + run.last_error());
  run.stop();
  return seen;
}

std::int32_t wire_count(const std::vector<std::uint8_t>& wire) {
  return sima_test::wire_count(g_decode.task, wire);
}

// The wire of the decode's task (support/boxdecode_wire.h): its size for top_k, its count, and
// zeros in all its unused capacity.
void require_wire(const std::vector<std::uint8_t>& wire, int top_k, const std::string& at) {
  sima_test::require_wire(g_decode.task, wire, top_k, at);
}

std::string boxes_text(const std::vector<std::uint8_t>& wire) {
  return sima_test::wire_text(g_decode.task, wire);
}

// The size of the frame a TensorList carries.
void size_of(const TensorList& frame, int& width, int& height) {
  const Tensor& t = frame.front();
  if (const Plane* luma = t.try_plane(PlaneRole::Y)) {
    height = static_cast<int>(luma->shape[0]);
    width = static_cast<int>(luma->shape[1]);
    return;
  }
  const std::size_t lead = t.shape.size() == 4U ? 1U : 0U;
  height = static_cast<int>(t.shape[lead]);
  width = static_cast<int>(t.shape[lead + 1U]);
}

// A backend's frames, field by field and byte by byte: recorded by the GStreamer run and compared
// by the Dataflow run (`row` names the record's keys); and, on Dataflow's, each frame's own
// original size and, for a wire, its boxes inside that frame.
void compare(sima_test::Reference& ref, const std::vector<Frame>& df,
             const std::vector<TensorList>& frames, bool wire, const std::string& row) {
  require(df.size() == frames.size(), row + ": every frame was produced");
  ref.equal(row + ": frames", df.size());
  std::size_t tensors = 0, bytes = 0;
  for (std::size_t i = 0; i < df.size(); ++i) {
    int width = 0, height = 0;
    size_of(frames[i], width, height);
    const std::string at = row + " frame " + std::to_string(i) + " (" + std::to_string(width) +
                           "x" + std::to_string(height) + "): ";
    const auto same = [&](const std::vector<std::string>& fields, const std::string& what) {
      ref.equal(at + what + " fields", fields.size());
      for (std::size_t k = 0; k < fields.size(); ++k) {
        ref.equal(at + what + " field " + std::to_string(k), fields[k]);
      }
    };
    ref.equal(at + "tensors", df[i].tensors.size());
    for (std::size_t t = 0; t < df[i].tensors.size(); ++t) {
      same(df[i].tensors[t], "tensor " + std::to_string(t));
      ref.equal(at + "tensor " + std::to_string(t) + " bytes",
                sima_test::bytes_digest(df[i].bytes[t]) +
                    (wire ? boxes_text(df[i].bytes[t]) : std::string()));
      if (wire) {
        ref.equal(at + "tensor " + std::to_string(t) + " parsed", df[i].parsed[t]);
      }
      ++tensors;
      bytes += df[i].bytes[t].size();
    }
    same(df[i].sample, "Sample");
    if (!sima_test::on_dataflow()) {
      continue;
    }
    require(df[i].original_width == width && df[i].original_height == height,
            at + "Semantic::preprocess is the frame's own (original " +
                std::to_string(df[i].original_width) + "x" + std::to_string(df[i].original_height) +
                ")");
    if (wire) {
      require_wire(df[i].bytes[0], kTopK, at);
      require(sima_test::wire_in_frame(g_decode.task, df[i].bytes[0], kTopK, width, height),
              at + "every detection lies in the frame:" + boxes_text(df[i].bytes[0]));
    }
  }
  std::cout << "[" << row << "] " << df.size() << " frames, " << tensors << " tensors, " << bytes
            << " bytes, " << df[0].tensors[0].size() << " fields a tensor\n";
  ref.note(row + " excluded: stage", df[0].stage_key);
}

// The images of the test: the real ones given, then generated ones.
std::vector<cv::Mat> images_of(const std::string& dir, std::size_t* real) {
  std::vector<cv::Mat> out;
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
    out.push_back(mat);
  }
  *real = out.size();
  for (unsigned p = 0; p < 2U; ++p) {
    out.push_back(pattern(kWidth, kHeight, p));
  }
  return out;
}

cv::Mat resized(const cv::Mat& image, int width, int height) {
  cv::Mat out;
  cv::resize(image, out, cv::Size(width, height), 0.0, 0.0,
             width < image.cols ? cv::INTER_AREA : cv::INTER_LINEAR);
  return out;
}

cv::Mat cropped(const cv::Mat& image, int width, int height) {
  return image(cv::Rect(0, 0, width, height)).clone();
}

// Each image whole, at half its size, whole again; then crops of the first image. (The seed,
// the first frame, is the Run's capacity on both backends: no frame is larger.)
std::vector<TensorList> sized_frames(const std::vector<cv::Mat>& images) {
  std::vector<TensorList> out;
  for (const cv::Mat& image : images) {
    const Tensor whole = bgr(image);
    out.push_back({whole});
    out.push_back({bgr(resized(image, image.cols / 2, image.rows / 2))});
    out.push_back({whole});
  }
  out.push_back({bgr(cropped(images[0], 640, 480))});
  out.push_back({bgr(cropped(images[0], 642, 362))});
  out.push_back({bgr(cropped(images[0], 1000, 700))});
  out.push_back({bgr(images[0])});
  return out;
}

void sizes(sima_test::Reference& ref, const Model& model, const std::vector<cv::Mat>& images,
           std::size_t real, std::vector<Frame>* kept) {
  const std::vector<TensorList> frames = sized_frames(images);
  std::vector<Frame> df = run_frames(route_graph(model), frames, on(), "sizes");
  compare(ref, df, frames, true, "sizes");
  int boxes = 0;
  for (std::size_t i = 0; i < 3U * images.size(); ++i) {
    const std::int32_t count = wire_count(df[i].bytes[0]);
    boxes += count;
    if (i < 6U) {
      std::cout << "  frame " << i << " (" << df[i].original_width << "x" << df[i].original_height
                << "): " << count << " boxes" << boxes_text(df[i].bytes[0]) << "\n";
    }
    require(i % 3U != 2U || df[i].bytes[0] == df[i - 2U].bytes[0],
            "sizes: the whole frame gives the same wire after the half one");
  }
  require(real == 0U || boxes > 0, "sizes: the real images give boxes");
  std::cout << "  " << boxes << " boxes in the " << 3U * images.size() << " image frames\n";
  *kept = std::move(df);
}

void black(const Model& model, const std::vector<cv::Mat>& images,
           const std::vector<Frame>& expected) {
  Graph g = route_graph(model);
  const TensorList whole{bgr(images[0])};
  const TensorList half{bgr(resized(images[0], kWidth / 2, kHeight / 2))};
  const TensorList dark{bgr(cv::Mat::zeros(kHeight, kWidth, CV_8UC3))};
  Run run = g.build(whole, on());
  const auto frame = [&](const TensorList& input, const std::string& what) {
    require(run.push(input), "black: push " + what);
    return pull_frame(run, true, "black " + what);
  };
  require(frame(whole, "whole").bytes[0] == expected[0].bytes[0], "black: the whole frame");
  require(frame(half, "half").bytes[0] == expected[1].bytes[0], "black: the half frame");
  require(frame(whole, "whole again").bytes[0] == expected[0].bytes[0], "black: whole again");
  for (int i = 0; i < 32; ++i) {
    const Frame out = frame(dark, "black frame " + std::to_string(i));
    require_wire(out.bytes[0], kTopK, "black: ");
    require(wire_count(out.bytes[0]) == 0, "black: a black frame gives no detection");
  }
  require(frame(half, "half after").bytes[0] == expected[1].bytes[0],
          "black: the Run goes on with the half frame");
  require(frame(whole, "whole after").bytes[0] == expected[0].bytes[0],
          "black: the Run goes on with the whole frame");
  require(run.last_error().empty(), "black: no error on the Run: " + run.last_error());
  run.close();
  std::cout << "[black] 32 black frames: count 0, no error; the frames before give their wires "
               "after\n";
}

// Two frame sizes in turn through the first `stages` stages.
void two_sizes(sima_test::Reference& ref, const Model& model, const std::vector<cv::Mat>& images,
               int stages, const std::string& row) {
  std::vector<TensorList> frames;
  for (std::size_t i = 0; i < 6U; ++i) {
    const cv::Mat& image = images[i % images.size()];
    frames.push_back({bgr(i % 2U == 0U ? image : resized(image, kWidth / 2, kHeight / 2))});
  }
  compare(ref, run_frames(route_graph(model, stages), frames, on(), row), frames, false, row);
}

void many(sima_test::Reference& ref, const Model& model, const std::vector<cv::Mat>& images) {
  // Eighteen sizes: more than the frames' table and an operator keep (sixteen each). The
  // first, the seed, is the largest: the stage's capacity.
  constexpr int kSizes[18][2] = {{1920, 1080}, {640, 360},  {1280, 720}, {640, 480},  {1000, 700},
                                 {642, 362},   {800, 600},  {320, 240},  {1024, 576}, {960, 540},
                                 {720, 480},   {1600, 900}, {1366, 768}, {854, 480},  {1152, 648},
                                 {480, 270},   {1440, 810}, {576, 324}};
  constexpr std::size_t kCount = sizeof kSizes / sizeof kSizes[0];
  std::vector<TensorList> frames;
  for (int round = 0; round < 3; ++round) {
    for (const auto& size : kSizes) {
      frames.push_back({bgr(resized(images[0], size[0], size[1]))});
    }
  }
  const std::vector<Frame> df = run_frames(route_graph(model), frames, on(), "many");
  compare(ref, df, frames, true, "many");
  for (std::size_t i = kCount; i < df.size(); ++i) {
    require(df[i].bytes[0] == df[i - kCount].bytes[0],
            "many: a size gives the same wire when its kernels are made again");
  }
}

void memory(const Model& model, const std::vector<cv::Mat>& images,
            const std::vector<Frame>& expected) {
  const cv::Mat half = resized(images[0], kWidth / 2, kHeight / 2);
  Graph g = route_graph(model);
  Run run = g.build(TensorList{bgr(images[0])}, on());
  const auto gives = [&](Tensor frame, std::size_t which, const std::string& what) {
    require(run.push(TensorList{std::move(frame)}), "memory: push " + what);
    require(pull_frame(run, false, "memory " + what).bytes[0] == expected[which].bytes[0],
            "memory: " + what + " gives the frame's wire");
  };
  for (int i = 0; i < 4; ++i) {
    // With the kernel's batch axis.
    Tensor batched = bgr(i % 2 == 0 ? images[0] : half);
    batched.shape.insert(batched.shape.begin(), 1);
    batched.strides_bytes.clear();
    gives(batched, static_cast<std::size_t>(i % 2), "a frame with a batch axis");
    // The DMS heap: copied into a slot of the input's EV74 pool.
    gives(bgr(i % 2 == 0 ? half : images[0], TensorMemory::MLA),
          static_cast<std::size_t>((i + 1) % 2), "a TensorMemory::MLA frame");
  }
  // A new tensor for every push, the sizes in turn.
  for (int i = 0; i < 40; ++i) {
    gives(bgr(i % 2 == 0 ? images[0] : half), static_cast<std::size_t>(i % 2), "a new tensor");
  }
  run.close();
  std::cout << "[memory] frames with a batch axis, in MLA memory and in 40 new tensors: the "
               "frames' wires\n";
}

void free_running(const Model& model, const std::vector<cv::Mat>& images,
                  const std::vector<Frame>& expected) {
  const TensorList whole{bgr(images[0])};
  const TensorList half{bgr(resized(images[0], kWidth / 2, kHeight / 2))};
  Graph g = route_graph(model);
  Run run = g.build(whole, on());
  for (int round = 0; round < 8; ++round) {
    for (int i = 0; i < 4; ++i) {
      require(run.push((round + i) % 2 == 0 ? whole : half), "free_running: push");
    }
    for (int i = 0; i < 4; ++i) {
      const std::size_t which = static_cast<std::size_t>((round + i) % 2);
      const Frame out = pull_frame(run, i % 2 == 1, "free_running");
      require(out.bytes[0] == expected[which].bytes[0] &&
                  out.original_width == expected[which].original_width,
              "free_running: frames in flight together come out in order, each with its own "
              "wire and facts");
    }
  }
  // Frames of both sizes in flight at the close.
  require(run.push(whole) && run.push(half) && run.push(whole), "free_running: push");
  run.close();
  std::cout << "[free_running] 32 frames, four in flight, sizes in turn: in order, their own "
               "wires; a close with three frames in flight\n";
}

void declared(sima_test::Reference& ref, const Model& model, const std::vector<cv::Mat>& images) {
  const TensorList whole{bgr(images[0])};
  const TensorList half{bgr(resized(images[0], kWidth / 2, kHeight / 2))};
  const TensorList large{bgr(resized(images[0], 1920, 1080))};
  const auto threw = [](Run& run, const TensorList& frame, const char* text,
                        const std::string& what) {
    try {
      (void)run.push(frame);
    } catch (const NeatError& error) {
      require_contains(error.what(), text, what + ": " + error.what());
      std::cout << "[declared] " << what << ": " << error.what() << "\n";
      return;
    }
    throw std::runtime_error("declared: " + what + " was pushed");
  };
  {
    // The frame's size declared: one kernel a stage, as before step Y4.
    InputOptions in;
    in.payload_type = PayloadType::Image;
    in.format = g_decode.input_format;
    in.width = kWidth;
    in.height = kHeight;
    std::vector<TensorList> frames;
    for (std::size_t i = 0; i < 6U; ++i) {
      frames.push_back({bgr(images[i % images.size()])});
    }
    const std::vector<Frame> df = run_frames(route_graph(model, 3, &in), frames, on(), "declared");
    compare(ref, df, frames, true, "declared");
    if (sima_test::on_dataflow()) {
      Graph g = route_graph(model, 3, &in);
      Run run = g.build(whole, on());
      threw(run, half, "where the Input declares", "a declared size, a frame of half of it");
      require(run.push(whole), "declared: the Run goes on");
      require(pull_frame(run, false, "declared").bytes[0] == df[0].bytes[0],
              "declared: the next frame of the declared size gives its wire");
      run.close();
    }
  }
  {
    // A bound declared: frames up to it, each with its own kernels.
    InputOptions in;
    in.payload_type = PayloadType::Image;
    in.format = g_decode.input_format;
    in.max_width = kWidth;
    in.max_height = kHeight;
    const std::vector<TensorList> frames{whole, half, whole, half};
    const std::vector<Frame> df =
        run_frames(route_graph(model, 3, &in), frames, on(), "declared bound");
    compare(ref, df, frames, true, "declared bound");
    if (!sima_test::on_dataflow()) {
      return; // GStreamer: the comparisons' records only
    }
    Graph g = route_graph(model, 3, &in);
    Run run = g.build(whole, on());
    threw(run, large, "", "a declared bound of 1280x720, a 1920x1080 frame");
    require(run.push(half), "declared: the Run goes on");
    require(pull_frame(run, false, "declared").bytes[0] == df[1].bytes[0],
            "declared: the next frame inside the bound gives its wire");
    run.close();
  }
}

void refused(const Model& model, const std::vector<cv::Mat>& images,
             const std::vector<Frame>& expected) {
  const TensorList whole{bgr(images[0])};
  Graph g = route_graph(model);
  Run run = g.build(whole, on());
  const auto fails = [&](const Tensor& frame, const char* text, const char* what) {
    try {
      (void)run.push(TensorList{frame});
    } catch (const NeatError& error) {
      require(error.report().error_code == error_codes::kInputShape,
              std::string(what) + ": an input-shape error: " + error.what());
      require_contains(error.what(), text, std::string(what) + ": " + error.what());
      std::cout << "[refused] " << what << ": " << error.what() << "\n";
      return;
    }
    throw std::runtime_error(std::string("refused: ") + what + " was pushed");
  };
  const auto goes_on = [&](const char* what) {
    require(run.push(whole), std::string("refused: push after ") + what);
    require(pull_frame(run, false, "refused").bytes[0] == expected[0].bytes[0],
            std::string("refused: the Run goes on after ") + what);
  };
  // Above the Run's capacity (the seed's size): GStreamer's refusal.
  try {
    (void)run.push(TensorList{bgr(resized(images[0], 1920, 1080))});
    throw std::runtime_error("refused: a frame above the seed's size was pushed");
  } catch (const NeatError& error) {
    require(error.report().error_code == error_codes::kInputCapacity,
            std::string("refused: a capacity error: ") + error.what());
    require_contains(error.what(), "exceeds configured capacity 1280",
                     std::string("refused: ") + error.what());
    std::cout << "[refused] a frame above the seed's size: " << error.what() << "\n";
  }
  goes_on("a frame above the Run's capacity");
  {
    // Above the stage's capacity, on a Run whose seed is that capacity.
    Graph large = route_graph(model);
    Run at_capacity = large.build(TensorList{bgr(resized(images[0], 1920, 1080))}, on());
    try {
      (void)at_capacity.push(TensorList{bgr(resized(images[0], 1922, 1082))});
      throw std::runtime_error("refused: a frame above the stage's capacity was pushed");
    } catch (const NeatError& error) {
      require_contains(error.what(), "takes frames up to 1920x1080",
                       std::string("refused: ") + error.what());
      std::cout << "[refused] a frame above the stage's capacity: " << error.what() << "\n";
    }
    require(at_capacity.push(whole), "refused: the Run goes on");
    require(pull_frame(at_capacity, false, "refused").bytes[0] == expected[0].bytes[0],
            "refused: a smaller frame gives its wire after the refusal");
    at_capacity.close();
  }
  Tensor roi = bgr(images[0]);
  PreprocessRuntimeMeta meta;
  meta.original_width = kWidth;
  meta.original_height = kHeight;
  meta.roi_list_enabled = true;
  meta.roi_input_batch_size = 1;
  meta.rois.push_back({0, 16, 16, 320, 240});
  roi.semantic.preprocess = meta;
  fails(roi, "ROI metadata", "a frame with an ROI");
  Tensor upstream = bgr(images[0]);
  PreprocessRuntimeMeta before;
  before.original_width = 2 * kWidth;
  before.original_height = 2 * kHeight;
  before.affine_m00 = 2.0;
  before.affine_m11 = 2.0;
  upstream.semantic.preprocess = before;
  require(run.push(TensorList{upstream}), "refused: a frame with an upstream transform is taken");
  require(pull_frame(run, false, "upstream").bytes[0] == expected[0].bytes[0],
          "refused: an upstream stage's preprocess metadata without an ROI is not read");
  goes_on("the metadata refusal");
  bool gray = false;
  try {
    (void)run.push(
        TensorList{Tensor::from_cv_mat(cv::Mat(kHeight, kWidth, CV_8UC1, cv::Scalar(7)),
                                       ImageSpec::PixelFormat::GRAY8, TensorMemory::EV74)});
  } catch (const std::exception& error) {
    gray = true;
    std::cout << "[refused] a GRAY8 frame: " << error.what() << "\n";
  }
  require(gray, "refused: a frame of another channel count is refused");
  bool cpu = false;
  try {
    (void)run.push(TensorList{bgr(images[0], TensorMemory::CPU)});
  } catch (const std::exception& error) {
    cpu = true;
    std::cout << "[refused] a CPU frame: " << error.what() << "\n";
  }
  require(cpu, "refused: a CPU frame is refused (GStreamer's guard of a device-visible route)");
  goes_on("every refusal");
  run.close();
}

void nv12_frames(sima_test::Reference& ref, const std::string& archive,
                 const std::vector<cv::Mat>& images) {
  const Model model(archive, image_options(PreprocessColorFormat::NV12));
  std::vector<TensorList> frames;
  for (std::size_t i = 0; i < 8U; ++i) {
    const cv::Mat& image = images[i % images.size()];
    frames.push_back({nv12(i % 4U == 1U ? resized(image, kWidth / 2, kHeight / 2) : image)});
  }
  for (int stages : {1, 3}) {
    const std::string row = stages == 1 ? "nv12 preprocess" : "nv12";
    // The Input declares the largest frame: frames of that size are the port's bytes as they
    // are (wrapped), the half ones are copied to the port's offsets.
    InputOptions in;
    in.payload_type = PayloadType::Image;
    in.format = FormatTag::NV12;
    in.max_width = kWidth;
    in.max_height = kHeight;
    // The reference, GStreamer with the bound of 1280x720 declared, is recorded once for each of
    // the three Dataflow builds below.
    const std::string rows[] = {row + ", a declared bound of 1280x720", row + ", no declaration",
                                row + ", a declared bound of 1920x1080"};
    if (!sima_test::on_dataflow()) {
      const std::vector<Frame> gst = run_frames(route_graph(model, stages, &in), frames, on(), row);
      for (const std::string& name : rows) {
        compare(ref, gst, frames, stages == 3, name);
      }
      continue;
    }
    compare(ref, run_frames(route_graph(model, stages, &in), frames, on(), row), frames,
            stages == 3, rows[0]);
    // Nothing declared: the port is the seed's frame, as above.
    compare(ref, run_frames(route_graph(model, stages), frames, on(), row), frames, stages == 3,
            rows[1]);
    // A larger bound declared: the port is 1920x1080 and every frame is copied to its offsets.
    InputOptions wide = in;
    wide.max_width = 1920;
    wide.max_height = 1080;
    compare(ref, run_frames(route_graph(model, stages, &wide), frames, on(), row), frames,
            stages == 3, rows[2]);
  }
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
    return skip_test("unit_dataflow_frame_route_test needs a YOLOv8 detection model package: "
                     "--model-package <X_mpk.tar.gz>, SIMA_TEST_BOXDECODE_MODEL_PACKAGE or "
                     "SIMA_YOLO_TAR");
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
  return sima_test::run_test("unit_dataflow_frame_route_test", [&] {
    g_decode = sima_test::decode_options(argc, argv, yolov8_options());
    sima_test::Reference ref = sima_test::Reference::from_args(argc, argv);
    const bool dataflow = sima_test::on_dataflow();
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name && std::string(name) != "sizes") {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    const Model model(archive, image_options());
    std::size_t real = 0;
    const std::vector<cv::Mat> images = images_of(dir, &real);
    if (!ev74_runs_kernels_firmware()) {
      if (!dataflow) {
        return;
      }
      try {
        Graph g = route_graph(model);
        (void)g.build(TensorList{bgr(images[0])}, on());
      } catch (const NeatError& error) {
        require_contains(error.what(), "sima-neat-native-cvu-fw",
                         std::string("stock firmware: ") + error.what());
        std::cout << "[stock firmware] " << error.what() << "\n";
        return;
      }
      throw std::runtime_error("the route built on Dataflow on the stock firmware");
    }
    // `sizes` runs with every row: the others compare with its frames 0 (whole) and 1 (half).
    std::vector<Frame> expected;
    // On GStreamer the comparison rows only (their record); on Dataflow every row.
    row("sizes", [&] { sizes(ref, model, images, real, &expected); });
    if (dataflow) {
      row("black", [&] { black(model, images, expected); });
    }
    row("heads", [&] { two_sizes(ref, model, images, 2, "heads"); });
    row("preprocess", [&] { two_sizes(ref, model, images, 1, "preprocess"); });
    row("many", [&] { many(ref, model, images); });
    if (dataflow) {
      row("memory", [&] { memory(model, images, expected); });
      row("free_running", [&] { free_running(model, images, expected); });
    }
    row("declared", [&] { declared(ref, model, images); });
    if (dataflow) {
      row("refused", [&] { refused(model, images, expected); });
    }
    row("nv12", [&] { nv12_frames(ref, archive, images); });
    if (only.empty() || !dataflow) {
      ref.finish();
    }
  });
}

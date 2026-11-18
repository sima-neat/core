// The image preproc stage on the Dataflow backend (H1, step Y1), on the public API only:
//   Input -> Model::preprocess() -> Output
// of a detection model whose pre stage is the Kernels package's "preproc" (EV74 only: resize,
// colour, normalize, quantize and tessellate a packed BGR frame), built with RunOptions::backend
// = RunBackend::Dataflow (Feed -> KernelOp -> Fetch, one kernel at the frame size the input has
// at the build) and, for the comparison rows, with RunBackend::GStreamer (appsrc !
// neatkernelrunner ! appsink, whose frame adapter gives the kernel each frame's geometry) in the
// same process. No runtime type is named.
// It needs the EV74 on the Kernels firmware (the stage has no A65 implementation, on either
// backend) and a compiled model package with an image input: --model-package <X_mpk.tar.gz>, or
// SIMA_TEST_PREPROC_MODEL_PACKAGE, SIMA_YOLO_TAR. Without either the test is skipped (it
// downloads nothing). YOLOv8s (640x640 INT8, letterbox) is the one it was written with.
// Rows:
//   differential     four 1280x720 BGR frames in rotation through both backends, pulled as
//                    TensorList and as Sample: every pulled tensor's dtype, layout, shape,
//                    strides, byte offset, route, tessellation description and every field of
//                    Semantic::preprocess; the Sample's tags and labels; and every byte of the
//                    tiled tensor. Excluded, printed with both values: the stage's name (GStreamer
//                    numbers its element), frame_id / sequence numbers / timestamps / caps_string
//                    (as on every Dataflow Run).
//   sizes            the same at 640x480, at 642x362 (a row pitch that is not a multiple of four)
//                    and at 1920x1080 (the model's capacity envelope): the kernel request and the
//                    preprocess facts come from the stage's policy for the size, not from numbers
//                    in the backend.
//   declared         a build without a seed whose Input declares the frame (format, width,
//                    height) runs on Dataflow and gives the seeded build's bytes.
//   input_contract   a frame with a batch axis of 1, a TensorMemory::MLA frame (copied into the
//                    input's EV74 pool) and a new tensor for every frame give the same bytes; a
//                    CPU tensor is refused.
//   refused          a frame of another size than the Run was built for, and a frame with ROI
//                    metadata, fail their push with misconfig.input_shape and the Run goes on.
//   declined         what stays off this backend at the build: a build that does not say the
//                    frame's size and an NV12 input (explicit Dataflow fails); under
//                    RunBackend::Auto with SIMA_NEAT_BACKEND=dataflow a graph it declines (a
//                    standalone BoxDecode node behind the model's stages) is GStreamer's.
//   lifecycle        close_input drains what was pushed; Runs one after the other give the same
//                    bytes.
// On a board whose EV74 runs the stock firmware every Dataflow build of the route must be refused
// at the build, and the test checks that instead (row firmware).

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "cli_utils.h"
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
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr int kPatterns = 4;

// A BGR "image": smooth gradients with texture, different for every index, so that the resize
// and the colour conversion have something to do and frames are told apart.
cv::Mat image(int width, int height, unsigned index) {
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

Tensor frame(int width, int height, unsigned index, TensorMemory memory = TensorMemory::EV74) {
  return Tensor::from_cv_mat(image(width, height, index), ImageSpec::PixelFormat::BGR, memory);
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

Model::Options model_options(PreprocessColorFormat format = PreprocessColorFormat::BGR) {
  Model::Options o;
  o.preprocess.kind = InputKind::Image;
  o.preprocess.enable = AutoFlag::On;
  o.preprocess.normalize.enable = AutoFlag::On;
  o.preprocess.color_convert.input_format = format;
  o.decode_type = BoxDecodeType::YoloV8;
  o.score_threshold = 0.49F;
  o.nms_iou_threshold = 0.5F;
  o.top_k = 100;
  return o;
}

Graph graph(const Model& model, const InputOptions* in = nullptr) {
  OutputOptions out;
  out.max_buffers = 4;
  out.sync = false;
  Graph g;
  g.add(in ? nodes::Input(*in) : nodes::Input());
  g.add(model.preprocess());
  g.add(nodes::Output(out));
  return g;
}

RunOptions on(RunBackend backend) {
  RunOptions o;
  o.backend = backend;
  o.queue_depth = 4;
  o.overflow_policy = OverflowPolicy::Block;
  return o;
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
  add("semantic.quant", t.semantic.quant ? "set" : "unset");
  add("semantic.detection", t.semantic.detection ? t.semantic.detection->format : "unset");
  add("semantic.encoded", t.semantic.encoded ? "set" : "unset");
  add("semantic.byte_stream", t.semantic.byte_stream ? "set" : "unset");
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
    const auto real = [](double value) {
      char text[40];
      std::snprintf(text, sizeof text, "%.17g", value);
      return std::string(text);
    };
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
          "media_type=" + s.media_type,
          "payload_type=" + std::to_string(static_cast<int>(s.payload_type)),
          "payload_tag=" + s.payload_tag,
          "format=" + s.format,
          "stream_id=" + s.stream_id,
          "stream_label=" + s.stream_label,
          "segment_name=" + s.segment_name,
          "port_name=" + s.port_name,
          "output_index=" + std::to_string(s.output_index),
          "logical_output_index=" + std::to_string(s.logical_output_index),
          "route_slot=" + std::to_string(s.route_slot),
          "memory_index=" + std::to_string(s.memory_index)};
}

// The tiled tensor's bytes: its storage, mapped (a dense read of a tiled tensor is not offered).
std::vector<std::uint8_t> bytes_of(const Tensor& tensor) {
  const Mapping map = tensor.map(MapMode::Read);
  require(map.data != nullptr && map.size_bytes != 0U, "a pulled tensor's storage maps for read");
  const auto* data = static_cast<const std::uint8_t*>(map.data);
  return {data, data + map.size_bytes};
}

struct Frame {
  std::vector<std::string> tensor;
  std::vector<std::string> sample; // empty: pulled as a TensorList
  std::vector<std::uint8_t> bytes;
  std::string stage_key;
  std::size_t storage_bytes = 0;
  std::size_t storage_segments = 0;
};

Frame keep(const Tensor& tensor) {
  Frame out;
  out.tensor = fields(tensor);
  out.bytes = bytes_of(tensor);
  out.stage_key = tensor.route.stage_key;
  require(static_cast<bool>(tensor.storage), "a pulled tensor has storage");
  out.storage_bytes = tensor.storage->size_bytes;
  out.storage_segments = tensor.storage->sima_segments.size();
  return out;
}

Frame pull_frame(Run& run, bool as_sample, const std::string& what) {
  if (!as_sample) {
    Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(out), what + ": pull failed: " + (out ? "" : out.error().message));
    require(out->size() == 1U, what + ": one tensor per frame");
    return keep(out->front());
  }
  Sample sample;
  PullError error;
  require(run.pull(30000, sample, &error) == PullStatus::Ok,
          what + ": Sample pull: " + error.message);
  require(sample.tensors.size() == 1U, what + ": the Sample holds one tensor");
  Frame out = keep(sample.tensors.front());
  out.sample = sample_fields(sample);
  return out;
}

std::vector<Frame> run_frames(Graph g, const std::vector<TensorList>& inputs, RunBackend backend,
                              int frames) {
  Run run = g.build(inputs[0], on(backend));
  std::vector<Frame> seen;
  for (int i = 0; i < frames; ++i) {
    require(run.push(inputs[static_cast<std::size_t>(i) % inputs.size()]), "push");
    seen.push_back(pull_frame(run, (i % 2) == 1, "differential"));
  }
  run.close_input();
  Sample none;
  PullError closed;
  require(run.pull(5000, none, &closed) == PullStatus::Closed,
          "closed after the drain: " + closed.message);
  run.stop();
  return seen;
}

void compare(const std::vector<Frame>& gst, const std::vector<Frame>& df, const std::string& row) {
  require(gst.size() == df.size(), row + ": both backends produced every frame");
  for (std::size_t i = 0; i < gst.size(); ++i) {
    const std::string at = row + " frame " + std::to_string(i) + ": ";
    const auto same = [&](const std::vector<std::string>& a, const std::vector<std::string>& b,
                          const char* what) {
      require(a.size() == b.size(), at + what + ": the same number of fields");
      for (std::size_t k = 0; k < a.size(); ++k) {
        require(a[k] == b[k],
                at + what + " field differs: GStreamer " + a[k] + ", Dataflow " + b[k]);
      }
    };
    same(gst[i].tensor, df[i].tensor, "tensor");
    same(gst[i].sample, df[i].sample, "Sample");
    require(gst[i].stage_key.rfind(df[i].stage_key, 0) == 0 && !df[i].stage_key.empty(),
            at + "route.stage_key: GStreamer '" + gst[i].stage_key + "' begins with Dataflow's '" +
                df[i].stage_key + "'");
    require(gst[i].bytes.size() == df[i].bytes.size() && gst[i].bytes == df[i].bytes,
            at + "the tensor's bytes are the same (" + std::to_string(gst[i].bytes.size()) +
                " and " + std::to_string(df[i].bytes.size()) + ")");
  }
  std::cout << "[" << row << "] " << gst.size() << " frames, " << gst[0].tensor.size()
            << " tensor fields and " << gst[1].sample.size() << " Sample fields and "
            << gst[0].bytes.size() << " bytes each the same; stage_key GStreamer '"
            << gst[0].stage_key << "' Dataflow '" << df[0].stage_key << "'; storage GStreamer "
            << gst[0].storage_bytes << " bytes in " << gst[0].storage_segments
            << " segment(s), Dataflow " << df[0].storage_bytes << " in " << df[0].storage_segments
            << "\n";
  for (const std::string& field : df[0].tensor) {
    if (field.rfind("preprocess.", 0) == 0 || field.rfind("tess.", 0) == 0) {
      std::cout << "  " << field << "\n";
    }
  }
}

std::vector<TensorList> frames_of(int width, int height) {
  std::vector<TensorList> inputs;
  for (int p = 0; p < kPatterns; ++p) {
    inputs.push_back(TensorList{frame(width, height, static_cast<unsigned>(p))});
  }
  return inputs;
}

void differential(const Model& model, int width, int height, int frames, const std::string& row) {
  const std::vector<TensorList> inputs = frames_of(width, height);
  compare(run_frames(graph(model), inputs, RunBackend::GStreamer, frames),
          run_frames(graph(model), inputs, RunBackend::Dataflow, frames), row);
}

// The first `count` outputs of a Dataflow Run built with `seed` (or, without one, from `in`).
std::vector<std::vector<std::uint8_t>> outputs(const Model& model,
                                               const std::vector<TensorList>& inputs,
                                               const InputOptions* in, bool seeded) {
  Graph g = graph(model, in);
  Run run =
      seeded ? g.build(inputs[0], on(RunBackend::Dataflow)) : g.build(on(RunBackend::Dataflow));
  std::vector<std::vector<std::uint8_t>> out;
  for (const TensorList& input : inputs) {
    require(run.push(input), "push");
    out.push_back(pull_frame(run, false, "outputs").bytes);
  }
  run.close();
  return out;
}

void declared(const Model& model) {
  const std::vector<TensorList> inputs = frames_of(1280, 720);
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::BGR;
  in.width = 1280;
  in.height = 720;
  require(outputs(model, inputs, &in, false) == outputs(model, inputs, nullptr, true),
          "declared: a build whose Input declares the frame gives the seeded build's bytes");
}

void input_contract(const Model& model) {
  const std::vector<TensorList> inputs = frames_of(1280, 720);
  const std::vector<std::vector<std::uint8_t>> expected = outputs(model, inputs, nullptr, true);
  Graph g = graph(model);
  Run run = g.build(inputs[0], on(RunBackend::Dataflow));
  for (int i = 0; i < 2 * kPatterns; ++i) {
    const auto p = static_cast<std::size_t>(i % kPatterns);
    // With the kernel's batch axis.
    Tensor batched = inputs[p].front();
    batched.shape.insert(batched.shape.begin(), 1);
    batched.strides_bytes.clear();
    require(run.push({batched}), "input_contract: push [1,H,W,3]");
    require(pull_frame(run, false, "batched").bytes == expected[p],
            "input_contract: a frame with a batch axis of 1 gives the same bytes");
    // The DMS heap: copied into a slot of the input's EV74 pool.
    require(run.push({frame(1280, 720, static_cast<unsigned>(p), TensorMemory::MLA)}),
            "input_contract: push a TensorMemory::MLA frame");
    require(pull_frame(run, false, "mla").bytes == expected[p],
            "input_contract: a TensorMemory::MLA frame gives the EV74 frame's bytes");
  }
  // A new tensor for every frame.
  for (int i = 0; i < 40; ++i) {
    const auto p = static_cast<std::size_t>(i % kPatterns);
    require(run.push({frame(1280, 720, static_cast<unsigned>(p))}),
            "input_contract: push a new tensor");
    require(pull_frame(run, false, "new").bytes == expected[p],
            "input_contract: new tensor " + std::to_string(i) + " gives its image's bytes");
  }
  // A CPU tensor: GStreamer's guard of a device-visible route.
  bool refused = false;
  try {
    (void)run.push({frame(1280, 720, 0, TensorMemory::CPU)});
  } catch (const std::exception& error) {
    refused = true;
    std::cout << "[input_contract] a CPU frame: " << error.what() << "\n";
  }
  require(refused, "input_contract: a CPU frame is refused");
  require(run.push(inputs[1]), "input_contract: push after the refusal");
  require(pull_frame(run, false, "after").bytes == expected[1],
          "input_contract: the Run goes on after a refused push");
  run.close();
}

void refused(const Model& model) {
  const std::vector<TensorList> inputs = frames_of(1280, 720);
  const std::vector<std::vector<std::uint8_t>> expected = outputs(model, inputs, nullptr, true);
  Graph g = graph(model);
  Run run = g.build(inputs[0], on(RunBackend::Dataflow));
  const auto fails = [&](const TensorList& item, const char* text, const char* what) {
    try {
      (void)run.push(item);
    } catch (const NeatError& error) {
      require(error.report().error_code == error_codes::kInputShape,
              std::string(what) + ": an input-shape error: " + error.what());
      require_contains(error.what(), text, std::string(what) + ": " + error.what());
      std::cout << "[refused] " << what << ": " << error.what() << "\n";
      return;
    }
    throw std::runtime_error(std::string("refused: ") + what + " was pushed");
  };
  fails({frame(640, 480, 0)}, "where the Run was built for 1280x720", "a 640x480 frame");
  Tensor roi = frame(1280, 720, 0);
  PreprocessRuntimeMeta meta;
  meta.original_width = 1280;
  meta.original_height = 720;
  meta.roi_list_enabled = true;
  meta.roi_input_batch_size = 1;
  meta.rois.push_back({0, 16, 16, 320, 240});
  roi.semantic.preprocess = meta;
  fails({roi}, "ROI metadata", "a frame with an ROI");
  for (int p = 0; p < kPatterns; ++p) {
    require(run.push(inputs[static_cast<std::size_t>(p)]), "refused: push after the refusals");
    require(pull_frame(run, false, "after").bytes == expected[static_cast<std::size_t>(p)],
            "refused: the Run goes on with the frames it was built for");
  }
  run.close();
}

void declined(const Model& model, const std::string& archive) {
  const std::vector<TensorList> inputs = frames_of(1280, 720);
  const auto fails = [&](Graph g, bool seeded, const char* text, const char* what) {
    try {
      (void)(seeded ? g.build(inputs[0], on(RunBackend::Dataflow))
                    : g.build(on(RunBackend::Dataflow)));
    } catch (const NeatError& error) {
      require_contains(error.what(), text, std::string(what) + ": " + error.what());
      std::cout << "[declined] " << what << ": " << error.what() << "\n";
      return;
    }
    throw std::runtime_error(std::string("declined: ") + what + " built on the Dataflow backend");
  };
  fails(graph(model), false, "the frame's size is not known at the build",
        "no seed, no declaration");
  {
    // The model told its frames are NV12 (two planes): whatever the seed, the stage is declined.
    const Model nv12(archive, model_options(PreprocessColorFormat::NV12));
    fails(graph(nv12), true, "only packed RGB and BGR frames", "an NV12 input");
  }
  {
    // A seed that is not a packed three-channel frame.
    bool built = false;
    try {
      const TensorList gray{Tensor::from_cv_mat(cv::Mat(720, 1280, CV_8UC1, cv::Scalar(7)),
                                                ImageSpec::PixelFormat::GRAY8, TensorMemory::EV74)};
      Graph g = graph(model);
      (void)g.build(gray, on(RunBackend::Dataflow));
      built = true;
    } catch (const std::exception& error) {
      std::cout << "[declined] a GRAY8 seed: " << error.what() << "\n";
    }
    require(!built, "declined: a GRAY8 seed does not build on the Dataflow backend");
  }
  // Under RunBackend::Auto with SIMA_NEAT_BACKEND=dataflow a declined build is GStreamer's: the
  // model's stages with a standalone BoxDecode node behind them (not a Model stage, so not a
  // route this backend takes) run there and give their one detection tensor.
  ::setenv("SIMA_NEAT_BACKEND", "dataflow", 1);
  {
    OutputOptions out;
    out.max_buffers = 4;
    out.sync = false;
    Graph g;
    g.add(nodes::Input());
    g.add(model.preprocess());
    g.add(model.inference());
    g.add(nodes::SimaBoxDecode(model, BoxDecodeType::YoloV8, 0.49, 0.5, 100));
    g.add(nodes::Output(out));
    Run run = g.build(inputs[0], on(RunBackend::Auto));
    require(run.push(inputs[1]), "declined: push on the fallback");
    Result<TensorList> pulled = run.pull(std::chrono::milliseconds(30000));
    require(static_cast<bool>(pulled) && pulled->size() == 1U &&
                pulled->front().semantic.detection.has_value(),
            "declined: a graph Dataflow declines runs on GStreamer under Auto and gives its "
            "detections");
    run.close();
  }
  ::unsetenv("SIMA_NEAT_BACKEND");
}

void lifecycle(const Model& model) {
  const std::vector<TensorList> inputs = frames_of(1280, 720);
  const std::vector<std::vector<std::uint8_t>> expected = outputs(model, inputs, nullptr, true);
  for (int round = 0; round < 3; ++round) {
    Graph g = graph(model);
    Run run = g.build(inputs[0], on(RunBackend::Dataflow));
    for (int p = 0; p < kPatterns; ++p) {
      require(run.push(inputs[static_cast<std::size_t>(p)]), "lifecycle: push");
    }
    run.close_input();
    for (int p = 0; p < kPatterns; ++p) {
      require(pull_frame(run, false, "drain").bytes == expected[static_cast<std::size_t>(p)],
              "lifecycle: what was pushed before close_input is pulled, in order");
    }
    Sample none;
    PullError closed;
    require(run.pull(5000, none, &closed) == PullStatus::Closed,
            "lifecycle: closed after the drain: " + closed.message);
    run.stop();
  }
}

void firmware(const Model& model) {
  // The EV74 runs the stock firmware: the stage has no A65 implementation, so the route is
  // refused at the build.
  const std::vector<TensorList> inputs = frames_of(1280, 720);
  try {
    Graph g = graph(model);
    (void)g.build(inputs[0], on(RunBackend::Dataflow));
  } catch (const NeatError& error) {
    require_contains(error.what(), "sima-neat-native-cvu-fw",
                     std::string("firmware: ") + error.what());
    std::cout << "[firmware] " << error.what() << "\n";
    return;
  }
  throw std::runtime_error(
      "firmware: the route built on the Dataflow backend on the stock firmware");
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string archive;
  sima_test::get_arg(argc, argv, "--model-package", archive);
  for (const char* name : {"SIMA_TEST_PREPROC_MODEL_PACKAGE", "SIMA_YOLO_TAR"}) {
    if (const char* staged = std::getenv(name); archive.empty() && staged) {
      archive = staged;
    }
  }
  if (archive.empty() || !file_exists(archive)) {
    return skip_test("unit_dataflow_preproc_route_test needs a model package with an image "
                     "preproc stage: --model-package <X_mpk.tar.gz>, "
                     "SIMA_TEST_PREPROC_MODEL_PACKAGE or SIMA_YOLO_TAR");
  }
  std::string only;
  sima_test::get_arg(argc, argv, "--row", only);
  return sima_test::run_test("unit_dataflow_preproc_route_test", [&] {
    const Model model(archive, model_options());
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    const bool ev74 = ev74_runs_kernels_firmware();
    std::cout << "[ev74] the EV74 " << (ev74 ? "runs" : "does not run")
              << " the Kernels firmware\n";
    if (!ev74) {
      row("firmware", [&] { firmware(model); });
      return;
    }
    row("differential", [&] { differential(model, 1280, 720, 64, "differential"); });
    row("sizes", [&] {
      differential(model, 640, 480, 8, "sizes 640x480");
      differential(model, 642, 362, 8, "sizes 642x362");
      differential(model, 1920, 1080, 8, "sizes 1920x1080");
    });
    row("declared", [&] { declared(model); });
    row("input_contract", [&] { input_contract(model); });
    row("refused", [&] { refused(model); });
    row("declined", [&] { declined(model, archive); });
    row("lifecycle", [&] { lifecycle(model); });
  });
}

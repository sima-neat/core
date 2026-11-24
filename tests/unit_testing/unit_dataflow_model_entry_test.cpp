// A Model's own entry points on the Dataflow backend, on the public API only: Model::build (with a
// seed and without), the Runner's verbs, Model::run and Model::execution_plan of a YOLOv8 detection
// model (the image preproc on the EV74, the MLA, BoxDecode), unchanged from what a program calls on
// GStreamer. A process runs on one backend (set_backend): the test runs twice, --backend=gstreamer
// --reference=<file> recording what the comparison rows see, then --backend=dataflow
// --reference=<file> comparing. It needs the EV74 on the Kernels firmware and a compiled YOLOv8
// detection package: --model-package <X_mpk.tar.gz>, or SIMA_TEST_BOXDECODE_MODEL_PACKAGE,
// SIMA_YOLO_TAR. Without one the test is skipped (it downloads nothing). Frames: generated ones,
// and in front of them the real images given with --frames-dir <dir> (or
// SIMA_TEST_BOXDECODE_FRAMES): raw 1280x720 BGR files named img<N>.bgr.
// Rows (each compares every pulled byte, and the Sample's fields both backends give):
//   runner      Model::build(seed) + Runner::push / Runner::pull, then close_input and the empty
//               Sample at the end; the first frame after the build is input_seq 1 (the build's
//               preflight took the seed), its Stamp's frame_id the pushed one.
//   run         Runner::run and Model::run (whose Runner is built at its first call).
//   seedless    Model::build() (the model's own dummy seed), then Runner::run.
//   plan        Model::execution_plan() and execution_plan(seed): the same JSON.
//   upstream    frames that carry an upstream stage's Semantic::preprocess without an ROI: not
//               read, the bytes and the pulled facts are the bare frame's; a seed that carries a
//               letterbox one builds and changes nothing; a seed that carries a stretch one
//               fails the build with the contract compiler's error.
//   format      a frame declared RGB into a Run of BGR frames: refused at the push with
//               GStreamer's text, and the Run goes on.
//   size        a frame of half the size: delivered, as on GStreamer.
//   mistakes    an empty seed or input list: the same error.
// On Dataflow only:
//   not_carried the package's default route, which the backend does not carry, fails at the
//               customer's call, Model::build or Model::run, in the one form.

#include "model/Model.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "cli_utils.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kTimeoutMs = 30000;

cv::Mat generated_image(int width, int height, unsigned index) {
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

// The real images given, then generated ones; each a cv::Mat kept beside its tensor.
std::vector<cv::Mat> images_of(const std::string& dir) {
  std::vector<cv::Mat> out;
  for (int n = 0; !dir.empty() && n < 8; ++n) {
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
  for (unsigned p = 0; p < 2U; ++p) {
    out.push_back(generated_image(kWidth, kHeight, p));
  }
  return out;
}

Tensor bgr(const cv::Mat& mat, ImageSpec::PixelFormat format = ImageSpec::PixelFormat::BGR) {
  return Tensor::from_cv_mat(mat, format, TensorMemory::EV74);
}

// Each pixel the mean of the four it covers.
cv::Mat half_of(const cv::Mat& m) {
  cv::Mat out(m.rows / 2, m.cols / 2, CV_8UC3);
  for (int r = 0; r < out.rows; ++r) {
    const std::uint8_t* a = m.ptr<std::uint8_t>(2 * r);
    const std::uint8_t* b = m.ptr<std::uint8_t>(2 * r + 1);
    std::uint8_t* o = out.ptr<std::uint8_t>(r);
    for (int c = 0; c < out.cols; ++c) {
      for (int k = 0; k < 3; ++k) {
        o[3 * c + k] = static_cast<std::uint8_t>(
            (a[6 * c + k] + a[6 * c + 3 + k] + b[6 * c + k] + b[6 * c + 3 + k] + 2) / 4);
      }
    }
  }
  return out;
}

// An upstream stage's transform without an ROI: a 3840x2160 original brought to this frame.
PreprocessRuntimeMeta upstream(const std::string& resize_mode) {
  PreprocessRuntimeMeta m;
  m.original_width = 3840;
  m.original_height = 2160;
  m.resized_width = kWidth;
  m.resized_height = kHeight;
  m.scaled_width = kWidth;
  m.scaled_height = kHeight;
  m.resize_mode = resize_mode;
  m.color_in = "BGR";
  m.color_out = "BGR";
  m.affine_m00 = 3.0;
  m.affine_m02 = 100.0;
  m.affine_m11 = 3.0;
  m.affine_m12 = 50.0;
  m.affine_scale_x = 3.0;
  m.affine_scale_y = 3.0;
  m.affine_offset_x = 100.0;
  m.affine_offset_y = 50.0;
  return m;
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

std::string real(double value) {
  char text[40];
  std::snprintf(text, sizeof text, "%.17g", value);
  return text;
}

// What a pulled tensor tells besides where its bytes live: its kind, shape and the frame's facts.
std::string tensor_text(const Tensor& t) {
  std::string out = "dtype=" + std::to_string(static_cast<int>(t.dtype)) +
                    " shape=" + sima_test::text(t.shape) +
                    " detection=" + (t.semantic.detection ? t.semantic.detection->format : "-");
  if (t.semantic.preprocess) {
    const PreprocessRuntimeMeta& p = *t.semantic.preprocess;
    out += " preprocess=" + std::to_string(p.original_width) + "x" +
           std::to_string(p.original_height) + "/" + std::to_string(p.scaled_width) + "x" +
           std::to_string(p.scaled_height) + " pad=" + std::to_string(p.pad_left) + "," +
           std::to_string(p.pad_right) + "," + std::to_string(p.pad_top) + "," +
           std::to_string(p.pad_bottom) + " " + p.resize_mode + " affine=" + real(p.affine_m00) +
           "," + real(p.affine_m02) + "," + real(p.affine_m11) + "," + real(p.affine_m12) +
           " rois=" + std::to_string(p.rois.size());
  }
  return out;
}

std::string bytes_of(const TensorList& tensors) {
  std::string out;
  for (const Tensor& t : tensors) {
    out += (out.empty() ? "" : " ") + sima_test::bytes_digest(t.copy_dense_bytes_tight());
  }
  return out;
}

// A pulled Sample: its bytes, its tensors' facts and the fields both backends give. Every push the
// rows compare carries a Stamp's frame_id (GStreamer gives an unstamped push's Sample a frame_id of
// its own, Dataflow none).
void record_sample(sima_test::Reference& ref, const std::string& at, const Sample& s) {
  require(!s.tensors.empty(), at + ": the Sample holds tensors");
  ref.equal(at + " bytes", bytes_of(s.tensors));
  for (std::size_t i = 0; i < s.tensors.size(); ++i) {
    ref.equal(at + " tensor " + std::to_string(i), tensor_text(s.tensors[i]));
  }
  ref.equal(at + " kind", static_cast<int>(s.kind));
  ref.equal(at + " media_type", s.media_type);
  ref.equal(at + " payload_tag", s.payload_tag);
  ref.equal(at + " format", s.format);
  ref.equal(at + " stream_id", s.stream_id);
  ref.equal(at + " frame_id", s.frame_id);
  ref.equal(at + " input_seq", s.input_seq);
  ref.equal(at + " orig_input_seq", s.orig_input_seq);
  ref.note(at + " stream_label", s.stream_label);
  ref.note(at + " caps_string", s.caps_string);
}

TensorList stamped(const Tensor& t, std::int64_t frame_id) {
  Stamp stamp;
  stamp.frame_id = frame_id;
  return TensorList{t.with_stamp(stamp)};
}

void runner_row(sima_test::Reference& ref, Model& model, const std::vector<cv::Mat>& images) {
  Model::Runner runner = model.build(TensorList{bgr(images[0])});
  for (std::size_t i = 0; i < images.size(); ++i) {
    require(runner.push(stamped(bgr(images[i]), 100 + static_cast<std::int64_t>(i))),
            "runner: push");
    const Sample s = runner.pull(kTimeoutMs);
    record_sample(ref, "runner frame " + std::to_string(i), s);
  }
  runner.close_input();
  const Sample end = runner.pull(5000);
  ref.equal("runner: end of stream gives an empty Sample", end.empty());
  runner.close();
}

void run_row(sima_test::Reference& ref, Model& model, const std::vector<cv::Mat>& images) {
  Model::Runner runner = model.build(TensorList{bgr(images[0])});
  for (std::size_t i = 0; i < images.size(); ++i) {
    ref.equal("Runner::run frame " + std::to_string(i),
              bytes_of(runner.run(TensorList{bgr(images[i])}, kTimeoutMs)));
  }
  runner.close();
  for (std::size_t i = 0; i < images.size(); ++i) {
    ref.equal("Model::run frame " + std::to_string(i),
              bytes_of(model.run(TensorList{bgr(images[i])}, kTimeoutMs)));
  }
}

void seedless_row(sima_test::Reference& ref, Model& model, const std::vector<cv::Mat>& images) {
  Model::Runner runner = model.build();
  for (std::size_t i = 0; i < 2U; ++i) {
    ref.equal("seedless frame " + std::to_string(i),
              bytes_of(runner.run(TensorList{bgr(images[i])}, kTimeoutMs)));
  }
  runner.close();
}

void plan_row(sima_test::Reference& ref, Model& model, const std::vector<cv::Mat>& images) {
  ref.equal("execution_plan()", model.execution_plan().to_json(2));
  ref.equal("execution_plan(seed)", model.execution_plan(TensorList{bgr(images[0])}).to_json(2));
}

void upstream_row(sima_test::Reference& ref, Model& model, const std::vector<cv::Mat>& images) {
  Model::Runner runner = model.build(TensorList{bgr(images[0])});
  for (std::size_t i = 0; i < 2U; ++i) {
    Tensor t = bgr(images[i]);
    t.semantic.preprocess = upstream("stretch");
    require(runner.push(stamped(t, 200 + static_cast<std::int64_t>(i))), "upstream: push");
    const Sample with = runner.pull(kTimeoutMs);
    require(runner.push(stamped(bgr(images[i]), 300 + static_cast<std::int64_t>(i))),
            "upstream: bare push");
    const Sample bare = runner.pull(kTimeoutMs);
    require(bytes_of(with.tensors) == bytes_of(bare.tensors) &&
                tensor_text(with.tensors.front()) == tensor_text(bare.tensors.front()),
            "upstream: the transform is not read (frame " + std::to_string(i) + ")");
    record_sample(ref, "upstream frame " + std::to_string(i), with);
  }
  runner.close();
  {
    Tensor seed = bgr(images[0]);
    seed.semantic.preprocess = upstream("letterbox");
    Model::Runner seeded = model.build(TensorList{seed});
    ref.equal("upstream: a letterbox seed, then a bare frame",
              bytes_of(seeded.run(TensorList{bgr(images[0])}, kTimeoutMs)));
    seeded.close();
  }
  Tensor seed = bgr(images[0]);
  seed.semantic.preprocess = upstream("stretch");
  std::string error = "built";
  try {
    Model::Runner refused = model.build(TensorList{seed});
    refused.close();
  } catch (const std::exception& e) {
    error = e.what();
  }
  ref.equal("upstream: a stretch seed", error);
}

void format_row(sima_test::Reference& ref, Model& model, const std::vector<cv::Mat>& images) {
  Model::Runner runner = model.build(TensorList{bgr(images[0])});
  std::string error = "taken";
  try {
    (void)runner.push(TensorList{bgr(images[1], ImageSpec::PixelFormat::RGB)});
  } catch (const std::exception& e) {
    error = e.what();
  }
  ref.equal("format: a frame declared RGB", error);
  require(runner.push(stamped(bgr(images[1]), 7)), "format: the Run goes on");
  record_sample(ref, "format: the next frame", runner.pull(kTimeoutMs));
  runner.close();
}

void size_row(sima_test::Reference& ref, Model& model, const std::vector<cv::Mat>& images) {
  Model::Runner runner = model.build(TensorList{bgr(images[0])});
  for (std::size_t i = 0; i < 2U; ++i) {
    require(runner.push(stamped(bgr(half_of(images[i])), 400 + static_cast<std::int64_t>(i))),
            "size: push");
    record_sample(ref, "size: a half-size frame " + std::to_string(i), runner.pull(kTimeoutMs));
    require(runner.push(stamped(bgr(images[i]), 500 + static_cast<std::int64_t>(i))), "size: push");
    record_sample(ref, "size: a whole frame " + std::to_string(i), runner.pull(kTimeoutMs));
  }
  runner.close();
}

void mistakes_row(sima_test::Reference& ref, Model& model) {
  const auto error_of = [](const std::function<void()>& call) {
    try {
      call();
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
    return std::string("no error");
  };
  ref.equal("mistakes: Model::build(empty)", error_of([&] { (void)model.build(TensorList{}); }));
  ref.equal("mistakes: Model::run(empty)",
            error_of([&] { (void)model.run(std::vector<Tensor>{}, kTimeoutMs); }));
}

// A route Dataflow does not carry (the package's own default route: a tensor input, its MLA's
// several heads behind it), from the customer's own calls: the error names the call, in the one
// form.
void not_carried_row(const std::string& archive) {
  Model model(archive);
  const std::vector<TensorSpec> specs = model.input_specs();
  require(!specs.empty() && !specs.front().shape.empty(), "not_carried: the model's input");
  std::vector<std::int64_t> shape = specs.front().shape;
  std::size_t count = 1;
  for (std::int64_t& d : shape) {
    d = d > 0 ? d : 1;
    count *= static_cast<std::size_t>(d);
  }
  const Tensor input =
      Tensor::from_vector(std::vector<float>(count, 0.0F), shape, TensorMemory::EV74);
  for (const char* call : {"Model::build", "Model::run"}) {
    try {
      if (std::string(call) == "Model::build") {
        (void)model.build(TensorList{input});
      } else {
        (void)model.run(TensorList{input}, kTimeoutMs);
      }
    } catch (const NeatError& e) {
      const std::string text = e.what();
      require_contains(text, std::string(call) + ": ", "names the call: " + text);
      require_contains(text, "is not carried by the Dataflow backend", "the form: " + text);
      require_contains(text, "simaai::neat::set_backend", "the hint: " + text);
      std::cout << "[not_carried] " << text << "\n";
      continue;
    }
    throw std::runtime_error(std::string("not_carried: ") + call + " built on Dataflow");
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
    return skip_test("unit_dataflow_model_entry_test needs a YOLOv8 detection model package: "
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
  return sima_test::run_test("unit_dataflow_model_entry_test", [&] {
    if (!ev74_runs_kernels_firmware()) {
      std::cout << "[skip] the EV74 does not run the Kernels firmware\n";
      return;
    }
    sima_test::Reference ref = sima_test::Reference::from_args(argc, argv);
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    Model model(archive, image_options());
    const std::vector<cv::Mat> images = images_of(dir);
    row("runner", [&] { runner_row(ref, model, images); });
    row("run", [&] { run_row(ref, model, images); });
    row("seedless", [&] { seedless_row(ref, model, images); });
    row("plan", [&] { plan_row(ref, model, images); });
    row("upstream", [&] { upstream_row(ref, model, images); });
    row("format", [&] { format_row(ref, model, images); });
    row("size", [&] { size_row(ref, model, images); });
    row("mistakes", [&] { mistakes_row(ref, model); });
    if (sima_test::on_dataflow()) {
      row("not_carried", [&] { not_carried_row(archive); });
    }
    if (only.empty()) {
      ref.finish();
    }
  });
}

#include "graph_migration/common/phase3_graph_test_utils.h"

#include "asset_utils.h"
#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include "yolov8_test_utils.h"
#endif

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgcodecs.hpp>
#endif

namespace fs = std::filesystem;

namespace graph_phase3_test {
namespace {

bool usable_file(const fs::path& path) {
  return sima_test::is_usable_regular_file(path);
}

std::string env_string(const char* name) {
  const char* raw = std::getenv(name);
  return (raw && *raw) ? std::string(raw) : std::string();
}

fs::path first_existing(std::initializer_list<fs::path> paths) {
  for (const auto& path : paths) {
    if (usable_file(path)) {
      return path;
    }
  }
  return {};
}

std::string with_tar_suffix(std::string_view filename) {
  std::string out(filename);
  constexpr std::string_view suffix = ".tar.gz";
  if (out.size() < suffix.size() || out.substr(out.size() - suffix.size()) != suffix) {
    out += suffix;
  }
  return out;
}

} // namespace

fs::path repo_root() {
  return sima_test::test_source_root();
}

fs::path resolve_yolov8n_variant_or_throw(const fs::path& root, std::string_view filename) {
  const std::string name = with_tar_suffix(filename);

  const fs::path env_path = env_string("SIMA_YOLO_BF16_BGR_MODEL_TAR");
  if (!env_path.empty() && env_path.filename() == name && usable_file(env_path)) {
    return env_path;
  }

  const fs::path found = first_existing({root / "tmp" / "yolov8n_6_variants" / name,
                                         root / "tmp" / "yolov8n_drive" / name,
                                         fs::current_path() / "tmp" / "yolov8n_6_variants" / name,
                                         fs::current_path() / "tmp" / "yolov8n_drive" / name,
                                         fs::path("/tmp") / "sima-yolov8n-drive" / name});
  if (!found.empty()) {
    return found;
  }

#if defined(SIMA_WITH_OPENCV)
  // Match the YOLO matrix tests: these variants are mandatory in CI, so a
  // missing staged tarball should be fetched on demand and validated instead
  // of failing this focused phase3 coverage first.
  const fs::path variants_dir = sima_yolov8_test::ensure_yolov8n_drive_variants(root);
  const fs::path downloaded = variants_dir / name;
  if (sima_test::is_listable_tar_gz(downloaded)) {
    return downloaded;
  }
#endif

  throw std::runtime_error("failed to resolve YOLOv8n variant tarball: " + name +
                           " (stage it under tmp/yolov8n_6_variants or tmp/yolov8n_drive, "
                           "or set SIMA_YOLOV8N_VARIANTS_BASE_URL to a mirror)");
}

std::vector<fs::path> resolve_yolov8n_variants_or_throw(const fs::path& root) {
  const std::vector<std::string> names = {
      "yolov8n_A_BF16_W_INT8_MLATess.tar.gz", "yolov8n_A_BF16_W_INT8_mpk.tar.gz",
      "yolov8n_A_W_BF16_MLATess.tar.gz",      "yolov8n_A_W_BF16_mpk.tar.gz",
      "yolov8n_A_W_INT8_MLATess.tar.gz",      "yolov8n_A_W_int8_mpk.tar.gz",
  };
  std::vector<fs::path> out;
  out.reserve(names.size());
  for (const auto& name : names) {
    out.push_back(resolve_yolov8n_variant_or_throw(root, name));
  }
  return out;
}

fs::path resolve_resnet50_or_throw(const fs::path& root) {
  const std::string resolved = sima_test::resolve_resnet50_tar(root);
  if (!resolved.empty() && usable_file(resolved)) {
    return resolved;
  }
  throw std::runtime_error("failed to resolve ResNet50 model tarball; set SIMA_RESNET50_TAR or "
                           "stage/download resnet_50_mpk.tar.gz");
}

fs::path resolve_mnist_or_throw(const fs::path& root) {
  if (const std::string env = env_string("SIMA_MNIST_MODEL_TAR");
      !env.empty() && usable_file(env)) {
    return env;
  }
  const std::string resolved = sima_test::resolve_modelzoo_tar("mnist_cnn", root);
  if (!resolved.empty() && usable_file(resolved)) {
    return resolved;
  }
  throw std::runtime_error("failed to resolve MNIST model tarball; set SIMA_MNIST_MODEL_TAR or "
                           "stage/download mnist_cnn");
}

#if defined(SIMA_WITH_OPENCV)
cv::Mat load_people_bgr_or_throw(const fs::path& root) {
  const fs::path path = first_existing(
      {root / "tests" / "images" / "people.jpg", root / "build" / "tests" / "images" / "people.jpg",
       root / "tmp" / "yolov8s_people.jpg", root / "tmp" / "coco_sample.jpg",
       fs::current_path() / "tests" / "images" / "people.jpg"});
  require(!path.empty(), "failed to resolve people.jpg test image");
  cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
  require(!img.empty(), "failed to read people image: " + path.string());
  return img;
}

cv::Mat load_goldfish_bgr_or_throw(const fs::path& root) {
  const fs::path path =
      first_existing({sima_test::default_goldfish_path(), root / "tmp" / "imagenet_goldfish.jpg",
                      root / "tests" / "images" / "goldfish.jpg",
                      fs::current_path() / "tmp" / "imagenet_goldfish.jpg"});
  require(!path.empty(), "failed to resolve goldfish image; run resnet50 tests with --goldfish "
                         "once or stage tmp/imagenet_goldfish.jpg");
  cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
  require(!img.empty(), "failed to read goldfish image: " + path.string());
  return img;
}
#endif

simaai::neat::Tensor make_rgb_tensor(int width, int height, std::uint8_t fill) {
  return make_color_tensor(width, height, simaai::neat::ImageSpec::PixelFormat::RGB, fill);
}

simaai::neat::Sample make_tensor_sample(int frame_id, std::string stream_id, int width,
                                        int height) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_rgb_tensor(width, height, static_cast<std::uint8_t>(0x30 + frame_id));
  sample.frame_id = frame_id;
  sample.stream_id = std::move(stream_id);
  return sample;
}

simaai::neat::Model::Options yolo_image_bgr_to_rgb_bf16_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.resize.width = 640;
  opt.preprocess.resize.height = 640;
  opt.preprocess.resize.mode = simaai::neat::ResizeMode::Letterbox;
  opt.preprocess.resize.pad_value = 114;
  opt.preprocess.resize.scaling_type = "BILINEAR";
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  opt.preprocess.color_convert.output_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.layout_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.layout_convert.perm = {0, 1, 2};
  opt.upstream_name = "decoder";
  return opt;
}

simaai::neat::Model::Options resnet_imagenet_bgr_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  opt.upstream_name = "decoder";
  return opt;
}

simaai::neat::Model::Options mnist_gray8_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.input_max_depth = 1;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::GRAY8;
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.1307f, 0.1307f, 0.1307f};
  opt.preprocess.normalize.stddev = {0.3081f, 0.3081f, 0.3081f};
  opt.preprocess.normalize.has_explicit_stats = true;
  return opt;
}

void require_nonempty_tensor_output(const simaai::neat::TensorList& out, std::string_view where) {
  require(!out.empty(), std::string(where) + ": expected non-empty TensorList output");
}

void require_sample_tensor_output(const simaai::neat::Sample& sample, std::string_view where) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    require(sample.tensor.has_value() && sample.tensor->storage != nullptr,
            std::string(where) + ": tensor sample has no storage");
    return;
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    require(!sample.tensors.empty(), std::string(where) + ": TensorSet sample is empty");
    return;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    require(!sample.fields.empty(), std::string(where) + ": Bundle sample is empty");
    return;
  }
  throw std::runtime_error(std::string(where) + ": expected tensor-like output sample");
}

} // namespace graph_phase3_test

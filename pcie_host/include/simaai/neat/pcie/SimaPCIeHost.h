#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<opencv2/core/mat.hpp>)
#include <opencv2/core/mat.hpp>
#define SIMA_PCIE_HAS_OPENCV_OVERLOAD 1
#endif

namespace simaai::neat::pcie {

enum class PipelineState {
  Uninitialized,
  Starting,
  Ready,
  Failed,
  Stopping,
  Exited,
};

struct ConnectionOptions {
  std::string card_host;
  int card_id = 0;
  std::string user = "sima";
  int queue = 0;
  int max_inflight = 0;
  std::string card_env;
  std::string card_gst_debug;
  std::string card_gst_debug_file;
  bool card_gst_debug_no_color = true;
};

enum class InputKind {
  Tensor,
  Image,
};

enum class AutoFlag {
  Auto,
  On,
  Off,
};

enum class ResizeMode {
  Stretch,
  Letterbox,
  Crop,
};

enum class ColorFormat {
  Auto,
  RGB,
  BGR,
  GRAY8,
  NV12,
  I420,
};

enum class NormalizePreset {
  None,
  ImageNet,
  COCO_YOLO,
};

enum class BoxDecodeType {
  Unspecified,
  YoloV5,
  YoloV6,
  YoloV7,
  YoloV8,
  YoloV26,
  YoloX,
};

enum class BoxDecodeTypeOption {
  Auto,
  Ultralytics,
  EfficientNMS,
};

struct ModelOptions {
  struct Preprocess {
    struct Resize {
      AutoFlag enable = AutoFlag::Auto;
      int width = 0;
      int height = 0;
      ResizeMode mode = ResizeMode::Letterbox;
      int pad_value = 114;
      std::string scaling_type = "BILINEAR";
    };

    struct ColorConvert {
      AutoFlag enable = AutoFlag::Auto;
      ColorFormat input_format = ColorFormat::Auto;
      ColorFormat output_format = ColorFormat::Auto;
    };

    struct Normalize {
      AutoFlag enable = AutoFlag::Auto;
      NormalizePreset preset = NormalizePreset::None;
      bool has_explicit_stats = false;
      std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
      std::array<float, 3> stddev{1.0f, 1.0f, 1.0f};
    };

    InputKind kind = InputKind::Tensor;
    AutoFlag enable = AutoFlag::Auto;
    int input_max_width = 0;
    int input_max_height = 0;
    int input_max_depth = 0;
    Resize resize;
    ColorConvert color_convert;
    Normalize normalize;
  };

  Preprocess preprocess;
  BoxDecodeType decode_type = BoxDecodeType::Unspecified;
  BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto;
  float score_threshold = 0.0f;
  float nms_iou_threshold = 0.0f;
  int top_k = 0;
  int num_classes = 0;
};

enum class TensorDType {
  UInt8,
  Int8,
  UInt16,
  Int16,
  Int32,
  BFloat16,
  Float32,
  Float64,
};

enum class TensorLayout {
  Unknown,
  HW,
  HWC,
  NHWC,
};

struct ImageSpec {
  enum class PixelFormat {
    Unknown,
    RGB,
    BGR,
    GRAY8,
    NV12,
    I420,
  };

  PixelFormat format = PixelFormat::Unknown;
  std::string color_space;
};

using PixelFormat = ImageSpec::PixelFormat;

enum class PlaneRole {
  Unknown,
  Y,
  U,
  V,
  UV,
};

struct Plane {
  PlaneRole role = PlaneRole::Unknown;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides_bytes;
  std::int64_t byte_offset = 0;
};

struct TensorRoute {
  std::string name;
  int logical_index = -1;
  int backend_output_index = -1;
  int physical_index = -1;
  int route_slot = -1;
  int memory_index = 0;
  std::int64_t physical_byte_offset = 0;
};

struct Tensor {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides_bytes;
  std::shared_ptr<void> owner;
  void* data = nullptr;
  std::size_t size_bytes = 0;
  std::int64_t byte_offset = 0;
  std::optional<ImageSpec> image;
  PixelFormat image_format = PixelFormat::Unknown;
  std::vector<Plane> planes;
  TensorRoute route;
  bool read_only = false;
};

using TensorList = std::vector<Tensor>;

struct TensorInfo {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t size_bytes = 0;
};

struct ModelInfo {
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  bool has_preprocess = false;
  bool has_boxdecode = false;
};

struct Status {
  PipelineState state = PipelineState::Uninitialized;
  int queue = -1;
  std::string message;
  std::string error_code;
};

class SimaPCIeHost {
public:
  explicit SimaPCIeHost(ConnectionOptions connection = {});
  ~SimaPCIeHost() noexcept;

  SimaPCIeHost(const SimaPCIeHost&) = delete;
  SimaPCIeHost& operator=(const SimaPCIeHost&) = delete;

  ModelInfo load_metadata(const std::string& model_path, const ModelOptions& options = {});
  ModelInfo init_pipeline(const std::string& model_path, const ModelOptions& options = {},
                          int readiness_timeout_ms = 180000);
  void stop();

  Status status() const;

  bool push(const Tensor& tensor);
  bool push(const TensorList& tensors);
#if defined(SIMA_PCIE_HAS_OPENCV_OVERLOAD)
  bool push(const cv::Mat& image);
#endif
  std::optional<TensorList> pull(int timeout_ms = -1);
  TensorList run(const Tensor& tensor, int timeout_ms = -1);
  TensorList run(const TensorList& tensors, int timeout_ms = -1);
#if defined(SIMA_PCIE_HAS_OPENCV_OVERLOAD)
  TensorList run(const cv::Mat& image, int timeout_ms = -1);
#endif

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

#if defined(SIMA_PCIE_HAS_OPENCV_OVERLOAD)
inline Tensor tensor_from_bgr_mat(const cv::Mat& image) {
  if (image.empty()) {
    throw std::runtime_error("PCIe cv::Mat image is empty");
  }
  if (image.depth() != CV_8U) {
    throw std::runtime_error("PCIe cv::Mat image must use uint8 storage");
  }
  const int channels = image.channels();
  if (channels != 1 && channels != 3) {
    throw std::runtime_error("PCIe cv::Mat image must be GRAY8 or BGR");
  }

  auto owner = std::make_shared<cv::Mat>(image.isContinuous() ? image : image.clone());
  Tensor tensor;
  tensor.dtype = TensorDType::UInt8;
  tensor.layout = channels == 1 ? TensorLayout::HW : TensorLayout::HWC;
  tensor.shape = channels == 1 ? std::vector<std::int64_t>{owner->rows, owner->cols}
                               : std::vector<std::int64_t>{owner->rows, owner->cols, channels};
  tensor.strides_bytes =
      channels == 1
          ? std::vector<std::int64_t>{static_cast<std::int64_t>(owner->step[0]), 1}
          : std::vector<std::int64_t>{static_cast<std::int64_t>(owner->step[0]), channels, 1};
  tensor.owner = owner;
  tensor.data = owner->data;
  tensor.size_bytes = static_cast<std::size_t>(owner->rows) * owner->step[0];
  tensor.byte_offset = 0;
  tensor.image = ImageSpec{channels == 1 ? PixelFormat::GRAY8 : PixelFormat::BGR, ""};
  tensor.image_format = tensor.image->format;
  tensor.route.name = "input_image";
  tensor.route.logical_index = 0;
  tensor.route.physical_index = 0;
  tensor.route.route_slot = 0;
  return tensor;
}

inline bool SimaPCIeHost::push(const cv::Mat& image) {
  return push(tensor_from_bgr_mat(image));
}

inline TensorList SimaPCIeHost::run(const cv::Mat& image, const int timeout_ms) {
  return run(tensor_from_bgr_mat(image), timeout_ms);
}
#endif

} // namespace simaai::neat::pcie

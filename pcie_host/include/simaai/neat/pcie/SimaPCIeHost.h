#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

enum class PixelFormat {
  Unknown,
  RGB,
  BGR,
  GRAY8,
  NV12,
  I420,
};

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

  ModelInfo load_metadata(const std::string& model_path,
                          const ModelOptions& options = {},
                          bool accelerator = false);
  ModelInfo init_pipeline(const std::string& model_path,
                          const ModelOptions& options = {},
                          bool accelerator = false,
                          int readiness_timeout_ms = 20000);
  void stop();

  Status status() const;

  bool push(const Tensor& tensor);
  bool push(const TensorList& tensors);
  std::optional<TensorList> pull(int timeout_ms = -1);
  TensorList run(const Tensor& tensor, int timeout_ms = -1);
  TensorList run(const TensorList& tensors, int timeout_ms = -1);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::pcie

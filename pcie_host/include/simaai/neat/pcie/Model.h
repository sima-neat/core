#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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
  int max_inflight = 10;
  std::string card_env;
  std::string card_gst_debug;
  std::string card_gst_debug_file;
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

namespace detail {

template <typename T> struct TensorDTypeFor;
template <> struct TensorDTypeFor<std::uint8_t> {
  static constexpr TensorDType value = TensorDType::UInt8;
};
template <> struct TensorDTypeFor<std::int8_t> {
  static constexpr TensorDType value = TensorDType::Int8;
};
template <> struct TensorDTypeFor<std::uint16_t> {
  static constexpr TensorDType value = TensorDType::UInt16;
};
template <> struct TensorDTypeFor<std::int16_t> {
  static constexpr TensorDType value = TensorDType::Int16;
};
template <> struct TensorDTypeFor<std::int32_t> {
  static constexpr TensorDType value = TensorDType::Int32;
};
template <> struct TensorDTypeFor<float> {
  static constexpr TensorDType value = TensorDType::Float32;
};
template <> struct TensorDTypeFor<double> {
  static constexpr TensorDType value = TensorDType::Float64;
};

template <typename T, typename = void> struct HasTensorDType : std::false_type {};
template <typename T>
struct HasTensorDType<T, std::void_t<decltype(TensorDTypeFor<T>::value)>> : std::true_type {};

inline TensorLayout infer_tensor_layout(const std::vector<std::int64_t>& shape) {
  if (shape.size() == 2U) {
    return TensorLayout::HW;
  }
  if (shape.size() == 3U) {
    return TensorLayout::HWC;
  }
  if (shape.size() == 4U) {
    return TensorLayout::NHWC;
  }
  return TensorLayout::Unknown;
}

inline std::size_t checked_shape_elements(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    throw std::invalid_argument("pcie::Tensor shape must not be empty");
  }
  std::size_t count = 1;
  for (const auto dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument("pcie::Tensor shape dimensions must be positive");
    }
    const auto udim = static_cast<std::size_t>(dim);
    if (count > std::numeric_limits<std::size_t>::max() / udim) {
      throw std::invalid_argument("pcie::Tensor shape is too large");
    }
    count *= udim;
  }
  return count;
}

inline std::vector<std::int64_t> contiguous_tensor_strides(const std::vector<std::int64_t>& shape,
                                                           const std::size_t elem_size) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = static_cast<std::int64_t>(elem_size);
  for (std::size_t index = shape.size(); index > 0; --index) {
    const std::size_t dim = index - 1;
    strides[dim] = stride;
    if (shape[dim] > 0 && stride > std::numeric_limits<std::int64_t>::max() / shape[dim]) {
      throw std::invalid_argument("pcie::Tensor stride is too large");
    }
    stride *= shape[dim];
  }
  return strides;
}

inline std::size_t checked_dense_bytes(const std::vector<std::int64_t>& shape,
                                       const std::size_t elem_size) {
  const std::size_t count = checked_shape_elements(shape);
  if (count > std::numeric_limits<std::size_t>::max() / elem_size) {
    throw std::invalid_argument("pcie::Tensor payload is too large");
  }
  return count * elem_size;
}

inline std::size_t checked_required_span_bytes(const std::vector<std::int64_t>& shape,
                                               const std::vector<std::int64_t>& strides_bytes,
                                               const std::size_t elem_size) {
  if (strides_bytes.empty()) {
    return checked_dense_bytes(shape, elem_size);
  }
  if (strides_bytes.size() != shape.size()) {
    throw std::invalid_argument("pcie::Tensor strides must match shape rank");
  }

  std::size_t max_offset = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      throw std::invalid_argument("pcie::Tensor shape dimensions must be positive");
    }
    if (strides_bytes[i] < static_cast<std::int64_t>(elem_size)) {
      throw std::invalid_argument("pcie::Tensor strides are too small for dtype");
    }
    const auto dim_minus_one = static_cast<std::size_t>(shape[i] - 1);
    const auto stride = static_cast<std::size_t>(strides_bytes[i]);
    if (dim_minus_one != 0 && stride > std::numeric_limits<std::size_t>::max() / dim_minus_one) {
      throw std::invalid_argument("pcie::Tensor strided payload is too large");
    }
    const std::size_t term = dim_minus_one * stride;
    if (max_offset > std::numeric_limits<std::size_t>::max() - term) {
      throw std::invalid_argument("pcie::Tensor strided payload is too large");
    }
    max_offset += term;
  }
  if (max_offset > std::numeric_limits<std::size_t>::max() - elem_size) {
    throw std::invalid_argument("pcie::Tensor strided payload is too large");
  }
  return max_offset + elem_size;
}

} // namespace detail

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

  template <typename T>
  static Tensor from_vector(std::vector<T> data, std::vector<std::int64_t> shape,
                            std::string route_name = "",
                            PixelFormat image_format = PixelFormat::Unknown) {
    using Value = std::remove_cv_t<T>;
    static_assert(detail::HasTensorDType<Value>::value,
                  "pcie::Tensor::from_vector supports uint8_t, int8_t, uint16_t, int16_t, "
                  "int32_t, float, and double");
    const std::size_t expected = detail::checked_shape_elements(shape);
    if (data.size() != expected) {
      throw std::invalid_argument("pcie::Tensor::from_vector data size does not match shape");
    }

    auto owner = std::make_shared<std::vector<Value>>(std::move(data));
    Tensor tensor;
    tensor.dtype = detail::TensorDTypeFor<Value>::value;
    tensor.layout = detail::infer_tensor_layout(shape);
    tensor.shape = std::move(shape);
    tensor.strides_bytes = detail::contiguous_tensor_strides(tensor.shape, sizeof(Value));
    tensor.owner = owner;
    tensor.data = owner->empty() ? nullptr : owner->data();
    tensor.size_bytes = owner->size() * sizeof(Value);
    tensor.byte_offset = 0;
    tensor.image_format = image_format;
    if (image_format != PixelFormat::Unknown) {
      tensor.image = ImageSpec{image_format, ""};
    }
    tensor.route.name = std::move(route_name);
    tensor.read_only = true;
    return tensor;
  }

  template <typename T>
  static Tensor from_external(T* base, std::size_t backing_element_count,
                              std::shared_ptr<void> owner, std::vector<std::int64_t> shape,
                              std::string route_name = "", std::int64_t byte_offset = 0,
                              std::vector<std::int64_t> strides_bytes = {},
                              PixelFormat image_format = PixelFormat::Unknown) {
    using Value = std::remove_cv_t<T>;
    static_assert(detail::HasTensorDType<Value>::value,
                  "pcie::Tensor::from_external supports uint8_t, int8_t, uint16_t, int16_t, "
                  "int32_t, float, and double");
    if (!base && backing_element_count != 0U) {
      throw std::invalid_argument("pcie::Tensor::from_external requires a data pointer");
    }
    if (!owner) {
      throw std::invalid_argument("pcie::Tensor::from_external requires an owner");
    }
    if (byte_offset < 0) {
      throw std::invalid_argument("pcie::Tensor::from_external byte_offset must be non-negative");
    }
    if (strides_bytes.empty()) {
      strides_bytes = detail::contiguous_tensor_strides(shape, sizeof(Value));
    }
    const std::size_t required_span =
        detail::checked_required_span_bytes(shape, strides_bytes, sizeof(Value));
    if (backing_element_count > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
      throw std::invalid_argument("pcie::Tensor::from_external backing buffer is too large");
    }
    const std::size_t backing_bytes = backing_element_count * sizeof(Value);
    const auto offset = static_cast<std::size_t>(byte_offset);
    if (offset > backing_bytes || required_span > backing_bytes - offset) {
      throw std::invalid_argument("pcie::Tensor::from_external tensor view exceeds backing buffer");
    }

    Tensor tensor;
    tensor.dtype = detail::TensorDTypeFor<Value>::value;
    tensor.layout = detail::infer_tensor_layout(shape);
    tensor.shape = std::move(shape);
    tensor.strides_bytes = std::move(strides_bytes);
    tensor.owner = std::move(owner);
    tensor.data = const_cast<Value*>(base);
    tensor.size_bytes = backing_bytes;
    tensor.byte_offset = byte_offset;
    tensor.image_format = image_format;
    if (image_format != PixelFormat::Unknown) {
      tensor.image = ImageSpec{image_format, ""};
    }
    tensor.route.name = std::move(route_name);
    tensor.read_only = true;
    return tensor;
  }
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

class Model {
public:
  explicit Model(std::string model_path, ModelOptions options = {},
                 ConnectionOptions connection = {});
  ~Model() noexcept;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  ModelInfo info() const;
  std::vector<TensorInfo> input_specs() const;
  std::vector<TensorInfo> output_specs() const;

  void build(int readiness_timeout_ms = 180000);
  bool running() const;
  void close();

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

inline bool Model::push(const cv::Mat& image) {
  return push(tensor_from_bgr_mat(image));
}

inline TensorList Model::run(const cv::Mat& image, const int timeout_ms) {
  return run(tensor_from_bgr_mat(image), timeout_ms);
}
#endif

} // namespace simaai::neat::pcie

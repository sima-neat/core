/**
 * @file
 * @ingroup tensors
 * @brief Tensor core types and storage/mapping utilities.
 */
#pragma once

#include "pipeline/TensorTypes.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace simaai::neat {

enum class DeviceType {
  CPU = 0,
  SIMA_APU,
  SIMA_CVU,
  SIMA_MLA,
  UNKNOWN,
};

struct Device {
  DeviceType type = DeviceType::CPU;
  int id = 0;
};

enum class StorageKind {
  CpuOwned = 0,
  CpuExternal,
  GstSample,
  DeviceHandle,
  Unknown,
};

enum class PlaneRole {
  Unknown = 0,
  Y,
  U,
  V,
  UV,
};

enum class MapMode {
  Read = 0,
  Write,
  ReadWrite,
};

struct ImageSpec {
  enum class PixelFormat {
    RGB = 0,
    BGR,
    GRAY8,
    NV12,
    I420,
    UNKNOWN,
  };

  PixelFormat format = PixelFormat::UNKNOWN;
  std::string color_space;
};

struct AudioSpec {
  int sample_rate = 0;
  int channels = 0;
  bool interleaved = true;
};

struct TokensSpec {
  int vocab_size = 0;
};

struct EncodedSpec {
  enum class Codec {
    H264 = 0,
    H265,
    RTP_H264,
    RTP_H265,
    JPEG,
    UNKNOWN,
  };

  Codec codec = Codec::UNKNOWN;
};

struct QuantSpec {
  float scale = 1.0f;
  int32_t zero_point = 0;
  int axis = -1;
  std::vector<float> scales;
  std::vector<int32_t> zero_points;
};

struct TessSpec {
  int tile_width = 0;
  int tile_height = 0;
  int tile_channels = 0;
  std::string format;
};

struct Semantic {
  std::optional<ImageSpec> image;
  std::optional<AudioSpec> audio;
  std::optional<TokensSpec> tokens;
  std::optional<TessSpec> tess;
  std::optional<EncodedSpec> encoded;
  std::optional<QuantSpec> quant;
};

struct Mapping {
  void* data = nullptr;
  std::size_t size_bytes = 0;
  std::function<void()> unmap;
  std::shared_ptr<void> keepalive;

  Mapping() = default;
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  Mapping(Mapping&& other) noexcept {
    *this = std::move(other);
  }
  Mapping& operator=(Mapping&& other) noexcept {
    if (this != &other) {
      if (unmap)
        unmap();
      data = other.data;
      size_bytes = other.size_bytes;
      unmap = std::move(other.unmap);
      keepalive = std::move(other.keepalive);
      other.data = nullptr;
      other.size_bytes = 0;
      other.unmap = nullptr;
      other.keepalive.reset();
    }
    return *this;
  }
  ~Mapping() {
    if (unmap)
      unmap();
  }
};

#if defined(SIMA_WITH_OPENCV)
struct CvMatView {
  Mapping mapping;
  cv::Mat mat;
};
#endif

struct Segment {
  std::string name;
  std::size_t size_bytes = 0;
};

struct Storage {
  StorageKind kind = StorageKind::Unknown;
  Device device{};
  std::size_t size_bytes = 0;
  std::shared_ptr<void> holder;
  void* data = nullptr;
  std::function<Mapping(MapMode)> map_fn;
  std::uint64_t sima_mem_target_flags = 0;
  std::uint64_t sima_mem_flags = 0;
  std::vector<Segment> sima_segments;

  Mapping map(MapMode mode) const {
    Mapping mapping;
    if (map_fn) {
      mapping = map_fn(mode);
    } else {
      mapping.data = data;
      mapping.size_bytes = size_bytes;
    }
    if (!mapping.keepalive) {
      mapping.keepalive = holder;
    }
    return mapping;
  }
};

std::shared_ptr<Storage> make_cpu_owned_storage(std::size_t size_bytes);
std::shared_ptr<Storage> make_cpu_external_storage(void* data, std::size_t size_bytes,
                                                   std::shared_ptr<void> holder = {},
                                                   bool read_only = true);

struct Plane {
  PlaneRole role = PlaneRole::Unknown;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;
  int64_t byte_offset = 0;
};

struct Nv12View {
  int width = 0;
  int height = 0;
  const uint8_t* y = nullptr;
  int64_t y_stride = 0;
  const uint8_t* uv = nullptr;
  int64_t uv_stride = 0;
};

struct Nv12Mapped {
  Mapping mapping;
  Nv12View view;
};

struct I420View {
  int width = 0;
  int height = 0;
  const uint8_t* y = nullptr;
  int64_t y_stride = 0;
  const uint8_t* u = nullptr;
  int64_t u_stride = 0;
  const uint8_t* v = nullptr;
  int64_t v_stride = 0;
};

struct I420Mapped {
  Mapping mapping;
  I420View view;
};

struct Tensor {
  std::shared_ptr<Storage> storage;
  simaai::neat::TensorDType dtype = simaai::neat::TensorDType::UInt8;
  simaai::neat::TensorLayout layout = simaai::neat::TensorLayout::Unknown;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;
  int64_t byte_offset = 0;
  Device device{};
  Semantic semantic{};
  std::vector<Plane> planes;
  bool read_only = true;

  bool is_dense() const {
    return planes.empty();
  }
  bool is_composite() const {
    return !planes.empty();
  }

  bool is_contiguous() const {
    if (shape.empty())
      return true;
    if (strides_bytes.empty())
      return true;
    std::size_t elem = dtype_bytes(dtype);
    if (elem == 0)
      return false;
    std::int64_t expected = static_cast<std::int64_t>(elem);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      if (strides_bytes[static_cast<size_t>(i)] != expected)
        return false;
      expected *= shape[static_cast<size_t>(i)];
    }
    return true;
  }

  const Plane* try_plane(PlaneRole role) const noexcept {
    for (const auto& plane : planes) {
      if (plane.role == role)
        return &plane;
    }
    return nullptr;
  }

  bool has_plane(PlaneRole role) const noexcept {
    return try_plane(role) != nullptr;
  }

  const Plane& plane(PlaneRole role) const {
    const Plane* found = try_plane(role);
    if (!found)
      throw std::runtime_error("Tensor::plane: plane not found");
    return *found;
  }

  Mapping map(MapMode mode) const {
    if (read_only && mode != MapMode::Read) {
      throw std::runtime_error("Tensor::map: tensor is read-only");
    }
    if (!storage)
      return {};
    Mapping base = storage->map(mode);
    if (!base.data)
      return base;
    Mapping out = std::move(base);
    if (!out.keepalive && storage) {
      out.keepalive = std::static_pointer_cast<void>(storage);
    }
    if (byte_offset != 0) {
      out.data = static_cast<uint8_t*>(out.data) + byte_offset;
      if (out.size_bytes > static_cast<std::size_t>(byte_offset)) {
        out.size_bytes = out.size_bytes - static_cast<std::size_t>(byte_offset);
      }
    }
#if defined(NEAT_VALIDATE_ON_MAP)
    std::string err;
    if (!validate(&err)) {
      throw std::runtime_error("Tensor::map: " + err);
    }
#endif
    return out;
  }

  Mapping map_read() const {
    return map(MapMode::Read);
  }
  Mapping map_write() const {
    return map(MapMode::Write);
  }
  Mapping view(MapMode mode = MapMode::Read) const;
  Mapping view_read() const {
    return view(MapMode::Read);
  }

  template <typename T> T* data_ptr() {
    if (read_only) {
      throw std::runtime_error("Tensor::data_ptr: tensor is read-only");
    }
    return const_cast<T*>(const_data_ptr<T>());
  }

  template <typename T> const T* data_ptr() const {
    return const_data_ptr<T>();
  }

  Tensor contiguous() const;
  Tensor clone() const;
  Tensor to(Device target) const;
  Tensor cpu() const;
  Tensor cvu() const;
  Tensor mla(bool force = false) const;
  Tensor to_cpu_if_needed() const;
  bool validate(std::string* err) const;

  std::optional<Nv12Mapped> map_nv12_read() const;
  std::size_t nv12_required_bytes() const;
  bool copy_nv12_contiguous_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_nv12_contiguous() const;

  std::optional<I420Mapped> map_i420_read() const;
  std::size_t i420_required_bytes() const;
  bool copy_i420_contiguous_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_i420_contiguous() const;

  std::size_t dense_bytes_tight() const;
  bool copy_dense_bytes_tight_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_dense_bytes_tight() const;

  bool copy_payload_bytes_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_payload_bytes() const;

  int width() const;
  int height() const;
  int channels() const;
  std::optional<ImageSpec::PixelFormat> image_format() const;
  bool is_nv12() const;
  bool is_i420() const;

  std::string debug_string() const;

#if defined(SIMA_WITH_OPENCV)
  static Tensor from_cv_mat(const cv::Mat& mat,
                            ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                            bool read_only = true);
  std::optional<CvMatView> map_cv_mat_view(ImageSpec::PixelFormat desired) const;
  cv::Mat to_cv_mat_copy(ImageSpec::PixelFormat desired) const;
#endif

private:
  template <typename T> const T* const_data_ptr() const {
    if (device.type != DeviceType::CPU) {
      throw std::runtime_error("Tensor::data_ptr: tensor is not on CPU");
    }
    if (!is_dense()) {
      throw std::runtime_error("Tensor::data_ptr: tensor is composite");
    }
    if (!is_contiguous()) {
      throw std::runtime_error("Tensor::data_ptr: call cpu().contiguous() first");
    }
    if (!storage || !storage->data) {
      throw std::runtime_error("Tensor::data_ptr: tensor storage is not mappable");
    }
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(storage->data) + byte_offset);
  }

  static std::size_t dtype_bytes(simaai::neat::TensorDType dtype) {
    switch (dtype) {
    case simaai::neat::TensorDType::UInt8:
    case simaai::neat::TensorDType::Int8:
      return 1;
    case simaai::neat::TensorDType::UInt16:
    case simaai::neat::TensorDType::Int16:
    case simaai::neat::TensorDType::BFloat16:
      return 2;
    case simaai::neat::TensorDType::Int32:
    case simaai::neat::TensorDType::Float32:
      return 4;
    case simaai::neat::TensorDType::Float64:
      return 8;
    }
    return 0;
  }
};

} // namespace simaai::neat

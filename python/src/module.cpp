#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/make_iterator.h>

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "genai/ASRModel.h"
#include "genai/GenAIModel.h"
#include "genai/GenAITypes.h"
#include "genai/GraphFragments.h"
#include "genai/GenAIServer.h"
#include "genai/VisionLanguageModel.h"
#include "graphs/Fragments.h"
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/groups/UdpH264OutputGroup.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/SimaDecode.h"
#include "nodes/sima/VisualFrontend.h"
#include "nodes/groups/GroupOutputSpec.h"
#include "nodes/groups/HttpMjpegDecodedInput.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/groups/VideoInputGroup.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/Input.h"
#include "nodes/io/MetadataSender.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/common/Caps.h"
#include "nodes/common/EncodedCapsFixup.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/JpegDecode.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/MultipartJpegDemux.h"
#include "nodes/common/VideoScale.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/io/HttpSource.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/sima/SimaRender.h"
#include "nodes/sima/SimaArgMax.h"
#include "nodes/sima/Cast.h"
#include "nodes/sima/CastTess.h"
#include "nodes/sima/Dequant.h"
#include "nodes/sima/Detess.h"
#include "nodes/sima/DetessCast.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/rtp/RTPJpegDepacketize.h"
#include "nodes/sima/PCIeSrc.h"
#include "nodes/sima/PCIeSink.h"
#include "pipeline/Run.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphMetrics.h"
#include "pipeline/NeatError.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/RunExport.h"
#include "pipeline/StageRun.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/EncodedSampleUtil.h"
#include "neat/version.h"
#include "neat/runtime.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using simaai::neat::ByteFormat;
using simaai::neat::ByteStreamSpec;
using simaai::neat::Device;
using simaai::neat::DeviceType;
using simaai::neat::Graph;
using simaai::neat::GraphElementMetrics;
using simaai::neat::GraphLinkOptions;
using simaai::neat::GraphLinkPolicy;
using simaai::neat::GraphNodeMetrics;
using simaai::neat::GraphOptions;
using simaai::neat::GraphReport;
using simaai::neat::GraphRunAutoExportOptions;
using simaai::neat::GraphRunExportOptions;
using simaai::neat::ImageSpec;
using simaai::neat::MapMode;
using simaai::neat::MeasureCounters;
using simaai::neat::MeasureInputStats;
using simaai::neat::MeasureLatencyStats;
using simaai::neat::MeasureOptions;
using simaai::neat::MeasurePathIdentity;
using simaai::neat::MeasurePathInterPluginGap;
using simaai::neat::MeasurePathNodeArrival;
using simaai::neat::MeasurePathOutputTail;
using simaai::neat::MeasurePathStat;
using simaai::neat::MeasurePathTiming;
using simaai::neat::MeasurePluginLatency;
using simaai::neat::MeasureReport;
using simaai::neat::MeasureScope;
using simaai::neat::NeatError;
using simaai::neat::NodeLatencySummary;
using simaai::neat::OutputMemory;
using simaai::neat::PowerFieldSummary;
using simaai::neat::PowerMonitorOptions;
using simaai::neat::PowerMonitorProfile;
using simaai::neat::PowerRailConfig;
using simaai::neat::PowerRailSummary;
using simaai::neat::PowerSummary;
using simaai::neat::PullError;
using simaai::neat::PullStatus;
using simaai::neat::Run;
using simaai::neat::RunAdvancedOptions;
using simaai::neat::RunAutoExportOptions;
using simaai::neat::RunExportOptions;
using simaai::neat::RunOptions;
using simaai::neat::RunPreset;
using simaai::neat::Sample;
using simaai::neat::SampleKind;
using simaai::neat::Tensor;
using simaai::neat::TensorConstraint;
using simaai::neat::TensorDType;
using simaai::neat::TensorLayout;
using simaai::neat::TensorList;
using simaai::neat::TensorMemory;
using simaai::neat::ValidateOptions;
using simaai::neat::dlpack::DLDataTypeCode;
using simaai::neat::dlpack::DLManagedTensor;

constexpr uint8_t kDlBfloat = 4;

std::size_t dtype_bytes(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
  case TensorDType::Int8:
    return 1;
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Int32:
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 0;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t* out) {
  if (!out)
    return false;
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  if (a > std::numeric_limits<std::size_t>::max() / b)
    return false;
  *out = a * b;
  return true;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t* out) {
  if (!out)
    return false;
  if (a > std::numeric_limits<std::size_t>::max() - b)
    return false;
  *out = a + b;
  return true;
}

nb::object json_to_python(const nlohmann::json& value) {
  if (value.is_null())
    return nb::none();
  if (value.is_boolean())
    return nb::bool_(value.get<bool>());
  if (value.is_number_integer())
    return nb::int_(value.get<std::int64_t>());
  if (value.is_number_unsigned())
    return nb::int_(value.get<std::uint64_t>());
  if (value.is_number_float())
    return nb::float_(value.get<double>());
  if (value.is_string())
    return nb::str(value.get_ref<const std::string&>().c_str());
  if (value.is_array()) {
    nb::list out;
    for (const auto& item : value) {
      out.append(json_to_python(item));
    }
    return std::move(out);
  }
  if (value.is_object()) {
    nb::dict out;
    for (auto it = value.begin(); it != value.end(); ++it) {
      out[nb::str(it.key().c_str())] = json_to_python(it.value());
    }
    return std::move(out);
  }
  throw std::runtime_error("unsupported nlohmann::json value in Python binding");
}

nlohmann::json python_to_json(nb::handle value) {
  if (value.is_none())
    return nullptr;
  if (PyBool_Check(value.ptr()))
    return nb::cast<bool>(value);
  if (PyLong_Check(value.ptr()))
    return nb::cast<std::int64_t>(value);
  if (PyFloat_Check(value.ptr()))
    return nb::cast<double>(value);
  if (PyUnicode_Check(value.ptr()))
    return nb::cast<std::string>(value);
  if (PyDict_Check(value.ptr())) {
    nlohmann::json out = nlohmann::json::object();
    for (auto item : nb::borrow<nb::dict>(value)) {
      out[nb::cast<std::string>(item.first)] = python_to_json(item.second);
    }
    return out;
  }
  if (PyList_Check(value.ptr()) || PyTuple_Check(value.ptr())) {
    nlohmann::json out = nlohmann::json::array();
    for (auto item : nb::borrow<nb::sequence>(value)) {
      out.push_back(python_to_json(item));
    }
    return out;
  }
  throw nb::type_error(
      "config_json must be None or a JSON-like Python value (dict, list, str, int, float, bool)");
}

std::optional<nlohmann::json> python_to_optional_json(nb::handle value) {
  if (value.is_none())
    return std::nullopt;
  return python_to_json(value);
}

simaai::neat::genai::Json python_to_genai_json(nb::handle value) {
  return simaai::neat::genai::Json::parse(python_to_json(value).dump());
}

nb::object genai_json_to_python(const simaai::neat::genai::Json& value) {
  return json_to_python(nlohmann::json::parse(value.dump()));
}

Sample make_text_sample_for_python(const std::string& port_name, const std::string& text) {
  Sample out = simaai::neat::make_tensor_sample(port_name, Tensor::from_text(text));
  out.port_name = port_name;
  out.stream_label = port_name;
  return out;
}

std::string sample_to_text_for_python(const Sample& sample) {
  if (sample.kind == SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("Sample.to_text: Tensor sample has no tensor payload");
    }
    return sample.tensor->to_text();
  }

  if (sample.kind == SampleKind::TensorSet) {
    if (sample.tensors.size() != 1U) {
      throw std::runtime_error("Sample.to_text: TensorSet sample must contain exactly one tensor");
    }
    return sample.tensors.front().to_text();
  }

  if (sample.kind == SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (field.port_name == "text" || field.stream_label == "text") {
        return sample_to_text_for_python(field);
      }
    }
    if (!sample.fields.empty()) {
      return sample_to_text_for_python(sample.fields.front());
    }
  }

  throw std::runtime_error("Sample.to_text: sample is not a text tensor");
}

// Format option fields are stored as C++ FormatSpec (a thin wrapper over FormatTag). In
// Python they are surfaced as the `Format` enum directly: reads return the underlying
// FormatTag, writes accept a FormatTag only. Passing a plain string raises TypeError —
// callers select a value from pyneat.Format (e.g. pyneat.Format.NV12).
template <typename C> auto format_enum_getter(simaai::neat::FormatSpec C::*member) {
  return [member](const C& self) { return (self.*member).tag; };
}
template <typename C> auto format_enum_setter(simaai::neat::FormatSpec C::*member) {
  return [member](C& self, simaai::neat::FormatTag value) { self.*member = value; };
}

std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                              std::size_t elem_bytes) {
  if (shape.empty())
    return {};
  std::vector<int64_t> strides(shape.size(), 0);
  std::size_t stride = elem_bytes;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = static_cast<int64_t>(stride);
    std::size_t next = 0;
    if (!checked_mul(stride, static_cast<std::size_t>(shape[static_cast<std::size_t>(i)]), &next)) {
      throw std::runtime_error("shape overflow while computing strides");
    }
    stride = next;
  }
  return strides;
}

std::size_t span_bytes(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides_bytes,
                       std::size_t elem_bytes) {
  if (shape.empty())
    return elem_bytes;
  if (shape.size() != strides_bytes.size())
    throw std::runtime_error("shape/strides rank mismatch");

  std::size_t max_offset = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const int64_t dim = shape[i];
    const int64_t stride = strides_bytes[i];
    if (dim <= 0)
      throw std::runtime_error("invalid non-positive dimension");
    if (stride < 0)
      throw std::runtime_error("negative strides are not supported");

    std::size_t term = 0;
    if (!checked_mul(static_cast<std::size_t>(dim - 1), static_cast<std::size_t>(stride), &term)) {
      throw std::runtime_error("stride overflow");
    }
    std::size_t next = 0;
    if (!checked_add(max_offset, term, &next))
      throw std::runtime_error("offset overflow");
    max_offset = next;
  }

  std::size_t out = 0;
  if (!checked_add(max_offset, elem_bytes, &out))
    throw std::runtime_error("span overflow");
  return out;
}

const char* pixel_format_name(ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "UNKNOWN";
}

void warn_chw_to_hwc_copy_once(ImageSpec::PixelFormat fmt) {
  static std::once_flag warned;
  std::call_once(warned, [fmt]() {
    const std::string msg =
        "pyneat: received CHW tensor for image/video input (format=" +
        std::string(pixel_format_name(fmt)) +
        "). Auto-converting to HWC so video/x-raw input contracts are satisfied. "
        "This helper requires a full memory copy (not zero-copy). "
        "For best performance, provide HWC uint8 input directly.";
    if (PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1) < 0) {
      throw nb::python_error();
    }
  });
}

void warn_deprecated_use_simaai_pool_python_once() {
  static std::once_flag warned;
  std::call_once(warned, []() {
    const char* msg = "pyneat.InputOptions.use_simaai_pool is deprecated. "
                      "Set InputOptions.memory_policy instead.";
    if (PyErr_WarnEx(PyExc_UserWarning, msg, 1) < 0) {
      throw nb::python_error();
    }
  });
}

void warn_deprecated_h264_decode_python() {
  const char* msg = "pyneat.nodes.h264_decode is deprecated; use pyneat.nodes.sima_decode with "
                    "SimaDecodeOptions.type = SimaDecodeType.H264";
  if (PyErr_WarnEx(PyExc_DeprecationWarning, msg, 1) < 0) {
    throw nb::python_error();
  }
}

std::size_t int64_to_size_or_throw(int64_t value, const char* what) {
  if (value < 0) {
    throw std::runtime_error(std::string("invalid negative ") + what);
  }
  return static_cast<std::size_t>(value);
}

Tensor maybe_convert_chw_image_to_hwc(Tensor in, bool* converted) {
  if (converted) {
    *converted = false;
  }
  if (in.layout != TensorLayout::CHW || !in.semantic.image.has_value()) {
    return in;
  }
  if (!in.is_dense()) {
    return in;
  }
  if (in.shape.size() != 3 || in.strides_bytes.size() != 3) {
    return in;
  }

  const ImageSpec::PixelFormat fmt = in.semantic.image->format;
  if (fmt != ImageSpec::PixelFormat::RGB && fmt != ImageSpec::PixelFormat::BGR &&
      fmt != ImageSpec::PixelFormat::GRAY8) {
    return in;
  }
  if (in.dtype != TensorDType::UInt8) {
    return in;
  }

  const int64_t c_i64 = in.shape[0];
  const int64_t h_i64 = in.shape[1];
  const int64_t w_i64 = in.shape[2];
  if (c_i64 <= 0 || h_i64 <= 0 || w_i64 <= 0) {
    return in;
  }

  const int64_t sc_i64 = in.strides_bytes[0];
  const int64_t sh_i64 = in.strides_bytes[1];
  const int64_t sw_i64 = in.strides_bytes[2];
  if (sc_i64 < 0 || sh_i64 < 0 || sw_i64 < 0) {
    throw std::runtime_error("negative strides are not supported for CHW->HWC conversion");
  }

  const std::size_t elem = dtype_bytes(in.dtype);
  if (elem == 0) {
    throw std::runtime_error("unknown dtype size during CHW->HWC conversion");
  }
  const std::size_t c = int64_to_size_or_throw(c_i64, "channel dimension");
  const std::size_t h = int64_to_size_or_throw(h_i64, "height dimension");
  const std::size_t w = int64_to_size_or_throw(w_i64, "width dimension");
  const std::size_t sc = int64_to_size_or_throw(sc_i64, "channel stride");
  const std::size_t sh = int64_to_size_or_throw(sh_i64, "row stride");
  const std::size_t sw = int64_to_size_or_throw(sw_i64, "column stride");

  std::size_t hw = 0;
  if (!checked_mul(h, w, &hw)) {
    throw std::runtime_error("shape overflow during CHW->HWC conversion");
  }
  std::size_t hwc = 0;
  if (!checked_mul(hw, c, &hwc)) {
    throw std::runtime_error("shape overflow during CHW->HWC conversion");
  }
  std::size_t total_bytes = 0;
  if (!checked_mul(hwc, elem, &total_bytes)) {
    throw std::runtime_error("byte-size overflow during CHW->HWC conversion");
  }

  simaai::neat::Mapping src = in.map_read();
  if (!src.data) {
    throw std::runtime_error("failed to map source tensor during CHW->HWC conversion");
  }
  if (src.size_bytes > 0) {
    std::size_t src_max = 0;
    std::size_t term = 0;
    if (!checked_mul(c - 1, sc, &term) || !checked_add(src_max, term, &src_max) ||
        !checked_mul(h - 1, sh, &term) || !checked_add(src_max, term, &src_max) ||
        !checked_mul(w - 1, sw, &term) || !checked_add(src_max, term, &src_max) ||
        !checked_add(src_max, elem, &src_max)) {
      throw std::runtime_error("source offset overflow during CHW->HWC conversion");
    }
    if (src_max > src.size_bytes) {
      throw std::runtime_error("source tensor mapping is too small for CHW->HWC conversion");
    }
  }

  Tensor out;
  out.dtype = in.dtype;
  out.layout = TensorLayout::HWC;
  out.shape = {h_i64, w_i64, c_i64};
  out.strides_bytes = contiguous_strides_bytes(out.shape, elem);
  out.byte_offset = 0;
  out.device = in.device;
  out.semantic = in.semantic;
  out.planes.clear();
  out.read_only = false;
  out.storage = simaai::neat::make_cpu_owned_storage(total_bytes);

  simaai::neat::Mapping dst = out.storage->map(MapMode::Write);
  if (!dst.data || dst.size_bytes < total_bytes) {
    throw std::runtime_error("failed to map destination tensor during CHW->HWC conversion");
  }

  const auto* src_bytes = static_cast<const uint8_t*>(src.data);
  auto* dst_bytes = static_cast<uint8_t*>(dst.data);
  std::size_t dst_row_stride = 0;
  std::size_t dst_pixel_stride = 0;
  if (!checked_mul(w, c, &dst_row_stride) || !checked_mul(dst_row_stride, elem, &dst_row_stride) ||
      !checked_mul(c, elem, &dst_pixel_stride)) {
    throw std::runtime_error("destination stride overflow during CHW->HWC conversion");
  }

  // Re-pack channel-first (CHW) source into packed channel-last (HWC) bytes.
  for (std::size_t ci = 0; ci < c; ++ci) {
    std::size_t src_c_off = 0;
    if (!checked_mul(ci, sc, &src_c_off)) {
      throw std::runtime_error("source offset overflow during CHW->HWC conversion");
    }
    uint8_t* dst_c = dst_bytes + ci * elem;
    for (std::size_t hi = 0; hi < h; ++hi) {
      std::size_t src_row_off = 0;
      std::size_t term = 0;
      if (!checked_mul(hi, sh, &term) || !checked_add(src_c_off, term, &src_row_off)) {
        throw std::runtime_error("source row offset overflow during CHW->HWC conversion");
      }
      std::size_t dst_row_off = 0;
      if (!checked_mul(hi, dst_row_stride, &dst_row_off)) {
        throw std::runtime_error("destination row offset overflow during CHW->HWC conversion");
      }
      uint8_t* dst_row = dst_c + dst_row_off;
      for (std::size_t wi = 0; wi < w; ++wi) {
        std::size_t src_off = 0;
        if (!checked_mul(wi, sw, &term) || !checked_add(src_row_off, term, &src_off)) {
          throw std::runtime_error("source column offset overflow during CHW->HWC conversion");
        }
        std::size_t dst_off = 0;
        if (!checked_mul(wi, dst_pixel_stride, &dst_off)) {
          throw std::runtime_error("destination column offset overflow during CHW->HWC conversion");
        }
        std::memcpy(dst_row + dst_off, src_bytes + src_off, elem);
      }
    }
  }

  out.read_only = true;
  warn_chw_to_hwc_copy_once(fmt);
  if (converted) {
    *converted = true;
  }
  return out;
}

TensorDType dtype_from_dl(const simaai::neat::dlpack::DLDataType& dtype) {
  if (dtype.lanes != 1)
    throw std::runtime_error("only lane=1 DLPack tensors are supported");

  if (dtype.code == DLDataTypeCode::kDLUInt) {
    if (dtype.bits == 8)
      return TensorDType::UInt8;
    if (dtype.bits == 16)
      return TensorDType::UInt16;
  }
  if (dtype.code == DLDataTypeCode::kDLInt) {
    if (dtype.bits == 8)
      return TensorDType::Int8;
    if (dtype.bits == 16)
      return TensorDType::Int16;
    if (dtype.bits == 32)
      return TensorDType::Int32;
  }
  if (dtype.code == DLDataTypeCode::kDLFloat) {
    if (dtype.bits == 32)
      return TensorDType::Float32;
    if (dtype.bits == 64)
      return TensorDType::Float64;
  }
  if (dtype.code == kDlBfloat && dtype.bits == 16) {
    return TensorDType::BFloat16;
  }

  throw std::runtime_error("unsupported DLPack dtype");
}

simaai::neat::dlpack::DLDataType dl_from_dtype(TensorDType dtype) {
  using simaai::neat::dlpack::DLDataType;
  if (dtype == TensorDType::UInt8)
    return DLDataType{DLDataTypeCode::kDLUInt, 8, 1};
  if (dtype == TensorDType::Int8)
    return DLDataType{DLDataTypeCode::kDLInt, 8, 1};
  if (dtype == TensorDType::UInt16)
    return DLDataType{DLDataTypeCode::kDLUInt, 16, 1};
  if (dtype == TensorDType::Int16)
    return DLDataType{DLDataTypeCode::kDLInt, 16, 1};
  if (dtype == TensorDType::Int32)
    return DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  if (dtype == TensorDType::BFloat16)
    return DLDataType{kDlBfloat, 16, 1};
  if (dtype == TensorDType::Float32)
    return DLDataType{DLDataTypeCode::kDLFloat, 32, 1};
  if (dtype == TensorDType::Float64)
    return DLDataType{DLDataTypeCode::kDLFloat, 64, 1};
  throw std::runtime_error("unsupported TensorDType");
}

struct DlpackExportOwner {
  Tensor tensor;
  simaai::neat::Mapping mapping;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_elems;
  DLManagedTensor managed{};
};

void delete_dlpack_managed(DLManagedTensor* managed) {
  if (!managed)
    return;
  auto* owner = static_cast<DlpackExportOwner*>(managed->manager_ctx);
  delete owner;
}

PyObject* tensor_to_dlpack_capsule(const Tensor& input) {
  Tensor tensor = input;
  if (tensor.device.type != DeviceType::CPU) {
    tensor = tensor.cpu();
  }
  if (!tensor.is_dense()) {
    throw std::runtime_error("__dlpack__ only supports dense tensors");
  }

  const std::size_t elem = dtype_bytes(tensor.dtype);
  if (elem == 0)
    throw std::runtime_error("unknown tensor dtype size");

  if (!tensor.shape.empty()) {
    if (tensor.strides_bytes.empty()) {
      tensor.strides_bytes = contiguous_strides_bytes(tensor.shape, elem);
    }
    for (int64_t stride_bytes : tensor.strides_bytes) {
      if (stride_bytes < 0 || (stride_bytes % static_cast<int64_t>(elem)) != 0) {
        throw std::runtime_error("tensor strides are not DLPack-compatible");
      }
    }
  }

  auto* owner = new DlpackExportOwner();
  owner->tensor = std::move(tensor);
  owner->mapping = owner->tensor.map_read();
  if (!owner->mapping.data)
    throw std::runtime_error("tensor mapping failed");

  owner->shape = owner->tensor.shape;
  owner->strides_elems.clear();
  owner->strides_elems.reserve(owner->tensor.strides_bytes.size());
  for (int64_t stride_bytes : owner->tensor.strides_bytes) {
    owner->strides_elems.push_back(stride_bytes / static_cast<int64_t>(elem));
  }

  owner->managed.dl_tensor.data = owner->mapping.data;
  owner->managed.dl_tensor.device = {simaai::neat::dlpack::DLDeviceType::kDLCPU, 0};
  owner->managed.dl_tensor.ndim = static_cast<int>(owner->shape.size());
  owner->managed.dl_tensor.dtype = dl_from_dtype(owner->tensor.dtype);
  owner->managed.dl_tensor.shape = owner->shape.empty() ? nullptr : owner->shape.data();
  owner->managed.dl_tensor.strides =
      owner->strides_elems.empty() ? nullptr : owner->strides_elems.data();
  owner->managed.dl_tensor.byte_offset = 0;
  owner->managed.manager_ctx = owner;
  owner->managed.deleter = delete_dlpack_managed;

  PyObject* capsule = PyCapsule_New(&owner->managed, "dltensor", [](PyObject* capsule_obj) {
    if (PyCapsule_IsValid(capsule_obj, "used_dltensor")) {
      return;
    }
    auto* managed_ptr =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_obj, "dltensor"));
    if (managed_ptr && managed_ptr->deleter) {
      managed_ptr->deleter(managed_ptr);
    }
  });
  if (!capsule) {
    delete owner;
  }
  return capsule;
}

Tensor tensor_from_dlpack_capsule_obj(PyObject* capsule_obj, bool copy,
                                      const std::optional<TensorLayout>& layout,
                                      const std::optional<ImageSpec::PixelFormat>& image_format,
                                      const std::optional<ByteFormat>& byte_format,
                                      TensorMemory memory) {
  if (!PyCapsule_IsValid(capsule_obj, "dltensor")) {
    throw std::runtime_error("expected an unconsumed dltensor capsule");
  }

  auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_obj, "dltensor"));
  if (!managed) {
    throw std::runtime_error("invalid dltensor capsule");
  }

  if (PyCapsule_SetName(capsule_obj, "used_dltensor") != 0) {
    throw std::runtime_error("failed to mark dltensor capsule as consumed");
  }
  if (PyCapsule_SetDestructor(capsule_obj, nullptr) != 0) {
    throw std::runtime_error("failed to detach dltensor capsule destructor");
  }

  auto deleter = [](DLManagedTensor* ptr) {
    if (ptr && ptr->deleter) {
      ptr->deleter(ptr);
    }
  };
  std::unique_ptr<DLManagedTensor, decltype(deleter)> managed_owner(managed, deleter);

  if (managed->dl_tensor.device.device_type != simaai::neat::dlpack::DLDeviceType::kDLCPU) {
    throw std::runtime_error("only CPU DLPack tensors are supported");
  }

  Tensor out;
  out.dtype = dtype_from_dl(managed->dl_tensor.dtype);
  out.layout = layout.value_or(TensorLayout::Unknown);
  out.device = Device{DeviceType::CPU, managed->dl_tensor.device.device_id};
  out.read_only = true;

  if (managed->dl_tensor.ndim < 0) {
    throw std::runtime_error("invalid DLPack ndim");
  }

  const std::size_t ndim = static_cast<std::size_t>(managed->dl_tensor.ndim);
  out.shape.assign(ndim, 0);
  for (std::size_t i = 0; i < ndim; ++i) {
    out.shape[i] = managed->dl_tensor.shape ? managed->dl_tensor.shape[i] : 0;
  }

  const std::size_t elem = dtype_bytes(out.dtype);
  if (elem == 0)
    throw std::runtime_error("unknown dtype size");

  if (managed->dl_tensor.strides) {
    out.strides_bytes.assign(ndim, 0);
    for (std::size_t i = 0; i < ndim; ++i) {
      const int64_t stride_elems = managed->dl_tensor.strides[i];
      if (stride_elems < 0)
        throw std::runtime_error("negative DLPack strides are not supported");
      std::size_t bytes = 0;
      if (!checked_mul(static_cast<std::size_t>(stride_elems), elem, &bytes)) {
        throw std::runtime_error("stride conversion overflow");
      }
      out.strides_bytes[i] = static_cast<int64_t>(bytes);
    }
  } else {
    out.strides_bytes = contiguous_strides_bytes(out.shape, elem);
  }

  std::size_t byte_offset = static_cast<std::size_t>(managed->dl_tensor.byte_offset);
  auto* base = static_cast<uint8_t*>(managed->dl_tensor.data);
  auto* data_ptr = base + byte_offset;

  std::size_t span = 0;
  if (ndim == 0) {
    span = elem;
  } else {
    span = span_bytes(out.shape, out.strides_bytes, elem);
  }

  auto holder = std::shared_ptr<void>(managed_owner.release(), [](void* ptr) {
    auto* dl = static_cast<DLManagedTensor*>(ptr);
    if (dl && dl->deleter) {
      dl->deleter(dl);
    }
  });

  out.storage = simaai::neat::make_cpu_external_storage(data_ptr, span, holder, true);
  out.byte_offset = 0;

  if (image_format.has_value()) {
    out.semantic.image = ImageSpec{*image_format, ""};
  }
  if (byte_format.has_value()) {
    if (image_format.has_value()) {
      throw std::runtime_error("byte_format tensors cannot also specify image_format");
    }
    ByteStreamSpec spec;
    spec.format = *byte_format;
    out.semantic.byte_stream = spec;
    out.layout = TensorLayout::Unknown;
  }

  bool chw_to_hwc_converted = false;
  out = maybe_convert_chw_image_to_hwc(std::move(out), &chw_to_hwc_converted);

  if (copy) {
    if (chw_to_hwc_converted) {
      out.read_only = false;
    } else {
      Tensor cloned = out.clone();
      cloned.layout = out.layout;
      cloned.semantic = out.semantic;
      cloned.read_only = false;
      out = std::move(cloned);
    }
  }

  if (memory == TensorMemory::Auto) {
    memory = TensorMemory::EV74;
  }
  if (memory == TensorMemory::CPU || memory == TensorMemory::A65) {
    return out;
  }
  if (memory == TensorMemory::EV74) {
    return out.cvu();
  }
  if (memory == TensorMemory::MLA) {
    return out.mla(true);
  }
  throw std::runtime_error("unsupported TensorMemory placement for Python tensor import");
}

Tensor tensor_from_dlpack_capsule(const nb::capsule& capsule, bool copy,
                                  const std::optional<TensorLayout>& layout,
                                  const std::optional<ImageSpec::PixelFormat>& image_format,
                                  TensorMemory memory,
                                  const std::optional<ByteFormat>& byte_format) {
  return tensor_from_dlpack_capsule_obj(capsule.ptr(), copy, layout, image_format, byte_format,
                                        memory);
}

std::optional<TensorLayout> infer_layout_from_object(const nb::object& obj,
                                                     const std::optional<TensorLayout>& layout) {
  if (layout.has_value()) {
    return layout;
  }

  const int has_ndim = PyObject_HasAttrString(obj.ptr(), "ndim");
  if (has_ndim < 0) {
    throw nb::python_error();
  }
  if (has_ndim == 0) {
    return TensorLayout::Unknown;
  }

  int64_t ndim = -1;
  try {
    ndim = nb::cast<int64_t>(obj.attr("ndim"));
  } catch (...) {
    return TensorLayout::Unknown;
  }

  if (ndim == 2) {
    return TensorLayout::HW;
  }
  if (ndim != 3) {
    return TensorLayout::Unknown;
  }

  bool prefer_chw = false;
  const int has_device = PyObject_HasAttrString(obj.ptr(), "device");
  if (has_device < 0) {
    throw nb::python_error();
  }
  if (has_device > 0) {
    prefer_chw = true;
  }

  const int has_shape = PyObject_HasAttrString(obj.ptr(), "shape");
  if (has_shape < 0) {
    throw nb::python_error();
  }
  if (has_shape > 0) {
    try {
      nb::object shape_obj = obj.attr("shape");
      if (PyTuple_Check(shape_obj.ptr())) {
        nb::tuple shape = nb::borrow<nb::tuple>(shape_obj);
        if (shape.size() == 3) {
          const int64_t c_first = nb::cast<int64_t>(shape[0]);
          const int64_t c_last = nb::cast<int64_t>(shape[2]);
          const bool first_channel_like = (c_first == 1 || c_first == 3 || c_first == 4);
          const bool last_channel_like = (c_last == 1 || c_last == 3 || c_last == 4);

          if (prefer_chw && first_channel_like && !last_channel_like) {
            return TensorLayout::CHW;
          }
          if (last_channel_like && !first_channel_like) {
            return TensorLayout::HWC;
          }
          if (first_channel_like && !last_channel_like) {
            return TensorLayout::CHW;
          }
          if (first_channel_like && last_channel_like) {
            return TensorLayout::Unknown;
          }
          if (!first_channel_like && !last_channel_like) {
            return TensorLayout::Unknown;
          }
        }
      } else if (PyList_Check(shape_obj.ptr())) {
        nb::list shape = nb::borrow<nb::list>(shape_obj);
        if (shape.size() == 3) {
          const int64_t c_first = nb::cast<int64_t>(shape[0]);
          const int64_t c_last = nb::cast<int64_t>(shape[2]);
          const bool first_channel_like = (c_first == 1 || c_first == 3 || c_first == 4);
          const bool last_channel_like = (c_last == 1 || c_last == 3 || c_last == 4);

          if (prefer_chw && first_channel_like && !last_channel_like) {
            return TensorLayout::CHW;
          }
          if (last_channel_like && !first_channel_like) {
            return TensorLayout::HWC;
          }
          if (first_channel_like && !last_channel_like) {
            return TensorLayout::CHW;
          }
          if (first_channel_like && last_channel_like) {
            return TensorLayout::Unknown;
          }
          if (!first_channel_like && !last_channel_like) {
            return TensorLayout::Unknown;
          }
        }
      }
    } catch (...) {
      // Fall through to default behavior.
    }
  }

  return TensorLayout::Unknown;
}

Tensor tensor_from_dlpack_like_object(const nb::object& input, bool copy,
                                      const std::optional<TensorLayout>& layout,
                                      const std::optional<ImageSpec::PixelFormat>& image_format,
                                      const std::optional<ByteFormat>& byte_format,
                                      TensorMemory memory) {
  nb::object source = input;

  // Match Python wrapper behavior: for torch tensors, move non-CPU tensors to CPU.
  const int has_device = PyObject_HasAttrString(source.ptr(), "device");
  if (has_device < 0) {
    throw nb::python_error();
  }
  if (has_device > 0) {
    try {
      nb::object device = source.attr("device");
      const int has_device_type = PyObject_HasAttrString(device.ptr(), "type");
      if (has_device_type < 0) {
        throw nb::python_error();
      }
      if (has_device_type > 0) {
        std::string device_type = nb::cast<std::string>(device.attr("type"));
        if (device_type != "cpu") {
          source = source.attr("cpu")();
        }
      }
    } catch (...) {
      // Ignore best-effort device probing failures and try direct DLPack export.
    }
  }

  const int has_dlpack = PyObject_HasAttrString(source.ptr(), "__dlpack__");
  if (has_dlpack < 0) {
    throw nb::python_error();
  }
  if (has_dlpack == 0) {
    throw std::runtime_error(
        "expected Tensor, NumPy/Torch tensor, or any object implementing __dlpack__()");
  }

  nb::object capsule_obj = source.attr("__dlpack__")();
  return tensor_from_dlpack_capsule_obj(capsule_obj.ptr(), copy,
                                        byte_format
                                            ? std::optional<TensorLayout>(TensorLayout::Unknown)
                                            : infer_layout_from_object(source, layout),
                                        image_format, byte_format, memory);
}

Tensor tensor_from_python_input(const nb::object& input, bool copy,
                                const std::optional<TensorLayout>& layout,
                                const std::optional<ImageSpec::PixelFormat>& image_format,
                                const std::optional<ByteFormat>& byte_format = std::nullopt,
                                TensorMemory memory = TensorMemory::EV74) {
  if (nb::isinstance<Tensor>(input)) {
    Tensor tensor = nb::cast<Tensor>(input);
    if (image_format.has_value() && !tensor.semantic.image.has_value()) {
      tensor.semantic.image = ImageSpec{*image_format, ""};
    }
    if (byte_format.has_value()) {
      if (image_format.has_value() || tensor.semantic.image.has_value()) {
        throw std::runtime_error("byte_format tensors cannot also specify image_format");
      }
      ByteStreamSpec spec;
      spec.format = *byte_format;
      tensor.semantic.byte_stream = spec;
      tensor.layout = TensorLayout::Unknown;
    }
    return tensor;
  }
  return tensor_from_dlpack_like_object(input, copy, layout, image_format, byte_format, memory);
}

std::vector<Tensor>
tensor_batch_from_python_input(const nb::object& input, bool copy,
                               const std::optional<TensorLayout>& layout,
                               const std::optional<ImageSpec::PixelFormat>& image_format,
                               const std::optional<ByteFormat>& byte_format = std::nullopt,
                               TensorMemory memory = TensorMemory::EV74) {
  std::vector<Tensor> tensors;
  if (PyList_Check(input.ptr())) {
    nb::list items = nb::borrow<nb::list>(input);
    tensors.reserve(items.size());
    for (nb::handle h : items) {
      tensors.emplace_back(tensor_from_python_input(nb::borrow<nb::object>(h), copy, layout,
                                                    image_format, byte_format, memory));
    }
    return tensors;
  }
  if (PyTuple_Check(input.ptr())) {
    nb::tuple items = nb::borrow<nb::tuple>(input);
    tensors.reserve(items.size());
    for (nb::handle h : items) {
      tensors.emplace_back(tensor_from_python_input(nb::borrow<nb::object>(h), copy, layout,
                                                    image_format, byte_format, memory));
    }
    return tensors;
  }
  throw std::runtime_error("expected list/tuple of Tensor or DLPack-compatible inputs");
}

#if defined(SIMA_WITH_OPENCV)
struct PythonPreprocImageBatch {
  std::vector<Tensor> tensors;
  std::vector<simaai::neat::Mapping> mappings;
  std::vector<cv::Mat> mats;
};

int int_dim_or_throw(int64_t value, const char* name) {
  if (value <= 0 || value > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string("pyneat.stages.preproc: invalid ") + name + " dimension");
  }
  return static_cast<int>(value);
}

std::vector<int64_t> contiguous_uint8_image_strides(const std::vector<int64_t>& shape) {
  if (shape.size() == 2) {
    return {shape[1], 1};
  }
  if (shape.size() == 3) {
    return {shape[1] * shape[2], shape[2], 1};
  }
  return {};
}

PythonPreprocImageBatch
python_preproc_images_to_cv_mats(const nb::object& images,
                                 const std::optional<ImageSpec::PixelFormat>& image_format,
                                 bool copy) {
  PythonPreprocImageBatch batch;
  batch.tensors = tensor_batch_from_python_input(images, copy, std::nullopt, image_format,
                                                 std::nullopt, TensorMemory::CPU);
  if (batch.tensors.empty()) {
    throw std::runtime_error("pyneat.stages.preproc: images must not be empty");
  }

  batch.mappings.reserve(batch.tensors.size());
  batch.mats.reserve(batch.tensors.size());

  for (Tensor& tensor : batch.tensors) {
    tensor = tensor.to_cpu_if_needed();
    if (!tensor.is_dense()) {
      throw std::runtime_error(
          "pyneat.stages.preproc: ROI/full-frame image inputs must be dense tensors");
    }
    if (tensor.dtype != TensorDType::UInt8) {
      throw std::runtime_error("pyneat.stages.preproc: images must be uint8");
    }
    if (tensor.shape.size() != 2 && tensor.shape.size() != 3) {
      throw std::runtime_error("pyneat.stages.preproc: each image must be rank-2 HW or rank-3 HWC");
    }
    if (tensor.layout == TensorLayout::CHW) {
      throw std::runtime_error(
          "pyneat.stages.preproc: CHW Tensor inputs are unsupported; pass HWC/HW images");
    }

    int height = int_dim_or_throw(tensor.shape[0], "height");
    int width = int_dim_or_throw(tensor.shape[1], "width");
    int channels = 1;
    if (tensor.shape.size() == 3) {
      channels = int_dim_or_throw(tensor.shape[2], "channel");
      if (channels != 1 && channels != 3) {
        throw std::runtime_error("pyneat.stages.preproc: rank-3 images must have 1 or 3 channels");
      }
    }

    std::vector<int64_t> strides = tensor.strides_bytes;
    if (strides.empty()) {
      strides = contiguous_uint8_image_strides(tensor.shape);
    }
    if (strides.size() != tensor.shape.size()) {
      throw std::runtime_error("pyneat.stages.preproc: image strides must match image rank");
    }

    const int64_t row_stride = strides[0];
    const int64_t min_row_bytes = static_cast<int64_t>(width) * channels;
    bool needs_contiguous = row_stride < min_row_bytes;
    if (tensor.shape.size() == 2) {
      needs_contiguous = needs_contiguous || strides[1] != 1;
    } else {
      needs_contiguous = needs_contiguous || strides[1] != channels || strides[2] != 1;
    }
    if (needs_contiguous || !tensor.is_contiguous()) {
      tensor = tensor.contiguous();
      strides = tensor.strides_bytes.empty() ? contiguous_uint8_image_strides(tensor.shape)
                                             : tensor.strides_bytes;
    }

    const int64_t final_row_stride = strides[0];
    if (final_row_stride < min_row_bytes || (tensor.shape.size() == 2 && strides[1] != 1) ||
        (tensor.shape.size() == 3 && (strides[1] != channels || strides[2] != 1))) {
      throw std::runtime_error(
          "pyneat.stages.preproc: images must be packed HW or packed HWC after conversion");
    }
    simaai::neat::Mapping mapping = tensor.map_read();
    if (!mapping.data) {
      throw std::runtime_error("pyneat.stages.preproc: failed to map image tensor");
    }
    const int64_t required_bytes =
        (static_cast<int64_t>(height) - 1) * final_row_stride + min_row_bytes;
    if (required_bytes < 0 || mapping.size_bytes < static_cast<std::size_t>(required_bytes)) {
      throw std::runtime_error("pyneat.stages.preproc: image mapping is smaller than image view");
    }

    auto* data = static_cast<std::uint8_t*>(mapping.data);
    batch.mappings.push_back(std::move(mapping));
    batch.mats.emplace_back(height, width, CV_8UC(channels), data,
                            static_cast<std::size_t>(final_row_stride));
  }

  return batch;
}
#endif

TensorList python_stage_preproc(const nb::object& images, const simaai::neat::Model& model,
                                const std::optional<std::vector<simaai::neat::PreprocessRoi>>& rois,
                                const std::optional<ImageSpec::PixelFormat>& image_format,
                                bool copy) {
#if defined(SIMA_WITH_OPENCV)
  PythonPreprocImageBatch batch = python_preproc_images_to_cv_mats(images, image_format, copy);
  TensorList out;
  {
    nb::gil_scoped_release release;
    if (rois.has_value()) {
      out = simaai::neat::stages::Preproc(batch.mats, model, *rois);
    } else {
      out = simaai::neat::stages::Preproc(batch.mats, model);
    }
  }
  return out;
#else
  (void)images;
  (void)model;
  (void)rois;
  (void)image_format;
  (void)copy;
  throw std::runtime_error(
      "pyneat.stages.preproc requires a NEAT build with SIMA_WITH_OPENCV enabled");
#endif
}

std::vector<Tensor> genai_image_tensors_from_python(const nb::object& input) {
  auto images = tensor_batch_from_python_input(
      input, true, TensorLayout::HWC, ImageSpec::PixelFormat::RGB, std::nullopt, TensorMemory::CPU);
  for (Tensor& image : images) {
    if (!image.semantic.image.has_value()) {
      image.semantic.image = ImageSpec{ImageSpec::PixelFormat::RGB, ""};
    }
    image = image.to_cpu_if_needed();
  }
  return images;
}

bool python_list_or_tuple(const nb::object& input) {
  return PyList_Check(input.ptr()) || PyTuple_Check(input.ptr());
}

bool python_sequence_all_samples(const nb::object& input) {
  if (!python_list_or_tuple(input)) {
    return false;
  }
  if (PyList_Check(input.ptr())) {
    nb::list items = nb::borrow<nb::list>(input);
    if (items.size() == 0) {
      return false;
    }
    for (nb::handle h : items) {
      if (!nb::isinstance<Sample>(h)) {
        return false;
      }
    }
    return true;
  }
  nb::tuple items = nb::borrow<nb::tuple>(input);
  if (items.size() == 0) {
    return false;
  }
  for (nb::handle h : items) {
    if (!nb::isinstance<Sample>(h)) {
      return false;
    }
  }
  return true;
}

Sample sample_batch_from_python_input(const nb::object& input) {
  Sample samples;
  if (PyList_Check(input.ptr())) {
    nb::list items = nb::borrow<nb::list>(input);
    samples.reserve(items.size());
    for (nb::handle h : items) {
      if (!nb::isinstance<Sample>(h)) {
        throw std::runtime_error("expected list/tuple of Sample");
      }
      samples.push_back(nb::cast<Sample>(h));
    }
    return samples;
  }
  if (PyTuple_Check(input.ptr())) {
    nb::tuple items = nb::borrow<nb::tuple>(input);
    samples.reserve(items.size());
    for (nb::handle h : items) {
      if (!nb::isinstance<Sample>(h)) {
        throw std::runtime_error("expected list/tuple of Sample");
      }
      samples.push_back(nb::cast<Sample>(h));
    }
    return samples;
  }
  throw std::runtime_error("expected list/tuple of Sample");
}

void reject_single_tensor_or_sample(const nb::object& input, const char* where) {
  if (nb::isinstance<Sample>(input)) {
    throw std::runtime_error(std::string(where) +
                             " expects Sample; pass [sample] instead of a single Sample");
  }
  if (nb::isinstance<Tensor>(input)) {
    throw std::runtime_error(std::string(where) +
                             " expects TensorList; pass [tensor] instead of a single Tensor");
  }
}

nb::bytes tensor_dense_bytes(const Tensor& t) {
  std::vector<uint8_t> data = t.copy_dense_bytes_tight();
  return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

nb::bytes tensor_payload_bytes(const Tensor& t) {
  std::vector<uint8_t> data = t.copy_payload_bytes();
  return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

std::optional<ImageSpec::PixelFormat> tensor_image_format_value(const Tensor& t) {
  return t.image_format();
}

nb::tuple tensor_validate(const Tensor& t) {
  std::string err;
  const bool ok = t.validate(&err);
  return nb::make_tuple(ok, err);
}

std::string format_neat_error_message(const NeatError& e) {
  std::string msg = e.what() ? std::string(e.what()) : std::string{};
  const GraphReport& rep = e.report();

  if (msg.empty() || msg == "[" || msg == "[]") {
    if (!rep.error_code.empty() && !rep.repro_note.empty()) {
      return "[" + rep.error_code + "] " + rep.repro_note;
    }
    if (!rep.error_code.empty()) {
      return "[" + rep.error_code + "] NeatError";
    }
    if (!rep.repro_note.empty()) {
      return rep.repro_note;
    }
    return "NeatError";
  }

  if (!rep.error_code.empty()) {
    const std::string tag = "[" + rep.error_code + "]";
    if (msg.rfind(tag, 0) != 0) {
      msg = tag + " " + msg;
    }
  }
  if (!rep.repro_note.empty() && msg.find(rep.repro_note) == std::string::npos) {
    msg += "\n" + rep.repro_note;
  }
  return msg;
}

void set_python_attr_or_throw(PyObject* obj, const char* key, const std::string& value) {
  if (!obj || !key) {
    return;
  }
  if (PyObject_SetAttrString(obj, key, nb::str(value.c_str(), value.size()).ptr()) != 0) {
    throw nb::python_error();
  }
}

bool detection_decode_type_error_message(const std::string& msg) {
  return msg.find("is not a ") != std::string::npos ||
         msg.find("expected a BBOX-family") != std::string::npos ||
         msg.find("format mismatch") != std::string::npos;
}

} // namespace

NB_MODULE(_pyneat_core, m) {
  m.doc() = "Python bindings for SiMa NEAT";

  nb::exception<NeatError> py_neat_error(m, "NeatError");
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        try {
          std::rethrow_exception(p);
        } catch (const NeatError& e) {
          PyObject* exc_type = reinterpret_cast<PyObject*>(payload);
          if (!exc_type) {
            throw;
          }

          const GraphReport& rep = e.report();
          const std::string msg = format_neat_error_message(e);
          nb::object exc_obj = nb::steal(PyObject_CallFunctionObjArgs(
              exc_type, nb::str(msg.c_str(), msg.size()).ptr(), nullptr));
          if (!exc_obj.is_valid()) {
            throw nb::python_error();
          }

          set_python_attr_or_throw(exc_obj.ptr(), "error_code", rep.error_code);
          set_python_attr_or_throw(exc_obj.ptr(), "repro_note", rep.repro_note);
          set_python_attr_or_throw(exc_obj.ptr(), "pipeline_string", rep.pipeline_string);
          set_python_attr_or_throw(exc_obj.ptr(), "report_json", rep.to_json());
          PyErr_SetObject(exc_type, exc_obj.ptr());
        }
      },
      py_neat_error.ptr());
#ifndef PYNEAT_VERSION
#define PYNEAT_VERSION "0.0.0"
#endif
  m.attr("__version__") = PYNEAT_VERSION;
  m.attr("_HAS_NATIVE_BUILD_OBJECT_OVERLOADS") = true;

  nb::enum_<TensorDType>(m, "TensorDType")
      .value("UInt8", TensorDType::UInt8)
      .value("Int8", TensorDType::Int8)
      .value("UInt16", TensorDType::UInt16)
      .value("Int16", TensorDType::Int16)
      .value("Int32", TensorDType::Int32)
      .value("BFloat16", TensorDType::BFloat16)
      .value("Float32", TensorDType::Float32)
      .value("Float64", TensorDType::Float64);

  nb::enum_<TensorLayout>(m, "TensorLayout")
      .value("Unknown", TensorLayout::Unknown)
      .value("HWC", TensorLayout::HWC)
      .value("CHW", TensorLayout::CHW)
      .value("HW", TensorLayout::HW);

  nb::enum_<TensorMemory>(m, "TensorMemory")
      .value("Auto", TensorMemory::Auto)
      .value("CPU", TensorMemory::CPU)
      .value("A65", TensorMemory::A65)
      .value("EV74", TensorMemory::EV74)
      .value("MLA", TensorMemory::MLA);

  nb::enum_<simaai::neat::TensorAxisSemantic>(m, "TensorAxisSemantic")
      .value("Unknown", simaai::neat::TensorAxisSemantic::Unknown)
      .value("N", simaai::neat::TensorAxisSemantic::N)
      .value("D", simaai::neat::TensorAxisSemantic::D)
      .value("H", simaai::neat::TensorAxisSemantic::H)
      .value("W", simaai::neat::TensorAxisSemantic::W)
      .value("C", simaai::neat::TensorAxisSemantic::C);

  nb::enum_<DeviceType>(m, "DeviceType")
      .value("CPU", DeviceType::CPU)
      .value("SIMA_APU", DeviceType::SIMA_APU)
      .value("SIMA_CVU", DeviceType::SIMA_CVU)
      .value("SIMA_MLA", DeviceType::SIMA_MLA)
      .value("UNKNOWN", DeviceType::UNKNOWN);

  nb::enum_<simaai::neat::StorageKind>(m, "StorageKind")
      .value("CpuOwned", simaai::neat::StorageKind::CpuOwned)
      .value("CpuExternal", simaai::neat::StorageKind::CpuExternal)
      .value("GstSample", simaai::neat::StorageKind::GstSample)
      .value("DeviceHandle", simaai::neat::StorageKind::DeviceHandle)
      .value("Unknown", simaai::neat::StorageKind::Unknown);

  nb::enum_<simaai::neat::PlaneRole>(m, "PlaneRole")
      .value("Unknown", simaai::neat::PlaneRole::Unknown)
      .value("Y", simaai::neat::PlaneRole::Y)
      .value("U", simaai::neat::PlaneRole::U)
      .value("V", simaai::neat::PlaneRole::V)
      .value("UV", simaai::neat::PlaneRole::UV);

  nb::enum_<MapMode>(m, "MapMode")
      .value("Read", MapMode::Read)
      .value("Write", MapMode::Write)
      .value("ReadWrite", MapMode::ReadWrite);

  nb::enum_<PowerMonitorProfile>(m, "PowerMonitorProfile")
      .value("Auto", PowerMonitorProfile::Auto)
      .value("ModalixSom", PowerMonitorProfile::ModalixSom)
      .value("ModalixDvt", PowerMonitorProfile::ModalixDvt)
      .value("Custom", PowerMonitorProfile::Custom);

  nb::enum_<simaai::neat::OverflowPolicy>(m, "OverflowPolicy")
      .value("Block", simaai::neat::OverflowPolicy::Block)
      .value("KeepLatest", simaai::neat::OverflowPolicy::KeepLatest)
      .value("DropIncoming", simaai::neat::OverflowPolicy::DropIncoming);

  nb::enum_<RunPreset>(m, "RunPreset")
      .value("Realtime", RunPreset::Realtime)
      .value("Balanced", RunPreset::Balanced)
      .value("Reliable", RunPreset::Reliable);

  nb::enum_<OutputMemory>(m, "OutputMemory")
      .value("Auto", OutputMemory::Auto)
      .value("ZeroCopy", OutputMemory::ZeroCopy)
      .value("Owned", OutputMemory::Owned);

  nb::enum_<SampleKind>(m, "SampleKind")
      .value("Tensor", SampleKind::Tensor)
      .value("TensorSet", SampleKind::TensorSet)
      .value("Bundle", SampleKind::Bundle)
      .value("Unknown", SampleKind::Unknown);

  nb::enum_<simaai::neat::PayloadType>(m, "PayloadType")
      .value("Auto", simaai::neat::PayloadType::Auto)
      .value("Image", simaai::neat::PayloadType::Image)
      .value("Tensor", simaai::neat::PayloadType::Tensor)
      .value("Encoded", simaai::neat::PayloadType::Encoded);
  m.attr("InputType") = m.attr("PayloadType");

  nb::enum_<simaai::neat::InputMemoryPolicy>(m, "InputMemoryPolicy")
      .value("Auto", simaai::neat::InputMemoryPolicy::Auto)
      .value("Ev74", simaai::neat::InputMemoryPolicy::Ev74)
      .value("Dms0", simaai::neat::InputMemoryPolicy::Dms0)
      .value("SystemMemory", simaai::neat::InputMemoryPolicy::SystemMemory);

  nb::enum_<PullStatus>(m, "PullStatus")
      .value("Ok", PullStatus::Ok)
      .value("Timeout", PullStatus::Timeout)
      .value("Closed", PullStatus::Closed)
      .value("Error", PullStatus::Error);

  nb::enum_<simaai::neat::InputRole>(m, "InputRole")
      .value("None", simaai::neat::InputRole::None)
      .value("Push", simaai::neat::InputRole::Push)
      .value("Source", simaai::neat::InputRole::Source);

  nb::enum_<simaai::neat::NodeCapsBehavior>(m, "NodeCapsBehavior")
      .value("Static", simaai::neat::NodeCapsBehavior::Static)
      .value("Dynamic", simaai::neat::NodeCapsBehavior::Dynamic);

  nb::enum_<simaai::neat::CapsMemory>(m, "CapsMemory")
      .value("Any", simaai::neat::CapsMemory::Any)
      .value("SystemMemory", simaai::neat::CapsMemory::SystemMemory);

  nb::class_<Device>(m, "Device")
      .def(nb::init<>())
      .def_rw("type", &Device::type)
      .def_rw("id", &Device::id);

  nb::class_<ImageSpec>(m, "ImageSpec")
      .def(nb::init<>())
      .def_rw("format", &ImageSpec::format)
      .def_rw("color_space", &ImageSpec::color_space);

  nb::enum_<ImageSpec::PixelFormat>(m, "PixelFormat")
      .value("RGB", ImageSpec::PixelFormat::RGB)
      .value("BGR", ImageSpec::PixelFormat::BGR)
      .value("GRAY8", ImageSpec::PixelFormat::GRAY8)
      .value("NV12", ImageSpec::PixelFormat::NV12)
      .value("I420", ImageSpec::PixelFormat::I420)
      .value("UNKNOWN", ImageSpec::PixelFormat::UNKNOWN);
  m.attr("Memory") = m.attr("TensorMemory");
  m.attr("ImageType") = m.attr("PixelFormat");

  nb::class_<simaai::neat::AudioSpec>(m, "AudioSpec")
      .def(nb::init<>())
      .def_rw("sample_rate", &simaai::neat::AudioSpec::sample_rate)
      .def_rw("channels", &simaai::neat::AudioSpec::channels)
      .def_rw("interleaved", &simaai::neat::AudioSpec::interleaved);

  nb::class_<simaai::neat::TokensSpec>(m, "TokensSpec")
      .def(nb::init<>())
      .def_rw("vocab_size", &simaai::neat::TokensSpec::vocab_size);

  nb::class_<simaai::neat::TextSpec>(m, "TextSpec")
      .def(nb::init<>())
      .def_rw("encoding", &simaai::neat::TextSpec::encoding);

  nb::class_<simaai::neat::EncodedSpec>(m, "EncodedSpec")
      .def(nb::init<>())
      .def_rw("codec", &simaai::neat::EncodedSpec::codec);

  nb::enum_<simaai::neat::ByteFormat>(m, "ByteFormat").value("Raw", simaai::neat::ByteFormat::Raw);

  nb::class_<simaai::neat::ByteStreamSpec>(m, "ByteStreamSpec")
      .def(nb::init<>())
      .def_rw("format", &simaai::neat::ByteStreamSpec::format)
      .def_rw("description", &simaai::neat::ByteStreamSpec::description);

  nb::enum_<simaai::neat::EncodedSpec::Codec>(m, "EncodedCodec")
      .value("H264", simaai::neat::EncodedSpec::Codec::H264)
      .value("H265", simaai::neat::EncodedSpec::Codec::H265)
      .value("RTP_H264", simaai::neat::EncodedSpec::Codec::RTP_H264)
      .value("RTP_H265", simaai::neat::EncodedSpec::Codec::RTP_H265)
      .value("JPEG", simaai::neat::EncodedSpec::Codec::JPEG)
      .value("UNKNOWN", simaai::neat::EncodedSpec::Codec::UNKNOWN);

  nb::class_<simaai::neat::QuantSpec>(m, "QuantSpec")
      .def(nb::init<>())
      .def_rw("scale", &simaai::neat::QuantSpec::scale)
      .def_rw("zero_point", &simaai::neat::QuantSpec::zero_point)
      .def_rw("axis", &simaai::neat::QuantSpec::axis)
      .def_rw("scales", &simaai::neat::QuantSpec::scales)
      .def_rw("zero_points", &simaai::neat::QuantSpec::zero_points);

  nb::class_<simaai::neat::TessSpec>(m, "TessSpec")
      .def(nb::init<>())
      .def_rw("slice_shape", &simaai::neat::TessSpec::slice_shape)
      .def("set_slice_shape", &simaai::neat::TessSpec::set_slice_shape, "shape"_a)
      .def_rw("format", &simaai::neat::TessSpec::format);

  nb::class_<simaai::neat::DetectionSpec>(
      m, "DetectionSpec",
      "Detection-decoder output metadata: identifies the wire format the consumer "
      "should parse (e.g. 'BBOX'). Lives in Semantic.detection.")
      .def(nb::init<>())
      .def_rw("format", &simaai::neat::DetectionSpec::format);

  nb::class_<simaai::neat::PreprocessRoi>(
      m, "PreprocessRoi",
      "Runtime ROI window consumed by Preproc. batch_index selects the source image; "
      "x/y/width/height are source-frame pixels.")
      .def(nb::init<>())
      .def(nb::init<int, int, int, int, int>(), "batch_index"_a = 0, "x"_a = 0, "y"_a = 0,
           "width"_a = 0, "height"_a = 0)
      .def_rw("batch_index", &simaai::neat::PreprocessRoi::batch_index)
      .def_rw("x", &simaai::neat::PreprocessRoi::x)
      .def_rw("y", &simaai::neat::PreprocessRoi::y)
      .def_rw("width", &simaai::neat::PreprocessRoi::width)
      .def_rw("height", &simaai::neat::PreprocessRoi::height);

  nb::class_<simaai::neat::PreprocessAffine>(
      m, "PreprocessAffine",
      "Per-ROI 2x3 affine from model/preprocessed coordinates back to source-frame "
      "coordinates.")
      .def(nb::init<>())
      .def_rw("m00", &simaai::neat::PreprocessAffine::m00)
      .def_rw("m01", &simaai::neat::PreprocessAffine::m01)
      .def_rw("m02", &simaai::neat::PreprocessAffine::m02)
      .def_rw("m10", &simaai::neat::PreprocessAffine::m10)
      .def_rw("m11", &simaai::neat::PreprocessAffine::m11)
      .def_rw("m12", &simaai::neat::PreprocessAffine::m12);

  nb::class_<simaai::neat::PreprocessRuntimeMeta>(
      m, "PreprocessRuntimeMeta",
      "Per-tensor preprocess metadata: resize/letterbox geometry, transform flags, "
      "and ROI-list breadcrumbs.")
      .def(nb::init<>())
      .def_rw("original_width", &simaai::neat::PreprocessRuntimeMeta::original_width)
      .def_rw("original_height", &simaai::neat::PreprocessRuntimeMeta::original_height)
      .def_rw("resized_width", &simaai::neat::PreprocessRuntimeMeta::resized_width)
      .def_rw("resized_height", &simaai::neat::PreprocessRuntimeMeta::resized_height)
      .def_rw("scaled_width", &simaai::neat::PreprocessRuntimeMeta::scaled_width)
      .def_rw("scaled_height", &simaai::neat::PreprocessRuntimeMeta::scaled_height)
      .def_rw("pad_left", &simaai::neat::PreprocessRuntimeMeta::pad_left)
      .def_rw("pad_right", &simaai::neat::PreprocessRuntimeMeta::pad_right)
      .def_rw("pad_top", &simaai::neat::PreprocessRuntimeMeta::pad_top)
      .def_rw("pad_bottom", &simaai::neat::PreprocessRuntimeMeta::pad_bottom)
      .def_rw("resize_mode", &simaai::neat::PreprocessRuntimeMeta::resize_mode)
      .def_rw("color_in", &simaai::neat::PreprocessRuntimeMeta::color_in)
      .def_rw("color_out", &simaai::neat::PreprocessRuntimeMeta::color_out)
      .def_rw("axis_perm", &simaai::neat::PreprocessRuntimeMeta::axis_perm)
      .def_rw("normalize", &simaai::neat::PreprocessRuntimeMeta::normalize)
      .def_rw("quantize", &simaai::neat::PreprocessRuntimeMeta::quantize)
      .def_rw("tessellate", &simaai::neat::PreprocessRuntimeMeta::tessellate)
      .def_rw("affine_m00", &simaai::neat::PreprocessRuntimeMeta::affine_m00)
      .def_rw("affine_m01", &simaai::neat::PreprocessRuntimeMeta::affine_m01)
      .def_rw("affine_m02", &simaai::neat::PreprocessRuntimeMeta::affine_m02)
      .def_rw("affine_m10", &simaai::neat::PreprocessRuntimeMeta::affine_m10)
      .def_rw("affine_m11", &simaai::neat::PreprocessRuntimeMeta::affine_m11)
      .def_rw("affine_m12", &simaai::neat::PreprocessRuntimeMeta::affine_m12)
      .def_rw("affine_scale_x", &simaai::neat::PreprocessRuntimeMeta::affine_scale_x)
      .def_rw("affine_scale_y", &simaai::neat::PreprocessRuntimeMeta::affine_scale_y)
      .def_rw("affine_offset_x", &simaai::neat::PreprocessRuntimeMeta::affine_offset_x)
      .def_rw("affine_offset_y", &simaai::neat::PreprocessRuntimeMeta::affine_offset_y)
      .def_rw("roi_list_enabled", &simaai::neat::PreprocessRuntimeMeta::roi_list_enabled)
      .def_rw("rois", &simaai::neat::PreprocessRuntimeMeta::rois)
      .def_rw("roi_input_batch_size", &simaai::neat::PreprocessRuntimeMeta::roi_input_batch_size)
      .def_rw("roi_source_width", &simaai::neat::PreprocessRuntimeMeta::roi_source_width)
      .def_rw("roi_source_height", &simaai::neat::PreprocessRuntimeMeta::roi_source_height)
      .def_rw("roi_source_stride_bytes",
              &simaai::neat::PreprocessRuntimeMeta::roi_source_stride_bytes)
      .def_rw("roi_pad_value", &simaai::neat::PreprocessRuntimeMeta::roi_pad_value)
      .def_rw("roi_capacity", &simaai::neat::PreprocessRuntimeMeta::roi_capacity)
      .def_rw("roi_valid_count", &simaai::neat::PreprocessRuntimeMeta::roi_valid_count)
      .def_rw("roi_input_count", &simaai::neat::PreprocessRuntimeMeta::roi_input_count)
      .def_rw("roi_dropped_invalid", &simaai::neat::PreprocessRuntimeMeta::roi_dropped_invalid)
      .def_rw("roi_dropped_overflow", &simaai::neat::PreprocessRuntimeMeta::roi_dropped_overflow)
      .def_rw("roi_affines", &simaai::neat::PreprocessRuntimeMeta::roi_affines)
      .def("has_axis_perm", &simaai::neat::PreprocessRuntimeMeta::has_axis_perm)
      .def("has_roi_list", &simaai::neat::PreprocessRuntimeMeta::has_roi_list);

  nb::class_<simaai::neat::Semantic>(m, "Semantic")
      .def(nb::init<>())
      .def_rw("image", &simaai::neat::Semantic::image)
      .def_rw("audio", &simaai::neat::Semantic::audio)
      .def_rw("tokens", &simaai::neat::Semantic::tokens)
      .def_rw("text", &simaai::neat::Semantic::text)
      .def_rw("byte_stream", &simaai::neat::Semantic::byte_stream)
      .def_rw("tess", &simaai::neat::Semantic::tess)
      .def_rw("encoded", &simaai::neat::Semantic::encoded)
      .def_rw("quant", &simaai::neat::Semantic::quant)
      .def_rw("detection", &simaai::neat::Semantic::detection)
      .def_rw("preprocess", &simaai::neat::Semantic::preprocess);

  nb::class_<simaai::neat::Segment>(m, "Segment")
      .def(nb::init<>())
      .def_rw("name", &simaai::neat::Segment::name)
      .def_rw("size_bytes", &simaai::neat::Segment::size_bytes);

  nb::class_<simaai::neat::Storage>(m, "Storage")
      .def_prop_ro("kind", [](const simaai::neat::Storage& s) { return s.kind; })
      .def_prop_ro("device", [](const simaai::neat::Storage& s) { return s.device; })
      .def_prop_ro("size_bytes", [](const simaai::neat::Storage& s) { return s.size_bytes; })
      .def_prop_ro("sima_segments", [](const simaai::neat::Storage& s) { return s.sima_segments; });

  nb::class_<simaai::neat::Plane>(m, "Plane")
      .def(nb::init<>())
      .def_rw("role", &simaai::neat::Plane::role)
      .def_rw("shape", &simaai::neat::Plane::shape)
      .def_rw("strides_bytes", &simaai::neat::Plane::strides_bytes)
      .def_rw("byte_offset", &simaai::neat::Plane::byte_offset);

  // Phase 1 (plan high-value slice S12): output-identity metadata — "which output is which" for
  // multi-output models. Bound as a single nested object on Tensor.route so the (jargon-y) routing
  // fields stay nested, not polluting the top-level surface. Full fields exposed for power users.
  nb::class_<simaai::neat::TensorRouteMeta>(m, "TensorRouteMeta")
      .def(nb::init<>())
      .def_rw("stage_key", &simaai::neat::TensorRouteMeta::stage_key)
      .def_rw("logical_index", &simaai::neat::TensorRouteMeta::logical_index)
      .def_rw("backend_output_index", &simaai::neat::TensorRouteMeta::backend_output_index)
      .def_rw("route_slot", &simaai::neat::TensorRouteMeta::route_slot)
      .def_rw("physical_index", &simaai::neat::TensorRouteMeta::physical_index)
      .def_rw("memory_index", &simaai::neat::TensorRouteMeta::memory_index)
      .def_rw("physical_byte_offset", &simaai::neat::TensorRouteMeta::physical_byte_offset)
      .def_rw("name", &simaai::neat::TensorRouteMeta::name)
      .def_rw("backend_name", &simaai::neat::TensorRouteMeta::backend_name)
      .def_rw("segment_name", &simaai::neat::TensorRouteMeta::segment_name);

  nb::class_<simaai::neat::Tensor>(m, "Tensor")
      .def(nb::init<>())
      .def_rw("dtype", &simaai::neat::Tensor::dtype)
      .def_rw("layout", &simaai::neat::Tensor::layout)
      .def_rw("shape", &simaai::neat::Tensor::shape)
      .def_rw("strides_bytes", &simaai::neat::Tensor::strides_bytes)
      .def_rw("byte_offset", &simaai::neat::Tensor::byte_offset)
      .def_rw("axis_semantics", &simaai::neat::Tensor::axis_semantics)
      .def_rw("device", &simaai::neat::Tensor::device)
      .def_rw("semantic", &simaai::neat::Tensor::semantic)
      .def_rw("planes", &simaai::neat::Tensor::planes)
      .def_rw("read_only", &simaai::neat::Tensor::read_only)
      .def_rw("route", &simaai::neat::Tensor::route)
      .def_prop_ro("storage", [](const simaai::neat::Tensor& t) { return t.storage; })
      .def("is_dense", &simaai::neat::Tensor::is_dense)
      .def("is_composite", &simaai::neat::Tensor::is_composite)
      .def("has_axis_semantics", &simaai::neat::Tensor::has_axis_semantics)
      .def("axis_semantics_match_shape", &simaai::neat::Tensor::axis_semantics_match_shape)
      .def("is_contiguous", &simaai::neat::Tensor::is_contiguous)
      .def("has_plane", &simaai::neat::Tensor::has_plane, "role"_a)
      .def("contiguous", &simaai::neat::Tensor::contiguous)
      .def("clone", &simaai::neat::Tensor::clone)
      .def("cpu", &simaai::neat::Tensor::cpu)
      .def("cvu", &simaai::neat::Tensor::cvu)
      .def("mla", &simaai::neat::Tensor::mla, "force"_a = false)
      .def("to_cpu_if_needed", &simaai::neat::Tensor::to_cpu_if_needed)
      .def("validate", &tensor_validate)
      .def("dense_bytes_tight", &simaai::neat::Tensor::dense_bytes_tight)
      .def("copy_dense_bytes_tight", &tensor_dense_bytes)
      .def("copy_payload_bytes", &tensor_payload_bytes)
      .def("width", &simaai::neat::Tensor::width)
      .def("height", &simaai::neat::Tensor::height)
      .def("channels", &simaai::neat::Tensor::channels)
      .def("to_text", &simaai::neat::Tensor::to_text)
      .def("image_format", &tensor_image_format_value)
      .def("is_nv12", &simaai::neat::Tensor::is_nv12)
      .def("is_i420", &simaai::neat::Tensor::is_i420)
      .def("debug_string", &simaai::neat::Tensor::debug_string)
      .def("__repr__",
           [](const simaai::neat::Tensor& t) { return "Tensor(" + t.debug_string() + ")"; })
      .def_static(
          "from_text",
          [](const std::string& text) { return simaai::neat::Tensor::from_text(text); }, "text"_a)
      .def_static(
          "_from_dlpack_capsule",
          [](const nb::capsule& capsule, bool copy, std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format, TensorMemory memory,
             std::optional<ByteFormat> byte_format) {
            return tensor_from_dlpack_capsule(capsule, copy, layout, image_format, memory,
                                              byte_format);
          },
          "capsule"_a, "copy"_a = false, "layout"_a = nb::none(), "image_format"_a = nb::none(),
          "memory"_a = TensorMemory::EV74, "byte_format"_a = nb::none())
      .def(
          "__dlpack__",
          [](const simaai::neat::Tensor& t, nb::object stream) {
            if (!stream.is_none()) {
              throw std::runtime_error("stream argument is unsupported for CPU tensors");
            }
            return nb::steal<nb::object>(tensor_to_dlpack_capsule(t));
          },
          "stream"_a = nb::none())
      .def("__dlpack_device__", [](const simaai::neat::Tensor&) {
        return nb::make_tuple(static_cast<int>(simaai::neat::dlpack::DLDeviceType::kDLCPU), 0);
      });

  nb::class_<TensorConstraint>(m, "TensorConstraint")
      .def(nb::init<>())
      .def_rw("dtypes", &TensorConstraint::dtypes)
      .def_rw("rank", &TensorConstraint::rank)
      .def_rw("shape", &TensorConstraint::shape)
      .def_rw("device", &TensorConstraint::device)
      .def_rw("allowed_devices", &TensorConstraint::allowed_devices)
      .def_rw("preferred_device", &TensorConstraint::preferred_device)
      .def_rw("image_format", &TensorConstraint::image_format)
      .def_rw("required_segments", &TensorConstraint::required_segments)
      .def_rw("required_segment_names", &TensorConstraint::required_segment_names)
      .def_rw("allow_composite", &TensorConstraint::allow_composite)
      .def("matches", &TensorConstraint::matches, "tensor"_a);

  nb::class_<simaai::neat::BusMessage>(m, "BusMessage")
      .def(nb::init<>())
      .def_rw("type", &simaai::neat::BusMessage::type)
      .def_rw("src", &simaai::neat::BusMessage::src)
      .def_rw("detail", &simaai::neat::BusMessage::detail)
      .def_rw("wall_time_us", &simaai::neat::BusMessage::wall_time_us);

  nb::class_<simaai::neat::BoundaryFlowStats>(m, "BoundaryFlowStats")
      .def(nb::init<>())
      .def_rw("boundary_name", &simaai::neat::BoundaryFlowStats::boundary_name)
      .def_rw("after_node_index", &simaai::neat::BoundaryFlowStats::after_node_index)
      .def_rw("before_node_index", &simaai::neat::BoundaryFlowStats::before_node_index)
      .def_rw("in_buffers", &simaai::neat::BoundaryFlowStats::in_buffers)
      .def_rw("out_buffers", &simaai::neat::BoundaryFlowStats::out_buffers)
      .def_rw("last_in_pts_ns", &simaai::neat::BoundaryFlowStats::last_in_pts_ns)
      .def_rw("last_out_pts_ns", &simaai::neat::BoundaryFlowStats::last_out_pts_ns)
      .def_rw("last_in_wall_us", &simaai::neat::BoundaryFlowStats::last_in_wall_us)
      .def_rw("last_out_wall_us", &simaai::neat::BoundaryFlowStats::last_out_wall_us);

  nb::class_<simaai::neat::NodeReport>(m, "NodeReport")
      .def(nb::init<>())
      .def_rw("index", &simaai::neat::NodeReport::index)
      .def_rw("kind", &simaai::neat::NodeReport::kind)
      .def_rw("user_label", &simaai::neat::NodeReport::user_label)
      .def_rw("backend_fragment", &simaai::neat::NodeReport::backend_fragment)
      .def_rw("elements", &simaai::neat::NodeReport::elements);

  nb::class_<simaai::neat::BuildAdaptationAction>(m, "BuildAdaptationAction")
      .def(nb::init<>())
      .def_rw("target", &simaai::neat::BuildAdaptationAction::target)
      .def_rw("applied", &simaai::neat::BuildAdaptationAction::applied)
      .def_rw("detail", &simaai::neat::BuildAdaptationAction::detail)
      .def_rw("reason", &simaai::neat::BuildAdaptationAction::reason);

  nb::class_<simaai::neat::BuildAdaptationSummary>(m, "BuildAdaptationSummary")
      .def(nb::init<>())
      .def_rw("shape_policy", &simaai::neat::BuildAdaptationSummary::shape_policy)
      .def_rw("dynamic_capability", &simaai::neat::BuildAdaptationSummary::dynamic_capability)
      .def_rw("seed_width", &simaai::neat::BuildAdaptationSummary::seed_width)
      .def_rw("seed_height", &simaai::neat::BuildAdaptationSummary::seed_height)
      .def_rw("seed_depth", &simaai::neat::BuildAdaptationSummary::seed_depth)
      .def_rw("seed_width_origin", &simaai::neat::BuildAdaptationSummary::seed_width_origin)
      .def_rw("seed_height_origin", &simaai::neat::BuildAdaptationSummary::seed_height_origin)
      .def_rw("seed_depth_origin", &simaai::neat::BuildAdaptationSummary::seed_depth_origin)
      .def_rw("max_width", &simaai::neat::BuildAdaptationSummary::max_width)
      .def_rw("max_height", &simaai::neat::BuildAdaptationSummary::max_height)
      .def_rw("max_depth", &simaai::neat::BuildAdaptationSummary::max_depth)
      .def_rw("max_width_origin", &simaai::neat::BuildAdaptationSummary::max_width_origin)
      .def_rw("max_height_origin", &simaai::neat::BuildAdaptationSummary::max_height_origin)
      .def_rw("max_depth_origin", &simaai::neat::BuildAdaptationSummary::max_depth_origin)
      .def_rw("max_input_bytes_guard", &simaai::neat::BuildAdaptationSummary::max_input_bytes_guard)
      .def_rw("byte_guard_origin", &simaai::neat::BuildAdaptationSummary::byte_guard_origin)
      .def_rw("allow_ingress_cvu_format_renegotiation",
              &simaai::neat::BuildAdaptationSummary::allow_ingress_cvu_format_renegotiation)
      .def_rw("actions", &simaai::neat::BuildAdaptationSummary::actions);

  nb::class_<GraphReport>(m, "GraphReport")
      .def(nb::init<>())
      .def_rw("pipeline_string", &GraphReport::pipeline_string)
      .def_rw("error_code", &GraphReport::error_code)
      .def_rw("nodes", &GraphReport::nodes)
      .def_rw("bus", &GraphReport::bus)
      .def_rw("boundaries", &GraphReport::boundaries)
      .def_rw("caps_dump", &GraphReport::caps_dump)
      .def_rw("dot_paths", &GraphReport::dot_paths)
      .def_rw("repro_gst_launch", &GraphReport::repro_gst_launch)
      .def_rw("repro_env", &GraphReport::repro_env)
      .def_rw("repro_note", &GraphReport::repro_note)
      .def_rw("has_build_adaptation", &GraphReport::has_build_adaptation)
      .def_rw("build_adaptation", &GraphReport::build_adaptation)
      .def("to_json", &GraphReport::to_json);

  nb::class_<PullError>(m, "PullError")
      .def(nb::init<>())
      .def_rw("message", &PullError::message)
      .def_rw("code", &PullError::code)
      .def_rw("report", &PullError::report);

  nb::class_<Sample>(m, "Sample")
      .def(nb::init<>())
      .def_rw("kind", &Sample::kind)
      .def_rw("owned", &Sample::owned)
      .def_rw("tensor", &Sample::tensor)
      .def_rw("tensors", &Sample::tensors)
      .def_rw("fields", &Sample::fields)
      .def_rw("caps_string", &Sample::caps_string)
      .def_rw("payload_type", &Sample::payload_type)
      .def_rw("media_type", &Sample::media_type)
      .def_rw("payload_tag", &Sample::payload_tag)
      .def_rw("format", &Sample::format)
      .def_rw("frame_id", &Sample::frame_id)
      .def_rw("stream_id", &Sample::stream_id)
      .def_rw("stream_label", &Sample::stream_label)
      .def_rw("port_name", &Sample::port_name)
      .def_rw("output_index", &Sample::output_index)
      // Phase 1 (plan high-value slice S12): canonical output-identity route fields. output_index
      // above is the deprecated alias for logical_output_index; expose the canonical names.
      .def_rw("logical_output_index", &Sample::logical_output_index)
      .def_rw("memory_index", &Sample::memory_index)
      .def_rw("route_slot", &Sample::route_slot)
      .def_rw("segment_name", &Sample::segment_name)
      .def_rw("input_seq", &Sample::input_seq)
      .def_rw("orig_input_seq", &Sample::orig_input_seq)
      .def_rw("pts_ns", &Sample::pts_ns)
      .def_rw("dts_ns", &Sample::dts_ns)
      .def_rw("duration_ns", &Sample::duration_ns)
      // Phase 1 (plan slice S10): Pythonic sequence protocol over Bundle samples. The raw C++
      // operator[] returns *this for any index on a non-Bundle sample (no bounds error), so
      // __getitem__ is explicitly bounds-checked (and supports negative indices). front()/back()/
      // reserve() are intentionally not exposed — use s[0]/s[-1].
      .def("__len__", [](const Sample& s) { return s.size(); })
      .def("__bool__", [](const Sample& s) { return !s.empty(); })
      .def(
          "__getitem__",
          [](Sample& s, Py_ssize_t i) -> Sample& {
            const auto n = static_cast<Py_ssize_t>(s.size());
            if (i < 0) {
              i += n;
            }
            if (i < 0 || i >= n) {
              throw nb::index_error("Sample index out of range");
            }
            return s[static_cast<std::size_t>(i)];
          },
          "index"_a, nb::rv_policy::reference_internal)
      .def(
          "__iter__",
          [](Sample& s) {
            return nb::make_iterator(nb::type<Sample>(), "SampleIterator", s.begin(), s.end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "append", [](Sample& s, Sample child) { s.push_back(std::move(child)); }, "sample"_a)
      .def("to_text", &sample_to_text_for_python);

  m.def("make_tensor_sample", &simaai::neat::make_tensor_sample, "port_name"_a, "tensor"_a);
  m.def("make_text_sample", &make_text_sample_for_python, "port_name"_a, "text"_a);
  // Phase 6 (plan slice): encoded-media sample construction (parity with make_tensor_sample). The
  // friendly Sample.from_encoded(...) wrapper is added in _wrappers.py. caps_to_codec is deferred.
  m.def(
      "make_encoded_sample",
      [](nb::bytes data, const std::string& caps_string, int64_t pts_ns, int64_t dts_ns,
         int64_t duration_ns) {
        const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
        std::vector<uint8_t> buf(p, p + data.size());
        return simaai::neat::make_encoded_sample(std::move(buf), caps_string, pts_ns, dts_ns,
                                                 duration_ns);
      },
      "data"_a, "caps_string"_a, "pts_ns"_a = -1, "dts_ns"_a = -1, "duration_ns"_a = -1);

  m.def(
      "decode_bbox",
      [](const TensorList& bbox_tensors, std::optional<std::pair<int, int>> clamp_to,
         std::optional<int> top_k, bool strict) {
        const int w = clamp_to ? clamp_to->first : 0;
        const int h = clamp_to ? clamp_to->second : 0;
        const int k = top_k.value_or(0);
        try {
          return simaai::neat::decode_bbox(bbox_tensors, w, h, k, strict);
        } catch (const std::runtime_error& e) {
          const std::string msg = e.what();
          // Wrong-type inputs surface as TypeError; malformed-payload (strict)
          // and other runtime failures stay RuntimeError.
          if (msg.find("is not a BBOX tensor") != std::string::npos ||
              msg.find("expected 'BBOX'") != std::string::npos ||
              msg.find("expected a BBOX-family") != std::string::npos ||
              msg.find("format mismatch") != std::string::npos) {
            throw nb::type_error(e.what());
          }
          throw;
        }
      },
      "Decode BBOX-format tensors into decoded-boxes tensors, positional 1:1.\n\n"
      "Pass a model's run output (a list of Tensors); each BBOX-format tensor is\n"
      "decoded into a float32 tensor of shape [num_detections, 6] with columns\n"
      "(x1, y1, x2, y2, score, class_id). The returned list has the same length as\n"
      "the input.\n\n"
      "Args:\n"
      "  bbox_tensors: list[Tensor] of BBOX-format tensors (e.g. model.run(...)).\n"
      "  clamp_to:     Optional (width, height) - clamp coordinates to that rectangle.\n"
      "  top_k:        Optional cap on detections per tensor.\n"
      "  strict:       When True, raise on malformed buffers instead of best-effort.\n\n"
      "Returns:\n"
      "  list[Tensor] - one [num_detections, 6] float32 tensor per input.\n\n"
      "Raises:\n"
      "  TypeError: an input tensor is not BBOX-format.\n"
      "  RuntimeError: strict=True and a payload is malformed.",
      "bbox_tensors"_a, nb::kw_only(), "clamp_to"_a = nb::none(), "top_k"_a = nb::none(),
      "strict"_a = false);

  nb::class_<simaai::neat::PoseDecodeTensors>(m, "PoseDecodeTensors")
      .def(nb::init<>())
      .def_rw("boxes", &simaai::neat::PoseDecodeTensors::boxes)
      .def_rw("keypoints", &simaai::neat::PoseDecodeTensors::keypoints);

  nb::class_<simaai::neat::SegmentationDecodeTensors>(m, "SegmentationDecodeTensors")
      .def(nb::init<>())
      .def_rw("boxes", &simaai::neat::SegmentationDecodeTensors::boxes)
      .def_rw("masks", &simaai::neat::SegmentationDecodeTensors::masks);

  m.def(
      "decode_pose",
      [](const TensorList& pose_tensors, std::optional<std::pair<int, int>> clamp_to,
         std::optional<int> top_k, bool strict) {
        const int w = clamp_to ? clamp_to->first : 0;
        const int h = clamp_to ? clamp_to->second : 0;
        const int k = top_k.value_or(0);
        try {
          return simaai::neat::decode_pose(pose_tensors, w, h, k, strict);
        } catch (const std::runtime_error& e) {
          const std::string msg = e.what();
          if (detection_decode_type_error_message(msg)) {
            throw nb::type_error(e.what());
          }
          throw;
        }
      },
      "Decode BoxDecode pose tensors into boxes and keypoints tensors, positional 1:1.\n\n"
      "Each result has `boxes`, a float32 tensor of shape [N, 6], and `keypoints`,\n"
      "a float32 tensor of shape [N, 17, 3] with columns (x, y, visibility).\n\n"
      "Args:\n"
      "  pose_tensors: list[Tensor] of BoxDecode pose-format tensors.\n"
      "  clamp_to:     Optional (width, height) - clamp box coordinates to that rectangle.\n"
      "  top_k:        Optional cap on detections per tensor.\n"
      "  strict:       When True, raise on malformed buffers instead of best-effort.\n\n"
      "Returns:\n"
      "  list[PoseDecodeTensors] - one result per input tensor.\n\n"
      "Raises:\n"
      "  TypeError: an input tensor is not pose/BBOX-compatible.\n"
      "  RuntimeError: strict=True and a payload is malformed.",
      "pose_tensors"_a, nb::kw_only(), "clamp_to"_a = nb::none(), "top_k"_a = nb::none(),
      "strict"_a = false);

  m.def(
      "decode_segmentation",
      [](const TensorList& segmentation_tensors, std::optional<std::pair<int, int>> clamp_to,
         std::optional<int> top_k, bool strict) {
        const int w = clamp_to ? clamp_to->first : 0;
        const int h = clamp_to ? clamp_to->second : 0;
        const int k = top_k.value_or(0);
        try {
          return simaai::neat::decode_segmentation(segmentation_tensors, w, h, k, strict);
        } catch (const std::runtime_error& e) {
          const std::string msg = e.what();
          if (detection_decode_type_error_message(msg)) {
            throw nb::type_error(e.what());
          }
          throw;
        }
      },
      "Decode BoxDecode segmentation tensors into boxes and masks tensors, positional 1:1.\n\n"
      "Each result has `boxes`, a float32 tensor of shape [N, 6], and `masks`,\n"
      "a uint8 tensor of shape [N, 160, 160].\n\n"
      "Args:\n"
      "  segmentation_tensors: list[Tensor] of BoxDecode segmentation-format tensors.\n"
      "  clamp_to:             Optional (width, height) - clamp box coordinates to that "
      "rectangle.\n"
      "  top_k:                Optional cap on detections per tensor.\n"
      "  strict:               When True, raise on malformed buffers instead of best-effort.\n\n"
      "Returns:\n"
      "  list[SegmentationDecodeTensors] - one result per input tensor.\n\n"
      "Raises:\n"
      "  TypeError: an input tensor is not segmentation/BBOX-compatible.\n"
      "  RuntimeError: strict=True and a payload is malformed.",
      "segmentation_tensors"_a, nb::kw_only(), "clamp_to"_a = nb::none(), "top_k"_a = nb::none(),
      "strict"_a = false);

  nb::enum_<simaai::neat::genai::GenAITask>(m, "GenAITask")
      .value("VisionLanguage", simaai::neat::genai::GenAITask::VisionLanguage)
      .value("ASR", simaai::neat::genai::GenAITask::ASR);

  nb::class_<simaai::neat::genai::ImageList>(m, "ImageList")
      .def(nb::init<>())
      .def(nb::init<std::vector<Tensor>>(), "images"_a)
      .def(
          "__init__",
          [](simaai::neat::genai::ImageList* self, const nb::object& images) {
            new (self) simaai::neat::genai::ImageList(genai_image_tensors_from_python(images));
          },
          "images"_a)
      .def("empty", &simaai::neat::genai::ImageList::empty)
      .def("size", &simaai::neat::genai::ImageList::size)
      .def_prop_rw(
          "tensors", [](const simaai::neat::genai::ImageList& images) { return images.tensors(); },
          [](simaai::neat::genai::ImageList& images, const nb::object& tensors) {
            images = genai_image_tensors_from_python(tensors);
          });

  nb::class_<simaai::neat::genai::ChatMessage>(m, "ChatMessage")
      .def(nb::init<>())
      .def_rw("role", &simaai::neat::genai::ChatMessage::role)
      .def_rw("content", &simaai::neat::genai::ChatMessage::content)
      .def_prop_rw(
          "images",
          [](const simaai::neat::genai::ChatMessage& message) { return message.images.tensors(); },
          [](simaai::neat::genai::ChatMessage& message, const nb::object& images) {
            message.images = genai_image_tensors_from_python(images);
          })
      .def_rw("use_cached_images", &simaai::neat::genai::ChatMessage::use_cached_images)
      .def_prop_rw(
          "tool_calls",
          [](const simaai::neat::genai::ChatMessage& message) {
            return genai_json_to_python(message.tool_calls);
          },
          [](simaai::neat::genai::ChatMessage& message, nb::handle value) {
            message.tool_calls = python_to_genai_json(value);
          })
      .def_rw("tool_call_id", &simaai::neat::genai::ChatMessage::tool_call_id)
      .def_rw("name", &simaai::neat::genai::ChatMessage::name);

  nb::class_<simaai::neat::genai::GenerationMetrics>(m, "GenerationMetrics")
      .def(nb::init<>())
      .def_rw("generated_tokens", &simaai::neat::genai::GenerationMetrics::generated_tokens)
      .def_rw("time_to_first_token_s",
              &simaai::neat::genai::GenerationMetrics::time_to_first_token_s)
      .def_rw("tokens_per_second", &simaai::neat::genai::GenerationMetrics::tokens_per_second);

  nb::class_<simaai::neat::genai::GenerationRequest>(m, "GenerationRequest")
      .def(nb::init<>())
      .def_rw("prompt", &simaai::neat::genai::GenerationRequest::prompt)
      .def_rw("system_prompt", &simaai::neat::genai::GenerationRequest::system_prompt)
      .def_rw("messages", &simaai::neat::genai::GenerationRequest::messages)
      .def_prop_rw(
          "images",
          [](const simaai::neat::genai::GenerationRequest& request) {
            return request.images.tensors();
          },
          [](simaai::neat::genai::GenerationRequest& request, const nb::object& images) {
            request.images = genai_image_tensors_from_python(images);
          })
      .def_rw("use_cached_images", &simaai::neat::genai::GenerationRequest::use_cached_images)
      .def_rw("audio", &simaai::neat::genai::GenerationRequest::audio)
      .def_rw("audio_file", &simaai::neat::genai::GenerationRequest::audio_file)
      .def_rw("language", &simaai::neat::genai::GenerationRequest::language)
      .def_rw("max_new_tokens", &simaai::neat::genai::GenerationRequest::max_new_tokens)
      .def_prop_rw(
          "tools",
          [](const simaai::neat::genai::GenerationRequest& request) {
            return genai_json_to_python(request.tools);
          },
          [](simaai::neat::genai::GenerationRequest& request, nb::handle value) {
            request.tools = python_to_genai_json(value);
          })
      .def_prop_rw(
          "tool_choice",
          [](const simaai::neat::genai::GenerationRequest& request) {
            return genai_json_to_python(request.tool_choice);
          },
          [](simaai::neat::genai::GenerationRequest& request, nb::handle value) {
            request.tool_choice = python_to_genai_json(value);
          });

  nb::class_<simaai::neat::genai::GenerationResult>(m, "GenerationResult")
      .def(nb::init<>())
      .def_rw("text", &simaai::neat::genai::GenerationResult::text)
      .def_rw("metrics", &simaai::neat::genai::GenerationResult::metrics)
      .def_rw("finish_reason", &simaai::neat::genai::GenerationResult::finish_reason)
      .def_prop_rw(
          "tool_calls",
          [](const simaai::neat::genai::GenerationResult& result) {
            return genai_json_to_python(result.tool_calls);
          },
          [](simaai::neat::genai::GenerationResult& result, nb::handle value) {
            result.tool_calls = python_to_genai_json(value);
          });

  nb::class_<simaai::neat::genai::TokenSample>(m, "TokenSample")
      .def(nb::init<>())
      .def_rw("text", &simaai::neat::genai::TokenSample::text)
      .def_rw("metrics", &simaai::neat::genai::TokenSample::metrics)
      .def_rw("is_final", &simaai::neat::genai::TokenSample::is_final)
      .def_rw("finish_reason", &simaai::neat::genai::TokenSample::finish_reason)
      .def_prop_rw(
          "tool_calls",
          [](const simaai::neat::genai::TokenSample& sample) {
            return genai_json_to_python(sample.tool_calls);
          },
          [](simaai::neat::genai::TokenSample& sample, nb::handle value) {
            sample.tool_calls = python_to_genai_json(value);
          });

  nb::class_<simaai::neat::genai::GenerationStream>(m, "GenerationStream")
      .def("next", &simaai::neat::genai::GenerationStream::next,
           nb::call_guard<nb::gil_scoped_release>())
      .def("cancel", &simaai::neat::genai::GenerationStream::cancel,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__iter__",
          [](simaai::neat::genai::GenerationStream& stream)
              -> simaai::neat::genai::GenerationStream& { return stream; },
          nb::rv_policy::reference_internal)
      .def("__next__", [](simaai::neat::genai::GenerationStream& stream) {
        std::optional<simaai::neat::genai::TokenSample> token;
        {
          nb::gil_scoped_release release;
          token = stream.next();
        }
        if (!token.has_value()) {
          throw nb::stop_iteration();
        }
        return std::move(*token);
      });

  nb::class_<simaai::neat::genai::VisionLanguageModel>(m, "VisionLanguageModel")
      .def(nb::init<std::filesystem::path>(), "model_dir"_a)
      .def("accepts_image", &simaai::neat::genai::VisionLanguageModel::accepts_image)
      .def("model_id", &simaai::neat::genai::VisionLanguageModel::model_id)
      .def("cached_image_count", &simaai::neat::genai::VisionLanguageModel::cached_image_count)
      .def(
          "encode",
          [](simaai::neat::genai::VisionLanguageModel& model, const nb::object& images) {
            auto tensors = genai_image_tensors_from_python(images);
            nb::gil_scoped_release release;
            return model.encode(tensors);
          },
          "images"_a)
      .def("run", &simaai::neat::genai::VisionLanguageModel::run, "request"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def("stream", &simaai::neat::genai::VisionLanguageModel::stream, "request"_a,
           nb::call_guard<nb::gil_scoped_release>());

  nb::class_<simaai::neat::genai::ASRModel>(m, "ASRModel")
      .def(nb::init<std::filesystem::path>(), "model_dir"_a)
      .def("accepts_audio", &simaai::neat::genai::ASRModel::accepts_audio)
      .def("model_id", &simaai::neat::genai::ASRModel::model_id)
      .def("run", &simaai::neat::genai::ASRModel::run, "request"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def("stream", &simaai::neat::genai::ASRModel::stream, "request"_a,
           nb::call_guard<nb::gil_scoped_release>());

  nb::class_<simaai::neat::genai::GenAIModel>(m, "GenAIModel")
      .def(nb::init<std::filesystem::path>(), "model_dir"_a)
      .def("task", &simaai::neat::genai::GenAIModel::task)
      .def("accepts_text", &simaai::neat::genai::GenAIModel::accepts_text)
      .def("accepts_image", &simaai::neat::genai::GenAIModel::accepts_image)
      .def("accepts_audio", &simaai::neat::genai::GenAIModel::accepts_audio)
      .def("model_id", &simaai::neat::genai::GenAIModel::model_id)
      .def("run", &simaai::neat::genai::GenAIModel::run, "request"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def("stream", &simaai::neat::genai::GenAIModel::stream, "request"_a,
           nb::call_guard<nb::gil_scoped_release>());

  nb::class_<simaai::neat::genai::GenAIServerOptions>(m, "GenAIServerOptions")
      .def(nb::init<>())
      .def_rw("host", &simaai::neat::genai::GenAIServerOptions::host)
      .def_rw("port", &simaai::neat::genai::GenAIServerOptions::port);

  nb::class_<simaai::neat::genai::GenAIServer>(m, "GenAIServer")
      .def(nb::init<simaai::neat::genai::GenAIServerOptions>(),
           "options"_a = simaai::neat::genai::GenAIServerOptions{})
      .def(
          "add_model",
          [](simaai::neat::genai::GenAIServer& server, const std::filesystem::path& model_dir) {
            return server.add_model(model_dir);
          },
          "model_dir"_a)
      .def(
          "add_model",
          [](simaai::neat::genai::GenAIServer& server, const std::filesystem::path& model_dir,
             const std::string& served_name) { return server.add_model(model_dir, served_name); },
          "model_dir"_a, "served_name"_a)
      .def("remove_model", &simaai::neat::genai::GenAIServer::remove_model, "served_name"_a)
      .def("model_names", &simaai::neat::genai::GenAIServer::model_names)
      .def("start", &simaai::neat::genai::GenAIServer::start,
           nb::call_guard<nb::gil_scoped_release>())
      .def("stop", &simaai::neat::genai::GenAIServer::stop,
           nb::call_guard<nb::gil_scoped_release>());

  nb::module_ genai_mod = m.def_submodule("genai", "Generative AI aliases and helpers");
  genai_mod.attr("GenAITask") = m.attr("GenAITask");
  genai_mod.attr("ImageList") = m.attr("ImageList");
  genai_mod.attr("ChatMessage") = m.attr("ChatMessage");
  genai_mod.attr("GenerationMetrics") = m.attr("GenerationMetrics");
  genai_mod.attr("GenerationRequest") = m.attr("GenerationRequest");
  genai_mod.attr("GenerationResult") = m.attr("GenerationResult");
  genai_mod.attr("TokenSample") = m.attr("TokenSample");
  genai_mod.attr("GenerationStream") = m.attr("GenerationStream");
  genai_mod.attr("VisionLanguageModel") = m.attr("VisionLanguageModel");
  genai_mod.attr("ASRModel") = m.attr("ASRModel");
  genai_mod.attr("GenAIModel") = m.attr("GenAIModel");
  genai_mod.attr("GenAIServerOptions") = m.attr("GenAIServerOptions");
  genai_mod.attr("GenAIServer") = m.attr("GenAIServer");

  nb::class_<simaai::neat::genai::VisionLanguageOptions>(genai_mod, "VisionLanguageOptions")
      .def(nb::init<>())
      .def_rw("system_prompt", &simaai::neat::genai::VisionLanguageOptions::system_prompt)
      .def_rw("max_new_tokens", &simaai::neat::genai::VisionLanguageOptions::max_new_tokens)
      .def_rw("streaming", &simaai::neat::genai::VisionLanguageOptions::streaming)
      .def_rw("encode_images_on_input",
              &simaai::neat::genai::VisionLanguageOptions::encode_images_on_input);
  nb::class_<simaai::neat::genai::SpeechTranscriberOptions>(genai_mod, "SpeechTranscriberOptions")
      .def(nb::init<>())
      .def_rw("language", &simaai::neat::genai::SpeechTranscriberOptions::language)
      .def_rw("streaming", &simaai::neat::genai::SpeechTranscriberOptions::streaming);

  nb::module_ genai_graphs_mod = genai_mod.def_submodule("graphs", "GenAI public Graph fragments");
  genai_graphs_mod.def(
      "vision_language",
      [](std::shared_ptr<simaai::neat::genai::VisionLanguageModel> model,
         simaai::neat::genai::VisionLanguageOptions options, const std::string& label) {
        return simaai::neat::genai::graphs::VisionLanguage(std::move(model), std::move(options),
                                                           label);
      },
      "model"_a, "options"_a = simaai::neat::genai::VisionLanguageOptions{},
      "label"_a = "vision_language");
  genai_graphs_mod.def(
      "speech_transcriber",
      [](std::shared_ptr<simaai::neat::genai::ASRModel> model,
         simaai::neat::genai::SpeechTranscriberOptions options, const std::string& label) {
        return simaai::neat::genai::graphs::SpeechTranscriber(std::move(model), std::move(options),
                                                              label);
      },
      "model"_a, "options"_a = simaai::neat::genai::SpeechTranscriberOptions{},
      "label"_a = "speech_transcriber");

  nb::class_<simaai::neat::RtspServerOptions>(m, "RtspServerOptions")
      .def(nb::init<>())
      .def_rw("mount", &simaai::neat::RtspServerOptions::mount)
      .def_rw("port", &simaai::neat::RtspServerOptions::port)
      .def_rw("rtp_port_base", &simaai::neat::RtspServerOptions::rtp_port_base)
      .def_rw("rtp_port_count", &simaai::neat::RtspServerOptions::rtp_port_count);

  nb::class_<ValidateOptions>(m, "ValidateOptions")
      .def(nb::init<>())
      .def_rw("parse_launch", &ValidateOptions::parse_launch)
      .def_rw("enforce_names", &ValidateOptions::enforce_names);

  // Phase 1 (plan slice S2/S3/S4): jargon-free execution surface. Bound as the ONLY execution
  // surface on GraphOptions (which previously exposed no execution knobs) — net power gain, no raw
  // legacy fields. Targets use the resolver's native tokens (AUTO/A65/EV74) per S3.
  nb::class_<simaai::neat::PreparedRunnerOptions>(m, "PreparedRunnerOptions")
      .def(nb::init<>())
      .def_rw("mode", &simaai::neat::PreparedRunnerOptions::mode)
      .def_rw("ring_depth", &simaai::neat::PreparedRunnerOptions::ring_depth)
      .def_rw("profile", &simaai::neat::PreparedRunnerOptions::profile)
      .def_rw("dequant_flags", &simaai::neat::PreparedRunnerOptions::dequant_flags);
  nb::class_<simaai::neat::AdvancedExecutionOptions>(m, "AdvancedExecutionOptions")
      .def(nb::init<>())
      .def_rw("preprocess_target", &simaai::neat::AdvancedExecutionOptions::preprocess_target)
      .def_rw("postprocess_target", &simaai::neat::AdvancedExecutionOptions::postprocess_target)
      .def_rw("preprocess_async", &simaai::neat::AdvancedExecutionOptions::preprocess_async)
      .def_rw("inference_async", &simaai::neat::AdvancedExecutionOptions::inference_async)
      .def_rw("inference_output_buffers",
              &simaai::neat::AdvancedExecutionOptions::inference_output_buffers)
      .def_rw("defer_output_cache_sync",
              &simaai::neat::AdvancedExecutionOptions::defer_output_cache_sync)
      .def_rw("prepared_runner", &simaai::neat::AdvancedExecutionOptions::prepared_runner)
      .def_rw("internal_queue_depth",
              &simaai::neat::AdvancedExecutionOptions::internal_queue_depth);

  nb::enum_<GraphLinkPolicy>(m, "GraphLinkPolicy")
      .value("Default", GraphLinkPolicy::Default)
      .value("RealtimeLatestByStream", GraphLinkPolicy::RealtimeLatestByStream);

  nb::class_<GraphLinkOptions>(m, "GraphLinkOptions")
      .def(nb::init<>())
      .def_rw("policy", &GraphLinkOptions::policy)
      .def_rw("queue_depth", &GraphLinkOptions::queue_depth)
      .def_rw("stream_id", &GraphLinkOptions::stream_id);

  nb::class_<GraphOptions>(m, "GraphOptions")
      .def(nb::init<>())
      .def_rw("callback_timeout_ms", &GraphOptions::callback_timeout_ms)
      .def_rw("element_name_prefix", &GraphOptions::element_name_prefix)
      .def_rw("element_name_suffix", &GraphOptions::element_name_suffix)
      .def_rw("verbose", &GraphOptions::verbose)
      .def_rw("advanced_execution", &GraphOptions::advanced_execution);

  nb::class_<simaai::neat::OutputTensorOptions>(m, "OutputTensorOptions")
      .def(nb::init<>())
      .def_prop_rw("format", format_enum_getter(&simaai::neat::OutputTensorOptions::format),
                   format_enum_setter(&simaai::neat::OutputTensorOptions::format))
      .def_rw("dtype", &simaai::neat::OutputTensorOptions::dtype)
      .def_rw("target_width", &simaai::neat::OutputTensorOptions::target_width)
      .def_rw("target_height", &simaai::neat::OutputTensorOptions::target_height)
      .def_rw("target_fps", &simaai::neat::OutputTensorOptions::target_fps);

  nb::class_<RunAdvancedOptions>(m, "RunAdvancedOptions")
      .def(nb::init<>())
      .def_rw("copy_input", &RunAdvancedOptions::copy_input)
      .def_rw("max_input_bytes", &RunAdvancedOptions::max_input_bytes)
      .def_rw("sync_num_buffers_override", &RunAdvancedOptions::sync_num_buffers_override)
      .def_rw("prepare_output_cpu_visible", &RunAdvancedOptions::prepare_output_cpu_visible);

  nb::class_<PowerRailConfig>(m, "PowerRailConfig")
      .def(nb::init<>())
      .def_rw("name", &PowerRailConfig::name)
      .def_rw("i2c_bus", &PowerRailConfig::i2c_bus)
      .def_rw("i2c_addr", &PowerRailConfig::i2c_addr)
      .def_rw("page", &PowerRailConfig::page)
      .def_rw("vout_exponent", &PowerRailConfig::vout_exponent)
      .def_rw("iout_exponent", &PowerRailConfig::iout_exponent)
      .def_rw("pout_exponent", &PowerRailConfig::pout_exponent);

  nb::class_<PowerMonitorOptions>(m, "PowerMonitorOptions")
      .def(nb::init<>())
      .def_rw("enabled", &PowerMonitorOptions::enabled)
      .def_rw("sample_interval_ms", &PowerMonitorOptions::sample_interval_ms)
      .def_rw("profile", &PowerMonitorOptions::profile)
      .def_rw("rails", &PowerMonitorOptions::rails);

  nb::class_<PowerFieldSummary>(m, "PowerFieldSummary")
      .def(nb::init<>())
      .def_rw("samples", &PowerFieldSummary::samples)
      .def_rw("errors", &PowerFieldSummary::errors)
      .def_rw("avg", &PowerFieldSummary::avg)
      .def_rw("min", &PowerFieldSummary::min)
      .def_rw("max", &PowerFieldSummary::max);

  nb::class_<PowerRailSummary>(m, "PowerRailSummary")
      .def(nb::init<>())
      .def_rw("config", &PowerRailSummary::config)
      .def_rw("voltage_v", &PowerRailSummary::voltage_v)
      .def_rw("current_a", &PowerRailSummary::current_a)
      .def_rw("power_w", &PowerRailSummary::power_w);

  nb::class_<PowerSummary>(m, "PowerSummary")
      .def(nb::init<>())
      .def_rw("enabled", &PowerSummary::enabled)
      .def_rw("samples", &PowerSummary::samples)
      .def_rw("duration_seconds", &PowerSummary::duration_seconds)
      .def_rw("total_avg_watts", &PowerSummary::total_avg_watts)
      .def_rw("total_min_watts", &PowerSummary::total_min_watts)
      .def_rw("total_max_watts", &PowerSummary::total_max_watts)
      .def_rw("energy_joules", &PowerSummary::energy_joules)
      .def_rw("rails", &PowerSummary::rails);

  nb::class_<RunAutoExportOptions> run_auto_export_options(m, "RunAutoExportOptions");
  run_auto_export_options.def(nb::init<>())
      .def_rw("path", &RunAutoExportOptions::path)
      .def_rw("label", &RunAutoExportOptions::label)
      .def_rw("include_metrics", &RunAutoExportOptions::include_metrics)
      .def_rw("include_power", &RunAutoExportOptions::include_power)
      .def_rw("include_node_metrics", &RunAutoExportOptions::include_node_metrics)
      .def_rw("include_plugin_metrics", &RunAutoExportOptions::include_plugin_metrics)
      .def_rw("include_empty_node_metrics", &RunAutoExportOptions::include_empty_node_metrics)
      .def_rw("indent", &RunAutoExportOptions::indent);
  m.attr("GraphRunAutoExportOptions") = run_auto_export_options;

  nb::class_<RunExportOptions> run_export_options(m, "RunExportOptions");
  run_export_options.def(nb::init<>())
      .def_rw("label", &RunExportOptions::label)
      .def_rw("include_metrics", &RunExportOptions::include_metrics)
      .def_rw("include_power", &RunExportOptions::include_power)
      .def_rw("include_node_metrics", &RunExportOptions::include_node_metrics)
      .def_rw("include_plugin_metrics", &RunExportOptions::include_plugin_metrics)
      .def_rw("include_empty_node_metrics", &RunExportOptions::include_empty_node_metrics)
      .def_rw("indent", &RunExportOptions::indent)
      .def_rw("metadata", &RunExportOptions::metadata);
  m.attr("GraphRunExportOptions") = run_export_options;

  nb::class_<simaai::neat::InputDropInfo>(m, "InputDropInfo")
      .def(nb::init<>())
      .def_rw("kind", &simaai::neat::InputDropInfo::kind)
      .def_rw("media_type", &simaai::neat::InputDropInfo::media_type)
      .def_rw("format", &simaai::neat::InputDropInfo::format)
      .def_rw("width", &simaai::neat::InputDropInfo::width)
      .def_rw("height", &simaai::neat::InputDropInfo::height)
      .def_rw("depth", &simaai::neat::InputDropInfo::depth)
      .def_rw("frame_id", &simaai::neat::InputDropInfo::frame_id)
      .def_rw("stream_id", &simaai::neat::InputDropInfo::stream_id)
      .def_rw("port_name", &simaai::neat::InputDropInfo::port_name)
      .def_rw("reason", &simaai::neat::InputDropInfo::reason);

  nb::class_<RunOptions>(m, "RunOptions")
      .def(nb::init<>())
      .def_rw("preset", &RunOptions::preset)
      .def_rw("queue_depth", &RunOptions::queue_depth)
      .def_rw("overflow_policy", &RunOptions::overflow_policy)
      .def_rw("output_memory", &RunOptions::output_memory)
      .def_rw("input_timeout_ms", &RunOptions::input_timeout_ms)
      .def_rw("startup_preflight", &RunOptions::startup_preflight)
      .def_rw("advanced", &RunOptions::advanced)
      .def_rw("power_monitor", &RunOptions::power_monitor)
      .def_rw("run_export", &RunOptions::run_export)
      .def_rw("graph_run_export", &RunOptions::graph_run_export)
      .def_rw("on_input_drop", &RunOptions::on_input_drop)
      .def("enable_board_power", &RunOptions::enable_board_power, "sample_interval_ms"_a = 100,
           nb::rv_policy::reference_internal)
      .def("enable_modalix_som_power", &RunOptions::enable_modalix_som_power,
           "sample_interval_ms"_a = 100, nb::rv_policy::reference_internal)
      .def("enable_modalix_dvt_power", &RunOptions::enable_modalix_dvt_power,
           "sample_interval_ms"_a = 100, nb::rv_policy::reference_internal)
      .def("disable_power_monitor", &RunOptions::disable_power_monitor,
           nb::rv_policy::reference_internal);

  m.def(
      "board_power_monitor_options",
      [](int sample_interval_ms, PowerMonitorProfile profile) {
        return simaai::neat::board_power_monitor_options(sample_interval_ms, profile);
      },
      "sample_interval_ms"_a = 100, "profile"_a = PowerMonitorProfile::Auto);
  m.def("modalix_som_power_monitor_options", &simaai::neat::modalix_som_power_monitor_options,
        "sample_interval_ms"_a = 100);
  m.def("modalix_dvt_power_monitor_options", &simaai::neat::modalix_dvt_power_monitor_options,
        "sample_interval_ms"_a = 100);
  m.def("power_monitor_profile_name", &simaai::neat::power_monitor_profile_name, "profile"_a);

  nb::class_<NodeLatencySummary>(m, "NodeLatencySummary")
      .def(nb::init<>())
      .def_rw("samples", &NodeLatencySummary::samples)
      .def_rw("total_ms", &NodeLatencySummary::total_ms)
      .def_rw("avg_ms", &NodeLatencySummary::avg_ms)
      .def_rw("min_ms", &NodeLatencySummary::min_ms)
      .def_rw("max_ms", &NodeLatencySummary::max_ms)
      .def_rw("min_max_available", &NodeLatencySummary::min_max_available);

  nb::class_<GraphElementMetrics>(m, "GraphElementMetrics")
      .def(nb::init<>())
      .def_rw("name", &GraphElementMetrics::name)
      .def_rw("latency", &GraphElementMetrics::latency);

  nb::class_<GraphNodeMetrics>(m, "GraphNodeMetrics")
      .def(nb::init<>())
      .def_rw("pipeline_segment_id", &GraphNodeMetrics::pipeline_segment_id)
      .def_rw("runtime_node_id", &GraphNodeMetrics::runtime_node_id)
      .def_rw("node_id", &GraphNodeMetrics::node_id)
      .def_rw("public_node_ids", &GraphNodeMetrics::public_node_ids)
      .def_rw("kind", &GraphNodeMetrics::kind)
      .def_rw("label", &GraphNodeMetrics::label)
      .def_rw("element_names", &GraphNodeMetrics::element_names)
      .def_rw("elements", &GraphNodeMetrics::elements)
      .def_rw("latency", &GraphNodeMetrics::latency);

  nb::class_<MeasureOptions>(m, "MeasureOptions")
      .def(nb::init<>())
      .def_rw("duration_ms", &MeasureOptions::duration_ms)
      .def_rw("warmup_ms", &MeasureOptions::warmup_ms)
      .def_rw("timeout_ms", &MeasureOptions::timeout_ms)
      .def_rw("include_plugin_latency", &MeasureOptions::include_plugin_latency)
      .def_rw("plugin_latency_source", &MeasureOptions::plugin_latency_source)
      .def_rw("include_edge_latency", &MeasureOptions::include_edge_latency)
      .def_rw("include_message_latency", &MeasureOptions::include_message_latency)
      .def_rw("message_latency_source", &MeasureOptions::message_latency_source)
      .def_rw("retain_metrics_trace", &MeasureOptions::retain_metrics_trace)
      .def_rw("metrics_trace_dir", &MeasureOptions::metrics_trace_dir)
      .def_rw("include_power", &MeasureOptions::include_power)
      .def_rw("title", &MeasureOptions::title)
      .def_rw("model", &MeasureOptions::model)
      .def_rw("input", &MeasureOptions::input)
      .def_rw("placement", &MeasureOptions::placement)
      .def_rw("logical_batch_size", &MeasureOptions::logical_batch_size);

  nb::class_<MeasureLatencyStats>(m, "MeasureLatencyStats")
      .def(nb::init<>())
      .def_rw("count", &MeasureLatencyStats::count)
      .def_rw("avg_ms", &MeasureLatencyStats::avg_ms)
      .def_rw("p50_ms", &MeasureLatencyStats::p50_ms)
      .def_rw("p90_ms", &MeasureLatencyStats::p90_ms)
      .def_rw("p95_ms", &MeasureLatencyStats::p95_ms)
      .def_rw("p99_ms", &MeasureLatencyStats::p99_ms)
      .def_rw("max_ms", &MeasureLatencyStats::max_ms);

  nb::class_<MeasurePluginLatency>(m, "MeasurePluginLatency")
      .def(nb::init<>())
      .def_rw("name", &MeasurePluginLatency::name)
      .def_rw("backend", &MeasurePluginLatency::backend)
      .def_rw("phase", &MeasurePluginLatency::phase)
      .def_rw("kernel_name", &MeasurePluginLatency::kernel_name)
      .def_rw("stage_name", &MeasurePluginLatency::stage_name)
      .def_rw("physical_input_index", &MeasurePluginLatency::physical_input_index)
      .def_rw("output_slot", &MeasurePluginLatency::output_slot)
      .def_rw("run_id_hash", &MeasurePluginLatency::run_id_hash)
      .def_rw("pipeline_segment_id", &MeasurePluginLatency::pipeline_segment_id)
      .def_rw("runtime_node_id", &MeasurePluginLatency::runtime_node_id)
      .def_rw("public_node_id", &MeasurePluginLatency::public_node_id)
      .def_rw("public_node_ids", &MeasurePluginLatency::public_node_ids)
      .def_rw("gst_element_name", &MeasurePluginLatency::gst_element_name)
      .def_rw("stream_id", &MeasurePluginLatency::stream_id)
      .def_rw("plugin_instance_id", &MeasurePluginLatency::plugin_instance_id)
      .def_rw("source", &MeasurePluginLatency::source)
      .def_rw("attribution_source", &MeasurePluginLatency::attribution_source)
      .def_rw("mapping_error", &MeasurePluginLatency::mapping_error)
      .def_rw("reliable", &MeasurePluginLatency::reliable)
      .def_rw("calls", &MeasurePluginLatency::calls)
      .def_rw("total_ms", &MeasurePluginLatency::total_ms)
      .def_rw("avg_ms", &MeasurePluginLatency::avg_ms)
      .def_rw("min_ms", &MeasurePluginLatency::min_ms)
      .def_rw("max_ms", &MeasurePluginLatency::max_ms);

  nb::class_<MeasureCounters>(m, "MeasureCounters")
      .def(nb::init<>())
      .def_rw("inputs_enqueued", &MeasureCounters::inputs_enqueued)
      .def_rw("inputs_pushed", &MeasureCounters::inputs_pushed)
      .def_rw("outputs_ready", &MeasureCounters::outputs_ready)
      .def_rw("outputs_pulled", &MeasureCounters::outputs_pulled)
      .def_rw("inputs_dropped", &MeasureCounters::inputs_dropped)
      .def_rw("outputs_dropped", &MeasureCounters::outputs_dropped);

  nb::class_<MeasureInputStats>(m, "MeasureInputStats")
      .def(nb::init<>())
      .def_rw("push_count", &MeasureInputStats::push_count)
      .def_rw("push_failures", &MeasureInputStats::push_failures)
      .def_rw("pull_count", &MeasureInputStats::pull_count)
      .def_rw("poll_count", &MeasureInputStats::poll_count)
      .def_rw("dropped_frames", &MeasureInputStats::dropped_frames)
      .def_rw("renegotiations", &MeasureInputStats::renegotiations)
      .def_rw("alloc_grows", &MeasureInputStats::alloc_grows)
      .def_rw("growth_blocked", &MeasureInputStats::growth_blocked)
      .def_rw("renegotiation_blocked", &MeasureInputStats::renegotiation_blocked)
      .def_rw("avg_alloc_us", &MeasureInputStats::avg_alloc_us)
      .def_rw("avg_map_us", &MeasureInputStats::avg_map_us)
      .def_rw("avg_copy_us", &MeasureInputStats::avg_copy_us)
      .def_rw("avg_push_us", &MeasureInputStats::avg_push_us)
      .def_rw("avg_pull_wait_us", &MeasureInputStats::avg_pull_wait_us)
      .def_rw("avg_decode_us", &MeasureInputStats::avg_decode_us);

  nb::class_<MeasureReport>(m, "MeasureReport")
      .def(nb::init<>())
      .def_rw("options", &MeasureReport::options)
      .def_rw("warmup_iterations", &MeasureReport::warmup_iterations)
      .def_rw("outputs", &MeasureReport::outputs)
      .def_rw("elapsed_s", &MeasureReport::elapsed_s)
      .def_rw("throughput_batches_per_s", &MeasureReport::throughput_batches_per_s)
      .def_rw("throughput_inferences_per_s", &MeasureReport::throughput_inferences_per_s)
      .def_rw("end_to_end", &MeasureReport::end_to_end)
      .def_rw("frame_gap", &MeasureReport::frame_gap)
      .def_rw("latency_samples_collected", &MeasureReport::latency_samples_collected)
      .def_rw("end_to_end_semantics", &MeasureReport::end_to_end_semantics)
      .def_rw("end_to_end_interpretation", &MeasureReport::end_to_end_interpretation)
      .def_rw("counters", &MeasureReport::counters)
      .def_rw("input", &MeasureReport::input)
      .def_rw("plugin_latency", &MeasureReport::plugin_latency)
      .def_rw("plugin_latency_unattributed", &MeasureReport::plugin_latency_unattributed)
      .def_rw("edge_latency", &MeasureReport::edge_latency)
      .def_rw("edge_latency_unattributed", &MeasureReport::edge_latency_unattributed)
      .def_rw("path_timing", &MeasureReport::path_timing)
      .def_rw("plugin_latency_status", &MeasureReport::plugin_latency_status)
      .def_rw("plugin_latency_source", &MeasureReport::plugin_latency_source)
      .def_rw("message_latency_status", &MeasureReport::message_latency_status)
      .def_rw("message_latency_source", &MeasureReport::message_latency_source)
      .def_rw("metrics_trace_dir", &MeasureReport::metrics_trace_dir)
      .def_rw("warnings", &MeasureReport::warnings)
      .def_rw("trace_loss_detected", &MeasureReport::trace_loss_detected)
      .def_rw("graph_sample_timing_unkeyed", &MeasureReport::graph_sample_timing_unkeyed)
      .def_rw("graph_sample_timing_misses", &MeasureReport::graph_sample_timing_misses)
      .def_rw("node_metrics", &MeasureReport::node_metrics)
      .def_rw("power", &MeasureReport::power)
      .def("text", &MeasureReport::to_text)
      .def("to_text", &MeasureReport::to_text)
      .def("to_json", &MeasureReport::to_json, "indent"_a = 2);

  nb::class_<MeasureScope>(m, "MeasureScope")
      .def("stop", &MeasureScope::stop)
      .def("stopped", &MeasureScope::stopped);

  nb::class_<Run>(m, "Run")
      .def(nb::init<>())
      .def("__bool__", [](const Run& run) { return static_cast<bool>(run); })
      .def("can_push", &Run::can_push)
      .def("can_pull", &Run::can_pull)
      .def("running", &Run::running)
      .def("input_names", &Run::input_names)
      .def("output_names", &Run::output_names)
      .def("push_tensors", static_cast<bool (Run::*)(const simaai::neat::TensorList&)>(&Run::push),
           "inputs"_a)
      .def("push_samples", static_cast<bool (Run::*)(const simaai::neat::Sample&)>(&Run::push),
           "inputs"_a)
      .def("try_push_tensors",
           static_cast<bool (Run::*)(const simaai::neat::TensorList&)>(&Run::try_push), "inputs"_a)
      .def("try_push_samples",
           static_cast<bool (Run::*)(const simaai::neat::Sample&)>(&Run::try_push), "inputs"_a)
      .def(
          "push",
          [](Run& run, std::string name, nb::object input, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "Run.push(name)");
            if (python_sequence_all_samples(input)) {
              return run.push(name, sample_batch_from_python_input(input));
            }
            return run.push(name,
                            tensor_batch_from_python_input(input, copy, layout, image_format));
          },
          "name"_a, "input"_a, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def(
          "push",
          [](Run& run, nb::object input, bool copy, std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "Run.push");
            if (python_sequence_all_samples(input)) {
              return run.push(sample_batch_from_python_input(input));
            }
            return run.push(tensor_batch_from_python_input(input, copy, layout, image_format));
          },
          "input"_a, "copy"_a = false, "layout"_a = nb::none(), "image_format"_a = nb::none())
      .def(
          "try_push",
          [](Run& run, std::string name, nb::object input, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "Run.try_push(name)");
            if (python_sequence_all_samples(input)) {
              return run.try_push(name, sample_batch_from_python_input(input));
            }
            return run.try_push(name,
                                tensor_batch_from_python_input(input, copy, layout, image_format));
          },
          "name"_a, "input"_a, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def(
          "try_push",
          [](Run& run, nb::object input, bool copy, std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "Run.try_push");
            if (python_sequence_all_samples(input)) {
              return run.try_push(sample_batch_from_python_input(input));
            }
            return run.try_push(tensor_batch_from_python_input(input, copy, layout, image_format));
          },
          "input"_a, "copy"_a = false, "layout"_a = nb::none(), "image_format"_a = nb::none())
      .def("close_input", &Run::close_input)
      .def(
          "pull",
          [](Run& run, std::string name, int timeout_ms) { return run.pull(name, timeout_ms); },
          "name"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("pull", static_cast<std::optional<Sample> (Run::*)(int)>(&Run::pull),
           "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "pull_tensors",
          [](Run& run, std::string name, int timeout_ms) {
            return run.pull_tensors(name, timeout_ms);
          },
          "name"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("pull_tensors", static_cast<TensorList (Run::*)(int)>(&Run::pull_tensors),
           "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "pull_samples",
          [](Run& run, std::string name, int timeout_ms) {
            return run.pull_samples(name, timeout_ms);
          },
          "name"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("pull_samples", static_cast<Sample (Run::*)(int)>(&Run::pull_samples),
           "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "run",
          [](Run& run, nb::object input, int timeout_ms, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) -> nb::object {
            reject_single_tensor_or_sample(input, "Run.run");
            if (python_sequence_all_samples(input)) {
              auto samples = sample_batch_from_python_input(input);
              simaai::neat::Sample out;
              {
                nb::gil_scoped_release release;
                out = run.run(samples, timeout_ms);
              }
              return nb::cast(std::move(out));
            }
            auto tensors = tensor_batch_from_python_input(input, copy, layout, image_format);
            simaai::neat::TensorList out;
            {
              nb::gil_scoped_release release;
              out = run.run(tensors, timeout_ms);
            }
            return nb::cast(std::move(out));
          },
          "input"_a, "timeout_ms"_a = -1, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def("start_measurement", static_cast<MeasureScope (Run::*)(bool)>(&Run::start_measurement),
           "include_plugin_latency"_a)
      .def("start_measurement",
           static_cast<MeasureScope (Run::*)(const MeasureOptions&)>(&Run::start_measurement),
           "options"_a = MeasureOptions{})
      .def("last_error", &Run::last_error)
      .def(
          "json",
          [](const Run& run, const RunExportOptions& options) {
            std::string err;
            const std::string body = simaai::neat::run_to_json(run, options, &err);
            if (body.empty()) {
              throw std::runtime_error(err.empty() ? "run_to_json failed" : err);
            }
            return body;
          },
          "options"_a = RunExportOptions{})
      .def(
          "json",
          [](const Run& run, const MeasureReport& report, const RunExportOptions& options) {
            std::string err;
            const std::string body = simaai::neat::run_to_json(run, report, options, &err);
            if (body.empty()) {
              throw std::runtime_error(err.empty() ? "run_to_json failed" : err);
            }
            return body;
          },
          "report"_a, "options"_a = RunExportOptions{})
      .def(
          "save_json",
          [](const Run& run, const std::string& path, const RunExportOptions& options) {
            std::string err;
            if (!simaai::neat::save_run_json(run, path, options, &err)) {
              throw std::runtime_error(err.empty() ? "save_run_json failed" : err);
            }
          },
          "path"_a, "options"_a = RunExportOptions{})
      .def(
          "save_json",
          [](const Run& run, const MeasureReport& report, const std::string& path,
             const RunExportOptions& options) {
            std::string err;
            if (!simaai::neat::save_run_json(run, report, path, options, &err)) {
              throw std::runtime_error(err.empty() ? "save_run_json failed" : err);
            }
          },
          "report"_a, "path"_a, "options"_a = RunExportOptions{})
      .def(
          "graph_run_json",
          [](const Run& run, const GraphRunExportOptions& options) {
            std::string err;
            const std::string body = simaai::neat::graph_run_to_json(run, options, &err);
            if (body.empty()) {
              throw std::runtime_error(err.empty() ? "graph_run_to_json failed" : err);
            }
            return body;
          },
          "options"_a = GraphRunExportOptions{})
      .def(
          "graph_run_json",
          [](const Run& run, const MeasureReport& report, const GraphRunExportOptions& options) {
            std::string err;
            const std::string body = simaai::neat::graph_run_to_json(run, report, options, &err);
            if (body.empty()) {
              throw std::runtime_error(err.empty() ? "graph_run_to_json failed" : err);
            }
            return body;
          },
          "report"_a, "options"_a = GraphRunExportOptions{})
      .def(
          "save_graph_run_json",
          [](const Run& run, const std::string& path, const GraphRunExportOptions& options) {
            std::string err;
            if (!simaai::neat::save_graph_run_json(run, path, options, &err)) {
              throw std::runtime_error(err.empty() ? "save_graph_run_json failed" : err);
            }
          },
          "path"_a, "options"_a = GraphRunExportOptions{})
      .def(
          "save_graph_run_json",
          [](const Run& run, const MeasureReport& report, const std::string& path,
             const GraphRunExportOptions& options) {
            std::string err;
            if (!simaai::neat::save_graph_run_json(run, report, path, options, &err)) {
              throw std::runtime_error(err.empty() ? "save_graph_run_json failed" : err);
            }
          },
          "report"_a, "path"_a, "options"_a = GraphRunExportOptions{})
      .def("stop", &Run::stop)
      .def("close", &Run::close);

  m.def(
      "run_to_json",
      [](const Run& run, const RunExportOptions& options) {
        std::string err;
        const std::string body = simaai::neat::run_to_json(run, options, &err);
        if (body.empty()) {
          throw std::runtime_error(err.empty() ? "run_to_json failed" : err);
        }
        return body;
      },
      "run"_a, "options"_a = RunExportOptions{});

  m.def(
      "run_to_json",
      [](const Run& run, const MeasureReport& report, const RunExportOptions& options) {
        std::string err;
        const std::string body = simaai::neat::run_to_json(run, report, options, &err);
        if (body.empty()) {
          throw std::runtime_error(err.empty() ? "run_to_json failed" : err);
        }
        return body;
      },
      "run"_a, "report"_a, "options"_a = RunExportOptions{});

  m.def(
      "save_run_json",
      [](const Run& run, const std::string& path, const RunExportOptions& options) {
        std::string err;
        if (!simaai::neat::save_run_json(run, path, options, &err)) {
          throw std::runtime_error(err.empty() ? "save_run_json failed" : err);
        }
      },
      "run"_a, "path"_a, "options"_a = RunExportOptions{});

  m.def(
      "save_run_json",
      [](const Run& run, const MeasureReport& report, const std::string& path,
         const RunExportOptions& options) {
        std::string err;
        if (!simaai::neat::save_run_json(run, report, path, options, &err)) {
          throw std::runtime_error(err.empty() ? "save_run_json failed" : err);
        }
      },
      "run"_a, "report"_a, "path"_a, "options"_a = RunExportOptions{});

  m.def(
      "graph_run_to_json",
      [](const Run& run, const GraphRunExportOptions& options) {
        std::string err;
        const std::string body = simaai::neat::graph_run_to_json(run, options, &err);
        if (body.empty()) {
          throw std::runtime_error(err.empty() ? "graph_run_to_json failed" : err);
        }
        return body;
      },
      "run"_a, "options"_a = GraphRunExportOptions{});

  m.def(
      "graph_run_to_json",
      [](const Run& run, const MeasureReport& report, const GraphRunExportOptions& options) {
        std::string err;
        const std::string body = simaai::neat::graph_run_to_json(run, report, options, &err);
        if (body.empty()) {
          throw std::runtime_error(err.empty() ? "graph_run_to_json failed" : err);
        }
        return body;
      },
      "run"_a, "report"_a, "options"_a = GraphRunExportOptions{});

  m.def(
      "save_graph_run_json",
      [](const Run& run, const std::string& path, const GraphRunExportOptions& options) {
        std::string err;
        if (!simaai::neat::save_graph_run_json(run, path, options, &err)) {
          throw std::runtime_error(err.empty() ? "save_graph_run_json failed" : err);
        }
      },
      "run"_a, "path"_a, "options"_a = GraphRunExportOptions{});

  m.def(
      "save_graph_run_json",
      [](const Run& run, const MeasureReport& report, const std::string& path,
         const GraphRunExportOptions& options) {
        std::string err;
        if (!simaai::neat::save_graph_run_json(run, report, path, options, &err)) {
          throw std::runtime_error(err.empty() ? "save_graph_run_json failed" : err);
        }
      },
      "run"_a, "report"_a, "path"_a, "options"_a = GraphRunExportOptions{});

  nb::class_<simaai::neat::RtspServerHandle>(m, "RtspServerHandle")
      .def(nb::init<>())
      .def("url", &simaai::neat::RtspServerHandle::url)
      .def("stop", &simaai::neat::RtspServerHandle::stop)
      .def("kill", &simaai::neat::RtspServerHandle::kill)
      .def("running", &simaai::neat::RtspServerHandle::running);

  nb::class_<simaai::neat::Node>(m, "Node")
      .def("kind", &simaai::neat::Node::kind)
      .def("user_label", &simaai::neat::Node::user_label)
      .def("caps_behavior", &simaai::neat::Node::caps_behavior)
      .def("input_role", &simaai::neat::Node::input_role);

  nb::class_<Graph>(m, "Graph")
      .def(nb::init<const GraphOptions&>(), "options"_a = GraphOptions{})
      .def(nb::init<std::string, const GraphOptions&>(), "name"_a, "options"_a = GraphOptions{})
      .def("set_name", &Graph::set_name, "name"_a, nb::rv_policy::reference_internal)
      .def_prop_ro("name", &Graph::name)
      .def("inputs", &Graph::inputs)
      .def("outputs", &Graph::outputs)
      .def(
          "add",
          [](Graph& self, const std::shared_ptr<simaai::neat::Node>& node) -> Graph& {
            return self.add(node);
          },
          "node"_a, nb::rv_policy::reference_internal)
      .def(
          "add", [](Graph& self, const Graph& fragment) -> Graph& { return self.add(fragment); },
          "fragment"_a, nb::rv_policy::reference_internal)
      .def(
          "add",
          [](Graph& self, const simaai::neat::Model& model) -> Graph& { return self.add(model); },
          "model"_a, nb::rv_policy::reference_internal)
      .def(
          "add_node",
          [](Graph& self, const std::shared_ptr<simaai::neat::Node>& node) -> Graph& {
            return self.add(node);
          },
          "node"_a, nb::rv_policy::reference_internal)
      .def(
          "add_graph",
          [](Graph& self, const Graph& fragment) -> Graph& { return self.add(fragment); },
          "fragment"_a, nb::rv_policy::reference_internal)
      .def(
          "add_model",
          [](Graph& self, const simaai::neat::Model& model) -> Graph& { return self.add(model); },
          "model"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const Graph& from, const Graph& to) -> Graph& {
            return self.connect(from, to);
          },
          "from_graph"_a, "to_graph"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const Graph& from, const Graph& to,
             const GraphLinkOptions& options) -> Graph& { return self.connect(from, to, options); },
          "from_graph"_a, "to_graph"_a, "options"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, std::string from_endpoint, std::string to_endpoint) -> Graph& {
            return self.connect(from_endpoint, to_endpoint);
          },
          "from_endpoint"_a, "to_endpoint"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const std::shared_ptr<simaai::neat::Node>& from,
             const std::shared_ptr<simaai::neat::Node>& to) -> Graph& {
            return self.connect(from, to);
          },
          "from_node"_a, "to_node"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const Graph& from, const std::shared_ptr<simaai::neat::Node>& to)
              -> Graph& { return self.connect(from, to); },
          "from_graph"_a, "to_node"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const std::shared_ptr<simaai::neat::Node>& from,
             const Graph& to) -> Graph& { return self.connect(from, to); },
          "from_node"_a, "to_graph"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const simaai::neat::Model& from,
             const simaai::neat::Model& to) -> Graph& { return self.connect(from, to); },
          "from_model"_a, "to_model"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const simaai::neat::Model& from, const Graph& to) -> Graph& {
            return self.connect(from, to);
          },
          "from_model"_a, "to_graph"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const Graph& from, const simaai::neat::Model& to) -> Graph& {
            return self.connect(from, to);
          },
          "from_graph"_a, "to_model"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const simaai::neat::Model& from,
             const std::shared_ptr<simaai::neat::Node>& to) -> Graph& {
            return self.connect(from, to);
          },
          "from_model"_a, "to_node"_a, nb::rv_policy::reference_internal)
      .def(
          "connect",
          [](Graph& self, const std::shared_ptr<simaai::neat::Node>& from,
             const simaai::neat::Model& to) -> Graph& { return self.connect(from, to); },
          "from_node"_a, "to_model"_a, nb::rv_policy::reference_internal)
      .def("custom", static_cast<Graph& (Graph::*)(std::string)>(&Graph::custom), "fragment"_a,
           nb::rv_policy::reference_internal)
      .def("custom_with_role",
           static_cast<Graph& (Graph::*)(std::string, simaai::neat::InputRole)>(&Graph::custom),
           "fragment"_a, "role"_a, nb::rv_policy::reference_internal)
      .def("run_source", static_cast<void (Graph::*)()>(&Graph::run),
           nb::call_guard<nb::gil_scoped_release>())
      .def("run", static_cast<void (Graph::*)()>(&Graph::run),
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "run",
          [](Graph& self, nb::object input, const RunOptions& options, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) -> nb::object {
            reject_single_tensor_or_sample(input, "Graph.run");
            if (python_sequence_all_samples(input)) {
              auto samples = sample_batch_from_python_input(input);
              simaai::neat::Sample out;
              {
                nb::gil_scoped_release release;
                out = self.run(samples, options);
              }
              return nb::cast(std::move(out));
            }
            auto tensors = tensor_batch_from_python_input(input, copy, layout, image_format);
            simaai::neat::TensorList out;
            {
              nb::gil_scoped_release release;
              out = self.run(tensors, options);
            }
            return nb::cast(std::move(out));
          },
          "input"_a, "options"_a = RunOptions{}, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def("build_source", static_cast<Run (Graph::*)(const RunOptions&)>(&Graph::build),
           "options"_a = RunOptions{}, nb::call_guard<nb::gil_scoped_release>())
      .def("build", static_cast<Run (Graph::*)(const RunOptions&)>(&Graph::build),
           "options"_a = RunOptions{}, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "build",
          [](Graph& self, nb::object input, const RunOptions& options, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "Graph.build");
            if (python_sequence_all_samples(input)) {
              auto samples = sample_batch_from_python_input(input);
              nb::gil_scoped_release release;
              return self.build(samples, options);
            }
            auto tensors = tensor_batch_from_python_input(input, copy, layout, image_format);
            nb::gil_scoped_release release;
            return self.build(tensors, options);
          },
          "input"_a, "options"_a = RunOptions{}, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def("run_rtsp", &Graph::run_rtsp, "options"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("validate",
           static_cast<GraphReport (Graph::*)(const ValidateOptions&) const>(&Graph::validate),
           "options"_a = ValidateOptions{})
      .def("add_output_tensor", &Graph::add_output_tensor,
           "options"_a = simaai::neat::OutputTensorOptions{}, nb::rv_policy::reference_internal)
      .def("describe", [](const Graph& self) { return self.describe(); })
      .def("describe_backend", &Graph::describe_backend, "insert_boundaries"_a = false)
      .def("save", &Graph::save, "path"_a)
      .def_static("load", &Graph::load, "path"_a)
      .def_prop_ro("last_pipeline", &Graph::last_pipeline);

  nb::class_<simaai::neat::OutputSpec>(m, "OutputSpec")
      .def(nb::init<>())
      .def_rw("payload_type", &simaai::neat::OutputSpec::payload_type)
      .def_rw("media_type", &simaai::neat::OutputSpec::media_type)
      .def_rw("format", &simaai::neat::OutputSpec::format)
      .def_rw("width", &simaai::neat::OutputSpec::width)
      .def_rw("height", &simaai::neat::OutputSpec::height)
      .def_rw("depth", &simaai::neat::OutputSpec::depth)
      .def_rw("fps_num", &simaai::neat::OutputSpec::fps_num)
      .def_rw("fps_den", &simaai::neat::OutputSpec::fps_den)
      .def_rw("memory", &simaai::neat::OutputSpec::memory)
      .def_rw("layout", &simaai::neat::OutputSpec::layout)
      .def_rw("dtype", &simaai::neat::OutputSpec::dtype)
      .def_rw("byte_size", &simaai::neat::OutputSpec::byte_size)
      .def_rw("certainty", &simaai::neat::OutputSpec::certainty)
      .def_rw("note", &simaai::neat::OutputSpec::note)
      .def("is_unknown", &simaai::neat::OutputSpec::is_unknown)
      .def("has_shape", &simaai::neat::OutputSpec::has_shape);

  nb::enum_<simaai::neat::SpecCertainty>(m, "SpecCertainty")
      .value("Unknown", simaai::neat::SpecCertainty::Unknown)
      .value("Hint", simaai::neat::SpecCertainty::Hint)
      .value("Derived", simaai::neat::SpecCertainty::Derived)
      .value("Authoritative", simaai::neat::SpecCertainty::Authoritative);

  // Phase 1 (plan slice): preprocess-metadata template attached to each input frame (resize/crop/
  // letterbox geometry → needed for mapping detections back to original-image coordinates).
  nb::class_<simaai::neat::PreprocessMetaTemplate>(m, "PreprocessMetaTemplate")
      .def(nb::init<>())
      .def_rw("enabled", &simaai::neat::PreprocessMetaTemplate::enabled)
      .def_rw("target_width", &simaai::neat::PreprocessMetaTemplate::target_width)
      .def_rw("target_height", &simaai::neat::PreprocessMetaTemplate::target_height)
      .def_rw("scaled_width", &simaai::neat::PreprocessMetaTemplate::scaled_width)
      .def_rw("scaled_height", &simaai::neat::PreprocessMetaTemplate::scaled_height)
      .def_rw("resize_mode", &simaai::neat::PreprocessMetaTemplate::resize_mode)
      .def_rw("pad_value", &simaai::neat::PreprocessMetaTemplate::pad_value)
      .def_rw("color_in", &simaai::neat::PreprocessMetaTemplate::color_in)
      .def_rw("color_out", &simaai::neat::PreprocessMetaTemplate::color_out)
      .def_rw("axis_perm", &simaai::neat::PreprocessMetaTemplate::axis_perm)
      .def_rw("normalize", &simaai::neat::PreprocessMetaTemplate::normalize)
      .def_rw("quantize", &simaai::neat::PreprocessMetaTemplate::quantize)
      .def_rw("tessellate", &simaai::neat::PreprocessMetaTemplate::tessellate)
      .def_rw("roi_list_enabled", &simaai::neat::PreprocessMetaTemplate::roi_list_enabled)
      .def_rw("rois", &simaai::neat::PreprocessMetaTemplate::rois)
      .def_rw("roi_input_batch_size", &simaai::neat::PreprocessMetaTemplate::roi_input_batch_size)
      .def_rw("roi_source_width", &simaai::neat::PreprocessMetaTemplate::roi_source_width)
      .def_rw("roi_source_height", &simaai::neat::PreprocessMetaTemplate::roi_source_height)
      .def_rw("roi_source_stride_bytes",
              &simaai::neat::PreprocessMetaTemplate::roi_source_stride_bytes)
      .def_rw("roi_pad_value", &simaai::neat::PreprocessMetaTemplate::roi_pad_value);

  nb::class_<simaai::neat::InputOptions>(m, "InputOptions")
      .def(nb::init<>())
      .def_rw("payload_type", &simaai::neat::InputOptions::payload_type)
      .def_prop_rw("format", format_enum_getter(&simaai::neat::InputOptions::format),
                   format_enum_setter(&simaai::neat::InputOptions::format))
      .def_rw("width", &simaai::neat::InputOptions::width)
      .def_rw("height", &simaai::neat::InputOptions::height)
      .def_rw("depth", &simaai::neat::InputOptions::depth)
      .def_rw("max_width", &simaai::neat::InputOptions::max_width)
      .def_rw("max_height", &simaai::neat::InputOptions::max_height)
      .def_rw("max_depth", &simaai::neat::InputOptions::max_depth)
      .def_rw("fps_n", &simaai::neat::InputOptions::fps_n)
      .def_rw("fps_d", &simaai::neat::InputOptions::fps_d)
      .def_rw("caps_override", &simaai::neat::InputOptions::caps_override)
      .def_rw("is_live", &simaai::neat::InputOptions::is_live)
      .def_rw("do_timestamp", &simaai::neat::InputOptions::do_timestamp)
      .def_rw("block", &simaai::neat::InputOptions::block)
      .def_rw("stream_type", &simaai::neat::InputOptions::stream_type)
      .def_rw("max_bytes", &simaai::neat::InputOptions::max_bytes)
      .def_prop_rw(
          "use_simaai_pool",
          [](const simaai::neat::InputOptions& opt) { return opt.use_simaai_pool; },
          [](simaai::neat::InputOptions& opt, bool value) {
            warn_deprecated_use_simaai_pool_python_once();
            opt.use_simaai_pool = value;
          })
      .def_rw("pool_min_buffers", &simaai::neat::InputOptions::pool_min_buffers)
      .def_rw("pool_max_buffers", &simaai::neat::InputOptions::pool_max_buffers)
      .def_rw("memory_policy", &simaai::neat::InputOptions::memory_policy)
      .def_rw("buffer_name", &simaai::neat::InputOptions::buffer_name)
      .def_rw("preprocess_meta", &simaai::neat::InputOptions::preprocess_meta);

  nb::enum_<simaai::neat::CombinePolicy>(m, "CombinePolicy")
      .value("None_", simaai::neat::CombinePolicy::None)
      .value("ByFrame", simaai::neat::CombinePolicy::ByFrame)
      .value("ByPts", simaai::neat::CombinePolicy::ByPts);

  nb::class_<simaai::neat::OutputOptions>(m, "OutputOptions")
      .def(nb::init<>())
      .def_rw("max_buffers", &simaai::neat::OutputOptions::max_buffers)
      .def_rw("drop", &simaai::neat::OutputOptions::drop)
      .def_rw("sync", &simaai::neat::OutputOptions::sync)
      .def_rw("combine_policy", &simaai::neat::OutputOptions::combine_policy)
      .def_static("latest", &simaai::neat::OutputOptions::Latest)
      .def_static("every_frame", &simaai::neat::OutputOptions::EveryFrame, "max_buffers"_a = 30)
      .def_static("clocked", &simaai::neat::OutputOptions::Clocked, "max_buffers"_a = 1);

  nb::class_<simaai::neat::CameraInputOptions>(m, "CameraInputOptions")
      .def(nb::init<>())
      .def_rw("camera_name", &simaai::neat::CameraInputOptions::camera_name)
      .def_rw("width", &simaai::neat::CameraInputOptions::width)
      .def_rw("height", &simaai::neat::CameraInputOptions::height)
      .def_rw("framerate_num", &simaai::neat::CameraInputOptions::framerate_num)
      .def_rw("framerate_den", &simaai::neat::CameraInputOptions::framerate_den)
      .def_rw("format", &simaai::neat::CameraInputOptions::format)
      .def_rw("buffer_name", &simaai::neat::CameraInputOptions::buffer_name)
      .def_rw("insert_queue", &simaai::neat::CameraInputOptions::insert_queue)
      .def_rw("leaky_queue", &simaai::neat::CameraInputOptions::leaky_queue)
      .def_rw("queue_depth", &simaai::neat::CameraInputOptions::queue_depth)
      .def_rw("allow_cpu_fallback", &simaai::neat::CameraInputOptions::allow_cpu_fallback);

  nb::module_ graphs_mod = m.def_submodule("graphs", "Reusable public Graph fragment helpers");
  graphs_mod.def("branch", &simaai::neat::graphs::Branch, "input"_a, "outputs"_a);
  graphs_mod.def("combine", &simaai::neat::graphs::Combine, "inputs"_a, "output"_a,
                 "policy"_a = simaai::neat::CombinePolicy::ByFrame);

  nb::class_<simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps>(
      m, "HttpMjpegDecodedInputOutputCaps")
      .def(nb::init<>())
      .def_rw("enable",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps::enable)
      .def_prop_rw(
          "format",
          format_enum_getter(
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps::format),
          format_enum_setter(
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps::format))
      .def_rw("width",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps::width)
      .def_rw("height",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps::height)
      .def_rw("fps", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps::fps)
      .def_rw("memory",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::OutputCaps::memory);

  nb::class_<simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions>(
      m, "HttpMjpegDecodedInputOptions")
      .def(nb::init<>())
      .def_rw("url", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::url)
      .def_rw("timeout_seconds",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::timeout_seconds)
      .def_rw("retries", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::retries)
      .def_rw("is_live", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::is_live)
      .def_rw("do_timestamp",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::do_timestamp)
      .def_rw("user_agent", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::user_agent)
      .def_rw("multipart_boundary",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::multipart_boundary)
      .def_rw("multipart_single_stream",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::multipart_single_stream)
      .def_rw("insert_queue",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::insert_queue)
      .def_rw("sync_mode", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::sync_mode)
      .def_rw("sima_allocator_type",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::sima_allocator_type)
      .def_prop_rw("out_format",
                   format_enum_getter(
                       &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::out_format),
                   format_enum_setter(
                       &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::out_format))
      .def_rw("decoder_name",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::decoder_name)
      .def_rw("decoder_raw_output",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::decoder_raw_output)
      .def_rw("decoder_next_element",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::decoder_next_element)
      .def_rw("dec_width", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::dec_width)
      .def_rw("dec_height", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::dec_height)
      .def_rw("dec_fps", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::dec_fps)
      .def_rw("num_buffers",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::num_buffers)
      .def_rw("use_videoconvert",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::use_videoconvert)
      .def_rw("use_videoscale",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::use_videoscale)
      .def_rw("output_caps",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::output_caps)
      .def_rw("extra_fragment",
              &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::extra_fragment)
      .def_rw("ssl_strict", &simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions::ssl_strict);

  nb::class_<simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps>(
      m, "ImageInputGroupOutputCaps")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::enable)
      .def_prop_rw("format",
                   format_enum_getter(
                       &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::format),
                   format_enum_setter(
                       &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::format))
      .def_rw("width", &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::width)
      .def_rw("height", &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::height)
      .def_rw("fps", &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::fps)
      .def_rw("memory", &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::memory);

  nb::enum_<simaai::neat::nodes::groups::ImageInputGroupOptions::Decoder>(m, "ImageInputDecoder")
      .value("Auto", simaai::neat::nodes::groups::ImageInputGroupOptions::Decoder::Auto)
      .value("ForceJpeg", simaai::neat::nodes::groups::ImageInputGroupOptions::Decoder::ForceJpeg)
      .value("ForcePng", simaai::neat::nodes::groups::ImageInputGroupOptions::Decoder::ForcePng)
      .value("Custom", simaai::neat::nodes::groups::ImageInputGroupOptions::Decoder::Custom);

  nb::class_<simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder>(
      m, "ImageInputSimaDecoder")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder::enable)
      .def_rw(
          "sima_allocator_type",
          &simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder::sima_allocator_type)
      .def_rw("decoder_name",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder::decoder_name)
      .def_rw("raw_output",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder::raw_output)
      .def_rw("next_element",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder::next_element)
      .def_rw("use_sw_encoder",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder::use_sw_encoder)
      .def_rw("sw_bitrate_kbps",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::SimaDecoder::sw_bitrate_kbps);

  nb::class_<simaai::neat::nodes::groups::ImageInputGroupOptions>(m, "ImageInputGroupOptions")
      .def(nb::init<>())
      .def_rw("path", &simaai::neat::nodes::groups::ImageInputGroupOptions::path)
      .def_rw("imagefreeze_num_buffers",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::imagefreeze_num_buffers)
      .def_rw("fps", &simaai::neat::nodes::groups::ImageInputGroupOptions::fps)
      .def_rw("sync_mode", &simaai::neat::nodes::groups::ImageInputGroupOptions::sync_mode)
      .def_rw("use_videorate", &simaai::neat::nodes::groups::ImageInputGroupOptions::use_videorate)
      .def_rw("use_videoconvert",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::use_videoconvert)
      .def_rw("use_videoscale",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::use_videoscale)
      .def_rw("output_caps", &simaai::neat::nodes::groups::ImageInputGroupOptions::output_caps)
      .def_rw("decoder", &simaai::neat::nodes::groups::ImageInputGroupOptions::decoder)
      .def_rw("custom_decoder_fragment",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::custom_decoder_fragment)
      .def_rw("sima_decoder", &simaai::neat::nodes::groups::ImageInputGroupOptions::sima_decoder)
      .def_rw("extra_fragment",
              &simaai::neat::nodes::groups::ImageInputGroupOptions::extra_fragment);

  nb::class_<simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps>(
      m, "VideoInputGroupOutputCaps")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::enable)
      .def_prop_rw("format",
                   format_enum_getter(
                       &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::format),
                   format_enum_setter(
                       &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::format))
      .def_rw("width", &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::width)
      .def_rw("height", &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::height)
      .def_rw("fps", &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::fps)
      .def_rw("memory", &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::memory);

  nb::class_<simaai::neat::nodes::groups::VideoInputGroupOptions>(m, "VideoInputGroupOptions")
      .def(nb::init<>())
      .def_rw("path", &simaai::neat::nodes::groups::VideoInputGroupOptions::path)
      .def_rw("demux_video_pad_index",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::demux_video_pad_index)
      .def_rw("insert_queue", &simaai::neat::nodes::groups::VideoInputGroupOptions::insert_queue)
      .def_rw("sync_mode", &simaai::neat::nodes::groups::VideoInputGroupOptions::sync_mode)
      .def_rw("parse_config_interval",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::parse_config_interval)
      .def_rw("parse_enforce_au",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::parse_enforce_au)
      .def_rw("sima_allocator_type",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::sima_allocator_type)
      .def_prop_rw(
          "out_format",
          format_enum_getter(&simaai::neat::nodes::groups::VideoInputGroupOptions::out_format),
          format_enum_setter(&simaai::neat::nodes::groups::VideoInputGroupOptions::out_format))
      .def_rw("use_videoconvert",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::use_videoconvert)
      .def_rw("use_videoscale",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::use_videoscale)
      .def_rw("output_caps", &simaai::neat::nodes::groups::VideoInputGroupOptions::output_caps)
      .def_rw("extra_fragment",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::extra_fragment);

  nb::enum_<simaai::neat::nodes::groups::RtspCodec>(m, "RtspCodec")
      .value("H264", simaai::neat::nodes::groups::RtspCodec::H264)
      .value("MJPEG", simaai::neat::nodes::groups::RtspCodec::MJPEG);

  nb::class_<simaai::neat::nodes::groups::RtspEncodedInputOptions>(m, "RtspEncodedInputOptions")
      .def(nb::init<>())
      .def_rw("url", &simaai::neat::nodes::groups::RtspEncodedInputOptions::url)
      .def_rw("codec", &simaai::neat::nodes::groups::RtspEncodedInputOptions::codec)
      .def_rw("latency_ms", &simaai::neat::nodes::groups::RtspEncodedInputOptions::latency_ms)
      .def_rw("tcp", &simaai::neat::nodes::groups::RtspEncodedInputOptions::tcp)
      .def_rw("drop_on_latency",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::drop_on_latency)
      .def_rw("buffer_mode", &simaai::neat::nodes::groups::RtspEncodedInputOptions::buffer_mode)
      .def_rw("insert_queue", &simaai::neat::nodes::groups::RtspEncodedInputOptions::insert_queue)
      .def_rw("sync_mode", &simaai::neat::nodes::groups::RtspEncodedInputOptions::sync_mode)
      .def_rw("h264_payload_type",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::h264_payload_type)
      .def_rw("mjpeg_payload_type",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::mjpeg_payload_type)
      .def_rw("h264_parse_config_interval",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::h264_parse_config_interval)
      .def_rw("h264_fps", &simaai::neat::nodes::groups::RtspEncodedInputOptions::h264_fps)
      .def_rw("h264_width", &simaai::neat::nodes::groups::RtspEncodedInputOptions::h264_width)
      .def_rw("h264_height", &simaai::neat::nodes::groups::RtspEncodedInputOptions::h264_height)
      .def_rw("auto_caps_from_stream",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::auto_caps_from_stream)
      .def_rw("fallback_h264_fps",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::fallback_h264_fps)
      .def_rw("fallback_h264_width",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::fallback_h264_width)
      .def_rw("fallback_h264_height",
              &simaai::neat::nodes::groups::RtspEncodedInputOptions::fallback_h264_height);

  nb::class_<simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps>(
      m, "RtspDecodedInputOutputCaps")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::enable)
      .def_prop_rw("format",
                   format_enum_getter(
                       &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::format),
                   format_enum_setter(
                       &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::format))
      .def_rw("width", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::width)
      .def_rw("height", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::height)
      .def_rw("fps", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::fps)
      .def_rw("memory", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::memory);

  nb::class_<simaai::neat::nodes::groups::RtspDecodedInputOptions>(m, "RtspDecodedInputOptions")
      .def(nb::init<>())
      .def_rw("url", &simaai::neat::nodes::groups::RtspDecodedInputOptions::url)
      .def_rw("latency_ms", &simaai::neat::nodes::groups::RtspDecodedInputOptions::latency_ms)
      .def_rw("tcp", &simaai::neat::nodes::groups::RtspDecodedInputOptions::tcp)
      .def_rw("drop_on_latency",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::drop_on_latency)
      .def_rw("buffer_mode", &simaai::neat::nodes::groups::RtspDecodedInputOptions::buffer_mode)
      .def_rw("payload_type", &simaai::neat::nodes::groups::RtspDecodedInputOptions::payload_type)
      .def_rw("mjpeg_payload_type",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::mjpeg_payload_type)
      .def_rw("h264_parse_config_interval",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::h264_parse_config_interval)
      .def_rw("h264_fps", &simaai::neat::nodes::groups::RtspDecodedInputOptions::h264_fps)
      .def_rw("h264_width", &simaai::neat::nodes::groups::RtspDecodedInputOptions::h264_width)
      .def_rw("h264_height", &simaai::neat::nodes::groups::RtspDecodedInputOptions::h264_height)
      .def_rw("insert_queue", &simaai::neat::nodes::groups::RtspDecodedInputOptions::insert_queue)
      .def_rw("sync_mode", &simaai::neat::nodes::groups::RtspDecodedInputOptions::sync_mode)
      .def_rw("auto_caps_from_stream",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::auto_caps_from_stream)
      .def_rw("fallback_h264_fps",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::fallback_h264_fps)
      .def_rw("fallback_h264_width",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::fallback_h264_width)
      .def_rw("fallback_h264_height",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::fallback_h264_height)
      .def_rw("sima_allocator_type",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::sima_allocator_type)
      .def_prop_rw(
          "out_format",
          format_enum_getter(&simaai::neat::nodes::groups::RtspDecodedInputOptions::out_format),
          format_enum_setter(&simaai::neat::nodes::groups::RtspDecodedInputOptions::out_format))
      .def_rw("decoder_name", &simaai::neat::nodes::groups::RtspDecodedInputOptions::decoder_name)
      .def_rw("decoder_raw_output",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::decoder_raw_output)
      .def_rw("decoder_next_element",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::decoder_next_element)
      .def_rw("dec_width", &simaai::neat::nodes::groups::RtspDecodedInputOptions::dec_width)
      .def_rw("dec_height", &simaai::neat::nodes::groups::RtspDecodedInputOptions::dec_height)
      .def_rw("dec_fps", &simaai::neat::nodes::groups::RtspDecodedInputOptions::dec_fps)
      .def_rw("num_buffers", &simaai::neat::nodes::groups::RtspDecodedInputOptions::num_buffers)
      .def_rw("use_videoconvert",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::use_videoconvert)
      .def_rw("use_videoscale",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::use_videoscale)
      .def_rw("output_caps", &simaai::neat::nodes::groups::RtspDecodedInputOptions::output_caps)
      .def_rw("extra_fragment",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::extra_fragment)
      .def_rw("codec", &simaai::neat::nodes::groups::RtspDecodedInputOptions::codec);

  nb::class_<simaai::neat::nodes::groups::UdpH264OutputGroupOptions>(m, "UdpH264OutputGroupOptions")
      .def(nb::init<>())
      .def_rw("h264_caps", &simaai::neat::nodes::groups::UdpH264OutputGroupOptions::h264_caps)
      .def_rw("payload_type", &simaai::neat::nodes::groups::UdpH264OutputGroupOptions::payload_type)
      .def_rw("config_interval",
              &simaai::neat::nodes::groups::UdpH264OutputGroupOptions::config_interval)
      .def_rw("udp_host", &simaai::neat::nodes::groups::UdpH264OutputGroupOptions::udp_host)
      .def_rw("udp_port", &simaai::neat::nodes::groups::UdpH264OutputGroupOptions::udp_port)
      .def_rw("udp_sync", &simaai::neat::nodes::groups::UdpH264OutputGroupOptions::udp_sync)
      .def_rw("udp_async", &simaai::neat::nodes::groups::UdpH264OutputGroupOptions::udp_async);

  nb::class_<simaai::neat::nodes::groups::VideoSenderRtpOptions>(m, "VideoSenderRtpOptions")
      .def(nb::init<>())
      .def_rw("payload_type", &simaai::neat::nodes::groups::VideoSenderRtpOptions::payload_type)
      .def_rw("config_interval",
              &simaai::neat::nodes::groups::VideoSenderRtpOptions::config_interval);

  nb::class_<simaai::neat::nodes::groups::VideoSenderEncoderOptions>(m, "VideoSenderEncoderOptions")
      .def(nb::init<>())
      .def_rw("bitrate_kbps", &simaai::neat::nodes::groups::VideoSenderEncoderOptions::bitrate_kbps)
      .def_rw("profile", &simaai::neat::nodes::groups::VideoSenderEncoderOptions::profile)
      .def_rw("level", &simaai::neat::nodes::groups::VideoSenderEncoderOptions::level);

  nb::class_<simaai::neat::nodes::groups::VideoSenderOptions>(m, "VideoSenderOptions")
      .def_static("h264_rtp_udp_from_raw",
                  &simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw, "width"_a,
                  "height"_a, "fps"_a)
      .def_static("h264_rtp_udp_from_encoded",
                  &simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded)
      .def("is_raw_input", &simaai::neat::nodes::groups::VideoSenderOptions::is_raw_input)
      .def("is_encoded_input", &simaai::neat::nodes::groups::VideoSenderOptions::is_encoded_input)
      .def_prop_ro("width", &simaai::neat::nodes::groups::VideoSenderOptions::width)
      .def_prop_ro("height", &simaai::neat::nodes::groups::VideoSenderOptions::height)
      .def_prop_ro("fps", &simaai::neat::nodes::groups::VideoSenderOptions::fps)
      .def_prop_ro("video_port", &simaai::neat::nodes::groups::VideoSenderOptions::video_port)
      .def_rw("host", &simaai::neat::nodes::groups::VideoSenderOptions::host)
      .def_rw("channel", &simaai::neat::nodes::groups::VideoSenderOptions::channel)
      .def_rw("video_port_base", &simaai::neat::nodes::groups::VideoSenderOptions::video_port_base)
      .def_rw("sync", &simaai::neat::nodes::groups::VideoSenderOptions::sync)
      .def_rw("async_", &simaai::neat::nodes::groups::VideoSenderOptions::async)
      .def_prop_rw(
          "async",
          [](const simaai::neat::nodes::groups::VideoSenderOptions& options) {
            return options.async;
          },
          [](simaai::neat::nodes::groups::VideoSenderOptions& options, bool value) {
            options.async = value;
          })
      .def_rw("rtp", &simaai::neat::nodes::groups::VideoSenderOptions::rtp)
      .def_rw("encoder", &simaai::neat::nodes::groups::VideoSenderOptions::encoder);

  nb::class_<simaai::neat::MetadataSenderOptions>(m, "MetadataSenderOptions")
      .def(nb::init<>())
      .def_rw("host", &simaai::neat::MetadataSenderOptions::host)
      .def_rw("channel", &simaai::neat::MetadataSenderOptions::channel)
      .def_rw("metadata_port_base", &simaai::neat::MetadataSenderOptions::metadata_port_base);

  nb::class_<simaai::neat::MetadataSender>(m, "MetadataSender")
      .def(
          "__init__",
          [](simaai::neat::MetadataSender* self, const simaai::neat::MetadataSenderOptions& opt) {
            std::string err;
            new (self) simaai::neat::MetadataSender(opt, &err);
            if (!self->ok()) {
              self->~MetadataSender();
              throw std::runtime_error(err.empty() ? "MetadataSender init failed" : err);
            }
          },
          "options"_a)
      .def("ok", &simaai::neat::MetadataSender::ok)
      .def("host", &simaai::neat::MetadataSender::host)
      .def("metadata_port", &simaai::neat::MetadataSender::metadata_port)
      .def(
          "send_raw_json",
          [](const simaai::neat::MetadataSender& self, const std::string& payload) {
            std::string err;
            const bool ok = self.send_raw_json(payload, &err);
            if (!ok && !err.empty()) {
              throw std::runtime_error(err);
            }
            return ok;
          },
          "payload"_a)
      .def(
          "send_metadata",
          [](const simaai::neat::MetadataSender& self, const std::string& type,
             const std::string& data_json, int64_t timestamp_ms, const std::string& frame_id) {
            std::string err;
            const bool ok = self.send_metadata(type, data_json, timestamp_ms, frame_id, &err);
            if (!ok && !err.empty()) {
              throw std::runtime_error(err);
            }
            return ok;
          },
          "type"_a, "data_json"_a, "timestamp_ms"_a, "frame_id"_a);

  nb::class_<simaai::neat::UdpOutputOptions>(m, "UdpOutputOptions")
      .def(nb::init<>())
      .def_rw("host", &simaai::neat::UdpOutputOptions::host)
      .def_rw("port", &simaai::neat::UdpOutputOptions::port)
      .def_rw("sync", &simaai::neat::UdpOutputOptions::sync)
      .def_rw("async_", &simaai::neat::UdpOutputOptions::async)
      .def_prop_rw(
          "async", [](const simaai::neat::UdpOutputOptions& options) { return options.async; },
          [](simaai::neat::UdpOutputOptions& options, bool value) { options.async = value; });

  nb::enum_<simaai::neat::H264ParseOptions::Alignment>(m, "H264ParseAlignment")
      .value("Auto", simaai::neat::H264ParseOptions::Alignment::Auto)
      .value("AU", simaai::neat::H264ParseOptions::Alignment::AU)
      .value("NAL", simaai::neat::H264ParseOptions::Alignment::NAL);

  nb::enum_<simaai::neat::H264ParseOptions::StreamFormat>(m, "H264ParseStreamFormat")
      .value("Auto", simaai::neat::H264ParseOptions::StreamFormat::Auto)
      .value("AVC", simaai::neat::H264ParseOptions::StreamFormat::AVC)
      .value("ByteStream", simaai::neat::H264ParseOptions::StreamFormat::ByteStream);

  nb::class_<simaai::neat::H264ParseOptions>(m, "H264ParseOptions")
      .def(nb::init<>())
      .def_rw("config_interval", &simaai::neat::H264ParseOptions::config_interval)
      .def_rw("alignment", &simaai::neat::H264ParseOptions::alignment)
      .def_rw("stream_format", &simaai::neat::H264ParseOptions::stream_format)
      .def_rw("enforce_caps", &simaai::neat::H264ParseOptions::enforce_caps);

  nb::class_<simaai::neat::FeatureHistogramOptions>(m, "FeatureHistogramOptions")
      .def(nb::init<>())
      .def_rw("width", &simaai::neat::FeatureHistogramOptions::width)
      .def_rw("height", &simaai::neat::FeatureHistogramOptions::height)
      .def_rw("batch_size", &simaai::neat::FeatureHistogramOptions::batch_size)
      .def_rw("debug", &simaai::neat::FeatureHistogramOptions::debug)
      .def_rw("num_buffers", &simaai::neat::FeatureHistogramOptions::num_buffers)
      .def_rw("element_name", &simaai::neat::FeatureHistogramOptions::element_name)
      .def_rw("input_name", &simaai::neat::FeatureHistogramOptions::input_name)
      .def_rw("output_name", &simaai::neat::FeatureHistogramOptions::output_name)
      .def("summary", &simaai::neat::FeatureHistogramOptions::summary)
      .def("__repr__",
           [](const simaai::neat::FeatureHistogramOptions& options) { return options.summary(); });

  nb::class_<simaai::neat::GriderFastOptions>(m, "GriderFastOptions")
      .def(nb::init<>())
      .def_rw("width", &simaai::neat::GriderFastOptions::width)
      .def_rw("height", &simaai::neat::GriderFastOptions::height)
      .def_rw("batch_size", &simaai::neat::GriderFastOptions::batch_size)
      .def_rw("debug", &simaai::neat::GriderFastOptions::debug)
      .def_rw("num_buffers", &simaai::neat::GriderFastOptions::num_buffers)
      .def_rw("element_name", &simaai::neat::GriderFastOptions::element_name)
      .def_rw("threshold", &simaai::neat::GriderFastOptions::threshold)
      .def_rw("max_features", &simaai::neat::GriderFastOptions::max_features)
      .def_rw("grid_x", &simaai::neat::GriderFastOptions::grid_x)
      .def_rw("grid_y", &simaai::neat::GriderFastOptions::grid_y)
      .def_rw("min_px_dist", &simaai::neat::GriderFastOptions::min_px_dist)
      .def_rw("input_name", &simaai::neat::GriderFastOptions::input_name)
      .def_rw("output_name", &simaai::neat::GriderFastOptions::output_name)
      .def("summary", &simaai::neat::GriderFastOptions::summary)
      .def("__repr__",
           [](const simaai::neat::GriderFastOptions& options) { return options.summary(); });

  nb::class_<simaai::neat::TrackDescriptorOptions>(m, "TrackDescriptorOptions")
      .def(nb::init<>())
      .def_rw("width", &simaai::neat::TrackDescriptorOptions::width)
      .def_rw("height", &simaai::neat::TrackDescriptorOptions::height)
      .def_rw("batch_size", &simaai::neat::TrackDescriptorOptions::batch_size)
      .def_rw("debug", &simaai::neat::TrackDescriptorOptions::debug)
      .def_rw("num_buffers", &simaai::neat::TrackDescriptorOptions::num_buffers)
      .def_rw("element_name", &simaai::neat::TrackDescriptorOptions::element_name)
      .def_rw("threshold", &simaai::neat::TrackDescriptorOptions::threshold)
      .def_rw("max_features", &simaai::neat::TrackDescriptorOptions::max_features)
      .def_rw("grid_x", &simaai::neat::TrackDescriptorOptions::grid_x)
      .def_rw("grid_y", &simaai::neat::TrackDescriptorOptions::grid_y)
      .def_rw("min_px_dist", &simaai::neat::TrackDescriptorOptions::min_px_dist)
      .def_rw("descriptor_words", &simaai::neat::TrackDescriptorOptions::descriptor_words)
      .def_rw("input_name", &simaai::neat::TrackDescriptorOptions::input_name)
      .def_rw("features_output_name", &simaai::neat::TrackDescriptorOptions::features_output_name)
      .def_rw("descriptors_output_name",
              &simaai::neat::TrackDescriptorOptions::descriptors_output_name)
      .def("summary", &simaai::neat::TrackDescriptorOptions::summary)
      .def("__repr__",
           [](const simaai::neat::TrackDescriptorOptions& options) { return options.summary(); });

  nb::class_<simaai::neat::TrackKLTOptions>(m, "TrackKLTOptions")
      .def(nb::init<>())
      .def_rw("width", &simaai::neat::TrackKLTOptions::width)
      .def_rw("height", &simaai::neat::TrackKLTOptions::height)
      .def_rw("batch_size", &simaai::neat::TrackKLTOptions::batch_size)
      .def_rw("num_points", &simaai::neat::TrackKLTOptions::num_points)
      .def_rw("win_half", &simaai::neat::TrackKLTOptions::win_half)
      .def_rw("max_iters", &simaai::neat::TrackKLTOptions::max_iters)
      .def_rw("max_level", &simaai::neat::TrackKLTOptions::max_level)
      .def_rw("detect_new_features", &simaai::neat::TrackKLTOptions::detect_new_features)
      .def_rw("fast_threshold", &simaai::neat::TrackKLTOptions::fast_threshold)
      .def_rw("max_features", &simaai::neat::TrackKLTOptions::max_features)
      .def_rw("grid_x", &simaai::neat::TrackKLTOptions::grid_x)
      .def_rw("grid_y", &simaai::neat::TrackKLTOptions::grid_y)
      .def_rw("min_px_dist", &simaai::neat::TrackKLTOptions::min_px_dist)
      .def_rw("debug", &simaai::neat::TrackKLTOptions::debug)
      .def_rw("num_buffers", &simaai::neat::TrackKLTOptions::num_buffers)
      .def_rw("element_name", &simaai::neat::TrackKLTOptions::element_name)
      .def_rw("prev_image_name", &simaai::neat::TrackKLTOptions::prev_image_name)
      .def_rw("cur_image_name", &simaai::neat::TrackKLTOptions::cur_image_name)
      .def_rw("input_points_name", &simaai::neat::TrackKLTOptions::input_points_name)
      .def_rw("output_points_name", &simaai::neat::TrackKLTOptions::output_points_name)
      .def_rw("output_status_name", &simaai::neat::TrackKLTOptions::output_status_name)
      .def_rw("output_features_name", &simaai::neat::TrackKLTOptions::output_features_name)
      .def("summary", &simaai::neat::TrackKLTOptions::summary)
      .def("__repr__",
           [](const simaai::neat::TrackKLTOptions& options) { return options.summary(); });

  // Phase 4 (plan slice): SiMa render/argmax postprocess node options (advanced/expert nodes;
  // factories live under nodes.* per settled S1 note, options top-level like other *Options).
  nb::class_<simaai::neat::SimaRenderOptions>(m, "SimaRenderOptions")
      .def(nb::init<>())
      .def_rw("config_path", &simaai::neat::SimaRenderOptions::config_path)
      .def_rw("sima_allocator_type", &simaai::neat::SimaRenderOptions::sima_allocator_type)
      .def_rw("silent", &simaai::neat::SimaRenderOptions::silent)
      .def_rw("emit_signals", &simaai::neat::SimaRenderOptions::emit_signals)
      .def_rw("transmit", &simaai::neat::SimaRenderOptions::transmit);
  nb::class_<simaai::neat::SimaArgMaxOptions>(m, "SimaArgMaxOptions")
      .def(nb::init<>())
      .def_rw("config_path", &simaai::neat::SimaArgMaxOptions::config_path)
      .def_rw("sima_allocator_type", &simaai::neat::SimaArgMaxOptions::sima_allocator_type)
      .def_rw("silent", &simaai::neat::SimaArgMaxOptions::silent)
      .def_rw("emit_signals", &simaai::neat::SimaArgMaxOptions::emit_signals)
      .def_rw("transmit", &simaai::neat::SimaArgMaxOptions::transmit);
  nb::enum_<simaai::neat::SimaDecodeType>(m, "SimaDecodeType")
      .value("H264", simaai::neat::SimaDecodeType::H264)
      .value("JPEG", simaai::neat::SimaDecodeType::JPEG)
      .value("MJPEG", simaai::neat::SimaDecodeType::MJPEG);
  nb::class_<simaai::neat::SimaDecodeOptions>(m, "SimaDecodeOptions")
      .def(nb::init<>())
      .def_rw("type", &simaai::neat::SimaDecodeOptions::type)
      .def_rw("sima_allocator_type", &simaai::neat::SimaDecodeOptions::sima_allocator_type)
      .def_prop_rw("out_format", format_enum_getter(&simaai::neat::SimaDecodeOptions::out_format),
                   format_enum_setter(&simaai::neat::SimaDecodeOptions::out_format))
      .def_rw("decoder_name", &simaai::neat::SimaDecodeOptions::decoder_name)
      .def_rw("raw_output", &simaai::neat::SimaDecodeOptions::raw_output)
      .def_rw("next_element", &simaai::neat::SimaDecodeOptions::next_element)
      .def_rw("dec_width", &simaai::neat::SimaDecodeOptions::dec_width)
      .def_rw("dec_height", &simaai::neat::SimaDecodeOptions::dec_height)
      .def_rw("dec_fps", &simaai::neat::SimaDecodeOptions::dec_fps)
      .def_rw("num_buffers", &simaai::neat::SimaDecodeOptions::num_buffers);
  nb::class_<simaai::neat::HttpSourceOptions>(m, "HttpSourceOptions")
      .def(nb::init<>())
      .def_rw("location", &simaai::neat::HttpSourceOptions::location)
      .def_rw("timeout_seconds", &simaai::neat::HttpSourceOptions::timeout_seconds)
      .def_rw("retries", &simaai::neat::HttpSourceOptions::retries)
      .def_rw("is_live", &simaai::neat::HttpSourceOptions::is_live)
      .def_rw("do_timestamp", &simaai::neat::HttpSourceOptions::do_timestamp)
      .def_rw("user_agent", &simaai::neat::HttpSourceOptions::user_agent)
      .def_rw("ssl_strict", &simaai::neat::HttpSourceOptions::ssl_strict);
  nb::class_<simaai::neat::MultipartJpegDemuxOptions>(m, "MultipartJpegDemuxOptions")
      .def(nb::init<>())
      .def_rw("boundary", &simaai::neat::MultipartJpegDemuxOptions::boundary)
      .def_rw("single_stream", &simaai::neat::MultipartJpegDemuxOptions::single_stream);
  nb::class_<simaai::neat::EncodedCapsFixupOptions>(m, "EncodedCapsFixupOptions")
      .def(nb::init<>())
      .def_rw("media_type", &simaai::neat::EncodedCapsFixupOptions::media_type)
      .def_rw("fallback_fps", &simaai::neat::EncodedCapsFixupOptions::fallback_fps)
      .def_rw("use_rtsp_sdp_fps", &simaai::neat::EncodedCapsFixupOptions::use_rtsp_sdp_fps);
  nb::class_<simaai::neat::JpegParseOptions>(m, "JpegParseOptions")
      .def(nb::init<>())
      .def_rw("disable_passthrough", &simaai::neat::JpegParseOptions::disable_passthrough);

  // Phase 4 (plan slice): CVU dtype-bridge atoms (cast/cast_tess/dequant/detess/detess_cast).
  // Advanced/expert nodes — the planner normally inserts them; the standalone forms exist for
  // manual graphs. Only user-facing scalar/enum fields are bound; internal contract fields
  // (compiled_contract / processcvu_compiled_contract / model_lineage / config_json) are
  // intentionally omitted (planner-set, not user surface). from-Model constructors are deferred
  // (they must register after Model); construct empty and set fields, or use the model-path groups.
  nb::enum_<simaai::neat::CastDirection>(m, "CastDirection")
      .value("Bf16ToFp32", simaai::neat::CastDirection::Bf16ToFp32)
      .value("Fp32ToBf16", simaai::neat::CastDirection::Fp32ToBf16);
  nb::class_<simaai::neat::CastOptions>(m, "CastOptions")
      .def(nb::init<>())
      .def_rw("direction", &simaai::neat::CastOptions::direction)
      .def_rw("element_name", &simaai::neat::CastOptions::element_name)
      .def_rw("silent", &simaai::neat::CastOptions::silent)
      .def_rw("num_buffers", &simaai::neat::CastOptions::num_buffers);
  auto cast_tess_opts_cls =
      nb::class_<simaai::neat::CastTessOptions>(m, "CastTessOptions")
          .def(nb::init<>())
          .def_rw("config_path", &simaai::neat::CastTessOptions::config_path)
          .def_rw("element_name", &simaai::neat::CastTessOptions::element_name)
          .def_rw("num_buffers", &simaai::neat::CastTessOptions::num_buffers)
          .def_rw("num_buffers_model", &simaai::neat::CastTessOptions::num_buffers_model)
          .def_rw("num_buffers_locked", &simaai::neat::CastTessOptions::num_buffers_locked);
  auto dequant_opts_cls =
      nb::class_<simaai::neat::DequantOptions>(m, "DequantOptions")
          .def(nb::init<>())
          .def_rw("element_name", &simaai::neat::DequantOptions::element_name)
          .def_rw("stage_id", &simaai::neat::DequantOptions::stage_id)
          .def_rw("model_managed", &simaai::neat::DequantOptions::model_managed)
          .def_rw("q_scale", &simaai::neat::DequantOptions::q_scale)
          .def_rw("q_zp", &simaai::neat::DequantOptions::q_zp)
          .def_rw("num_buffers", &simaai::neat::DequantOptions::num_buffers)
          .def_rw("num_buffers_model", &simaai::neat::DequantOptions::num_buffers_model)
          .def_rw("num_buffers_locked", &simaai::neat::DequantOptions::num_buffers_locked);
  auto detess_opts_cls =
      nb::class_<simaai::neat::DetessOptions>(m, "DetessOptions")
          .def(nb::init<>())
          .def_rw("config_path", &simaai::neat::DetessOptions::config_path)
          .def_rw("config_dir", &simaai::neat::DetessOptions::config_dir)
          .def_rw("keep_config", &simaai::neat::DetessOptions::keep_config)
          .def_rw("no_json_path", &simaai::neat::DetessOptions::no_json_path)
          .def_rw("upstream_name", &simaai::neat::DetessOptions::upstream_name)
          .def_rw("element_name", &simaai::neat::DetessOptions::element_name)
          .def_rw("num_buffers", &simaai::neat::DetessOptions::num_buffers)
          .def_rw("num_buffers_model", &simaai::neat::DetessOptions::num_buffers_model)
          .def_rw("num_buffers_locked", &simaai::neat::DetessOptions::num_buffers_locked);
  auto detess_cast_opts_cls =
      nb::class_<simaai::neat::DetessCastOptions>(m, "DetessCastOptions")
          .def(nb::init<>())
          .def_rw("config_path", &simaai::neat::DetessCastOptions::config_path)
          .def_rw("upstream_name", &simaai::neat::DetessCastOptions::upstream_name)
          .def_rw("element_name", &simaai::neat::DetessCastOptions::element_name)
          .def_rw("num_buffers", &simaai::neat::DetessCastOptions::num_buffers)
          .def_rw("num_buffers_model", &simaai::neat::DetessCastOptions::num_buffers_model)
          .def_rw("num_buffers_locked", &simaai::neat::DetessCastOptions::num_buffers_locked);

  // Phase 4 (plan slice): PCIe transport node options (host<->board zero-copy).
  nb::class_<simaai::neat::PCIeSrcOptions>(m, "PCIeSrcOptions")
      .def(nb::init<>())
      .def_rw("buffer_size", &simaai::neat::PCIeSrcOptions::buffer_size)
      .def_rw("format", &simaai::neat::PCIeSrcOptions::format)
      .def_rw("width", &simaai::neat::PCIeSrcOptions::width)
      .def_rw("height", &simaai::neat::PCIeSrcOptions::height)
      .def_rw("fps_n", &simaai::neat::PCIeSrcOptions::fps_n)
      .def_rw("fps_d", &simaai::neat::PCIeSrcOptions::fps_d);
  nb::class_<simaai::neat::PCIeSinkOptions>(m, "PCIeSinkOptions")
      .def(nb::init<>())
      .def_rw("config_file", &simaai::neat::PCIeSinkOptions::config_file)
      .def_rw("data_buf_name", &simaai::neat::PCIeSinkOptions::data_buf_name)
      .def_rw("data_buffer_size", &simaai::neat::PCIeSinkOptions::data_buffer_size)
      .def_rw("num_buffers", &simaai::neat::PCIeSinkOptions::num_buffers)
      .def_rw("queue", &simaai::neat::PCIeSinkOptions::queue)
      .def_rw("param_buf_name", &simaai::neat::PCIeSinkOptions::param_buf_name)
      .def_rw("param_buffer_size", &simaai::neat::PCIeSinkOptions::param_buffer_size)
      .def_rw("use_multi_buffers", &simaai::neat::PCIeSinkOptions::use_multi_buffers)
      .def_rw("sync", &simaai::neat::PCIeSinkOptions::sync)
      .def_rw("async_state", &simaai::neat::PCIeSinkOptions::async_state)
      .def_rw("max_lateness_ns", &simaai::neat::PCIeSinkOptions::max_lateness_ns)
      .def_rw("processing_deadline_ns", &simaai::neat::PCIeSinkOptions::processing_deadline_ns)
      .def_rw("transmit_kpi", &simaai::neat::PCIeSinkOptions::transmit_kpi)
      .def_rw("qos", &simaai::neat::PCIeSinkOptions::qos);

  nb::module_ nodes_mod = m.def_submodule("nodes", "Node factory helpers");
  nodes_mod.def("queue", &simaai::neat::nodes::Queue);
  nodes_mod.def("rtsp_input", &simaai::neat::nodes::RTSPInput, "url"_a, "latency_ms"_a = 200,
                "tcp"_a = true, "drop_on_latency"_a = false, "buffer_mode"_a = "");
  nodes_mod.def("h264_depacketize", &simaai::neat::nodes::H264Depacketize, "payload_type"_a = 96,
                "h264_parse_config_interval"_a = -1, "h264_fps"_a = -1, "h264_width"_a = -1,
                "h264_height"_a = -1, "enforce_h264_caps"_a = true);
  nodes_mod.def("input",
                static_cast<std::shared_ptr<simaai::neat::Node> (*)(simaai::neat::InputOptions)>(
                    &simaai::neat::nodes::Input),
                "options"_a = simaai::neat::InputOptions{});
  nodes_mod.def(
      "input",
      static_cast<std::shared_ptr<simaai::neat::Node> (*)(std::string, simaai::neat::InputOptions)>(
          &simaai::neat::nodes::Input),
      "name"_a, "options"_a = simaai::neat::InputOptions{});
  nodes_mod.def("camera_input", &simaai::neat::nodes::CameraInput,
                "options"_a = simaai::neat::CameraInputOptions{});
  nodes_mod.def("output",
                static_cast<std::shared_ptr<simaai::neat::Node> (*)(simaai::neat::OutputOptions)>(
                    &simaai::neat::nodes::Output),
                "options"_a = simaai::neat::OutputOptions{});
  nodes_mod.def("output",
                static_cast<std::shared_ptr<simaai::neat::Node> (*)(
                    std::string, simaai::neat::OutputOptions)>(&simaai::neat::nodes::Output),
                "name"_a, "options"_a = simaai::neat::OutputOptions{});
  nodes_mod.def("video_convert", &simaai::neat::nodes::VideoConvert);
  nodes_mod.def(
      "feature_histogram",
      static_cast<std::shared_ptr<simaai::neat::Node> (*)(simaai::neat::FeatureHistogramOptions)>(
          &simaai::neat::nodes::FeatureHistogram),
      "options"_a = simaai::neat::FeatureHistogramOptions{});
  nodes_mod.def(
      "grider_fast",
      static_cast<std::shared_ptr<simaai::neat::Node> (*)(simaai::neat::GriderFastOptions)>(
          &simaai::neat::nodes::GriderFast),
      "options"_a = simaai::neat::GriderFastOptions{});
  nodes_mod.def(
      "track_descriptor",
      static_cast<std::shared_ptr<simaai::neat::Node> (*)(simaai::neat::TrackDescriptorOptions)>(
          &simaai::neat::nodes::TrackDescriptor),
      "options"_a = simaai::neat::TrackDescriptorOptions{});
  nodes_mod.def("track_klt",
                static_cast<std::shared_ptr<simaai::neat::Node> (*)(simaai::neat::TrackKLTOptions)>(
                    &simaai::neat::nodes::TrackKLT),
                "options"_a = simaai::neat::TrackKLTOptions{});

  // ── Phase 4 (plan slice): caps / custom / common-media node factories ────────────────────────
  // S5: nodes.custom returns a connect()-able shared_ptr<Node> for non-linear topology — this is
  // NOT redundant with the linear-only Graph.custom. S10: format_filter is the friendly name over
  // CapsRaw with optional width/height/fps; caps_raw / caps_nv12_sys_mem are bound as raw parity
  // (caps_i420 deferred per S8/S10). All return shared_ptr<Node> usable in connect()/add().
  nodes_mod.def("custom", &simaai::neat::nodes::Custom, "fragment"_a,
                "role"_a = simaai::neat::InputRole::None);
  nodes_mod.def("caps_raw", &simaai::neat::nodes::CapsRaw, "format"_a, "width"_a = -1,
                "height"_a = -1, "fps"_a = -1, "memory"_a = simaai::neat::CapsMemory::Any);
  nodes_mod.def("format_filter", &simaai::neat::nodes::CapsRaw, "format"_a, "width"_a = -1,
                "height"_a = -1, "fps"_a = -1, "memory"_a = simaai::neat::CapsMemory::Any);
  nodes_mod.def("caps_nv12_sys_mem", &simaai::neat::nodes::CapsNV12SysMem, "width"_a, "height"_a,
                "fps"_a);
  nodes_mod.def("http_source", &simaai::neat::nodes::HttpSource, "options"_a);
  nodes_mod.def("file_input", &simaai::neat::nodes::FileInput, "path"_a);
  nodes_mod.def("image_decode", &simaai::neat::nodes::ImageDecode);
  nodes_mod.def("jpeg_decode", &simaai::neat::nodes::JpegDecode);
  nodes_mod.def("multipart_jpeg_demux", &simaai::neat::nodes::MultipartJpegDemux,
                "options"_a = simaai::neat::MultipartJpegDemuxOptions{});
  nodes_mod.def("encoded_caps_fixup", &simaai::neat::nodes::EncodedCapsFixup, "options"_a);
  nodes_mod.def("jpeg_parse", &simaai::neat::nodes::JpegParse,
                "options"_a = simaai::neat::JpegParseOptions{});
  nodes_mod.def("video_scale", &simaai::neat::nodes::VideoScale);
  nodes_mod.def("video_rate", &simaai::neat::nodes::VideoRate);
  nodes_mod.def("image_freeze", &simaai::neat::nodes::ImageFreeze, "num_buffers"_a = -1);
  nodes_mod.def("video_track_select", &simaai::neat::nodes::VideoTrackSelect,
                "video_pad_index"_a = 0);
  // still_image_input: wrap the strong-typedef geometry args so the Python signature is plain ints.
  nodes_mod.def(
      "still_image_input",
      [](std::string image_path, int content_width, int content_height, int encode_width,
         int encode_height, int fps) {
        return simaai::neat::nodes::StillImageInput(
            std::move(image_path), simaai::neat::StillImageInput::ContentWidth{content_width},
            simaai::neat::StillImageInput::ContentHeight{content_height},
            simaai::neat::StillImageInput::EncodeWidth{encode_width},
            simaai::neat::StillImageInput::EncodeHeight{encode_height},
            simaai::neat::StillImageInput::FramesPerSecond{fps});
      },
      "image_path"_a, "content_width"_a, "content_height"_a, "encode_width"_a, "encode_height"_a,
      "fps"_a = 30);
  // SimaRender (S6): render bbox overlays onto a frame — the render stage UdpOutputGroupG needs, so
  // binding it here keeps the overlay→UDP power reachable while that group stays deferred.
  nodes_mod.def("sima_render", &simaai::neat::nodes::SimaRender,
                "options"_a = simaai::neat::SimaRenderOptions{});
  nodes_mod.def("sima_argmax", &simaai::neat::nodes::SimaArgMax,
                "options"_a = simaai::neat::SimaArgMaxOptions{});
  nodes_mod.def("cast", &simaai::neat::nodes::Cast, "options"_a = simaai::neat::CastOptions{});
  nodes_mod.def("cast_tess", &simaai::neat::nodes::CastTess,
                "options"_a = simaai::neat::CastTessOptions{});
  nodes_mod.def("dequant", &simaai::neat::nodes::Dequant,
                "options"_a = simaai::neat::DequantOptions{});
  nodes_mod.def("detess", &simaai::neat::nodes::Detess,
                "options"_a = simaai::neat::DetessOptions{});
  nodes_mod.def("detess_cast", &simaai::neat::nodes::DetessCast,
                "options"_a = simaai::neat::DetessCastOptions{});
  // RTP/H264 + PCIe transport nodes.
  nodes_mod.def("h264_caps_fixup", &simaai::neat::nodes::H264CapsFixup, "fallback_fps"_a = 30,
                "fallback_width"_a = 1280, "fallback_height"_a = 720);
  nodes_mod.def("rtp_jpeg_depacketize", &simaai::neat::nodes::RTPJpegDepacketize,
                "payload_type"_a = 26);
  nodes_mod.def("h264_encode_sw", &simaai::neat::nodes::H264EncodeSW, "bitrate_kbps"_a = 4000);
  nodes_mod.def("pcie_src", &simaai::neat::nodes::PCIeSrc,
                "options"_a = simaai::neat::PCIeSrcOptions{});
  nodes_mod.def("pcie_sink", &simaai::neat::nodes::PCIeSink,
                "options"_a = simaai::neat::PCIeSinkOptions{});

  nb::module_ groups_mod = m.def_submodule("groups", "Reusable Graph fragment helpers");
  groups_mod.def("image_input", &simaai::neat::nodes::groups::ImageInputGroup, "options"_a);
  groups_mod.def("video_input", &simaai::neat::nodes::groups::VideoInputGroup, "options"_a);
  groups_mod.def("http_mjpeg_decoded_input", &simaai::neat::nodes::groups::HttpMjpegDecodedInput,
                 "options"_a);
  groups_mod.def("rtsp_encoded_input", &simaai::neat::nodes::groups::RtspEncodedInput, "options"_a);
  groups_mod.def("rtsp_decoded_input", &simaai::neat::nodes::groups::RtspDecodedInput, "options"_a);
  groups_mod.def("udp_h264_output_group", &simaai::neat::nodes::groups::UdpH264OutputGroup,
                 "options"_a);
  groups_mod.def("video_sender", &simaai::neat::nodes::groups::VideoSender, "options"_a);
  groups_mod.def("mla", &simaai::neat::nodes::groups::MLA, "model"_a);
  groups_mod.def("image_input_output_spec", &simaai::neat::nodes::groups::ImageInputGroupOutputSpec,
                 "options"_a);
  groups_mod.def("video_input_output_spec", &simaai::neat::nodes::groups::VideoInputGroupOutputSpec,
                 "options"_a);
  groups_mod.def("http_mjpeg_decoded_output_spec",
                 &simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec, "options"_a);
  groups_mod.def("rtsp_encoded_output_spec",
                 &simaai::neat::nodes::groups::RtspEncodedInputOutputSpec, "options"_a);
  groups_mod.def("rtsp_decoded_output_spec",
                 &simaai::neat::nodes::groups::RtspDecodedInputOutputSpec, "options"_a);

  // ── Phase 6 (plan slice S7): object-detection decode helpers ─────────────────────────────────
  // Typed boxes + raw-tensor/byte decoders live in pyneat.detections. The top-level
  // decode_bbox/decode_pose/decode_segmentation are kept as-is — this is NOT a mirror of them; it
  // adds the genuinely-new typed-Box surface (decode_bbox_tensor returns a structured
  // BoxDecodeResult, distinct from the TensorList contract). boxes_to_tensor and the tensor-level
  // pose/seg decoders are intentionally dropped.
  nb::module_ detections_mod =
      m.def_submodule("detections", "Object-detection decode helpers (typed boxes).");
  nb::class_<simaai::neat::Box>(detections_mod, "Box")
      .def(nb::init<>())
      .def_rw("x1", &simaai::neat::Box::x1)
      .def_rw("y1", &simaai::neat::Box::y1)
      .def_rw("x2", &simaai::neat::Box::x2)
      .def_rw("y2", &simaai::neat::Box::y2)
      .def_rw("score", &simaai::neat::Box::score)
      .def_rw("class_id", &simaai::neat::Box::class_id)
      .def("__repr__", [](const simaai::neat::Box& b) {
        return nb::str("Box(x1={}, y1={}, x2={}, y2={}, score={}, class_id={})")
            .format(b.x1, b.y1, b.x2, b.y2, b.score, b.class_id);
      });
  nb::class_<simaai::neat::BoxDecodeResult>(detections_mod, "BoxDecodeResult")
      .def(nb::init<>())
      .def_ro("boxes", &simaai::neat::BoxDecodeResult::boxes)
      .def_prop_ro("raw", [](const simaai::neat::BoxDecodeResult& r) {
        return nb::bytes(reinterpret_cast<const char*>(r.raw.data()), r.raw.size());
      });
  detections_mod.def("decode_bbox_tensor", &simaai::neat::decode_bbox_tensor, "tensor"_a, "img_w"_a,
                     "img_h"_a, "expected_topk"_a = 0, "strict"_a = false);
  detections_mod.def(
      "parse_bbox_bytes",
      [](nb::bytes data, int img_w, int img_h, int expected_topk, bool strict) {
        const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
        std::vector<uint8_t> buf(p, p + data.size());
        return simaai::neat::parse_bbox_bytes(buf, img_w, img_h, expected_topk, strict);
      },
      "data"_a, "img_w"_a, "img_h"_a, "expected_topk"_a = 0, "strict"_a = false);
  detections_mod.def("read_detection_format", &simaai::neat::read_detection_format, "tensor"_a);
  detections_mod.def("format_is_bbox", &simaai::neat::detection_format_is_bbox, "format"_a);
  detections_mod.def("format_is_pose", &simaai::neat::detection_format_is_pose, "format"_a);
  detections_mod.def("format_is_segmentation", &simaai::neat::detection_format_is_segmentation,
                     "format"_a);
  detections_mod.def("format_is_bbox_family", &simaai::neat::detection_format_is_bbox_family,
                     "format"_a);

  // ── Phase 6 (plan slice S9): board power telemetry → pyneat.power ─────────────────────────────
  // New snapshot/rail structs + PowerMonitor lifecycle + the rail/profile free functions live here.
  // Already-bound config/summary types and the *_power_monitor_options factories are REUSED via
  // aliases (not rebound/relocated — relocation would break test_api_surface). format_power_summary
  // / power_summary_to_json (formatting helpers) are deferred.
  nb::module_ power_mod = m.def_submodule("power", "Board power telemetry (PMBus rail readings).");
  nb::class_<simaai::neat::PowerFieldReading>(power_mod, "PowerFieldReading")
      .def_ro("available", &simaai::neat::PowerFieldReading::available)
      .def_ro("raw", &simaai::neat::PowerFieldReading::raw)
      .def_ro("value", &simaai::neat::PowerFieldReading::value)
      .def_ro("error", &simaai::neat::PowerFieldReading::error);
  nb::class_<simaai::neat::PowerRailReading>(power_mod, "PowerRailReading")
      .def_ro("config", &simaai::neat::PowerRailReading::config)
      .def_ro("voltage_v", &simaai::neat::PowerRailReading::voltage_v)
      .def_ro("current_a", &simaai::neat::PowerRailReading::current_a)
      .def_ro("power_w", &simaai::neat::PowerRailReading::power_w);
  nb::class_<simaai::neat::PowerSnapshot>(power_mod, "PowerSnapshot")
      .def_ro("rails", &simaai::neat::PowerSnapshot::rails)
      .def_ro("total_watts", &simaai::neat::PowerSnapshot::total_watts)
      .def_ro("rails_with_power", &simaai::neat::PowerSnapshot::rails_with_power);
  nb::class_<simaai::neat::PowerMonitor>(power_mod, "PowerMonitor")
      .def(nb::init<simaai::neat::PowerMonitorOptions>(),
           "options"_a = simaai::neat::PowerMonitorOptions{})
      .def("start", &simaai::neat::PowerMonitor::start)
      .def("stop", &simaai::neat::PowerMonitor::stop)
      .def("sample_once", &simaai::neat::PowerMonitor::sample_once)
      .def("summary", &simaai::neat::PowerMonitor::summary)
      .def("running", &simaai::neat::PowerMonitor::running);
  power_mod.def("read_power_snapshot", &simaai::neat::read_power_snapshot, "options"_a);
  power_mod.def("default_modalix_som_power_rails", &simaai::neat::default_modalix_som_power_rails);
  power_mod.def("default_modalix_dvt_power_rails", &simaai::neat::default_modalix_dvt_power_rails);
  power_mod.def("detect_default_power_monitor_profile",
                &simaai::neat::detect_default_power_monitor_profile);
  power_mod.def("power_rails_for_profile", &simaai::neat::power_rails_for_profile, "profile"_a);
  // Reuse already-bound config/summary types + option factories (S9: alias, don't relocate).
  power_mod.attr("PowerMonitorProfile") = m.attr("PowerMonitorProfile");
  power_mod.attr("PowerMonitorOptions") = m.attr("PowerMonitorOptions");
  power_mod.attr("PowerRailConfig") = m.attr("PowerRailConfig");
  power_mod.attr("PowerFieldSummary") = m.attr("PowerFieldSummary");
  power_mod.attr("PowerRailSummary") = m.attr("PowerRailSummary");
  power_mod.attr("PowerSummary") = m.attr("PowerSummary");
  power_mod.attr("board_power_monitor_options") = m.attr("board_power_monitor_options");
  power_mod.attr("modalix_som_power_monitor_options") = m.attr("modalix_som_power_monitor_options");
  power_mod.attr("modalix_dvt_power_monitor_options") = m.attr("modalix_dvt_power_monitor_options");

  nb::module_ stages_mod = m.def_submodule("stages", "Standalone model stage helpers");
  stages_mod.def(
      "preproc", &python_stage_preproc,
      "Run the model's Preproc stage on one or more uint8 HWC/HW images.\n\n"
      "Args:\n"
      "  images:       list/tuple of NumPy/Torch/pyneat.Tensor image inputs.\n"
      "  model:        pyneat.Model whose resolved preprocess route should run.\n"
      "  rois:         Optional list[PreprocessRoi]. When set, output order follows ROI order.\n"
      "  image_format: Optional PixelFormat hint. Use BGR for cv2.imread images, RGB for RGB.\n"
      "  copy:         Copy Python image buffers before running Preproc.\n\n"
      "Returns:\n"
      "  list[Tensor] with dtype/layout selected by the model's preprocess route.",
      "images"_a, "model"_a, nb::kw_only(), "rois"_a = std::nullopt,
      "image_format"_a = std::nullopt, "copy"_a = false);

  nb::enum_<simaai::neat::AutoFlag>(m, "AutoFlag")
      .value("Auto", simaai::neat::AutoFlag::Auto)
      .value("On", simaai::neat::AutoFlag::On)
      .value("Off", simaai::neat::AutoFlag::Off);

  nb::enum_<simaai::neat::InputKind>(m, "InputKind")
      .value("Auto", simaai::neat::InputKind::Auto)
      .value("Image", simaai::neat::InputKind::Image)
      .value("Tensor", simaai::neat::InputKind::Tensor);

  nb::enum_<simaai::neat::ResizeMode>(m, "ResizeMode")
      .value("Stretch", simaai::neat::ResizeMode::Stretch)
      .value("Letterbox", simaai::neat::ResizeMode::Letterbox)
      .value("Crop", simaai::neat::ResizeMode::Crop);

  nb::enum_<simaai::neat::PreprocessColorFormat>(m, "PreprocessColorFormat")
      .value("Auto", simaai::neat::PreprocessColorFormat::Auto)
      .value("RGB", simaai::neat::PreprocessColorFormat::RGB)
      .value("BGR", simaai::neat::PreprocessColorFormat::BGR)
      .value("GRAY8", simaai::neat::PreprocessColorFormat::GRAY8)
      .value("NV12", simaai::neat::PreprocessColorFormat::NV12)
      .value("I420", simaai::neat::PreprocessColorFormat::I420);

  nb::enum_<simaai::neat::NormalizePreset>(m, "NormalizePreset")
      .value("None", simaai::neat::NormalizePreset::None)
      .value("ImageNet", simaai::neat::NormalizePreset::ImageNet)
      .value("COCO_YOLO", simaai::neat::NormalizePreset::COCO_YOLO);

  nb::enum_<simaai::neat::TransformType>(m, "TransformType")
      .value("Resize", simaai::neat::TransformType::Resize)
      .value("ColorConvert", simaai::neat::TransformType::ColorConvert)
      .value("LayoutConvert", simaai::neat::TransformType::LayoutConvert)
      .value("Normalize", simaai::neat::TransformType::Normalize)
      .value("Quantize", simaai::neat::TransformType::Quantize)
      .value("Tessellate", simaai::neat::TransformType::Tessellate);

  nb::enum_<simaai::neat::BoxDecodeType>(m, "BoxDecodeType")
      .value("Unspecified", simaai::neat::BoxDecodeType::Unspecified)
      .value("Yolo", simaai::neat::BoxDecodeType::Yolo)
      .value("YoloV5", simaai::neat::BoxDecodeType::YoloV5)
      .value("YoloV5Seg", simaai::neat::BoxDecodeType::YoloV5Seg)
      .value("YoloV7", simaai::neat::BoxDecodeType::YoloV7)
      .value("YoloV7Seg", simaai::neat::BoxDecodeType::YoloV7Seg)
      .value("YoloV8", simaai::neat::BoxDecodeType::YoloV8)
      .value("YoloV8Seg", simaai::neat::BoxDecodeType::YoloV8Seg)
      .value("YoloV8Pose", simaai::neat::BoxDecodeType::YoloV8Pose)
      .value("YoloV9", simaai::neat::BoxDecodeType::YoloV9)
      .value("YoloV9Seg", simaai::neat::BoxDecodeType::YoloV9Seg)
      .value("YoloV10", simaai::neat::BoxDecodeType::YoloV10)
      .value("YoloV10Seg", simaai::neat::BoxDecodeType::YoloV10Seg)
      .value("YoloV26", simaai::neat::BoxDecodeType::YoloV26)
      .value("YoloV26Pose", simaai::neat::BoxDecodeType::YoloV26Pose)
      .value("YoloV26Seg", simaai::neat::BoxDecodeType::YoloV26Seg)
      .value("YoloV6", simaai::neat::BoxDecodeType::YoloV6)
      .value("YoloX", simaai::neat::BoxDecodeType::YoloX)
      .value("Detr", simaai::neat::BoxDecodeType::Detr)
      .value("EffDet", simaai::neat::BoxDecodeType::EffDet)
      .value("RcnnStage1", simaai::neat::BoxDecodeType::RcnnStage1)
      .value("Centernet", simaai::neat::BoxDecodeType::Centernet);

  nb::enum_<simaai::neat::BoxDecodeTypeOption>(m, "BoxDecodeTypeOption")
      .value("Auto", simaai::neat::BoxDecodeTypeOption::Auto)
      .value("PackedPerHead", simaai::neat::BoxDecodeTypeOption::PackedPerHead)
      .value("InterleavedByHead", simaai::neat::BoxDecodeTypeOption::InterleavedByHead)
      .value("GroupedByRole", simaai::neat::BoxDecodeTypeOption::GroupedByRole)
      .value("Split3Interleaved", simaai::neat::BoxDecodeTypeOption::Split3Interleaved)
      .value("Split3Grouped", simaai::neat::BoxDecodeTypeOption::Split3Grouped)
      .value("InterleavedByHeadProbability",
             simaai::neat::BoxDecodeTypeOption::InterleavedByHeadProbability)
      .value("InterleavedByHeadLogit", simaai::neat::BoxDecodeTypeOption::InterleavedByHeadLogit)
      .value("GroupedByRoleProbability",
             simaai::neat::BoxDecodeTypeOption::GroupedByRoleProbability)
      .value("GroupedByRoleLogit", simaai::neat::BoxDecodeTypeOption::GroupedByRoleLogit);

  nb::enum_<simaai::neat::VerbosityLevel>(m, "VerbosityLevel")
      .value("Quiet", simaai::neat::VerbosityLevel::Quiet)
      .value("Production", simaai::neat::VerbosityLevel::Production)
      .value("Verbose", simaai::neat::VerbosityLevel::Verbose);

  nb::class_<simaai::neat::ResizeSpec>(m, "ResizeSpec")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::ResizeSpec::enable)
      .def_rw("width", &simaai::neat::ResizeSpec::width)
      .def_rw("height", &simaai::neat::ResizeSpec::height)
      .def_rw("mode", &simaai::neat::ResizeSpec::mode)
      .def_rw("pad_value", &simaai::neat::ResizeSpec::pad_value)
      .def_rw("scaling_type", &simaai::neat::ResizeSpec::scaling_type);

  nb::class_<simaai::neat::ColorConvertSpec>(m, "ColorConvertSpec")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::ColorConvertSpec::enable)
      .def_rw("input_format", &simaai::neat::ColorConvertSpec::input_format)
      .def_rw("output_format", &simaai::neat::ColorConvertSpec::output_format);

  nb::class_<simaai::neat::LayoutConvertSpec>(m, "LayoutConvertSpec")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::LayoutConvertSpec::enable)
      .def_rw("perm", &simaai::neat::LayoutConvertSpec::perm)
      .def("has_perm", &simaai::neat::LayoutConvertSpec::has_perm);

  nb::class_<simaai::neat::NormalizeSpec>(m, "NormalizeSpec")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::NormalizeSpec::enable)
      .def_rw("mean", &simaai::neat::NormalizeSpec::mean)
      .def_rw("stddev", &simaai::neat::NormalizeSpec::stddev)
      .def_rw("has_explicit_stats", &simaai::neat::NormalizeSpec::has_explicit_stats);

  nb::class_<simaai::neat::QuantizeSpec>(m, "QuantizeSpec")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::QuantizeSpec::enable)
      .def_rw("zero_point", &simaai::neat::QuantizeSpec::zero_point)
      .def_rw("scale", &simaai::neat::QuantizeSpec::scale)
      .def_rw("output_dtype", &simaai::neat::QuantizeSpec::output_dtype);

  nb::class_<simaai::neat::TessellateSpec>(m, "TessellateSpec")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::TessellateSpec::enable)
      .def_rw("slice_shape", &simaai::neat::TessellateSpec::slice_shape)
      .def("set_slice_shape", &simaai::neat::TessellateSpec::set_slice_shape, "shape"_a)
      .def("has_slice_shape", &simaai::neat::TessellateSpec::has_slice_shape);

  nb::class_<simaai::neat::Transform>(m, "Transform")
      .def(nb::init<>())
      .def_rw("type", &simaai::neat::Transform::type)
      .def_rw("resize", &simaai::neat::Transform::resize)
      .def_rw("color_convert", &simaai::neat::Transform::color_convert)
      .def_rw("layout_convert", &simaai::neat::Transform::layout_convert)
      .def_rw("normalize", &simaai::neat::Transform::normalize)
      .def_rw("quantize", &simaai::neat::Transform::quantize)
      .def_rw("tessellate", &simaai::neat::Transform::tessellate);

  nb::class_<simaai::neat::PreprocessOptions>(m, "PreprocessOptions")
      .def(nb::init<>())
      .def_rw("kind", &simaai::neat::PreprocessOptions::kind)
      .def_rw("enable", &simaai::neat::PreprocessOptions::enable)
      .def_rw("input_max_width", &simaai::neat::PreprocessOptions::input_max_width)
      .def_rw("input_max_height", &simaai::neat::PreprocessOptions::input_max_height)
      .def_rw("input_max_depth", &simaai::neat::PreprocessOptions::input_max_depth)
      .def_rw("resize", &simaai::neat::PreprocessOptions::resize)
      .def_rw("color_convert", &simaai::neat::PreprocessOptions::color_convert)
      .def_rw("layout_convert", &simaai::neat::PreprocessOptions::layout_convert)
      .def_rw("normalize", &simaai::neat::PreprocessOptions::normalize)
      .def_rw("quantize", &simaai::neat::PreprocessOptions::quantize)
      .def_rw("tessellate", &simaai::neat::PreprocessOptions::tessellate)
      .def_rw("transforms", &simaai::neat::PreprocessOptions::transforms)
      .def_rw("preset", &simaai::neat::PreprocessOptions::preset);

  nb::class_<simaai::neat::VerboseOptions>(m, "VerboseOptions")
      .def(nb::init<>())
      .def_rw("level", &simaai::neat::VerboseOptions::level)
      .def_rw("progress", &simaai::neat::VerboseOptions::progress)
      .def_rw("progress_force", &simaai::neat::VerboseOptions::progress_force)
      .def_rw("gstreamer", &simaai::neat::VerboseOptions::gstreamer)
      .def_rw("planner", &simaai::neat::VerboseOptions::planner)
      .def_rw("graph", &simaai::neat::VerboseOptions::graph)
      .def_rw("pipeline", &simaai::neat::VerboseOptions::pipeline)
      .def_rw("inputstream", &simaai::neat::VerboseOptions::inputstream)
      .def_rw("tensor", &simaai::neat::VerboseOptions::tensor)
      .def_rw("plugins", &simaai::neat::VerboseOptions::plugins)
      .def_static("quiet", &simaai::neat::VerboseOptions::quiet)
      .def_static("production", &simaai::neat::VerboseOptions::production)
      .def_static("debug_plugins", &simaai::neat::VerboseOptions::debug_plugins)
      .def_static("debug_all", &simaai::neat::VerboseOptions::debug_all);

  nb::class_<simaai::neat::ProcessCvuOptions>(m, "ProcessCvuOptions")
      .def(nb::init<>())
      .def_rw("pre_run_target", &simaai::neat::ProcessCvuOptions::pre_run_target)
      .def_rw("post_run_target", &simaai::neat::ProcessCvuOptions::post_run_target)
      .def_rw("async_", &simaai::neat::ProcessCvuOptions::async);

  nb::class_<simaai::neat::ProcessMlaOptions>(m, "ProcessMlaOptions")
      .def(nb::init<>())
      .def_rw("async_", &simaai::neat::ProcessMlaOptions::async)
      .def_rw("output_pool_buffers", &simaai::neat::ProcessMlaOptions::output_pool_buffers)
      .def_rw("defer_output_invalidate", &simaai::neat::ProcessMlaOptions::defer_output_invalidate);

  nb::class_<simaai::neat::Model::InferenceTerminalPolicy>(m, "InferenceTerminalPolicy")
      .def(nb::init<>())
      .def_rw("mla_only", &simaai::neat::Model::InferenceTerminalPolicy::mla_only)
      .def_rw("last_stage_index", &simaai::neat::Model::InferenceTerminalPolicy::last_stage_index)
      .def_rw("last_stage_name", &simaai::neat::Model::InferenceTerminalPolicy::last_stage_name)
      .def_rw("last_plugin_id", &simaai::neat::Model::InferenceTerminalPolicy::last_plugin_id)
      .def_rw("last_processor", &simaai::neat::Model::InferenceTerminalPolicy::last_processor);

  nb::class_<simaai::neat::Model::Options>(m, "ModelOptions")
      .def(nb::init<>())
      .def_rw("preprocess", &simaai::neat::Model::Options::preprocess)
      .def_rw("decode_type", &simaai::neat::Model::Options::decode_type)
      .def_rw("decode_type_option", &simaai::neat::Model::Options::decode_type_option)
      .def_rw("score_threshold", &simaai::neat::Model::Options::score_threshold)
      .def_rw("nms_iou_threshold", &simaai::neat::Model::Options::nms_iou_threshold)
      .def_rw("top_k", &simaai::neat::Model::Options::top_k)
      .def_rw("num_classes", &simaai::neat::Model::Options::num_classes)
      .def_rw("boxdecode_original_width", &simaai::neat::Model::Options::boxdecode_original_width)
      .def_rw("boxdecode_original_height", &simaai::neat::Model::Options::boxdecode_original_height)
      .def_rw("boxdecode_resize_mode", &simaai::neat::Model::Options::boxdecode_resize_mode)
      .def_rw("upstream_name", &simaai::neat::Model::Options::upstream_name)
      .def_rw("name_suffix", &simaai::neat::Model::Options::name_suffix)
      .def_rw("cleanup_extracted_model_data",
              &simaai::neat::Model::Options::cleanup_extracted_model_data)
      .def_rw("verbose", &simaai::neat::Model::Options::verbose)
      .def_rw("inference_terminal", &simaai::neat::Model::Options::inference_terminal)
      .def_rw("processcvu", &simaai::neat::Model::Options::processcvu)
      .def_rw("processmla", &simaai::neat::Model::Options::processmla)
      .def_rw("advanced_execution", &simaai::neat::Model::Options::advanced_execution);

  nb::class_<simaai::neat::Model::RouteOptions>(m, "ModelRouteOptions")
      .def(nb::init<>())
      .def_rw("include_input", &simaai::neat::Model::RouteOptions::include_input)
      .def_rw("include_output", &simaai::neat::Model::RouteOptions::include_output)
      .def_rw("expose_all_outputs", &simaai::neat::Model::RouteOptions::expose_all_outputs)
      .def_rw("upstream_name", &simaai::neat::Model::RouteOptions::upstream_name)
      .def_rw("name_suffix", &simaai::neat::Model::RouteOptions::name_suffix)
      .def_rw("buffer_name", &simaai::neat::Model::RouteOptions::buffer_name)
      .def_rw("advanced_execution", &simaai::neat::Model::RouteOptions::advanced_execution);

  nb::enum_<simaai::neat::Model::Stage>(m, "ModelStage")
      .value("Preprocess", simaai::neat::Model::Stage::Preprocess)
      .value("Inference", simaai::neat::Model::Stage::Inference)
      .value("Postprocess", simaai::neat::Model::Stage::Postprocess)
      .value("Full", simaai::neat::Model::Stage::Full);

  nb::class_<simaai::neat::Model::Runner>(m, "ModelRunner")
      .def(nb::init<>())
      .def("__bool__", [](const simaai::neat::Model::Runner& r) { return static_cast<bool>(r); })
      .def("push_tensors",
           static_cast<bool (simaai::neat::Model::Runner::*)(const simaai::neat::TensorList&)>(
               &simaai::neat::Model::Runner::push),
           "inputs"_a)
      .def("push_samples",
           static_cast<bool (simaai::neat::Model::Runner::*)(const simaai::neat::Sample&)>(
               &simaai::neat::Model::Runner::push),
           "inputs"_a)
      .def(
          "push",
          [](simaai::neat::Model::Runner& runner, nb::object input, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "ModelRunner.push");
            if (python_sequence_all_samples(input)) {
              return runner.push(sample_batch_from_python_input(input));
            }
            return runner.push(tensor_batch_from_python_input(input, copy, layout, image_format));
          },
          "input"_a, "copy"_a = false, "layout"_a = nb::none(), "image_format"_a = nb::none())
      .def("pull", &simaai::neat::Model::Runner::pull, "timeout_ms"_a = -1,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "run",
          [](simaai::neat::Model::Runner& runner, nb::object input, int timeout_ms, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) -> nb::object {
            reject_single_tensor_or_sample(input, "ModelRunner.run");
            if (python_sequence_all_samples(input)) {
              auto samples = sample_batch_from_python_input(input);
              simaai::neat::Sample out;
              {
                nb::gil_scoped_release release;
                out = runner.run(samples, timeout_ms);
              }
              return nb::cast(std::move(out));
            }
            auto tensors = tensor_batch_from_python_input(input, copy, layout, image_format);
            simaai::neat::TensorList out;
            {
              nb::gil_scoped_release release;
              out = runner.run(tensors, timeout_ms);
            }
            return nb::cast(std::move(out));
          },
          "input"_a, "timeout_ms"_a = -1, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def("start_measurement",
           static_cast<MeasureScope (simaai::neat::Model::Runner::*)(bool)>(
               &simaai::neat::Model::Runner::start_measurement),
           "include_plugin_latency"_a)
      .def("start_measurement",
           static_cast<MeasureScope (simaai::neat::Model::Runner::*)(const MeasureOptions&)>(
               &simaai::neat::Model::Runner::start_measurement),
           "options"_a = MeasureOptions{})
      .def("close_input", &simaai::neat::Model::Runner::close_input)
      .def("close", &simaai::neat::Model::Runner::close);

  nb::class_<simaai::neat::BenchmarkReport>(m, "BenchmarkReport")
      .def(nb::init<>())
      .def_rw("latency_ms", &simaai::neat::BenchmarkReport::latency_ms)
      .def_rw("fps", &simaai::neat::BenchmarkReport::fps)
      .def_rw("avg_power_watts", &simaai::neat::BenchmarkReport::avg_power_watts)
      .def_rw("energy_joules", &simaai::neat::BenchmarkReport::energy_joules);

  // ── Phase 3: model / preprocess introspection (plan slice S1/S14) ────────────────────────────
  // Tiering per S1 ("advanced = by namespace, not by docs"): the diagnostic snapshots that are the
  // direct return of the primary Model methods (ModelInfo, PreprocessRequirements) live at top
  // level; the deep resolved-plan contract tree lives under `pyneat.advanced`, reachable only via
  // Model.resolved_preprocess_plan(). These are read-only result structs (def_ro), not user-built.
  nb::module_ advanced_mod = m.def_submodule(
      "advanced", "Advanced / power-user API surface (raw contracts, expert knobs).");

  // ── Phase 6 (plan slice S8): user-facing format vocabulary ───────────────────────────────────
  // pyneat.Format is the friendly enum (FormatTag alias retained). Only meaningful formats are
  // exposed; the EVXX_*/MLA/BBOX/ARGMAX/DETESSDEQUANT caps-layer spellings are intentionally NOT
  // bound as enum members. Format option fields (FormatSpec in C++) are surfaced as this enum via
  // format_enum_getter/format_enum_setter — assignment takes a pyneat.Format value, not a string.
  nb::enum_<simaai::neat::FormatTag>(m, "Format")
      .value("Auto", simaai::neat::FormatTag::Auto)
      .value("RGB", simaai::neat::FormatTag::RGB)
      .value("BGR", simaai::neat::FormatTag::BGR)
      .value("GRAY8", simaai::neat::FormatTag::GRAY8)
      .value("NV12", simaai::neat::FormatTag::NV12)
      .value("I420", simaai::neat::FormatTag::I420)
      .value("YUYV", simaai::neat::FormatTag::YUYV)
      .value("ENCODED", simaai::neat::FormatTag::ENCODED)
      .value("H264", simaai::neat::FormatTag::H264)
      .value("ByteStream", simaai::neat::FormatTag::ByteStream)
      .value("FP32", simaai::neat::FormatTag::FP32)
      .value("INT8", simaai::neat::FormatTag::INT8)
      .value("UINT8", simaai::neat::FormatTag::UINT8)
      .value("BF16", simaai::neat::FormatTag::BF16);
  m.attr("FormatTag") = m.attr("Format");
  // Format string<->tag converters are caps/jargon utilities → advanced tier (S8).
  advanced_mod.def("format_tag_name", &simaai::neat::format_tag_name, "tag"_a);
  advanced_mod.def("format_tag_to_string", &simaai::neat::format_tag_to_string, "tag"_a);
  advanced_mod.def("format_tag_from_string", &simaai::neat::format_tag_from_string, "value"_a);
  advanced_mod.def("is_raw_video_format", &simaai::neat::is_raw_video_format, "tag"_a);
  advanced_mod.def("is_tensor_payload_format", &simaai::neat::is_tensor_payload_format, "tag"_a);

  // ── Phase 2 (plan slice): metrics/diagnostics types in the advanced tier ─────────────────────
  // MetricsTraceSource is used by MeasureOptions. Edge/path timing rows are detailed
  // diagnostics, so keep their class names under advanced while MeasureReport exposes the data.
  nb::enum_<simaai::neat::MetricsTraceSource>(advanced_mod, "MetricsTraceSource")
      .value("Auto", simaai::neat::MetricsTraceSource::Auto)
      .value("Off", simaai::neat::MetricsTraceSource::Off)
      .value("Lttng", simaai::neat::MetricsTraceSource::Lttng);
  nb::class_<simaai::neat::MeasureEdgeLatency>(advanced_mod, "MeasureEdgeLatency")
      .def_ro("edge_id", &simaai::neat::MeasureEdgeLatency::edge_id)
      .def_ro("name", &simaai::neat::MeasureEdgeLatency::name)
      .def_ro("from_node_id", &simaai::neat::MeasureEdgeLatency::from_node_id)
      .def_ro("to_node_id", &simaai::neat::MeasureEdgeLatency::to_node_id)
      .def_ro("from_runtime_node_id", &simaai::neat::MeasureEdgeLatency::from_runtime_node_id)
      .def_ro("to_runtime_node_id", &simaai::neat::MeasureEdgeLatency::to_runtime_node_id)
      .def_ro("from_element_name", &simaai::neat::MeasureEdgeLatency::from_element_name)
      .def_ro("to_element_name", &simaai::neat::MeasureEdgeLatency::to_element_name)
      .def_ro("from_plugin_instance_id", &simaai::neat::MeasureEdgeLatency::from_plugin_instance_id)
      .def_ro("to_plugin_instance_id", &simaai::neat::MeasureEdgeLatency::to_plugin_instance_id)
      .def_ro("stream_id", &simaai::neat::MeasureEdgeLatency::stream_id)
      .def_ro("samples", &simaai::neat::MeasureEdgeLatency::samples)
      .def_ro("total_ms", &simaai::neat::MeasureEdgeLatency::total_ms)
      .def_ro("avg_ms", &simaai::neat::MeasureEdgeLatency::avg_ms)
      .def_ro("min_ms", &simaai::neat::MeasureEdgeLatency::min_ms)
      .def_ro("max_ms", &simaai::neat::MeasureEdgeLatency::max_ms)
      .def_ro("p50_ms", &simaai::neat::MeasureEdgeLatency::p50_ms)
      .def_ro("p95_ms", &simaai::neat::MeasureEdgeLatency::p95_ms)
      .def_ro("source", &simaai::neat::MeasureEdgeLatency::source)
      .def_ro("timing_semantics", &simaai::neat::MeasureEdgeLatency::timing_semantics)
      .def_ro("attribution_source", &simaai::neat::MeasureEdgeLatency::attribution_source)
      .def_ro("mapping_error", &simaai::neat::MeasureEdgeLatency::mapping_error)
      .def_ro("non_additive", &simaai::neat::MeasureEdgeLatency::non_additive)
      .def_ro("reliable", &simaai::neat::MeasureEdgeLatency::reliable);

  nb::class_<MeasurePathStat>(advanced_mod, "MeasurePathStat")
      .def(nb::init<>())
      .def_rw("samples", &MeasurePathStat::samples)
      .def_rw("avg_ms", &MeasurePathStat::avg_ms)
      .def_rw("p50_ms", &MeasurePathStat::p50_ms)
      .def_rw("p95_ms", &MeasurePathStat::p95_ms)
      .def_rw("max_ms", &MeasurePathStat::max_ms)
      .def_rw("reliable", &MeasurePathStat::reliable);

  nb::class_<MeasurePathIdentity>(advanced_mod, "MeasurePathIdentity")
      .def(nb::init<>())
      .def_rw("primary_key", &MeasurePathIdentity::primary_key)
      .def_rw("fallback_key", &MeasurePathIdentity::fallback_key)
      .def_rw("used_public_fields", &MeasurePathIdentity::used_public_fields)
      .def_rw("sample_identity_source", &MeasurePathIdentity::sample_identity_source);

  nb::class_<MeasurePathNodeArrival>(advanced_mod, "MeasurePathNodeArrival")
      .def(nb::init<>())
      .def_rw("customer_node_id", &MeasurePathNodeArrival::customer_node_id)
      .def_rw("lowered_node_id", &MeasurePathNodeArrival::lowered_node_id)
      .def_rw("runtime_node_id", &MeasurePathNodeArrival::runtime_node_id)
      .def_rw("plugin_instance_id", &MeasurePathNodeArrival::plugin_instance_id)
      .def_rw("stream_id", &MeasurePathNodeArrival::stream_id)
      .def_rw("semantics", &MeasurePathNodeArrival::semantics)
      .def_rw("latency", &MeasurePathNodeArrival::latency);

  nb::class_<MeasurePathInterPluginGap>(advanced_mod, "MeasurePathInterPluginGap")
      .def(nb::init<>())
      .def_rw("customer_edge_id", &MeasurePathInterPluginGap::customer_edge_id)
      .def_rw("lowered_edge_id", &MeasurePathInterPluginGap::lowered_edge_id)
      .def_rw("from_customer_node_id", &MeasurePathInterPluginGap::from_customer_node_id)
      .def_rw("to_customer_node_id", &MeasurePathInterPluginGap::to_customer_node_id)
      .def_rw("from_runtime_node_id", &MeasurePathInterPluginGap::from_runtime_node_id)
      .def_rw("to_runtime_node_id", &MeasurePathInterPluginGap::to_runtime_node_id)
      .def_rw("from_plugin_instance_id", &MeasurePathInterPluginGap::from_plugin_instance_id)
      .def_rw("to_plugin_instance_id", &MeasurePathInterPluginGap::to_plugin_instance_id)
      .def_rw("stream_id", &MeasurePathInterPluginGap::stream_id)
      .def_rw("semantics", &MeasurePathInterPluginGap::semantics)
      .def_rw("latency", &MeasurePathInterPluginGap::latency);

  nb::class_<MeasurePathOutputTail>(advanced_mod, "MeasurePathOutputTail")
      .def(nb::init<>())
      .def_rw("output_endpoint", &MeasurePathOutputTail::output_endpoint)
      .def_rw("customer_output_node_id", &MeasurePathOutputTail::customer_output_node_id)
      .def_rw("lowered_edge_id", &MeasurePathOutputTail::lowered_edge_id)
      .def_rw("stream_id", &MeasurePathOutputTail::stream_id)
      .def_rw("semantics", &MeasurePathOutputTail::semantics)
      .def_rw("latency", &MeasurePathOutputTail::latency);

  nb::class_<MeasurePathTiming>(advanced_mod, "MeasurePathTiming")
      .def(nb::init<>())
      .def_rw("available", &MeasurePathTiming::available)
      .def_rw("status", &MeasurePathTiming::status)
      .def_rw("source", &MeasurePathTiming::source)
      .def_rw("reason", &MeasurePathTiming::reason)
      .def_rw("aggregation", &MeasurePathTiming::aggregation)
      .def_rw("warnings", &MeasurePathTiming::warnings)
      .def_rw("identity", &MeasurePathTiming::identity)
      .def_rw("node_arrival", &MeasurePathTiming::node_arrival)
      .def_rw("inter_plugin_gap", &MeasurePathTiming::inter_plugin_gap)
      .def_rw("output_tail", &MeasurePathTiming::output_tail);

  // ── Phase 6 (plan slice): runtime warm-up + build/version info ───────────────────────────────
  // prewarm_runtime kept (S-naming: no warm_up_runtime alias); prewarm_runtime_async dropped.
  // build_info() preferred over version_info() — returns a dict of the three version strings.
  m.def("prewarm_runtime", &simaai::neat::prewarm_runtime,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("build_info", []() {
    nb::dict info;
    info["version"] = ::sima_neat_version();
    info["platform_version"] = ::sima_neat_platform_version();
    info["abi_version"] = ::sima_neat_abi_version();
    return info;
  });

  using ModelInfoT = simaai::neat::Model::ModelInfo;
  auto model_info_cls = nb::class_<ModelInfoT>(m, "ModelInfo");
  nb::class_<ModelInfoT::RouteNeeds>(model_info_cls, "RouteNeeds")
      .def_ro("pre_quantization", &ModelInfoT::RouteNeeds::pre_quantization)
      .def_ro("pre_tessellation", &ModelInfoT::RouteNeeds::pre_tessellation)
      .def_ro("pre_cast", &ModelInfoT::RouteNeeds::pre_cast)
      .def_ro("post_detessellation", &ModelInfoT::RouteNeeds::post_detessellation)
      .def_ro("post_dequantization", &ModelInfoT::RouteNeeds::post_dequantization)
      .def_ro("post_cast", &ModelInfoT::RouteNeeds::post_cast);
  nb::class_<ModelInfoT::RouteCapabilities>(model_info_cls, "RouteCapabilities")
      .def_ro("has_pre_quantization", &ModelInfoT::RouteCapabilities::has_pre_quantization)
      .def_ro("has_pre_tessellation", &ModelInfoT::RouteCapabilities::has_pre_tessellation)
      .def_ro("has_pre_cast", &ModelInfoT::RouteCapabilities::has_pre_cast)
      .def_ro("has_post_detessellation", &ModelInfoT::RouteCapabilities::has_post_detessellation)
      .def_ro("has_post_dequantization", &ModelInfoT::RouteCapabilities::has_post_dequantization)
      .def_ro("has_post_cast", &ModelInfoT::RouteCapabilities::has_post_cast)
      .def_ro("has_post_boxdecode", &ModelInfoT::RouteCapabilities::has_post_boxdecode);
  nb::class_<ModelInfoT::RouteSelection>(model_info_cls, "RouteSelection")
      .def_ro("include_preprocess_stage", &ModelInfoT::RouteSelection::include_preprocess_stage)
      .def_ro("include_postprocess_stage", &ModelInfoT::RouteSelection::include_postprocess_stage)
      .def_ro("infer_only", &ModelInfoT::RouteSelection::infer_only)
      .def_ro("preprocess_graph", &ModelInfoT::RouteSelection::preprocess_graph)
      .def_ro("selected_post_kind", &ModelInfoT::RouteSelection::selected_post_kind);
  nb::class_<ModelInfoT::OutputTopology>(model_info_cls, "OutputTopology")
      .def_ro("physical_outputs", &ModelInfoT::OutputTopology::physical_outputs)
      .def_ro("logical_outputs", &ModelInfoT::OutputTopology::logical_outputs)
      .def_ro("packed_outputs", &ModelInfoT::OutputTopology::packed_outputs);
  model_info_cls.def_ro("mpk_json_path", &ModelInfoT::mpk_json_path)
      .def_ro("model_name", &ModelInfoT::model_name)
      .def_ro("needs", &ModelInfoT::needs)
      .def_ro("capabilities", &ModelInfoT::capabilities)
      .def_ro("selection", &ModelInfoT::selection)
      .def_ro("output_topology", &ModelInfoT::output_topology)
      .def_ro("pre_kernels", &ModelInfoT::pre_kernels)
      .def_ro("post_kernels", &ModelInfoT::post_kernels)
      .def_ro("warnings", &ModelInfoT::warnings);

  using PreprocReqT = simaai::neat::Model::PreprocessRequirements;
  nb::class_<PreprocReqT>(m, "PreprocessRequirements")
      .def_ro("has_preproc_stage", &PreprocReqT::has_preproc_stage)
      .def_ro("quant_needed", &PreprocReqT::quant_needed)
      .def_ro("tess_needed", &PreprocReqT::tess_needed)
      .def_ro("input_media_type", &PreprocReqT::input_media_type)
      .def_ro("input_format", &PreprocReqT::input_format)
      .def_ro("output_format", &PreprocReqT::output_format)
      .def_ro("output_dtype", &PreprocReqT::output_dtype)
      .def_ro("axis_perm", &PreprocReqT::axis_perm)
      .def_ro("output_shape", &PreprocReqT::output_shape)
      .def_ro("slice_shape", &PreprocReqT::slice_shape)
      .def_ro("q_scale", &PreprocReqT::q_scale)
      .def_ro("q_zp", &PreprocReqT::q_zp);

  // Advanced tier — resolved preprocess plan contract tree.
  nb::enum_<simaai::neat::PreprocessGraphFamily>(advanced_mod, "PreprocessGraphFamily")
      .value("Disabled", simaai::neat::PreprocessGraphFamily::Disabled)
      .value("Preproc", simaai::neat::PreprocessGraphFamily::Preproc)
      .value("Quant", simaai::neat::PreprocessGraphFamily::Quant)
      .value("Tess", simaai::neat::PreprocessGraphFamily::Tess)
      .value("QuantTess", simaai::neat::PreprocessGraphFamily::QuantTess);
  nb::class_<simaai::neat::PreprocessExplicitKnobs>(advanced_mod, "PreprocessExplicitKnobs")
      .def_ro("resize", &simaai::neat::PreprocessExplicitKnobs::resize)
      .def_ro("color_convert", &simaai::neat::PreprocessExplicitKnobs::color_convert)
      .def_ro("layout_convert", &simaai::neat::PreprocessExplicitKnobs::layout_convert)
      .def_ro("normalize", &simaai::neat::PreprocessExplicitKnobs::normalize)
      .def_ro("normalize_stats", &simaai::neat::PreprocessExplicitKnobs::normalize_stats)
      .def_ro("quantize_enable", &simaai::neat::PreprocessExplicitKnobs::quantize_enable)
      .def_ro("quantize_params", &simaai::neat::PreprocessExplicitKnobs::quantize_params)
      .def_ro("tessellate_enable", &simaai::neat::PreprocessExplicitKnobs::tessellate_enable)
      .def_ro("tessellate_geometry", &simaai::neat::PreprocessExplicitKnobs::tessellate_geometry);
  nb::class_<simaai::neat::PreprocessContract>(advanced_mod, "PreprocessContract")
      .def_ro("media_type", &simaai::neat::PreprocessContract::media_type)
      .def_ro("format", &simaai::neat::PreprocessContract::format)
      .def_ro("width", &simaai::neat::PreprocessContract::width)
      .def_ro("height", &simaai::neat::PreprocessContract::height)
      .def_ro("depth", &simaai::neat::PreprocessContract::depth)
      .def_ro("max_width", &simaai::neat::PreprocessContract::max_width)
      .def_ro("max_height", &simaai::neat::PreprocessContract::max_height)
      .def_ro("max_depth", &simaai::neat::PreprocessContract::max_depth);
  nb::class_<simaai::neat::PreprocessMetaContract>(advanced_mod, "PreprocessMetaContract")
      .def_ro("meta_name", &simaai::neat::PreprocessMetaContract::meta_name)
      .def_ro("required_fields", &simaai::neat::PreprocessMetaContract::required_fields);
  nb::class_<simaai::neat::ResolvedPreprocessPlan>(advanced_mod, "ResolvedPreprocessPlan")
      .def_ro("requested", &simaai::neat::ResolvedPreprocessPlan::requested)
      .def_ro("effective", &simaai::neat::ResolvedPreprocessPlan::effective)
      .def_ro("explicit_knobs", &simaai::neat::ResolvedPreprocessPlan::explicit_knobs)
      .def_ro("resolved_kind", &simaai::neat::ResolvedPreprocessPlan::resolved_kind)
      .def_ro("transforms_override", &simaai::neat::ResolvedPreprocessPlan::transforms_override)
      .def_ro("enabled", &simaai::neat::ResolvedPreprocessPlan::enabled)
      .def_ro("graph_family", &simaai::neat::ResolvedPreprocessPlan::graph_family)
      .def_ro("graph_kernel", &simaai::neat::ResolvedPreprocessPlan::graph_kernel)
      .def_ro("graph_config_path", &simaai::neat::ResolvedPreprocessPlan::graph_config_path)
      .def_ro("ingress_contracts", &simaai::neat::ResolvedPreprocessPlan::ingress_contracts)
      .def_ro("mla_contract", &simaai::neat::ResolvedPreprocessPlan::mla_contract)
      .def_ro("meta_contract", &simaai::neat::ResolvedPreprocessPlan::meta_contract)
      .def_ro("warnings", &simaai::neat::ResolvedPreprocessPlan::warnings)
      .def("to_debug_string", &simaai::neat::ResolvedPreprocessPlan::to_debug_string)
      .def("__repr__", &simaai::neat::ResolvedPreprocessPlan::to_debug_string);

  nb::class_<simaai::neat::Model>(m, "Model")
      .def(nb::init<const std::string&>(), "model_path"_a)
      .def(nb::init<const std::string&, const simaai::neat::Model::Options&>(), "model_path"_a,
           "options"_a)
      .def("preprocess", &simaai::neat::Model::preprocess)
      .def("inference", &simaai::neat::Model::inference)
      .def("postprocess", &simaai::neat::Model::postprocess)
      .def("graph", static_cast<simaai::neat::Graph (simaai::neat::Model::*)() const>(
                        &simaai::neat::Model::graph))
      .def("graph",
           static_cast<simaai::neat::Graph (simaai::neat::Model::*)(
               simaai::neat::Model::RouteOptions) const>(&simaai::neat::Model::graph),
           "options"_a)
      .def("graph_with_options",
           static_cast<simaai::neat::Graph (simaai::neat::Model::*)(
               simaai::neat::Model::RouteOptions) const>(&simaai::neat::Model::graph),
           "options"_a)
      .def("input_specs", &simaai::neat::Model::input_specs)
      .def("output_specs", &simaai::neat::Model::output_specs)
      .def("metadata", &simaai::neat::Model::metadata)
      .def("compiled_batch_size", &simaai::neat::Model::compiled_batch_size)
      .def("info", &simaai::neat::Model::info)
      // S11: text-only summary() (human-readable string) alongside structured info(). Composed from
      // the diagnostic snapshot so Python users get a glanceable model report without formatting
      // code.
      .def("summary",
           [](const simaai::neat::Model& model) {
             const auto info = model.info();
             std::string out =
                 "Model: " + (info.model_name.empty() ? "<unnamed>" : info.model_name);
             out += "\n  mpk: " + info.mpk_json_path;
             out += "\n  preprocess: " + (info.selection.preprocess_graph.empty()
                                              ? "none"
                                              : info.selection.preprocess_graph);
             out += "\n  postprocess: " + (info.selection.selected_post_kind.empty()
                                               ? "none"
                                               : info.selection.selected_post_kind);
             out += "\n  outputs: " + std::to_string(info.output_topology.logical_outputs) +
                    " logical / " + std::to_string(info.output_topology.physical_outputs) +
                    " physical" + (info.output_topology.packed_outputs ? " (packed)" : "");
             if (!info.warnings.empty()) {
               out += "\n  warnings: " + std::to_string(info.warnings.size());
             }
             return out;
           })
      .def("preprocess_requirements", &simaai::neat::Model::preprocess_requirements)
      .def("resolved_preprocess_plan", &simaai::neat::Model::resolved_preprocess_plan)
      // Friendly alias (plan "Other final naming decisions"): preprocess_plan() ==
      // resolved_preprocess_plan(); "resolved" is accurate but unnecessary for users.
      .def("preprocess_plan", &simaai::neat::Model::resolved_preprocess_plan)
      .def("fragment", &simaai::neat::Model::fragment, "stage"_a)
      .def("backend_fragment", &simaai::neat::Model::backend_fragment, "stage"_a)
      .def("input_appsrc_options", &simaai::neat::Model::input_appsrc_options, "tensor_mode"_a)
      .def("find_config_path_by_plugin", &simaai::neat::Model::find_config_path_by_plugin,
           "plugin_id"_a)
      .def("find_config_path_by_processor", &simaai::neat::Model::find_config_path_by_processor,
           "processor"_a)
      .def("infer_output_name", &simaai::neat::Model::infer_output_name)
      .def("build", static_cast<simaai::neat::Model::Runner (simaai::neat::Model::*)()>(
                        &simaai::neat::Model::build))
      .def("build_with_route_options",
           static_cast<simaai::neat::Model::Runner (simaai::neat::Model::*)(
               const simaai::neat::Model::RouteOptions&)>(&simaai::neat::Model::build),
           "options"_a)
      .def(
          "build",
          [](simaai::neat::Model& model, nb::object input,
             const simaai::neat::Model::RouteOptions& route_options, const RunOptions& run_options,
             bool copy) {
            reject_single_tensor_or_sample(input, "Model.build");
            if (python_sequence_all_samples(input)) {
              return model.build(sample_batch_from_python_input(input), route_options, run_options);
            }
            auto tensors = tensor_batch_from_python_input(input, copy, std::nullopt, std::nullopt);
            return model.build(tensors, route_options, run_options);
          },
          "input"_a, "route_options"_a = simaai::neat::Model::RouteOptions{},
          "run_options"_a = RunOptions{}, "copy"_a = false)
      .def(
          "run",
          [](simaai::neat::Model& model, nb::object input, int timeout_ms,
             bool copy) -> nb::object {
            reject_single_tensor_or_sample(input, "Model.run");
            if (python_sequence_all_samples(input)) {
              auto samples = sample_batch_from_python_input(input);
              simaai::neat::Sample out;
              {
                nb::gil_scoped_release release;
                out = model.run(samples, timeout_ms);
              }
              return nb::cast(std::move(out));
            }
            auto tensors = tensor_batch_from_python_input(input, copy, std::nullopt, std::nullopt);
            simaai::neat::TensorList out;
            {
              nb::gil_scoped_release release;
              out = model.run(tensors, timeout_ms);
            }
            return nb::cast(std::move(out));
          },
          "input"_a, "timeout_ms"_a = -1, "copy"_a = false)
      .def("benchmark",
           static_cast<simaai::neat::BenchmarkReport (simaai::neat::Model::*)(int, bool)>(
               &simaai::neat::Model::benchmark),
           "num_samples"_a = 100, "include_plugin_latency"_a = false,
           nb::call_guard<nb::gil_scoped_release>());

  // from-Model constructors for the CVU-atom options (registered here, after Model — pulls tile
  // geometry / quant params / model-managed buffer counts so the standalone nodes are actually
  // usable, e.g. detess_cast/dequant require model-managed params). Mirrors PreprocOptions(Model).
  cast_tess_opts_cls.def(nb::init<const simaai::neat::Model&>(), "model"_a);
  dequant_opts_cls.def(nb::init<const simaai::neat::Model&>(), "model"_a);
  detess_opts_cls.def(nb::init<const simaai::neat::Model&>(), "model"_a);
  detess_cast_opts_cls.def(nb::init<const simaai::neat::Model&>(), "model"_a);

  nb::class_<simaai::neat::PreprocOptions>(m, "PreprocOptions")
      .def(nb::init<>())
      .def(nb::init<const simaai::neat::Model&>(), "model"_a)
      .def_rw("input_shape", &simaai::neat::PreprocOptions::input_shape)
      .def_rw("output_shape", &simaai::neat::PreprocOptions::output_shape)
      .def_rw("slice_shape", &simaai::neat::PreprocOptions::slice_shape)
      .def("set_input_shape", &simaai::neat::PreprocOptions::set_input_shape, "shape"_a)
      .def("set_output_shape", &simaai::neat::PreprocOptions::set_output_shape, "shape"_a)
      .def("set_slice_shape", &simaai::neat::PreprocOptions::set_slice_shape, "shape"_a)
      .def("has_input_shape", &simaai::neat::PreprocOptions::has_input_shape)
      .def("has_output_shape", &simaai::neat::PreprocOptions::has_output_shape)
      .def("has_slice_shape", &simaai::neat::PreprocOptions::has_slice_shape)
      .def_rw("scaled_width", &simaai::neat::PreprocOptions::scaled_width)
      .def_rw("scaled_height", &simaai::neat::PreprocOptions::scaled_height)
      .def_rw("batch_size", &simaai::neat::PreprocOptions::batch_size)
      .def_rw("normalize", &simaai::neat::PreprocOptions::normalize)
      .def_rw("aspect_ratio", &simaai::neat::PreprocOptions::aspect_ratio)
      .def_rw("tessellate", &simaai::neat::PreprocOptions::tessellate)
      .def_rw("dynamic_input_dims", &simaai::neat::PreprocOptions::dynamic_input_dims)
      .def_rw("single_output_handoff", &simaai::neat::PreprocOptions::single_output_handoff)
      .def_rw("input_offset", &simaai::neat::PreprocOptions::input_offset)
      .def_rw("input_stride", &simaai::neat::PreprocOptions::input_stride)
      .def_rw("output_stride", &simaai::neat::PreprocOptions::output_stride)
      .def_rw("q_zp", &simaai::neat::PreprocOptions::q_zp)
      .def_rw("q_scale", &simaai::neat::PreprocOptions::q_scale)
      .def_rw("channel_mean", &simaai::neat::PreprocOptions::channel_mean)
      .def_rw("channel_stddev", &simaai::neat::PreprocOptions::channel_stddev)
      .def_rw("input_img_type", &simaai::neat::PreprocOptions::input_img_type)
      .def_rw("output_img_type", &simaai::neat::PreprocOptions::output_img_type)
      .def_rw("output_dtype", &simaai::neat::PreprocOptions::output_dtype)
      .def_rw("scaling_type", &simaai::neat::PreprocOptions::scaling_type)
      .def_rw("padding_type", &simaai::neat::PreprocOptions::padding_type)
      .def_rw("pad_value", &simaai::neat::PreprocOptions::pad_value)
      .def_rw("graph_name", &simaai::neat::PreprocOptions::graph_name)
      .def_rw("node_name", &simaai::neat::PreprocOptions::node_name)
      .def_rw("element_name", &simaai::neat::PreprocOptions::element_name)
      .def_rw("cpu", &simaai::neat::PreprocOptions::cpu)
      .def_rw("next_cpu", &simaai::neat::PreprocOptions::next_cpu)
      .def_rw("debug", &simaai::neat::PreprocOptions::debug)
      .def_rw("upstream_name", &simaai::neat::PreprocOptions::upstream_name)
      .def_rw("graph_input_name", &simaai::neat::PreprocOptions::graph_input_name)
      .def_rw("num_buffers", &simaai::neat::PreprocOptions::num_buffers)
      .def_rw("num_buffers_model", &simaai::neat::PreprocOptions::num_buffers_model)
      .def_rw("num_buffers_locked", &simaai::neat::PreprocOptions::num_buffers_locked)
      .def_rw("model_managed_contract", &simaai::neat::PreprocOptions::model_managed_contract);

  nb::class_<simaai::neat::QuantTessOptions>(m, "QuantTessOptions")
      .def(nb::init<>())
      .def(nb::init<const simaai::neat::Model&>(), "model"_a)
      .def_rw("config_path", &simaai::neat::QuantTessOptions::config_path)
      .def_prop_rw(
          "config_json",
          [](const simaai::neat::QuantTessOptions& options) -> nb::object {
            if (!options.config_json.has_value())
              return nb::none();
            return json_to_python(*options.config_json);
          },
          [](simaai::neat::QuantTessOptions& options, nb::object value) {
            options.config_json = python_to_optional_json(value);
          },
          "value"_a.none())
      .def_rw("element_name", &simaai::neat::QuantTessOptions::element_name)
      .def_rw("num_buffers", &simaai::neat::QuantTessOptions::num_buffers)
      .def_rw("num_buffers_model", &simaai::neat::QuantTessOptions::num_buffers_model)
      .def_rw("num_buffers_locked", &simaai::neat::QuantTessOptions::num_buffers_locked);

  nb::class_<simaai::neat::DetessDequantOptions>(m, "DetessDequantOptions")
      .def(nb::init<>())
      .def(nb::init<const simaai::neat::Model&>(), "model"_a)
      .def_rw("config_path", &simaai::neat::DetessDequantOptions::config_path)
      .def_prop_rw(
          "config_json",
          [](const simaai::neat::DetessDequantOptions& options) -> nb::object {
            if (!options.config_json.has_value())
              return nb::none();
            return json_to_python(*options.config_json);
          },
          [](simaai::neat::DetessDequantOptions& options, nb::object value) {
            options.config_json = python_to_optional_json(value);
          },
          "value"_a.none())
      .def_rw("upstream_name", &simaai::neat::DetessDequantOptions::upstream_name)
      .def_rw("element_name", &simaai::neat::DetessDequantOptions::element_name)
      .def_rw("num_buffers", &simaai::neat::DetessDequantOptions::num_buffers)
      .def_rw("num_buffers_model", &simaai::neat::DetessDequantOptions::num_buffers_model)
      .def_rw("num_buffers_locked", &simaai::neat::DetessDequantOptions::num_buffers_locked);

  nodes_mod.def("preproc", &simaai::neat::nodes::Preproc,
                "options"_a = simaai::neat::PreprocOptions{});
  nodes_mod.def("quant_tess", &simaai::neat::nodes::QuantTess,
                "options"_a = simaai::neat::QuantTessOptions{});
  nodes_mod.def("udp_output", &simaai::neat::nodes::UdpOutput,
                "options"_a = simaai::neat::UdpOutputOptions{});
  nodes_mod.def("h264_encode_sima", &simaai::neat::nodes::H264EncodeSima, "width"_a, "height"_a,
                "fps"_a, "bitrate_kbps"_a = 4000, "profile"_a = "baseline", "level"_a = "4.0");
  nodes_mod.def(
      "h264_decode",
      [](int sima_allocator_type, std::string out_format, std::string decoder_name, bool raw_output,
         std::string next_element, int dec_width, int dec_height, int dec_fps, int num_buffers) {
        warn_deprecated_h264_decode_python();
        return simaai::neat::nodes::H264Decode(
            sima_allocator_type, std::move(out_format), std::move(decoder_name), raw_output,
            std::move(next_element), dec_width, dec_height, dec_fps, num_buffers);
      },
      "sima_allocator_type"_a = 2, "out_format"_a = "NV12", "decoder_name"_a = "",
      "raw_output"_a = false, "next_element"_a = "", "dec_width"_a = -1, "dec_height"_a = -1,
      "dec_fps"_a = -1, "num_buffers"_a = -1);
  nodes_mod.def("sima_decode", &simaai::neat::nodes::SimaDecode,
                "options"_a = simaai::neat::SimaDecodeOptions{});
  nodes_mod.def(
      "h264_parse",
      static_cast<std::shared_ptr<simaai::neat::Node> (*)(simaai::neat::H264ParseOptions)>(
          &simaai::neat::nodes::H264Parse),
      "options"_a);
  nodes_mod.def(
      "h264_parse",
      [](int config_interval) { return simaai::neat::nodes::H264Parse(config_interval); },
      "config_interval"_a = 1);
  nodes_mod.def(
      "h264_packetize",
      [](int payload_type, int config_interval) {
        return simaai::neat::nodes::H264Packetize(
            simaai::neat::H264Packetize::PayloadType{payload_type},
            simaai::neat::H264Packetize::ConfigInterval{config_interval});
      },
      "payload_type"_a = 96, "config_interval"_a = 1);
  nodes_mod.def("detess_dequant", &simaai::neat::nodes::DetessDequant,
                "options"_a = simaai::neat::DetessDequantOptions{});
  nodes_mod.def(
      "sima_box_decode",
      [](const simaai::neat::Model& model, simaai::neat::BoxDecodeType decode_type,
         int original_width, int original_height, double detection_threshold,
         double nms_iou_threshold, int top_k) {
        return simaai::neat::nodes::SimaBoxDecode(
            model, decode_type, detection_threshold, nms_iou_threshold, top_k,
            /*element_name=*/"", std::nullopt, std::nullopt, original_width, original_height);
      },
      "model"_a, "decode_type"_a, "original_width"_a, "original_height"_a, "detection_threshold"_a,
      "nms_iou_threshold"_a, "top_k"_a);
  nodes_mod.def(
      "sima_box_decode",
      [](const simaai::neat::Model& model, simaai::neat::BoxDecodeType decode_type,
         int original_width, int original_height, int model_width, int model_height,
         double detection_threshold, double nms_iou_threshold, int top_k,
         std::optional<simaai::neat::ResizeMode> resize_mode) {
        // When a Model is provided, always go through the model-bound
        // constructor: it pulls the model's compiled boxdecode contract
        // (decode family, quant scales/zero-points, tensor layout, etc.) and
        // populates `compiled_contract`, which lets `compile_node_contract`
        // short-circuit upstream-inference at session build time. Geometry
        // and resize-mode overrides flow through the same path as optional
        // refinements; they never cause the binding to silently fall back to
        // the raw-geometry (no-model) constructor — that path can't recover
        // the model's quant metadata and surfaces as
        //   "boxdecode inferred quantized contract requires upstream q_scale/q_zp"
        // for any composition whose immediate upstream stage is MLA-only.
        //
        // `resize_mode` is for compositions that skip the model's Preproc
        // stage (e.g. CPU-letterboxed FP32 tensor directly into
        // quanttess → mla → boxdecode). It relaxes the per-buffer required-
        // meta contract by stripping `preproc_resize_mode`, so a buffer that
        // has no upstream Preproc emitter doesn't get rejected at chain time.
        return simaai::neat::nodes::SimaBoxDecode(
            model, decode_type, detection_threshold, nms_iou_threshold, top_k,
            /*element_name=*/"", std::nullopt, std::nullopt, original_width, original_height,
            model_width, model_height, resize_mode);
      },
      "model"_a, "decode_type"_a = simaai::neat::BoxDecodeType::Unspecified, "original_width"_a = 0,
      "original_height"_a = 0, "model_width"_a = 0, "model_height"_a = 0,
      "detection_threshold"_a = 0.0, "nms_iou_threshold"_a = 0.0, "top_k"_a = 0,
      "resize_mode"_a = std::nullopt);

  // Do not bind the low-level graph::Graph / GraphRun substrate into Python.
  // Python applications use pyneat.Graph/pyneat.Run only; C++ runtime/compiler
  // tests cover the internal substrate directly.

  m.attr("ERROR_PIPELINE_SHAPE") = simaai::neat::error_codes::kPipelineShape;
  m.attr("ERROR_CAPS") = simaai::neat::error_codes::kCaps;
  m.attr("ERROR_INPUT_SHAPE") = simaai::neat::error_codes::kInputShape;
  m.attr("ERROR_PARSE_LAUNCH") = simaai::neat::error_codes::kParseLaunch;
  m.attr("ERROR_RUNTIME_PULL") = simaai::neat::error_codes::kRuntimePull;
  m.attr("ERROR_IO_PARSE") = simaai::neat::error_codes::kIoParse;
  m.attr("ERROR_IO_OPEN") = simaai::neat::error_codes::kIoOpen;
}

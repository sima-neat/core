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

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "genai/ASRModel.h"
#include "genai/GenAIModel.h"
#include "genai/GenAITypes.h"
#include "genai/GraphFragments.h"
#include "genai/OpenAIServer.h"
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
#include "nodes/sima/VisualFrontend.h"
#include "nodes/groups/GroupOutputSpec.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/VideoInputGroup.h"
#include "nodes/io/Input.h"
#include "nodes/io/MetadataSender.h"
#include "nodes/io/UdpOutput.h"
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
using simaai::neat::GraphMetricsReport;
using simaai::neat::GraphNodeMetrics;
using simaai::neat::GraphOptions;
using simaai::neat::GraphReport;
using simaai::neat::GraphRunAutoExportOptions;
using simaai::neat::GraphRunExportOptions;
using simaai::neat::ImageSpec;
using simaai::neat::MapMode;
using simaai::neat::MeasureLatencyStats;
using simaai::neat::MeasureOptions;
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
using simaai::neat::RunDiagSnapshot;
using simaai::neat::RunElementFlowStats;
using simaai::neat::RunElementPadTimingStats;
using simaai::neat::RunElementTimingStats;
using simaai::neat::RunExportOptions;
using simaai::neat::RunMeasurementSummary;
using simaai::neat::RunMode;
using simaai::neat::RunOptions;
using simaai::neat::RunPreset;
using simaai::neat::RunReportOptions;
using simaai::neat::RunStageStats;
using simaai::neat::RunStats;
using simaai::neat::RuntimeCounters;
using simaai::neat::RuntimeLatencyMetrics;
using simaai::neat::RuntimeMetricGroup;
using simaai::neat::RuntimeMetrics;
using simaai::neat::RuntimeMetricsFormat;
using simaai::neat::RuntimeMetricsOptions;
using simaai::neat::RuntimeMetricValue;
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

simaai::neat::FormatSpec python_to_format_spec(nb::handle value) {
  if (value.is_none()) {
    return simaai::neat::FormatSpec{};
  }
  if (PyUnicode_Check(value.ptr())) {
    return simaai::neat::FormatSpec(nb::cast<std::string>(value));
  }
  throw nb::type_error("format must be a string token such as 'RGB', 'NV12', or 'FP32'");
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

  nb::enum_<RunMode>(m, "RunMode").value("Async", RunMode::Async).value("Sync", RunMode::Sync);

  nb::enum_<RuntimeMetricsFormat>(m, "RuntimeMetricsFormat")
      .value("Text", RuntimeMetricsFormat::Text)
      .value("Json", RuntimeMetricsFormat::Json)
      .value("CompactText", RuntimeMetricsFormat::CompactText);

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
      .def_rw("input_seq", &Sample::input_seq)
      .def_rw("orig_input_seq", &Sample::orig_input_seq)
      .def_rw("pts_ns", &Sample::pts_ns)
      .def_rw("dts_ns", &Sample::dts_ns)
      .def_rw("duration_ns", &Sample::duration_ns)
      .def("to_text", &sample_to_text_for_python);

  m.def("make_tensor_sample", &simaai::neat::make_tensor_sample, "port_name"_a, "tensor"_a);
  m.def("make_text_sample", &make_text_sample_for_python, "port_name"_a, "text"_a);

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

  nb::class_<simaai::neat::genai::OpenAIServerOptions>(m, "OpenAIServerOptions")
      .def(nb::init<>())
      .def_rw("host", &simaai::neat::genai::OpenAIServerOptions::host)
      .def_rw("port", &simaai::neat::genai::OpenAIServerOptions::port);

  nb::class_<simaai::neat::genai::OpenAIServer>(m, "OpenAIServer")
      .def(nb::init<simaai::neat::genai::OpenAIServerOptions>(),
           "options"_a = simaai::neat::genai::OpenAIServerOptions{})
      .def(
          "add_model",
          [](simaai::neat::genai::OpenAIServer& server, const std::filesystem::path& model_dir) {
            return server.add_model(model_dir);
          },
          "model_dir"_a)
      .def(
          "add_model",
          [](simaai::neat::genai::OpenAIServer& server, const std::filesystem::path& model_dir,
             const std::string& served_name) { return server.add_model(model_dir, served_name); },
          "model_dir"_a, "served_name"_a)
      .def("remove_model", &simaai::neat::genai::OpenAIServer::remove_model, "served_name"_a)
      .def("model_names", &simaai::neat::genai::OpenAIServer::model_names)
      .def("start", &simaai::neat::genai::OpenAIServer::start,
           nb::call_guard<nb::gil_scoped_release>())
      .def("stop", &simaai::neat::genai::OpenAIServer::stop,
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
  genai_mod.attr("OpenAIServerOptions") = m.attr("OpenAIServerOptions");
  genai_mod.attr("OpenAIServer") = m.attr("OpenAIServer");

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

  nb::class_<GraphOptions>(m, "GraphOptions")
      .def(nb::init<>())
      .def_rw("callback_timeout_ms", &GraphOptions::callback_timeout_ms)
      .def_rw("element_name_prefix", &GraphOptions::element_name_prefix)
      .def_rw("element_name_suffix", &GraphOptions::element_name_suffix);

  nb::class_<simaai::neat::OutputTensorOptions>(m, "OutputTensorOptions")
      .def(nb::init<>())
      .def_rw("format", &simaai::neat::OutputTensorOptions::format)
      .def_rw("dtype", &simaai::neat::OutputTensorOptions::dtype)
      .def_rw("target_width", &simaai::neat::OutputTensorOptions::target_width)
      .def_rw("target_height", &simaai::neat::OutputTensorOptions::target_height)
      .def_rw("target_fps", &simaai::neat::OutputTensorOptions::target_fps);

  nb::class_<RunAdvancedOptions>(m, "RunAdvancedOptions")
      .def(nb::init<>())
      .def_rw("copy_input", &RunAdvancedOptions::copy_input)
      .def_rw("max_input_bytes", &RunAdvancedOptions::max_input_bytes);

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
      .def_rw("enable_metrics", &RunOptions::enable_metrics)
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

  nb::class_<simaai::neat::InputStreamStats>(m, "InputStreamStats")
      .def(nb::init<>())
      .def_rw("push_count", &simaai::neat::InputStreamStats::push_count)
      .def_rw("push_failures", &simaai::neat::InputStreamStats::push_failures)
      .def_rw("pull_count", &simaai::neat::InputStreamStats::pull_count)
      .def_rw("poll_count", &simaai::neat::InputStreamStats::poll_count)
      .def_rw("dropped_frames", &simaai::neat::InputStreamStats::dropped_frames)
      .def_rw("renegotiations", &simaai::neat::InputStreamStats::renegotiations)
      .def_rw("alloc_grows", &simaai::neat::InputStreamStats::alloc_grows)
      .def_rw("growth_blocked", &simaai::neat::InputStreamStats::growth_blocked)
      .def_rw("renegotiation_blocked", &simaai::neat::InputStreamStats::renegotiation_blocked)
      .def_rw("avg_alloc_us", &simaai::neat::InputStreamStats::avg_alloc_us)
      .def_rw("avg_map_us", &simaai::neat::InputStreamStats::avg_map_us)
      .def_rw("avg_copy_us", &simaai::neat::InputStreamStats::avg_copy_us)
      .def_rw("avg_push_us", &simaai::neat::InputStreamStats::avg_push_us)
      .def_rw("avg_pull_wait_us", &simaai::neat::InputStreamStats::avg_pull_wait_us)
      .def_rw("avg_decode_us", &simaai::neat::InputStreamStats::avg_decode_us);

  nb::class_<RunStats>(m, "RunStats")
      .def(nb::init<>())
      .def_rw("inputs_enqueued", &RunStats::inputs_enqueued)
      .def_rw("inputs_dropped", &RunStats::inputs_dropped)
      .def_rw("inputs_pushed", &RunStats::inputs_pushed)
      .def_rw("outputs_ready", &RunStats::outputs_ready)
      .def_rw("outputs_pulled", &RunStats::outputs_pulled)
      .def_rw("outputs_dropped", &RunStats::outputs_dropped)
      .def_rw("avg_latency_ms", &RunStats::avg_latency_ms)
      .def_rw("min_latency_ms", &RunStats::min_latency_ms)
      .def_rw("max_latency_ms", &RunStats::max_latency_ms);

  nb::class_<RuntimeMetricsOptions>(m, "RuntimeMetricsOptions")
      .def(nb::init<>())
      .def_rw("include_power", &RuntimeMetricsOptions::include_power)
      .def_rw("include_diagnostics", &RuntimeMetricsOptions::include_diagnostics)
      .def_rw("include_pipeline", &RuntimeMetricsOptions::include_pipeline)
      .def_rw("include_percentiles", &RuntimeMetricsOptions::include_percentiles);

  nb::class_<RuntimeLatencyMetrics>(m, "RuntimeLatencyMetrics")
      .def(nb::init<>())
      .def_rw("avg_ms", &RuntimeLatencyMetrics::avg_ms)
      .def_rw("min_ms", &RuntimeLatencyMetrics::min_ms)
      .def_rw("max_ms", &RuntimeLatencyMetrics::max_ms)
      .def_rw("p50_ms", &RuntimeLatencyMetrics::p50_ms)
      .def_rw("p95_ms", &RuntimeLatencyMetrics::p95_ms)
      .def_rw("has_percentiles", &RuntimeLatencyMetrics::has_percentiles);

  nb::class_<RuntimeCounters>(m, "RuntimeCounters")
      .def(nb::init<>())
      .def_rw("inputs_enqueued", &RuntimeCounters::inputs_enqueued)
      .def_rw("inputs_dropped", &RuntimeCounters::inputs_dropped)
      .def_rw("inputs_pushed", &RuntimeCounters::inputs_pushed)
      .def_rw("outputs_ready", &RuntimeCounters::outputs_ready)
      .def_rw("outputs_pulled", &RuntimeCounters::outputs_pulled)
      .def_rw("outputs_dropped", &RuntimeCounters::outputs_dropped);

  nb::class_<RuntimeMetricValue>(m, "RuntimeMetricValue")
      .def(nb::init<>())
      .def_rw("name", &RuntimeMetricValue::name)
      .def_rw("value", &RuntimeMetricValue::value)
      .def_rw("unit", &RuntimeMetricValue::unit);

  nb::class_<RuntimeMetricGroup>(m, "RuntimeMetricGroup")
      .def(nb::init<>())
      .def_rw("name", &RuntimeMetricGroup::name)
      .def_rw("values", &RuntimeMetricGroup::values);

  nb::class_<RuntimeMetrics>(m, "RuntimeMetrics")
      .def(nb::init<>())
      .def_rw("source_kind", &RuntimeMetrics::source_kind)
      .def_rw("source_name", &RuntimeMetrics::source_name)
      .def_rw("elapsed_seconds", &RuntimeMetrics::elapsed_seconds)
      .def_rw("throughput_fps", &RuntimeMetrics::throughput_fps)
      .def_rw("latency", &RuntimeMetrics::latency)
      .def_rw("counters", &RuntimeMetrics::counters)
      .def_rw("power", &RuntimeMetrics::power)
      .def_rw("metadata", &RuntimeMetrics::metadata)
      .def_rw("groups", &RuntimeMetrics::groups);

  nb::class_<RunMeasurementSummary>(m, "RunMeasurementSummary")
      .def(nb::init<>())
      .def_rw("stats", &RunMeasurementSummary::stats)
      .def_rw("input_stats", &RunMeasurementSummary::input_stats)
      .def_rw("elapsed_seconds", &RunMeasurementSummary::elapsed_seconds)
      .def_rw("throughput_fps", &RunMeasurementSummary::throughput_fps)
      .def_rw("power", &RunMeasurementSummary::power);

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

  nb::class_<GraphMetricsReport>(m, "GraphMetricsReport")
      .def(nb::init<>())
      .def_rw("graph_metrics", &GraphMetricsReport::graph_metrics)
      .def_rw("aggregation", &GraphMetricsReport::aggregation)
      .def_rw("latency_semantics", &GraphMetricsReport::latency_semantics)
      .def_rw("throughput_counting", &GraphMetricsReport::throughput_counting)
      .def_rw("node_metrics", &GraphMetricsReport::node_metrics);

  nb::class_<MeasureOptions>(m, "MeasureOptions")
      .def(nb::init<>())
      .def_rw("duration_ms", &MeasureOptions::duration_ms)
      .def_rw("warmup_ms", &MeasureOptions::warmup_ms)
      .def_rw("timeout_ms", &MeasureOptions::timeout_ms)
      .def_rw("include_plugin_latency", &MeasureOptions::include_plugin_latency)
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
      .def_rw("calls", &MeasurePluginLatency::calls)
      .def_rw("total_ms", &MeasurePluginLatency::total_ms)
      .def_rw("avg_ms", &MeasurePluginLatency::avg_ms)
      .def_rw("min_ms", &MeasurePluginLatency::min_ms)
      .def_rw("max_ms", &MeasurePluginLatency::max_ms);

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
      .def_rw("plugin_latency", &MeasureReport::plugin_latency)
      .def_rw("node_metrics", &MeasureReport::node_metrics)
      .def_rw("inputs_pushed", &MeasureReport::inputs_pushed)
      .def_rw("outputs_pulled", &MeasureReport::outputs_pulled)
      .def_rw("inputs_dropped", &MeasureReport::inputs_dropped)
      .def_rw("outputs_dropped", &MeasureReport::outputs_dropped)
      .def_rw("final_run_stats", &MeasureReport::final_run_stats)
      .def_rw("power", &MeasureReport::power)
      .def("text", &MeasureReport::to_text)
      .def("to_text", &MeasureReport::to_text);

  nb::class_<RunStageStats>(m, "RunStageStats")
      .def(nb::init<>())
      .def_rw("stage_name", &RunStageStats::stage_name)
      .def_rw("samples", &RunStageStats::samples)
      .def_rw("total_us", &RunStageStats::total_us)
      .def_rw("max_us", &RunStageStats::max_us);

  nb::class_<RunElementTimingStats>(m, "RunElementTimingStats")
      .def(nb::init<>())
      .def_rw("element_name", &RunElementTimingStats::element_name)
      .def_rw("samples", &RunElementTimingStats::samples)
      .def_rw("total_us", &RunElementTimingStats::total_us)
      .def_rw("max_us", &RunElementTimingStats::max_us)
      .def_rw("min_us", &RunElementTimingStats::min_us)
      .def_rw("missed_in", &RunElementTimingStats::missed_in)
      .def_rw("missed_out", &RunElementTimingStats::missed_out);

  nb::class_<RunElementFlowStats>(m, "RunElementFlowStats")
      .def(nb::init<>())
      .def_rw("element_name", &RunElementFlowStats::element_name)
      .def_rw("in_buffers", &RunElementFlowStats::in_buffers)
      .def_rw("out_buffers", &RunElementFlowStats::out_buffers)
      .def_rw("in_bytes", &RunElementFlowStats::in_bytes)
      .def_rw("out_bytes", &RunElementFlowStats::out_bytes)
      .def_rw("caps_changes", &RunElementFlowStats::caps_changes);

  nb::class_<RunElementPadTimingStats>(m, "RunElementPadTimingStats")
      .def(nb::init<>())
      .def_rw("element_name", &RunElementPadTimingStats::element_name)
      .def_rw("pad_name", &RunElementPadTimingStats::pad_name)
      .def_rw("is_sink", &RunElementPadTimingStats::is_sink)
      .def_rw("samples", &RunElementPadTimingStats::samples)
      .def_rw("inter_arrival_total_us", &RunElementPadTimingStats::inter_arrival_total_us)
      .def_rw("inter_arrival_max_us", &RunElementPadTimingStats::inter_arrival_max_us)
      .def_rw("queue_wait_samples", &RunElementPadTimingStats::queue_wait_samples)
      .def_rw("queue_wait_total_us", &RunElementPadTimingStats::queue_wait_total_us)
      .def_rw("queue_wait_max_us", &RunElementPadTimingStats::queue_wait_max_us)
      .def_rw("bytes", &RunElementPadTimingStats::bytes);

  nb::class_<RunDiagSnapshot>(m, "RunDiagSnapshot")
      .def(nb::init<>())
      .def_rw("stages", &RunDiagSnapshot::stages)
      .def_rw("boundaries", &RunDiagSnapshot::boundaries)
      .def_rw("element_timings", &RunDiagSnapshot::element_timings)
      .def_rw("element_flows", &RunDiagSnapshot::element_flows)
      .def_rw("element_pad_timings", &RunDiagSnapshot::element_pad_timings);

  nb::class_<RunReportOptions>(m, "RunReportOptions")
      .def(nb::init<>())
      .def_rw("include_pipeline", &RunReportOptions::include_pipeline)
      .def_rw("include_stage_timings", &RunReportOptions::include_stage_timings)
      .def_rw("include_element_timings", &RunReportOptions::include_element_timings)
      .def_rw("include_boundaries", &RunReportOptions::include_boundaries)
      .def_rw("include_flow_stats", &RunReportOptions::include_flow_stats)
      .def_rw("include_node_reports", &RunReportOptions::include_node_reports)
      .def_rw("include_next_cpu", &RunReportOptions::include_next_cpu)
      .def_rw("include_queue_depth", &RunReportOptions::include_queue_depth)
      .def_rw("include_num_buffers", &RunReportOptions::include_num_buffers)
      .def_rw("include_run_stats", &RunReportOptions::include_run_stats)
      .def_rw("include_input_stats", &RunReportOptions::include_input_stats)
      .def_rw("include_power", &RunReportOptions::include_power)
      .def_rw("include_system_info", &RunReportOptions::include_system_info);

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
      .def("stats", &Run::stats)
      .def("input_stats", &Run::input_stats)
      .def("diag_snapshot", &Run::diag_snapshot)
      .def("power_summary", &Run::power_summary)
      .def("measurement_summary", &Run::measurement_summary)
      .def("metrics", &Run::metrics, "options"_a = RuntimeMetricsOptions{})
      .def(
          "metrics_report",
          [](const Run& run, const RuntimeMetricsOptions& options, RuntimeMetricsFormat format) {
            return run.metrics_report(options, format);
          },
          "options"_a = RuntimeMetricsOptions{}, "format"_a = RuntimeMetricsFormat::Text)
      .def("start_measurement", &Run::start_measurement, "options"_a = MeasureOptions{})
      .def("report", &Run::report, "options"_a = RunReportOptions{})
      .def("last_error", &Run::last_error)
      .def("diagnostics_summary", &Run::diagnostics_summary)
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

  m.def("build_graph_metrics_report_run_lifetime",
        &simaai::neat::build_graph_metrics_report_run_lifetime, "run"_a,
        "options"_a = RuntimeMetricsOptions{});

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
          [](Graph& self, nb::object input, RunMode mode, const RunOptions& options, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "Graph.build");
            if (python_sequence_all_samples(input)) {
              auto samples = sample_batch_from_python_input(input);
              nb::gil_scoped_release release;
              return self.build(samples, mode, options);
            }
            auto tensors = tensor_batch_from_python_input(input, copy, layout, image_format);
            nb::gil_scoped_release release;
            return self.build(tensors, mode, options);
          },
          "input"_a, "mode"_a = RunMode::Async, "options"_a = RunOptions{}, "copy"_a = false,
          "layout"_a = nb::none(), "image_format"_a = nb::none())
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

  nb::class_<simaai::neat::InputOptions>(m, "InputOptions")
      .def(nb::init<>())
      .def_rw("payload_type", &simaai::neat::InputOptions::payload_type)
      .def_prop_rw(
          "format", [](const simaai::neat::InputOptions& options) { return options.format.str(); },
          [](simaai::neat::InputOptions& options, nb::handle value) {
            options.format = python_to_format_spec(value);
          },
          "value"_a.none())
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
      .def_rw("use_simaai_pool", &simaai::neat::InputOptions::use_simaai_pool)
      .def_rw("pool_min_buffers", &simaai::neat::InputOptions::pool_min_buffers)
      .def_rw("pool_max_buffers", &simaai::neat::InputOptions::pool_max_buffers)
      .def_rw("memory_policy", &simaai::neat::InputOptions::memory_policy)
      .def_rw("buffer_name", &simaai::neat::InputOptions::buffer_name);

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

  nb::module_ graphs_mod = m.def_submodule("graphs", "Reusable public Graph fragment helpers");
  graphs_mod.def("branch", &simaai::neat::graphs::Branch, "input"_a, "outputs"_a);
  graphs_mod.def("combine", &simaai::neat::graphs::Combine, "inputs"_a, "output"_a,
                 "policy"_a = simaai::neat::CombinePolicy::ByFrame);

  nb::class_<simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps>(
      m, "ImageInputGroupOutputCaps")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::enable)
      .def_rw("format", &simaai::neat::nodes::groups::ImageInputGroupOptions::OutputCaps::format)
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
      .def_rw("format", &simaai::neat::nodes::groups::VideoInputGroupOptions::OutputCaps::format)
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
      .def_rw("out_format", &simaai::neat::nodes::groups::VideoInputGroupOptions::out_format)
      .def_rw("use_videoconvert",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::use_videoconvert)
      .def_rw("use_videoscale",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::use_videoscale)
      .def_rw("output_caps", &simaai::neat::nodes::groups::VideoInputGroupOptions::output_caps)
      .def_rw("extra_fragment",
              &simaai::neat::nodes::groups::VideoInputGroupOptions::extra_fragment);

  nb::class_<simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps>(
      m, "RtspDecodedInputOutputCaps")
      .def(nb::init<>())
      .def_rw("enable", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::enable)
      .def_rw("format", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::format)
      .def_rw("width", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::width)
      .def_rw("height", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::height)
      .def_rw("fps", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::fps)
      .def_rw("memory", &simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps::memory);

  nb::class_<simaai::neat::nodes::groups::RtspDecodedInputOptions>(m, "RtspDecodedInputOptions")
      .def(nb::init<>())
      .def_rw("url", &simaai::neat::nodes::groups::RtspDecodedInputOptions::url)
      .def_rw("latency_ms", &simaai::neat::nodes::groups::RtspDecodedInputOptions::latency_ms)
      .def_rw("tcp", &simaai::neat::nodes::groups::RtspDecodedInputOptions::tcp)
      .def_rw("payload_type", &simaai::neat::nodes::groups::RtspDecodedInputOptions::payload_type)
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
      .def_rw("out_format", &simaai::neat::nodes::groups::RtspDecodedInputOptions::out_format)
      .def_rw("decoder_name", &simaai::neat::nodes::groups::RtspDecodedInputOptions::decoder_name)
      .def_rw("decoder_raw_output",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::decoder_raw_output)
      .def_rw("decoder_next_element",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::decoder_next_element)
      .def_rw("use_videoconvert",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::use_videoconvert)
      .def_rw("use_videoscale",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::use_videoscale)
      .def_rw("output_caps", &simaai::neat::nodes::groups::RtspDecodedInputOptions::output_caps)
      .def_rw("extra_fragment",
              &simaai::neat::nodes::groups::RtspDecodedInputOptions::extra_fragment);

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

  nb::module_ groups_mod = m.def_submodule("groups", "Reusable Graph fragment helpers");
  groups_mod.def("image_input", &simaai::neat::nodes::groups::ImageInputGroup, "options"_a);
  groups_mod.def("video_input", &simaai::neat::nodes::groups::VideoInputGroup, "options"_a);
  groups_mod.def("rtsp_decoded_input", &simaai::neat::nodes::groups::RtspDecodedInput, "options"_a);
  groups_mod.def("udp_h264_output_group", &simaai::neat::nodes::groups::UdpH264OutputGroup,
                 "options"_a);
  groups_mod.def("video_sender", &simaai::neat::nodes::groups::VideoSender, "options"_a);
  groups_mod.def("mla", &simaai::neat::nodes::groups::MLA, "model"_a);
  groups_mod.def("image_input_output_spec", &simaai::neat::nodes::groups::ImageInputGroupOutputSpec,
                 "options"_a);
  groups_mod.def("video_input_output_spec", &simaai::neat::nodes::groups::VideoInputGroupOutputSpec,
                 "options"_a);
  groups_mod.def("rtsp_decoded_output_spec",
                 &simaai::neat::nodes::groups::RtspDecodedInputOutputSpec, "options"_a);

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
      .def_rw("processmla", &simaai::neat::Model::Options::processmla);

  nb::class_<simaai::neat::Model::RouteOptions>(m, "ModelRouteOptions")
      .def(nb::init<>())
      .def_rw("include_input", &simaai::neat::Model::RouteOptions::include_input)
      .def_rw("include_output", &simaai::neat::Model::RouteOptions::include_output)
      .def_rw("expose_all_outputs", &simaai::neat::Model::RouteOptions::expose_all_outputs)
      .def_rw("upstream_name", &simaai::neat::Model::RouteOptions::upstream_name)
      .def_rw("name_suffix", &simaai::neat::Model::RouteOptions::name_suffix)
      .def_rw("buffer_name", &simaai::neat::Model::RouteOptions::buffer_name);

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
      .def("warmup", &simaai::neat::Model::Runner::warmup, "inputs"_a, "warm"_a = -1,
           "timeout_ms"_a = -1)
      .def(
          "warmup",
          [](simaai::neat::Model::Runner& runner, nb::object input, int warm, int timeout_ms,
             bool copy, std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            reject_single_tensor_or_sample(input, "ModelRunner.warmup");
            auto tensors = tensor_batch_from_python_input(input, copy, layout, image_format);
            return runner.warmup(tensors, warm, timeout_ms);
          },
          "input"_a, "warm"_a = -1, "timeout_ms"_a = -1, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def("close", &simaai::neat::Model::Runner::close);

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
      .def("input_spec", &simaai::neat::Model::input_spec)
      .def("output_spec", &simaai::neat::Model::output_spec)
      .def("metadata", &simaai::neat::Model::metadata)
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
          "input"_a, "timeout_ms"_a = -1, "copy"_a = false);

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
  nodes_mod.def("h264_decode", &simaai::neat::nodes::H264Decode, "sima_allocator_type"_a = 2,
                "out_format"_a = "NV12", "decoder_name"_a = "", "raw_output"_a = false,
                "next_element"_a = "", "dec_width"_a = -1, "dec_height"_a = -1, "dec_fps"_a = -1,
                "num_buffers"_a = -1);
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

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "builder/OutputSpec.h"
#include "graph/Graph.h"
#include "graph/GraphPrinter.h"
#include "graph/GraphRun.h"
#include "graph/GraphSession.h"
#include "graph/Node.h"
#include "graph/nodes/FanOut.h"
#include "graph/nodes/JoinBundle.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageModelExecutor.h"
#include "graph/nodes/StampFrameId.h"
#include "graph/nodes/StreamScheduler.h"
#include "model/Model.h"
#include "mpk/MpKLoader.h"
#include "mpk/MpKPipelineAdapter.h"
#include "mpk/PipelineSequence.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/groups/UdpH264OutputGroup.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/groups/GroupOutputSpec.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/VideoInputGroup.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/io/Input.h"
#include "nodes/io/MetadataSender.h"
#include "nodes/io/UdpOutput.h"
#include "pipeline/Run.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Session.h"
#include "pipeline/SessionError.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
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
using simaai::neat::ImageSpec;
using simaai::neat::MapMode;
using simaai::neat::OutputMemory;
using simaai::neat::PullError;
using simaai::neat::PullStatus;
using simaai::neat::Run;
using simaai::neat::RunAdvancedOptions;
using simaai::neat::RunDiagSnapshot;
using simaai::neat::RunElementFlowStats;
using simaai::neat::RunElementTimingStats;
using simaai::neat::RunMode;
using simaai::neat::RunOptions;
using simaai::neat::RunPreset;
using simaai::neat::RunReportOptions;
using simaai::neat::RunStageStats;
using simaai::neat::RunStats;
using simaai::neat::Sample;
using simaai::neat::SampleKind;
using simaai::neat::Session;
using simaai::neat::SessionError;
using simaai::neat::SessionOptions;
using simaai::neat::SessionReport;
using simaai::neat::Tensor;
using simaai::neat::TensorConstraint;
using simaai::neat::TensorDType;
using simaai::neat::TensorLayout;
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

std::string format_session_error_message(const SessionError& e) {
  std::string msg = e.what() ? std::string(e.what()) : std::string{};
  const SessionReport& rep = e.report();

  if (msg.empty() || msg == "[" || msg == "[]") {
    if (!rep.error_code.empty() && !rep.repro_note.empty()) {
      return "[" + rep.error_code + "] " + rep.repro_note;
    }
    if (!rep.error_code.empty()) {
      return "[" + rep.error_code + "] SessionError";
    }
    if (!rep.repro_note.empty()) {
      return rep.repro_note;
    }
    return "SessionError";
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

} // namespace

NB_MODULE(_pyneat_core, m) {
  m.doc() = "Python bindings for SiMa NEAT";

  nb::exception<SessionError> py_session_error(m, "SessionError");
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        try {
          std::rethrow_exception(p);
        } catch (const SessionError& e) {
          PyObject* exc_type = reinterpret_cast<PyObject*>(payload);
          if (!exc_type) {
            throw;
          }

          const SessionReport& rep = e.report();
          const std::string msg = format_session_error_message(e);
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
      py_session_error.ptr());
  nb::exception<simaai::neat::mpk::MpKError>(m, "MpKError");

  m.attr("__version__") = "0.1.0";
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

  nb::class_<simaai::neat::AudioSpec>(m, "AudioSpec")
      .def(nb::init<>())
      .def_rw("sample_rate", &simaai::neat::AudioSpec::sample_rate)
      .def_rw("channels", &simaai::neat::AudioSpec::channels)
      .def_rw("interleaved", &simaai::neat::AudioSpec::interleaved);

  nb::class_<simaai::neat::TokensSpec>(m, "TokensSpec")
      .def(nb::init<>())
      .def_rw("vocab_size", &simaai::neat::TokensSpec::vocab_size);

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

  nb::class_<simaai::neat::Semantic>(m, "Semantic")
      .def(nb::init<>())
      .def_rw("image", &simaai::neat::Semantic::image)
      .def_rw("audio", &simaai::neat::Semantic::audio)
      .def_rw("tokens", &simaai::neat::Semantic::tokens)
      .def_rw("byte_stream", &simaai::neat::Semantic::byte_stream)
      .def_rw("tess", &simaai::neat::Semantic::tess)
      .def_rw("encoded", &simaai::neat::Semantic::encoded)
      .def_rw("quant", &simaai::neat::Semantic::quant);

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
      .def("image_format", &tensor_image_format_value)
      .def("is_nv12", &simaai::neat::Tensor::is_nv12)
      .def("is_i420", &simaai::neat::Tensor::is_i420)
      .def("debug_string", &simaai::neat::Tensor::debug_string)
      .def("__repr__",
           [](const simaai::neat::Tensor& t) { return "Tensor(" + t.debug_string() + ")"; })
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

  nb::class_<SessionReport>(m, "SessionReport")
      .def(nb::init<>())
      .def_rw("pipeline_string", &SessionReport::pipeline_string)
      .def_rw("error_code", &SessionReport::error_code)
      .def_rw("nodes", &SessionReport::nodes)
      .def_rw("bus", &SessionReport::bus)
      .def_rw("boundaries", &SessionReport::boundaries)
      .def_rw("caps_dump", &SessionReport::caps_dump)
      .def_rw("dot_paths", &SessionReport::dot_paths)
      .def_rw("repro_gst_launch", &SessionReport::repro_gst_launch)
      .def_rw("repro_env", &SessionReport::repro_env)
      .def_rw("repro_note", &SessionReport::repro_note)
      .def_rw("has_build_adaptation", &SessionReport::has_build_adaptation)
      .def_rw("build_adaptation", &SessionReport::build_adaptation)
      .def("to_json", &SessionReport::to_json);

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
      .def_rw("media_type", &Sample::media_type)
      .def_rw("payload_tag", &Sample::payload_tag)
      .def_rw("format", &Sample::format)
      .def_rw("frame_id", &Sample::frame_id)
      .def_rw("stream_id", &Sample::stream_id)
      .def_rw("port_name", &Sample::port_name)
      .def_rw("output_index", &Sample::output_index)
      .def_rw("input_seq", &Sample::input_seq)
      .def_rw("orig_input_seq", &Sample::orig_input_seq)
      .def_rw("pts_ns", &Sample::pts_ns)
      .def_rw("dts_ns", &Sample::dts_ns)
      .def_rw("duration_ns", &Sample::duration_ns);

  m.def("make_tensor_sample", &simaai::neat::make_tensor_sample, "port_name"_a, "tensor"_a);

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

  nb::class_<SessionOptions>(m, "SessionOptions")
      .def(nb::init<>())
      .def_rw("callback_timeout_ms", &SessionOptions::callback_timeout_ms)
      .def_rw("element_name_prefix", &SessionOptions::element_name_prefix)
      .def_rw("element_name_suffix", &SessionOptions::element_name_suffix);

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
      .def_rw("advanced", &RunOptions::advanced)
      .def_rw("on_input_drop", &RunOptions::on_input_drop);

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

  nb::class_<RunDiagSnapshot>(m, "RunDiagSnapshot")
      .def(nb::init<>())
      .def_rw("stages", &RunDiagSnapshot::stages)
      .def_rw("boundaries", &RunDiagSnapshot::boundaries)
      .def_rw("element_timings", &RunDiagSnapshot::element_timings)
      .def_rw("element_flows", &RunDiagSnapshot::element_flows);

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
      .def_rw("include_system_info", &RunReportOptions::include_system_info);

  nb::class_<Run>(m, "Run")
      .def(nb::init<>())
      .def("__bool__", [](const Run& run) { return static_cast<bool>(run); })
      .def("can_push", &Run::can_push)
      .def("can_pull", &Run::can_pull)
      .def("running", &Run::running)
      .def(
          "push_tensor",
          [](Run& run, const Tensor& input) { return run.push(simaai::neat::TensorList{input}); },
          "input"_a)
      .def(
          "try_push_tensor",
          [](Run& run, const Tensor& input) {
            return run.try_push(simaai::neat::TensorList{input});
          },
          "input"_a)
      .def(
          "push_sample",
          [](Run& run, const Sample& input) { return run.push(simaai::neat::SampleList{input}); },
          "input"_a)
      .def(
          "try_push_sample",
          [](Run& run, const Sample& input) {
            return run.try_push(simaai::neat::SampleList{input});
          },
          "input"_a)
      .def(
          "push",
          [](Run& run, nb::object input, bool copy, std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            if (nb::isinstance<Sample>(input)) {
              return run.push(simaai::neat::SampleList{nb::cast<Sample>(input)});
            }
            return run.push(simaai::neat::TensorList{
                tensor_from_python_input(input, copy, layout, image_format)});
          },
          "input"_a, "copy"_a = false, "layout"_a = nb::none(), "image_format"_a = nb::none())
      .def(
          "try_push",
          [](Run& run, nb::object input, bool copy, std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            if (nb::isinstance<Sample>(input)) {
              return run.try_push(simaai::neat::SampleList{nb::cast<Sample>(input)});
            }
            return run.try_push(simaai::neat::TensorList{
                tensor_from_python_input(input, copy, layout, image_format)});
          },
          "input"_a, "copy"_a = false, "layout"_a = nb::none(), "image_format"_a = nb::none())
      .def("close_input", &Run::close_input)
      .def("pull", static_cast<std::optional<Sample> (Run::*)(int)>(&Run::pull),
           "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("pull_tensors", &Run::pull_tensors, "timeout_ms"_a = -1,
           nb::call_guard<nb::gil_scoped_release>())
      .def("pull_samples", &Run::pull_samples, "timeout_ms"_a = -1,
           nb::call_guard<nb::gil_scoped_release>())
      .def("run_tensors",
           static_cast<simaai::neat::TensorList (Run::*)(const simaai::neat::TensorList&, int)>(
               &Run::run),
           "inputs"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("run_samples",
           static_cast<simaai::neat::SampleList (Run::*)(const simaai::neat::SampleList&, int)>(
               &Run::run),
           "inputs"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "run",
          [](Run& run, nb::object input, int timeout_ms, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) -> nb::object {
            if (nb::isinstance<Sample>(input)) {
              auto sample = nb::cast<Sample>(input);
              simaai::neat::SampleList out;
              {
                nb::gil_scoped_release release;
                out = run.run(simaai::neat::SampleList{sample}, timeout_ms);
              }
              return nb::cast(std::move(out));
            }
            auto tensor = tensor_from_python_input(input, copy, layout, image_format);
            simaai::neat::TensorList out;
            {
              nb::gil_scoped_release release;
              out = run.run(simaai::neat::TensorList{tensor}, timeout_ms);
            }
            return nb::cast(std::move(out));
          },
          "input"_a, "timeout_ms"_a = -1, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def("stats", &Run::stats)
      .def("input_stats", &Run::input_stats)
      .def("diag_snapshot", &Run::diag_snapshot)
      .def("report", &Run::report, "options"_a = RunReportOptions{})
      .def("last_error", &Run::last_error)
      .def("diagnostics_summary", &Run::diagnostics_summary)
      .def("stop", &Run::stop)
      .def("close", &Run::close);

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

  nb::class_<simaai::neat::NodeGroup>(m, "NodeGroup")
      .def(nb::init<>())
      .def(nb::init<const std::vector<std::shared_ptr<simaai::neat::Node>>&>(), "nodes"_a)
      .def("nodes", [](const simaai::neat::NodeGroup& g) { return g.nodes(); })
      .def("empty", &simaai::neat::NodeGroup::empty)
      .def("size", &simaai::neat::NodeGroup::size)
      .def("caps_behavior", &simaai::neat::NodeGroup::caps_behavior)
      .def("is_static_group", &simaai::neat::NodeGroup::is_static_group);

  nb::class_<Session>(m, "Session")
      .def(nb::init<const SessionOptions&>(), "options"_a = SessionOptions{})
      .def(
          "add_node",
          [](Session& self, const std::shared_ptr<simaai::neat::Node>& node) -> Session& {
            return self.add(node);
          },
          "node"_a, nb::rv_policy::reference_internal)
      .def(
          "add_group",
          [](Session& self, const simaai::neat::NodeGroup& group) -> Session& {
            return self.add(group);
          },
          "group"_a, nb::rv_policy::reference_internal)
      .def("custom", static_cast<Session& (Session::*)(std::string)>(&Session::custom),
           "fragment"_a, nb::rv_policy::reference_internal)
      .def("custom_with_role",
           static_cast<Session& (Session::*)(std::string, simaai::neat::InputRole)>(
               &Session::custom),
           "fragment"_a, "role"_a, nb::rv_policy::reference_internal)
      .def("run_source", static_cast<void (Session::*)()>(&Session::run),
           nb::call_guard<nb::gil_scoped_release>())
      .def("run_tensors",
           static_cast<simaai::neat::TensorList (Session::*)(const simaai::neat::TensorList&,
                                                             const RunOptions&)>(&Session::run),
           "inputs"_a, "options"_a = RunOptions{}, nb::call_guard<nb::gil_scoped_release>())
      .def("run_samples",
           static_cast<simaai::neat::SampleList (Session::*)(const simaai::neat::SampleList&,
                                                             const RunOptions&)>(&Session::run),
           "inputs"_a, "options"_a = RunOptions{}, nb::call_guard<nb::gil_scoped_release>())
      .def("build_tensors",
           static_cast<Run (Session::*)(const simaai::neat::TensorList&, RunMode,
                                        const RunOptions&)>(&Session::build),
           "inputs"_a, "mode"_a = RunMode::Async, "options"_a = RunOptions{},
           nb::call_guard<nb::gil_scoped_release>())
      .def("build_samples",
           static_cast<Run (Session::*)(const simaai::neat::SampleList&, RunMode,
                                        const RunOptions&)>(&Session::build),
           "inputs"_a, "mode"_a = RunMode::Async, "options"_a = RunOptions{},
           nb::call_guard<nb::gil_scoped_release>())
      .def("build_source", static_cast<Run (Session::*)(const RunOptions&)>(&Session::build),
           "options"_a = RunOptions{}, nb::call_guard<nb::gil_scoped_release>())
      .def("build", static_cast<Run (Session::*)(const RunOptions&)>(&Session::build),
           "options"_a = RunOptions{}, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "build",
          [](Session& self, nb::object input, RunMode mode, const RunOptions& options, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            if (nb::isinstance<Sample>(input)) {
              auto sample = nb::cast<Sample>(input);
              nb::gil_scoped_release release;
              return self.build(simaai::neat::SampleList{sample}, mode, options);
            }
            auto tensor = tensor_from_python_input(input, copy, layout, image_format);
            nb::gil_scoped_release release;
            return self.build(simaai::neat::TensorList{tensor}, mode, options);
          },
          "input"_a, "mode"_a = RunMode::Async, "options"_a = RunOptions{}, "copy"_a = false,
          "layout"_a = nb::none(), "image_format"_a = nb::none())
      .def("run_rtsp", &Session::run_rtsp, "options"_a, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "validate",
          static_cast<SessionReport (Session::*)(const ValidateOptions&) const>(&Session::validate),
          "options"_a = ValidateOptions{})
      .def("add_output_tensor", &Session::add_output_tensor,
           "options"_a = simaai::neat::OutputTensorOptions{}, nb::rv_policy::reference_internal)
      .def("describe", [](const Session& self) { return self.describe(); })
      .def("describe_backend", &Session::describe_backend, "insert_boundaries"_a = false)
      .def("save", &Session::save, "path"_a)
      .def_static("load", &Session::load, "path"_a)
      .def_prop_ro("last_pipeline", &Session::last_pipeline);

  nb::class_<simaai::neat::OutputSpec>(m, "OutputSpec")
      .def(nb::init<>())
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
      .def_rw("media_type", &simaai::neat::InputOptions::media_type)
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
      .def_rw("buffer_name", &simaai::neat::InputOptions::buffer_name);

  nb::class_<simaai::neat::OutputOptions>(m, "OutputOptions")
      .def(nb::init<>())
      .def_rw("max_buffers", &simaai::neat::OutputOptions::max_buffers)
      .def_rw("drop", &simaai::neat::OutputOptions::drop)
      .def_rw("sync", &simaai::neat::OutputOptions::sync)
      .def_static("latest", &simaai::neat::OutputOptions::Latest)
      .def_static("every_frame", &simaai::neat::OutputOptions::EveryFrame, "max_buffers"_a = 30)
      .def_static("clocked", &simaai::neat::OutputOptions::Clocked, "max_buffers"_a = 1);

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
            if (!ok && !err.empty())
              throw std::runtime_error(err);
            return ok;
          },
          "payload"_a)
      .def(
          "send_metadata",
          [](const simaai::neat::MetadataSender& self, const std::string& type,
             const std::string& data_json, int64_t timestamp_ms, const std::string& frame_id) {
            std::string err;
            const bool ok = self.send_metadata(type, data_json, timestamp_ms, frame_id, &err);
            if (!ok && !err.empty())
              throw std::runtime_error(err);
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

  nb::module_ nodes_mod = m.def_submodule("nodes", "Node factory helpers");
  nodes_mod.def("queue", &simaai::neat::nodes::Queue);
  nodes_mod.def("rtsp_input", &simaai::neat::nodes::RTSPInput, "url"_a, "latency_ms"_a = 200,
                "tcp"_a = true, "drop_on_latency"_a = false, "buffer_mode"_a = "");
  nodes_mod.def("h264_depacketize", &simaai::neat::nodes::H264Depacketize, "payload_type"_a = 96,
                "h264_parse_config_interval"_a = -1, "h264_fps"_a = -1, "h264_width"_a = -1,
                "h264_height"_a = -1, "enforce_h264_caps"_a = true);
  nodes_mod.def("input", &simaai::neat::nodes::Input, "options"_a = simaai::neat::InputOptions{});
  nodes_mod.def("output", &simaai::neat::nodes::Output,
                "options"_a = simaai::neat::OutputOptions{});
  nodes_mod.def("video_convert", &simaai::neat::nodes::VideoConvert);

  nb::module_ groups_mod = m.def_submodule("groups", "NodeGroup factory helpers");
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
      .value("Detr", simaai::neat::BoxDecodeType::Detr)
      .value("EffDet", simaai::neat::BoxDecodeType::EffDet)
      .value("RcnnStage1", simaai::neat::BoxDecodeType::RcnnStage1)
      .value("Centernet", simaai::neat::BoxDecodeType::Centernet);

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
      .def_rw("score_threshold", &simaai::neat::Model::Options::score_threshold)
      .def_rw("nms_iou_threshold", &simaai::neat::Model::Options::nms_iou_threshold)
      .def_rw("top_k", &simaai::neat::Model::Options::top_k)
      .def_rw("boxdecode_original_width", &simaai::neat::Model::Options::boxdecode_original_width)
      .def_rw("boxdecode_original_height", &simaai::neat::Model::Options::boxdecode_original_height)
      .def_rw("upstream_name", &simaai::neat::Model::Options::upstream_name)
      .def_rw("name_suffix", &simaai::neat::Model::Options::name_suffix)
      .def_rw("cleanup_extracted_model_data",
              &simaai::neat::Model::Options::cleanup_extracted_model_data)
      .def_rw("verbose", &simaai::neat::Model::Options::verbose)
      .def_rw("inference_terminal", &simaai::neat::Model::Options::inference_terminal)
      .def_rw("processcvu", &simaai::neat::Model::Options::processcvu)
      .def_rw("processmla", &simaai::neat::Model::Options::processmla);

  nb::class_<simaai::neat::Model::SessionOptions>(m, "ModelSessionOptions")
      .def(nb::init<>())
      .def_rw("include_appsrc", &simaai::neat::Model::SessionOptions::include_appsrc)
      .def_rw("include_appsink", &simaai::neat::Model::SessionOptions::include_appsink)
      .def_rw("upstream_name", &simaai::neat::Model::SessionOptions::upstream_name)
      .def_rw("name_suffix", &simaai::neat::Model::SessionOptions::name_suffix)
      .def_rw("buffer_name", &simaai::neat::Model::SessionOptions::buffer_name);

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
           static_cast<bool (simaai::neat::Model::Runner::*)(const simaai::neat::SampleList&)>(
               &simaai::neat::Model::Runner::push),
           "inputs"_a)
      .def(
          "push_tensor",
          [](simaai::neat::Model::Runner& runner, const Tensor& input) {
            return runner.push(simaai::neat::TensorList{input});
          },
          "input"_a)
      .def(
          "push_sample",
          [](simaai::neat::Model::Runner& runner, const Sample& input) {
            return runner.push(simaai::neat::SampleList{input});
          },
          "input"_a)
      .def(
          "push",
          [](simaai::neat::Model::Runner& runner, nb::object input, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) {
            if (nb::isinstance<Sample>(input)) {
              return runner.push(simaai::neat::SampleList{nb::cast<Sample>(input)});
            }
            return runner.push(simaai::neat::TensorList{
                tensor_from_python_input(input, copy, layout, image_format)});
          },
          "input"_a, "copy"_a = false, "layout"_a = nb::none(), "image_format"_a = nb::none())
      .def("pull", &simaai::neat::Model::Runner::pull, "timeout_ms"_a = -1,
           nb::call_guard<nb::gil_scoped_release>())
      .def("run_tensors",
           static_cast<simaai::neat::TensorList (simaai::neat::Model::Runner::*)(
               const simaai::neat::TensorList&, int)>(&simaai::neat::Model::Runner::run),
           "inputs"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("run_samples",
           static_cast<simaai::neat::SampleList (simaai::neat::Model::Runner::*)(
               const simaai::neat::SampleList&, int)>(&simaai::neat::Model::Runner::run),
           "inputs"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "run",
          [](simaai::neat::Model::Runner& runner, nb::object input, int timeout_ms, bool copy,
             std::optional<TensorLayout> layout,
             std::optional<ImageSpec::PixelFormat> image_format) -> nb::object {
            if (nb::isinstance<Sample>(input)) {
              auto sample = nb::cast<Sample>(input);
              simaai::neat::SampleList out;
              {
                nb::gil_scoped_release release;
                out = runner.run(simaai::neat::SampleList{sample}, timeout_ms);
              }
              return nb::cast(std::move(out));
            }
            auto tensor = tensor_from_python_input(input, copy, layout, image_format);
            simaai::neat::TensorList out;
            {
              nb::gil_scoped_release release;
              out = runner.run(simaai::neat::TensorList{tensor}, timeout_ms);
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
            auto tensor = tensor_from_python_input(input, copy, layout, image_format);
            return runner.warmup(simaai::neat::TensorList{tensor}, warm, timeout_ms);
          },
          "input"_a, "warm"_a = -1, "timeout_ms"_a = -1, "copy"_a = false, "layout"_a = nb::none(),
          "image_format"_a = nb::none())
      .def("close", &simaai::neat::Model::Runner::close);

  nb::class_<simaai::neat::Model>(m, "Model")
      .def(nb::init<const std::string&>(), "mpk_path"_a)
      .def(nb::init<const std::string&, const simaai::neat::Model::Options&>(), "mpk_path"_a,
           "options"_a)
      .def("preprocess", &simaai::neat::Model::preprocess)
      .def("inference", &simaai::neat::Model::inference)
      .def("postprocess", &simaai::neat::Model::postprocess)
      .def("session_group", static_cast<simaai::neat::NodeGroup (simaai::neat::Model::*)() const>(
                                &simaai::neat::Model::session))
      .def("session_group_with_options",
           static_cast<simaai::neat::NodeGroup (simaai::neat::Model::*)(
               simaai::neat::Model::SessionOptions) const>(&simaai::neat::Model::session),
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
      .def("build_with_session_options",
           static_cast<simaai::neat::Model::Runner (simaai::neat::Model::*)(
               const simaai::neat::Model::SessionOptions&)>(&simaai::neat::Model::build),
           "options"_a)
      .def("build_tensors",
           static_cast<simaai::neat::Model::Runner (simaai::neat::Model::*)(
               const simaai::neat::TensorList&, const simaai::neat::Model::SessionOptions&,
               const RunOptions&)>(&simaai::neat::Model::build),
           "inputs"_a, "session_options"_a = simaai::neat::Model::SessionOptions{},
           "run_options"_a = RunOptions{})
      .def("build_samples",
           static_cast<simaai::neat::Model::Runner (simaai::neat::Model::*)(
               const simaai::neat::SampleList&, const simaai::neat::Model::SessionOptions&,
               const RunOptions&)>(&simaai::neat::Model::build),
           "inputs"_a, "session_options"_a = simaai::neat::Model::SessionOptions{},
           "run_options"_a = RunOptions{})
      .def(
          "build",
          [](simaai::neat::Model& model, nb::object input,
             const simaai::neat::Model::SessionOptions& session_options,
             const RunOptions& run_options, bool copy) {
            if (nb::isinstance<Sample>(input)) {
              auto sample = nb::cast<Sample>(input);
              return model.build(simaai::neat::SampleList{sample}, session_options, run_options);
            }
            auto tensor = tensor_from_python_input(input, copy, std::nullopt, std::nullopt);
            return model.build(simaai::neat::TensorList{tensor}, session_options, run_options);
          },
          "input"_a, "session_options"_a = simaai::neat::Model::SessionOptions{},
          "run_options"_a = RunOptions{}, "copy"_a = false)
      .def("run_tensors",
           static_cast<simaai::neat::TensorList (simaai::neat::Model::*)(
               const simaai::neat::TensorList&, int)>(&simaai::neat::Model::run),
           "inputs"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("run_samples",
           static_cast<simaai::neat::SampleList (simaai::neat::Model::*)(
               const simaai::neat::SampleList&, int)>(&simaai::neat::Model::run),
           "inputs"_a, "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "run",
          [](simaai::neat::Model& model, nb::object input, int timeout_ms,
             bool copy) -> nb::object {
            if (nb::isinstance<Sample>(input)) {
              auto sample = nb::cast<Sample>(input);
              simaai::neat::SampleList out;
              {
                nb::gil_scoped_release release;
                out = model.run(simaai::neat::SampleList{sample}, timeout_ms);
              }
              return nb::cast(std::move(out));
            }
            if (PyList_Check(input.ptr()) || PyTuple_Check(input.ptr())) {
              auto batch = tensor_batch_from_python_input(input, copy, std::nullopt, std::nullopt);
              simaai::neat::TensorList out;
              {
                nb::gil_scoped_release release;
                out = model.run(batch, timeout_ms);
              }
              return nb::cast(std::move(out));
            }
            auto tensor = tensor_from_python_input(input, copy, std::nullopt, std::nullopt);
            simaai::neat::TensorList out;
            {
              nb::gil_scoped_release release;
              out = model.run(simaai::neat::TensorList{tensor}, timeout_ms);
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
      .def_rw("num_buffers_locked", &simaai::neat::PreprocOptions::num_buffers_locked);

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

  nb::module_ graph_mod = m.def_submodule("graph", "Hybrid graph runtime and helper nodes");

  nb::enum_<simaai::neat::graph::Backend>(graph_mod, "Backend")
      .value("Pipeline", simaai::neat::graph::Backend::Pipeline)
      .value("Stage", simaai::neat::graph::Backend::Stage);

  nb::class_<simaai::neat::graph::Node>(graph_mod, "Node")
      .def("kind", &simaai::neat::graph::Node::kind)
      .def("user_label", &simaai::neat::graph::Node::user_label)
      .def("backend", &simaai::neat::graph::Node::backend);

  nb::class_<simaai::neat::graph::Edge>(graph_mod, "Edge")
      .def(nb::init<>())
      .def_rw("from_node", &simaai::neat::graph::Edge::from)
      .def_rw("from_port", &simaai::neat::graph::Edge::from_port)
      .def_rw("to_node", &simaai::neat::graph::Edge::to)
      .def_rw("to_port", &simaai::neat::graph::Edge::to_port);

  nb::class_<simaai::neat::graph::Graph>(graph_mod, "Graph")
      .def(nb::init<>())
      .def("add", &simaai::neat::graph::Graph::add, "node"_a)
      .def("connect", &simaai::neat::graph::Graph::connect, "from_node"_a, "to_node"_a,
           "from_port"_a = "out", "to_port"_a = "in")
      .def("node_count", &simaai::neat::graph::Graph::node_count)
      .def("port_count", &simaai::neat::graph::Graph::port_count)
      .def("port_names", &simaai::neat::graph::Graph::port_names)
      .def("topo_order", &simaai::neat::graph::Graph::topo_order)
      .def("is_dag", &simaai::neat::graph::Graph::is_dag)
      .def("edges", [](const simaai::neat::graph::Graph& g) { return g.edges(); });

  graph_mod.def(
      "to_text",
      [](const simaai::neat::graph::Graph& g) {
        return simaai::neat::graph::GraphPrinter::to_text(g);
      },
      "graph"_a);
  graph_mod.def(
      "to_dot",
      [](const simaai::neat::graph::Graph& g) {
        return simaai::neat::graph::GraphPrinter::to_dot(g);
      },
      "graph"_a);

  nb::class_<simaai::neat::graph::GraphRunOptions>(graph_mod, "GraphRunOptions")
      .def(nb::init<>())
      .def_rw("edge_queue", &simaai::neat::graph::GraphRunOptions::edge_queue)
      .def_rw("push_timeout_ms", &simaai::neat::graph::GraphRunOptions::push_timeout_ms)
      .def_rw("pull_timeout_ms", &simaai::neat::graph::GraphRunOptions::pull_timeout_ms)
      .def_rw("pipeline", &simaai::neat::graph::GraphRunOptions::pipeline);

  nb::class_<simaai::neat::graph::GraphRun::Input>(graph_mod, "Input")
      .def("push", &simaai::neat::graph::GraphRun::Input::push, "sample"_a,
           nb::call_guard<nb::gil_scoped_release>());

  nb::class_<simaai::neat::graph::GraphRun::Output>(graph_mod, "Output")
      .def(
          "pull",
          [](const simaai::neat::graph::GraphRun::Output& output, int timeout_ms) {
            return output.pull(timeout_ms);
          },
          "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "pull_or_throw",
          [](const simaai::neat::graph::GraphRun::Output& output, int timeout_ms) {
            return output.pull_or_throw(timeout_ms);
          },
          "timeout_ms"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def("node_id", &simaai::neat::graph::GraphRun::Output::node_id);

  nb::class_<simaai::neat::graph::GraphRun>(graph_mod, "GraphRun")
      .def(nb::init<>())
      .def("__bool__",
           [](const simaai::neat::graph::GraphRun& run) { return static_cast<bool>(run); })
      .def("running", &simaai::neat::graph::GraphRun::running)
      .def("push",
           nb::overload_cast<simaai::neat::graph::NodeId, const Sample&>(
               &simaai::neat::graph::GraphRun::push),
           "node_id"_a, "sample"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("push_port",
           nb::overload_cast<simaai::neat::graph::NodeId, simaai::neat::graph::PortId,
                             const Sample&>(&simaai::neat::graph::GraphRun::push),
           "node_id"_a, "port_id"_a, "sample"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("pull", &simaai::neat::graph::GraphRun::pull, "node_id"_a, "timeout_ms"_a = -1,
           nb::call_guard<nb::gil_scoped_release>())
      .def("input",
           nb::overload_cast<simaai::neat::graph::NodeId>(&simaai::neat::graph::GraphRun::input),
           "node_id"_a, nb::keep_alive<0, 1>())
      .def("input_port",
           nb::overload_cast<simaai::neat::graph::NodeId, simaai::neat::graph::PortId>(
               &simaai::neat::graph::GraphRun::input),
           "node_id"_a, "port_id"_a, nb::keep_alive<0, 1>())
      .def("output", &simaai::neat::graph::GraphRun::output, "node_id"_a, nb::keep_alive<0, 1>())
      .def("describe", &simaai::neat::graph::GraphRun::describe)
      .def("stop", &simaai::neat::graph::GraphRun::stop)
      .def("last_error", &simaai::neat::graph::GraphRun::last_error)
      .def("last_error_or_throw", &simaai::neat::graph::GraphRun::last_error_or_throw);

  nb::class_<simaai::neat::graph::GraphSession>(graph_mod, "GraphSession")
      .def(nb::init<simaai::neat::graph::Graph>(), "graph"_a)
      .def("build", &simaai::neat::graph::GraphSession::build,
           "options"_a = simaai::neat::graph::GraphRunOptions{},
           nb::call_guard<nb::gil_scoped_release>());

  nb::module_ graph_nodes_mod = graph_mod.def_submodule("nodes", "Graph node factory helpers");

  nb::enum_<simaai::neat::graph::nodes::StreamDropPolicy>(graph_nodes_mod, "StreamDropPolicy")
      .value("DropOldest", simaai::neat::graph::nodes::StreamDropPolicy::DropOldest)
      .value("DropNewest", simaai::neat::graph::nodes::StreamDropPolicy::DropNewest);

  nb::class_<simaai::neat::graph::nodes::StreamSchedulerOptions>(graph_nodes_mod,
                                                                 "StreamSchedulerOptions")
      .def(nb::init<>())
      .def_rw("per_stream_queue",
              &simaai::neat::graph::nodes::StreamSchedulerOptions::per_stream_queue)
      .def_rw("drop_policy", &simaai::neat::graph::nodes::StreamSchedulerOptions::drop_policy)
      .def_rw("max_batch", &simaai::neat::graph::nodes::StreamSchedulerOptions::max_batch);

  nb::class_<simaai::neat::graph::nodes::StageModelExecutorOptions>(graph_nodes_mod,
                                                                    "StageModelExecutorOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "model",
          [](const simaai::neat::graph::nodes::StageModelExecutorOptions& opt) {
            return std::const_pointer_cast<simaai::neat::Model>(opt.model);
          },
          [](simaai::neat::graph::nodes::StageModelExecutorOptions& opt,
             const std::shared_ptr<simaai::neat::Model>& model) { opt.model = model; })
      .def_rw("do_preproc", &simaai::neat::graph::nodes::StageModelExecutorOptions::do_preproc)
      .def_rw("do_mla", &simaai::neat::graph::nodes::StageModelExecutorOptions::do_mla)
      .def_rw("do_boxdecode", &simaai::neat::graph::nodes::StageModelExecutorOptions::do_boxdecode);

  graph_nodes_mod.def(
      "pipeline_node",
      [](const simaai::neat::NodeGroup& group, const std::string& label) {
        return std::static_pointer_cast<simaai::neat::graph::Node>(
            std::make_shared<simaai::neat::graph::nodes::PipelineNode>(group, label));
      },
      "group"_a, "label"_a = "");
  graph_nodes_mod.def(
      "pipeline_node",
      [](std::shared_ptr<simaai::neat::Node> node, const std::string& label) {
        return std::static_pointer_cast<simaai::neat::graph::Node>(
            std::make_shared<simaai::neat::graph::nodes::PipelineNode>(std::move(node), label));
      },
      "node"_a, "label"_a = "");
  graph_nodes_mod.def("stamp_frame_id", &simaai::neat::graph::nodes::StampFrameIdNode,
                      "label"_a = "");
  graph_nodes_mod.def(
      "stage_model_executor",
      [](const simaai::neat::graph::nodes::StageModelExecutorOptions& options,
         const std::string& label) {
        return simaai::neat::graph::nodes::StageModelExecutorNode(options, label);
      },
      "options"_a, "label"_a = "");
  graph_nodes_mod.def("stream_scheduler", &simaai::neat::graph::nodes::StreamSchedulerNode,
                      "options"_a = simaai::neat::graph::nodes::StreamSchedulerOptions{},
                      "label"_a = "", "input"_a = "in", "output"_a = "out");
  graph_nodes_mod.def("fan_out", &simaai::neat::graph::nodes::FanOutNode, "outputs"_a,
                      "label"_a = "", "input"_a = "in");
  graph_nodes_mod.def(
      "join_bundle",
      [](const std::vector<std::string>& inputs, const std::string& label,
         const std::string& output) {
        return simaai::neat::graph::nodes::JoinBundleNode(inputs, label, output);
      },
      "inputs"_a, "label"_a = "", "output"_a = "bundle");

  nb::module_ mpk_mod = m.def_submodule("mpk", "MPK inspection and sequence helpers");

  nb::enum_<simaai::neat::mpk::ErrorClass>(mpk_mod, "ErrorClass")
      .value("InvalidArchive", simaai::neat::mpk::ErrorClass::InvalidArchive)
      .value("PathTraversal", simaai::neat::mpk::ErrorClass::PathTraversal)
      .value("SchemaError", simaai::neat::mpk::ErrorClass::SchemaError)
      .value("UnsupportedVersion", simaai::neat::mpk::ErrorClass::UnsupportedVersion)
      .value("SizeLimitExceeded", simaai::neat::mpk::ErrorClass::SizeLimitExceeded);

  nb::class_<simaai::neat::mpk::ArchiveEntry>(mpk_mod, "ArchiveEntry")
      .def(nb::init<>())
      .def_rw("path", &simaai::neat::mpk::ArchiveEntry::path)
      .def_rw("normalized_path", &simaai::neat::mpk::ArchiveEntry::normalized_path)
      .def_rw("type", &simaai::neat::mpk::ArchiveEntry::type)
      .def_rw("size_bytes", &simaai::neat::mpk::ArchiveEntry::size_bytes);

  nb::class_<simaai::neat::mpk::MpKManifest>(mpk_mod, "MpKManifest")
      .def(nb::init<>())
      .def_rw("archive_path", &simaai::neat::mpk::MpKManifest::archive_path)
      .def_rw("package_name", &simaai::neat::mpk::MpKManifest::package_name)
      .def_rw("version", &simaai::neat::mpk::MpKManifest::version)
      .def_rw("archive_size_bytes", &simaai::neat::mpk::MpKManifest::archive_size_bytes)
      .def_rw("has_pipeline_sequence", &simaai::neat::mpk::MpKManifest::has_pipeline_sequence)
      .def_rw("has_model_binary", &simaai::neat::mpk::MpKManifest::has_model_binary)
      .def_rw("entries", &simaai::neat::mpk::MpKManifest::entries);

  nb::class_<simaai::neat::mpk::MpKLoaderOptions>(mpk_mod, "MpKLoaderOptions")
      .def(nb::init<>())
      .def_rw("max_archive_bytes", &simaai::neat::mpk::MpKLoaderOptions::max_archive_bytes)
      .def_rw("max_entry_bytes", &simaai::neat::mpk::MpKLoaderOptions::max_entry_bytes)
      .def_rw("max_total_json_bytes", &simaai::neat::mpk::MpKLoaderOptions::max_total_json_bytes)
      .def_rw("max_entries", &simaai::neat::mpk::MpKLoaderOptions::max_entries)
      .def_rw("max_json_depth", &simaai::neat::mpk::MpKLoaderOptions::max_json_depth)
      .def_rw("require_pipeline_sequence",
              &simaai::neat::mpk::MpKLoaderOptions::require_pipeline_sequence)
      .def_rw("require_model_binary", &simaai::neat::mpk::MpKLoaderOptions::require_model_binary)
      .def_rw("reject_unsupported_file_types",
              &simaai::neat::mpk::MpKLoaderOptions::reject_unsupported_file_types)
      .def_rw("reject_duplicate_json_keys",
              &simaai::neat::mpk::MpKLoaderOptions::reject_duplicate_json_keys)
      .def_rw("reject_invalid_utf8_paths",
              &simaai::neat::mpk::MpKLoaderOptions::reject_invalid_utf8_paths)
      .def_rw("reject_unicode_path_confusables",
              &simaai::neat::mpk::MpKLoaderOptions::reject_unicode_path_confusables);

  nb::class_<simaai::neat::mpk::MpKExtractResult>(mpk_mod, "MpKExtractResult")
      .def(nb::init<>())
      .def_rw("package_root", &simaai::neat::mpk::MpKExtractResult::package_root)
      .def_rw("etc_dir", &simaai::neat::mpk::MpKExtractResult::etc_dir)
      .def_rw("lib_dir", &simaai::neat::mpk::MpKExtractResult::lib_dir)
      .def_rw("share_dir", &simaai::neat::mpk::MpKExtractResult::share_dir)
      .def_rw("manifest", &simaai::neat::mpk::MpKExtractResult::manifest);

  nb::class_<simaai::neat::mpk::SequenceEntry>(mpk_mod, "SequenceEntry")
      .def(nb::init<>())
      .def_rw("sequence_id", &simaai::neat::mpk::SequenceEntry::sequence_id)
      .def_rw("name", &simaai::neat::mpk::SequenceEntry::name)
      .def_rw("plugin_id", &simaai::neat::mpk::SequenceEntry::plugin_id)
      .def_rw("config_path", &simaai::neat::mpk::SequenceEntry::config_path)
      .def_rw("processor", &simaai::neat::mpk::SequenceEntry::processor)
      .def_rw("kernel", &simaai::neat::mpk::SequenceEntry::kernel);

  nb::class_<simaai::neat::mpk::SequenceSplit>(mpk_mod, "SequenceSplit")
      .def(nb::init<>())
      .def_rw("pre", &simaai::neat::mpk::SequenceSplit::pre)
      .def_rw("infer", &simaai::neat::mpk::SequenceSplit::infer)
      .def_rw("post", &simaai::neat::mpk::SequenceSplit::post);

  nb::class_<simaai::neat::mpk::MpKPipelineAdapterOptions>(mpk_mod, "MpKPipelineAdapterOptions")
      .def(nb::init<>())
      .def_rw("include_pre", &simaai::neat::mpk::MpKPipelineAdapterOptions::include_pre)
      .def_rw("include_infer", &simaai::neat::mpk::MpKPipelineAdapterOptions::include_infer)
      .def_rw("include_post", &simaai::neat::mpk::MpKPipelineAdapterOptions::include_post)
      .def_rw("mla_only", &simaai::neat::mpk::MpKPipelineAdapterOptions::mla_only);

  mpk_mod.def("inspect", &simaai::neat::mpk::MpKLoader::inspect, "archive_path"_a,
              "options"_a = simaai::neat::mpk::MpKLoaderOptions{});
  mpk_mod.def("extract", &simaai::neat::mpk::MpKLoader::extract, "archive_path"_a, "output_root"_a,
              "options"_a = simaai::neat::mpk::MpKLoaderOptions{});
  mpk_mod.def("load_pipeline_sequence", &simaai::neat::mpk::load_pipeline_sequence, "etc_dir"_a);
  mpk_mod.def("split_sequence_for_infer", &simaai::neat::mpk::split_sequence_for_infer,
              "sequence"_a);
  mpk_mod.def("is_pre_adapter_kernel", &simaai::neat::mpk::is_pre_adapter_kernel, "kernel"_a);
  mpk_mod.def("is_post_adapter_kernel", &simaai::neat::mpk::is_post_adapter_kernel, "kernel"_a);
  mpk_mod.def("adapt_pipeline_sequence",
              nb::overload_cast<const std::vector<simaai::neat::mpk::SequenceEntry>&,
                                const simaai::neat::mpk::MpKPipelineAdapterOptions&>(
                  &simaai::neat::mpk::MpKPipelineAdapter::adapt),
              "sequence"_a, "options"_a = simaai::neat::mpk::MpKPipelineAdapterOptions{});

  m.attr("ERROR_PIPELINE_SHAPE") = simaai::neat::error_codes::kPipelineShape;
  m.attr("ERROR_CAPS") = simaai::neat::error_codes::kCaps;
  m.attr("ERROR_INPUT_SHAPE") = simaai::neat::error_codes::kInputShape;
  m.attr("ERROR_PARSE_LAUNCH") = simaai::neat::error_codes::kParseLaunch;
  m.attr("ERROR_RUNTIME_PULL") = simaai::neat::error_codes::kRuntimePull;
  m.attr("ERROR_IO_PARSE") = simaai::neat::error_codes::kIoParse;
  m.attr("ERROR_IO_OPEN") = simaai::neat::error_codes::kIoOpen;
}

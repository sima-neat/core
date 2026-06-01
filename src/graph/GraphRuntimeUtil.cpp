#include "graph/GraphBuild.h"

#include "graph/nodes/PipelineNode.h"
#include "graph/runtime/BlockingQueue.h"
#include "graph/runtime/StageMailbox.h"
#include "internal/GraphRunState.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/TensorCore.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cctype>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace simaai::neat::graph {
using simaai::neat::pipeline_internal::env_bool;
using simaai::neat::pipeline_internal::env_int;

bool has_input_appsrc(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  for (const auto& n : nodes) {
    if (!n)
      continue;
    if (dynamic_cast<const simaai::neat::Input*>(n.get()))
      return true;
  }
  return false;
}

bool has_output_appsink(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  for (const auto& n : nodes) {
    if (!n)
      continue;
    if (dynamic_cast<const simaai::neat::Output*>(n.get()))
      return true;
  }
  return false;
}

bool has_internal_source(std::span<const std::shared_ptr<simaai::neat::Node>> nodes) {
  for (const auto& n : nodes) {
    if (!n)
      continue;
    if (n->input_role() == simaai::neat::InputRole::Source)
      return true;
  }
  return false;
}

bool tensor_is_zero_copy(const simaai::neat::Tensor& t) {
  return t.storage && t.storage->kind == simaai::neat::StorageKind::GstSample;
}

int zero_copy_backpressure_cap() {
  const char* env = std::getenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP");
  if (env && *env)
    return std::max(0, std::atoi(env));
  env = std::getenv("SIMA_GRAPH_ZERO_COPY_MAX_INFLIGHT");
  if (env && *env)
    return std::max(0, std::atoi(env));
  return 1;
}

std::size_t identity_map_capacity() {
  const int cap = env_int("SIMA_GRAPH_IDENTITY_MAP_CAPACITY", 1024);
  return (cap > 0) ? static_cast<std::size_t>(cap) : 0;
}

void force_copy_tensor(simaai::neat::Tensor& t) {
  if (!tensor_is_zero_copy(t))
    return;
  t = t.clone();
  t.read_only = false;
}

void force_copy_sample_if_zero_copy(Sample& sample) {
  if (sample.kind == SampleKind::TensorSet) {
    for (auto& tensor : sample.tensors) {
      force_copy_tensor(tensor);
    }
    sample.owned = true;
    return;
  }
  if (!sample_is_multi_output(sample))
    return;
  for (auto& field : sample.fields) {
    force_copy_sample_if_zero_copy(field);
  }
  sample.owned = true;
}

bool sample_has_zero_copy_tensor(const Sample& sample) {
  if (sample.kind == SampleKind::TensorSet) {
    for (const auto& tensor : sample.tensors) {
      if (tensor_is_zero_copy(tensor))
        return true;
    }
    return false;
  }
  if (!sample_is_multi_output(sample))
    return false;
  for (const auto& field : sample.fields) {
    if (sample_has_zero_copy_tensor(field))
      return true;
  }
  return false;
}

void maybe_force_copy_for_backpressure(Sample& sample, std::size_t qsize, const char* where,
                                       std::size_t seg_id) {
  if (!sample_has_zero_copy_tensor(sample))
    return;
  const int cap = zero_copy_backpressure_cap();
  if (cap <= 0 || qsize < static_cast<std::size_t>(cap))
    return;
  force_copy_sample_if_zero_copy(sample);
  if (env_bool("SIMA_GRAPH_ZERO_COPY_DEBUG", false)) {
    std::fprintf(stderr,
                 "[GRAPH] zero_copy_backpressure seg=%zu where=%s qsize=%zu cap=%d stream_id=%s\n",
                 seg_id, where ? where : "queue", qsize, cap, sample.stream_id.c_str());
  }
}

InputOptions input_opts_from_spec(const OutputSpec& spec, bool complete) {
  InputOptions opt;
  if (!spec.media_type.empty())
    opt.payload_type = input_type_from_media_type(spec.media_type);
  if (complete) {
    if (!spec.format.empty())
      opt.format = spec.format;
    if (spec.width > 0)
      opt.width = spec.width;
    if (spec.height > 0)
      opt.height = spec.height;
    if (spec.depth > 0)
      opt.depth = spec.depth;
  }
  return opt;
}

bool is_encoded_sample(const Sample& sample) {
  if (sample.kind != SampleKind::TensorSet || sample.tensors.size() != 1U) {
    return false;
  }
  return sample.tensors.front().semantic.encoded.has_value();
}

std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                              int64_t elem_bytes) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = elem_bytes;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = stride;
    stride *= shape[static_cast<std::size_t>(i)];
  }
  return strides;
}

std::string upper_copy(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return out;
}

bool is_system_memory_spec(const std::string& memory) {
  if (memory.empty())
    return true;
  const std::string up = upper_copy(memory);
  return up == "SYSTEMMEMORY" || up == "UNKNOWN";
}

TensorDType dtype_from_format_or_dtype(const OutputSpec& spec) {
  const std::string fmt = upper_copy(spec.format);
  if (!fmt.empty()) {
    if (fmt == "DETESS")
      return TensorDType::UInt16;
    if (fmt == "DETESSDEQUANT" || fmt == "FP32")
      return TensorDType::Float32;
    if (fmt == "EVXX_INT8" || fmt == "EV74_INT8" || fmt == "INT8")
      return TensorDType::Int8;
    if (fmt == "MLA")
      return TensorDType::Int8;
    if (fmt == "EVXX_BFLOAT16" || fmt == "BF16" || fmt == "BFLOAT16") {
      return TensorDType::BFloat16;
    }
    if (fmt == "UINT8")
      return TensorDType::UInt8;
    if (fmt == "UINT16")
      return TensorDType::UInt16;
    if (fmt == "INT16")
      return TensorDType::Int16;
    if (fmt == "INT32")
      return TensorDType::Int32;
    if (fmt == "FP64")
      return TensorDType::Float64;
  }

  const std::string dt = upper_copy(spec.dtype);
  if (dt == "INT8")
    return TensorDType::Int8;
  if (dt == "UINT8")
    return TensorDType::UInt8;
  if (dt == "INT16")
    return TensorDType::Int16;
  if (dt == "UINT16")
    return TensorDType::UInt16;
  if (dt == "INT32")
    return TensorDType::Int32;
  if (dt == "BF16" || dt == "BFLOAT16")
    return TensorDType::BFloat16;
  if (dt == "FP32" || dt == "FLOAT32")
    return TensorDType::Float32;
  if (dt == "FP64" || dt == "FLOAT64")
    return TensorDType::Float64;
  return TensorDType::UInt8;
}

size_t dtype_bytes(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return 1;
  case TensorDType::Int8:
    return 1;
  case TensorDType::UInt16:
    return 2;
  case TensorDType::Int16:
    return 2;
  case TensorDType::Int32:
    return 4;
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 0;
}

TensorLayout layout_from_spec(const OutputSpec& spec) {
  const std::string up = upper_copy(spec.layout);
  if (up.find("CHW") != std::string::npos)
    return TensorLayout::CHW;
  if (up.find("HWC") != std::string::npos)
    return TensorLayout::HWC;
  if (up.find("HW") != std::string::npos)
    return TensorLayout::HW;
  return TensorLayout::Unknown;
}

std::optional<Sample> sample_from_input_spec(const OutputSpec& spec, std::string* err) {
  if (!is_system_memory_spec(spec.memory)) {
    if (err)
      *err = "prebuild skipped: non-system memory spec";
    return std::nullopt;
  }
  if (spec.media_type == "video/x-raw") {
    const std::string fmt = upper_copy(spec.format);
    if (fmt.empty()) {
      if (err)
        *err = "prebuild skipped: missing video format";
      return std::nullopt;
    }
    if (spec.width <= 0 || spec.height <= 0) {
      if (err)
        *err = "prebuild skipped: missing video width/height";
      return std::nullopt;
    }
    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8" && fmt != "NV12" && fmt != "I420") {
      if (err)
        *err = "prebuild skipped: unsupported video format";
      return std::nullopt;
    }

    size_t bytes = expected_byte_size(spec);
    if (bytes == 0) {
      if (fmt == "NV12" || fmt == "I420") {
        bytes = static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * 3 / 2;
      } else if (fmt == "GRAY8") {
        bytes = static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height);
      } else {
        bytes = static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * 3;
      }
    }
    if (bytes == 0) {
      if (err)
        *err = "prebuild skipped: invalid video byte size";
      return std::nullopt;
    }

    simaai::neat::Tensor t;
    t.storage = simaai::neat::make_cpu_owned_storage(bytes);
    t.dtype = TensorDType::UInt8;
    t.device = {simaai::neat::DeviceType::CPU, 0};
    t.read_only = true;

    simaai::neat::ImageSpec::PixelFormat pixel = simaai::neat::ImageSpec::PixelFormat::UNKNOWN;
    if (fmt == "RGB")
      pixel = simaai::neat::ImageSpec::PixelFormat::RGB;
    else if (fmt == "BGR")
      pixel = simaai::neat::ImageSpec::PixelFormat::BGR;
    else if (fmt == "GRAY8")
      pixel = simaai::neat::ImageSpec::PixelFormat::GRAY8;
    else if (fmt == "NV12")
      pixel = simaai::neat::ImageSpec::PixelFormat::NV12;
    else if (fmt == "I420")
      pixel = simaai::neat::ImageSpec::PixelFormat::I420;
    t.semantic.image = simaai::neat::ImageSpec{pixel, {}};

    if (fmt == "NV12" || fmt == "I420") {
      if ((spec.width % 2) != 0 || (spec.height % 2) != 0) {
        if (err)
          *err = "prebuild skipped: NV12/I420 requires even width/height";
        return std::nullopt;
      }
      t.shape = {spec.height, spec.width};
      t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W};

      const int64_t w = spec.width;
      const int64_t h = spec.height;
      simaai::neat::Plane y;
      y.role = simaai::neat::PlaneRole::Y;
      y.shape = {h, w};
      y.strides_bytes = {w, 1};
      y.byte_offset = 0;
      t.planes.push_back(y);

      if (fmt == "NV12") {
        simaai::neat::Plane uv;
        uv.role = simaai::neat::PlaneRole::UV;
        uv.shape = {h / 2, w};
        uv.strides_bytes = {w, 1};
        uv.byte_offset = w * h;
        t.planes.push_back(uv);
      } else {
        const int64_t half_w = w / 2;
        const int64_t half_h = h / 2;
        simaai::neat::Plane u;
        u.role = simaai::neat::PlaneRole::U;
        u.shape = {half_h, half_w};
        u.strides_bytes = {half_w, 1};
        u.byte_offset = w * h;
        t.planes.push_back(u);

        simaai::neat::Plane v;
        v.role = simaai::neat::PlaneRole::V;
        v.shape = {half_h, half_w};
        v.strides_bytes = {half_w, 1};
        v.byte_offset = w * h + (half_w * half_h);
        t.planes.push_back(v);
      }
    } else {
      const int64_t w = spec.width;
      const int64_t h = spec.height;
      if (fmt == "GRAY8") {
        t.shape = {h, w};
        t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W};
        t.strides_bytes = contiguous_strides_bytes(t.shape, 1);
      } else {
        t.shape = {h, w, 3};
        t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
        t.strides_bytes = contiguous_strides_bytes(t.shape, 1);
      }
    }

    Sample sample = sample_from_tensors(TensorList{std::move(t)});
    sample.payload_type = spec.payload_type != PayloadType::Auto
                              ? spec.payload_type
                              : payload_type_from_media_type(spec.media_type);
    sample.media_type = spec.media_type;
    sample.payload_tag = spec.format;
    return sample;
  }

  if (spec.media_type == "application/vnd.simaai.tensor") {
    if (spec.width <= 0 || spec.height <= 0 || spec.depth <= 0) {
      if (err)
        *err = "prebuild skipped: missing tensor width/height/depth";
      return std::nullopt;
    }
    TensorLayout layout = layout_from_spec(spec);
    if (layout == TensorLayout::Unknown && spec.depth > 1) {
      if (err)
        *err = "prebuild skipped: missing explicit tensor layout for depth > 1";
      return std::nullopt;
    }
    if (layout == TensorLayout::HW && spec.depth > 1) {
      if (err)
        *err = "prebuild skipped: HW layout with depth > 1";
      return std::nullopt;
    }

    simaai::neat::Tensor t;
    t.dtype = dtype_from_format_or_dtype(spec);
    t.device = {simaai::neat::DeviceType::CPU, 0};
    t.read_only = true;

    const int64_t w = spec.width;
    const int64_t h = spec.height;
    const int64_t d = spec.depth;
    if (layout == TensorLayout::HWC) {
      t.shape = {h, w, d};
      t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
    } else if (layout == TensorLayout::CHW) {
      t.shape = {d, h, w};
      t.axis_semantics = {TensorAxisSemantic::C, TensorAxisSemantic::H, TensorAxisSemantic::W};
    } else {
      t.shape = {h, w};
      t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W};
    }

    const size_t elem = dtype_bytes(t.dtype);
    if (elem == 0) {
      if (err)
        *err = "prebuild skipped: unknown tensor dtype";
      return std::nullopt;
    }
    size_t bytes = expected_byte_size(spec);
    if (bytes == 0) {
      bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(d) * elem;
    }
    if (bytes == 0) {
      if (err)
        *err = "prebuild skipped: invalid tensor byte size";
      return std::nullopt;
    }

    t.storage = simaai::neat::make_cpu_owned_storage(bytes);
    t.strides_bytes = contiguous_strides_bytes(t.shape, static_cast<int64_t>(elem));

    Sample sample = sample_from_tensors(TensorList{std::move(t)});
    sample.payload_type = spec.payload_type != PayloadType::Auto
                              ? spec.payload_type
                              : payload_type_from_media_type(spec.media_type);
    sample.media_type = spec.media_type;
    sample.payload_tag = spec.format;
    return sample;
  }

  if (err)
    *err = "prebuild skipped: unsupported media_type";
  return std::nullopt;
}

Sample make_bundle_carrier_sample() {
  Sample sample;
  sample.payload_type = PayloadType::Tensor;
  sample.media_type = "application/vnd.simaai.tensor";
  sample.format = "FP32";

  constexpr int64_t kW = 1;
  constexpr int64_t kH = 1;
  constexpr int64_t kD = 1;
  constexpr std::size_t kBytes = sizeof(float);

  auto storage = simaai::neat::make_cpu_owned_storage(kBytes);
  auto mapping = storage->map(simaai::neat::MapMode::Write);
  if (mapping.data && mapping.size_bytes >= kBytes) {
    std::memset(mapping.data, 0, kBytes);
  }

  simaai::neat::Tensor t;
  t.storage = std::move(storage);
  t.dtype = TensorDType::Float32;
  t.shape = {kH, kW, kD};
  t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
  t.strides_bytes = contiguous_strides_bytes(t.shape, static_cast<int64_t>(sizeof(float)));
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;

  sample = sample_from_tensors(TensorList{std::move(t)});
  sample.payload_type = PayloadType::Tensor;
  sample.media_type = "application/vnd.simaai.tensor";
  sample.format = "FP32";
  return sample;
}

} // namespace simaai::neat::graph

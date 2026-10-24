#include "HostPcieChannel.h"

#include "HostPcieTensorPayload.h"
#include "HostPcieTensorSetMeta.h"

#include <simaai/scheduler/pcie/channel.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::pcie::internal {
namespace {

std::string upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return value;
}

std::optional<ImageSpec> tensor_image_spec(const Tensor& tensor) {
  if (tensor.image.has_value() && tensor.image->format != PixelFormat::Unknown) {
    return tensor.image;
  }
  if (tensor.image_format != PixelFormat::Unknown) {
    return ImageSpec{.format = tensor.image_format};
  }
  return std::nullopt;
}

const char* pixel_format_caps_name(const PixelFormat format) {
  switch (format) {
  case PixelFormat::RGB:
    return "RGB";
  case PixelFormat::BGR:
    return "BGR";
  case PixelFormat::GRAY8:
    return "GRAY8";
  case PixelFormat::NV12:
    return "NV12";
  case PixelFormat::I420:
    return "I420";
  case PixelFormat::Unknown:
    break;
  }
  return nullptr;
}

const char* tensor_dtype_caps_name(const TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "EVXX_UINT8";
  case TensorDType::Int8:
    return "EVXX_INT8";
  case TensorDType::Int16:
    return "EVXX_INT16";
  case TensorDType::Int32:
    return "EVXX_INT32";
  case TensorDType::BFloat16:
    return "EVXX_BFLOAT16";
  case TensorDType::Float32:
    return "EVXX_FLOAT32";
  case TensorDType::UInt16:
  case TensorDType::Float64:
    break;
  }
  throw std::runtime_error("PCIe tensor-set caps do not support this input dtype");
}

std::string tensor_set_caps_for_primary_tensor(const Tensor& tensor) {
  if (tensor.shape.empty()) {
    throw std::runtime_error("PCIe tensor-set caps require a non-empty primary tensor shape");
  }

  const char* dtype = tensor_dtype_caps_name(tensor.dtype);
  std::ostringstream caps;
  caps << "application/vnd.simaai.tensor, format=(string)" << dtype << ", dtype=(string)" << dtype
       << ", rank=(int)" << tensor.shape.size();
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (tensor.shape[i] <= 0 || tensor.shape[i] > std::numeric_limits<int>::max()) {
      throw std::runtime_error("PCIe tensor-set caps require positive 32-bit shape dimensions");
    }
    caps << ", dim" << i << "=(int)" << tensor.shape[i];
  }
  caps << ", shape=(string)\"";
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i != 0U) {
      caps << ',';
    }
    caps << tensor.shape[i];
  }
  caps << "\", representation=(string)tensor-set, storage=(string)tensorbuffer";
  return caps.str();
}

std::pair<std::int64_t, std::int64_t> image_height_width(const Tensor& tensor,
                                                         const PixelFormat format) {
  if (tensor.layout == TensorLayout::NHWC) {
    if (tensor.shape.size() != 4U || tensor.shape[0] != 1) {
      throw std::runtime_error(
          "PCIe raw image transport requires NHWC shape [1, height, width, channels]");
    }
    return {tensor.shape[1], tensor.shape[2]};
  }
  if (!tensor.planes.empty() && tensor.planes.front().shape.size() >= 2U) {
    return {tensor.planes.front().shape[0], tensor.planes.front().shape[1]};
  }
  if (tensor.shape.size() >= 2U) {
    std::int64_t height = tensor.shape[0];
    if (tensor.shape.size() == 2U && (format == PixelFormat::NV12 || format == PixelFormat::I420)) {
      if (height <= 0 || height % 3 != 0) {
        throw std::runtime_error("packed NV12/I420 image height must equal visible height * 3 / 2");
      }
      height = height * 2 / 3;
    }
    return {height, tensor.shape[1]};
  }
  return {0, 0};
}

void validate_image_storage(const Tensor& tensor, const PixelFormat format) {
  if (tensor.dtype != TensorDType::UInt8) {
    throw std::runtime_error("PCIe raw image transport requires UInt8 storage");
  }

  const bool hw = tensor.layout == TensorLayout::HW && tensor.shape.size() == 2U;
  const bool hwc = tensor.layout == TensorLayout::HWC && tensor.shape.size() == 3U;
  const bool nhwc =
      tensor.layout == TensorLayout::NHWC && tensor.shape.size() == 4U && tensor.shape[0] == 1;
  switch (format) {
  case PixelFormat::RGB:
  case PixelFormat::BGR:
    if ((!hwc || tensor.shape[2] != 3) && (!nhwc || tensor.shape[3] != 3)) {
      throw std::runtime_error("PCIe RGB/BGR images require HWC or singleton NHWC storage with "
                               "three channels");
    }
    return;
  case PixelFormat::GRAY8:
    if (!hw && (!hwc || tensor.shape[2] != 1) && (!nhwc || tensor.shape[3] != 1)) {
      throw std::runtime_error(
          "PCIe GRAY8 images require HW or single-channel HWC/singleton NHWC storage");
    }
    return;
  case PixelFormat::NV12:
  case PixelFormat::I420:
    if (tensor.planes.empty()) {
      if (!hw) {
        throw std::runtime_error("PCIe packed NV12/I420 images require two-dimensional HW storage");
      }
      if (tensor.shape[1] % 2 != 0) {
        throw std::runtime_error("PCIe packed NV12/I420 images require an even width");
      }
    }
    return;
  case PixelFormat::Unknown:
    return;
  }
}

std::string image_caps_for_tensor(const Tensor& tensor, const ImageSpec& image) {
  const char* format = pixel_format_caps_name(image.format);
  if (!format) {
    throw std::runtime_error("image tensor has unknown pixel format");
  }

  validate_image_storage(tensor, image.format);
  const auto [height, width] = image_height_width(tensor, image.format);
  if (width <= 0 || height <= 0 || width > std::numeric_limits<int>::max() ||
      height > std::numeric_limits<int>::max()) {
    throw std::runtime_error("image tensor requires positive width/height in shape or planes");
  }

  return std::string("video/x-raw,format=(string)") + format + ",width=(int)" +
         std::to_string(width) + ",height=(int)" + std::to_string(height);
}

TensorDType dtype_from_fact(const std::string& dtype) {
  const std::string v = upper_copy(dtype);
  if (v == "UINT8")
    return TensorDType::UInt8;
  if (v == "INT8" || v == "EVXX_INT8" || v == "EV74_INT8")
    return TensorDType::Int8;
  if (v == "UINT16")
    return TensorDType::UInt16;
  if (v == "INT16")
    return TensorDType::Int16;
  if (v == "INT32")
    return TensorDType::Int32;
  if (v == "BF16" || v == "BFLOAT16" || v == "EVXX_BFLOAT16" || v == "EV74_BFLOAT16")
    return TensorDType::BFloat16;
  if (v == "FP32" || v == "FLOAT32")
    return TensorDType::Float32;
  if (v == "FP64" || v == "FLOAT64")
    return TensorDType::Float64;
  throw std::runtime_error("unsupported tensor dtype from model facts: " + dtype);
}

void free_wrapped_payload(gpointer user_data) {
  auto* holder = static_cast<std::shared_ptr<void>*>(user_data);
  delete holder;
}

} // namespace

TensorList tensors_from_output_payload(const std::shared_ptr<MappedSample>& owner,
                                       const PcieModelFacts& facts) {
  TensorList out;
  if (!owner || !owner->map.data) {
    return out;
  }
  HostPcieChannel::validate_output_payload_size(owner->map.size, facts.packed_output_bytes);
  std::shared_ptr<std::vector<std::uint8_t>> dense;
  if (facts.dense_output_bytes > 0U) {
    dense = std::make_shared<std::vector<std::uint8_t>>(facts.dense_output_bytes);
  }
  for (std::size_t i = 0; i < facts.outputs.size(); ++i) {
    const auto& fact = facts.outputs[i];
    if (fact.payload_offset > owner->map.size ||
        fact.size_bytes > owner->map.size - fact.payload_offset) {
      throw std::runtime_error("PCIe output '" + fact.name + "' exceeds the received payload span");
    }

    Tensor tensor;
    tensor.owner = owner;
    tensor.data = static_cast<std::uint8_t*>(owner->map.data) + fact.payload_offset;
    tensor.size_bytes = fact.size_bytes;
    tensor.dtype = dtype_from_fact(fact.dtype);
    if (dense) {
      std::uint8_t* dst = dense->data() + fact.dense_offset;
      if (!copy_dense_rows(static_cast<const std::uint8_t*>(tensor.data),
                           owner->map.size - fact.payload_offset, fact.shape,
                           fact.transport_strides_bytes, tensor_dtype_bytes(tensor.dtype), 0U,
                           &dst)) {
        throw std::runtime_error("PCIe output '" + fact.name +
                                 "' exceeds the received payload span");
      }
      tensor.owner = dense;
      tensor.data = dense->data() + fact.dense_offset;
    }
    tensor.layout = TensorLayout::Unknown;
    tensor.shape = fact.shape;
    tensor.strides_bytes =
        contiguous_tensor_strides(tensor.shape, tensor_dtype_bytes(tensor.dtype));
    tensor.byte_offset = 0;
    tensor.read_only = true;
    tensor.route.name = fact.name.empty() ? "tensor_" + std::to_string(i) : fact.name;
    tensor.route.logical_index = fact.tensor_index >= 0 ? fact.tensor_index : static_cast<int>(i);
    tensor.route.backend_output_index = fact.tensor_index;
    tensor.route.physical_index = fact.physical_index;
    tensor.route.physical_byte_offset = fact.byte_offset;
    out.push_back(std::move(tensor));
  }
  return out;
}

namespace {

bool sample_has_bbox_caps(GstSample* sample) {
  GstCaps* caps = gst_sample_get_caps(sample);
  if (!caps || gst_caps_is_empty(caps)) {
    return false;
  }

  const GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    return false;
  }

  const char* media_type = gst_structure_get_name(structure);
  const char* format = gst_structure_get_string(structure, "format");
  return media_type && std::strcmp(media_type, "application/vnd.simaai.tensor") == 0 && format &&
         std::strcmp(format, "BBOX") == 0;
}

TensorList bbox_tensor_from_output_payload(const std::shared_ptr<MappedSample>& owner) {
  TensorList out;
  if (!owner || !owner->map.data) {
    return out;
  }

  Tensor tensor;
  tensor.owner = owner;
  tensor.data = owner->map.data;
  tensor.size_bytes = owner->map.size;
  tensor.dtype = TensorDType::UInt8;
  tensor.layout = TensorLayout::Unknown;
  tensor.shape = {static_cast<std::int64_t>(owner->map.size)};
  tensor.strides_bytes = {1};
  tensor.byte_offset = 0;
  tensor.read_only = true;
  tensor.route.name = "BBOX";
  tensor.route.logical_index = 0;
  tensor.route.backend_output_index = 0;
  tensor.route.physical_index = 0;
  out.push_back(std::move(tensor));
  return out;
}

TensorList tensors_from_output_sample(const std::shared_ptr<MappedSample>& owner,
                                      const PcieModelFacts& facts, const bool expects_bbox_output) {
  if (expects_bbox_output || (owner && sample_has_bbox_caps(owner->sample))) {
    return bbox_tensor_from_output_payload(owner);
  }
  return tensors_from_output_payload(owner, facts);
}

} // namespace

HostPcieChannel::HostPcieChannel()
    : transport_(std::make_unique<simaai::scheduler::pcie::Channel>()) {}
HostPcieChannel::~HostPcieChannel() = default;
void HostPcieChannel::configure(const PcieModelFacts& facts, int queue, int card, int inflight,
                                bool bbox, std::size_t output_bytes) {
  transport_->configure(queue, card, inflight,
                        required_transport_buffer_size(facts.packed_input_bytes,
                                                       facts.packed_output_bytes, 0, output_bytes));
  facts_ = facts;
  expects_bbox_output_ = bbox;
  output_layout_.reset();
}
void HostPcieChannel::request_stop() {
  transport_->request_stop();
}
void HostPcieChannel::stop() {
  transport_->stop();
}
bool HostPcieChannel::is_running() const {
  return transport_->is_running();
}
std::size_t HostPcieChannel::required_transport_buffer_size(std::size_t a, std::size_t b,
                                                            std::size_t c, std::size_t d) {
  return simaai::scheduler::pcie::Channel::required_buffer_size(a, b, c, d);
}
void HostPcieChannel::attach_request_id(GstBuffer* buffer, std::int32_t id) {
  simaai::scheduler::pcie::Channel::attach_request_id(buffer, id);
}
std::optional<std::int32_t> HostPcieChannel::request_id_from_buffer(GstBuffer* buffer) {
  return simaai::scheduler::pcie::Channel::request_id_from_buffer(buffer);
}
std::string HostPcieChannel::tensor_set_caps() {
  return "application/vnd.simaai.tensor, representation=(string)tensor-set, "
         "storage=(string)tensorbuffer";
}

void HostPcieChannel::validate_output_payload_size(const std::size_t received_bytes,
                                                   const std::size_t expected_bytes) {
  if (received_bytes < expected_bytes) {
    throw std::runtime_error("PCIe output payload is shorter than the model output contract: " +
                             std::to_string(received_bytes) + " bytes received, " +
                             std::to_string(expected_bytes) + " bytes expected");
  }
}

std::string HostPcieChannel::caps_for_tensors(const TensorList& tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("PCIe payload requires at least one tensor");
  }

  std::optional<ImageSpec> image;
  for (const auto& tensor : tensors) {
    const std::optional<ImageSpec> current = tensor_image_spec(tensor);
    if (!current.has_value()) {
      continue;
    }
    if (image.has_value()) {
      throw std::runtime_error("PCIe raw image transport supports one image tensor per push");
    }
    image = current;
  }

  if (!image.has_value()) {
    return tensor_set_caps_for_primary_tensor(tensors.front());
  }
  if (tensors.size() != 1U) {
    throw std::runtime_error("PCIe raw image transport cannot mix image and tensor payloads");
  }
  return image_caps_for_tensor(tensors.front(), *image);
}

bool HostPcieChannel::push(const TensorList& tensors) {
  return push_payload(next_request_id_.fetch_add(1), tensors, true);
}
bool HostPcieChannel::try_push(std::int32_t id, const TensorList& tensors) {
  return push_payload(id, tensors, false);
}
bool HostPcieChannel::push_payload(std::int32_t id, const TensorList& tensors, bool wait) {
  struct Context {
    HostPcieChannel& channel;
    const TensorList& tensors;
  } context{*this, tensors};
  return transport_->push_prepared(
      id,
      [](void* opaque) {
        auto& context = *static_cast<Context*>(opaque);
        simaai::scheduler::pcie::PreparedInput input;
        input.caps = caps_for_tensors(context.tensors);
        auto payload = prepare_tensor_payload(context.tensors);
        if (!payload.owner || !payload.data || !payload.size_bytes)
          throw std::runtime_error("prepared PCIe payload is invalid");
        auto* holder = new std::shared_ptr<void>(std::move(payload.owner));
        input.buffer =
            gst_buffer_new_wrapped_full(payload.flags, payload.data, payload.size_bytes, 0,
                                        payload.size_bytes, holder, free_wrapped_payload);
        if (!input.buffer) {
          delete holder;
          throw std::runtime_error("failed to allocate GstBuffer");
        }
        try {
          if (input.caps.find("representation=(string)tensor-set") != std::string::npos)
            attach_tensor_set_meta(input.buffer, payload.spans, context.channel.facts_.inputs,
                                   context.channel.facts_.packed_input
                                       ? &*context.channel.facts_.packed_input
                                       : nullptr);
        } catch (...) {
          gst_buffer_unref(input.buffer);
          throw;
        }
        return input;
      },
      &context, wait);
}
std::optional<TensorList> HostPcieChannel::pull(int timeout_ms) {
  auto result = pull_result(timeout_ms);
  if (!result)
    return std::nullopt;
  return std::move(result->outputs);
}
std::optional<RuntimeInferenceResult> HostPcieChannel::pull_result(int timeout_ms) {
  std::lock_guard lock(receive_mutex_);
  auto received = transport_->pull(timeout_ms);
  if (!received)
    return std::nullopt;
  auto owner = std::make_shared<MappedSample>();
  owner->sample = gst_sample_ref(received->sample.get());
  owner->buffer = gst_sample_get_buffer(owner->sample);
  if (!owner->buffer || !gst_buffer_map(owner->buffer, &owner->map, GST_MAP_READ))
    throw std::runtime_error("PCIe output cannot be mapped for public tensor access");
  owner->mapped = true;
  if (!expects_bbox_output_ && !sample_has_bbox_caps(owner->sample)) {
    if (!output_layout_)
      output_layout_ = resolve_output_layout(facts_, gst_sample_get_caps(owner->sample));
    if (output_layout_->raw_bytes && owner->map.size != output_layout_->raw_bytes)
      throw std::runtime_error("PCIe raw output extent differs from the negotiated layout");
  }
  return RuntimeInferenceResult{
      received->request_id,
      tensors_from_output_sample(owner, output_layout_ ? output_layout_->facts : facts_,
                                 expects_bbox_output_)};
}
} // namespace simaai::neat::pcie::internal

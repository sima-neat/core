#include "HostPcieChannel.h"

#include "HostPcieTensorPayload.h"
#include "HostPcieTensorSetMeta.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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

std::pair<std::int64_t, std::int64_t> image_height_width(const Tensor& tensor) {
  if (!tensor.planes.empty() && tensor.planes.front().shape.size() >= 2U) {
    return {tensor.planes.front().shape[0], tensor.planes.front().shape[1]};
  }
  if (tensor.layout == TensorLayout::NHWC && tensor.shape.size() >= 4U) {
    return {tensor.shape[1], tensor.shape[2]};
  }
  if (tensor.shape.size() >= 2U) {
    return {tensor.shape[0], tensor.shape[1]};
  }
  return {0, 0};
}

std::string image_caps_for_tensor(const Tensor& tensor, const ImageSpec& image) {
  const char* format = pixel_format_caps_name(image.format);
  if (!format) {
    throw std::runtime_error("image tensor has unknown pixel format");
  }

  const auto [height, width] = image_height_width(tensor);
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

TensorList tensors_from_output_payload(const std::shared_ptr<MappedSample>& owner,
                                       const PcieModelFacts& facts) {
  TensorList out;
  if (!owner || !owner->map.data) {
    return out;
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < facts.outputs.size(); ++i) {
    const auto& fact = facts.outputs[i];
    if (offset + fact.size_bytes > owner->map.size) {
      break;
    }

    Tensor tensor;
    tensor.owner = owner;
    tensor.data = static_cast<std::uint8_t*>(owner->map.data) + offset;
    tensor.size_bytes = fact.size_bytes;
    tensor.dtype = dtype_from_fact(fact.dtype);
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
    offset += fact.size_bytes;
  }
  return out;
}

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
                                      const PcieModelFacts& facts) {
  if (owner && sample_has_bbox_caps(owner->sample)) {
    return bbox_tensor_from_output_payload(owner);
  }
  return tensors_from_output_payload(owner, facts);
}

} // namespace

HostPcieChannel::HostPcieChannel() {
  static std::once_flag once;
  std::call_once(once, []() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
  });
}

HostPcieChannel::~HostPcieChannel() {
  stop();
}

void HostPcieChannel::configure(const PcieModelFacts& facts, const int queue, const int card_id,
                                const int max_inflight) {
  if (running_.load()) {
    throw std::runtime_error("cannot configure HostPcieChannel while running");
  }
  if (max_inflight < 0) {
    throw std::invalid_argument("max_inflight must be non-negative");
  }
  facts_ = facts;
  pcie_queue_ = queue;
  card_id_ = card_id;
  max_inflight_ = max_inflight;
  stop_requested_.store(false);
}

std::string HostPcieChannel::tensor_set_caps() {
  return "application/vnd.simaai.tensor, representation=(string)tensor-set, "
         "storage=(string)tensorbuffer";
}

std::string HostPcieChannel::caps_for_tensors(const TensorList& tensors) {
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
    return tensor_set_caps();
  }
  if (tensors.size() != 1U) {
    throw std::runtime_error("PCIe raw image transport cannot mix image and tensor payloads");
  }
  return image_caps_for_tensor(tensors.front(), *image);
}

void HostPcieChannel::start_with_caps(const std::string& caps_string) {
  if (running_.load()) {
    if (caps_string != caps_) {
      GstCaps* caps = gst_caps_from_string(caps_string.c_str());
      if (!caps) {
        throw std::runtime_error("failed to parse caps: " + caps_string);
      }
      g_object_set(G_OBJECT(appsrc_), "caps", caps, nullptr);
      gst_caps_unref(caps);
      caps_ = caps_string;
    }
    return;
  }

  caps_ = caps_string;
  pipeline_ = gst_pipeline_new("sima_neat_pcie_host");
  appsrc_ = gst_element_factory_make("appsrc", "src");
  queue_element_ = gst_element_factory_make("queue", "q");
  pciehost_ = gst_element_factory_make("neatpciehost", "pcie");
  appsink_ = gst_element_factory_make("appsink", "sink");

  if (!pipeline_ || !appsrc_ || !queue_element_ || !pciehost_ || !appsink_) {
    stop();
    throw std::runtime_error("failed to create appsrc/queue/neatpciehost/appsink elements");
  }

  GstCaps* caps = gst_caps_from_string(caps_.c_str());
  if (!caps) {
    stop();
    throw std::runtime_error("failed to parse caps: " + caps_);
  }

  const guint queue_depth = static_cast<guint>(max_inflight_);
  g_object_set(G_OBJECT(appsrc_), "caps", caps, "is-live", TRUE, "do-timestamp", TRUE, "block",
               TRUE, "format", GST_FORMAT_TIME, "max-buffers", queue_depth, "max-bytes",
               static_cast<guint64>(0), "max-time", static_cast<guint64>(0), nullptr);
  gst_caps_unref(caps);

  const std::size_t auto_buffer = std::max(facts_.packed_input_bytes, facts_.packed_output_bytes);
  const guint64 buffer_size =
      static_cast<guint64>(std::max<std::size_t>(auto_buffer, 512U * 1024U));
  g_object_set(G_OBJECT(pciehost_), "buffersize", buffer_size, "card-number", card_id_, "queue",
               pcie_queue_, "queuedepth", queue_depth, nullptr);

  g_object_set(G_OBJECT(appsink_), "emit-signals", TRUE, "sync", FALSE, "max-buffers", 256, "drop",
               FALSE, nullptr);
  g_object_set(G_OBJECT(queue_element_), "max-size-buffers", queue_depth, "max-size-bytes", 0,
               "max-size-time", static_cast<guint64>(0), "leaky", 0, nullptr);

  g_signal_connect(appsink_, "new-sample", G_CALLBACK(on_new_sample_static), this);

  gst_bin_add_many(GST_BIN(pipeline_), appsrc_, queue_element_, pciehost_, appsink_, nullptr);
  if (!gst_element_link(appsrc_, queue_element_) || !gst_element_link(queue_element_, pciehost_) ||
      !gst_element_link(pciehost_, appsink_)) {
    stop();
    throw std::runtime_error("failed to link host PCIe pipeline");
  }

  if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    stop();
    throw std::runtime_error("failed to set host PCIe pipeline to PLAYING");
  }
  stop_requested_.store(false);
  running_.store(true);
}

void HostPcieChannel::stop() {
  stop_requested_.store(true);
  running_.store(false);
  if (appsrc_) {
    GstFlowReturn ret = GST_FLOW_OK;
    g_signal_emit_by_name(appsrc_, "end-of-stream", &ret);
  }
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
  }
  pipeline_ = nullptr;
  appsrc_ = nullptr;
  queue_element_ = nullptr;
  pciehost_ = nullptr;
  appsink_ = nullptr;
  caps_.clear();
  {
    std::lock_guard<std::mutex> lock(receive_mutex_);
    received_results_.clear();
    receive_error_.reset();
  }
  receive_cv_.notify_all();
}

bool HostPcieChannel::is_running() const {
  return running_.load();
}

bool HostPcieChannel::push(const TensorList& tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("PCIe payload requires at least one tensor");
  }
  PreparedPayload payload = prepare_tensor_payload(tensors);
  start_with_caps(caps_for_tensors(tensors));
  return push_prepared_payload(std::move(payload));
}

bool HostPcieChannel::push_prepared_payload(PreparedPayload&& payload) {
  if (!running_.load()) {
    throw std::runtime_error("host PCIe channel is not running");
  }
  if (!payload.owner || !payload.data || payload.size_bytes == 0) {
    throw std::runtime_error("prepared PCIe payload is invalid");
  }

  auto* holder = new std::shared_ptr<void>(std::move(payload.owner));
  GstBuffer* buffer =
      gst_buffer_new_wrapped_full(payload.flags, payload.data, payload.size_bytes, 0,
                                  payload.size_bytes, holder, free_wrapped_payload);
  if (!buffer) {
    delete holder;
    throw std::runtime_error("failed to allocate GstBuffer");
  }
  if (caps_ == tensor_set_caps()) {
    try {
      attach_tensor_set_meta(buffer, payload.spans);
    } catch (...) {
      gst_buffer_unref(buffer);
      throw;
    }
  }
  GstFlowReturn ret = GST_FLOW_OK;
  g_signal_emit_by_name(appsrc_, "push-buffer", buffer, &ret);
  gst_buffer_unref(buffer);
  if (ret != GST_FLOW_OK) {
    throw std::runtime_error("push-buffer failed: " + std::to_string(static_cast<int>(ret)));
  }
  return true;
}

std::optional<TensorList> HostPcieChannel::pull(const int timeout_ms) {
  std::unique_lock<std::mutex> lock(receive_mutex_);
  if (timeout_ms < 0) {
    receive_cv_.wait(lock, [&] {
      return !received_results_.empty() || receive_error_.has_value() || stop_requested_.load();
    });
  } else {
    const bool ready = receive_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
      return !received_results_.empty() || receive_error_.has_value() || stop_requested_.load();
    });
    if (!ready) {
      return std::nullopt;
    }
  }
  if (receive_error_) {
    throw std::runtime_error(*receive_error_);
  }
  if (received_results_.empty()) {
    return std::nullopt;
  }
  TensorList result = std::move(received_results_.front());
  received_results_.pop_front();
  return result;
}

GstFlowReturn HostPcieChannel::on_new_sample_static(GstElement* sink, gpointer user_data) {
  return static_cast<HostPcieChannel*>(user_data)->on_new_sample(sink);
}

GstFlowReturn HostPcieChannel::on_new_sample(GstElement* sink) {
  try {
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
      return GST_FLOW_ERROR;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
      gst_sample_unref(sample);
      return GST_FLOW_ERROR;
    }

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      gst_sample_unref(sample);
      return GST_FLOW_ERROR;
    }

    auto owner = std::make_shared<MappedSample>();
    owner->sample = sample;
    owner->buffer = buffer;
    owner->map = map;
    owner->mapped = true;

    TensorList result = tensors_from_output_sample(owner, facts_);

    {
      std::lock_guard<std::mutex> lock(receive_mutex_);
      received_results_.push_back(result);
    }
    receive_cv_.notify_one();

    return GST_FLOW_OK;
  } catch (const std::exception& e) {
    {
      std::lock_guard<std::mutex> lock(receive_mutex_);
      receive_error_ = std::string("PCIe appsink callback failed: ") + e.what();
    }
    receive_cv_.notify_all();
    return GST_FLOW_ERROR;
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(receive_mutex_);
      receive_error_ = "PCIe appsink callback failed with an unknown exception";
    }
    receive_cv_.notify_all();
    return GST_FLOW_ERROR;
  }
}

} // namespace simaai::neat::pcie::internal

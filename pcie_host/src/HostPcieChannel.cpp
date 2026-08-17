#include "HostPcieChannel.h"

#include "HostPcieTensorPayload.h"
#include "HostPcieTensorSetMeta.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

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

constexpr std::size_t kMinimumTransportBufferSize = 512U * 1024U;
constexpr std::size_t kMaximumTransportBufferSize = 128U * 1024U * 1024U;

void ensure_gstreamer_initialized() {
  static std::once_flag once;
  std::call_once(once, []() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
    static const gchar* tags[] = {nullptr};
    if (!gst_meta_get_info("GstSimaHostMeta")) {
      gst_meta_register_custom("GstSimaHostMeta", tags, nullptr, nullptr, nullptr);
    }
  });
}

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

TensorList tensors_from_output_payload(const std::shared_ptr<MappedSample>& owner,
                                       const PcieModelFacts& facts) {
  TensorList out;
  if (!owner || !owner->map.data) {
    return out;
  }
  HostPcieChannel::validate_output_payload_size(owner->map.size, facts.packed_output_bytes);
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

HostPcieChannel::HostPcieChannel() {
  ensure_gstreamer_initialized();
}

HostPcieChannel::~HostPcieChannel() {
  stop();
}

void HostPcieChannel::configure(const PcieModelFacts& facts, const int queue, const int card_id,
                                const int max_inflight, const bool expects_bbox_output) {
  std::lock_guard<std::mutex> lock(send_mutex_);
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
  expects_bbox_output_ = expects_bbox_output;
  stop_requested_.store(false);
  configured_ = true;
  {
    std::lock_guard<std::mutex> inflight_lock(inflight_mutex_);
    inflight_ = 0;
  }
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

std::size_t
HostPcieChannel::required_transport_buffer_size(const std::size_t packed_input_bytes,
                                                const std::size_t packed_output_bytes,
                                                const std::size_t submitted_payload_bytes) {
  const std::size_t required = std::max({kMinimumTransportBufferSize, packed_input_bytes,
                                         packed_output_bytes, submitted_payload_bytes});
  if (required > kMaximumTransportBufferSize) {
    throw std::runtime_error("PCIe transport payload exceeds the 128 MiB buffer limit: " +
                             std::to_string(required) + " bytes required");
  }
  return required;
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

void HostPcieChannel::start_with_caps(const std::string& caps_string,
                                      const std::size_t submitted_payload_bytes) {
  throw_if_stopped();
  if (!configured_) {
    throw std::runtime_error("host PCIe channel is not configured");
  }
  if (running_.load()) {
    if (submitted_payload_bytes > transport_buffer_size_) {
      throw std::runtime_error(
          "PCIe input payload exceeds the transport buffer selected by the first submission: " +
          std::to_string(submitted_payload_bytes) + " bytes submitted, " +
          std::to_string(transport_buffer_size_) +
          " bytes available; close and rebuild the model for the new input geometry");
    }
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

  transport_buffer_size_ = required_transport_buffer_size(
      facts_.packed_input_bytes, facts_.packed_output_bytes, submitted_payload_bytes);
  caps_ = caps_string;
  pipeline_ = gst_pipeline_new("sima_neat_pcie_host");
  appsrc_ = gst_element_factory_make("appsrc", "src");
  queue_element_ = gst_element_factory_make("queue", "q");
  pciehost_ = gst_element_factory_make("neatpciehost", "pcie");
  appsink_ = gst_element_factory_make("appsink", "sink");

  if (!pipeline_ || !appsrc_ || !queue_element_ || !pciehost_ || !appsink_) {
    stop_locked();
    throw std::runtime_error("failed to create appsrc/queue/neatpciehost/appsink elements");
  }

  GstCaps* caps = gst_caps_from_string(caps_.c_str());
  if (!caps) {
    stop_locked();
    throw std::runtime_error("failed to parse caps: " + caps_);
  }

  const guint queue_depth = static_cast<guint>(max_inflight_);
  g_object_set(G_OBJECT(appsrc_), "caps", caps, "is-live", TRUE, "do-timestamp", TRUE, "block",
               FALSE, "format", GST_FORMAT_TIME, "max-buffers", queue_depth, "max-bytes",
               static_cast<guint64>(0), "max-time", static_cast<guint64>(0), nullptr);
  gst_caps_unref(caps);

  g_object_set(G_OBJECT(pciehost_), "buffersize", static_cast<guint64>(transport_buffer_size_),
               "card-number", card_id_, "queue", pcie_queue_, "queuedepth", queue_depth, nullptr);

  g_object_set(G_OBJECT(appsink_), "emit-signals", TRUE, "sync", FALSE, "max-buffers", 256, "drop",
               FALSE, nullptr);
  g_object_set(G_OBJECT(queue_element_), "max-size-buffers", queue_depth, "max-size-bytes", 0,
               "max-size-time", static_cast<guint64>(0), "leaky", 0, nullptr);

  g_signal_connect(appsink_, "new-sample", G_CALLBACK(on_new_sample_static), this);

  gst_bin_add_many(GST_BIN(pipeline_), appsrc_, queue_element_, pciehost_, appsink_, nullptr);
  if (!gst_element_link(appsrc_, queue_element_)) {
    stop_locked();
    throw std::runtime_error("failed to link host PCIe pipeline");
  }

  GstPad* queue_src = gst_element_get_static_pad(queue_element_, "src");
  GstPad* appsink_sink = gst_element_get_static_pad(appsink_, "sink");
  pciehost_sink_pad_ = gst_element_request_pad_simple(pciehost_, "sink_0");
  pciehost_src_pad_ = gst_element_request_pad_simple(pciehost_, "src_0");
  const bool pads_linked = queue_src && appsink_sink && pciehost_sink_pad_ && pciehost_src_pad_ &&
                           gst_pad_link(queue_src, pciehost_sink_pad_) == GST_PAD_LINK_OK &&
                           gst_pad_link(pciehost_src_pad_, appsink_sink) == GST_PAD_LINK_OK;
  if (queue_src) {
    gst_object_unref(queue_src);
  }
  if (appsink_sink) {
    gst_object_unref(appsink_sink);
  }
  if (!pads_linked) {
    stop_locked();
    throw std::runtime_error("failed to link explicit neatpciehost sink_0/src_0 pads");
  }

  if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    stop_locked();
    throw std::runtime_error("failed to set host PCIe pipeline to PLAYING");
  }
  running_.store(true);
  GstBus* bus = gst_element_get_bus(pipeline_);
  if (!bus) {
    stop_locked();
    throw std::runtime_error("failed to get host PCIe pipeline bus");
  }
  try {
    bus_thread_ = std::thread(&HostPcieChannel::monitor_bus, this, bus);
  } catch (...) {
    gst_object_unref(bus);
    stop_locked();
    throw;
  }
}

void HostPcieChannel::request_stop() {
  stop_requested_.store(true);
  running_.store(false);
  inflight_cv_.notify_all();
  receive_cv_.notify_all();
}

void HostPcieChannel::stop() {
  request_stop();
  std::lock_guard<std::mutex> lock(send_mutex_);
  stop_locked();
}

void HostPcieChannel::stop_locked() {
  request_stop();
  configured_ = false;
  if (bus_thread_.joinable()) {
    bus_thread_.join();
  }
  if (appsrc_) {
    GstFlowReturn ret = GST_FLOW_OK;
    g_signal_emit_by_name(appsrc_, "end-of-stream", &ret);
    if (pipeline_ && ret == GST_FLOW_OK) {
      GstBus* bus = gst_element_get_bus(pipeline_);
      if (bus) {
        GstMessage* message = gst_bus_timed_pop_filtered(
            bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        if (message) {
          gst_message_unref(message);
        }
        gst_object_unref(bus);
      }
    }
  }
  if (pipeline_) {
    const GstStateChangeReturn state_change = gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (state_change == GST_STATE_CHANGE_ASYNC) {
      (void)gst_element_get_state(pipeline_, nullptr, nullptr, 5 * GST_SECOND);
    }
    if (pciehost_ && pciehost_sink_pad_) {
      gst_element_release_request_pad(pciehost_, pciehost_sink_pad_);
      gst_object_unref(pciehost_sink_pad_);
    }
    if (pciehost_ && pciehost_src_pad_) {
      gst_element_release_request_pad(pciehost_, pciehost_src_pad_);
      gst_object_unref(pciehost_src_pad_);
    }
    pciehost_sink_pad_ = nullptr;
    pciehost_src_pad_ = nullptr;
    gst_object_unref(pipeline_);
  }
  pipeline_ = nullptr;
  appsrc_ = nullptr;
  queue_element_ = nullptr;
  pciehost_ = nullptr;
  appsink_ = nullptr;
  pciehost_sink_pad_ = nullptr;
  pciehost_src_pad_ = nullptr;
  caps_.clear();
  transport_buffer_size_ = 0;
  {
    std::lock_guard<std::mutex> inflight_lock(inflight_mutex_);
    inflight_ = 0;
  }
  inflight_cv_.notify_all();
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

void HostPcieChannel::monitor_bus(GstBus* bus) {
  while (!stop_requested_.load()) {
    GstMessage* message = gst_bus_timed_pop_filtered(
        bus, 100 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!message) {
      continue;
    }

    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
      GError* error = nullptr;
      gchar* detail = nullptr;
      gst_message_parse_error(message, &error, &detail);
      std::ostringstream text;
      text << "PCIe pipeline error";
      if (GST_MESSAGE_SRC(message)) {
        text << " from " << GST_OBJECT_NAME(GST_MESSAGE_SRC(message));
      }
      if (error && error->message) {
        text << ": " << error->message;
      }
      if (detail && *detail) {
        text << " (" << detail << ")";
      }
      {
        std::lock_guard<std::mutex> lock(receive_mutex_);
        if (!receive_error_) {
          receive_error_ = text.str();
        }
      }
      g_clear_error(&error);
      g_free(detail);
      gst_message_unref(message);
      request_stop();
      break;
    }

    gst_message_unref(message);
    request_stop();
    break;
  }
  gst_object_unref(bus);
}

bool HostPcieChannel::push(const TensorList& tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("PCIe payload requires at least one tensor");
  }
  throw_if_stopped();
  PreparedPayload payload = prepare_tensor_payload(tensors);
  if (!wait_and_reserve_inflight()) {
    throw_if_stopped();
    throw std::runtime_error("host PCIe channel could not reserve transport capacity");
  }
  std::lock_guard<std::mutex> lock(send_mutex_);
  try {
    throw_if_stopped();
    start_with_caps(caps_for_tensors(tensors), payload.size_bytes);
    return push_prepared_payload(next_request_id_.fetch_add(1), std::move(payload));
  } catch (...) {
    release_inflight();
    throw;
  }
}

bool HostPcieChannel::try_push(const std::int32_t request_id, const TensorList& tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("PCIe payload requires at least one tensor");
  }
  throw_if_stopped();
  if (!reserve_inflight()) {
    throw_if_stopped();
    return false;
  }
  try {
    PreparedPayload payload = prepare_tensor_payload(tensors);
    std::lock_guard<std::mutex> lock(send_mutex_);
    throw_if_stopped();
    start_with_caps(caps_for_tensors(tensors), payload.size_bytes);
    return push_prepared_payload(request_id, std::move(payload));
  } catch (...) {
    release_inflight();
    throw;
  }
}

void HostPcieChannel::throw_if_stopped() {
  if (!stop_requested_.load()) {
    return;
  }
  std::lock_guard<std::mutex> lock(receive_mutex_);
  if (receive_error_) {
    throw std::runtime_error(*receive_error_);
  }
  throw std::runtime_error("host PCIe channel has stopped");
}

bool HostPcieChannel::reserve_inflight() {
  if (stop_requested_.load()) {
    return false;
  }
  if (max_inflight_ == 0) {
    return true;
  }
  std::lock_guard<std::mutex> lock(inflight_mutex_);
  if (stop_requested_.load() || inflight_ >= max_inflight_) {
    return false;
  }
  ++inflight_;
  return true;
}

bool HostPcieChannel::wait_and_reserve_inflight() {
  if (max_inflight_ == 0) {
    return !stop_requested_.load();
  }
  std::unique_lock<std::mutex> lock(inflight_mutex_);
  inflight_cv_.wait(lock, [&] { return inflight_ < max_inflight_ || stop_requested_.load(); });
  if (stop_requested_.load()) {
    return false;
  }
  ++inflight_;
  return true;
}

void HostPcieChannel::release_inflight() {
  if (max_inflight_ == 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    if (inflight_ > 0) {
      --inflight_;
    }
  }
  inflight_cv_.notify_one();
}

void HostPcieChannel::attach_request_id(GstBuffer* buffer, const std::int32_t request_id) {
  ensure_gstreamer_initialized();
  if (!buffer) {
    throw std::invalid_argument("cannot attach a request ID to a null GstBuffer");
  }
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaHostMeta");
  if (!meta) {
    throw std::runtime_error("failed to attach GstSimaHostMeta to PCIe input");
  }
  GstStructure* structure = gst_custom_meta_get_structure(meta);
  if (!structure) {
    throw std::runtime_error("GstSimaHostMeta has no writable structure");
  }
  const auto request_bits = std::bit_cast<std::uint32_t>(request_id);
  gst_structure_set(structure, "frame-identifier", G_TYPE_UINT64,
                    static_cast<guint64>(request_bits), "stream-id", G_TYPE_UINT,
                    static_cast<guint>(0), "frame-id", G_TYPE_UINT,
                    static_cast<guint>(request_bits), nullptr);
}

std::optional<std::int32_t> HostPcieChannel::request_id_from_buffer(GstBuffer* buffer) {
  if (!buffer) {
    return std::nullopt;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaHostMeta");
  if (!meta) {
    return std::nullopt;
  }
  const GstStructure* structure = gst_custom_meta_get_structure(meta);
  guint frame_id = 0;
  if (structure && gst_structure_get_uint(structure, "frame-id", &frame_id)) {
    return std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(frame_id));
  }

  guint64 request_id = 0;
  if (!structure || !gst_structure_get_uint64(structure, "frame-identifier", &request_id)) {
    return std::nullopt;
  }
  if (request_id > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("PCIe frame-identifier exceeds the PCIe request ID width");
  }
  return std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(request_id));
}

bool HostPcieChannel::push_prepared_payload(const std::int32_t request_id,
                                            PreparedPayload&& payload) {
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
  try {
    attach_request_id(buffer, request_id);
  } catch (...) {
    gst_buffer_unref(buffer);
    throw;
  }
  if (caps_.find("representation=(string)tensor-set") != std::string::npos) {
    try {
      attach_tensor_set_meta(buffer, payload.spans, facts_.inputs);
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
  auto result = pull_result(timeout_ms);
  if (!result) {
    return std::nullopt;
  }
  return std::move(result->outputs);
}

std::optional<RuntimeInferenceResult> HostPcieChannel::pull_result(const int timeout_ms) {
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
  RuntimeInferenceResult result = std::move(received_results_.front());
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
      release_inflight();
      return GST_FLOW_ERROR;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
      gst_sample_unref(sample);
      release_inflight();
      return GST_FLOW_ERROR;
    }
    const auto request_id = request_id_from_buffer(buffer);
    if (!request_id) {
      gst_sample_unref(sample);
      throw std::runtime_error("PCIe output is missing GstSimaHostMeta frame-identifier");
    }

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      gst_sample_unref(sample);
      release_inflight();
      return GST_FLOW_ERROR;
    }

    auto owner = std::make_shared<MappedSample>();
    owner->sample = sample;
    owner->buffer = buffer;
    owner->map = map;
    owner->mapped = true;

    RuntimeInferenceResult result{
        .request_id = *request_id,
        .outputs = tensors_from_output_sample(owner, facts_, expects_bbox_output_),
    };

    {
      std::lock_guard<std::mutex> lock(receive_mutex_);
      received_results_.push_back(std::move(result));
    }
    release_inflight();
    receive_cv_.notify_one();

    return GST_FLOW_OK;
  } catch (const std::exception& e) {
    release_inflight();
    {
      std::lock_guard<std::mutex> lock(receive_mutex_);
      receive_error_ = std::string("PCIe appsink callback failed: ") + e.what();
    }
    receive_cv_.notify_all();
    return GST_FLOW_ERROR;
  } catch (...) {
    release_inflight();
    {
      std::lock_guard<std::mutex> lock(receive_mutex_);
      receive_error_ = "PCIe appsink callback failed with an unknown exception";
    }
    receive_cv_.notify_all();
    return GST_FLOW_ERROR;
  }
}

} // namespace simaai::neat::pcie::internal

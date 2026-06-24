#include "HostPcieChannel.h"

#include "gst/SimaTensorSetMetaAbi.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

int tensor_set_dtype(const TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  case TensorDType::Int8:
    return SIMA_TENSOR_SET_DTYPE_INT8_V1;
  case TensorDType::UInt16:
    return SIMA_TENSOR_SET_DTYPE_UINT16_V1;
  case TensorDType::Int16:
    return SIMA_TENSOR_SET_DTYPE_INT16_V1;
  case TensorDType::Int32:
    return SIMA_TENSOR_SET_DTYPE_INT32_V1;
  case TensorDType::BFloat16:
    return SIMA_TENSOR_SET_DTYPE_BF16_V1;
  case TensorDType::Float32:
    return SIMA_TENSOR_SET_DTYPE_FP32_V1;
  case TensorDType::Float64:
    return SIMA_TENSOR_SET_DTYPE_FP64_V1;
  }
  return SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1;
}

int tensor_set_layout(const Tensor& tensor) {
  if (tensor.layout == TensorLayout::HW)
    return SIMA_TENSOR_SET_LAYOUT_HW_V1;
  if (tensor.layout == TensorLayout::HWC)
    return SIMA_TENSOR_SET_LAYOUT_HWC_V1;
  if (tensor.layout == TensorLayout::NHWC)
    return SIMA_TENSOR_SET_LAYOUT_NHWC_V1;
  if (tensor.shape.size() == 2)
    return SIMA_TENSOR_SET_LAYOUT_HW_V1;
  if (tensor.shape.size() == 3)
    return SIMA_TENSOR_SET_LAYOUT_HWC_V1;
  if (tensor.shape.size() == 4)
    return SIMA_TENSOR_SET_LAYOUT_NHWC_V1;
  return SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1;
}

std::size_t dtype_bytes(const TensorDType dtype) {
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

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& shape,
                                             const std::size_t elem_size) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = static_cast<std::int64_t>(elem_size);
  for (std::size_t index = shape.size(); index > 0; --index) {
    const std::size_t dim = index - 1;
    strides[dim] = stride;
    stride *= shape[dim];
  }
  return strides;
}

std::size_t dense_size_bytes(const Tensor& tensor) {
  const std::size_t elem = dtype_bytes(tensor.dtype);
  if (elem == 0) {
    return 0;
  }
  std::size_t bytes = elem;
  for (const auto dim : tensor.shape) {
    if (dim < 0) {
      return 0;
    }
    const auto udim = static_cast<std::size_t>(dim);
    if (udim != 0 && bytes > std::numeric_limits<std::size_t>::max() / udim) {
      return 0;
    }
    bytes *= udim;
  }
  return bytes;
}

const std::uint8_t* tensor_data(const Tensor& tensor, std::size_t* size_out) {
  if (!tensor.data) {
    throw std::runtime_error("tensor has no data pointer");
  }
  if (tensor.byte_offset < 0 || static_cast<std::size_t>(tensor.byte_offset) > tensor.size_bytes) {
    throw std::runtime_error("tensor byte offset is outside tensor payload");
  }
  const auto offset = static_cast<std::size_t>(tensor.byte_offset);
  if (size_out) {
    *size_out = tensor.size_bytes - offset;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + offset;
}

bool copy_dense_rows(const std::uint8_t* src, const std::size_t src_size,
                     const std::vector<std::int64_t>& shape,
                     const std::vector<std::int64_t>& strides, const std::size_t elem_size,
                     const std::size_t dim, std::uint8_t** dst) {
  if (dim + 1U == shape.size()) {
    const auto elements = static_cast<std::size_t>(shape[dim]);
    if (strides[dim] < static_cast<std::int64_t>(elem_size)) {
      return false;
    }
    for (std::size_t i = 0; i < elements; ++i) {
      const auto src_offset = static_cast<std::size_t>(static_cast<std::int64_t>(i) * strides[dim]);
      if (src_offset + elem_size > src_size) {
        return false;
      }
      std::memcpy(*dst, src + src_offset, elem_size);
      *dst += elem_size;
    }
    return true;
  }

  const auto count = static_cast<std::size_t>(shape[dim]);
  for (std::size_t i = 0; i < count; ++i) {
    const auto src_offset = static_cast<std::size_t>(static_cast<std::int64_t>(i) * strides[dim]);
    if (src_offset > src_size) {
      return false;
    }
    if (!copy_dense_rows(src + src_offset, src_size - src_offset, shape, strides, elem_size,
                         dim + 1U, dst)) {
      return false;
    }
  }
  return true;
}

std::vector<std::uint8_t> copy_dense_tensor_payload(const Tensor& tensor) {
  const std::size_t elem = dtype_bytes(tensor.dtype);
  const std::size_t bytes = dense_size_bytes(tensor);
  if (bytes == 0) {
    throw std::runtime_error("dense tensor has unknown or empty byte size");
  }

  std::size_t mapped_size = 0;
  const auto* src = tensor_data(tensor, &mapped_size);
  if (mapped_size < bytes && tensor.strides_bytes.empty()) {
    throw std::runtime_error("dense tensor mapping is smaller than required payload");
  }

  std::vector<std::uint8_t> out(bytes);
  if (tensor.strides_bytes.empty()) {
    std::memcpy(out.data(), src, bytes);
    return out;
  }

  std::vector<std::int64_t> strides = tensor.strides_bytes;
  if (strides.size() != tensor.shape.size()) {
    throw std::runtime_error("dense tensor strides must match shape rank");
  }
  std::uint8_t* dst = out.data();
  if (!copy_dense_rows(src, mapped_size, tensor.shape, strides, elem, 0U, &dst)) {
    throw std::runtime_error("dense tensor strided copy failed");
  }
  return out;
}

std::vector<std::uint8_t> copy_raw_tensor_payload(const Tensor& tensor) {
  std::size_t mapped_size = 0;
  const auto* src = tensor_data(tensor, &mapped_size);
  return std::vector<std::uint8_t>(src, src + mapped_size);
}

std::size_t tensor_payload_size(const Tensor& tensor) {
  if (!tensor.planes.empty()) {
    if (tensor.byte_offset < 0 ||
        static_cast<std::size_t>(tensor.byte_offset) > tensor.size_bytes) {
      return 0;
    }
    return tensor.size_bytes - static_cast<std::size_t>(tensor.byte_offset);
  }
  if (!tensor.shape.empty()) {
    return dense_size_bytes(tensor);
  }
  if (tensor.byte_offset < 0 || static_cast<std::size_t>(tensor.byte_offset) > tensor.size_bytes) {
    return 0;
  }
  return tensor.size_bytes - static_cast<std::size_t>(tensor.byte_offset);
}

std::vector<std::uint8_t> copy_tensor_payload(const Tensor& tensor) {
  if (!tensor.planes.empty()) {
    return copy_raw_tensor_payload(tensor);
  }
  if (!tensor.shape.empty()) {
    return copy_dense_tensor_payload(tensor);
  }
  return copy_raw_tensor_payload(tensor);
}

std::vector<std::uint8_t> pack_tensor_payloads(const TensorList& tensors) {
  std::vector<std::uint8_t> out;
  std::size_t total = 0;
  for (const auto& tensor : tensors) {
    total += tensor_payload_size(tensor);
  }
  out.reserve(total);
  for (const auto& tensor : tensors) {
    std::vector<std::uint8_t> bytes = copy_tensor_payload(tensor);
    out.insert(out.end(), bytes.begin(), bytes.end());
  }
  return out;
}

void attach_tensor_set_meta(GstBuffer* buffer, const TensorList& tensors) {
  if (!buffer || tensors.empty()) {
    throw std::runtime_error("tensor-set metadata requires a buffer and at least one tensor");
  }
  if (gst_meta_get_info(SIMA_TENSOR_SET_META_NAME) == nullptr) {
    static const gchar* tags[] = {"memory", "tensor", nullptr};
    if (gst_meta_register_custom(SIMA_TENSOR_SET_META_NAME, tags, nullptr, nullptr, nullptr) ==
        nullptr) {
      throw std::runtime_error("failed to register GstSimaTensorSetMeta");
    }
  }

  std::vector<std::string> names;
  names.reserve(tensors.size());
  std::vector<SimaTensorDescriptorV2> descriptors;
  descriptors.reserve(tensors.size());

  std::int64_t running_offset = 0;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    names.push_back(tensor.route.name.empty() ? "tensor_" + std::to_string(i) : tensor.route.name);
    const std::size_t size_bytes = tensor_payload_size(tensor);

    SimaTensorDescriptorV2 desc{};
    desc.logical_index = static_cast<gint>(i);
    desc.physical_index =
        tensor.route.physical_index >= 0 ? tensor.route.physical_index : static_cast<int>(i);
    desc.backend_output_index = tensor.route.backend_output_index;
    desc.route_slot = tensor.route.route_slot >= 0 ? tensor.route.route_slot : static_cast<gint>(i);
    desc.memory_index = tensor.route.memory_index;
    desc.logical_name_id = static_cast<gint>(i);
    desc.backend_name_id = static_cast<gint>(i);
    desc.segment_name_id = static_cast<gint>(i);
    desc.byte_offset = running_offset;
    desc.size_bytes = size_bytes;
    desc.dtype = tensor_set_dtype(tensor.dtype);
    desc.layout = tensor_set_layout(tensor);
    desc.rank =
        static_cast<guint>(std::min<std::size_t>(tensor.shape.size(), SIMA_TENSOR_SET_MAX_RANK));
    for (guint d = 0; d < desc.rank; ++d) {
      desc.shape[d] = tensor.shape[d];
    }
    const std::vector<std::int64_t> strides =
        tensor.strides_bytes.empty() ? contiguous_strides(tensor.shape, dtype_bytes(tensor.dtype))
                                     : tensor.strides_bytes;
    for (guint d = 0; d < desc.rank && d < strides.size(); ++d) {
      desc.stride_bytes[d] = strides[d];
    }
    descriptors.push_back(desc);
    running_offset = desc.byte_offset + static_cast<std::int64_t>(size_bytes);
  }

  std::vector<gchar*> name_table;
  name_table.reserve(names.size() + 1U);
  for (const auto& name : names) {
    name_table.push_back(const_cast<gchar*>(name.c_str()));
  }
  name_table.push_back(nullptr);

  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME);
  if (!meta) {
    throw std::runtime_error("failed to attach GstSimaTensorSetMeta");
  }

  GBytes* descriptor_bytes =
      g_bytes_new(descriptors.data(), descriptors.size() * sizeof(SimaTensorDescriptorV2));
  GstStructure* structure = gst_custom_meta_get_structure(meta);
  gst_structure_set(
      structure, SIMA_TENSOR_SET_META_FIELD_VERSION, G_TYPE_UINT, SIMA_TENSOR_SET_META_VERSION,
      SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, G_TYPE_UINT, static_cast<guint>(descriptors.size()),
      SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, G_TYPE_UINT,
      static_cast<guint>(sizeof(SimaTensorDescriptorV2)), SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS,
      G_TYPE_BYTES, descriptor_bytes, SIMA_TENSOR_SET_META_FIELD_STAGE_KEY, G_TYPE_STRING,
      "pcie-input", SIMA_TENSOR_SET_META_FIELD_NAME_TABLE, G_TYPE_STRV, name_table.data(), nullptr);
  g_bytes_unref(descriptor_bytes);
}

void free_wrapped_payload(gpointer user_data) {
  auto* holder = static_cast<std::shared_ptr<std::vector<std::uint8_t>>*>(user_data);
  delete holder;
}

struct MappedSample {
  GstSample* sample = nullptr;
  GstBuffer* buffer = nullptr;
  GstMapInfo map{};
  bool mapped = false;

  ~MappedSample() {
    if (mapped && buffer) {
      gst_buffer_unmap(buffer, &map);
    }
    if (sample) {
      gst_sample_unref(sample);
    }
  }
};

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
    tensor.strides_bytes = contiguous_strides(tensor.shape, dtype_bytes(tensor.dtype));
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

void HostPcieChannel::configure(const PcieModelFacts& facts, const int queue, const int card_id) {
  if (running_.load()) {
    throw std::runtime_error("cannot configure HostPcieChannel while running");
  }
  facts_ = facts;
  pcie_queue_ = queue;
  card_id_ = card_id;
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
      throw std::runtime_error("host PCIe channel already running with different caps");
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

  g_object_set(G_OBJECT(appsrc_), "caps", caps, "is-live", TRUE, "do-timestamp", TRUE, "block",
               TRUE, "format", GST_FORMAT_TIME, nullptr);
  gst_caps_unref(caps);

  const std::size_t auto_buffer = std::max(facts_.packed_input_bytes, facts_.packed_output_bytes);
  const guint64 buffer_size =
      static_cast<guint64>(std::max<std::size_t>(auto_buffer, 512U * 1024U));
  g_object_set(G_OBJECT(pciehost_), "buffersize", buffer_size, "card-number", card_id_, "queue",
               pcie_queue_, nullptr);

  g_object_set(G_OBJECT(appsink_), "emit-signals", TRUE, "sync", FALSE, "max-buffers", 256, "drop",
               FALSE, nullptr);
  g_object_set(G_OBJECT(queue_element_), "max-size-buffers", 64, "max-size-bytes", 0,
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
  running_.store(true);
}

void HostPcieChannel::stop() {
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
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_sequences_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(receive_mutex_);
    received_results_.clear();
  }
  receive_cv_.notify_all();
}

bool HostPcieChannel::is_running() const {
  return running_.load();
}

std::uint64_t HostPcieChannel::next_sequence() {
  return ++sequence_;
}

bool HostPcieChannel::push(const TensorList& tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("PCIe payload requires at least one tensor");
  }
  std::vector<std::uint8_t> payload = pack_tensor_payloads(tensors);
  if (payload.empty()) {
    throw std::runtime_error("PCIe payload is empty");
  }
  start_with_caps(caps_for_tensors(tensors));
  return push_bytes(payload, tensors);
}

bool HostPcieChannel::push_bytes(const std::vector<std::uint8_t>& payload,
                                 const TensorList& tensors) {
  if (!running_.load()) {
    throw std::runtime_error("host PCIe channel is not running");
  }
  auto owned = std::make_shared<std::vector<std::uint8_t>>(payload);
  const std::uint64_t seq = next_sequence();
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_sequences_.push_back(seq);
  }
  const auto remove_pending_sequence = [this, seq]() {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    const auto it = std::find(pending_sequences_.begin(), pending_sequences_.end(), seq);
    if (it != pending_sequences_.end()) {
      pending_sequences_.erase(it);
    }
  };

  auto* holder = new std::shared_ptr<std::vector<std::uint8_t>>(owned);
  GstBuffer* buffer =
      gst_buffer_new_wrapped_full(static_cast<GstMemoryFlags>(0), owned->data(), owned->size(), 0,
                                  owned->size(), holder, free_wrapped_payload);
  if (!buffer) {
    delete holder;
    remove_pending_sequence();
    throw std::runtime_error("failed to allocate GstBuffer");
  }
  if (caps_ == tensor_set_caps()) {
    try {
      attach_tensor_set_meta(buffer, tensors);
    } catch (...) {
      gst_buffer_unref(buffer);
      remove_pending_sequence();
      throw;
    }
  }
  GstFlowReturn ret = GST_FLOW_OK;
  g_signal_emit_by_name(appsrc_, "push-buffer", buffer, &ret);
  gst_buffer_unref(buffer);
  if (ret != GST_FLOW_OK) {
    remove_pending_sequence();
    throw std::runtime_error("push-buffer failed: " + std::to_string(static_cast<int>(ret)));
  }
  return true;
}

std::optional<TensorList> HostPcieChannel::pull(const int timeout_ms) {
  std::unique_lock<std::mutex> lock(receive_mutex_);
  if (timeout_ms < 0) {
    receive_cv_.wait(lock, [&] { return !received_results_.empty() || !running_.load(); });
  } else {
    const bool ready = receive_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
      return !received_results_.empty() || !running_.load();
    });
    if (!ready) {
      return std::nullopt;
    }
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

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    if (!pending_sequences_.empty()) {
      pending_sequences_.pop_front();
    }
  }

  TensorList result = tensors_from_output_sample(owner, facts_);

  {
    std::lock_guard<std::mutex> lock(receive_mutex_);
    received_results_.push_back(result);
  }
  receive_cv_.notify_one();

  return GST_FLOW_OK;
}

} // namespace simaai::neat::pcie::internal

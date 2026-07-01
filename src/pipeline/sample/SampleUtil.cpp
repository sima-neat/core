// src/pipeline/internal/SampleUtil.cpp
#include "pipeline/internal/SampleUtil.h"

#include "InputStreamUtil.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"

#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal {

const char* storage_kind_name(simaai::neat::StorageKind kind);

namespace {

constexpr const char* kSampleMetaName = "GstSimaSampleMeta";

const char* plane_role_name(simaai::neat::PlaneRole role) {
  switch (role) {
  case simaai::neat::PlaneRole::Y:
    return "Y";
  case simaai::neat::PlaneRole::U:
    return "U";
  case simaai::neat::PlaneRole::V:
    return "V";
  case simaai::neat::PlaneRole::UV:
    return "UV";
  case simaai::neat::PlaneRole::Unknown:
  default:
    return "Unknown";
  }
}

std::string join_dims(const std::vector<int64_t>& dims, char sep) {
  std::ostringstream ss;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i)
      ss << sep;
    ss << dims[i];
  }
  return ss.str();
}

bool sample_debug_enabled() {
  return env_bool("SIMA_SAMPLE_DEBUG", false);
}

bool sample_bytes_enabled() {
  return env_bool("SIMA_SAMPLE_BYTES", false);
}

struct TensorBufferRuntimeSidecar {
  std::weak_ptr<const TensorBuffer> owner;
  bool has_producer_stream_lifetime = false;
  std::weak_ptr<void> producer_stream_lifetime;
  bool holder_loan_release_attached = false;
  std::weak_ptr<void> zero_copy_loan;
};

std::mutex& tensor_buffer_sidecar_mutex() {
  static std::mutex mu;
  return mu;
}

std::unordered_map<const TensorBuffer*, TensorBufferRuntimeSidecar>& tensor_buffer_sidecars() {
  static std::unordered_map<const TensorBuffer*, TensorBufferRuntimeSidecar> sidecars;
  return sidecars;
}

TensorBufferRuntimeSidecar*
tensor_buffer_sidecar_locked(const std::shared_ptr<TensorBuffer>& storage, bool create) {
  if (!storage) {
    return nullptr;
  }
  auto& sidecars = tensor_buffer_sidecars();
  const TensorBuffer* key = storage.get();
  auto it = sidecars.find(key);
  if (it != sidecars.end()) {
    const auto owner = it->second.owner.lock();
    if (!owner || owner.get() != key) {
      sidecars.erase(it);
      it = sidecars.end();
    }
  }
  if (it == sidecars.end()) {
    if (!create) {
      return nullptr;
    }
    TensorBufferRuntimeSidecar sidecar;
    sidecar.owner = storage;
    it = sidecars.emplace(key, std::move(sidecar)).first;
  }
  return &it->second;
}

void mark_tensor_producer_lifetime(const Tensor& tensor, const std::shared_ptr<void>& lifetime) {
  if (!tensor.storage || tensor.storage->kind != simaai::neat::StorageKind::GstSample ||
      !lifetime) {
    return;
  }
  std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
  auto* sidecar = tensor_buffer_sidecar_locked(tensor.storage, /*create=*/true);
  if (!sidecar) {
    return;
  }
  sidecar->has_producer_stream_lifetime = true;
  sidecar->producer_stream_lifetime = lifetime;
}

bool tensor_has_device_gstsample_holder_local(const Tensor& tensor);

bool tensor_has_device_gstsample_producer_lifetime_local(const Tensor& tensor,
                                                         bool require_expired) {
  if (!tensor.storage || tensor.storage->kind != simaai::neat::StorageKind::GstSample ||
      !tensor_has_device_gstsample_holder_local(tensor)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
  const auto* sidecar = tensor_buffer_sidecar_locked(tensor.storage, /*create=*/false);
  if (!sidecar || !sidecar->has_producer_stream_lifetime) {
    return false;
  }
  return !require_expired || sidecar->producer_stream_lifetime.expired();
}

bool tensor_has_device_gstsample_holder_local(const Tensor& tensor) {
  if (!tensor.storage || tensor.storage->kind != simaai::neat::StorageKind::GstSample ||
      !tensor.storage->holder) {
    return false;
  }
  return tensor.device.type != simaai::neat::DeviceType::CPU ||
         tensor.storage->device.type != simaai::neat::DeviceType::CPU ||
         tensor.storage->sima_mem_target_flags != 0 || !tensor.storage->sima_segments.empty();
}

struct ZeroCopyLoanToken {
  ZeroCopyLoanToken(HolderLoanGatePtr gate_in, std::shared_ptr<void> producer_lifetime_in)
      : gate(std::move(gate_in)), producer_lifetime(std::move(producer_lifetime_in)) {}
  ~ZeroCopyLoanToken() {
    release_once();
  }

  void release_once() noexcept {
    bool expected = false;
    if (released.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
      if (gate) {
        gate->release();
      }
    }
  }

  HolderLoanGatePtr gate;
  std::shared_ptr<void> producer_lifetime;
  std::atomic<bool> released{false};
};

struct GstBufferLoanKeepalive {
  std::vector<std::shared_ptr<void>> loans;
};

GQuark zero_copy_loan_quark() {
  static GQuark quark = g_quark_from_static_string("simaai-neat-zero-copy-loans");
  return quark;
}

void destroy_gst_buffer_loan_keepalive(gpointer data) {
  delete static_cast<GstBufferLoanKeepalive*>(data);
}

void collect_zero_copy_loans_from_sample(const Sample& sample,
                                         std::vector<std::shared_ptr<void>>* out) {
  if (!out) {
    return;
  }
  auto add_tensor = [&](const Tensor& tensor) {
    if (!tensor.storage) {
      return;
    }
    std::shared_ptr<void> loan;
    {
      std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
      const auto* sidecar = tensor_buffer_sidecar_locked(tensor.storage, /*create=*/false);
      if (sidecar) {
        loan = sidecar->zero_copy_loan.lock();
      }
    }
    if (!loan) {
      return;
    }
    const auto found = std::find_if(out->begin(), out->end(), [&](const std::shared_ptr<void>& v) {
      return v.get() == loan.get();
    });
    if (found == out->end()) {
      out->push_back(loan);
    }
  };
  auto walk = [&](auto&& self, const Sample& s) -> void {
    if (sample_has_tensor_list(s)) {
      for (const auto& tensor : s.tensors) {
        add_tensor(tensor);
      }
    }
    if (s.tensor.has_value()) {
      add_tensor(*s.tensor);
    }
    if (s.kind == SampleKind::Bundle) {
      for (const auto& field : s.fields) {
        self(self, field);
      }
    }
  };
  walk(walk, sample);
}

bool restore_preprocess_meta_after_make_writable(GstBuffer* dst, GstBuffer* src, std::string* err) {
  if (!dst || !src || !has_simaai_preprocess_meta(src) || has_simaai_preprocess_meta(dst)) {
    return true;
  }
  return copy_simaai_preprocess_meta(dst, src, err);
}

int tensor_runtime_memory_index(const Tensor& tensor) {
  if (tensor.route.memory_index >= 0) {
    return tensor.route.memory_index;
  }
  if (tensor.route.physical_index >= 0) {
    return tensor.route.physical_index;
  }
  return 0;
}

std::string tensor_runtime_segment_name(const Tensor& tensor) {
  // route.segment_name is the consumer-side identity authored by the contract
  // layer (e.g. "cast_0"/"cast_1" for a multi-IFM MLA's ingress, or
  // "MLA_0_ifm_pack_transform" for a monolithic-pack ingress). It is the name
  // downstream segment-aware buffer resolvers (build_segmented_buffer +
  // gst_simaai_memory_get_segment) need to honor when they map a binding's
  // source_segment_name to a runtime memory slot.
  //
  // The tensor.storage->sima_segments[i] list, by contrast, reflects the
  // UPSTREAM producer's own naming ("memory0" from a cast plugin's default
  // allocation), which is the wrong identity for any multi-tensor bundle.
  // Falling back to that name in the multi-tensor case caused two failure
  // modes:
  //   1. When memory_index was out of range, the previous code clamped to
  //      sima_segments[0] and every tensor reported the same name, making
  //      the segmented buffer's unique-naming step append "#1" / "#2"
  //      suffixes ("memory0", "memory0#1") that no contract binding asks for.
  //   2. Even when memory_index was in range, sima_segments[memory_index].name
  //      was the producer's local label, not the consumer's segment name, so
  //      the resolver still couldn't find the segment by the contract's name.
  //
  // Prefer route.segment_name whenever it is set; fall back to the storage
  // segment name only when route is unset, preserving legacy behavior for
  // tensors that have never been routed.
  if (!tensor.route.segment_name.empty()) {
    return tensor.route.segment_name;
  }
  const int memory_index = tensor_runtime_memory_index(tensor);
  if (tensor.storage && !tensor.storage->sima_segments.empty() && memory_index >= 0 &&
      static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size()) {
    const auto& segment = tensor.storage->sima_segments[static_cast<std::size_t>(memory_index)];
    if (!segment.name.empty()) {
      return segment.name;
    }
  }
  return tensor.route.segment_name;
}

std::size_t tensor_runtime_segment_size(const Tensor& tensor) {
  const int memory_index = tensor_runtime_memory_index(tensor);
  if (tensor.storage && !tensor.storage->sima_segments.empty()) {
    std::size_t segment_index = 0U;
    if (memory_index >= 0 &&
        static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size()) {
      segment_index = static_cast<std::size_t>(memory_index);
    }
    return tensor.storage->sima_segments[segment_index].size_bytes;
  }
  return tensor.storage ? tensor.storage->size_bytes : 0U;
}

int tensor_source_memory_index(const Tensor& tensor, GstBuffer* source_buffer);

std::size_t tensor_transport_span_bytes_for_materialization(const Tensor& tensor) {
  const std::size_t tight_bytes = tensor_bytes_tight(tensor);
  const std::size_t runtime_segment_bytes = tensor_runtime_segment_size(tensor);
  if (runtime_segment_bytes == 0U) {
    return tight_bytes;
  }

  const bool preserve_runtime_segment =
      tensor.byte_offset == 0 &&
      (tensor.semantic.tess.has_value() || runtime_segment_bytes > tight_bytes);
  return preserve_runtime_segment ? runtime_segment_bytes : tight_bytes;
}

bool copy_tensor_transport_payload_to(const Tensor& tensor, std::uint8_t* dst,
                                      std::size_t transport_bytes, std::string* err) {
  const std::size_t logical_bytes = tensor_bytes_tight(tensor);
  if (transport_bytes <= logical_bytes) {
    return copy_tensor_payload_to(tensor, dst, transport_bytes, err);
  }

  std::string source_err;
  GstBuffer* source_buffer = buffer_from_holder_if_gstsample(tensor, &source_err);
  if (!source_buffer) {
    if (err) {
      *err = source_err.empty() ? "tensor transport copy: missing source buffer" : source_err;
    }
    return false;
  }

  const int source_memory_index = tensor_source_memory_index(tensor, source_buffer);
  if (source_memory_index < 0 ||
      static_cast<guint>(source_memory_index) >= gst_buffer_n_memory(source_buffer)) {
    gst_buffer_unref(source_buffer);
    if (err) {
      *err = "tensor transport copy: invalid source memory index";
    }
    return false;
  }

  GstMemory* memory =
      gst_buffer_peek_memory(source_buffer, static_cast<guint>(source_memory_index));
  if (!memory) {
    gst_buffer_unref(source_buffer);
    if (err) {
      *err = "tensor transport copy: missing source memory";
    }
    return false;
  }

  GstMapInfo map{};
  if (!gst_memory_map(memory, &map, GST_MAP_READ)) {
    gst_buffer_unref(source_buffer);
    if (err) {
      *err = "tensor transport copy: map failed";
    }
    return false;
  }

  bool ok = true;
  if (transport_bytes > map.size) {
    ok = false;
    if (err) {
      *err = "tensor transport copy: buffer out of range";
    }
  } else {
    std::memcpy(dst, map.data, transport_bytes);
  }
  gst_memory_unmap(memory, &map);
  gst_buffer_unref(source_buffer);
  return ok;
}

int tensor_source_memory_index(const Tensor& tensor, GstBuffer* source_buffer) {
  if (!source_buffer) {
    return -1;
  }
  const guint source_memory_count = gst_buffer_n_memory(source_buffer);
  if (source_memory_count == 0U) {
    return -1;
  }

  const int runtime_index = tensor_runtime_memory_index(tensor);
  if (runtime_index >= 0 && static_cast<guint>(runtime_index) < source_memory_count) {
    return runtime_index;
  }

  const std::string runtime_segment_name = tensor_runtime_segment_name(tensor);
  if (!runtime_segment_name.empty() && tensor.storage) {
    for (std::size_t i = 0; i < tensor.storage->sima_segments.size(); ++i) {
      const auto& segment = tensor.storage->sima_segments[i];
      if (segment.name == runtime_segment_name && static_cast<guint>(i) < source_memory_count) {
        return static_cast<int>(i);
      }
    }
  }

  if (source_memory_count == 1U) {
    return 0;
  }
  return -1;
}

Sample tensor_sample_from_tensor(const Tensor& tensor, std::size_t /*index*/) {
  Sample field;
  field.kind = SampleKind::TensorSet;
  field.owned = true;
  field.tensors = TensorList{tensor};
  field.payload_type = tensor.semantic.image.has_value() ? PayloadType::Image : PayloadType::Tensor;
  field.media_type =
      tensor.semantic.image.has_value() ? "video/x-raw" : "application/vnd.simaai.tensor";
  if (tensor.semantic.image.has_value()) {
    field.format = Sample::image_format_string(tensor.semantic.image->format);
    field.payload_tag = field.format;
  } else if (tensor.semantic.byte_stream.has_value()) {
    field.format = format_tag_to_string(FormatTag::ByteStream);
    field.payload_tag = field.format;
  }
  field.output_index = tensor.route.logical_index;
  field.logical_output_index = tensor.route.logical_index;
  field.memory_index = pipeline_internal::tensor_runtime_memory_index(tensor);
  field.route_slot = tensor.route.route_slot;
  field.segment_name = pipeline_internal::tensor_runtime_segment_name(tensor);
  field.stream_label = tensor.route.name;
  return field;
}

void log_bundle_field(const Sample& field) {
  std::ostringstream ss;
  const std::string name = !field.stream_label.empty()
                               ? field.stream_label
                               : (field.segment_name.empty() ? "field" : field.segment_name);
  ss << "[SAMPLE] field name=" << name;
  if (!field.caps_string.empty()) {
    ss << " caps=" << field.caps_string;
  }
  if (!sample_has_tensor_list(field) || field.tensors.empty()) {
    ss << " neat=<missing>";
    std::fprintf(stderr, "%s\n", ss.str().c_str());
    return;
  }
  const simaai::neat::Tensor& t = field.tensors.front();
  ss << " " << t.debug_string();
  if (t.storage) {
    ss << " storage=" << storage_kind_name(t.storage->kind) << " size=" << t.storage->size_bytes;
  } else {
    ss << " storage=<none>";
  }
  for (size_t i = 0; i < t.planes.size(); ++i) {
    const simaai::neat::Plane& plane = t.planes[i];
    ss << " plane[" << i << "]=" << plane_role_name(plane.role)
       << " shape=" << join_dims(plane.shape, 'x')
       << " strides=" << join_dims(plane.strides_bytes, ',') << " offset=" << plane.byte_offset;
  }
  std::fprintf(stderr, "%s\n", ss.str().c_str());
}

void log_bundle(const Sample& bundle) {
  std::ostringstream ss;
  const std::size_t count =
      sample_has_tensor_list(bundle) ? bundle.tensors.size() : bundle.fields.size();
  ss << "[SAMPLE] bundle fields=" << count << " frame_id=" << bundle.frame_id;
  if (!bundle.stream_id.empty()) {
    ss << " stream_id=" << bundle.stream_id;
  }
  std::fprintf(stderr, "%s\n", ss.str().c_str());
  if (sample_has_tensor_list(bundle)) {
    for (const auto& tensor : bundle.tensors) {
      Sample field = tensor_sample_from_tensor(tensor, 0);
      log_bundle_field(field);
    }
    return;
  }
  for (const auto& field : bundle.fields) {
    log_bundle_field(field);
  }
}

gint tensor_set_dtype_from_tensor(const Tensor& tensor) {
  switch (tensor.dtype) {
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

gint tensor_set_layout_from_tensor(const Tensor& tensor) {
  switch (tensor.layout) {
  case TensorLayout::HW:
    return SIMA_TENSOR_SET_LAYOUT_HW_V1;
  case TensorLayout::HWC:
    return SIMA_TENSOR_SET_LAYOUT_HWC_V1;
  case TensorLayout::CHW:
    return SIMA_TENSOR_SET_LAYOUT_CHW_V1;
  case TensorLayout::Unknown:
  default:
    return SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1;
  }
}

std::string tensor_set_stage_key_from_tensors(const TensorList& tensors) {
  std::string stage_key;
  for (const auto& tensor : tensors) {
    if (tensor.route.stage_key.empty()) {
      continue;
    }
    if (stage_key.empty()) {
      stage_key = tensor.route.stage_key;
      continue;
    }
    if (stage_key != tensor.route.stage_key) {
      return {};
    }
  }
  return stage_key;
}

bool tensor_buffer_descriptor_from_tensors(const TensorList& tensors, TensorBufferView* out,
                                           std::string* err) {
  if (!out) {
    if (err) {
      *err = "tensor buffer descriptor: missing output storage";
    }
    return false;
  }
  out->stage_key = tensor_set_stage_key_from_tensors(tensors);
  out->tensors.clear();
  if (tensors.empty()) {
    if (err) {
      *err = "tensor buffer descriptor: empty tensor list";
    }
    return false;
  }

  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    if (!tensor.storage) {
      if (err) {
        *err = "tensor buffer descriptor: tensor missing storage";
      }
      return false;
    }

    const int memory_index = tensor_runtime_memory_index(tensor);
    const std::string segment_name = tensor_runtime_segment_name(tensor);
    const std::size_t tensor_bytes = tensor_bytes_tight(tensor);
    if (tensor.byte_offset < 0) {
      if (err) {
        *err = "tensor buffer descriptor: tensor byte offset is negative";
      }
      return false;
    }
    const std::size_t segment_size = tensor_runtime_segment_size(tensor);
    if (segment_size > 0U &&
        static_cast<std::size_t>(tensor.byte_offset) + tensor_bytes > segment_size) {
      if (err) {
        *err = "tensor buffer descriptor: tensor span exceeds runtime segment";
      }
      return false;
    }

    TensorBufferTensorDescriptor descriptor_tensor;
    descriptor_tensor.logical_index =
        tensor.route.logical_index >= 0 ? tensor.route.logical_index : static_cast<int>(i);
    descriptor_tensor.physical_index =
        tensor.route.physical_index >= 0 ? tensor.route.physical_index : memory_index;
    descriptor_tensor.backend_output_index = tensor.route.backend_output_index >= 0
                                                 ? tensor.route.backend_output_index
                                                 : descriptor_tensor.logical_index;
    descriptor_tensor.route_slot =
        tensor.route.route_slot >= 0 ? tensor.route.route_slot : descriptor_tensor.logical_index;
    descriptor_tensor.memory_index = memory_index;
    descriptor_tensor.logical_name =
        !tensor.route.name.empty() ? tensor.route.name
                                   : ("output" + std::to_string(descriptor_tensor.logical_index));
    descriptor_tensor.backend_name = tensor.route.backend_name;
    descriptor_tensor.segment_name = !segment_name.empty()
                                         ? segment_name
                                         : ("memory" + std::to_string(static_cast<std::size_t>(i)));
    descriptor_tensor.byte_offset = tensor.byte_offset;
    descriptor_tensor.size_bytes = tensor_bytes;
    descriptor_tensor.dtype = tensor_set_dtype_from_tensor(tensor);
    descriptor_tensor.layout = tensor_set_layout_from_tensor(tensor);
    descriptor_tensor.shape = tensor.shape;
    descriptor_tensor.stride_bytes = tensor.strides_bytes;
    if (tensor.semantic.quant.has_value()) {
      TensorBufferQuantDescriptor quant;
      const QuantSpec& source = *tensor.semantic.quant;
      quant.axis = source.axis;
      if (!source.scales.empty()) {
        quant.scales.assign(source.scales.begin(), source.scales.end());
      } else {
        quant.scales.push_back(source.scale);
      }
      if (!source.zero_points.empty()) {
        quant.zero_points.assign(source.zero_points.begin(), source.zero_points.end());
      } else {
        quant.zero_points.push_back(source.zero_point);
      }
      descriptor_tensor.quant = std::move(quant);
    }
    out->tensors.push_back(std::move(descriptor_tensor));
  }

  return true;
}

bool tensor_buffer_view_from_handle(simaai::gst::SimaTensorBufferHandle* handle,
                                    TensorBufferView* out, std::string* err) {
  if (!handle || !out) {
    if (err) {
      *err = "tensor buffer view: missing tensorbuffer handle";
    }
    return false;
  }
  out->stage_key = simaai::gst::sima_tensor_buffer_handle_stage_key(handle)
                       ? simaai::gst::sima_tensor_buffer_handle_stage_key(handle)
                       : "";
  out->tensors.clear();
  const gsize tensor_count = simaai::gst::sima_tensor_buffer_handle_logical_tensor_count(handle);
  gsize quant_scale_count = 0U;
  gsize quant_zp_count = 0U;
  const gdouble* quant_scales =
      simaai::gst::sima_tensor_buffer_handle_quant_scales(handle, &quant_scale_count);
  const gint64* quant_zero_points =
      simaai::gst::sima_tensor_buffer_handle_quant_zero_points(handle, &quant_zp_count);
  out->tensors.reserve(tensor_count);
  for (gsize i = 0; i < tensor_count; ++i) {
    SimaTensorDescriptorV2 descriptor{};
    if (!simaai::gst::sima_tensor_buffer_handle_tensor_descriptor(handle, i, &descriptor)) {
      if (err) {
        *err = "tensor buffer view: failed to read tensor descriptor";
      }
      out->tensors.clear();
      return false;
    }
    TensorBufferTensorDescriptor local;
    local.logical_index = descriptor.logical_index;
    local.physical_index = descriptor.physical_index;
    local.backend_output_index = descriptor.backend_output_index;
    local.route_slot = descriptor.route_slot;
    local.memory_index = descriptor.memory_index;
    local.logical_name =
        simaai::gst::sima_tensor_buffer_handle_name_at(handle, descriptor.logical_name_id)
            ? simaai::gst::sima_tensor_buffer_handle_name_at(handle, descriptor.logical_name_id)
            : "";
    local.backend_name =
        simaai::gst::sima_tensor_buffer_handle_name_at(handle, descriptor.backend_name_id)
            ? simaai::gst::sima_tensor_buffer_handle_name_at(handle, descriptor.backend_name_id)
            : "";
    local.segment_name =
        simaai::gst::sima_tensor_buffer_handle_name_at(handle, descriptor.segment_name_id)
            ? simaai::gst::sima_tensor_buffer_handle_name_at(handle, descriptor.segment_name_id)
            : "";
    local.byte_offset = descriptor.byte_offset;
    local.size_bytes = descriptor.size_bytes;
    local.dtype = descriptor.dtype;
    local.layout = descriptor.layout;
    for (guint dim = 0; dim < descriptor.rank; ++dim) {
      local.shape.push_back(descriptor.shape[dim]);
      local.stride_bytes.push_back(descriptor.stride_bytes[dim]);
    }
    if (descriptor.has_quant != 0U) {
      TensorBufferQuantDescriptor quant;
      quant.granularity = descriptor.quant_granularity;
      quant.axis = descriptor.quant_axis;
      const gsize scale_end = descriptor.quant_scales_offset + descriptor.quant_scales_len;
      const gsize zp_end = descriptor.quant_zero_points_offset + descriptor.quant_zero_points_len;
      if (quant_scales && scale_end <= quant_scale_count) {
        quant.scales.assign(quant_scales + descriptor.quant_scales_offset,
                            quant_scales + scale_end);
      }
      if (quant_zero_points && zp_end <= quant_zp_count) {
        quant.zero_points.assign(quant_zero_points + descriptor.quant_zero_points_offset,
                                 quant_zero_points + zp_end);
      }
      local.quant = std::move(quant);
    }
    out->tensors.push_back(std::move(local));
  }
  return true;
}

bool tensor_buffer_view_from_tensors_impl(const TensorList& tensors, TensorBufferView* out,
                                          std::string* err) {
  if (!out) {
    if (err) {
      *err = "tensor buffer view: missing output view";
    }
    return false;
  }
  *out = TensorBufferView{};
  if (tensors.empty()) {
    if (err) {
      *err = "tensor buffer view: empty tensor list";
    }
    return false;
  }

  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    if (!tensor.storage || tensor.storage->kind != simaai::neat::StorageKind::GstSample ||
        !tensor.storage->holder) {
      if (err) {
        *err = "tensor buffer view: tensor is not GstSample-backed";
      }
      return false;
    }

    auto* sample = static_cast<GstSample*>(tensor.storage->holder.get());
    if (!sample || !GST_IS_SAMPLE(sample)) {
      if (err) {
        *err = "tensor buffer view: invalid GstSample holder";
      }
      return false;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
      if (err) {
        *err = "tensor buffer view: missing GstBuffer";
      }
      return false;
    }
    GstCaps* caps = gst_sample_get_caps(sample);

    if (!out->holder) {
      out->holder = tensor.storage->holder;
      out->sample = sample;
      out->buffer = buffer;
      out->caps = caps;
      char* c_err = nullptr;
      simaai::gst::SimaTensorBufferHandle* handle =
          simaai::gst::sima_tensor_buffer_create_view_handle_from_sample(sample, &c_err);
      if (!handle) {
        if (err && c_err) {
          *err = c_err;
        }
        g_free(c_err);
        return false;
      }
      const bool ok = tensor_buffer_view_from_handle(handle, out, err);
      simaai::gst::sima_tensor_buffer_handle_unref(handle);
      g_free(c_err);
      if (!ok) {
        return false;
      }
    } else if (tensor.storage->holder != out->holder || sample != out->sample ||
               buffer != out->buffer) {
      if (err) {
        *err = "tensor buffer view: tensors do not share the same backing sample";
      }
      return false;
    }
  }

  return true;
}

bool tensor_buffer_view_from_sample_impl(const Sample& sample, TensorBufferView* out,
                                         std::string* err) {
  if (!out) {
    if (err) {
      *err = "tensor buffer view: missing output view";
    }
    return false;
  }
  if (sample_has_tensor_list(sample)) {
    return tensor_buffer_view_from_tensors_impl(sample.tensors, out, err);
  }
  if (sample.kind == SampleKind::Bundle) {
    TensorList tensors;
    tensors.reserve(sample.fields.size());
    for (const auto& field : sample.fields) {
      if (!sample_has_tensor_list(field) || field.tensors.empty()) {
        if (err) {
          *err = "tensor buffer view: bundle contains non-tensor field";
        }
        return false;
      }
      tensors.push_back(field.tensors.front());
    }
    return tensor_buffer_view_from_tensors_impl(tensors, out, err);
  }
  if (err) {
    *err = "tensor buffer view: sample does not contain tensor-list payload";
  }
  return false;
}

bool tensor_buffer_descriptor_from_sample_impl(GstSample* sample, TensorBufferView* out,
                                               std::string* err) {
  if (!out) {
    if (err) {
      *err = "tensor buffer view: missing output view";
    }
    return false;
  }
  *out = TensorBufferView{};
  if (!sample) {
    if (err) {
      *err = "tensor buffer view: missing GstSample";
    }
    return false;
  }
  out->holder = std::shared_ptr<void>(
      gst_sample_ref(sample), [](void* p) { gst_sample_unref(static_cast<GstSample*>(p)); });
  out->sample = sample;
  out->buffer = gst_sample_get_buffer(sample);
  out->caps = gst_sample_get_caps(sample);
  char* c_err = nullptr;
  simaai::gst::SimaTensorBufferHandle* handle =
      simaai::gst::sima_tensor_buffer_create_view_handle_from_sample(sample, &c_err);
  if (!handle) {
    if (err && c_err) {
      *err = c_err;
    }
    g_free(c_err);
    return false;
  }
  const bool ok = tensor_buffer_view_from_handle(handle, out, err);
  simaai::gst::sima_tensor_buffer_handle_unref(handle);
  g_free(c_err);
  return ok;
}

bool attach_tensor_set_meta_from_descriptor_view_impl(GstBuffer* buffer,
                                                      const TensorBufferView& descriptor,
                                                      std::string* err) {
  if (!buffer) {
    if (err) {
      *err = "tensor-set meta attach requires destination buffer";
    }
    return false;
  }
  std::vector<std::string> name_table_storage;
  std::vector<SimaTensorDescriptorV2> flat_descriptors;
  std::vector<gdouble> quant_scales;
  std::vector<gint64> quant_zero_points;

  const auto intern_name = [&](const std::string& name) -> gint {
    if (name.empty()) {
      return -1;
    }
    auto it = std::find(name_table_storage.begin(), name_table_storage.end(), name);
    if (it != name_table_storage.end()) {
      return static_cast<gint>(std::distance(name_table_storage.begin(), it));
    }
    name_table_storage.push_back(name);
    return static_cast<gint>(name_table_storage.size() - 1U);
  };

  flat_descriptors.reserve(descriptor.tensors.size());
  for (const auto& tensor : descriptor.tensors) {
    SimaTensorDescriptorV2 flat{};
    flat.logical_index = tensor.logical_index;
    flat.physical_index = tensor.physical_index;
    flat.backend_output_index = tensor.backend_output_index;
    flat.route_slot = tensor.route_slot;
    flat.memory_index = tensor.memory_index;
    flat.logical_name_id = intern_name(tensor.logical_name);
    flat.backend_name_id = intern_name(tensor.backend_name);
    flat.segment_name_id = intern_name(tensor.segment_name);
    flat.byte_offset = tensor.byte_offset;
    flat.size_bytes = tensor.size_bytes;
    flat.dtype = tensor.dtype;
    flat.layout = tensor.layout;
    flat.rank =
        static_cast<guint>(std::min<std::size_t>(tensor.shape.size(), SIMA_TENSOR_SET_MAX_RANK));
    for (guint dim = 0; dim < flat.rank; ++dim) {
      flat.shape[dim] = tensor.shape[dim];
      if (dim < tensor.stride_bytes.size()) {
        flat.stride_bytes[dim] = tensor.stride_bytes[dim];
      }
    }
    if (tensor.quant.has_value()) {
      flat.has_quant = 1U;
      flat.quant_granularity = tensor.quant->granularity;
      flat.quant_axis = tensor.quant->axis;
      flat.quant_scales_offset = static_cast<guint>(quant_scales.size());
      flat.quant_scales_len = static_cast<guint>(tensor.quant->scales.size());
      flat.quant_zero_points_offset = static_cast<guint>(quant_zero_points.size());
      flat.quant_zero_points_len = static_cast<guint>(tensor.quant->zero_points.size());
      quant_scales.insert(quant_scales.end(), tensor.quant->scales.begin(),
                          tensor.quant->scales.end());
      quant_zero_points.insert(quant_zero_points.end(), tensor.quant->zero_points.begin(),
                               tensor.quant->zero_points.end());
    }
    flat_descriptors.push_back(flat);
  }

  std::vector<const gchar*> name_table;
  name_table.reserve(name_table_storage.size() + 1U);
  for (const auto& name : name_table_storage) {
    name_table.push_back(name.c_str());
  }
  name_table.push_back(nullptr);

  char* c_err = nullptr;
  const gboolean ok = simaai::gst::sima_tensor_buffer_attach_meta_flat(
      buffer, descriptor.stage_key.empty() ? nullptr : descriptor.stage_key.c_str(),
      flat_descriptors.data(), flat_descriptors.size(),
      name_table_storage.empty() ? nullptr : name_table.data(),
      quant_scales.empty() ? nullptr : quant_scales.data(), quant_scales.size(),
      quant_zero_points.empty() ? nullptr : quant_zero_points.data(), quant_zero_points.size(),
      &c_err);
  if (!ok) {
    if (err) {
      *err = c_err ? c_err : "tensor-set meta attach failed";
    }
    g_free(c_err);
    return false;
  }
  g_free(c_err);
  return true;
}

void attach_tensor_set_meta_from_tensors_impl(GstBuffer* buffer, const TensorList& tensors) {
  if (!buffer || tensors.empty()) {
    return;
  }
  TensorBufferView descriptor;
  std::string descriptor_err;
  if (!tensor_buffer_descriptor_from_tensors(tensors, &descriptor, &descriptor_err)) {
    return;
  }
  std::string attach_err;
  (void)attach_tensor_set_meta_from_descriptor_view_impl(buffer, descriptor, &attach_err);
}

GstBuffer* buffer_from_tensor_or_copy(const Sample& field, const SampleSpec& spec, std::string* err,
                                      bool allow_zero_copy = true) {
  if (!sample_has_tensor_list(field) || field.tensors.empty()) {
    if (err)
      *err = "Sample field missing tensor";
    return nullptr;
  }
  const simaai::neat::Tensor& t = field.tensors.front();
  const char* name = !field.stream_label.empty()
                         ? field.stream_label.c_str()
                         : (field.segment_name.empty() ? "field" : field.segment_name.c_str());

  auto should_force_copy = [&]() -> bool {
    if (!t.storage || t.storage->kind != simaai::neat::StorageKind::GstSample) {
      return false;
    }
    if (t.byte_offset != 0 || !t.planes.empty()) {
      return true;
    }
    if (!t.storage->holder) {
      return true;
    }
    GstBuffer* holder_buf = buffer_from_tensor_holder(t.storage->holder);
    if (!holder_buf) {
      return true;
    }
    const std::size_t holder_bytes = static_cast<std::size_t>(gst_buffer_get_size(holder_buf));
    gst_buffer_unref(holder_buf);
    if (spec.required_bytes_actual == 0) {
      return true;
    }
    return holder_bytes != spec.required_bytes_actual;
  };

  GstBufferBuildPolicy policy;
  policy.allow_zero_copy = allow_zero_copy && !should_force_copy();
  policy.require_video_meta = true;
  policy.allow_appsrc_pool = false;
  policy.require_contiguous = true;
  policy.allow_device_memory = false;

  GstBuffer* buf = build_gst_buffer_from_tensor(t, spec, policy, err);
  if (!buf)
    return nullptr;

  GstBuffer* source_preproc_meta_buffer = nullptr;
  if (t.storage && t.storage->holder) {
    source_preproc_meta_buffer = buffer_from_tensor_holder(t.storage->holder);
  }
  if (source_preproc_meta_buffer && has_simaai_preprocess_meta(source_preproc_meta_buffer) &&
      !has_simaai_preprocess_meta(buf)) {
    std::string copy_err;
    if (!copy_simaai_preprocess_meta(buf, source_preproc_meta_buffer, &copy_err)) {
      gst_buffer_unref(source_preproc_meta_buffer);
      gst_buffer_unref(buf);
      if (err) {
        *err = copy_err.empty() ? "Sample field preprocess meta copy failed" : copy_err;
      }
      return nullptr;
    }
  }
  if (!has_simaai_preprocess_meta(buf) && t.semantic.preprocess.has_value() &&
      !write_simaai_preprocess_meta(buf, *t.semantic.preprocess)) {
    if (source_preproc_meta_buffer) {
      gst_buffer_unref(source_preproc_meta_buffer);
    }
    gst_buffer_unref(buf);
    if (err) {
      *err = "Sample field preprocess meta apply failed";
    }
    return nullptr;
  }
  // Plan 1: framework owns preproc_axis_perm. The plugin writes geometry/affine/flags
  // but never axis_perm; merge the user-resolved layout_convert.perm onto the
  // existing meta here without overwriting plugin-authored fields.
  if (t.semantic.preprocess.has_value() && t.semantic.preprocess->has_axis_perm()) {
    merge_simaai_preprocess_axis_perm(buf, t.semantic.preprocess->axis_perm);
  }
  if (source_preproc_meta_buffer) {
    gst_buffer_unref(source_preproc_meta_buffer);
  }
  if (sample_bytes_enabled()) {
    const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(buf));
    std::fprintf(stderr, "[SAMPLE] field name=%s source=adapter bytes=%zu\n", name, buf_bytes);
  }
  return buf;
}

bool add_field_to_list(GValue* list, const Sample& field, GstBuffer* buf,
                       const std::string& buffer_name) {
  if (!list || !buf)
    return false;
  const char* field_name =
      !field.stream_label.empty()
          ? field.stream_label.c_str()
          : (field.segment_name.empty() ? "field" : field.segment_name.c_str());
  const char* caps = field.caps_string.empty() ? nullptr : field.caps_string.c_str();

  GstStructure* entry = gst_structure_new("simaai-sample-field", "name", G_TYPE_STRING, field_name,
                                          "buffer", GST_TYPE_BUFFER, buf, "buffer-name",
                                          G_TYPE_STRING, buffer_name.c_str(), nullptr);
  if (caps) {
    gst_structure_set(entry, "caps", G_TYPE_STRING, caps, nullptr);
  }
  if (field.logical_output_index >= 0) {
    gst_structure_set(entry, "logical-output-index", G_TYPE_INT, field.logical_output_index,
                      nullptr);
  }
  if (field.memory_index >= 0) {
    gst_structure_set(entry, "memory-index", G_TYPE_INT, field.memory_index, nullptr);
  }
  if (field.route_slot >= 0) {
    gst_structure_set(entry, "route-slot", G_TYPE_INT, field.route_slot, nullptr);
  }
  if (!field.segment_name.empty()) {
    gst_structure_set(entry, "segment-name", G_TYPE_STRING, field.segment_name.c_str(), nullptr);
  }

  GValue entry_val = G_VALUE_INIT;
  g_value_init(&entry_val, GST_TYPE_STRUCTURE);
  g_value_take_boxed(&entry_val, entry);

  gst_value_list_append_value(list, &entry_val);
  g_value_unset(&entry_val);
  return true;
}

bool ensure_sima_meta_fields(GstBuffer* buffer, const std::optional<int64_t>& frame_id,
                             const std::optional<int64_t>& input_seq,
                             const std::optional<int64_t>& orig_input_seq,
                             const std::optional<std::string>& stream_id,
                             const std::optional<std::string>& buffer_name) {
  if (!buffer)
    return false;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  }
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return false;

  gint64 existing_frame_id = -1;
  const bool has_frame_id = gst_structure_get_int64(s, "frame-id", &existing_frame_id);
  const gint64 effective_frame_id =
      frame_id.has_value()
          ? static_cast<gint64>(*frame_id)
          : ((has_frame_id && existing_frame_id >= 0) ? existing_frame_id
                                                      : static_cast<gint64>(next_input_frame_id()));
  if (frame_id.has_value() || !has_frame_id || existing_frame_id < 0) {
    gst_structure_set(s, "frame-id", G_TYPE_INT64, effective_frame_id, nullptr);
  }
  if (input_seq.has_value()) {
    gst_structure_set(s, "input-seq", G_TYPE_INT64, static_cast<gint64>(*input_seq), nullptr);
  } else if (!gst_structure_has_field(s, "input-seq")) {
    gst_structure_set(s, "input-seq", G_TYPE_INT64, effective_frame_id, nullptr);
  }
  if (orig_input_seq.has_value()) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(*orig_input_seq),
                      nullptr);
  } else if (input_seq.has_value()) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(*input_seq), nullptr);
  } else if (!gst_structure_has_field(s, "orig-input-seq")) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, effective_frame_id, nullptr);
  }
  if (stream_id.has_value()) {
    gst_structure_set(s, "stream-id", G_TYPE_STRING, stream_id->c_str(), nullptr);
    gst_structure_set(s, "orig-stream-id", G_TYPE_STRING, stream_id->c_str(), nullptr);
  } else {
    const gchar* existing_stream_id = gst_structure_get_string(s, "stream-id");
    if (!existing_stream_id || !*existing_stream_id) {
      gst_structure_set(s, "stream-id", G_TYPE_STRING, "0", nullptr);
      gst_structure_set(s, "orig-stream-id", G_TYPE_STRING, "0", nullptr);
    } else if (!gst_structure_has_field(s, "orig-stream-id")) {
      gst_structure_set(s, "orig-stream-id", G_TYPE_STRING, existing_stream_id, nullptr);
    }
  }
  if (buffer_name.has_value()) {
    gst_structure_set(s, "buffer-name", G_TYPE_STRING, buffer_name->c_str(), nullptr);
  }
  gint64 existing_buffer_id = 0;
  const bool has_buffer_id = gst_structure_get_int64(s, "buffer-id", &existing_buffer_id);
  gint64 phys_addr = 0;
  if (gst_buffer_n_memory(buffer) > 0U) {
    phys_addr = gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(buffer, 0U));
  }
  if (!has_buffer_id || existing_buffer_id <= 0) {
    gst_structure_set(s, "buffer-id", G_TYPE_INT64, phys_addr, nullptr);
  }
  gint64 existing_buffer_offset = 0;
  if (!gst_structure_get_int64(s, "buffer-offset", &existing_buffer_offset)) {
    gst_structure_set(s, "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0), nullptr);
  }
  guint64 existing_timestamp = 0;
  if (!gst_structure_get_uint64(s, "timestamp", &existing_timestamp)) {
    gst_structure_set(s, "timestamp", G_TYPE_UINT64, static_cast<guint64>(0), nullptr);
  }
  return true;
}

bool try_collect_shared_bundle_backing(const Sample& bundle, GstBuffer** out_buffer,
                                       GstCaps** out_caps) {
  if (!out_buffer || !out_caps) {
    return false;
  }
  *out_buffer = nullptr;
  *out_caps = nullptr;
  TensorBufferView view;
  std::string view_err;
  if (!tensor_buffer_view_from_sample_impl(bundle, &view, &view_err) || !view.buffer) {
    return false;
  }
  char* c_err = nullptr;
  *out_buffer = simaai::gst::sima_tensor_buffer_clone_envelope(view.buffer, &c_err);
  g_free(c_err);
  if (!*out_buffer) {
    return false;
  }
  *out_caps = view.caps ? gst_caps_ref(view.caps) : nullptr;
  return true;
}

bool build_tensor_set_envelope_caps(const Sample& bundle, GstCaps** out_caps, std::string* err) {
  if (!out_caps) {
    if (err) {
      *err = "tensor-set envelope caps require output storage";
    }
    return false;
  }
  *out_caps = nullptr;
  try {
    const SampleSpec spec = derive_sample_spec_or_throw(bundle);
    if (!spec.caps_string.empty()) {
      *out_caps = gst_caps_from_string(spec.caps_string.c_str());
      if (!*out_caps) {
        if (err) {
          *err = "tensor-set envelope caps parse failed";
        }
        return false;
      }
    }
    return true;
  } catch (const std::exception& e) {
    if (err) {
      *err = e.what();
    }
    return false;
  }
}

bool copy_bundle_tensor_preprocess_meta(GstBuffer* dst, const TensorList& tensors,
                                        std::string* err) {
  if (!dst) {
    if (err) {
      *err = "tensor-set preprocess meta copy requires destination buffer";
    }
    return false;
  }
  for (const auto& tensor : tensors) {
    if (!tensor.storage || !tensor.storage->holder) {
      continue;
    }
    std::string source_err;
    GstBuffer* src = buffer_from_holder_if_gstsample(tensor, &source_err);
    if (!src) {
      continue;
    }
    if (!has_simaai_preprocess_meta(src)) {
      gst_buffer_unref(src);
      continue;
    }
    std::string copy_err;
    const bool ok = copy_simaai_preprocess_meta(dst, src, &copy_err);
    gst_buffer_unref(src);
    if (!ok) {
      if (err) {
        *err = copy_err.empty() ? "tensor-set preprocess meta copy failed" : copy_err;
      }
      return false;
    }
    return true;
  }
  return true;
}

bool try_build_multi_source_tensor_set_backing(const Sample& bundle, GstBuffer** out_buffer,
                                               GstCaps** out_caps, std::string* err) {
  if (!out_buffer || !out_caps) {
    return false;
  }
  *out_buffer = nullptr;
  *out_caps = nullptr;
  if (!sample_has_tensor_list(bundle) || bundle.tensors.size() <= 1U) {
    return false;
  }
  if (!build_tensor_set_envelope_caps(bundle, out_caps, err)) {
    return false;
  }

  GstBuffer* assembled = gst_buffer_new();
  if (!assembled) {
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = "tensor-set multi-source backing allocation failed";
    }
    return false;
  }

  GstBuffer* first_source_buffer = nullptr;
  bool appended_memory = false;
  for (const auto& tensor : bundle.tensors) {
    if (!tensor.storage || tensor.storage->kind != simaai::neat::StorageKind::GstSample ||
        !tensor.storage->holder) {
      if (first_source_buffer) {
        gst_buffer_unref(first_source_buffer);
      }
      gst_buffer_unref(assembled);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      return false;
    }

    std::string source_err;
    GstBuffer* source_buffer = buffer_from_holder_if_gstsample(tensor, &source_err);
    if (!source_buffer) {
      if (first_source_buffer) {
        gst_buffer_unref(first_source_buffer);
      }
      gst_buffer_unref(assembled);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = source_err.empty() ? "tensor-set multi-source missing source buffer" : source_err;
      }
      return false;
    }
    if (!first_source_buffer) {
      first_source_buffer = gst_buffer_ref(source_buffer);
    }

    const int source_memory_index = tensor_source_memory_index(tensor, source_buffer);
    if (source_memory_index < 0 ||
        static_cast<guint>(source_memory_index) >= gst_buffer_n_memory(source_buffer)) {
      gst_buffer_unref(source_buffer);
      if (first_source_buffer) {
        gst_buffer_unref(first_source_buffer);
      }
      gst_buffer_unref(assembled);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = "tensor-set multi-source tensor references invalid source memory index";
      }
      return false;
    }

    GstMemory* memory =
        gst_buffer_peek_memory(source_buffer, static_cast<guint>(source_memory_index));
    if (!memory) {
      gst_buffer_unref(source_buffer);
      if (first_source_buffer) {
        gst_buffer_unref(first_source_buffer);
      }
      gst_buffer_unref(assembled);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = "tensor-set multi-source missing source memory";
      }
      return false;
    }

    gst_buffer_append_memory(assembled, gst_memory_ref(memory));
    appended_memory = true;
    gst_buffer_unref(source_buffer);
  }

  if (!appended_memory) {
    if (first_source_buffer) {
      gst_buffer_unref(first_source_buffer);
    }
    gst_buffer_unref(assembled);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = "tensor-set multi-source backing has no source memories";
    }
    return false;
  }

  if (first_source_buffer) {
    GST_BUFFER_PTS(assembled) = GST_BUFFER_PTS(first_source_buffer);
    GST_BUFFER_DTS(assembled) = GST_BUFFER_DTS(first_source_buffer);
    GST_BUFFER_DURATION(assembled) = GST_BUFFER_DURATION(first_source_buffer);
    GST_BUFFER_OFFSET(assembled) = GST_BUFFER_OFFSET(first_source_buffer);
    GST_BUFFER_OFFSET_END(assembled) = GST_BUFFER_OFFSET_END(first_source_buffer);
    GST_MINI_OBJECT_FLAGS(assembled) = GST_MINI_OBJECT_FLAGS(first_source_buffer);
    gst_buffer_unref(first_source_buffer);
  }

  std::string preprocess_err;
  if (!copy_bundle_tensor_preprocess_meta(assembled, bundle.tensors, &preprocess_err)) {
    gst_buffer_unref(assembled);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = preprocess_err.empty() ? "tensor-set multi-source preprocess meta failed"
                                    : preprocess_err;
    }
    return false;
  }

  *out_buffer = assembled;
  return true;
}

struct TensorSetSegmentMaterialization {
  std::string buffer_name;
  GstBuffer* buffer = nullptr;
  std::size_t size_bytes = 0U;
};

void release_tensor_set_segments(std::vector<TensorSetSegmentMaterialization>* fields) {
  if (!fields) {
    return;
  }
  for (auto& entry : *fields) {
    if (entry.buffer) {
      gst_buffer_unref(entry.buffer);
      entry.buffer = nullptr;
    }
  }
}

std::string
unique_tensor_set_segment_name(const std::string& requested_name,
                               const std::vector<TensorSetSegmentMaterialization>& existing,
                               std::size_t index) {
  const std::string base =
      requested_name.empty() ? ("memory" + std::to_string(index)) : requested_name;
  auto is_taken = [&](const std::string& candidate) {
    return std::any_of(existing.begin(), existing.end(),
                       [&](const TensorSetSegmentMaterialization& field) {
                         return field.buffer_name == candidate;
                       });
  };
  if (!is_taken(base)) {
    return base;
  }
  for (std::size_t suffix = 1U;; ++suffix) {
    const std::string candidate = base + "#" + std::to_string(suffix);
    if (!is_taken(candidate)) {
      return candidate;
    }
  }
}

std::optional<std::string> packed_tensor_set_parent_segment_name(const Sample& bundle) {
  if (!sample_has_tensor_list(bundle) || bundle.tensors.size() <= 1U) {
    return std::nullopt;
  }

  int shared_physical_index = -1;
  std::string parent_segment_name;
  for (const auto& tensor : bundle.tensors) {
    if (tensor.route.physical_index < 0) {
      return std::nullopt;
    }
    if (shared_physical_index < 0) {
      shared_physical_index = tensor.route.physical_index;
    } else if (shared_physical_index != tensor.route.physical_index) {
      return std::nullopt;
    }

    if (!tensor.route.segment_name.empty()) {
      if (parent_segment_name.empty()) {
        parent_segment_name = tensor.route.segment_name;
      } else if (parent_segment_name != tensor.route.segment_name) {
        return std::nullopt;
      }
    }
  }

  if (parent_segment_name.empty() && !bundle.segment_name.empty()) {
    parent_segment_name = bundle.segment_name;
  }
  if (parent_segment_name.empty()) {
    return std::nullopt;
  }
  return parent_segment_name;
}

bool tensor_buffer_descriptor_from_packed_tensors(const TensorList& tensors,
                                                  const std::string& parent_segment_name,
                                                  TensorBufferView* out, std::string* err) {
  if (!out) {
    if (err) {
      *err = "tensor buffer descriptor: missing packed output storage";
    }
    return false;
  }
  out->stage_key = tensor_set_stage_key_from_tensors(tensors);
  out->tensors.clear();
  if (tensors.empty()) {
    if (err) {
      *err = "tensor buffer descriptor: packed tensor list is empty";
    }
    return false;
  }

  const int shared_physical_index =
      tensors.front().route.physical_index >= 0 ? tensors.front().route.physical_index : 0;
  std::size_t running_offset = 0U;
  out->tensors.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    const std::size_t logical_bytes = tensor_bytes_tight(tensor);
    const std::size_t transport_bytes = tensor_transport_span_bytes_for_materialization(tensor);
    if (logical_bytes == 0U || transport_bytes == 0U) {
      if (err) {
        *err = "tensor buffer descriptor: packed tensor has zero byte size";
      }
      return false;
    }

    TensorBufferTensorDescriptor descriptor_tensor;
    descriptor_tensor.logical_index =
        tensor.route.logical_index >= 0 ? tensor.route.logical_index : static_cast<int>(i);
    descriptor_tensor.physical_index = shared_physical_index;
    descriptor_tensor.backend_output_index = tensor.route.backend_output_index >= 0
                                                 ? tensor.route.backend_output_index
                                                 : descriptor_tensor.logical_index;
    descriptor_tensor.route_slot =
        tensor.route.route_slot >= 0 ? tensor.route.route_slot : descriptor_tensor.logical_index;
    descriptor_tensor.memory_index = 0;
    descriptor_tensor.logical_name =
        !tensor.route.name.empty() ? tensor.route.name
                                   : ("output" + std::to_string(descriptor_tensor.logical_index));
    descriptor_tensor.backend_name = tensor.route.backend_name;
    descriptor_tensor.segment_name = parent_segment_name;
    descriptor_tensor.byte_offset = running_offset;
    descriptor_tensor.size_bytes = logical_bytes;
    descriptor_tensor.dtype = tensor_set_dtype_from_tensor(tensor);
    descriptor_tensor.layout = tensor_set_layout_from_tensor(tensor);
    descriptor_tensor.shape = tensor.shape;
    descriptor_tensor.stride_bytes = tensor.strides_bytes;
    if (tensor.semantic.quant.has_value()) {
      TensorBufferQuantDescriptor quant;
      const QuantSpec& source = *tensor.semantic.quant;
      quant.axis = source.axis;
      if (!source.scales.empty()) {
        quant.scales.assign(source.scales.begin(), source.scales.end());
      } else {
        quant.scales.push_back(source.scale);
      }
      if (!source.zero_points.empty()) {
        quant.zero_points.assign(source.zero_points.begin(), source.zero_points.end());
      } else {
        quant.zero_points.push_back(source.zero_point);
      }
      descriptor_tensor.quant = std::move(quant);
    }
    out->tensors.push_back(std::move(descriptor_tensor));
    if (transport_bytes > (std::numeric_limits<std::size_t>::max() - running_offset)) {
      if (err) {
        *err = "tensor buffer descriptor: packed tensor offsets overflow";
      }
      return false;
    }
    running_offset += transport_bytes;
  }
  return true;
}

bool attach_tensor_set_meta_from_packed_tensors(GstBuffer* buffer, const TensorList& tensors,
                                                const std::string& parent_segment_name,
                                                std::string* err) {
  TensorBufferView descriptor;
  std::string descriptor_err;
  if (!tensor_buffer_descriptor_from_packed_tensors(tensors, parent_segment_name, &descriptor,
                                                    &descriptor_err)) {
    if (err) {
      *err = descriptor_err.empty() ? "tensor-set packed descriptor build failed" : descriptor_err;
    }
    return false;
  }
  return attach_tensor_set_meta_from_descriptor_view_impl(buffer, descriptor, err);
}

bool build_packed_tensor_set_backing(const Sample& bundle, const std::string& parent_segment_name,
                                     GstBuffer** out_buffer, GstCaps** out_caps, std::string* err) {
  if (!out_buffer || !out_caps) {
    if (err) {
      *err = "tensor-set packed backing missing output pointers";
    }
    return false;
  }
  *out_buffer = nullptr;
  *out_caps = nullptr;
  if (!sample_has_tensor_list(bundle) || bundle.tensors.empty()) {
    if (err) {
      *err = "tensor-set packed backing requires tensor list";
    }
    return false;
  }
  if (!build_tensor_set_envelope_caps(bundle, out_caps, err)) {
    return false;
  }

  std::size_t total_bytes = 0U;
  std::vector<std::size_t> tensor_transport_bytes;
  tensor_transport_bytes.reserve(bundle.tensors.size());
  for (const auto& tensor : bundle.tensors) {
    const std::size_t bytes = tensor_transport_span_bytes_for_materialization(tensor);
    if (bytes == 0U) {
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = "tensor-set packed backing encountered zero-sized tensor";
      }
      return false;
    }
    if (bytes > (std::numeric_limits<std::size_t>::max() - total_bytes)) {
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = "tensor-set packed backing size overflow";
      }
      return false;
    }
    tensor_transport_bytes.push_back(bytes);
    total_bytes += bytes;
  }

  GstBuffer* source_buffer =
      gst_buffer_new_allocate(nullptr, static_cast<gsize>(total_bytes), nullptr);
  if (!source_buffer) {
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = "tensor-set packed backing allocation failed";
    }
    return false;
  }

  GstMapInfo map{};
  if (!gst_buffer_map(source_buffer, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(source_buffer);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = "tensor-set packed backing map failed";
    }
    return false;
  }

  std::size_t running_offset = 0U;
  for (std::size_t i = 0; i < bundle.tensors.size(); ++i) {
    std::string copy_err;
    if (!copy_tensor_transport_payload_to(bundle.tensors[i],
                                          static_cast<std::uint8_t*>(map.data) + running_offset,
                                          tensor_transport_bytes[i], &copy_err)) {
      gst_buffer_unmap(source_buffer, &map);
      gst_buffer_unref(source_buffer);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = copy_err.empty() ? "tensor-set packed backing tensor copy failed" : copy_err;
      }
      return false;
    }
    running_offset += tensor_transport_bytes[i];
  }
  gst_buffer_unmap(source_buffer, &map);

  simaai::gst::SimaTensorBufferBuildSegmentV1 segment{};
  segment.name = parent_segment_name.c_str();
  segment.source_buffer = source_buffer;
  segment.copy_bytes = static_cast<gsize>(total_bytes);

  GstBuffer* segmented = nullptr;
  char* c_err = nullptr;
  const gboolean ok =
      simaai::gst::sima_tensor_buffer_build_segmented_buffer(&segment, 1U, &segmented, &c_err);
  gst_buffer_unref(source_buffer);
  if (!ok || !segmented) {
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = c_err ? c_err : "tensor-set packed segmented backing allocation failed";
    }
    g_free(c_err);
    return false;
  }
  g_free(c_err);

  std::string preprocess_err;
  if (!copy_bundle_tensor_preprocess_meta(segmented, bundle.tensors, &preprocess_err)) {
    gst_buffer_unref(segmented);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = preprocess_err.empty() ? "tensor-set packed preprocess meta failed" : preprocess_err;
    }
    return false;
  }

  std::string attach_err;
  if (!attach_tensor_set_meta_from_packed_tensors(segmented, bundle.tensors, parent_segment_name,
                                                  &attach_err)) {
    gst_buffer_unref(segmented);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = attach_err.empty() ? "tensor-set packed meta attach failed" : attach_err;
    }
    return false;
  }

  *out_buffer = segmented;
  return true;
}

bool build_materialized_tensor_set_backing(const Sample& bundle, GstBuffer** out_buffer,
                                           GstCaps** out_caps, std::string* err) {
  if (!out_buffer || !out_caps) {
    if (err) {
      *err = "tensor-set materialized backing missing output pointers";
    }
    return false;
  }
  *out_buffer = nullptr;
  *out_caps = nullptr;
  if (!sample_has_tensor_list(bundle) || bundle.tensors.empty()) {
    if (err) {
      *err = "tensor-set materialized backing requires tensor list";
    }
    return false;
  }
  if (!build_tensor_set_envelope_caps(bundle, out_caps, err)) {
    return false;
  }

  std::vector<TensorSetSegmentMaterialization> fields;
  fields.reserve(bundle.tensors.size());
  for (std::size_t i = 0; i < bundle.tensors.size(); ++i) {
    const auto& tensor = bundle.tensors[i];
    const std::size_t tensor_bytes = tensor_bytes_tight(tensor);
    if (tensor_bytes == 0U) {
      release_tensor_set_segments(&fields);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = "tensor-set materialized backing encountered zero-sized tensor";
      }
      return false;
    }

    GstBuffer* source_buffer =
        gst_buffer_new_allocate(nullptr, static_cast<gsize>(tensor_bytes), nullptr);
    if (!source_buffer) {
      release_tensor_set_segments(&fields);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = "tensor-set materialized backing buffer allocation failed";
      }
      return false;
    }
    GstMapInfo map{};
    if (!gst_buffer_map(source_buffer, &map, GST_MAP_WRITE)) {
      gst_buffer_unref(source_buffer);
      release_tensor_set_segments(&fields);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = "tensor-set materialized backing buffer map failed";
      }
      return false;
    }
    std::string copy_err;
    if (!copy_tensor_payload_to(tensor, static_cast<std::uint8_t*>(map.data), tensor_bytes,
                                &copy_err)) {
      gst_buffer_unmap(source_buffer, &map);
      gst_buffer_unref(source_buffer);
      release_tensor_set_segments(&fields);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = copy_err.empty() ? "tensor-set materialized backing tensor copy failed" : copy_err;
      }
      return false;
    }
    gst_buffer_unmap(source_buffer, &map);

    const std::string requested_name =
        !tensor_runtime_segment_name(tensor).empty()
            ? tensor_runtime_segment_name(tensor)
            : (!tensor.route.name.empty() ? tensor.route.name : std::string("memory"));
    TensorSetSegmentMaterialization entry;
    entry.buffer_name = unique_tensor_set_segment_name(requested_name, fields, i);
    entry.buffer = source_buffer;
    entry.size_bytes = tensor_bytes;
    fields.push_back(std::move(entry));
  }

  std::vector<simaai::gst::SimaTensorBufferBuildSegmentV1> segments;
  segments.reserve(fields.size());
  for (const auto& field : fields) {
    simaai::gst::SimaTensorBufferBuildSegmentV1 segment{};
    segment.name = field.buffer_name.c_str();
    segment.source_buffer = field.buffer;
    segment.copy_bytes = static_cast<gsize>(field.size_bytes);
    segments.push_back(std::move(segment));
  }

  GstBuffer* segmented = nullptr;
  char* c_err = nullptr;
  const gboolean ok = simaai::gst::sima_tensor_buffer_build_segmented_buffer(
      segments.data(), segments.size(), &segmented, &c_err);
  if (!ok || !segmented) {
    release_tensor_set_segments(&fields);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = c_err ? c_err : "tensor-set materialized backing allocation failed";
    }
    g_free(c_err);
    return false;
  }
  g_free(c_err);

  std::string preprocess_err;
  if (!copy_bundle_tensor_preprocess_meta(segmented, bundle.tensors, &preprocess_err)) {
    gst_buffer_unref(segmented);
    release_tensor_set_segments(&fields);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = preprocess_err.empty() ? "tensor-set materialized preprocess meta failed"
                                    : preprocess_err;
    }
    return false;
  }

  attach_tensor_set_meta_from_tensors(segmented, bundle.tensors);
  release_tensor_set_segments(&fields);
  *out_buffer = segmented;
  return true;
}

struct BundleTensorFieldMaterialization {
  Sample field;
  SampleSpec spec;
  std::string buffer_name;
  GstBuffer* buffer = nullptr;
};

std::string unique_segment_name(const std::string& requested_name,
                                const std::vector<BundleTensorFieldMaterialization>& existing,
                                std::size_t index) {
  const std::string base =
      requested_name.empty() ? ("memory" + std::to_string(index)) : requested_name;
  auto is_taken = [&](const std::string& candidate) {
    return std::any_of(existing.begin(), existing.end(),
                       [&](const BundleTensorFieldMaterialization& field) {
                         return field.buffer_name == candidate;
                       });
  };
  if (!is_taken(base)) {
    return base;
  }
  for (std::size_t suffix = 1U;; ++suffix) {
    const std::string candidate = base + "#" + std::to_string(suffix);
    if (!is_taken(candidate)) {
      return candidate;
    }
  }
}

bool tensor_buffer_descriptor_from_materialized_fields(
    GstBuffer* buffer, const Sample& bundle,
    const std::vector<BundleTensorFieldMaterialization>& fields, TensorBufferView* out,
    std::string* err) {
  if (!out) {
    if (err) {
      *err = "tensor buffer descriptor: missing output storage";
    }
    return false;
  }
  out->stage_key.clear();
  out->tensors.clear();
  if (!sample_has_tensor_list(bundle)) {
    if (err) {
      *err = "tensor buffer descriptor: bundle tensor list expected";
    }
    return false;
  }
  if (bundle.tensors.size() != fields.size()) {
    if (err) {
      *err = "tensor buffer descriptor: rebuilt field count mismatch";
    }
    return false;
  }

  out->stage_key = tensor_set_stage_key_from_tensors(bundle.tensors);
  out->tensors.reserve(fields.size());
  // `build_segmented_bundle_backing()` is the non-packed transport path. Even if
  // GStreamer represents that backing with a single GstMemory, the consumer-
  // visible tensor contract still needs one memory/segment slot per field.
  const bool use_parent_span_layout = false;
  const std::string parent_segment_name = (!fields.empty() && !fields.front().buffer_name.empty())
                                              ? fields.front().buffer_name
                                              : "memory0";
  std::size_t running_offset = 0U;
  for (std::size_t i = 0; i < bundle.tensors.size(); ++i) {
    const Tensor& tensor = bundle.tensors[i];
    const auto& field = fields[i];
    if (field.spec.required_bytes_actual <= 0) {
      if (err) {
        *err = "tensor buffer descriptor: rebuilt field has invalid byte size";
      }
      return false;
    }

    TensorBufferTensorDescriptor descriptor_tensor;
    descriptor_tensor.logical_index =
        tensor.route.logical_index >= 0 ? tensor.route.logical_index : static_cast<int>(i);
    descriptor_tensor.physical_index =
        tensor.route.physical_index >= 0 ? tensor.route.physical_index : static_cast<int>(i);
    descriptor_tensor.backend_output_index = tensor.route.backend_output_index >= 0
                                                 ? tensor.route.backend_output_index
                                                 : descriptor_tensor.logical_index;
    descriptor_tensor.route_slot =
        tensor.route.route_slot >= 0 ? tensor.route.route_slot : descriptor_tensor.logical_index;
    descriptor_tensor.memory_index = use_parent_span_layout ? 0 : static_cast<int>(i);
    descriptor_tensor.logical_name =
        !tensor.route.name.empty() ? tensor.route.name
                                   : ("output" + std::to_string(descriptor_tensor.logical_index));
    descriptor_tensor.backend_name = tensor.route.backend_name;
    descriptor_tensor.segment_name =
        use_parent_span_layout ? parent_segment_name : field.buffer_name;
    descriptor_tensor.byte_offset = use_parent_span_layout ? running_offset : 0U;
    descriptor_tensor.size_bytes = static_cast<std::size_t>(field.spec.required_bytes_actual);
    descriptor_tensor.dtype = tensor_set_dtype_from_tensor(tensor);
    descriptor_tensor.layout = tensor_set_layout_from_tensor(tensor);
    descriptor_tensor.shape = tensor.shape;
    descriptor_tensor.stride_bytes = tensor.strides_bytes;
    if (tensor.semantic.quant.has_value()) {
      TensorBufferQuantDescriptor quant;
      const QuantSpec& source = *tensor.semantic.quant;
      quant.axis = source.axis;
      if (!source.scales.empty()) {
        quant.scales.assign(source.scales.begin(), source.scales.end());
      } else {
        quant.scales.push_back(source.scale);
      }
      if (!source.zero_points.empty()) {
        quant.zero_points.assign(source.zero_points.begin(), source.zero_points.end());
      } else {
        quant.zero_points.push_back(source.zero_point);
      }
      descriptor_tensor.quant = std::move(quant);
    }
    if (use_parent_span_layout) {
      const std::size_t field_bytes = static_cast<std::size_t>(field.spec.required_bytes_actual);
      if (field_bytes > (std::numeric_limits<std::size_t>::max() - running_offset)) {
        if (err) {
          *err = "tensor buffer descriptor: materialized field offsets overflow";
        }
        return false;
      }
      running_offset += field_bytes;
    }
    out->tensors.push_back(std::move(descriptor_tensor));
  }

  return true;
}

bool attach_tensor_set_meta_from_materialized_fields(
    GstBuffer* buffer, const Sample& bundle,
    const std::vector<BundleTensorFieldMaterialization>& fields, std::string* err) {
  TensorBufferView descriptor;
  std::string descriptor_err;
  if (!tensor_buffer_descriptor_from_materialized_fields(buffer, bundle, fields, &descriptor,
                                                         &descriptor_err)) {
    if (err) {
      *err = descriptor_err.empty() ? "tensor buffer descriptor: materialized fields failed"
                                    : descriptor_err;
    }
    return false;
  }
  std::vector<std::string> name_table_storage;
  std::vector<SimaTensorDescriptorV2> flat_descriptors;
  std::vector<gdouble> quant_scales;
  std::vector<gint64> quant_zero_points;

  const auto intern_name = [&](const std::string& name) -> gint {
    if (name.empty()) {
      return -1;
    }
    auto it = std::find(name_table_storage.begin(), name_table_storage.end(), name);
    if (it != name_table_storage.end()) {
      return static_cast<gint>(std::distance(name_table_storage.begin(), it));
    }
    name_table_storage.push_back(name);
    return static_cast<gint>(name_table_storage.size() - 1U);
  };

  flat_descriptors.reserve(descriptor.tensors.size());
  for (const auto& tensor : descriptor.tensors) {
    SimaTensorDescriptorV2 flat{};
    flat.logical_index = tensor.logical_index;
    flat.physical_index = tensor.physical_index;
    flat.backend_output_index = tensor.backend_output_index;
    flat.route_slot = tensor.route_slot;
    flat.memory_index = tensor.memory_index;
    flat.logical_name_id = intern_name(tensor.logical_name);
    flat.backend_name_id = intern_name(tensor.backend_name);
    flat.segment_name_id = intern_name(tensor.segment_name);
    flat.byte_offset = tensor.byte_offset;
    flat.size_bytes = tensor.size_bytes;
    flat.dtype = tensor.dtype;
    flat.layout = tensor.layout;
    flat.rank =
        static_cast<guint>(std::min<std::size_t>(tensor.shape.size(), SIMA_TENSOR_SET_MAX_RANK));
    for (guint dim = 0; dim < flat.rank; ++dim) {
      flat.shape[dim] = tensor.shape[dim];
      if (dim < tensor.stride_bytes.size()) {
        flat.stride_bytes[dim] = tensor.stride_bytes[dim];
      }
    }
    if (tensor.quant.has_value()) {
      flat.has_quant = 1U;
      flat.quant_granularity = tensor.quant->granularity;
      flat.quant_axis = tensor.quant->axis;
      flat.quant_scales_offset = static_cast<guint>(quant_scales.size());
      flat.quant_scales_len = static_cast<guint>(tensor.quant->scales.size());
      flat.quant_zero_points_offset = static_cast<guint>(quant_zero_points.size());
      flat.quant_zero_points_len = static_cast<guint>(tensor.quant->zero_points.size());
      quant_scales.insert(quant_scales.end(), tensor.quant->scales.begin(),
                          tensor.quant->scales.end());
      quant_zero_points.insert(quant_zero_points.end(), tensor.quant->zero_points.begin(),
                               tensor.quant->zero_points.end());
    }
    flat_descriptors.push_back(flat);
  }

  std::vector<const gchar*> name_table;
  name_table.reserve(name_table_storage.size() + 1U);
  for (const auto& name : name_table_storage) {
    name_table.push_back(name.c_str());
  }
  name_table.push_back(nullptr);

  char* c_err = nullptr;
  const gboolean ok = simaai::gst::sima_tensor_buffer_attach_meta_flat(
      buffer, descriptor.stage_key.empty() ? nullptr : descriptor.stage_key.c_str(),
      flat_descriptors.data(), flat_descriptors.size(),
      name_table_storage.empty() ? nullptr : name_table.data(),
      quant_scales.empty() ? nullptr : quant_scales.data(), quant_scales.size(),
      quant_zero_points.empty() ? nullptr : quant_zero_points.data(), quant_zero_points.size(),
      &c_err);
  if (!ok) {
    if (err) {
      *err = c_err ? c_err : "tensor buffer descriptor: meta attach failed";
    }
    g_free(c_err);
    return false;
  }
  g_free(c_err);
  return true;
}

void release_bundle_tensor_fields(std::vector<BundleTensorFieldMaterialization>* fields) {
  if (!fields) {
    return;
  }
  for (auto& entry : *fields) {
    if (entry.buffer) {
      gst_buffer_unref(entry.buffer);
      entry.buffer = nullptr;
    }
  }
}

bool collect_bundle_tensor_fields(const Sample& bundle,
                                  std::vector<BundleTensorFieldMaterialization>* out_fields,
                                  GstCaps** out_caps, std::string* err, bool allow_zero_copy) {
  if (!out_fields) {
    if (err) {
      *err = "bundle field collection missing output vector";
    }
    return false;
  }
  if (out_caps) {
    *out_caps = nullptr;
  }

  auto append_field = [&](Sample field) -> bool {
    SampleSpec spec;
    std::string caps_err;
    if (!derive_field_spec(field, &spec, &caps_err)) {
      if (err) {
        *err = caps_err.empty() ? "Sample field caps missing" : caps_err;
      }
      return false;
    }

    std::string field_err;
    GstBuffer* buf = buffer_from_tensor_or_copy(field, spec, &field_err, allow_zero_copy);
    if (!buf) {
      if (err) {
        *err = field_err.empty() ? "Sample field buffer failed" : field_err;
      }
      return false;
    }
    buf = gst_buffer_make_writable(buf);
    if (!buf) {
      if (err) {
        *err = "Sample field buffer not writable";
      }
      return false;
    }

    if (out_caps && !*out_caps && !spec.caps_string.empty()) {
      *out_caps = gst_caps_from_string(spec.caps_string.c_str());
    }

    BundleTensorFieldMaterialization entry;
    entry.field = std::move(field);
    entry.spec = std::move(spec);
    const std::string requested_name =
        !entry.field.segment_name.empty()
            ? entry.field.segment_name
            : (!entry.field.stream_label.empty() ? entry.field.stream_label : std::string("field"));
    entry.buffer_name = unique_segment_name(requested_name, *out_fields, out_fields->size());
    entry.buffer = buf;
    out_fields->push_back(std::move(entry));
    return true;
  };

  if (sample_has_tensor_list(bundle)) {
    out_fields->reserve(bundle.tensors.size());
    for (const auto& tensor : bundle.tensors) {
      Sample field = tensor_sample_from_tensor(tensor, 0);
      field.owned = bundle.owned;
      field.payload_type =
          tensor.semantic.image.has_value() ? PayloadType::Image : PayloadType::Tensor;
      field.media_type =
          tensor.semantic.image.has_value() ? "video/x-raw" : "application/vnd.simaai.tensor";
      if (!append_field(std::move(field))) {
        return false;
      }
    }
    return true;
  }

  out_fields->reserve(bundle.fields.size());
  for (const auto& field : bundle.fields) {
    if (!append_field(field)) {
      return false;
    }
  }
  return true;
}

bool build_segmented_bundle_backing(const Sample& bundle, GstBuffer** out_buffer,
                                    GstCaps** out_caps, std::string* err,
                                    bool allow_zero_copy = true) {
  if (!out_buffer || !out_caps) {
    if (err) {
      *err = "bundle segmented backing missing output pointers";
    }
    return false;
  }
  *out_buffer = nullptr;
  *out_caps = nullptr;

  std::vector<BundleTensorFieldMaterialization> fields;
  if (!collect_bundle_tensor_fields(bundle, &fields, out_caps, err, allow_zero_copy)) {
    release_bundle_tensor_fields(&fields);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    return false;
  }
  if (fields.empty()) {
    release_bundle_tensor_fields(&fields);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    return false;
  }

  std::vector<simaai::gst::SimaTensorBufferBuildSegmentV1> segments;
  segments.reserve(fields.size());
  for (const auto& field : fields) {
    simaai::gst::SimaTensorBufferBuildSegmentV1 segment{};
    segment.name = field.buffer_name.c_str();
    segment.source_buffer = field.buffer;
    segment.copy_bytes = static_cast<gsize>(field.spec.required_bytes_actual);
    segments.push_back(std::move(segment));
  }

  GstBuffer* segmented = nullptr;
  char* c_err = nullptr;
  if (!simaai::gst::sima_tensor_buffer_build_segmented_buffer(segments.data(), segments.size(),
                                                              &segmented, &c_err) ||
      !segmented) {
    release_bundle_tensor_fields(&fields);
    if (*out_caps) {
      gst_caps_unref(*out_caps);
      *out_caps = nullptr;
    }
    if (err) {
      *err = c_err ? c_err : "bundle segmented backing allocation failed";
    }
    g_free(c_err);
    return false;
  }
  g_free(c_err);

  if (!has_simaai_preprocess_meta(segmented)) {
    for (const auto& field : fields) {
      if (!field.buffer || !has_simaai_preprocess_meta(field.buffer)) {
        continue;
      }
      std::string copy_err;
      if (!copy_simaai_preprocess_meta(segmented, field.buffer, &copy_err)) {
        gst_buffer_unref(segmented);
        release_bundle_tensor_fields(&fields);
        if (*out_caps) {
          gst_caps_unref(*out_caps);
          *out_caps = nullptr;
        }
        if (err) {
          *err =
              copy_err.empty() ? "bundle segmented backing preprocess meta copy failed" : copy_err;
        }
        return false;
      }
      break;
    }
  }

  if (sample_has_tensor_list(bundle)) {
    std::string attach_err;
    if (!attach_tensor_set_meta_from_materialized_fields(segmented, bundle, fields, &attach_err)) {
      gst_buffer_unref(segmented);
      release_bundle_tensor_fields(&fields);
      if (*out_caps) {
        gst_caps_unref(*out_caps);
        *out_caps = nullptr;
      }
      if (err) {
        *err = attach_err.empty() ? "bundle segmented backing tensor-set meta failed" : attach_err;
      }
      return false;
    }
  }

  release_bundle_tensor_fields(&fields);
  *out_buffer = segmented;
  return true;
}

} // namespace

void attach_tensor_set_meta_from_tensors(GstBuffer* buffer, const TensorList& tensors) {
  attach_tensor_set_meta_from_tensors_impl(buffer, tensors);
}

bool sample_has_device_gstsample_producer_lifetime(const Sample& sample, bool require_expired) {
  if (sample_has_tensor_list(sample)) {
    for (const auto& tensor : sample.tensors) {
      if (tensor_has_device_gstsample_producer_lifetime_local(tensor, require_expired)) {
        return true;
      }
    }
  }
  if (sample.tensor.has_value() &&
      tensor_has_device_gstsample_producer_lifetime_local(*sample.tensor, require_expired)) {
    return true;
  }
  if (sample.kind == SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (sample_has_device_gstsample_producer_lifetime(field, require_expired)) {
        return true;
      }
    }
  }
  return false;
}

std::string cross_run_zero_copy_sample_error(const char* where) {
  return std::string(where ? where : "Run::push") +
         ": Cannot pass this zero-copy frame into another running graph. The frame was produced "
         "by another graph and must carry a live zero-copy loan before it can be reused safely. "
         "Use the normal public output path so the runtime can attach a loan, keep producer and "
         "consumer in one graph, or request owned/copy output for unsupported boundaries.";
}

bool sample_has_device_gstsample_holder(const Sample& sample) {
  if (sample_has_tensor_list(sample)) {
    for (const auto& tensor : sample.tensors) {
      if (tensor_has_device_gstsample_holder_local(tensor)) {
        return true;
      }
    }
  }
  if (sample.tensor.has_value() && tensor_has_device_gstsample_holder_local(*sample.tensor)) {
    return true;
  }
  if (sample.kind == SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (sample_has_device_gstsample_holder(field)) {
        return true;
      }
    }
  }
  return false;
}

int count_distinct_device_gstsample_holders(const Sample& sample) {
  std::vector<const void*> seen_storage;
  auto add_tensor = [&](const Tensor& tensor) {
    if (!tensor_has_device_gstsample_holder_local(tensor)) {
      return;
    }
    const void* key = tensor.storage.get();
    if (std::find(seen_storage.begin(), seen_storage.end(), key) == seen_storage.end()) {
      seen_storage.push_back(key);
    }
  };
  auto walk = [&](auto&& self, const Sample& s) -> void {
    if (sample_has_tensor_list(s)) {
      for (const auto& tensor : s.tensors) {
        add_tensor(tensor);
      }
    }
    if (s.tensor.has_value()) {
      add_tensor(*s.tensor);
    }
    if (s.kind == SampleKind::Bundle) {
      for (const auto& field : s.fields) {
        self(self, field);
      }
    }
  };
  walk(walk, sample);
  return static_cast<int>(seen_storage.size());
}

bool attach_zero_copy_loan_to_sample(const Sample& sample, const HolderLoanGatePtr& gate,
                                     std::string* err) {
  if (!gate || !gate->enabled()) {
    if (err) {
      *err = "producer output is not loan-managed";
    }
    return false;
  }
  std::vector<const void*> seen_storage;
  std::vector<std::shared_ptr<ZeroCopyLoanToken>> acquired;
  struct AcquiredStorage {
    std::shared_ptr<TensorBuffer> storage;
    std::shared_ptr<void> original_holder;
  };
  std::vector<AcquiredStorage> acquired_storage;
  auto attach_tensor = [&](const Tensor& tensor) -> bool {
    if (!tensor_has_device_gstsample_holder_local(tensor)) {
      return true;
    }
    const void* key = tensor.storage.get();
    if (std::find(seen_storage.begin(), seen_storage.end(), key) != seen_storage.end()) {
      return true;
    }
    seen_storage.push_back(key);
    std::weak_ptr<void> producer_lifetime_weak;
    {
      std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
      auto* sidecar = tensor_buffer_sidecar_locked(tensor.storage, /*create=*/true);
      if (!sidecar) {
        if (err) {
          *err = "zero-copy output loan bookkeeping is unavailable";
        }
        return false;
      }
      if (auto existing = sidecar->zero_copy_loan.lock()) {
        return true;
      }
      if (sidecar->has_producer_stream_lifetime) {
        producer_lifetime_weak = sidecar->producer_stream_lifetime;
      }
    }
    if (!gate->try_acquire()) {
      if (err) {
        *err = "zero-copy output loan credits are exhausted";
      }
      return false;
    }
    std::shared_ptr<void> producer_lifetime = producer_lifetime_weak.lock();
    auto loan = std::make_shared<ZeroCopyLoanToken>(gate, std::move(producer_lifetime));
    auto original_holder = tensor.storage->holder;
    tensor.storage->holder =
        std::shared_ptr<void>(original_holder.get(), [original_holder, loan](void*) mutable {
          original_holder.reset();
          loan.reset();
        });
    {
      std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
      auto* sidecar = tensor_buffer_sidecar_locked(tensor.storage, /*create=*/true);
      if (sidecar) {
        sidecar->zero_copy_loan = loan;
      }
    }
    acquired.push_back(std::move(loan));
    acquired_storage.push_back({tensor.storage, std::move(original_holder)});
    return true;
  };
  bool ok = true;
  auto walk = [&](auto&& self, const Sample& s) -> void {
    if (!ok) {
      return;
    }
    if (sample_has_tensor_list(s)) {
      for (const auto& tensor : s.tensors) {
        if (!attach_tensor(tensor)) {
          ok = false;
          return;
        }
      }
    }
    if (s.tensor.has_value() && !attach_tensor(*s.tensor)) {
      ok = false;
      return;
    }
    if (s.kind == SampleKind::Bundle) {
      for (const auto& field : s.fields) {
        self(self, field);
        if (!ok) {
          return;
        }
      }
    }
  };
  walk(walk, sample);
  if (!ok) {
    for (auto& storage : acquired_storage) {
      if (storage.storage) {
        storage.storage->holder = std::move(storage.original_holder);
        std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
        auto* sidecar = tensor_buffer_sidecar_locked(storage.storage, /*create=*/false);
        if (sidecar) {
          sidecar->zero_copy_loan.reset();
        }
      }
    }
    for (auto& loan : acquired) {
      if (loan) {
        loan->release_once();
      }
    }
    return false;
  }
  return true;
}

bool sample_has_transferable_zero_copy_loan(const Sample& sample, std::string* reason) {
  bool saw_producer_backed_tensor = false;
  bool ok = true;
  auto check_tensor = [&](const Tensor& tensor) {
    if (!tensor_has_device_gstsample_producer_lifetime_local(tensor, /*require_expired=*/false)) {
      return;
    }
    saw_producer_backed_tensor = true;
    std::weak_ptr<void> producer_lifetime;
    std::weak_ptr<void> zero_copy_loan;
    {
      std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
      const auto* sidecar = tensor_buffer_sidecar_locked(tensor.storage, /*create=*/false);
      if (sidecar) {
        producer_lifetime = sidecar->producer_stream_lifetime;
        zero_copy_loan = sidecar->zero_copy_loan;
      }
    }
    if (producer_lifetime.expired()) {
      ok = false;
      if (reason) {
        *reason = "the producer graph is no longer running";
      }
      return;
    }
    if (zero_copy_loan.expired()) {
      ok = false;
      if (reason) {
        *reason = "the frame does not carry a live zero-copy loan";
      }
    }
  };
  auto walk = [&](auto&& self, const Sample& s) -> void {
    if (!ok) {
      return;
    }
    if (sample_has_tensor_list(s)) {
      for (const auto& tensor : s.tensors) {
        check_tensor(tensor);
        if (!ok) {
          return;
        }
      }
    }
    if (s.tensor.has_value()) {
      check_tensor(*s.tensor);
    }
    if (s.kind == SampleKind::Bundle) {
      for (const auto& field : s.fields) {
        self(self, field);
        if (!ok) {
          return;
        }
      }
    }
  };
  walk(walk, sample);
  if (!saw_producer_backed_tensor && reason) {
    *reason = "the sample does not contain a producer-owned zero-copy frame";
  }
  return ok && saw_producer_backed_tensor;
}

void attach_zero_copy_loans_to_gst_buffer(GstBuffer* buffer, const Sample& sample) {
  if (!buffer) {
    return;
  }
  std::vector<std::shared_ptr<void>> loans;
  collect_zero_copy_loans_from_sample(sample, &loans);
  if (loans.empty()) {
    return;
  }
  auto* keepalive = static_cast<GstBufferLoanKeepalive*>(
      gst_mini_object_get_qdata(GST_MINI_OBJECT(buffer), zero_copy_loan_quark()));
  if (!keepalive) {
    keepalive = new GstBufferLoanKeepalive();
    gst_mini_object_set_qdata(GST_MINI_OBJECT(buffer), zero_copy_loan_quark(), keepalive,
                              destroy_gst_buffer_loan_keepalive);
  }
  for (const auto& loan : loans) {
    const auto found =
        std::find_if(keepalive->loans.begin(), keepalive->loans.end(),
                     [&](const std::shared_ptr<void>& v) { return v.get() == loan.get(); });
    if (found == keepalive->loans.end()) {
      keepalive->loans.push_back(loan);
    }
  }
}

void attach_holder_release_to_sample(const Sample& sample, std::function<void()> on_release) {
  if (!on_release) {
    return;
  }
  auto shared_release = std::make_shared<std::function<void()>>(std::move(on_release));
  auto seen = std::make_shared<std::vector<const void*>>();
  auto attach_tensor = [&](const Tensor& tensor) {
    if (!tensor_has_device_gstsample_holder_local(tensor)) {
      return;
    }
    const void* key = tensor.storage.get();
    if (std::find(seen->begin(), seen->end(), key) != seen->end()) {
      return;
    }
    seen->push_back(key);
    {
      std::lock_guard<std::mutex> lock(tensor_buffer_sidecar_mutex());
      auto* sidecar = tensor_buffer_sidecar_locked(tensor.storage, /*create=*/true);
      if (!sidecar || sidecar->holder_loan_release_attached) {
        return;
      }
      sidecar->holder_loan_release_attached = true;
    }
    auto original_holder = tensor.storage->holder;
    tensor.storage->holder = std::shared_ptr<void>(
        original_holder.get(), [original_holder, shared_release](void*) mutable {
          original_holder.reset();
          if (*shared_release) {
            (*shared_release)();
          }
        });
  };
  auto walk = [&](auto&& self, const Sample& s) -> void {
    if (sample_has_tensor_list(s)) {
      for (const auto& tensor : s.tensors) {
        attach_tensor(tensor);
      }
    }
    if (s.tensor.has_value()) {
      attach_tensor(*s.tensor);
    }
    if (s.kind == SampleKind::Bundle) {
      for (const auto& field : s.fields) {
        self(self, field);
      }
    }
  };
  walk(walk, sample);
}

void mark_sample_producer_stream_lifetime(Sample& sample, std::shared_ptr<void> lifetime_token) {
  if (!lifetime_token) {
    return;
  }
  auto walk = [&](auto&& self, Sample& s) -> void {
    if (s.tensor.has_value()) {
      mark_tensor_producer_lifetime(*s.tensor, lifetime_token);
    }
    for (auto& tensor : s.tensors) {
      mark_tensor_producer_lifetime(tensor, lifetime_token);
    }
    for (auto& field : s.fields) {
      self(self, field);
    }
  };
  walk(walk, sample);
}

bool build_bundled_input_gst_buffer(const TensorList& tensors, GstBuffer** out_buffer,
                                    std::string* err) {
  if (!out_buffer) {
    if (err)
      *err = "bundled input: missing out_buffer";
    return false;
  }
  *out_buffer = nullptr;
  if (tensors.empty()) {
    if (err)
      *err = "bundled input: empty tensor list";
    return false;
  }

  // Path A: produce ONE GstSimaaiSegmentMemory with N segments, one per tensor.
  // The plugin's job_builder peeks memory[0] and walks the sima allocator's
  // segment table (gst_simaai_memory_get_segment_count / _at) to find each
  // logical input's carrier — the symmetric counterpart of how the post side
  // (detessdequant) emits multi-output buffers. N appended GstMemory objects
  // would not be readable through that API and would mis-bind at dispatch
  // validation. The trade-off vs. zero-copy: we copy CPU tensor bytes into
  // device-accessible segments. The framework holds the host-side payload as
  // CPU memory (FP32 image_l, image_uv) which the MLA/CVU cannot read directly,
  // so a copy was unavoidable; the legacy branch_sessions path materialized
  // the same bytes via per-ingress casttess output buffers.

  // 1. Describe per-tensor sizes + names for the segmented allocation.
  GstSimaaiAllocationParams params;
  gst_simaai_memory_allocation_params_init(&params);
  gst_allocation_params_init(&params.parent);

  std::vector<std::string> segment_names;
  segment_names.reserve(tensors.size());
  std::vector<std::size_t> segment_bytes;
  segment_bytes.reserve(tensors.size());
  gsize total_size = 0;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& t = tensors[i];
    if (!t.storage || !t.storage->data) {
      if (err)
        *err = std::string("bundled input: tensor ") + std::to_string(i) + " has no CPU data";
      return false;
    }
    if (!t.is_dense() || !t.is_contiguous()) {
      if (err)
        *err =
            std::string("bundled input: tensor ") + std::to_string(i) + " is not dense/contiguous";
      return false;
    }
    const std::size_t bytes = t.dense_bytes_tight();
    if (bytes == 0U) {
      if (err)
        *err = std::string("bundled input: tensor ") + std::to_string(i) + " has zero dense bytes";
      return false;
    }
    std::string seg_name = t.route.segment_name;
    if (seg_name.empty()) {
      seg_name = std::string("ifm") + std::to_string(i);
    }
    segment_names.push_back(std::move(seg_name));
    segment_bytes.push_back(bytes);
    if (!gst_simaai_memory_allocation_params_add_segment(&params, static_cast<gsize>(bytes),
                                                         segment_names.back().c_str())) {
      if (err)
        *err = std::string("bundled input: failed to add segment ") + std::to_string(i);
      return false;
    }
    total_size += static_cast<gsize>(bytes);
  }

  // 2. Allocate ONE GstSimaaiSegmentMemory with N segments via the standard
  //    sima allocator. The allocator constructs N simaai_memory_t handles
  //    backed by one segmented allocation (simaai_memory_alloc_segments_flags).
  GstAllocator* allocator = gst_simaai_memory_get_segment_allocator();
  if (!allocator) {
    if (err)
      *err = "bundled input: simaai segment allocator unavailable";
    return false;
  }
  GstBuffer* assembled = gst_buffer_new_allocate(allocator, total_size,
                                                 reinterpret_cast<GstAllocationParams*>(&params));
  gst_object_unref(allocator);
  if (!assembled) {
    if (err)
      *err = "bundled input: gst_buffer_new_allocate failed";
    return false;
  }
  GstMemory* assembled_memory = gst_buffer_peek_memory(assembled, 0U);
  if (!assembled_memory) {
    gst_buffer_unref(assembled);
    if (err)
      *err = "bundled input: assembled buffer missing memory";
    return false;
  }
  if (multi_io_bundled_debug_enabled()) {
    std::fprintf(stderr, "[bundled] assembled buf=%p mem=%p total=%zu n_mem=%u tensors=%zu\n",
                 static_cast<void*>(assembled), static_cast<void*>(assembled_memory),
                 static_cast<std::size_t>(total_size), gst_buffer_n_memory(assembled),
                 tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      void* seg = gst_simaai_memory_get_segment(assembled_memory, segment_names[i].c_str());
      std::fprintf(stderr, "[bundled]  seg[%zu] name=%s bytes=%zu seg_ptr=%p\n", i,
                   segment_names[i].c_str(), segment_bytes[i], seg);
    }
  }

  // 3. Copy each tensor's bytes into the matching segment by name.
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    void* segment = gst_simaai_memory_get_segment(assembled_memory, segment_names[i].c_str());
    if (!segment) {
      gst_buffer_unref(assembled);
      if (err)
        *err = std::string("bundled input: segment lookup failed for '") + segment_names[i] +
               "' at index " + std::to_string(i);
      return false;
    }
    const auto& t = tensors[i];
    const auto* src = static_cast<const std::uint8_t*>(t.storage->data) +
                      static_cast<std::size_t>(std::max<std::int64_t>(t.byte_offset, 0));
    std::string copy_err;
    if (!pipeline_internal::copy_into_simaai_segment_memory(segment, src, segment_bytes[i],
                                                            &copy_err)) {
      gst_buffer_unref(assembled);
      if (err)
        *err = std::string("bundled input: segment copy failed at index ") + std::to_string(i) +
               ": " + copy_err;
      return false;
    }
  }

  attach_tensor_set_meta_from_tensors_impl(assembled, tensors);
  *out_buffer = assembled;
  return true;
}

bool tensor_buffer_view_from_tensors(const TensorList& tensors, TensorBufferView* out,
                                     std::string* err) {
  return tensor_buffer_view_from_tensors_impl(tensors, out, err);
}

bool tensor_buffer_view_from_sample(const Sample& sample, TensorBufferView* out, std::string* err) {
  return tensor_buffer_view_from_sample_impl(sample, out, err);
}

bool tensor_buffer_descriptor_from_sample(GstSample* sample, TensorBufferView* out,
                                          std::string* err) {
  return tensor_buffer_descriptor_from_sample_impl(sample, out, err);
}

std::shared_ptr<void> make_sample_holder_from_bundle(const Sample& bundle, std::string* err,
                                                     bool allow_zero_copy) {
  if (!sample_has_tensor_list(bundle) && bundle.kind != SampleKind::Bundle) {
    if (err)
      *err = "Sample tensor-list or bundle payload expected";
    return {};
  }
  if (!sample_has_tensor_list(bundle) && bundle.fields.empty()) {
    if (err)
      *err = "Sample multi-output payload has no fields";
    return {};
  }
  if (sample_debug_enabled()) {
    log_bundle(bundle);
  }

  GstBuffer* sample_buf = nullptr;
  GstCaps* sample_caps = nullptr;
  if (sample_has_tensor_list(bundle)) {
    GstBuffer* shared_buffer = nullptr;
    GstCaps* shared_caps = nullptr;
    if (allow_zero_copy &&
        try_collect_shared_bundle_backing(bundle, &shared_buffer, &shared_caps)) {
      sample_buf = gst_buffer_ref(shared_buffer);
      gst_buffer_unref(shared_buffer);
      sample_caps = shared_caps;
      if (sample_debug_enabled() && sample_buf) {
        std::fprintf(stderr, "[SAMPLE] tensor-set reusing shared backing buffer bytes=%zu\n",
                     static_cast<size_t>(gst_buffer_get_size(sample_buf)));
      }
      // Plan 1: framework owns preproc_axis_perm. Plugin-written meta on the
      // shared backing buffer carries geometry/affine/flags but never
      // axis_perm; merge the user-resolved layout_convert.perm onto it
      // here without overwriting plugin-authored fields.
      for (const Tensor& t : bundle.tensors) {
        if (t.semantic.preprocess.has_value() && t.semantic.preprocess->has_axis_perm()) {
          (void)merge_simaai_preprocess_axis_perm(sample_buf, t.semantic.preprocess->axis_perm);
          break;
        }
      }
    } else if (bundle.tensors.size() == 1U && bundle.fields.empty()) {
      Sample first_field = tensor_sample_from_tensor(bundle.tensors.front(), 0);
      first_field.owned = bundle.owned;
      if (bundle.payload_type != PayloadType::Auto) {
        first_field.payload_type = bundle.payload_type;
      }
      if (!bundle.media_type.empty()) {
        first_field.media_type = bundle.media_type;
        if (first_field.payload_type == PayloadType::Auto) {
          first_field.payload_type = payload_type_from_media_type(bundle.media_type);
        }
      }
      if (!bundle.payload_tag.empty()) {
        first_field.payload_tag = bundle.payload_tag;
      }
      if (!bundle.format.empty()) {
        first_field.format = bundle.format;
      }

      SampleSpec direct_spec;
      std::string direct_err;
      if (!derive_field_spec(first_field, &direct_spec, &direct_err)) {
        if (err) {
          *err = direct_err.empty() ? "Sample single tensor field spec failed" : direct_err;
        }
        return {};
      }
      sample_buf =
          buffer_from_tensor_or_copy(first_field, direct_spec, &direct_err, allow_zero_copy);
      if (!sample_buf) {
        if (err) {
          *err = direct_err.empty() ? "Sample single tensor buffer materialization failed"
                                    : direct_err;
        }
        return {};
      }
      if (!direct_spec.caps_string.empty()) {
        sample_caps = gst_caps_from_string(direct_spec.caps_string.c_str());
      }
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] tensor-set direct single-tensor backing bytes=%zu\n",
                     static_cast<size_t>(gst_buffer_get_size(sample_buf)));
      }
    } else {
      std::string materialized_err;
      const auto packed_parent_segment_name = packed_tensor_set_parent_segment_name(bundle);
      if (multi_io_bundled_debug_enabled()) {
        std::fprintf(stderr,
                     "[bundled-path] tensors=%zu bundle.segment_name='%s' packed_parent='%s'\n",
                     bundle.tensors.size(), bundle.segment_name.c_str(),
                     packed_parent_segment_name.has_value() ? packed_parent_segment_name->c_str()
                                                            : "<nullopt>");
        for (std::size_t i = 0; i < bundle.tensors.size(); ++i) {
          const auto& t = bundle.tensors[i];
          std::fprintf(
              stderr, "[bundled-path]   tensor[%zu] segment='%s' name='%s' phys=%d mem=%d log=%d\n",
              i, t.route.segment_name.c_str(), t.route.name.c_str(), t.route.physical_index,
              t.route.memory_index, t.route.logical_index);
        }
      }
      const bool built =
          packed_parent_segment_name.has_value()
              ? build_packed_tensor_set_backing(bundle, *packed_parent_segment_name, &sample_buf,
                                                &sample_caps, &materialized_err)
              : build_materialized_tensor_set_backing(bundle, &sample_buf, &sample_caps,
                                                      &materialized_err);
      if (!built) {
        if (err) {
          *err = materialized_err.empty() ? "Sample tensor-set materialized backing failed"
                                          : materialized_err;
        }
        return {};
      }
      if (sample_debug_enabled() && sample_buf) {
        std::fprintf(stderr, "[SAMPLE] tensor-set %s tensor backing bytes=%zu\n",
                     packed_parent_segment_name.has_value() ? "packed" : "materialized",
                     static_cast<size_t>(gst_buffer_get_size(sample_buf)));
      }
    }
  } else {
    GstBuffer* shared_buffer = nullptr;
    GstCaps* shared_caps = nullptr;
    std::string segmented_err;
    if (build_segmented_bundle_backing(bundle, &sample_buf, &sample_caps, &segmented_err,
                                       allow_zero_copy)) {
      if (sample_debug_enabled() && sample_buf) {
        std::fprintf(stderr, "[SAMPLE] bundle materialized segmented backing bytes=%zu\n",
                     static_cast<size_t>(gst_buffer_get_size(sample_buf)));
      }
    } else if (allow_zero_copy &&
               try_collect_shared_bundle_backing(bundle, &shared_buffer, &shared_caps)) {
      sample_buf = gst_buffer_ref(shared_buffer);
      gst_buffer_unref(shared_buffer);
      sample_caps = shared_caps;
      if (sample_debug_enabled() && sample_buf) {
        std::fprintf(stderr, "[SAMPLE] bundle reusing shared backing buffer bytes=%zu\n",
                     static_cast<size_t>(gst_buffer_get_size(sample_buf)));
      }
    } else {
      sample_buf = gst_buffer_new();
    }
  }
  if (!sample_buf) {
    if (sample_caps) {
      gst_caps_unref(sample_caps);
    }
    if (err)
      *err = "Sample buffer allocation failed";
    return {};
  }
  if (!gst_buffer_is_writable(sample_buf)) {
    GstBuffer* sample_buf_before_writable = gst_buffer_ref(sample_buf);
    sample_buf = gst_buffer_make_writable(sample_buf);
    if (!sample_buf) {
      gst_buffer_unref(sample_buf_before_writable);
      if (sample_caps) {
        gst_caps_unref(sample_caps);
      }
      if (err) {
        *err = "Sample buffer not writable";
      }
      return {};
    }
    std::string sample_meta_err;
    if (!restore_preprocess_meta_after_make_writable(sample_buf, sample_buf_before_writable,
                                                     &sample_meta_err)) {
      gst_buffer_unref(sample_buf_before_writable);
      gst_buffer_unref(sample_buf);
      if (sample_caps) {
        gst_caps_unref(sample_caps);
      }
      if (err) {
        *err = sample_meta_err.empty() ? "Sample buffer preprocess meta restore failed"
                                       : sample_meta_err;
      }
      return {};
    }
    gst_buffer_unref(sample_buf_before_writable);
  }

  if (sample_has_tensor_list(bundle) && !bundle.tensors.empty()) {
    for (const Tensor& t : bundle.tensors) {
      if (t.semantic.preprocess.has_value()) {
        if (!has_simaai_preprocess_meta(sample_buf)) {
          (void)write_simaai_preprocess_meta(sample_buf, *t.semantic.preprocess);
        }
        if (t.semantic.preprocess->has_axis_perm()) {
          (void)merge_simaai_preprocess_axis_perm(sample_buf, t.semantic.preprocess->axis_perm);
        }
        break;
      }
    }
  }

  if (bundle.media_type == "video/x-raw") {
    Sample first_field;
    bool have_first_field = false;
    if (sample_has_tensor_list(bundle) && !bundle.tensors.empty()) {
      first_field = tensor_sample_from_tensor(bundle.tensors.front(), 0);
      first_field.owned = bundle.owned;
      first_field.payload_type = PayloadType::Image;
      first_field.media_type = "video/x-raw";
      if (bundle.tensors.front().semantic.image.has_value()) {
        first_field.payload_tag =
            Sample::image_format_string(bundle.tensors.front().semantic.image->format);
        first_field.format = first_field.payload_tag;
      }
      have_first_field = true;
    } else if (!bundle.fields.empty()) {
      first_field = bundle.fields.front();
      have_first_field = true;
    }

    if (have_first_field) {
      SampleSpec outer_spec;
      std::string outer_spec_err;
      if (!derive_field_spec(first_field, &outer_spec, &outer_spec_err)) {
        gst_buffer_unref(sample_buf);
        if (sample_caps) {
          gst_caps_unref(sample_caps);
        }
        if (err) {
          *err = outer_spec_err.empty() ? "Sample outer video caps missing" : outer_spec_err;
        }
        return {};
      }
      std::string outer_meta_err;
      if (!attach_video_meta(&sample_buf, outer_spec, &outer_meta_err)) {
        gst_buffer_unref(sample_buf);
        if (sample_caps) {
          gst_caps_unref(sample_caps);
        }
        if (err) {
          *err = outer_meta_err.empty() ? "Sample outer video meta attach failed" : outer_meta_err;
        }
        return {};
      }
    }
  }

  GstCustomMeta* meta = gst_buffer_get_custom_meta(sample_buf, kSampleMetaName);
  if (!meta) {
    meta = gst_buffer_add_custom_meta(sample_buf, kSampleMetaName);
  }
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    gst_buffer_unref(sample_buf);
    if (sample_caps) {
      gst_caps_unref(sample_caps);
    }
    if (err)
      *err = "Sample meta attach failed";
    return {};
  }
  const std::optional<int64_t> bundle_input_seq =
      bundle.input_seq >= 0 ? std::optional<int64_t>(bundle.input_seq) : std::nullopt;
  const std::optional<int64_t> bundle_orig_input_seq =
      bundle.orig_input_seq >= 0 ? std::optional<int64_t>(bundle.orig_input_seq) : bundle_input_seq;
  ensure_sima_meta_fields(
      sample_buf, bundle.frame_id >= 0 ? std::optional<int64_t>(bundle.frame_id) : std::nullopt,
      bundle_input_seq, bundle_orig_input_seq,
      bundle.stream_id.empty() ? std::nullopt : std::optional<std::string>(bundle.stream_id),
      std::optional<std::string>("bundle"));

  if (sample_has_tensor_list(bundle)) {
    gst_structure_remove_field(s, "fields");
    if (!gst_buffer_get_custom_meta(sample_buf, SIMA_TENSOR_SET_META_NAME)) {
      attach_tensor_set_meta_from_tensors(sample_buf, bundle.tensors);
    }
    if (sample_debug_enabled()) {
      std::string accessor_err;
      auto accessor = simaai::gst::TensorBufferAccessor::create(sample_buf, nullptr, &accessor_err);
      std::fprintf(stderr,
                   "[SAMPLE] tensor-set final buffer memories=%u bytes=%zu accessor_ok=%d err=%s\n",
                   static_cast<unsigned>(gst_buffer_n_memory(sample_buf)),
                   static_cast<size_t>(gst_buffer_get_size(sample_buf)), accessor.valid() ? 1 : 0,
                   accessor_err.empty() ? "<empty>" : accessor_err.c_str());
    }
  } else {
    GValue list = G_VALUE_INIT;
    g_value_init(&list, GST_TYPE_LIST);

    for (const auto& field : bundle.fields) {
      SampleSpec spec;
      std::string caps_err;
      if (!derive_field_spec(field, &spec, &caps_err)) {
        gst_buffer_unref(sample_buf);
        if (err)
          *err = caps_err.empty() ? "Sample field caps missing" : caps_err;
        return {};
      }

      std::string field_err;
      GstBuffer* buf = buffer_from_tensor_or_copy(field, spec, &field_err, allow_zero_copy);
      if (!buf) {
        gst_buffer_unref(sample_buf);
        if (err)
          *err = field_err.empty() ? "Sample field buffer failed" : field_err;
        return {};
      }
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] field name=%s caps_string=%s\n",
                     !field.stream_label.empty()
                         ? field.stream_label.c_str()
                         : (field.segment_name.empty() ? "field" : field.segment_name.c_str()),
                     spec.caps_string.empty() ? "<empty>" : spec.caps_string.c_str());
      }
      if (sample_bytes_enabled()) {
        const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(buf));
        std::fprintf(stderr, "[SAMPLE] bundle field=%s buffer-bytes=%zu\n",
                     !field.stream_label.empty()
                         ? field.stream_label.c_str()
                         : (field.segment_name.empty() ? "field" : field.segment_name.c_str()),
                     buf_bytes);
      }
      GstBuffer* field_buf_before_writable = gst_buffer_ref(buf);
      buf = gst_buffer_make_writable(buf);
      if (!buf) {
        gst_buffer_unref(field_buf_before_writable);
        gst_buffer_unref(sample_buf);
        if (err)
          *err = "Sample field buffer not writable";
        return {};
      }
      std::string field_meta_err;
      if (!restore_preprocess_meta_after_make_writable(buf, field_buf_before_writable,
                                                       &field_meta_err)) {
        gst_buffer_unref(field_buf_before_writable);
        gst_buffer_unref(buf);
        gst_buffer_unref(sample_buf);
        if (err) {
          *err = field_meta_err.empty() ? "Sample field preprocess meta restore failed"
                                        : field_meta_err;
        }
        return {};
      }
      gst_buffer_unref(field_buf_before_writable);

      const std::string buffer_name =
          !field.segment_name.empty()
              ? field.segment_name
              : (!field.stream_label.empty() ? field.stream_label : std::string("field"));
      const std::optional<int64_t> field_input_seq =
          bundle.input_seq >= 0
              ? std::optional<int64_t>(bundle.input_seq)
              : (field.input_seq >= 0 ? std::optional<int64_t>(field.input_seq) : std::nullopt);
      const std::optional<int64_t> field_orig_input_seq =
          bundle.orig_input_seq >= 0
              ? std::optional<int64_t>(bundle.orig_input_seq)
              : (field.orig_input_seq >= 0 ? std::optional<int64_t>(field.orig_input_seq)
                                           : field_input_seq);
      update_simaai_meta_fields(
          buf, bundle.frame_id >= 0 ? std::optional<int64_t>(bundle.frame_id) : std::nullopt,
          field_input_seq, field_orig_input_seq,
          bundle.stream_id.empty() ? std::nullopt : std::optional<std::string>(bundle.stream_id),
          buffer_name);
      ensure_sima_meta_fields(
          buf, bundle.frame_id >= 0 ? std::optional<int64_t>(bundle.frame_id) : std::nullopt,
          field_input_seq, field_orig_input_seq,
          bundle.stream_id.empty() ? std::nullopt : std::optional<std::string>(bundle.stream_id),
          std::optional<std::string>(buffer_name));

      Sample field_with_caps = field;
      field_with_caps.caps_string = spec.caps_string;
      if (!add_field_to_list(&list, field_with_caps, buf, buffer_name)) {
        gst_buffer_unref(buf);
        gst_buffer_unref(sample_buf);
        if (err)
          *err = "Sample meta field insert failed";
        return {};
      }
      gst_buffer_unref(buf);
    }
    gst_structure_set_value(s, "fields", &list);
    g_value_unset(&list);
  }

  GstSample* sample = gst_sample_new(sample_buf, sample_caps, nullptr, nullptr);
  gst_buffer_unref(sample_buf);
  if (sample_caps) {
    gst_caps_unref(sample_caps);
  }
  if (!sample) {
    if (err)
      *err = "Sample wrap failed";
    return {};
  }
  auto holder = std::shared_ptr<void>(
      gst_sample_ref(sample), [](void* p) { gst_sample_unref(static_cast<GstSample*>(p)); });
  gst_sample_unref(sample);
  return holder;
}

std::shared_ptr<void> tensor_to_gst_envelope_holder(const Tensor& tensor, std::string* err) {
  Sample sample = sample_from_tensors(TensorList{tensor});
  sample.owned = true;
  sample.payload_type =
      tensor.semantic.image.has_value() ? PayloadType::Image : PayloadType::Tensor;
  sample.media_type =
      tensor.semantic.image.has_value() ? "video/x-raw" : "application/vnd.simaai.tensor";
  sample.payload_tag =
      tensor.semantic.image.has_value() ? Sample::image_format_string(tensor.semantic.image->format)
      : tensor.semantic.byte_stream.has_value() ? format_tag_to_string(FormatTag::ByteStream)
                                                : std::string();
  sample.format = sample.payload_tag;
  sample.segment_name = tensor_runtime_segment_name(tensor);
  sample.stream_label = tensor.route.name;
  sample.logical_output_index = tensor.route.logical_index;
  sample.output_index = tensor.route.logical_index;
  sample.memory_index = tensor_runtime_memory_index(tensor);
  sample.route_slot = tensor.route.route_slot;
  return sample_to_gst_envelope_holder(sample, err);
}

std::shared_ptr<void> tensor_list_to_gst_envelope_holder(const TensorList& tensors,
                                                         const Sample& envelope_meta,
                                                         std::string* err, bool allow_zero_copy) {
  if (tensors.empty()) {
    if (err) {
      *err = "TensorList envelope payload is empty";
    }
    return {};
  }
  Sample sample = envelope_meta;
  sample.kind = SampleKind::TensorSet;
  sample.owned = envelope_meta.owned;
  sample.tensor.reset();
  sample.fields.clear();
  sample.tensors = tensors;
  if (sample.media_type.empty()) {
    const bool all_image_tensors =
        std::all_of(tensors.begin(), tensors.end(),
                    [](const Tensor& tensor) { return tensor.semantic.image.has_value(); });
    sample.payload_type = all_image_tensors ? PayloadType::Image : PayloadType::Tensor;
    sample.media_type = all_image_tensors ? "video/x-raw" : "application/vnd.simaai.tensor";
  } else if (sample.payload_type == PayloadType::Auto) {
    sample.payload_type = payload_type_from_media_type(sample.media_type);
  }
  if (sample.segment_name.empty()) {
    sample.segment_name = tensor_runtime_segment_name(tensors.front());
  }
  if (sample.stream_label.empty() && !tensors.front().route.name.empty()) {
    sample.stream_label = tensors.front().route.name;
  }
  return make_sample_holder_from_bundle(sample, err, allow_zero_copy);
}

std::shared_ptr<void> sample_to_gst_envelope_holder(const Sample& sample, std::string* err,
                                                    bool allow_zero_copy) {
  const Sample canonical = canonicalize_tensor_transport_sample(sample);
  if (sample_has_tensor_list(canonical)) {
    return tensor_list_to_gst_envelope_holder(canonical.tensors, canonical, err, allow_zero_copy);
  }
  if (canonical.kind == SampleKind::Bundle) {
    return make_sample_holder_from_bundle(canonical, err, allow_zero_copy);
  }
  if (err) {
    *err = "Sample tensor envelope conversion requires TensorSet/Bundle";
  }
  return {};
}

} // namespace simaai::neat::pipeline_internal

namespace simaai::neat {

namespace {

Tensor tensor_with_route_meta(const Sample& sample) {
  const Tensor& base = *sample.tensor;
  Tensor out = base;
  if (out.route.logical_index < 0) {
    if (sample.logical_output_index >= 0) {
      out.route.logical_index = sample.logical_output_index;
    } else if (sample.output_index >= 0) {
      out.route.logical_index = sample.output_index;
    }
  }
  if (out.route.memory_index < 0 && sample.memory_index >= 0) {
    out.route.memory_index = sample.memory_index;
  }
  if (out.route.segment_name.empty() && !sample.segment_name.empty()) {
    out.route.segment_name = sample.segment_name;
  }
  return out;
}

void collect_tensors_from_sample_recursive_public(const Sample& sample, TensorList& out) {
  if (sample_has_tensor_list(sample)) {
    out.insert(out.end(), sample.tensors.begin(), sample.tensors.end());
    return;
  }
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    out.emplace_back(tensor_with_route_meta(sample));
    return;
  }
  if (sample.kind != SampleKind::Bundle) {
    return;
  }
  for (const auto& field : sample.fields) {
    collect_tensors_from_sample_recursive_public(field, out);
  }
}

Sample tensor_sample_from_tensor(const Tensor& tensor, std::size_t index) {
  Sample field;
  field.kind = SampleKind::TensorSet;
  field.owned = true;
  field.tensors = TensorList{tensor};
  field.payload_type = tensor.semantic.image.has_value() ? PayloadType::Image : PayloadType::Tensor;
  field.media_type =
      tensor.semantic.image.has_value() ? "video/x-raw" : "application/vnd.simaai.tensor";
  if (tensor.semantic.image.has_value()) {
    field.format = Sample::image_format_string(tensor.semantic.image->format);
    field.payload_tag = field.format;
  } else if (tensor.semantic.byte_stream.has_value()) {
    field.format = format_tag_to_string(FormatTag::ByteStream);
    field.payload_tag = field.format;
  }
  field.output_index = tensor.route.logical_index;
  field.logical_output_index = tensor.route.logical_index;
  field.memory_index = pipeline_internal::tensor_runtime_memory_index(tensor);
  field.route_slot = tensor.route.route_slot;
  field.segment_name = pipeline_internal::tensor_runtime_segment_name(tensor);
  field.stream_label = tensor.route.name;
  return field;
}

} // namespace

TensorList& sample_tensor_list(Sample& sample, const char* where) {
  if (sample.kind == SampleKind::TensorSet) {
    if (sample.tensors.empty()) {
      throw std::runtime_error(std::string(where ? where : "sample_tensor_list") +
                               ": TensorSet has no tensors");
    }
    return sample.tensors;
  }
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    sample = pipeline_internal::canonicalize_tensor_transport_sample(sample);
    if (!sample.tensors.empty()) {
      return sample.tensors;
    }
  }
  throw std::runtime_error(std::string(where ? where : "sample_tensor_list") +
                           ": sample does not contain tensor list payload");
}

const TensorList& sample_tensor_list(const Sample& sample, const char* where) {
  return sample_tensor_list(const_cast<Sample&>(sample), where);
}

Tensor& require_single_tensor(Sample& sample, const char* where) {
  TensorList& tensors = sample_tensor_list(sample, where);
  if (tensors.size() != 1U) {
    throw std::runtime_error(std::string(where ? where : "require_single_tensor") +
                             ": expected exactly one tensor, got " +
                             std::to_string(tensors.size()));
  }
  return tensors.front();
}

const Tensor& require_single_tensor(const Sample& sample, const char* where) {
  const TensorList& tensors = sample_tensor_list(sample, where);
  if (tensors.size() != 1U) {
    throw std::runtime_error(std::string(where ? where : "require_single_tensor") +
                             ": expected exactly one tensor, got " +
                             std::to_string(tensors.size()));
  }
  return tensors.front();
}

Sample pipeline_internal::canonicalize_tensor_transport_sample(const Sample& sample) {
  if (sample.kind == SampleKind::Bundle) {
    Sample out = sample;
    out.fields.clear();
    out.fields.reserve(sample.fields.size());
    for (const auto& field : sample.fields) {
      out.fields.push_back(canonicalize_tensor_transport_sample(field));
    }
    return out;
  }

  if (sample.kind != SampleKind::Tensor || !sample.tensor.has_value()) {
    return sample;
  }

  Tensor tensor = tensor_with_route_meta(sample);
  Sample out = sample;
  out.kind = SampleKind::TensorSet;
  out.tensor.reset();
  out.fields.clear();
  out.tensors = TensorList{tensor};
  if (out.media_type.empty()) {
    out.payload_type = tensor.semantic.image.has_value() ? PayloadType::Image : PayloadType::Tensor;
    out.media_type =
        tensor.semantic.image.has_value() ? "video/x-raw" : "application/vnd.simaai.tensor";
  } else if (out.payload_type == PayloadType::Auto) {
    out.payload_type = payload_type_from_media_type(out.media_type);
  }
  if (out.payload_tag.empty() && tensor.semantic.image.has_value()) {
    out.payload_tag = Sample::image_format_string(tensor.semantic.image->format);
  }
  if (out.format.empty()) {
    out.format = out.payload_tag;
  }
  if (out.segment_name.empty()) {
    out.segment_name = tensor.route.segment_name;
  }
  if (out.stream_label.empty()) {
    out.stream_label = !sample.port_name.empty() ? sample.port_name : tensor.route.name;
  }
  if (out.logical_output_index < 0) {
    out.logical_output_index = tensor.route.logical_index;
  }
  if (out.output_index < 0) {
    out.output_index = tensor.route.logical_index;
  }
  if (out.memory_index < 0) {
    out.memory_index = tensor.route.memory_index;
  }
  if (out.route_slot < 0) {
    out.route_slot = tensor.route.route_slot;
  }
  return out;
}

TensorList tensors_from_sample(const Sample& sample, bool require_nonempty) {
  TensorList out;
  collect_tensors_from_sample_recursive_public(sample, out);
  if (require_nonempty && out.empty()) {
    throw std::runtime_error("tensors_from_sample: sample contains no tensor outputs");
  }
  return out;
}

Sample sample_from_tensors(const TensorList& tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("sample_from_tensors: empty tensor list");
  }
  Sample out;
  out.kind = SampleKind::TensorSet;
  out.owned = true;
  out.tensors = tensors;
  // Stamp positional identity for multi-tensor sets so the SIMA_TENSOR_SET_META
  // descriptor (built downstream by tensor_buffer_descriptor_from_tensors) gives
  // each region a distinct logical slot. Each binding shares the same
  // physical input slot (the packed parent buffer), with memory_index the
  // disambiguator across regions — same convention the multi-IO renderer
  // emits (physical_index=0 / memory_index per tensor). Sharing
  // physical_index lets `packed_tensor_set_parent_segment_name` route the
  // bundle into the existing packed-buffer materialization path so the
  // dispatcher consumes one carrier with byte-offset disambiguation.
  // Only stamps when the caller hasn't already assigned an explicit identity.
  if (out.tensors.size() > 1U) {
    bool any_route_unstamped = false;
    for (const auto& t : out.tensors) {
      if (t.route.memory_index < 0 && t.route.physical_index < 0) {
        any_route_unstamped = true;
        break;
      }
    }
    for (std::size_t i = 0; i < out.tensors.size(); ++i) {
      Tensor& t = out.tensors[i];
      if (t.route.memory_index < 0 && t.route.physical_index < 0) {
        t.route.memory_index = static_cast<int>(i);
        t.route.physical_index = 0;
      }
      if (t.route.logical_index < 0) {
        t.route.logical_index = static_cast<int>(i);
      }
    }
    // Synthesize a parent segment name when the user did not supply one and
    // we just stamped positional identity. The packed-buffer path requires
    // a non-empty segment_name (tensor-level OR bundle-level) to recognize
    // the shared parent. Empty name disables the packed path.
    if (any_route_unstamped && out.segment_name.empty()) {
      bool tensor_segment_name_present = false;
      for (const auto& t : out.tensors) {
        if (!t.route.segment_name.empty()) {
          tensor_segment_name_present = true;
          break;
        }
      }
      if (!tensor_segment_name_present) {
        out.segment_name = "bundled_input_tensor";
      }
    }
  }
  const bool all_image_tensors =
      std::all_of(out.tensors.begin(), out.tensors.end(),
                  [](const Tensor& tensor) { return tensor.semantic.image.has_value(); });
  out.payload_type = all_image_tensors ? PayloadType::Image : PayloadType::Tensor;
  out.media_type = all_image_tensors ? "video/x-raw" : "application/vnd.simaai.tensor";
  if (out.tensors.size() == 1U && out.tensors.front().semantic.byte_stream.has_value()) {
    out.format = format_tag_to_string(FormatTag::ByteStream);
    out.payload_tag = out.format;
  }
  return out;
}

Sample pipeline_internal::collapse_single_tensor_sample(Sample sample) {
  if (sample.kind != SampleKind::TensorSet || sample.tensors.size() != 1U ||
      !sample.fields.empty()) {
    return sample;
  }

  Tensor tensor = std::move(sample.tensors.front());
  sample.tensors.clear();
  sample.kind = SampleKind::Tensor;
  sample.tensor = std::move(tensor);

  if (sample.media_type.empty()) {
    sample.payload_type =
        sample.tensor->semantic.image.has_value() ? PayloadType::Image : PayloadType::Tensor;
    sample.media_type =
        sample.tensor->semantic.image.has_value() ? "video/x-raw" : "application/vnd.simaai.tensor";
  } else if (sample.payload_type == PayloadType::Auto) {
    sample.payload_type = payload_type_from_media_type(sample.media_type);
  }
  if (sample.payload_tag.empty() && sample.tensor->semantic.image.has_value()) {
    sample.payload_tag = Sample::image_format_string(sample.tensor->semantic.image->format);
  } else if (sample.payload_tag.empty() && sample.tensor->semantic.byte_stream.has_value()) {
    sample.payload_tag = format_tag_to_string(FormatTag::ByteStream);
  }
  if (sample.format.empty()) {
    sample.format = sample.payload_tag;
  }
  if (sample.segment_name.empty()) {
    sample.segment_name = sample.tensor->route.segment_name;
  }
  if (sample.stream_label.empty()) {
    sample.stream_label = sample.tensor->route.name;
  }
  if (sample.output_index < 0) {
    sample.output_index = sample.tensor->route.logical_index;
  }
  if (sample.logical_output_index < 0) {
    sample.logical_output_index = sample.tensor->route.logical_index;
  }
  if (sample.memory_index < 0) {
    sample.memory_index = sample.tensor->route.memory_index;
  }
  if (sample.route_slot < 0) {
    sample.route_slot = sample.tensor->route.route_slot;
  }
  return sample;
}

Sample pipeline_internal::sample_from_tensors_for_input(const TensorList& tensors,
                                                        const InputOptions& opt) {
  Sample out = sample_from_tensors(tensors);
  if (out.media_type == "application/vnd.simaai.tensor" &&
      lower_copy(resolve_input_media_type(opt)) == "application/vnd.simaai.tensor") {
    const std::string format = normalize_caps_format_for_media(out.media_type, opt.format.str());
    if (!format.empty()) {
      out.format = format;
    }
  }
  return out;
}

} // namespace simaai::neat

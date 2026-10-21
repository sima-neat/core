#include "pipeline/internal/StageInputCacheIdentity.h"

#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/InputStreamUtil.h"

#include <gst/gst.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace simaai::neat::pipeline_internal {
namespace {

using Json = nlohmann::ordered_json;

[[noreturn]] void invalid(const std::string& message) {
  throw std::runtime_error("StageRun cache input metadata: " + message);
}

// Pointer equality is used only during description. First-seen ordinals (never addresses)
// preserve sharing topology across fresh, otherwise equivalent allocations.
struct Ordinals {
  std::unordered_map<const void*, std::size_t> values;
  Json get(const void* pointer) {
    if (!pointer)
      return nullptr;
    const auto entry = values.emplace(pointer, values.size()).first;
    return entry->second;
  }
};

// Aliasing shared_ptr holders may point at different views of one allocation. Ownership,
// not the alias pointer, identifies that sharing. Weak keys are transient and retain no loan.
struct OwnerOrdinals {
  std::map<std::weak_ptr<void>, std::size_t, std::owner_less<std::weak_ptr<void>>> values;
  Json get(const std::shared_ptr<void>& owner) {
    if (owner.use_count() == 0)
      return nullptr;
    const auto entry = values.emplace(std::weak_ptr<void>(owner), values.size()).first;
    return entry->second;
  }
};

Json device_json(const Device& device) {
  return {{"type", static_cast<int>(device.type)}, {"id", device.id}};
}

void validate_shape_strides(const std::vector<std::int64_t>& shape,
                            const std::vector<std::int64_t>& strides) {
  if (shape.empty())
    invalid("tensor has no shape authority");
  for (const auto dim : shape)
    if (dim < 0)
      invalid("tensor shape contains a negative dimension");
  if (!strides.empty() && strides.size() != shape.size())
    invalid("tensor stride rank does not match shape rank");
  for (const auto stride : strides)
    if (stride < 0)
      invalid("tensor has a negative byte stride");
}

void validate_scales(const std::vector<double>& scales) {
  for (const auto scale : scales)
    if (!std::isfinite(scale) || scale <= 0.0)
      invalid("quantization scale must be finite and positive");
}

Json descriptor_json(const TensorBufferTensorDescriptor& descriptor) {
  validate_shape_strides(descriptor.shape, descriptor.stride_bytes);
  if (descriptor.byte_offset < 0 || descriptor.size_bytes == 0)
    invalid("tensorbuffer descriptor has an invalid offset or span");
  Json quant = nullptr;
  if (descriptor.quant) {
    validate_scales(descriptor.quant->scales);
    quant = {{"granularity", descriptor.quant->granularity},
             {"axis", descriptor.quant->axis},
             {"scales", descriptor.quant->scales},
             {"zero_points", descriptor.quant->zero_points}};
  }
  return {{"logical_index", descriptor.logical_index},
          {"physical_index", descriptor.physical_index},
          {"backend_output_index", descriptor.backend_output_index},
          {"route_slot", descriptor.route_slot},
          {"memory_index", descriptor.memory_index},
          {"logical_name", descriptor.logical_name},
          {"backend_name", descriptor.backend_name},
          {"segment_name", descriptor.segment_name},
          {"byte_offset", descriptor.byte_offset},
          {"size_bytes", descriptor.size_bytes},
          {"dtype", descriptor.dtype},
          {"layout", descriptor.layout},
          {"shape", descriptor.shape},
          {"stride_bytes", descriptor.stride_bytes},
          {"quant", std::move(quant)}};
}

Json preprocess_json(const PreprocessRuntimeMeta& meta, StageInputCacheGeometryPolicy policy) {
  Json rois = Json::array();
  for (const auto& roi : meta.rois)
    rois.push_back({{"batch_index", roi.batch_index},
                    {"x", roi.x},
                    {"y", roi.y},
                    {"width", roi.width},
                    {"height", roi.height}});
  Json roi_affines = Json::array();
  for (const auto& affine : meta.roi_affines)
    roi_affines.push_back({affine.m00, affine.m01, affine.m02, affine.m10, affine.m11, affine.m12});
  Json out = {{"resized_width", meta.resized_width},
              {"resized_height", meta.resized_height},
              {"scaled_width", meta.scaled_width},
              {"scaled_height", meta.scaled_height},
              {"pad_left", meta.pad_left},
              {"pad_right", meta.pad_right},
              {"pad_top", meta.pad_top},
              {"pad_bottom", meta.pad_bottom},
              {"resize_mode", meta.resize_mode},
              {"color_in", meta.color_in},
              {"color_out", meta.color_out},
              {"axis_perm", meta.axis_perm},
              {"normalize", meta.normalize},
              {"quantize", meta.quantize},
              {"tessellate", meta.tessellate},
              {"affine",
               {meta.affine_m00, meta.affine_m01, meta.affine_m02, meta.affine_m10, meta.affine_m11,
                meta.affine_m12}},
              {"affine_scale_x", meta.affine_scale_x},
              {"affine_scale_y", meta.affine_scale_y},
              {"affine_offset_x", meta.affine_offset_x},
              {"affine_offset_y", meta.affine_offset_y},
              {"roi_list_enabled", meta.roi_list_enabled},
              {"rois", std::move(rois)},
              {"roi_input_batch_size", meta.roi_input_batch_size},
              {"roi_source_width", meta.roi_source_width},
              {"roi_source_height", meta.roi_source_height},
              {"roi_source_stride_bytes", meta.roi_source_stride_bytes},
              {"roi_pad_value", meta.roi_pad_value},
              {"roi_capacity", meta.roi_capacity},
              {"roi_valid_count", meta.roi_valid_count},
              {"roi_input_count", meta.roi_input_count},
              {"roi_dropped_invalid", meta.roi_dropped_invalid},
              {"roi_dropped_overflow", meta.roi_dropped_overflow},
              {"roi_affines", std::move(roi_affines)}};
  if (policy == StageInputCacheGeometryPolicy::Exact) {
    out["original_width"] = meta.original_width;
    out["original_height"] = meta.original_height;
  }
  return out;
}

Json semantic_json(const Semantic& semantic, StageInputCacheGeometryPolicy policy) {
  Json out = Json::object();
  if (semantic.image)
    out["image"] = {{"format", static_cast<int>(semantic.image->format)},
                    {"color_space", semantic.image->color_space}};
  if (semantic.audio)
    out["audio"] = {{"sample_rate", semantic.audio->sample_rate},
                    {"channels", semantic.audio->channels},
                    {"interleaved", semantic.audio->interleaved}};
  if (semantic.tokens)
    out["tokens"] = semantic.tokens->vocab_size;
  if (semantic.text)
    out["text"] = semantic.text->encoding;
  if (semantic.byte_stream)
    out["byte_stream"] = {{"format", static_cast<int>(semantic.byte_stream->format)},
                          {"description", semantic.byte_stream->description}};
  if (semantic.tess)
    out["tess"] = {{"slice_shape", semantic.tess->slice_shape}, {"format", semantic.tess->format}};
  if (semantic.encoded)
    out["encoded"] = static_cast<int>(semantic.encoded->codec);
  if (semantic.quant) {
    const auto& quant = *semantic.quant;
    validate_scales({quant.scale});
    validate_scales(std::vector<double>(quant.scales.begin(), quant.scales.end()));
    out["quant"] = {{"scale", quant.scale},
                    {"zero_point", quant.zero_point},
                    {"axis", quant.axis},
                    {"scales", quant.scales},
                    {"zero_points", quant.zero_points}};
  }
  if (semantic.detection)
    out["detection"] = semantic.detection->format;
  if (semantic.feature)
    out["feature"] = semantic.feature->format;
  if (semantic.preprocess && policy != StageInputCacheGeometryPolicy::FramePassthrough)
    out["preprocess"] = preprocess_json(*semantic.preprocess, policy);
  // Only original image size has a known dynamic BoxDecode resolver. In particular resize
  // mode and geometry used to derive model dimensions remain checked preparation facts.
  return out;
}

struct Description {
  StageInputCacheGeometryPolicy geometry_policy = StageInputCacheGeometryPolicy::Exact;
  Ordinals storages;
  OwnerOrdinals holders;
  Ordinals data;
  Ordinals buffers;
  Ordinals memories;
  Ordinals parents;
  // Memoization is local to this description. Actual buffer/caps facts, not
  // incidental GstSample wrappers, select one validated value-only table entry.
  std::unordered_map<const GstCaps*, std::string> caps_texts;
  std::unordered_map<const GstBuffer*, std::unordered_map<std::string, std::size_t>> envelopes;
  Json envelope_values = Json::array();
  std::string reason;

  bool ineligible(const std::string& message) {
    reason = message;
    return false;
  }

  bool gst_envelope(const TensorBuffer& storage, Json* out) {
    auto* sample = static_cast<GstSample*>(storage.holder.get());
    if (!sample || !GST_IS_SAMPLE(sample))
      invalid("GstSample storage has no valid sample holder");
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer || gst_buffer_n_memory(buffer) == 0)
      invalid("GstSample storage has no visible memory");
    const GstCaps* caps = gst_sample_get_caps(sample);
    const auto [caps_entry, new_caps] = caps_texts.try_emplace(caps);
    if (new_caps) {
      gchar* text = caps ? gst_caps_to_string(caps) : nullptr;
      caps_entry->second = text ? text : "";
      g_free(text);
    }
    const auto& caps_text = caps_entry->second;
    if (const auto existing = envelopes.find(buffer); existing != envelopes.end()) {
      if (const auto entry = existing->second.find(caps_text); entry != existing->second.end()) {
        *out = entry->second;
        return true;
      }
    }
    TensorBufferView view;
    std::string error;
    if (!tensor_buffer_descriptor_from_sample(sample, &view, &error)) {
      if (gst_buffer_get_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME))
        invalid("invalid tensorbuffer descriptor: " + error);
      return ineligible("GstSample has no inspectable tensorbuffer descriptor: " + error);
    }
    if (view.tensors.empty())
      invalid("tensorbuffer descriptor has no tensors");

    // Core's descriptor projection intentionally contains logical views only. Reuse also
    // needs the actual physical carrier realization, exposed by Runtime without mapping.
    std::size_t binding_count = 0;
    if (!simaai::gst::tensor_buffer_physical_binding_count(buffer, &binding_count, &error))
      return ineligible("GstSample has no inspectable physical binding authority: " + error);
    Json bindings = Json::array();
    for (std::size_t i = 0; i < binding_count; ++i) {
      simaai::gst::TensorBufferPhysicalBinding binding;
      if (!simaai::gst::tensor_buffer_physical_binding_at(buffer, i, &binding, &error))
        invalid("invalid tensorbuffer physical binding: " + error);
      if (binding.memory_index < 0 ||
          static_cast<guint>(binding.memory_index) >= gst_buffer_n_memory(buffer))
        invalid("tensorbuffer physical binding references an invalid memory slot");
      const auto capacity = gst_memory_get_sizes(
          gst_buffer_peek_memory(buffer, static_cast<guint>(binding.memory_index)), nullptr,
          nullptr);
      if (binding.byte_offset > capacity || binding.size_bytes > capacity - binding.byte_offset)
        invalid("tensorbuffer physical binding extends beyond visible carrier capacity");
      bindings.push_back({{"memory_index", binding.memory_index},
                          {"byte_offset", binding.byte_offset},
                          {"size_bytes", binding.size_bytes}});
    }
    Json memory_json = Json::array();
    for (guint i = 0; i < gst_buffer_n_memory(buffer); ++i) {
      GstMemory* memory = gst_buffer_peek_memory(buffer, i);
      if (!memory)
        invalid("GstBuffer memory slot is absent");
      gsize offset = 0;
      gsize maxsize = 0;
      const gsize visible = gst_memory_get_sizes(memory, &offset, &maxsize);
      if (visible == 0)
        invalid("GstBuffer memory slot has zero visible capacity");
      const GstMemory* parent = memory;
      while (parent->parent)
        parent = parent->parent;
      memory_json.push_back(
          {{"sharing", memories.get(memory)},
           {"parent_sharing", parents.get(parent)},
           {"visible_bytes", visible},
           {"offset", offset},
           {"maxsize", maxsize},
           {"alignment", memory->align},
           {"flags", GST_MEMORY_FLAGS(memory)},
           {"allocator_type",
            memory->allocator && memory->allocator->mem_type ? memory->allocator->mem_type : ""}});
    }
    Json tensors = Json::array();
    for (const auto& descriptor : view.tensors) {
      if (descriptor.memory_index < 0 ||
          static_cast<guint>(descriptor.memory_index) >= gst_buffer_n_memory(buffer))
        invalid("tensorbuffer descriptor references an invalid memory slot");
      const auto capacity = gst_memory_get_sizes(
          gst_buffer_peek_memory(buffer, static_cast<guint>(descriptor.memory_index)), nullptr,
          nullptr);
      if (descriptor.byte_offset < 0 ||
          static_cast<std::uint64_t>(descriptor.byte_offset) > capacity ||
          descriptor.size_bytes > capacity - static_cast<std::uint64_t>(descriptor.byte_offset))
        invalid("tensorbuffer descriptor extends beyond visible carrier capacity");
      tensors.push_back(descriptor_json(descriptor));
    }
    // Keep holder-side facts as well as semantic facts: ingress materialization preserves
    // existing producer metadata, and the frame resolver reads compact authority first.
    // This read does not materialize legacy metadata or touch tensor payloads.
    const auto preprocess = geometry_policy == StageInputCacheGeometryPolicy::FramePassthrough
                                ? std::optional<PreprocessRuntimeMeta>{}
                                : read_simaai_preprocess_meta(buffer);
    Json facts = {
        {"buffer_sharing", buffers.get(buffer)},
        {"preprocess", preprocess ? preprocess_json(*preprocess, geometry_policy) : Json(nullptr)},
        {"caps", caps_text},
        {"stage_key", view.stage_key},
        {"physical_bindings", std::move(bindings)},
        {"memories", std::move(memory_json)},
        {"tensors", std::move(tensors)}};
    const auto index = envelope_values.size();
    envelope_values.push_back(std::move(facts));
    envelopes[buffer].emplace(caps_text, index);
    *out = index;
    return true;
  }

  bool tensor(const Tensor& tensor, Json* out) {
    if (!tensor.storage)
      invalid("tensor has no storage");
    const auto& storage = *tensor.storage;
    if (storage.kind != StorageKind::CpuOwned && storage.kind != StorageKind::CpuExternal &&
        storage.kind != StorageKind::GstSample)
      return ineligible("unsupported tensor storage kind");
    validate_shape_strides(tensor.shape, tensor.strides_bytes);
    if (!tensor.axis_semantics_match_shape())
      invalid("tensor axis semantic rank does not match shape rank");
    if (tensor.byte_offset < 0 || tensor.route.physical_byte_offset < 0)
      invalid("tensor has a negative source offset");
    if (dtype_bytes(tensor.dtype) == 0)
      invalid("tensor dtype has no byte-width authority");
    if (storage.size_bytes == 0)
      return ineligible("storage has no declared capacity authority");
    if (storage.kind != StorageKind::GstSample && !storage.data)
      return ineligible("CPU storage has no direct metadata-describable payload");

    Json segments = Json::array();
    for (const auto& segment : storage.sima_segments)
      segments.push_back({{"name", segment.name}, {"size_bytes", segment.size_bytes}});
    Json planes = Json::array();
    for (const auto& plane : tensor.planes) {
      validate_shape_strides(plane.shape, plane.strides_bytes);
      if (plane.byte_offset < 0)
        invalid("tensor plane has a negative source offset");
      planes.push_back({{"role", static_cast<int>(plane.role)},
                        {"shape", plane.shape},
                        {"strides_bytes", plane.strides_bytes},
                        {"byte_offset", plane.byte_offset}});
    }
    Json axes = Json::array();
    for (const auto axis : tensor.axis_semantics)
      axes.push_back(static_cast<int>(axis));
    Json envelope = nullptr;
    Json storage_sharing;
    Json holder_sharing;
    if (storage.kind == StorageKind::GstSample) {
      if (!gst_envelope(storage, &envelope))
        return false;
      // Cold descriptor projection can create one TensorBuffer/loan holder per head,
      // while warm projection reuses one wrapper for a packed carrier. Neither C++
      // wrapper partition describes the backing allocation. Use the inspected carrier
      // partition instead; its exact GstMemory/parent partitions and extents remain in
      // the envelope. CPU owners and direct-data sharing retain their original identity.
      storage_sharing = envelope_values.at(envelope.get<std::size_t>()).at("buffer_sharing");
      holder_sharing = storage_sharing;
    } else {
      storage_sharing = storages.get(tensor.storage.get());
      holder_sharing = holders.get(storage.holder);
    }
    *out = {{"dtype", static_cast<int>(tensor.dtype)},
            {"layout", static_cast<int>(tensor.layout)},
            {"shape", tensor.shape},
            {"strides_bytes", tensor.strides_bytes},
            {"byte_offset", tensor.byte_offset},
            {"axis_semantics", std::move(axes)},
            {"device", device_json(tensor.device)},
            {"read_only", tensor.read_only},
            {"semantic", semantic_json(tensor.semantic, geometry_policy)},
            {"planes", std::move(planes)},
            {"route",
             {{"stage_key", tensor.route.stage_key},
              {"logical_index", tensor.route.logical_index},
              {"backend_output_index", tensor.route.backend_output_index},
              {"route_slot", tensor.route.route_slot},
              {"physical_index", tensor.route.physical_index},
              {"memory_index", tensor.route.memory_index},
              {"physical_byte_offset", tensor.route.physical_byte_offset},
              {"name", tensor.route.name},
              {"backend_name", tensor.route.backend_name},
              {"segment_name", tensor.route.segment_name}}},
            {"storage",
             {{"kind", static_cast<int>(storage.kind)},
              {"device", device_json(storage.device)},
              {"size_bytes", storage.size_bytes},
              {"storage_sharing", std::move(storage_sharing)},
              {"holder_sharing", std::move(holder_sharing)},
              {"data_sharing", data.get(storage.data)},
              {"has_map_callback", static_cast<bool>(storage.map_fn)},
              {"sima_mem_target_flags", storage.sima_mem_target_flags},
              {"sima_mem_flags", storage.sima_mem_flags},
              {"segments", std::move(segments)},
              {"envelope", std::move(envelope)}}}};
    return true;
  }

  bool sample(const Sample& sample, Json* out) {
    Json payload = Json::array();
    switch (sample.kind) {
    case SampleKind::Tensor:
      if (!sample.tensor)
        invalid("Tensor Sample has no tensor");
      payload.emplace_back();
      if (!tensor(*sample.tensor, &payload.back()))
        return false;
      break;
    case SampleKind::TensorSet:
      if (sample.tensors.empty())
        invalid("TensorSet Sample has no tensors");
      for (const auto& input : sample.tensors) {
        payload.emplace_back();
        if (!tensor(input, &payload.back()))
          return false;
      }
      break;
    case SampleKind::Bundle:
      if (sample.fields.empty())
        invalid("Bundle Sample has no fields");
      for (const auto& field : sample.fields) {
        payload.emplace_back();
        if (!this->sample(field, &payload.back()))
          return false;
      }
      break;
    default:
      return ineligible("unsupported Sample representation");
    }
    *out = {{"kind", static_cast<int>(sample.kind)},
            {"owned", sample.owned},
            {"caps", sample.caps_string},
            {"payload_type", static_cast<int>(sample.payload_type)},
            {"media_type", sample.media_type},
            {"payload_tag", sample.payload_tag},
            {"format", sample.format},
            {"port_name", sample.port_name},
            // Bundle ingress can use this label as its field/segment routing fallback.
            {"stream_label", sample.stream_label},
            {"output_index", sample.output_index},
            {"logical_output_index", sample.logical_output_index},
            {"memory_index", sample.memory_index},
            {"route_slot", sample.route_slot},
            {"segment_name", sample.segment_name},
            {"payload", std::move(payload)}};
    return true;
  }
};

} // namespace

std::optional<StageInputCacheIdentity>
stage_input_cache_identity(const Sample& sample, std::string* ineligible_reason,
                           StageInputCacheGeometryPolicy geometry_policy) {
  if (ineligible_reason)
    ineligible_reason->clear();
  Description description;
  description.geometry_policy = geometry_policy;
  Json values;
  if (!description.sample(sample, &values)) {
    if (ineligible_reason)
      *ineligible_reason = description.reason;
    return std::nullopt;
  }
  values["envelopes"] = std::move(description.envelope_values);
  return StageInputCacheIdentity{values.dump()};
}

} // namespace simaai::neat::pipeline_internal

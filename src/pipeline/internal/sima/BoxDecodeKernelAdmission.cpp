#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/sima/stagesemantics/SsdDecodeContract.h"

#include "pipeline/internal/TensorBufferEnvelope.h"

#include <gst/gst.h>

#include <algorithm>
#include <stdexcept>

namespace simaai::neat::pipeline_internal::sima {
CompiledKernelContract compile_boxdecode_kernel(const CompiledBoxDecodeContract& compiled,
                                                const ContractCompileInput& input,
                                                BoxDecodeGeometryOptions geometry,
                                                std::optional<ResizeMode> resize_mode) {
  namespace si = ::simaai::neat::pipeline_internal::sima;
  // The old plugin resolved source carriers dynamically from metadata.
  // Runner requires their allocation extents once, before admitting work.
  // Import actual producer/storage contracts; tensor shapes cannot establish
  // the capacity of a shared carrier or identify its GstMemory slot.
  auto admitted = compiled;
  auto& runtime = admitted.runtime_contract;
  const auto* upstream =
      compiled_runtime_contract_from_stage(last_effective_child_stage(input.immediate_upstream));
  TensorBufferView ingress;
  std::shared_ptr<void> ingress_holder;
  if (!upstream && input.ingress.ingress_sample) {
    // Use the same envelope as graph ingress. Owned outputs may retain producer
    // route offsets after their payloads have been copied into independent
    // storage. Those historical offsets are not offsets in the new carrier.
    // This is cold admission only; no new per-frame conversion is introduced.
    std::string error;
    ingress_holder = sample_to_gst_envelope_holder(*input.ingress.ingress_sample, &error);
    if (!ingress_holder || !tensor_buffer_descriptor_from_sample(
                               static_cast<GstSample*>(ingress_holder.get()), &ingress, &error))
      throw std::invalid_argument("SimaBoxDecode: cannot resolve ingress carriers: " + error);
  }
  runtime.physical_inputs.clear();
  for (auto& logical : runtime.logical_inputs) {
    auto binding = std::find_if(runtime.input_bindings.begin(), runtime.input_bindings.end(),
                                [&](const auto& value) {
                                  return value.local_logical_input_index == logical.logical_index;
                                });
    if (binding == runtime.input_bindings.end())
      throw std::invalid_argument("SimaBoxDecode: missing authored head input binding");
    si::PhysicalBufferStaticSpec physical;
    if (upstream) {
      const auto source =
          std::find_if(upstream->physical_outputs.begin(), upstream->physical_outputs.end(),
                       [&](const auto& value) {
                         return value.physical_index == binding->src_physical_output_index;
                       });
      if (source == upstream->physical_outputs.end())
        throw std::invalid_argument(
            "SimaBoxDecode: head has no authoritative upstream physical carrier");
      physical = si::specbuilders::build_physical_input_from_output(*source, *upstream);
      physical.physical_index = logical.physical_index;
      binding->src_physical_size_bytes = source->size_bytes;
      // Bind the complete carrier; the native port applies the logical
      // tensor offset exactly once, including packed multi-head carriers.
      binding->src_physical_byte_offset = 0;
    } else if (ingress_holder) {
      const auto source =
          std::find_if(ingress.tensors.begin(), ingress.tensors.end(), [&](const auto& tensor) {
            return tensor.logical_index == binding->src_logical_output_index;
          });
      if (source == ingress.tensors.end() || source->memory_index < 0 ||
          static_cast<guint>(source->memory_index) >= gst_buffer_n_memory(ingress.buffer))
        throw std::invalid_argument("SimaBoxDecode: ingress head has no physical carrier");
      const int memory = source->memory_index;
      logical.physical_index = memory;
      logical.byte_offset = source->byte_offset;
      physical = si::specbuilders::build_physical_buffer_static_spec(
          memory, memory,
          gst_memory_get_sizes(gst_buffer_peek_memory(ingress.buffer, memory), nullptr, nullptr),
          si::DeviceKind::Cpu, logical.segment_name);
      binding->src_stage_index = -1;
      binding->src_stage_id.clear();
      binding->src_logical_output_index = -1;
      binding->src_output_slot = -1;
      binding->src_physical_output_index = -1;
      binding->src_physical_size_bytes = 0;
      binding->src_physical_byte_offset = 0;
    } else {
      const auto source =
          std::find_if(compiled.runtime_contract.physical_inputs.begin(),
                       compiled.runtime_contract.physical_inputs.end(), [&](const auto& value) {
                         return value.physical_index == logical.physical_index;
                       });
      if (source == compiled.runtime_contract.physical_inputs.end())
        throw std::invalid_argument("SimaBoxDecode: typed head contract requires authoritative "
                                    "physical input carriers");
      physical = *source;
    }
    if (!physical.size_bytes || logical.byte_offset < 0 ||
        static_cast<std::uint64_t>(logical.byte_offset) >= physical.size_bytes)
      throw std::invalid_argument("SimaBoxDecode: head offset exceeds its physical carrier");
    logical.size_bytes = physical.size_bytes - static_cast<std::uint64_t>(logical.byte_offset);
    const auto existing = std::find_if(
        runtime.physical_inputs.begin(), runtime.physical_inputs.end(),
        [&](const auto& value) { return value.physical_index == physical.physical_index; });
    if (existing == runtime.physical_inputs.end())
      runtime.physical_inputs.push_back(std::move(physical));
    else if (existing->size_bytes != physical.size_bytes ||
             existing->source_byte_offset != physical.source_byte_offset ||
             existing->address_source != physical.address_source)
      throw std::invalid_argument("SimaBoxDecode: shared head carrier contracts disagree");
  }
  std::optional<PreprocessRuntimeMeta> observed;
  if (input.ingress.ingress_sample) {
    for (const auto& tensor : tensors_from_sample(*input.ingress.ingress_sample, true)) {
      if (!tensor.semantic.preprocess)
        continue;
      const auto& facts = *tensor.semantic.preprocess;
      if (observed && (observed->original_width != facts.original_width ||
                       observed->original_height != facts.original_height))
        throw std::invalid_argument("SimaBoxDecode: ingress tensors disagree on image geometry");
      observed = facts;
    }
  }
  if (observed && geometry.model_width <= 0 && geometry.model_height <= 0) {
    const auto width =
        observed->scaled_width > 0 ? observed->scaled_width : observed->resized_width;
    const auto height =
        observed->scaled_height > 0 ? observed->scaled_height : observed->resized_height;
    const std::int64_t padded_width =
        std::int64_t(width) + observed->pad_left + observed->pad_right;
    const std::int64_t padded_height =
        std::int64_t(height) + observed->pad_top + observed->pad_bottom;
    if (padded_width < 1 || padded_width > 1048576 || padded_height < 1 || padded_height > 1048576)
      throw std::invalid_argument("SimaBoxDecode: ingress model geometry is invalid");
    geometry.model_width = static_cast<int>(padded_width);
    geometry.model_height = static_cast<int>(padded_height);
  }
  if ((geometry.model_width <= 0 || geometry.model_height <= 0) &&
      compiled.payload.decode_type == BoxDecodeType::Ssd) {
    if (const auto* recipe =
            si::stagesemantics::find_ssd_recipe_descriptor(compiled.payload.ssd_recipe_id)) {
      geometry.model_width = recipe->model_width;
      geometry.model_height = recipe->model_height;
    }
  }
  if (geometry.model_width <= 0 || geometry.model_height <= 0)
    throw std::invalid_argument("SimaBoxDecode: model dimensions require explicit options, "
                                "model contract or ingress preprocess metadata");
  const bool dynamic_image = geometry.original_width <= 0 || geometry.original_height <= 0;
  geometry.original_width = dynamic_image
                                ? (observed ? observed->original_width : geometry.model_width)
                                : geometry.original_width;
  geometry.original_height = dynamic_image
                                 ? (observed ? observed->original_height : geometry.model_height)
                                 : geometry.original_height;
  geometry.resize = (compiled.payload.decode_type == BoxDecodeType::Ssd ||
                     compiled.payload.decode_type == BoxDecodeType::Detr)
                        ? ResizeMode::Stretch
                        : ResizeMode::Letterbox;
  if (resize_mode)
    geometry.resize = *resize_mode;
  else if (observed && !observed->resize_mode.empty()) {
    if (observed->resize_mode == "stretch")
      geometry.resize = ResizeMode::Stretch;
    else if (observed->resize_mode == "letterbox")
      geometry.resize = ResizeMode::Letterbox;
    else if (compiled.payload.decode_type == BoxDecodeType::SuperPoint &&
             observed->resize_mode == "crop")
      geometry.resize = ResizeMode::Crop;
    else
      throw std::invalid_argument("SimaBoxDecode: unsupported preprocess resize geometry");
  }
  auto native = si::prepare_boxdecode_kernel(admitted, geometry);
  if (compiled.payload.decode_type != BoxDecodeType::SuperPoint) {
    auto recipe = std::make_shared<si::KernelStageRecipe>(*native.compiled.recipe);
    const bool require_mode =
        std::find(compiled.runtime_contract.required_preprocess_meta_fields.begin(),
                  compiled.runtime_contract.required_preprocess_meta_fields.end(),
                  "preproc_resize_mode") !=
        compiled.runtime_contract.required_preprocess_meta_fields.end();
    si::attach_boxdecode_geometry_resolver(*recipe, native.decode_member, geometry, dynamic_image,
                                           require_mode);
    native.compiled.recipe = std::move(recipe);
  }
  return std::move(native.compiled);
}
} // namespace simaai::neat::pipeline_internal::sima

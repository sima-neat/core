#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string>

namespace simaai::neat {
namespace {

using pipeline_internal::sima::LogicalInputStaticSpec;
using pipeline_internal::sima::LogicalTensorStaticSpec;
using pipeline_internal::sima::ManifestBuildDiagnostics;
using pipeline_internal::sima::ProcessCvuGraphFamily;
using pipeline_internal::sima::ProcessCvuOutputSemanticKind;
using pipeline_internal::sima::ProcessCvuOutputTransportKind;
using pipeline_internal::sima::ProcessCvuStagePayload;
using pipeline_internal::sima::SimaPluginStaticManifest;
using pipeline_internal::sima::StageOutputRoute;
using pipeline_internal::sima::StagePayloadKind;
using pipeline_internal::sima::StageStaticSpec;
using pipeline_internal::sima::TensorStaticSpec;

std::string upper_copy_local(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

void set_tensor_wh_from_shape(const std::vector<std::int64_t>& shape,
                              const std::string& layout,
                              int* out_w,
                              int* out_h) {
  if (out_w) {
    *out_w = 0;
  }
  if (out_h) {
    *out_h = 0;
  }
  if (shape.empty()) {
    return;
  }

  const std::string up_layout = pipeline_internal::sima::tensorsemantics::normalize_layout_token(layout);
  auto positive_or_zero = [](std::int64_t value) {
    return value > 0 ? static_cast<int>(value) : 0;
  };
  if (up_layout == "CHW") {
    if (shape.size() >= 3U) {
      if (out_h) {
        *out_h = positive_or_zero(shape[shape.size() - 2U]);
      }
      if (out_w) {
        *out_w = positive_or_zero(shape.back());
      }
    }
    return;
  }
  if (up_layout == "HW") {
    if (shape.size() >= 2U) {
      if (out_h) {
        *out_h = positive_or_zero(shape[shape.size() - 2U]);
      }
      if (out_w) {
        *out_w = positive_or_zero(shape.back());
      }
    }
    return;
  }
  if (up_layout == "HWC" && shape.size() >= 3U) {
    if (out_h) {
      *out_h = positive_or_zero(shape[shape.size() - 3U]);
    }
    if (out_w) {
      *out_w = positive_or_zero(shape[shape.size() - 2U]);
    }
    return;
  }
  if (shape.size() >= 2U) {
    if (out_h) {
      *out_h = positive_or_zero(shape[shape.size() - 2U]);
    }
    if (out_w) {
      *out_w = positive_or_zero(shape.back());
    }
  }
}

TensorStaticSpec tensor_from_logical_input(const LogicalInputStaticSpec& logical) {
  TensorStaticSpec tensor;
  tensor.tensor_index = logical.backend_input_index;
  tensor.shape = logical.shape;
  tensor.dtype = logical.dtype;
  tensor.layout = logical.layout;
  tensor.semantic_tag = !logical.logical_name.empty() ? logical.logical_name : logical.backend_name;
  set_tensor_wh_from_shape(tensor.shape, tensor.layout, &tensor.max_w, &tensor.max_h);
  return tensor;
}

TensorStaticSpec tensor_from_logical_output(const LogicalTensorStaticSpec& logical) {
  TensorStaticSpec tensor;
  tensor.tensor_index = logical.tensor_index;
  tensor.shape = logical.shape;
  tensor.dtype = logical.dtype;
  tensor.layout = logical.layout;
  tensor.semantic_tag = !logical.logical_name.empty()
                            ? logical.logical_name
                            : (!logical.backend_name.empty() ? logical.backend_name
                                                             : logical.segment_name);
  set_tensor_wh_from_shape(tensor.shape, tensor.layout, &tensor.max_w, &tensor.max_h);
  return tensor;
}

std::vector<TensorStaticSpec> tensors_from_logical_inputs(
    const std::vector<LogicalInputStaticSpec>& logical_inputs) {
  std::vector<TensorStaticSpec> tensors;
  tensors.reserve(logical_inputs.size());
  for (const auto& logical : logical_inputs) {
    tensors.push_back(tensor_from_logical_input(logical));
  }
  return tensors;
}

std::vector<TensorStaticSpec> tensors_from_logical_outputs(
    const std::vector<LogicalTensorStaticSpec>& logical_outputs) {
  std::vector<TensorStaticSpec> tensors;
  tensors.reserve(logical_outputs.size());
  for (const auto& logical : logical_outputs) {
    tensors.push_back(tensor_from_logical_output(logical));
  }
  return tensors;
}

std::vector<StageOutputRoute> require_exposed_output_routes(const CompiledExposedView& exposed_view) {
  if (exposed_view.exposed_output_order.empty()) {
    throw std::invalid_argument(
        "processcvu exposed view missing explicit output_order for render");
  }
  for (const auto& route : exposed_view.exposed_output_order) {
    if (route.cm_output_name.empty() && route.segment_name.empty()) {
      throw std::invalid_argument(
          "processcvu exposed view output_order route missing output name");
    }
  }
  return exposed_view.exposed_output_order;
}

bool stage_is_cast_transport(const StageStaticSpec& stage) {
  return upper_copy_local(stage.plugin_kind) == "CAST" ||
         upper_copy_local(stage.kernel_kind) == "CAST";
}

bool stage_is_distinct_mla_consumer(const StageStaticSpec& stage) {
  return stage.consumer_keeps_distinct_physical_inputs &&
         upper_copy_local(stage.kernel_kind) == "MLA" &&
         stage.physical_inputs.size() > 1U;
}

std::string physical_segment_name(const pipeline_internal::sima::PhysicalBufferStaticSpec& physical) {
  return physical.segment_name;
}

bool logical_matches_segment_name(const LogicalTensorStaticSpec& logical,
                                  const std::string& segment_name) {
  return !segment_name.empty() &&
         (logical.segment_name == segment_name || logical.backend_name == segment_name ||
          logical.logical_name == segment_name);
}

void normalize_cast_outputs_for_distinct_mla_boundary(
    StageStaticSpec* producer,
    const StageStaticSpec& consumer) {
  if (!producer || !stage_is_cast_transport(*producer) ||
      !stage_is_distinct_mla_consumer(consumer) ||
      producer->logical_outputs.size() != consumer.physical_inputs.size()) {
    return;
  }

  std::vector<std::size_t> logical_by_physical(consumer.physical_inputs.size(),
                                               producer->logical_outputs.size());
  std::vector<bool> used(producer->logical_outputs.size(), false);
  for (std::size_t physical_idx = 0; physical_idx < consumer.physical_inputs.size();
       ++physical_idx) {
    const auto& consumer_physical = consumer.physical_inputs[physical_idx];
    const std::string segment_name = physical_segment_name(consumer_physical);
    if (segment_name.empty() || consumer_physical.size_bytes == 0U) {
      return;
    }
    for (std::size_t logical_idx = 0; logical_idx < producer->logical_outputs.size();
         ++logical_idx) {
      if (used[logical_idx] ||
          !logical_matches_segment_name(producer->logical_outputs[logical_idx], segment_name)) {
        continue;
      }
      const auto& logical = producer->logical_outputs[logical_idx];
      if (logical.size_bytes != 0U && logical.size_bytes != consumer_physical.size_bytes) {
        return;
      }
      logical_by_physical[physical_idx] = logical_idx;
      used[logical_idx] = true;
      break;
    }
    if (logical_by_physical[physical_idx] == producer->logical_outputs.size()) {
      return;
    }
  }

  const auto old_physical_outputs = producer->physical_outputs;
  const auto fallback_device =
      old_physical_outputs.empty() ? pipeline_internal::sima::DeviceKind::Cpu
                                   : old_physical_outputs.front().device_kind;
  producer->physical_outputs.clear();
  producer->physical_outputs.reserve(consumer.physical_inputs.size());
  for (std::size_t physical_idx = 0; physical_idx < consumer.physical_inputs.size();
       ++physical_idx) {
    const auto& consumer_physical = consumer.physical_inputs[physical_idx];
    producer->physical_outputs.push_back(
        pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
            static_cast<int>(physical_idx),
            static_cast<int>(physical_idx),
            consumer_physical.size_bytes,
            fallback_device,
            physical_segment_name(consumer_physical)));
    auto& logical = producer->logical_outputs[logical_by_physical[physical_idx]];
    logical.physical_index = static_cast<int>(physical_idx);
    logical.byte_offset = 0;
    logical.segment_name = physical_segment_name(consumer_physical);
  }

  for (auto& route : producer->output_order) {
    if (route.logical_output_index < 0) {
      continue;
    }
    const auto it = std::find_if(
        producer->logical_outputs.begin(), producer->logical_outputs.end(),
        [&](const LogicalTensorStaticSpec& logical) {
          return logical.logical_index == route.logical_output_index;
        });
    if (it != producer->logical_outputs.end()) {
      route.segment_name = it->segment_name;
    }
  }
  producer->consumer_keeps_distinct_physical_inputs = true;
}

void normalize_manifest_boundaries(SimaPluginStaticManifest* manifest) {
  if (!manifest || manifest->stages.size() < 2U) {
    return;
  }
  for (std::size_t i = 0; i + 1U < manifest->stages.size(); ++i) {
    normalize_cast_outputs_for_distinct_mla_boundary(&manifest->stages[i],
                                                     manifest->stages[i + 1U]);
  }
}

void populate_processcvu_runtime_output_contract(ProcessCvuStagePayload* payload,
                                                 const CompiledRuntimeContract& runtime) {
  if (!payload) {
    return;
  }
  const bool preproc_graph = upper_copy_local(payload->graph_family) == "PREPROC";
  const bool preproc_single_output =
      preproc_graph && payload->preproc_single_output_handoff;
  // Payload runtime outputs describe the CM/runtime boundary, not the exposed
  // logical publication fanout. For routes like detessdequant, many logical
  // published outputs can alias one runtime output / physical buffer.
  const std::size_t expected_runtime_output_count =
      !payload->default_output_names.empty()
          ? payload->default_output_names.size()
          : (!runtime.physical_outputs.empty() ? runtime.physical_outputs.size()
                                               : runtime.logical_outputs.size());
  const auto existing_default_output_names = payload->default_output_names;
  const auto existing_output_shapes = payload->output_shapes;
  const auto existing_runtime_output_logical_index_list =
      payload->runtime_output_logical_index_list;
  const auto existing_runtime_output_output_slot_list =
      payload->runtime_output_output_slot_list;
  const auto existing_runtime_output_physical_index_list =
      payload->runtime_output_physical_index_list;
  const auto existing_runtime_output_dtype_list = payload->runtime_output_dtype_list;
  const auto existing_runtime_output_transport_kind_list =
      payload->runtime_output_transport_kind_list;
  const auto existing_runtime_output_semantic_kind_list =
      payload->runtime_output_semantic_kind_list;
  const auto existing_runtime_output_logical_shapes =
      payload->runtime_output_logical_shapes;
  const auto existing_runtime_output_logical_layout_list =
      payload->runtime_output_logical_layout_list;
  const bool already_complete =
      !payload->output_shapes.empty() &&
      payload->output_shapes.size() == expected_runtime_output_count &&
      payload->runtime_output_transport_kind_list.size() == payload->output_shapes.size() &&
      payload->runtime_output_semantic_kind_list.size() == payload->output_shapes.size() &&
      payload->runtime_output_logical_shapes.size() == payload->output_shapes.size() &&
      payload->runtime_output_logical_layout_list.size() == payload->output_shapes.size();
  if (already_complete) {
    return;
  }
  payload->output_shapes.clear();
  payload->runtime_output_logical_index_list.clear();
  payload->runtime_output_output_slot_list.clear();
  payload->runtime_output_physical_index_list.clear();
  payload->runtime_output_dtype_list.clear();
  payload->runtime_output_transport_kind_list.clear();
  payload->runtime_output_semantic_kind_list.clear();
  payload->runtime_output_logical_shapes.clear();
  payload->runtime_output_logical_layout_list.clear();

  auto shape_i64_to_int = [](const std::vector<std::int64_t>& shape_i64) {
    std::vector<int> shape;
    shape.reserve(shape_i64.size());
    for (const auto dim : shape_i64) {
      shape.push_back(static_cast<int>(dim));
    }
    return shape;
  };

  auto append_runtime_output = [&](std::vector<int> output_shape,
                                   const LogicalTensorStaticSpec& logical,
                                   int fallback_output_slot,
                                   const std::string& dtype,
                                   ProcessCvuOutputTransportKind transport_kind,
                                   ProcessCvuOutputSemanticKind semantic_kind,
                                   std::vector<int> logical_shape,
                                   const std::string& logical_layout) {
    payload->output_shapes.push_back(std::move(output_shape));
    payload->runtime_output_logical_index_list.push_back(logical.logical_index);
    payload->runtime_output_output_slot_list.push_back(
        logical.output_slot >= 0 ? logical.output_slot : fallback_output_slot);
    payload->runtime_output_physical_index_list.push_back(logical.physical_index);
    payload->runtime_output_dtype_list.push_back(dtype);
    payload->runtime_output_transport_kind_list.push_back(transport_kind);
    payload->runtime_output_semantic_kind_list.push_back(semantic_kind);
    payload->runtime_output_logical_shapes.push_back(std::move(logical_shape));
    payload->runtime_output_logical_layout_list.push_back(logical_layout);
  };

  auto transport_kind_from_logical = [](const LogicalTensorStaticSpec& logical) {
    if (logical.shape.size() == 1U && logical.size_bytes > 0U &&
        upper_copy_local(logical.layout) == "HW") {
      return ProcessCvuOutputTransportKind::Packed;
    }
    return ProcessCvuOutputTransportKind::Dense;
  };

  auto semantic_kind_from_output_name = [](const std::string& output_name,
                                           ProcessCvuOutputTransportKind transport_kind) {
    (void)transport_kind;
    if (output_name == "output_rgb_image") {
      return ProcessCvuOutputSemanticKind::Image;
    }
    if (output_name == "output_tessellated_image") {
      return ProcessCvuOutputSemanticKind::TessellatedImage;
    }
    return ProcessCvuOutputSemanticKind::Tensor;
  };

  auto payload_transport_kind_at = [&](std::size_t index,
                                       ProcessCvuOutputTransportKind fallback) {
    return index < existing_runtime_output_transport_kind_list.size()
               ? existing_runtime_output_transport_kind_list[index]
               : fallback;
  };

  auto payload_semantic_kind_at = [&](std::size_t index,
                                      ProcessCvuOutputSemanticKind fallback) {
    return index < existing_runtime_output_semantic_kind_list.size()
               ? existing_runtime_output_semantic_kind_list[index]
               : fallback;
  };

  auto payload_output_shape_at = [&](std::size_t index,
                                     std::vector<int> fallback) -> std::vector<int> {
    if (index < existing_output_shapes.size() && !existing_output_shapes[index].empty()) {
      return existing_output_shapes[index];
    }
    return fallback;
  };

  auto payload_logical_shape_at = [&](std::size_t index,
                                      std::vector<int> fallback) -> std::vector<int> {
    if (index < existing_runtime_output_logical_shapes.size() &&
        !existing_runtime_output_logical_shapes[index].empty()) {
      return existing_runtime_output_logical_shapes[index];
    }
    return fallback;
  };

  auto payload_logical_layout_at = [&](std::size_t index, const std::string& fallback) {
    return index < existing_runtime_output_logical_layout_list.size() &&
                   !existing_runtime_output_logical_layout_list[index].empty()
               ? existing_runtime_output_logical_layout_list[index]
               : fallback;
  };

  const std::size_t payload_output_npos = static_cast<std::size_t>(-1);
  auto payload_output_index_for = [&](const LogicalTensorStaticSpec& logical,
                                      const StageOutputRoute* route,
                                      const std::string& preferred_name) {
    auto find_name = [&](const std::string& candidate) -> std::size_t {
      if (candidate.empty()) {
        return payload_output_npos;
      }
      for (std::size_t index = 0; index < existing_default_output_names.size(); ++index) {
        if (existing_default_output_names[index] == candidate) {
          return index;
        }
      }
      return payload_output_npos;
    };

    for (const auto& candidate : {preferred_name,
                                  route ? route->cm_output_name : std::string(),
                                  route ? route->segment_name : std::string(),
                                  logical.logical_name,
                                  logical.backend_name,
                                  logical.segment_name}) {
      const std::size_t index = find_name(candidate);
      if (index != payload_output_npos) {
        return index;
      }
    }

    if (logical.physical_index >= 0) {
      for (std::size_t index = 0; index < existing_runtime_output_physical_index_list.size();
           ++index) {
        if (existing_runtime_output_physical_index_list[index] == logical.physical_index) {
          return index;
        }
      }
    }

    if (logical.logical_index >= 0) {
      for (std::size_t index = 0; index < existing_runtime_output_logical_index_list.size();
           ++index) {
        if (existing_runtime_output_logical_index_list[index] == logical.logical_index) {
          return index;
        }
      }
    }

    return payload_output_npos;
  };

  auto find_runtime_logical_by_name = [&](const std::string& output_name)
      -> const LogicalTensorStaticSpec* {
    for (const auto& logical : runtime.logical_outputs) {
      if (logical.logical_name == output_name || logical.backend_name == output_name ||
          logical.segment_name == output_name) {
        return &logical;
      }
    }
    return nullptr;
  };

  if (preproc_graph && !preproc_single_output) {
    const auto payload_outputs = !payload->default_output_names.empty()
                                     ? payload->default_output_names
                                     : std::vector<std::string>{"output_rgb_image",
                                                                "output_tessellated_image"};
    payload->output_shapes.reserve(payload_outputs.size());
    payload->runtime_output_logical_index_list.reserve(payload_outputs.size());
    payload->runtime_output_output_slot_list.reserve(payload_outputs.size());
    payload->runtime_output_physical_index_list.reserve(payload_outputs.size());
    payload->runtime_output_dtype_list.reserve(payload_outputs.size());
    payload->runtime_output_transport_kind_list.reserve(payload_outputs.size());
    payload->runtime_output_semantic_kind_list.reserve(payload_outputs.size());
    payload->runtime_output_logical_shapes.reserve(payload_outputs.size());
    payload->runtime_output_logical_layout_list.reserve(payload_outputs.size());

    for (std::size_t i = 0; i < payload_outputs.size(); ++i) {
      const std::string& output_name = payload_outputs[i];
      const auto* logical = find_runtime_logical_by_name(output_name);
      if (!logical) {
        throw std::invalid_argument(
            "processcvu preproc runtime output could not resolve a logical output");
      }
      const auto logical_shape = shape_i64_to_int(logical->shape);
      const std::string dtype = !payload->output_dtype.empty() ? payload->output_dtype
                                : (!payload->out_dtype.empty() ? payload->out_dtype
                                                               : logical->dtype);
      const std::string layout = payload->logical_output_layout_token(i);
      const auto transport_kind =
          payload_transport_kind_at(i, transport_kind_from_logical(*logical));
      const auto semantic_kind =
          payload_semantic_kind_at(i, semantic_kind_from_output_name(output_name, transport_kind));
      const auto output_shape = payload_output_shape_at(i, logical_shape);
      const auto runtime_logical_shape = payload_logical_shape_at(i, logical_shape);
      const std::string logical_layout = payload_logical_layout_at(i, layout);
      append_runtime_output(output_shape, *logical,
                            logical->output_slot >= 0 ? logical->output_slot
                                                       : static_cast<int>(i),
                            dtype, transport_kind, semantic_kind, runtime_logical_shape,
                            logical_layout);
    }
    return;
  }

  payload->output_shapes.reserve(runtime.logical_outputs.size());
  payload->runtime_output_logical_index_list.reserve(runtime.logical_outputs.size());
  payload->runtime_output_output_slot_list.reserve(runtime.logical_outputs.size());
  payload->runtime_output_physical_index_list.reserve(runtime.logical_outputs.size());
  payload->runtime_output_dtype_list.reserve(runtime.logical_outputs.size());
  payload->runtime_output_transport_kind_list.reserve(runtime.logical_outputs.size());
  payload->runtime_output_semantic_kind_list.reserve(runtime.logical_outputs.size());
  payload->runtime_output_logical_shapes.reserve(runtime.logical_outputs.size());
  payload->runtime_output_logical_layout_list.reserve(runtime.logical_outputs.size());

  auto find_runtime_logical =
      [&](const StageOutputRoute& route) -> const LogicalTensorStaticSpec* {
    for (const auto& logical : runtime.logical_outputs) {
      if (route.logical_output_index >= 0 && logical.logical_index == route.logical_output_index) {
        return &logical;
      }
      if (route.output_slot >= 0 && logical.output_slot == route.output_slot) {
        return &logical;
      }
      if (route.tensor_index >= 0 && logical.tensor_index == route.tensor_index) {
        return &logical;
      }
      if (!route.cm_output_name.empty() && logical.backend_name == route.cm_output_name) {
        return &logical;
      }
      if (!route.segment_name.empty() && logical.segment_name == route.segment_name) {
        return &logical;
      }
    }
    return nullptr;
  };

  if (runtime.output_order.empty()) {
    throw std::invalid_argument(
        "processcvu runtime contract missing explicit output_order for payload render");
  }

  for (std::size_t i = 0; i < runtime.output_order.size(); ++i) {
    const auto& route = runtime.output_order[i];
    const auto* logical = find_runtime_logical(route);
    if (!logical) {
      throw std::invalid_argument(
          "processcvu runtime contract output_order could not resolve a logical output");
    }
    const auto logical_shape = shape_i64_to_int(logical->shape);
    const auto transport_kind =
        payload_transport_kind_at(payload_output_index_for(
                                      *logical,
                                      &route,
                                      !route.cm_output_name.empty() ? route.cm_output_name
                                                                    : route.segment_name),
                                  transport_kind_from_logical(*logical));
    const auto semantic_kind =
        payload_semantic_kind_at(payload_output_index_for(
                                     *logical,
                                     &route,
                                     !route.cm_output_name.empty() ? route.cm_output_name
                                                                   : route.segment_name),
                                 semantic_kind_from_output_name(
                                     !route.cm_output_name.empty() ? route.cm_output_name
                                                                   : route.segment_name,
                                     transport_kind));
    const std::size_t payload_index = payload_output_index_for(
        *logical, &route, !route.cm_output_name.empty() ? route.cm_output_name : route.segment_name);
    const auto output_shape = payload_output_shape_at(payload_index, logical_shape);
    const auto runtime_logical_shape = payload_logical_shape_at(payload_index, logical_shape);
    const std::string logical_layout = payload_logical_layout_at(payload_index, logical->layout);
    append_runtime_output(output_shape, *logical,
                          route.output_slot >= 0 ? route.output_slot : static_cast<int>(i),
                          logical->dtype, transport_kind, semantic_kind, runtime_logical_shape,
                          logical_layout);
  }
  if (payload->output_shapes.size() != runtime.output_order.size()) {
    throw std::invalid_argument(
        "processcvu runtime contract output_order render count mismatch");
  }
}

std::string stage_match_key(const StageStaticSpec& stage) {
  if (!stage.logical_stage_id.empty()) {
    return "id:" + stage.logical_stage_id;
  }
  if (!stage.element_name.empty()) {
    return "element:" + stage.element_name;
  }
  return {};
}

StageStaticSpec render_processcvu_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.processcvu;
  if (pipeline_internal::env_bool("SIMA_RENDER_STAGE_DEBUG", false) &&
      compiled_stage.logical_stage_id.find("post_dequant") != std::string::npos) {
    std::fprintf(stderr,
                 "[render-stage-debug] stage=%s id=%s runtime.logical_inputs=%zu "
                 "runtime.logical_outputs=%zu runtime.output_order=%zu "
                 "exposed.logical_outputs=%zu exposed.output_order=%zu primary=%s\n",
                 compiled_stage.element_name.c_str(), compiled_stage.logical_stage_id.c_str(),
                 compiled.runtime_contract.logical_inputs.size(),
                 compiled.runtime_contract.logical_outputs.size(),
                 compiled.runtime_contract.output_order.size(),
                 compiled.exposed_view.exposed_logical_outputs.size(),
                 compiled.exposed_view.exposed_output_order.size(),
                 compiled.exposed_view.primary_output_name.c_str());
  }
  StageStaticSpec stage;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind.empty() ? "processcvu"
                                                                    : compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = !compiled.payload.graph_family.empty() ? compiled.payload.graph_family
                                                             : compiled.payload.graph_name;
  stage.payload_kind = StagePayloadKind::ProcessCvu;
  stage.processcvu = compiled.payload;
  stage.processcvu.preproc_single_output_handoff = compiled.preproc_single_output_handoff;
  populate_processcvu_runtime_output_contract(&stage.processcvu, compiled.runtime_contract);
  if (!compiled.exposed_view.primary_output_name.empty()) {
    stage.processcvu.primary_output_name = compiled.exposed_view.primary_output_name;
  }

  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  // Render only the exposed outputs into the manifest stage. Runtime outputs
  // still live in processcvu payload/default_output_names + physical_outputs,
  // but downstream tensor publication must honor single-output handoff.
  stage.logical_outputs = compiled.exposed_view.exposed_logical_outputs;
  stage.output_order = compiled.exposed_view.exposed_output_order;
  stage.output_quant = compiled.runtime_contract.output_quant;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.consumer_keeps_distinct_physical_inputs =
      compiled.runtime_contract.consumer_keeps_distinct_physical_inputs;
  return stage;
}

StageStaticSpec render_processmla_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.processmla;
  StageStaticSpec stage;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind.empty() ? "processmla"
                                                                    : compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = "mla";
  stage.payload_kind = StagePayloadKind::ProcessMla;
  stage.processmla = compiled.payload;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.output_quant = compiled.runtime_contract.output_quant;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.consumer_keeps_distinct_physical_inputs =
      compiled.runtime_contract.consumer_keeps_distinct_physical_inputs;
  stage.elf_ifm_symbol_names = compiled.runtime_contract.elf_ifm_symbol_names;
  stage.elf_ofm_symbol_names = compiled.runtime_contract.elf_ofm_symbol_names;
  return stage;
}

StageStaticSpec render_boxdecode_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.boxdecode;
  StageStaticSpec stage;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind.empty() ? "boxdecode"
                                                                    : compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = "boxdecode";
  stage.payload_kind = StagePayloadKind::BoxDecode;
  stage.boxdecode = compiled.payload;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  return stage;
}

StageStaticSpec render_dequant_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.dequant;
  StageStaticSpec stage;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.model_managed_stage = false;
  stage.plugin_kind =
      compiled.runtime_contract.plugin_kind.empty() ? "dequant" : compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = "dequant";
  stage.payload_kind = StagePayloadKind::Dequant;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  return stage;
}

StageStaticSpec render_transport_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.transport;
  StageStaticSpec stage;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.model_managed_stage = compiled.model_managed_stage;
  stage.plugin_kind = compiled.plugin_kind;
  stage.kernel_kind = compiled.kernel_kind;
  stage.payload_kind = compiled.payload_kind;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.consumer_keeps_distinct_physical_inputs =
      compiled.runtime_contract.consumer_keeps_distinct_physical_inputs;
  if (compiled.payload_kind == StagePayloadKind::ProcessCvu && compiled.processcvu_payload.has_value()) {
    stage.processcvu = *compiled.processcvu_payload;
    populate_processcvu_runtime_output_contract(&stage.processcvu, compiled.runtime_contract);
    if (stage.processcvu.primary_output_name.empty() && !compiled.runtime_contract.output_order.empty()) {
      const std::string primary_name =
          output_name_from_route(compiled.runtime_contract.output_order.front());
      if (!primary_name.empty()) {
        stage.processcvu.primary_output_name = primary_name;
      }
    }
  }
  if (pipeline_internal::env_bool("SIMA_TRANSPORT_CONTRACT_DEBUG", false)) {
    std::fprintf(stderr,
                 "[render-transport-debug] stage=%s id=%s plugin=%s kernel=%s phys=%zu logical=%zu routes=%zu first_segment=%s\n",
                 stage.element_name.c_str(), stage.logical_stage_id.c_str(),
                 stage.plugin_kind.c_str(), stage.kernel_kind.c_str(),
                 stage.physical_outputs.size(), stage.logical_outputs.size(),
                 stage.output_order.size(),
                 (!stage.physical_outputs.empty() && !stage.physical_outputs.front().segment_name.empty())
                     ? stage.physical_outputs.front().segment_name.c_str()
                     : "<empty>");
  }
  return stage;
}

std::vector<StageStaticSpec> render_stages_from_compiled_contract(
    const CompiledNodeContract& stage, ManifestBuildDiagnostics* diagnostics) {
  if (!stage.child_stages.empty()) {
    std::vector<StageStaticSpec> rendered;
    for (const auto& child : stage.child_stages) {
      auto child_rendered = render_stages_from_compiled_contract(child, diagnostics);
      rendered.insert(rendered.end(),
                      std::make_move_iterator(child_rendered.begin()),
                      std::make_move_iterator(child_rendered.end()));
    }
    return rendered;
  }
  if (stage.processcvu.has_value()) {
    try {
      return {render_processcvu_stage(stage)};
    } catch (const std::exception& ex) {
      if (diagnostics) {
        diagnostics->errors.push_back("contract render: processcvu stage '" +
                                      stage.node_kind + "' failed: " + ex.what());
      }
      return {};
    }
  }
  if (stage.processmla.has_value()) {
    try {
      return {render_processmla_stage(stage)};
    } catch (const std::exception& ex) {
      if (diagnostics) {
        diagnostics->errors.push_back("contract render: processmla stage '" +
                                      stage.node_kind + "' failed: " + ex.what());
      }
      return {};
    }
  }
  if (stage.boxdecode.has_value()) {
    try {
      return {render_boxdecode_stage(stage)};
    } catch (const std::exception& ex) {
      if (diagnostics) {
        diagnostics->errors.push_back("contract render: boxdecode stage '" +
                                      stage.node_kind + "' failed: " + ex.what());
      }
      return {};
    }
  }
  if (stage.dequant.has_value()) {
    try {
      return {render_dequant_stage(stage)};
    } catch (const std::exception& ex) {
      if (diagnostics) {
        diagnostics->errors.push_back("contract render: dequant stage '" +
                                      stage.node_kind + "' failed: " + ex.what());
      }
      return {};
    }
  }
  if (stage.transport.has_value()) {
    try {
      return {render_transport_stage(stage)};
    } catch (const std::exception& ex) {
      if (diagnostics) {
        diagnostics->errors.push_back("contract render: transport stage '" +
                                      stage.node_kind + "' failed: " + ex.what());
      }
      return {};
    }
  }
  if (diagnostics && stage.node_kind != "Input" && stage.node_kind != "Output") {
    diagnostics->warnings.push_back("contract render: skipping non-renderable node kind '" +
                                    stage.node_kind + "'");
  }
  return {};
}

} // namespace

std::optional<pipeline_internal::sima::SimaPluginStaticManifest>
render_manifest_from_compiled_contracts(const CompiledPipelineContracts& compiled,
                                        const ContractCompileInput& compile_input,
                                        pipeline_internal::sima::ManifestBuildDiagnostics* diagnostics) {
  SimaPluginStaticManifest manifest;

  for (const auto& stage : compiled.stages) {
    for (const auto& rendered : render_stages_from_compiled_contract(stage, diagnostics)) {
      StageStaticSpec resolved = rendered;
      if (resolved.payload_kind == StagePayloadKind::ProcessCvu) {
        const std::string stage_identity =
            !resolved.logical_stage_id.empty() ? resolved.logical_stage_id
                                               : resolved.element_name;
        resolve_processcvu_run_target(&resolved.processcvu, compile_input, stage_identity);
      }
      manifest.stages.push_back(std::move(resolved));
    }
  }

  normalize_manifest_boundaries(&manifest);

  if (diagnostics && !diagnostics->errors.empty()) {
    return std::nullopt;
  }

  if (manifest.stages.empty()) {
    if (diagnostics && !compiled.fully_renderable) {
      diagnostics->warnings.push_back(
          "contract render: compiled contract set is partial and no renderable stages were produced");
    }
    return std::nullopt;
  }
  if (diagnostics && !compiled.fully_renderable) {
    diagnostics->warnings.push_back(
        "contract render: compiled contract set is partial; rendered stages will require overlay");
  }
  return manifest;
}

} // namespace simaai::neat

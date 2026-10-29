#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
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

void set_tensor_wh_from_shape(const std::vector<std::int64_t>& shape, const std::string& layout,
                              int* out_w, int* out_h) {
  if (out_w) {
    *out_w = 0;
  }
  if (out_h) {
    *out_h = 0;
  }
  if (shape.empty()) {
    return;
  }

  const std::string up_layout =
      pipeline_internal::sima::tensorsemantics::normalize_layout_token(layout);
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
  tensor.semantic_tag =
      !logical.logical_name.empty()
          ? logical.logical_name
          : (!logical.backend_name.empty() ? logical.backend_name : logical.segment_name);
  set_tensor_wh_from_shape(tensor.shape, tensor.layout, &tensor.max_w, &tensor.max_h);
  return tensor;
}

std::vector<TensorStaticSpec>
tensors_from_logical_inputs(const std::vector<LogicalInputStaticSpec>& logical_inputs) {
  std::vector<TensorStaticSpec> tensors;
  tensors.reserve(logical_inputs.size());
  for (const auto& logical : logical_inputs) {
    tensors.push_back(tensor_from_logical_input(logical));
  }
  return tensors;
}

std::vector<TensorStaticSpec>
tensors_from_logical_outputs(const std::vector<LogicalTensorStaticSpec>& logical_outputs) {
  std::vector<TensorStaticSpec> tensors;
  tensors.reserve(logical_outputs.size());
  for (const auto& logical : logical_outputs) {
    tensors.push_back(tensor_from_logical_output(logical));
  }
  return tensors;
}

std::vector<StageOutputRoute>
require_exposed_output_routes(const CompiledExposedView& exposed_view) {
  if (exposed_view.exposed_output_order.empty()) {
    throw std::invalid_argument("processcvu exposed view missing explicit output_order for render");
  }
  for (const auto& route : exposed_view.exposed_output_order) {
    if (route.cm_output_name.empty() && route.segment_name.empty()) {
      throw std::invalid_argument("processcvu exposed view output_order route missing output name");
    }
  }
  return exposed_view.exposed_output_order;
}

void populate_processcvu_runtime_output_contract(ProcessCvuStagePayload* payload,
                                                 const CompiledRuntimeContract& runtime) {
  if (!payload) {
    return;
  }
  // Typed image policy owns one image canvas separately from its numerical planes.
  // Do not rewrite that canvas or descriptor selection from publication fanout.
  if (payload->image_transform)
    return;
  const bool preproc_graph = upper_copy_local(payload->graph_family) == "PREPROC";
  const bool preproc_single_output = preproc_graph && payload->preproc_single_output_handoff;
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
  const auto existing_runtime_output_output_slot_list = payload->runtime_output_output_slot_list;
  const auto existing_runtime_output_physical_index_list =
      payload->runtime_output_physical_index_list;
  const auto existing_runtime_output_dtype_list = payload->runtime_output_dtype_list;
  const auto existing_runtime_output_transport_kind_list =
      payload->runtime_output_transport_kind_list;
  const auto existing_runtime_output_semantic_kind_list =
      payload->runtime_output_semantic_kind_list;
  const auto existing_runtime_output_logical_shapes = payload->runtime_output_logical_shapes;
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

  auto append_runtime_output =
      [&](std::vector<int> output_shape, const LogicalTensorStaticSpec& logical,
          int fallback_output_slot, const std::string& dtype,
          ProcessCvuOutputTransportKind transport_kind, ProcessCvuOutputSemanticKind semantic_kind,
          std::vector<int> logical_shape, const std::string& logical_layout) {
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

  auto payload_transport_kind_at = [&](std::size_t index, ProcessCvuOutputTransportKind fallback) {
    return index < existing_runtime_output_transport_kind_list.size()
               ? existing_runtime_output_transport_kind_list[index]
               : fallback;
  };

  auto payload_semantic_kind_at = [&](std::size_t index, ProcessCvuOutputSemanticKind fallback) {
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

    for (const auto& candidate : {preferred_name, route ? route->cm_output_name : std::string(),
                                  route ? route->segment_name : std::string(), logical.logical_name,
                                  logical.backend_name, logical.segment_name}) {
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

  auto find_runtime_logical_by_name =
      [&](const std::string& output_name) -> const LogicalTensorStaticSpec* {
    for (const auto& logical : runtime.logical_outputs) {
      if (logical.logical_name == output_name || logical.backend_name == output_name ||
          logical.segment_name == output_name) {
        return &logical;
      }
    }
    return nullptr;
  };

  if (preproc_graph && !preproc_single_output) {
    const auto payload_outputs =
        !payload->default_output_names.empty()
            ? payload->default_output_names
            : std::vector<std::string>{"output_rgb_image", "output_tessellated_image"};
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
      const std::string dtype =
          !payload->output_dtype.empty()
              ? payload->output_dtype
              : (!payload->out_dtype.empty() ? payload->out_dtype : logical->dtype);
      const std::string layout = payload->logical_output_layout_token(i);
      const auto transport_kind =
          payload_transport_kind_at(i, transport_kind_from_logical(*logical));
      const auto semantic_kind =
          payload_semantic_kind_at(i, semantic_kind_from_output_name(output_name, transport_kind));
      const auto output_shape = payload_output_shape_at(i, logical_shape);
      const auto runtime_logical_shape = payload_logical_shape_at(i, logical_shape);
      const std::string logical_layout = payload_logical_layout_at(i, layout);
      append_runtime_output(output_shape, *logical,
                            logical->output_slot >= 0 ? logical->output_slot : static_cast<int>(i),
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

  auto find_runtime_logical = [&](const StageOutputRoute& route) -> const LogicalTensorStaticSpec* {
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
    const auto transport_kind = payload_transport_kind_at(
        payload_output_index_for(*logical, &route,
                                 !route.cm_output_name.empty() ? route.cm_output_name
                                                               : route.segment_name),
        transport_kind_from_logical(*logical));
    const auto semantic_kind = payload_semantic_kind_at(
        payload_output_index_for(*logical, &route,
                                 !route.cm_output_name.empty() ? route.cm_output_name
                                                               : route.segment_name),
        semantic_kind_from_output_name(!route.cm_output_name.empty() ? route.cm_output_name
                                                                     : route.segment_name,
                                       transport_kind));
    const std::size_t payload_index = payload_output_index_for(
        *logical, &route,
        !route.cm_output_name.empty() ? route.cm_output_name : route.segment_name);
    const auto output_shape = payload_output_shape_at(payload_index, logical_shape);
    const auto runtime_logical_shape = payload_logical_shape_at(payload_index, logical_shape);
    const std::string logical_layout = payload_logical_layout_at(payload_index, logical->layout);
    append_runtime_output(
        output_shape, *logical, route.output_slot >= 0 ? route.output_slot : static_cast<int>(i),
        logical->dtype, transport_kind, semantic_kind, runtime_logical_shape, logical_layout);
  }
  if (payload->output_shapes.size() != runtime.output_order.size()) {
    throw std::invalid_argument("processcvu runtime contract output_order render count mismatch");
  }
}

StageStaticSpec render_processcvu_stage(const CompiledProcessCvuContract& compiled,
                                        const std::string& element_name,
                                        const std::string& logical_stage_id) {
  if (pipeline_internal::env_bool("SIMA_RENDER_STAGE_DEBUG", false) &&
      logical_stage_id.find("post_dequant") != std::string::npos) {
    std::fprintf(stderr,
                 "[render-stage-debug] stage=%s id=%s runtime.logical_inputs=%zu "
                 "runtime.logical_outputs=%zu runtime.output_order=%zu "
                 "exposed.logical_outputs=%zu exposed.output_order=%zu primary=%s\n",
                 element_name.c_str(), logical_stage_id.c_str(),
                 compiled.runtime_contract.logical_inputs.size(),
                 compiled.runtime_contract.logical_outputs.size(),
                 compiled.runtime_contract.output_order.size(),
                 compiled.exposed_view.exposed_logical_outputs.size(),
                 compiled.exposed_view.exposed_output_order.size(),
                 compiled.exposed_view.primary_output_name.c_str());
  }
  StageStaticSpec stage;
  stage.element_name = element_name;
  stage.logical_stage_id = logical_stage_id;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind.empty()
                          ? "processcvu"
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

  stage.image_publication = compiled.runtime_contract.image_publication;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.logical_outputs = compiled.exposed_view.exposed_logical_outputs;
  // Checked Preproc produces the selected carrier; an unselected RGB descriptor
  // is a semantic alternative, not another materialized output.
  if (stage.processcvu.graph_family_enum == ProcessCvuGraphFamily::Preproc)
    std::erase_if(stage.physical_outputs, [&](const auto& physical) {
      return std::none_of(
          stage.logical_outputs.begin(), stage.logical_outputs.end(),
          [&](const auto& logical) { return logical.physical_index == physical.physical_index; });
    });
  stage.output_order = compiled.exposed_view.exposed_output_order;
  stage.output_quant = compiled.runtime_contract.output_quant;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.allocation_policy = compiled.runtime_contract.allocation_policy;
  stage.reuse_input_pad = compiled.runtime_contract.reuse_input_pad;
  stage.frame_layout = compiled.runtime_contract.frame_layout;
  stage.consumer_keeps_distinct_physical_inputs =
      compiled.runtime_contract.consumer_keeps_distinct_physical_inputs;
  return stage;
}

StageStaticSpec render_processmla_stage(const CompiledMlaContract& compiled,
                                        const std::string& element_name,
                                        const std::string& logical_stage_id) {
  StageStaticSpec stage;
  stage.element_name = element_name;
  stage.logical_stage_id = logical_stage_id;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind.empty()
                          ? "processmla"
                          : compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = "mla";
  stage.payload_kind = StagePayloadKind::ProcessMla;
  stage.processmla = compiled.payload;
  stage.image_publication = compiled.runtime_contract.image_publication;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.output_quant = compiled.runtime_contract.output_quant;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.allocation_policy = compiled.runtime_contract.allocation_policy;
  stage.reuse_input_pad = compiled.runtime_contract.reuse_input_pad;
  stage.frame_layout = compiled.runtime_contract.frame_layout;
  stage.consumer_keeps_distinct_physical_inputs =
      compiled.runtime_contract.consumer_keeps_distinct_physical_inputs;
  stage.elf_ifm_symbol_names = compiled.runtime_contract.elf_ifm_symbol_names;
  stage.elf_ofm_symbol_names = compiled.runtime_contract.elf_ofm_symbol_names;
  auto program = std::make_shared<::simaai::scheduler::client::MlaProgram>();
  program->model = {compiled.payload.model_path, compiled.payload.executable_bytes,
                    compiled.payload.executable_sha256};
  program->batch_size = compiled.payload.batch_size;
  program->model_batch_size = compiled.payload.batch_sz_model;
  const auto ports = [](const auto& names, const auto& physical) {
    if (physical.size() != names.size())
      throw std::invalid_argument("MLA port names and physical alignment table differ");
    std::vector<::simaai::scheduler::client::MlaPort> result(names.size());
    for (const auto& port : physical) {
      if (port.physical_index < 0 ||
          static_cast<std::size_t>(port.physical_index) >= names.size() ||
          !port.required_alignment_bytes)
        throw std::invalid_argument("MLA port has no exact ordered alignment contract");
      const auto index = static_cast<std::size_t>(port.physical_index);
      result[index] = {names[index], static_cast<std::size_t>(port.required_alignment_bytes)};
    }
    return result;
  };
  program->inputs = ports(compiled.runtime_contract.elf_ifm_symbol_names, stage.physical_inputs);
  program->outputs = ports(compiled.runtime_contract.elf_ofm_symbol_names, stage.physical_outputs);
  stage.mla_program = std::move(program);

  return stage;
}

StageStaticSpec render_kernel_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.kernel;
  if (!compiled.recipe || compiled.recipe->members.empty())
    throw std::invalid_argument("native kernel stage requires checked executable members");
  const auto& runtime = compiled.runtime_contract;
  StageStaticSpec stage;
  stage.native_kernel = compiled.recipe;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.plugin_kind = "kernel";
  stage.kernel_kind = compiled_stage.node_kind;
  stage.image_publication = runtime.image_publication;
  stage.logical_inputs = runtime.logical_inputs;
  stage.input_bindings = runtime.input_bindings;
  stage.physical_inputs = runtime.physical_inputs;
  stage.physical_outputs = runtime.physical_outputs;
  stage.physical_outputs.insert(stage.physical_outputs.end(),
                                compiled.recipe->private_buffers.begin(),
                                compiled.recipe->private_buffers.end());
  stage.logical_outputs = runtime.logical_outputs;
  stage.output_order = runtime.output_order;
  stage.required_preprocess_meta_fields = runtime.required_preprocess_meta_fields;
  stage.allocation_policy = runtime.allocation_policy;
  stage.reuse_input_pad = runtime.reuse_input_pad;
  stage.frame_layout = runtime.frame_layout;
  return stage;
}

StageStaticSpec render_boxdecode_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.boxdecode;
  StageStaticSpec stage;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind.empty()
                          ? "boxdecode"
                          : compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = "boxdecode";
  stage.payload_kind = StagePayloadKind::BoxDecode;
  stage.boxdecode = compiled.payload;
  stage.image_publication = compiled.runtime_contract.image_publication;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.allocation_policy = compiled.runtime_contract.allocation_policy;
  stage.reuse_input_pad = compiled.runtime_contract.reuse_input_pad;
  stage.frame_layout = compiled.runtime_contract.frame_layout;
  return stage;
}

// Keep the public Dequant contract and its publication/routing identities. Only
// execution is lowered here: each logical tensor becomes one checked kernel
// member, with its original strides and offsets into its physical carrier.
ProcessCvuStagePayload dequant_execution_payload(StageStaticSpec& stage) {
  namespace semantics = pipeline_internal::sima::stagesemantics;
  if (stage.logical_inputs.empty() || stage.logical_inputs.size() != stage.logical_outputs.size() ||
      stage.physical_inputs.empty() || stage.physical_outputs.empty()) {
    throw std::invalid_argument(
        "Dequant requires matching typed input/output tensors and carriers");
  }
  semantics::CompiledProcessCvuRuntimeConfig config;
  config.graph_family = config.graph_name = "dequant";

  config.input_dtype = "INT8";
  config.output_dtype = config.out_dtype = "FP32";
  config.primary_output_transport_kind = ProcessCvuOutputTransportKind::Dense;
  config.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Tensor;

  const auto descriptor = [](const auto& tensor, const auto& iteration_shape) {
    if (tensor.shape.size() != iteration_shape.size() || iteration_shape.empty() ||
        iteration_shape.size() > SIMA_EV_MAX_RANK || tensor.byte_offset < 0 ||
        (!tensor.stride_bytes.empty() && tensor.stride_bytes.size() != tensor.shape.size())) {
      throw std::invalid_argument("Dequant requires a bounded shape and matching strides/offset");
    }
    std::vector<int> shape;
    for (std::size_t axis = 0; axis < iteration_shape.size(); ++axis) {
      if (iteration_shape[axis] <= 0 || iteration_shape[axis] > tensor.shape[axis] ||
          tensor.shape[axis] > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Dequant output must be a positive input-shaped slice");
      }
      shape.push_back(static_cast<int>(tensor.shape[axis]));
    }
    sima_ev_tensor_desc result{};
    std::string error;
    if (!pipeline_internal::sima::tensorsemantics::build_dense_tensor_desc(
            shape, tensor.dtype, tensor.layout, &result, &error, "Dequant: missing descriptor",
            "Dequant: invalid rank", "Dequant: invalid dimension", "Dequant: invalid dtype",
            "Dequant: missing strides")) {
      throw std::invalid_argument(error);
    }
    // A sliced output keeps the input's original row/plane pitches. Shrinking
    // the logical iteration bounds must not recompute those pitches.
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      result.shape.sizes[axis] = iteration_shape[axis];
      if (!tensor.stride_bytes.empty()) {
        if (tensor.stride_bytes[axis] <= 0)
          throw std::invalid_argument("Dequant requires positive byte strides");
        result.layout.strided.strides_bytes[axis] = tensor.stride_bytes[axis];
      }
    }
    result.storage.addr = static_cast<std::uint64_t>(tensor.byte_offset);
    return result;
  };
  for (std::size_t i = 0; i < stage.logical_inputs.size(); ++i) {
    auto& input = stage.logical_inputs[i];
    auto& output = stage.logical_outputs[i];
    if (upper_copy_local(input.dtype) != "INT8" ||
        (upper_copy_local(output.dtype) != "FP32" && upper_copy_local(output.dtype) != "FLOAT32")) {
      throw std::invalid_argument("Dequant requires INT8 input and FP32 output");
    }
    if (!input.quant || input.quant->scales.size() != 1 || input.quant->zero_points.size() != 1 ||
        input.quant->granularity != pipeline_internal::sima::QuantGranularity::PerTensor ||
        !std::isfinite(input.quant->scales.front()) || input.quant->scales.front() <= 0 ||
        input.quant->zero_points.front() < -128 || input.quant->zero_points.front() > 127) {
      throw std::invalid_argument(
          "Dequant requires a finite positive per-tensor scale and INT8 zero point");
    }
    // Backend indices identify executable members, not the upstream's output
    // order. Logical identities and source bindings remain unchanged.
    input.backend_input_index = static_cast<int>(i);
    output.backend_output_index = static_cast<int>(i);
    config.input_tensors.emplace_back(descriptor(input, output.shape), input.dtype);
    config.output_tensors.emplace_back(descriptor(output, output.shape), output.dtype);
    config.q_scale_list.push_back(input.quant->scales.front());
    config.q_zp_list.push_back(input.quant->zero_points.front());
    config.published_output_names.push_back(output.logical_name);
  }
  for (const auto& physical : stage.physical_inputs)
    config.physical_input_names.push_back(physical.segment_name);
  for (const auto& physical : stage.physical_outputs) {
    config.physical_output_names.push_back(physical.segment_name);
    const auto logical = std::find_if(
        stage.logical_outputs.begin(), stage.logical_outputs.end(),
        [&](const auto& value) { return value.physical_index == physical.physical_index; });
    if (logical == stage.logical_outputs.end())
      throw std::invalid_argument("Dequant output carrier has no logical tensor");
    // Runtime-output arrays describe physical carriers, not semantic members.
    // Six logical output views packed into one carrier still have one entry.
    config.runtime_output_logical_shapes.emplace_back(logical->shape.begin(), logical->shape.end());
  }
  config.default_input_name = config.physical_input_names.front();
  config.runtime_input_names = config.physical_input_names;
  config.runtime_output_names = config.physical_output_names;
  config.primary_output_name = config.published_output_names.front();
  auto payload = semantics::build_processcvu_payload_from_runtime_config_internal(config);
  // Standalone Dequant has always been a host transform. Preserve placement;
  // model-authored ProcessCVU Dequant keeps its existing explicit policy.
  payload.requested_run_target = payload.run_target = payload.resolved_exec_backend = "A65";
  return payload;
}

StageStaticSpec render_dequant_stage(const CompiledNodeContract& compiled_stage) {
  const auto& compiled = *compiled_stage.dequant;
  StageStaticSpec stage;
  stage.element_name = compiled_stage.element_name;
  stage.logical_stage_id = compiled_stage.logical_stage_id;
  stage.model_managed_stage = false;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind.empty()
                          ? "dequant"
                          : compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = "dequant";
  stage.payload_kind = StagePayloadKind::ProcessCvu;
  stage.image_publication = compiled.runtime_contract.image_publication;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.allocation_policy = compiled.runtime_contract.allocation_policy;
  stage.reuse_input_pad = compiled.runtime_contract.reuse_input_pad;
  stage.frame_layout = compiled.runtime_contract.frame_layout;
  stage.processcvu = dequant_execution_payload(stage);
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
  stage.image_publication = compiled.runtime_contract.image_publication;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.required_preprocess_meta_fields = compiled.runtime_contract.required_preprocess_meta_fields;
  stage.tvm_program = compiled.tvm_program;
  stage.allocation_policy = compiled.runtime_contract.allocation_policy;
  stage.reuse_input_pad = compiled.runtime_contract.reuse_input_pad;
  stage.frame_layout = compiled.runtime_contract.frame_layout;
  stage.consumer_keeps_distinct_physical_inputs =
      compiled.runtime_contract.consumer_keeps_distinct_physical_inputs;
  if (compiled.payload_kind == StagePayloadKind::ProcessCvu &&
      compiled.processcvu_payload.has_value()) {
    stage.processcvu = *compiled.processcvu_payload;
    populate_processcvu_runtime_output_contract(&stage.processcvu, compiled.runtime_contract);
    if (stage.processcvu.primary_output_name.empty() &&
        !compiled.runtime_contract.output_order.empty()) {
      const std::string primary_name =
          output_name_from_route(compiled.runtime_contract.output_order.front());
      if (!primary_name.empty()) {
        stage.processcvu.primary_output_name = primary_name;
      }
    }
  }
  if (pipeline_internal::env_bool("SIMA_TRANSPORT_CONTRACT_DEBUG", false)) {
    std::fprintf(
        stderr,
        "[render-transport-debug] stage=%s id=%s plugin=%s kernel=%s phys=%zu logical=%zu "
        "routes=%zu first_segment=%s\n",
        stage.element_name.c_str(), stage.logical_stage_id.c_str(), stage.plugin_kind.c_str(),
        stage.kernel_kind.c_str(), stage.physical_outputs.size(), stage.logical_outputs.size(),
        stage.output_order.size(),
        (!stage.physical_outputs.empty() && !stage.physical_outputs.front().segment_name.empty())
            ? stage.physical_outputs.front().segment_name.c_str()
            : "<empty>");
  }
  return stage;
}

std::vector<StageStaticSpec>
render_stages_from_compiled_contract(const CompiledNodeContract& stage,
                                     ManifestBuildDiagnostics* diagnostics) {
  if (!stage.child_stages.empty()) {
    std::vector<StageStaticSpec> rendered;
    for (const auto& child : stage.child_stages) {
      auto stages = render_stages_from_compiled_contract(child, diagnostics);
      rendered.insert(rendered.end(), std::make_move_iterator(stages.begin()),
                      std::make_move_iterator(stages.end()));
    }
    return rendered;
  }
  if (stage.kernel)
    return {render_kernel_stage(stage)};
  if (stage.processcvu) {
    auto rendered =
        render_processcvu_stage(*stage.processcvu, stage.element_name, stage.logical_stage_id);
    rendered.native_execution_capacity = stage.native_execution_capacity;
    return {std::move(rendered)};
  }
  if (stage.processmla)
    return {render_processmla_stage(*stage.processmla, stage.element_name, stage.logical_stage_id)};
  if (stage.boxdecode)
    return {render_boxdecode_stage(stage)};
  if (stage.dequant)
    return {render_dequant_stage(stage)};
  if (stage.transport)
    return {render_transport_stage(stage)};
  if (stage.node_kind != "Input" && stage.node_kind != "Output")
    throw std::invalid_argument("no stage renderer for node kind '" + stage.node_kind + "'");
  return {};
}

} // namespace

pipeline_internal::sima::StageStaticSpec
project_processcvu_stage_contract(const CompiledProcessCvuContract& compiled) {
  return render_processcvu_stage(compiled, {}, {});
}

pipeline_internal::sima::StageStaticSpec
project_mla_stage_contract(const CompiledMlaContract& compiled) {
  return render_processmla_stage(compiled, {}, {});
}

std::optional<pipeline_internal::sima::SimaPluginStaticManifest>
render_manifest_from_compiled_contracts(
    const CompiledPipelineContracts& compiled, const ContractCompileInput&,
    pipeline_internal::sima::ManifestBuildDiagnostics* diagnostics) {
  SimaPluginStaticManifest manifest;
  manifest.model_storage_scopes = compiled.model_storage_scopes;

  try {
    for (std::size_t node_index = 0; node_index < compiled.stages.size(); ++node_index)
      for (auto& rendered :
           render_stages_from_compiled_contract(compiled.stages[node_index], diagnostics)) {
        rendered.compile_stage_index = manifest.stages.size();
        rendered.compile_node_index = node_index;
        manifest.stages.push_back(std::move(rendered));
      }
  } catch (const std::exception& error) {
    if (!diagnostics)
      throw;
    diagnostics->errors.push_back(std::string("contract render: ") + error.what());
    return std::nullopt;
  }

  if (diagnostics && !diagnostics->errors.empty()) {
    return std::nullopt;
  }

  if (manifest.stages.empty())
    return std::nullopt;
  return manifest;
}

} // namespace simaai::neat

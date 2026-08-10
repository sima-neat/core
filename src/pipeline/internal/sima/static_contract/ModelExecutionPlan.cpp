#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {

bool fail(std::string* error, std::string detail) {
  if (error) {
    *error = std::move(detail);
  }
  return false;
}

bool config_matches_kind(const OpSpec& op) {
  switch (op.kind) {
  case OpKind::Cast:
    return std::holds_alternative<CastOpConfig>(op.config);
  case OpKind::Quantize:
    return std::holds_alternative<QuantizeOpConfig>(op.config);
  case OpKind::Tessellate:
    return std::holds_alternative<TessellateOpConfig>(op.config);
  case OpKind::Pack:
    return std::holds_alternative<PackOpConfig>(op.config);
  case OpKind::Mla:
    return std::holds_alternative<MlaOpConfig>(op.config);
  case OpKind::Unpack:
    return std::holds_alternative<UnpackOpConfig>(op.config);
  case OpKind::Slice:
    return std::holds_alternative<SliceOpConfig>(op.config);
  case OpKind::Reshape:
    return std::holds_alternative<ReshapeOpConfig>(op.config);
  case OpKind::Detessellate:
    return std::holds_alternative<DetessellateOpConfig>(op.config);
  case OpKind::Dequantize:
    return std::holds_alternative<DequantizeOpConfig>(op.config);
  case OpKind::HostTvm:
    return std::holds_alternative<HostTvmOpConfig>(op.config);
  case OpKind::PassThrough:
    return std::holds_alternative<PassThroughOpConfig>(op.config);
  }
  return false;
}

bool checked_mul(const std::uint64_t lhs, const std::uint64_t rhs, std::uint64_t* result) {
  if (!result || (rhs != 0U && lhs > std::numeric_limits<std::uint64_t>::max() / rhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_add(const std::uint64_t lhs, const std::uint64_t rhs, std::uint64_t* result) {
  if (!result || lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool validate_read_expression(const ModelExecutionPlanData& data, const ValueSpec& value,
                              std::string* error) {
  if (!value.read_expression.has_value()) {
    return true;
  }
  const auto& expression = *value.read_expression;
  if (expression.source_value_id >= data.values.size() || expression.source_value_id >= value.id) {
    return fail(error, "execution-plan read expression has a missing or forward carrier");
  }
  const auto& source = data.values[expression.source_value_id];
  if (source.read_expression.has_value()) {
    return fail(error, "execution-plan read expression is not composed to its root carrier");
  }
  if (!value.logical_shape.has_value() || value.logical_shape->empty() ||
      expression.stride_bytes.size() != value.logical_shape->size()) {
    return fail(error, "execution-plan read expression has no exact shape/stride relation");
  }

  std::uint64_t element_count = 1U;
  for (const auto dimension : *value.logical_shape) {
    if (dimension <= 0 ||
        !checked_mul(element_count, static_cast<std::uint64_t>(dimension), &element_count)) {
      return fail(error, "execution-plan read expression shape overflows");
    }
  }
  if (element_count == 0U || value.required_bytes % element_count != 0U) {
    return fail(error, "execution-plan read expression has no exact element width");
  }
  const std::uint64_t element_bytes = value.required_bytes / element_count;
  if (element_bytes == 0U) {
    return fail(error, "execution-plan read expression has a zero element width");
  }

  std::uint64_t physical_span = element_bytes;
  for (std::size_t axis = 0; axis < expression.stride_bytes.size(); ++axis) {
    const auto stride = expression.stride_bytes[axis];
    if (stride <= 0) {
      return fail(error, "execution-plan read expression has a non-positive byte stride");
    }
    std::uint64_t axis_span = 0U;
    if (!checked_mul(static_cast<std::uint64_t>((*value.logical_shape)[axis] - 1),
                     static_cast<std::uint64_t>(stride), &axis_span) ||
        !checked_add(physical_span, axis_span, &physical_span)) {
      return fail(error, "execution-plan read expression span overflows");
    }
  }
  const std::uint64_t required_span = std::max(value.required_bytes, physical_span);
  std::uint64_t end = 0U;
  if (!checked_add(expression.byte_offset, required_span, &end) || end > source.required_bytes) {
    return fail(error, "execution-plan read expression exceeds its root carrier");
  }
  return true;
}

bool validate(const ModelExecutionPlanData& data, std::string* error) {
  if (data.contract_version.empty()) {
    return fail(error, "execution plan has an empty contract version");
  }
  if (data.values.empty()) {
    return fail(error, "execution plan has no values");
  }

  std::unordered_set<std::string> value_names;
  for (std::size_t index = 0; index < data.values.size(); ++index) {
    const auto& value = data.values[index];
    if (value.id != index) {
      return fail(error, "execution-plan ValueIds must be dense and ordered");
    }
    if (value.name.empty() || !value_names.emplace(value.name).second) {
      return fail(error, "execution-plan value names must be non-empty and unique");
    }
    if (value.required_bytes == 0U) {
      return fail(error, "execution-plan values must have a non-zero byte extent");
    }
    if (!validate_read_expression(data, value, error)) {
      return false;
    }
  }

  std::unordered_set<ValueId> produced;
  for (const ValueId id : data.model_inputs) {
    if (id >= data.values.size() || !produced.emplace(id).second) {
      return fail(error, "execution-plan model input is invalid or duplicated");
    }
  }

  std::unordered_set<std::string> operation_names;
  std::vector<const OpSpec*> mla_stages;
  for (std::size_t index = 0; index < data.ops.size(); ++index) {
    const auto& op = data.ops[index];
    if (op.id != index || op.sequence != index + 1U) {
      return fail(error, "execution-plan operations must have dense ids and sequences");
    }
    if (op.name.empty() || !operation_names.emplace(op.name).second || op.processor.empty() ||
        !config_matches_kind(op)) {
      return fail(error, "execution-plan operation identity/configuration is invalid");
    }
    if (op.inputs.empty() || op.outputs.empty()) {
      return fail(error, "execution-plan operations must have input and output values");
    }
    for (const ValueId id : op.inputs) {
      if (id >= data.values.size() || !produced.contains(id)) {
        return fail(error, "execution-plan operation references a missing or forward input");
      }
    }
    for (const ValueId id : op.outputs) {
      if (id >= data.values.size() || !produced.emplace(id).second) {
        return fail(error, "execution-plan value has duplicate producers");
      }
    }
    if (op.kind == OpKind::Pack) {
      const auto& pack = std::get<PackOpConfig>(op.config);
      if (op.outputs.size() != 1U || pack.components.size() != op.inputs.size()) {
        return fail(error, "execution-plan Pack has no exact component placement");
      }
      std::uint64_t previous_end = 0U;
      for (std::size_t component_index = 0; component_index < pack.components.size();
           ++component_index) {
        const auto& component = pack.components[component_index];
        const auto input_id = op.inputs[component_index];
        std::uint64_t component_end = 0U;
        if (component.value_id != input_id || component.parent_offset != previous_end ||
            component.parent_offset % 16U != 0U || component.stored_bytes == 0U ||
            component.stored_bytes % 16U != 0U ||
            component.stored_bytes < data.values[input_id].required_bytes ||
            !checked_add(component.parent_offset, component.stored_bytes, &component_end) ||
            component_end > data.values[op.outputs.front()].required_bytes) {
          return fail(error, "execution-plan Pack component placement is invalid");
        }
        previous_end = component_end;
      }
    }
    if (op.kind == OpKind::Mla) {
      const auto* mla = std::get_if<MlaOpConfig>(&op.config);
      if (!mla || mla->executable.empty()) {
        return fail(error, "execution-plan MLA operation has no exact executable identity");
      }
      mla_stages.push_back(&op);
    }
    if (op.kind == OpKind::Reshape) {
      if (op.inputs.size() != 1U || op.outputs.size() != 1U ||
          data.values[op.inputs.front()].required_bytes !=
              data.values[op.outputs.front()].required_bytes ||
          !data.values[op.outputs.front()].read_expression.has_value()) {
        return fail(error, "execution-plan Reshape is not an exact address view");
      }
    }
    if (op.kind == OpKind::HostTvm) {
      const auto& host = std::get<HostTvmOpConfig>(op.config);
      if (host.executable.empty() || host.input_names.size() != op.inputs.size() ||
          host.input_types.size() != op.inputs.size() ||
          host.output_types.size() != op.outputs.size() ||
          host.output_alias_input.size() != op.outputs.size()) {
        return fail(error, "execution-plan HostTVM port contract is incomplete");
      }
      for (std::size_t output_index = 0; output_index < host.output_alias_input.size();
           ++output_index) {
        const auto input_index = host.output_alias_input[output_index];
        if (input_index >= 0 &&
            (static_cast<std::size_t>(input_index) >= op.inputs.size() ||
             data.values[op.inputs[static_cast<std::size_t>(input_index)]].required_bytes !=
                 data.values[op.outputs[output_index]].required_bytes ||
             !data.values[op.outputs[output_index]].read_expression.has_value())) {
          return fail(error, "execution-plan HostTVM alias output is not an exact address view");
        }
      }
    }
  }
  if (produced.size() != data.values.size()) {
    return fail(error, "execution-plan contains a value without a producer");
  }

  std::set<std::tuple<std::size_t, BackendPortDirection, std::size_t>> port_keys;
  std::size_t expected_backend_port_count = 0U;
  for (const auto* stage : mla_stages) {
    expected_backend_port_count += stage->inputs.size() + stage->outputs.size();
  }
  if (data.backend_ports.size() != expected_backend_port_count) {
    return fail(error, "execution-plan backend port count does not cover MLA stage arity");
  }
  for (const auto& port : data.backend_ports) {
    if (port.value_id >= data.values.size() || port.elf_symbol.empty() ||
        port.required_bytes != data.values[port.value_id].required_bytes ||
        port.required_alignment_bytes == 0U ||
        (port.required_alignment_bytes & (port.required_alignment_bytes - 1U)) != 0U) {
      return fail(error, "execution-plan backend port contract is invalid");
    }
    if (port.alignment_authority == BackendPortAlignmentAuthority::LegacyPolicy &&
        port.required_alignment_bytes != kLegacyEvoCmaRegionAlignmentBytes) {
      return fail(error, "execution-plan legacy alignment contradicts the fixed policy");
    }
    if ((port.direction == BackendPortDirection::Input &&
         port.access != BackendPortAccess::ReadOnly) ||
        (port.direction == BackendPortDirection::Output &&
         port.access != BackendPortAccess::WriteOnly)) {
      return fail(error, "execution-plan backend port access contradicts its direction");
    }
    if (!port_keys.emplace(port.stage_index, port.direction, port.port_index).second) {
      return fail(error, "execution-plan backend port index is duplicated");
    }
    if (port.stage_index >= mla_stages.size()) {
      return fail(error, "execution-plan backend port references a missing MLA stage");
    }
    const auto& stage_values = port.direction == BackendPortDirection::Input
                                   ? mla_stages[port.stage_index]->inputs
                                   : mla_stages[port.stage_index]->outputs;
    if (port.port_index >= stage_values.size() || stage_values[port.port_index] != port.value_id) {
      return fail(error, "execution-plan backend port order contradicts the MLA operation");
    }
  }

  std::unordered_set<ValueId> public_values;
  for (std::size_t index = 0; index < data.model_outputs.size(); ++index) {
    const auto& output = data.model_outputs[index];
    if (output.public_index != index || output.name.empty() ||
        output.value_id >= data.values.size() || !public_values.emplace(output.value_id).second) {
      return fail(error, "execution-plan public output contract is invalid");
    }
  }
  if (data.model_outputs.empty()) {
    return fail(error, "execution plan has no public outputs");
  }
  return true;
}

void build_mla_index(ModelExecutionPlanData* data) {
  data->mla_stages.clear();
  std::sort(data->backend_ports.begin(), data->backend_ports.end(),
            [](const BackendPortSpec& lhs, const BackendPortSpec& rhs) {
              return std::tie(lhs.stage_index, lhs.direction, lhs.port_index) <
                     std::tie(rhs.stage_index, rhs.direction, rhs.port_index);
            });

  std::size_t stage_index = 0U;
  for (const auto& op : data->ops) {
    if (op.kind != OpKind::Mla) {
      continue;
    }
    MlaStageSpec stage;
    stage.key.stage_index = stage_index;
    stage.key.op_id = op.id;
    stage.key.logical_stage_id = op.name;
    stage.key.executable = std::get<MlaOpConfig>(op.config).executable;

    const auto first =
        std::lower_bound(data->backend_ports.begin(), data->backend_ports.end(), stage_index,
                         [](const BackendPortSpec& port, const std::size_t wanted) {
                           return port.stage_index < wanted;
                         });
    const auto last = std::upper_bound(first, data->backend_ports.end(), stage_index,
                                       [](const std::size_t wanted, const BackendPortSpec& port) {
                                         return wanted < port.stage_index;
                                       });
    const auto output = std::find_if(first, last, [](const BackendPortSpec& port) {
      return port.direction == BackendPortDirection::Output;
    });
    stage.input_port_begin = static_cast<std::size_t>(first - data->backend_ports.begin());
    stage.input_port_count = static_cast<std::size_t>(output - first);
    stage.output_port_begin = static_cast<std::size_t>(output - data->backend_ports.begin());
    stage.output_port_count = static_cast<std::size_t>(last - output);
    data->mla_stages.push_back(std::move(stage));
    ++stage_index;
  }
}

} // namespace

ModelExecutionPlan::ModelExecutionPlan(std::shared_ptr<const ModelExecutionPlanData> data)
    : data_(std::move(data)) {}

std::optional<ModelExecutionPlan> ModelExecutionPlan::create(ModelExecutionPlanData data,
                                                             std::string* error) {
  if (!validate(data, error)) {
    return std::nullopt;
  }
  build_mla_index(&data);
  if (error) {
    error->clear();
  }
  return ModelExecutionPlan(std::make_shared<const ModelExecutionPlanData>(std::move(data)));
}

const std::string& ModelExecutionPlan::contract_version() const noexcept {
  return data_->contract_version;
}
const std::vector<ValueSpec>& ModelExecutionPlan::values() const noexcept {
  return data_->values;
}
const std::vector<ValueId>& ModelExecutionPlan::model_inputs() const noexcept {
  return data_->model_inputs;
}
const std::vector<OpSpec>& ModelExecutionPlan::ops() const noexcept {
  return data_->ops;
}
const std::vector<BackendPortSpec>& ModelExecutionPlan::backend_ports() const noexcept {
  return data_->backend_ports;
}
std::size_t ModelExecutionPlan::mla_stage_count() const noexcept {
  return data_->mla_stages.size();
}
const MlaStageSpec* ModelExecutionPlan::mla_stage(const std::size_t stage_index) const noexcept {
  return stage_index < data_->mla_stages.size() ? &data_->mla_stages[stage_index] : nullptr;
}
const MlaStageSpec* ModelExecutionPlan::mla_stage_for_op(const OpId op_id) const noexcept {
  const auto found =
      std::find_if(data_->mla_stages.begin(), data_->mla_stages.end(),
                   [op_id](const MlaStageSpec& stage) { return stage.key.op_id == op_id; });
  return found == data_->mla_stages.end() ? nullptr : &*found;
}
const MlaStageSpec*
ModelExecutionPlan::mla_stage_for_identity(const std::string_view logical_stage_id,
                                           const std::string_view executable) const noexcept {
  const auto found = std::find_if(
      data_->mla_stages.begin(), data_->mla_stages.end(), [&](const MlaStageSpec& stage) {
        return stage.key.logical_stage_id == logical_stage_id && stage.key.executable == executable;
      });
  return found == data_->mla_stages.end() ? nullptr : &*found;
}
std::span<const BackendPortSpec>
ModelExecutionPlan::backend_ports(const std::size_t stage_index,
                                  const BackendPortDirection direction) const noexcept {
  const auto* stage = mla_stage(stage_index);
  if (!stage) {
    return {};
  }
  const auto begin =
      direction == BackendPortDirection::Input ? stage->input_port_begin : stage->output_port_begin;
  const auto count =
      direction == BackendPortDirection::Input ? stage->input_port_count : stage->output_port_count;
  return std::span<const BackendPortSpec>(data_->backend_ports).subspan(begin, count);
}
const std::vector<ModelOutputSpec>& ModelExecutionPlan::model_outputs() const noexcept {
  return data_->model_outputs;
}
const ValueSpec* ModelExecutionPlan::value(const ValueId id) const noexcept {
  return id < data_->values.size() ? &data_->values[id] : nullptr;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

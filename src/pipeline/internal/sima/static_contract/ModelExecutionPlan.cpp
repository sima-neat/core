#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
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

bool is_power_of_two(const std::size_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

std::optional<std::uint64_t> physical_span_for(const ValueSpec& value,
                                               const std::vector<std::int64_t>& strides) {
  if (!value.logical_shape.has_value() || value.logical_shape->empty() ||
      strides.size() != value.logical_shape->size()) {
    return std::nullopt;
  }
  std::uint64_t element_count = 1U;
  for (const auto dimension : *value.logical_shape) {
    if (dimension <= 0 ||
        !checked_mul(element_count, static_cast<std::uint64_t>(dimension), &element_count)) {
      return std::nullopt;
    }
  }
  if (element_count == 0U || value.required_bytes % element_count != 0U) {
    return std::nullopt;
  }
  const std::uint64_t element_bytes = value.required_bytes / element_count;
  if (element_bytes == 0U) {
    return std::nullopt;
  }
  std::uint64_t span = element_bytes;
  for (std::size_t reverse = strides.size(); reverse > 0U; --reverse) {
    const std::size_t axis = reverse - 1U;
    if (strides[axis] <= 0 || static_cast<std::uint64_t>(strides[axis]) < span) {
      return std::nullopt;
    }
    std::uint64_t axis_span = 0U;
    if (!checked_mul(static_cast<std::uint64_t>((*value.logical_shape)[axis] - 1),
                     static_cast<std::uint64_t>(strides[axis]), &axis_span) ||
        !checked_add(span, axis_span, &span)) {
      return std::nullopt;
    }
  }
  return span;
}

struct ByteInterval {
  std::uint64_t begin = 0U;
  std::uint64_t end = 0U;
};

std::optional<std::vector<ByteInterval>> authored_write_intervals(
    const ValueSpec& value, const std::uint64_t backend_write_extent = 0U) {
  if (!value.storage_binding) {
    return std::nullopt;
  }
  const auto& binding = *value.storage_binding;
  std::uint64_t contiguous_end = 0U;
  // An MLA backend port owns its complete physical zone, including row or
  // tail padding which is not part of the logical tensor view.
  if (backend_write_extent != 0U) {
    if (!checked_add(binding.byte_offset, backend_write_extent, &contiguous_end)) {
      return std::nullopt;
    }
    return std::vector<ByteInterval>{{binding.byte_offset, contiguous_end}};
  }
  if (binding.stride_bytes.empty() || binding.physical_span == value.required_bytes) {
    if (!checked_add(binding.byte_offset, binding.physical_span, &contiguous_end)) {
      return std::nullopt;
    }
    return std::vector<ByteInterval>{{binding.byte_offset, contiguous_end}};
  }
  if (!value.logical_shape || value.logical_shape->empty()) {
    return std::nullopt;
  }

  // Preserve the existing exact disjoint footprint for direct shared-carrier
  // batch placement when its specialized form is proven.
  const auto& shape = *value.logical_shape;
  const auto& strides = binding.stride_bytes;
  const auto exact_batch = [&]() -> std::optional<std::vector<ByteInterval>> {
    if (strides.size() != shape.size() || shape.front() <= 0) {
      return std::nullopt;
    }
    std::uint64_t elements = 1U;
    for (const auto dimension : shape) {
      if (dimension <= 0 ||
          !checked_mul(elements, static_cast<std::uint64_t>(dimension), &elements)) {
        return std::nullopt;
      }
    }
    if (elements == 0U || value.required_bytes % elements != 0U) {
      return std::nullopt;
    }
    const std::uint64_t element_bytes = value.required_bytes / elements;
    std::uint64_t expected_stride = element_bytes;
    for (std::size_t axis = shape.size(); axis-- > 1U;) {
      if (strides[axis] <= 0 ||
          static_cast<std::uint64_t>(strides[axis]) != expected_stride ||
          !checked_mul(expected_stride, static_cast<std::uint64_t>(shape[axis]),
                       &expected_stride)) {
        return std::nullopt;
      }
    }
    const std::uint64_t row_bytes =
        value.required_bytes / static_cast<std::uint64_t>(shape.front());
    if (expected_stride != row_bytes || strides.front() <= 0 ||
        static_cast<std::uint64_t>(strides.front()) < row_bytes) {
      return std::nullopt;
    }
    std::uint64_t expected_span = 0U;
    if (!checked_mul(static_cast<std::uint64_t>(shape.front() - 1),
                     static_cast<std::uint64_t>(strides.front()), &expected_span) ||
        !checked_add(expected_span, row_bytes, &expected_span) ||
        expected_span > binding.physical_span) {
      return std::nullopt;
    }
    std::vector<ByteInterval> result;
    result.reserve(static_cast<std::size_t>(shape.front()));
    for (std::int64_t batch = 0; batch < shape.front(); ++batch) {
      std::uint64_t row_offset = 0U;
      std::uint64_t begin = 0U;
      std::uint64_t end = 0U;
      if (!checked_mul(static_cast<std::uint64_t>(batch),
                       static_cast<std::uint64_t>(strides.front()), &row_offset) ||
          !checked_add(binding.byte_offset, row_offset, &begin) ||
          !checked_add(begin, row_bytes, &end)) {
        return std::nullopt;
      }
      result.push_back({begin, end});
    }
    return result;
  }();
  if (exact_batch) {
    return exact_batch;
  }

  // Any other validated positive monotonic affine root write is accounted as
  // its complete physical span. This is conservative and never creates holes.
  if (binding.kind != StorageBindingKind::Root) {
    return std::nullopt;
  }
  std::uint64_t end = 0U;
  if (!checked_add(binding.byte_offset, binding.physical_span, &end)) {
    return std::nullopt;
  }
  return std::vector<ByteInterval>{{binding.byte_offset, end}};
}

bool normalize_storage(ModelExecutionPlanData* data, std::string* error) {
  if (!data) {
    return fail(error, "execution plan has no mutable construction data");
  }
  std::unordered_set<ValueId> model_inputs(data->model_inputs.begin(), data->model_inputs.end());
  for (auto& value : data->values) {
    if (!value.storage_binding.has_value()) {
      StorageBinding binding;
      if (value.read_expression.has_value()) {
        const auto& expression = *value.read_expression;
        if (expression.source_value_id >= value.id ||
            expression.source_value_id >= data->values.size() ||
            !data->values[expression.source_value_id].storage_binding.has_value()) {
          return fail(error, "execution-plan read expression has no normalized root carrier");
        }
        const auto& source = *data->values[expression.source_value_id].storage_binding;
        binding.kind = StorageBindingKind::View;
        binding.carrier_id = source.carrier_id;
        if (!checked_add(source.byte_offset, expression.byte_offset, &binding.byte_offset)) {
          return fail(error, "execution-plan view offset overflows");
        }
        const auto span = physical_span_for(value, expression.stride_bytes);
        if (!span) {
          return fail(error, "execution-plan view '" + value.name +
                                 "' has no exact physical span");
        }
        binding.physical_span = *span;
        binding.stride_bytes = expression.stride_bytes;
        binding.access = StorageAccess::ReadOnly;
        binding.source_value_id = expression.source_value_id;
      } else {
        binding.kind = model_inputs.contains(value.id) ? StorageBindingKind::External
                                                       : StorageBindingKind::Root;
        binding.carrier_id = value.id;
        binding.physical_span = value.required_bytes;
        binding.access = model_inputs.contains(value.id) ? StorageAccess::ReadOnly
                                                         : StorageAccess::ReadWrite;
      }
      value.storage_binding = std::move(binding);
    } else if (value.storage_binding->kind == StorageBindingKind::View &&
               !value.read_expression.has_value()) {
      if (!value.storage_binding->source_value_id.has_value()) {
        return fail(error, "execution-plan view binding has no source value");
      }
      value.read_expression = ReadExpression{*value.storage_binding->source_value_id,
                                             value.storage_binding->byte_offset,
                                             value.storage_binding->stride_bytes};
    }
  }

  if (data->carriers.empty()) {
    std::map<CarrierId, CarrierSpec> carriers;
    for (const auto& value : data->values) {
      const auto& binding = *value.storage_binding;
      std::uint64_t end = 0U;
      if (!checked_add(binding.byte_offset, binding.physical_span, &end)) {
        return fail(error, "execution-plan carrier extent overflows");
      }
      auto [iterator, inserted] = carriers.emplace(
          binding.carrier_id,
          CarrierSpec{binding.carrier_id, end, kLegacyEvoCmaRegionAlignmentBytes,
                      value.representation});
      if (!inserted) {
        iterator->second.required_bytes = std::max(iterator->second.required_bytes, end);
      }
    }
    for (const auto& port : data->backend_ports) {
      if (port.value_id >= data->values.size() ||
          !data->values[port.value_id].storage_binding.has_value()) {
        return fail(error, "execution-plan backend port has no storage binding");
      }
      const auto id = data->values[port.value_id].storage_binding->carrier_id;
      const auto found = carriers.find(id);
      if (found == carriers.end() || port.physical_extent_bytes == 0U) {
        return fail(error, "execution-plan backend port has no storage carrier extent");
      }
      std::uint64_t port_end = 0U;
      const auto& binding = *data->values[port.value_id].storage_binding;
      if (!checked_add(binding.byte_offset, port.physical_extent_bytes, &port_end)) {
        return fail(error, "execution-plan backend port carrier extent overflows");
      }
      found->second.required_bytes = std::max(found->second.required_bytes, port_end);
      found->second.required_alignment_bytes =
          std::max(found->second.required_alignment_bytes, port.required_alignment_bytes);
    }
    data->carriers.reserve(carriers.size());
    for (auto& [id, carrier] : carriers) {
      (void)id;
      data->carriers.push_back(std::move(carrier));
    }
  }
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

  const auto required_span = physical_span_for(value, expression.stride_bytes);
  if (!required_span) {
    return fail(error, "execution-plan read expression has no exact physical span");
  }
  std::uint64_t end = 0U;
  const auto& source_binding = *source.storage_binding;
  const auto carrier = std::find_if(data.carriers.begin(), data.carriers.end(),
                                    [&](const CarrierSpec& item) {
                                      return item.id == source_binding.carrier_id;
                                    });
  if (carrier == data.carriers.end() ||
      !checked_add(expression.byte_offset, *required_span, &end) ||
      end > carrier->required_bytes) {
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

  std::unordered_set<CarrierId> carrier_ids;
  for (const auto& carrier : data.carriers) {
    if (!carrier_ids.emplace(carrier.id).second || carrier.required_bytes == 0U ||
        !is_power_of_two(carrier.required_alignment_bytes)) {
      return fail(error, "execution-plan carrier identity/extent/alignment is invalid");
    }
  }
  if (data.carriers.empty()) {
    return fail(error, "execution plan has no storage carriers");
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
    if (!value.storage_binding.has_value()) {
      return fail(error, "execution-plan value has no storage binding");
    }
    const auto& binding = *value.storage_binding;
    const auto carrier = std::find_if(data.carriers.begin(), data.carriers.end(),
                                      [&](const CarrierSpec& item) {
                                        return item.id == binding.carrier_id;
                                      });
    std::uint64_t binding_end = 0U;
    if (carrier == data.carriers.end() || binding.physical_span == 0U ||
        !checked_add(binding.byte_offset, binding.physical_span, &binding_end) ||
        binding_end > carrier->required_bytes) {
      return fail(error, "execution-plan value binding exceeds its carrier");
    }
    if (!binding.stride_bytes.empty()) {
      const auto span = physical_span_for(value, binding.stride_bytes);
      if (!span || *span > binding.physical_span) {
        return fail(error, "execution-plan storage strides disagree with its physical span");
      }
    } else if (binding.physical_span < value.required_bytes) {
      return fail(error, "execution-plan opaque storage is smaller than its logical value");
    }
    if (binding.kind == StorageBindingKind::View) {
      if (!binding.source_value_id.has_value() || *binding.source_value_id >= value.id ||
          *binding.source_value_id >= data.values.size() ||
          !data.values[*binding.source_value_id].storage_binding.has_value() ||
          data.values[*binding.source_value_id].storage_binding->carrier_id != binding.carrier_id) {
        return fail(error, "execution-plan view binding has no earlier source on the same carrier");
      }
    } else if (binding.source_value_id.has_value()) {
      return fail(error, "execution-plan root/external binding unexpectedly names a source value");
    }
    if (!validate_read_expression(data, value, error)) {
      return false;
    }
  }

  std::unordered_set<ValueId> produced;
  std::vector<std::optional<OpId>> value_producer(data.values.size());
  for (const ValueId id : data.model_inputs) {
    if (id >= data.values.size() || !produced.emplace(id).second) {
      return fail(error, "execution-plan model input is invalid or duplicated");
    }
  }

  const auto value_is_available = [&](const ValueId id) {
    if (produced.contains(id)) {
      return true;
    }
    const auto& value = data.values[id];
    return value.storage_binding->kind == StorageBindingKind::View &&
           value.storage_binding->source_value_id.has_value() &&
           produced.contains(*value.storage_binding->source_value_id);
  };

  std::unordered_set<std::string> operation_names;
  std::vector<const OpSpec*> mla_stages;
  std::unordered_map<CarrierId, std::vector<ByteInterval>> authored_writes;
  std::uint64_t previous_sequence = 0U;
  for (std::size_t index = 0; index < data.ops.size(); ++index) {
    const auto& op = data.ops[index];
    if (op.id != index || op.sequence == 0U || op.sequence <= previous_sequence) {
      return fail(error, "execution-plan commands need dense ids and strictly increasing order");
    }
    previous_sequence = op.sequence;
    if (op.name.empty() || !operation_names.emplace(op.name).second || op.processor.empty() ||
        !config_matches_kind(op)) {
      return fail(error, "execution-plan operation identity/configuration is invalid");
    }
    if (op.inputs.empty() || op.outputs.empty()) {
      return fail(error, "execution-plan operations must have input and output values");
    }
    std::unordered_set<OpId> dependencies;
    for (const auto dependency : op.dependencies) {
      if (dependency >= op.id || !dependencies.emplace(dependency).second) {
        return fail(error, "execution-plan command dependency is invalid or duplicated");
      }
    }
    for (const ValueId id : op.inputs) {
      if (id >= data.values.size() || !value_is_available(id)) {
        return fail(error, "execution-plan operation references a missing or forward input");
      }
      const auto& binding = *data.values[id].storage_binding;
      const ValueId producer_value =
          binding.kind == StorageBindingKind::View && binding.source_value_id.has_value()
              ? *binding.source_value_id
              : id;
      const auto producer = value_producer[producer_value];
      if (producer.has_value() && !dependencies.empty() && !dependencies.contains(*producer)) {
        return fail(error,
                    "execution-plan command omits a dependency for an authored input producer");
      }
    }
    for (const ValueId id : op.outputs) {
      if (id >= data.values.size() || !produced.emplace(id).second) {
        return fail(error, "execution-plan value has duplicate producers");
      }
      value_producer[id] = op.id;
    }
    const bool pack_relation =
        op.kind == OpKind::Pack && !std::get<PackOpConfig>(op.config).materializes;
    if (!pack_relation) {
      for (const ValueId id : op.outputs) {
        const auto& binding = *data.values[id].storage_binding;
        // A view/alias output publishes existing storage; it is not a write.
        if (binding.kind == StorageBindingKind::View) {
          continue;
        }
        const auto& output_value = data.values[id];
        const auto carrier = std::find_if(data.carriers.begin(), data.carriers.end(),
                                          [&](const CarrierSpec& item) {
                                            return item.id == binding.carrier_id;
                                          });
        std::uint64_t backend_write_extent = 0U;
        if (op.kind == OpKind::Mla) {
          const auto port = std::find_if(
              data.backend_ports.begin(), data.backend_ports.end(),
              [&](const BackendPortSpec& candidate) {
                return candidate.stage_index == mla_stages.size() &&
                       candidate.direction == BackendPortDirection::Output &&
                       candidate.value_id == id;
              });
          if (port == data.backend_ports.end()) {
            return fail(error,
                        "execution-plan MLA output has no exact backend write extent");
          }
          backend_write_extent = port->physical_extent_bytes;
        }
        const auto intervals =
            carrier == data.carriers.end()
                ? std::optional<std::vector<ByteInterval>>{}
                : authored_write_intervals(output_value, backend_write_extent);
        if (!intervals) {
          return fail(error,
                      "execution-plan command output has no exact bounded write footprint");
        }
        auto& writes = authored_writes[binding.carrier_id];
        for (const auto& interval : *intervals) {
          for (const auto& previous : writes) {
            if (interval.begin < previous.end && previous.begin < interval.end) {
              return fail(error,
                          "execution-plan commands have overlapping authored carrier writes");
            }
          }
          writes.push_back(interval);
        }
      }
    }
    if (op.kind == OpKind::Pack) {
      const auto& pack = std::get<PackOpConfig>(op.config);
      if (op.outputs.size() != 1U) {
        return fail(error, "execution-plan Pack has no exact component placement");
      }
      if (!pack.spans.empty()) {
        if (pack.batch_count == 0U ||
            pack.parent_required_bytes != data.values[op.outputs.front()].required_bytes ||
            pack.spans.size() != op.inputs.size() * pack.batch_count) {
          return fail(error, "execution-plan Pack span cardinality/parent extent is invalid");
        }
        std::set<std::pair<ValueId, std::uint32_t>> members;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> destinations;
        for (const auto& span : pack.spans) {
          std::uint64_t source_end = 0U;
          std::uint64_t destination_end = 0U;
          if (std::find(op.inputs.begin(), op.inputs.end(), span.value_id) == op.inputs.end() ||
              span.batch_index >= pack.batch_count ||
              !members.emplace(span.value_id, span.batch_index).second ||
              span.logical_bytes == 0U || span.stored_bytes < span.logical_bytes ||
              span.padding_policy.empty() ||
              !checked_add(span.source_byte_offset, span.logical_bytes, &source_end) ||
              source_end > data.values[span.value_id].required_bytes ||
              !checked_add(span.parent_offset, span.stored_bytes, &destination_end) ||
              destination_end > pack.parent_required_bytes) {
            return fail(error, "execution-plan Pack span is invalid");
          }
          for (const auto& [begin, end] : destinations) {
            if (span.parent_offset < end && begin < destination_end) {
              return fail(error, "execution-plan Pack destination spans overlap");
            }
          }
          destinations.emplace_back(span.parent_offset, destination_end);
        }
        std::sort(destinations.begin(), destinations.end());
        std::uint64_t covered = 0U;
        for (const auto& [begin, end] : destinations) {
          if (begin != covered) {
            return fail(error, "execution-plan Pack destination has a gap");
          }
          covered = end;
        }
        if (covered != pack.parent_required_bytes) {
          return fail(error, "execution-plan Pack does not cover its exact parent carrier");
        }
        if (!pack.materializes) {
          const auto& parent = *data.values[op.outputs.front()].storage_binding;
          if (pack.parent_required_bytes % pack.batch_count != 0U) {
            return fail(error, "execution-plan direct Pack parent is not batch-divisible");
          }
          const std::uint64_t parent_row_bytes =
              pack.parent_required_bytes / static_cast<std::uint64_t>(pack.batch_count);
          for (const auto input_id : op.inputs) {
            const auto& input = data.values[input_id];
            const auto& binding = *input.storage_binding;
            if (!input.logical_shape || input.logical_shape->empty() ||
                input.logical_shape->front() != pack.batch_count ||
                input.required_bytes % pack.batch_count != 0U ||
                binding.kind == StorageBindingKind::View ||
                binding.carrier_id != parent.carrier_id) {
              return fail(error, "execution-plan direct Pack child has no shared carrier binding");
            }
            const auto footprint = authored_write_intervals(input);
            if (!footprint || footprint->size() != pack.batch_count) {
              return fail(error, "execution-plan direct Pack child has no pitched write footprint");
            }
            const std::uint64_t row_bytes = input.required_bytes / pack.batch_count;
            for (std::uint32_t batch = 0U; batch < pack.batch_count; ++batch) {
              const auto member = std::find_if(
                  pack.spans.begin(), pack.spans.end(), [&](const PackSpan& span) {
                    return span.value_id == input_id && span.batch_index == batch;
                  });
              if (member == pack.spans.end() ||
                  member->source_byte_offset != batch * row_bytes ||
                  member->logical_bytes != row_bytes || member->stored_bytes != row_bytes ||
                  member->padding_policy != "none" ||
                  member->parent_offset != (*footprint)[batch].begin ||
                  member->parent_offset + member->stored_bytes != (*footprint)[batch].end ||
                  (batch > 0U &&
                   member->parent_offset - (*footprint)[batch - 1U].begin != parent_row_bytes)) {
                return fail(error,
                            "execution-plan direct Pack spans contradict child placement");
              }
            }
          }
        }
      } else {
        if (pack.components.size() != op.inputs.size()) {
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
      std::unordered_set<std::string> host_arguments;
      for (const auto& name : host.input_names) {
        if (name.empty() || !host_arguments.emplace(name).second) {
          return fail(error, "execution-plan HostTVM external input names are invalid");
        }
      }
      for (const auto& name : host.linked_parameter_names) {
        if (name.empty() || !host_arguments.emplace(name).second) {
          return fail(error,
                      "execution-plan HostTVM linked parameters overlap or are duplicated");
        }
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
  for (const auto& value : data.values) {
    if (!produced.contains(value.id) &&
        !(value.storage_binding->kind == StorageBindingKind::View &&
          value.storage_binding->source_value_id.has_value() &&
          produced.contains(*value.storage_binding->source_value_id))) {
      return fail(error, "execution-plan contains a value without a producer or view source");
    }
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
        port.physical_extent_bytes < data.values[port.value_id].required_bytes ||
        port.required_alignment_bytes == 0U ||
        (port.required_alignment_bytes & (port.required_alignment_bytes - 1U)) != 0U) {
      return fail(error, "execution-plan backend port contract is invalid");
    }
    const auto& binding = *data.values[port.value_id].storage_binding;
    const auto carrier = std::find_if(data.carriers.begin(), data.carriers.end(),
                                      [&](const CarrierSpec& item) {
                                        return item.id == binding.carrier_id;
                                      });
    std::uint64_t port_end = 0U;
    if (carrier == data.carriers.end() ||
        !checked_add(binding.byte_offset, port.physical_extent_bytes, &port_end) ||
        port_end > carrier->required_bytes) {
      return fail(error,
                  "execution-plan backend extent exceeds its storage carrier");
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
  if (!normalize_storage(&data, error)) {
    return std::nullopt;
  }
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
const std::vector<CarrierSpec>& ModelExecutionPlan::carriers() const noexcept {
  return data_->carriers;
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
const CarrierSpec* ModelExecutionPlan::carrier(const CarrierId id) const noexcept {
  const auto found = std::find_if(data_->carriers.begin(), data_->carriers.end(),
                                  [id](const CarrierSpec& carrier) { return carrier.id == id; });
  return found == data_->carriers.end() ? nullptr : &*found;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

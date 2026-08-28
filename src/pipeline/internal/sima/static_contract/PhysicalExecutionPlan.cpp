#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <gst/SimaCvuCapabilityAbi.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {

struct DraftCommand {
  std::uint64_t stable_order = 0;
  PhysicalCohortId cohort_id = 0;
  PhysicalEngine engine = PhysicalEngine::Cvu;
  PhysicalCommandRole role = PhysicalCommandRole::NonCvu;
  std::string implementation_id;
  std::uint32_t graph_id = 0U;
  std::uint32_t batch_size = 1U;
  std::uint32_t maximum_members = 0U;
  std::vector<PhysicalCommandMember> members;
  std::set<std::size_t> predecessor_drafts;
};

bool record_error(std::string* error, std::string detail) {
  if (error) {
    *error = std::move(detail);
  }
  return false;
}

bool is_relation(const OpSpec& op) {
  if (op.kind == OpKind::Unpack || op.kind == OpKind::Slice || op.kind == OpKind::Reshape ||
      op.kind == OpKind::PassThrough) {
    return true;
  }
  const auto* pack = std::get_if<PackOpConfig>(&op.config);
  return op.kind == OpKind::Pack && pack && !pack->materializes;
}

bool is_dense_order_preserving_view(const ValueSpec& root, const ValueSpec& previous,
                                    const ValueSpec& view) {
  if (root.read_expression || !root.storage_binding || !previous.storage_binding ||
      !view.read_expression || !view.storage_binding || !view.logical_shape ||
      view.logical_shape->empty() || root.required_bytes != previous.required_bytes ||
      previous.required_bytes != view.required_bytes) {
    return false;
  }
  const auto& expression = *view.read_expression;
  const auto& root_binding = *root.storage_binding;
  const auto& previous_binding = *previous.storage_binding;
  const auto& view_binding = *view.storage_binding;
  if (expression.source_value_id != root.id || expression.byte_offset != 0U ||
      view_binding.kind != StorageBindingKind::View ||
      view_binding.source_value_id != std::optional<ValueId>{root.id} ||
      previous_binding.carrier_id != root_binding.carrier_id ||
      view_binding.carrier_id != previous_binding.carrier_id ||
      previous_binding.byte_offset != root_binding.byte_offset ||
      view_binding.byte_offset != previous_binding.byte_offset ||
      view_binding.physical_span != view.required_bytes ||
      view_binding.stride_bytes != expression.stride_bytes) {
    return false;
  }

  std::uint64_t elements = 1U;
  for (const auto dimension : *view.logical_shape) {
    if (dimension <= 0 ||
        elements > std::numeric_limits<std::uint64_t>::max() /
                       static_cast<std::uint64_t>(dimension)) {
      return false;
    }
    elements *= static_cast<std::uint64_t>(dimension);
  }
  if (elements == 0U || view.required_bytes % elements != 0U ||
      expression.stride_bytes.size() != view.logical_shape->size()) {
    return false;
  }
  std::uint64_t stride = view.required_bytes / elements;
  for (std::size_t reverse = view.logical_shape->size(); reverse > 0U; --reverse) {
    const auto axis = reverse - 1U;
    if (stride > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        expression.stride_bytes[axis] != static_cast<std::int64_t>(stride)) {
      return false;
    }
    if (axis > 0U) {
      const auto dimension = static_cast<std::uint64_t>(view.logical_shape->at(axis));
      if (stride > std::numeric_limits<std::uint64_t>::max() / dimension) {
        return false;
      }
      stride *= dimension;
    }
  }
  return stride == view.required_bytes;
}

std::optional<std::uint32_t> cvu_semantic_op(const OpKind kind) {
  switch (kind) {
  case OpKind::Cast:
    return SIMA_CVU_SEMANTIC_OP_CAST;
  case OpKind::Quantize:
    return SIMA_CVU_SEMANTIC_OP_QUANTIZE;
  case OpKind::Tessellate:
    return SIMA_CVU_SEMANTIC_OP_TESSELLATE;
  case OpKind::Detessellate:
    return SIMA_CVU_SEMANTIC_OP_DETESSELLATE;
  case OpKind::Dequantize:
    return SIMA_CVU_SEMANTIC_OP_DEQUANTIZE;
  case OpKind::Pack:
  case OpKind::Mla:
  case OpKind::Unpack:
  case OpKind::Slice:
  case OpKind::Reshape:
  case OpKind::HostTvm:
  case OpKind::PassThrough:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<SimaCvuCapabilityAbiRecord>
cvu_capability(const std::vector<OpId>& chain, const ModelExecutionPlan& plan) {
  if (chain.empty() || chain.size() > 2U) {
    return std::nullopt;
  }
  const auto first = chain.front() < plan.ops().size()
                         ? cvu_semantic_op(plan.ops()[chain.front()].kind)
                         : std::nullopt;
  const auto second = chain.size() == 2U && chain.back() < plan.ops().size()
                          ? cvu_semantic_op(plan.ops()[chain.back()].kind)
                          : std::optional<std::uint32_t>{SIMA_CVU_SEMANTIC_OP_NONE};
  if (!first || !second) {
    return std::nullopt;
  }
  SimaCvuCapabilityAbiRecord record{};
  if (!sima_cvu_capability_abi_lookup_semantic_pattern(
          static_cast<std::uint32_t>(chain.size()), *first, *second, &record) ||
      record.maximum_members == 0U) {
    return std::nullopt;
  }
  return record;
}

bool is_groupable_cvu(const OpSpec& op, const ModelExecutionPlan& plan) {
  if (op.processor != "EV74" || op.inputs.size() != 1U || op.outputs.size() != 1U ||
      !cvu_semantic_op(op.kind)) {
    return false;
  }
  return cvu_capability(std::vector<OpId>{op.id}, plan).has_value();
}

std::string cvu_implementation_id(const SimaCvuCapabilityAbiRecord& capability) {
  return "cvu.graph" + std::to_string(capability.graph_id) + "." +
         capability.canonical_token + ".v" +
         std::to_string(capability.descriptor_contract_version);
}

std::string implementation_id(const OpSpec& op, const ModelExecutionPlan& plan) {
  if (const auto capability = cvu_capability(std::vector<OpId>{op.id}, plan)) {
    return cvu_implementation_id(*capability);
  }
  switch (op.kind) {
  case OpKind::Cast:
  case OpKind::Quantize:
  case OpKind::Tessellate:
  case OpKind::Detessellate:
  case OpKind::Dequantize:
    return {};
  case OpKind::Mla:
    return "mla.elf.v1:" + std::get<MlaOpConfig>(op.config).executable;
  case OpKind::HostTvm:
    return "a65.tvm-graph-executor.v1:" + std::get<HostTvmOpConfig>(op.config).executable;
  case OpKind::Pack:
    // No generic materializer exists. A materializing Pack needs an explicit
    // registered device implementation; relation-only Pack was removed above.
    return {};
  case OpKind::Unpack:
  case OpKind::Slice:
  case OpKind::Reshape:
  case OpKind::PassThrough:
    break;
  }
  return {};
}

PhysicalEngine engine_for(const OpSpec& op) {
  if (op.kind == OpKind::Mla) {
    return PhysicalEngine::Mla;
  }
  return op.kind == OpKind::HostTvm ? PhysicalEngine::A65 : PhysicalEngine::Cvu;
}

std::int64_t batch_for(const OpSpec& op) {
  if (op.input_shapes.empty() || op.input_shapes.front().empty()) {
    return 1;
  }
  return op.input_shapes.front().front();
}

std::string canonical_dtype(std::string value) {
  std::string result;
  result.reserve(value.size());
  for (const auto character : value) {
    if (character != '_' && character != '-' && !std::isspace(static_cast<unsigned char>(character))) {
      result.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(character))));
    }
  }
  if (result == "FLOAT" || result == "FLOAT32" || result == "F32") {
    return "FP32";
  }
  if (result == "BFLOAT16" || result == "B16") {
    return "BF16";
  }
  if (result == "I8") {
    return "INT8";
  }
  return result;
}

bool value_has_dtype(const ModelExecutionPlan& plan, const ValueId value_id,
                     const std::string_view expected, std::string* detail) {
  const auto* value = plan.value(value_id);
  if (!value || !value->logical_dtype ||
      canonical_dtype(*value->logical_dtype) != expected) {
    if (detail) {
      *detail = value ? "value '" + value->name + "' must have dtype " +
                            std::string(expected)
                      : "fused transform references an absent value";
    }
    return false;
  }
  return true;
}

bool positive_quantization(const std::vector<QuantizationSpec>& params) {
  // Tensor-transform-pair ABI v1 carries one scale/zero-point per member. Do
  // not silently collapse an authored per-channel contract to its first row.
  return params.size() == 1U &&
         std::all_of(params.begin(), params.end(), [](const auto& param) {
           return std::isfinite(param.scale) && param.scale > 0.0;
         });
}

bool validate_fused_member(const ModelExecutionPlan& plan,
                           const std::vector<OpId>& chain,
                           std::string* detail) {
  if (chain.size() != 2U || chain[0] >= plan.ops().size() ||
      chain[1] >= plan.ops().size()) {
    return record_error(detail, "fused transform must contain two valid operations");
  }
  const auto& first = plan.ops()[chain[0]];
  const auto& second = plan.ops()[chain[1]];
  if (first.processor != second.processor || first.processor != "EV74" ||
      first.inputs.size() != 1U || first.outputs.size() != 1U ||
      second.inputs.size() != 1U || second.outputs.size() != 1U ||
      !resolve_exact_private_ordered_relation_path(plan, first.id, second.id)) {
    return record_error(
        detail,
        "registered fused transform requires one exact direct or private ordered-view path");
  }
  const bool direct_edge = first.outputs.front() == second.inputs.front();
  if (direct_edge && !first.output_shapes.empty() && !second.input_shapes.empty() &&
      first.output_shapes.front() != second.input_shapes.front()) {
    return record_error(detail, "registered fused transform has incompatible intermediate shapes");
  }
  if (batch_for(first) != batch_for(second)) {
    return record_error(detail, "registered fused transform has incompatible batch contracts");
  }

  const auto outer_input = first.inputs.front();
  const auto intermediate = first.outputs.front();
  const auto transformed_intermediate = second.inputs.front();
  const auto outer_output = second.outputs.front();
  if (first.kind == OpKind::Quantize && second.kind == OpKind::Tessellate) {
    const auto* quantize = std::get_if<QuantizeOpConfig>(&first.config);
    const auto* tessellate = std::get_if<TessellateOpConfig>(&second.config);
    if (!quantize || !tessellate || canonical_dtype(quantize->output_dtype) != "INT8" ||
        canonical_dtype(tessellate->frame_type) != "INT8" ||
        quantize->rounding.empty() || !positive_quantization(quantize->channel_params) ||
        tessellate->slice_shape.empty() ||
        !value_has_dtype(plan, outer_input, "FP32", detail) ||
        !value_has_dtype(plan, intermediate, "INT8", detail) ||
        !value_has_dtype(plan, transformed_intermediate, "INT8", detail) ||
        !value_has_dtype(plan, outer_output, "INT8", detail)) {
      if (detail && detail->empty()) {
        *detail = "quantize+tessellate does not satisfy the graph contract";
      }
      return false;
    }
    return true;
  }
  if (first.kind == OpKind::Cast && second.kind == OpKind::Tessellate) {
    const auto* cast = std::get_if<CastOpConfig>(&first.config);
    const auto* tessellate = std::get_if<TessellateOpConfig>(&second.config);
    if (!cast || !tessellate || canonical_dtype(cast->output_dtype) != "BF16" ||
        canonical_dtype(tessellate->frame_type) != "BF16" ||
        tessellate->slice_shape.empty() ||
        !value_has_dtype(plan, outer_input, "FP32", detail) ||
        !value_has_dtype(plan, intermediate, "BF16", detail) ||
        !value_has_dtype(plan, transformed_intermediate, "BF16", detail) ||
        !value_has_dtype(plan, outer_output, "BF16", detail)) {
      if (detail && detail->empty()) {
        *detail = "cast+tessellate does not satisfy the graph contract";
      }
      return false;
    }
    return true;
  }
  if (first.kind == OpKind::Detessellate && second.kind == OpKind::Cast) {
    const auto* detessellate = std::get_if<DetessellateOpConfig>(&first.config);
    const auto* cast = std::get_if<CastOpConfig>(&second.config);
    if (!detessellate || !cast || canonical_dtype(detessellate->frame_type) != "BF16" ||
        canonical_dtype(cast->output_dtype) != "FP32" ||
        detessellate->slice_shape.empty() || detessellate->frame_shape.empty() ||
        !value_has_dtype(plan, outer_input, "BF16", detail) ||
        !value_has_dtype(plan, intermediate, "BF16", detail) ||
        !value_has_dtype(plan, transformed_intermediate, "BF16", detail) ||
        !value_has_dtype(plan, outer_output, "FP32", detail)) {
      if (detail && detail->empty()) {
        *detail = "detessellate+cast does not satisfy the graph contract";
      }
      return false;
    }
    return true;
  }
  if (first.kind == OpKind::Detessellate && second.kind == OpKind::Dequantize) {
    const auto* detessellate = std::get_if<DetessellateOpConfig>(&first.config);
    const auto* dequantize = std::get_if<DequantizeOpConfig>(&second.config);
    if (!detessellate || !dequantize ||
        canonical_dtype(detessellate->frame_type) != "INT8" ||
        canonical_dtype(dequantize->input_dtype) != "INT8" ||
        detessellate->slice_shape.empty() || detessellate->frame_shape.empty() ||
        !positive_quantization(dequantize->channel_params) ||
        !value_has_dtype(plan, outer_input, "INT8", detail) ||
        !value_has_dtype(plan, intermediate, "INT8", detail) ||
        !value_has_dtype(plan, transformed_intermediate, "INT8", detail) ||
        !value_has_dtype(plan, outer_output, "FP32", detail)) {
      if (detail && detail->empty()) {
        *detail = "detessellate+dequantize does not satisfy the graph contract";
      }
      return false;
    }
    return true;
  }
  return record_error(detail, "generated fused semantic pattern has no typed validator");
}

struct Lane {
  std::uint32_t ordinal = 0;
  std::vector<OpId> operations;
};

struct ReducedMember {
  PhysicalCommandMember physical;
  SimaCvuCapabilityAbiRecord capability{};
  std::int64_t batch = 1;
  std::string processor;
  bool align_c16 = false;
  bool cblock = false;
};

using ProducerTable = std::vector<std::optional<OpId>>;
using ConsumerTable = std::vector<std::vector<OpId>>;

struct IngressFrontierValue {
  ValueId value_id = 0;
  OpId expected_consumer = 0;
};

std::optional<OpId> executable_producer(const ModelExecutionPlan& plan,
                                        const ProducerTable& producers, ValueId value_id,
                                        std::unordered_set<ValueId>* visiting = nullptr) {
  std::unordered_set<ValueId> local;
  if (!visiting) {
    visiting = &local;
  }
  if (value_id >= plan.values().size() || !visiting->emplace(value_id).second) {
    return std::nullopt;
  }
  if (producers[value_id]) {
    const auto& producer = plan.ops()[*producers[value_id]];
    if (!is_relation(producer)) {
      return producer.id;
    }
    for (const auto input : producer.inputs) {
      if (auto resolved = executable_producer(plan, producers, input, visiting)) {
        return resolved;
      }
    }
  }
  const auto* value = plan.value(value_id);
  if (value && value->read_expression) {
    return executable_producer(plan, producers, value->read_expression->source_value_id, visiting);
  }
  return std::nullopt;
}

std::vector<OpId> ingress_lane(const ModelExecutionPlan& plan, const ProducerTable& producers,
                               const ConsumerTable& consumers, ValueId value_id, OpId mla_id,
                               const std::unordered_set<ValueId>& public_values) {
  std::vector<OpId> reverse;
  OpId expected_consumer = mla_id;
  while (true) {
    const auto producer = executable_producer(plan, producers, value_id);
    if (!producer || !is_groupable_cvu(plan.ops()[*producer], plan)) {
      break;
    }
    const auto& op = plan.ops()[*producer];
    const auto output = op.outputs.front();
    if (public_values.contains(output) || consumers[output].size() != 1U ||
        consumers[output].front() != expected_consumer) {
      break;
    }
    reverse.push_back(op.id);
    expected_consumer = op.id;
    value_id = op.inputs.front();
  }
  std::reverse(reverse.begin(), reverse.end());
  return reverse;
}

std::optional<OpId> next_groupable_cvu_consumer(
    const ModelExecutionPlan& plan, const ConsumerTable& consumers,
    const std::unordered_set<ValueId>& public_values, ValueId value_id) {
  for (std::size_t traversed = 0U; traversed <= plan.ops().size(); ++traversed) {
    if (value_id >= consumers.size() || public_values.contains(value_id) ||
        consumers[value_id].size() != 1U) {
      return std::nullopt;
    }
    const auto candidate_id = consumers[value_id].front();
    const auto& candidate = plan.ops()[candidate_id];
    if (is_groupable_cvu(candidate, plan)) {
      return candidate_id;
    }
    if (!is_relation(candidate) || candidate.inputs.size() != 1U ||
        candidate.outputs.size() != 1U || candidate.inputs.front() != value_id) {
      return std::nullopt;
    }
    value_id = candidate.outputs.front();
  }
  return std::nullopt;
}

// Walk an MLA input backward through address-only relations.  A relation-only
// Pack is the exact ingress dual of egress Unpack: it fans one physical parent
// boundary into its compiler-authored child order so every direct producer is
// retained as one CVU member.  Materializing/ambiguous Pack and any branch or
// public observation remain hard barriers.
void expand_ingress_relation_frontier(
    const ModelExecutionPlan& plan, const ProducerTable& producers,
    const ConsumerTable& consumers, const ValueId value_id,
    const OpId expected_consumer,
    const std::unordered_set<ValueId>& public_values,
    std::unordered_set<ValueId>* visiting,
    std::vector<IngressFrontierValue>* frontier) {
  if (!visiting || !frontier || value_id >= producers.size() ||
      !visiting->emplace(value_id).second || public_values.contains(value_id) ||
      consumers[value_id].size() != 1U ||
      consumers[value_id].front() != expected_consumer || !producers[value_id]) {
    if (frontier && value_id < producers.size()) {
      frontier->push_back({value_id, expected_consumer});
    }
    return;
  }
  const auto producer_id = *producers[value_id];
  if (producer_id >= plan.ops().size()) {
    frontier->push_back({value_id, expected_consumer});
    return;
  }
  const auto& relation = plan.ops()[producer_id];
  if (!is_relation(relation) || relation.outputs.empty() ||
      std::find(relation.outputs.begin(), relation.outputs.end(), value_id) ==
          relation.outputs.end()) {
    frontier->push_back({value_id, expected_consumer});
    return;
  }

  if (relation.kind == OpKind::Pack) {
    const auto* pack = std::get_if<PackOpConfig>(&relation.config);
    const bool exact_components =
        pack && !pack->materializes && relation.outputs.size() == 1U &&
        relation.outputs.front() == value_id && !relation.inputs.empty() &&
        ((pack->spans.empty() && pack->components.size() == relation.inputs.size()) ||
         (!pack->spans.empty() &&
          std::all_of(relation.inputs.begin(), relation.inputs.end(), [&](const auto input) {
            return std::any_of(pack->spans.begin(), pack->spans.end(),
                               [&](const auto& span) { return span.value_id == input; });
          })));
    if (!exact_components) {
      frontier->push_back({value_id, expected_consumer});
      return;
    }
    for (const auto input : relation.inputs) {
      expand_ingress_relation_frontier(plan, producers, consumers, input, relation.id,
                                       public_values, visiting, frontier);
    }
    return;
  }

  // Only exact one-to-one view relations can be inverted without selecting an
  // arbitrary branch. Unpack fan-out and joins remain barriers on ingress.
  if (relation.inputs.size() != 1U || relation.outputs.size() != 1U ||
      relation.outputs.front() != value_id) {
    frontier->push_back({value_id, expected_consumer});
    return;
  }
  expand_ingress_relation_frontier(plan, producers, consumers,
                                   relation.inputs.front(), relation.id,
                                   public_values, visiting, frontier);
}

std::vector<OpId> egress_lane(const ModelExecutionPlan& plan, const ConsumerTable& consumers,
                              ValueId value_id,
                              const std::unordered_set<ValueId>& public_values) {
  std::vector<OpId> result;
  std::optional<OpId> previous;
  while (value_id < consumers.size()) {
    const auto consumer_id =
        next_groupable_cvu_consumer(plan, consumers, public_values, value_id);
    if (!consumer_id ||
        (previous &&
         !resolve_exact_private_ordered_relation_path(plan, *previous, *consumer_id))) {
      break;
    }
    const auto& consumer = plan.ops()[*consumer_id];
    if (!previous && consumer.inputs.front() != value_id) {
      break;
    }
    result.push_back(*consumer_id);
    previous = *consumer_id;
    value_id = consumer.outputs.front();
  }
  return result;
}

// Relation-only operations do not own bytes and therefore must not sever the
// physical lane proof.  In particular, stock AFE represents one packed MLA
// OFM followed by N independently materialized branches as
//
//   MLA -> Unpack(view x N) -> Detess[x N] -> Cast/Dequant[x N].
//
// The Unpack is an ordered address projection, not a kernel submission.  Walk
// through only the exact relation forms whose output order is compiler-authored
// and preserve that order as the physical member ordinal.  Branches, joins,
// public intermediates, and materializing Pack operations remain barriers.
void expand_egress_relation_frontier(const ModelExecutionPlan& plan,
                                     const ConsumerTable& consumers,
                                     const ValueId value_id,
                                     const std::unordered_set<ValueId>& public_values,
                                     std::unordered_set<ValueId>* visiting,
                                     std::vector<ValueId>* frontier) {
  if (!visiting || !frontier || value_id >= consumers.size() ||
      public_values.contains(value_id) || !visiting->emplace(value_id).second ||
      consumers[value_id].size() != 1U) {
    if (frontier && value_id < consumers.size()) {
      frontier->push_back(value_id);
    }
    return;
  }

  const auto consumer_id = consumers[value_id].front();
  if (consumer_id >= plan.ops().size()) {
    frontier->push_back(value_id);
    return;
  }
  const auto& relation = plan.ops()[consumer_id];
  if (!is_relation(relation) || relation.inputs.size() != 1U ||
      relation.inputs.front() != value_id || relation.outputs.empty()) {
    frontier->push_back(value_id);
    return;
  }

  // Unpack is the one admitted fan-out relation.  Slice, Reshape, and
  // PassThrough may be crossed only as exact one-to-one views.  A
  // relation-only Pack is a fan-in and cannot be traversed from one input.
  const bool ordered_unpack = relation.kind == OpKind::Unpack;
  const bool exact_one_to_one = relation.kind != OpKind::Pack &&
                                relation.outputs.size() == 1U;
  if (!ordered_unpack && !exact_one_to_one) {
    frontier->push_back(value_id);
    return;
  }

  for (const auto output : relation.outputs) {
    expand_egress_relation_frontier(plan, consumers, output, public_values, visiting,
                                    frontier);
  }
}

void implementation_flags(const ModelExecutionPlan& plan, const std::vector<OpId>& chain,
                          bool* align_c16, bool* cblock) {
  *align_c16 = false;
  *cblock = false;
  for (const auto op_id : chain) {
    const auto& op = plan.ops()[op_id];
    if (const auto* tessellate = std::get_if<TessellateOpConfig>(&op.config)) {
      *align_c16 = tessellate->align_c16;
      *cblock = tessellate->cblock;
    } else if (const auto* detessellate = std::get_if<DetessellateOpConfig>(&op.config)) {
      *align_c16 = detessellate->align_c16;
      *cblock = detessellate->cblock;
    }
  }
}

std::optional<std::vector<ReducedMember>>
reduce_lane(const ModelExecutionPlan& plan, const Lane& lane, std::string* error) {
  std::vector<ReducedMember> result;
  std::vector<std::optional<SimaCvuCapabilityAbiRecord>> pair_matches(
      lane.operations.size() > 1U ? lane.operations.size() - 1U : 0U);
  for (std::size_t index = 0; index < pair_matches.size(); ++index) {
    const auto& first = plan.ops()[lane.operations[index]];
    const auto& second = plan.ops()[lane.operations[index + 1U]];
    const bool exact_private_edge =
        resolve_exact_private_ordered_relation_path(plan, first.id, second.id);
    if (exact_private_edge) {
      pair_matches[index] =
          cvu_capability(std::vector<OpId>{first.id, second.id}, plan);
    }
  }
  for (std::size_t index = 1; index < pair_matches.size(); ++index) {
    if (pair_matches[index - 1U] && pair_matches[index] &&
        (pair_matches[index - 1U]->mandatory_when_matched != 0U ||
         pair_matches[index]->mandatory_when_matched != 0U)) {
      record_error(error, "overlapping mandatory CVU patterns at semantic operation '" +
                              plan.ops()[lane.operations[index]].name +
                              "' require a registered longer implementation");
      return std::nullopt;
    }
  }
  for (std::size_t index = 0; index < lane.operations.size();) {
    std::vector<OpId> chain{lane.operations[index]};
    std::optional<SimaCvuCapabilityAbiRecord> capability;
    if (index < pair_matches.size() && pair_matches[index]) {
      const auto& first = plan.ops()[lane.operations[index]];
      const auto& second = plan.ops()[lane.operations[index + 1U]];
      std::vector<OpId> candidate{first.id, second.id};
      std::string detail;
      if (!validate_fused_member(plan, candidate, &detail)) {
        if (pair_matches[index]->mandatory_when_matched != 0U) {
          record_error(error, "mandatory fused CVU chain '" + first.name + " -> " +
                                  second.name + "' is invalid: " + detail);
          return std::nullopt;
        }
      } else {
        chain = std::move(candidate);
        capability = *pair_matches[index];
      }
    }
    if (!capability) {
      capability = cvu_capability(chain, plan);
    }
    if (!capability) {
      record_error(error, "physical lowering has no registered CVU implementation for '" +
                              plan.ops()[chain.front()].name + "'");
      return std::nullopt;
    }

    const auto& first = plan.ops()[chain.front()];
    const auto& last = plan.ops()[chain.back()];
    ReducedMember member;
    member.physical.ordinal = lane.ordinal;
    member.physical.semantic_chain = chain;
    member.physical.outer_inputs = first.inputs;
    member.physical.outer_outputs = last.outputs;
    member.capability = *capability;
    member.batch = batch_for(first);
    if (member.batch <= 0 ||
        static_cast<std::uint64_t>(member.batch) >
            std::numeric_limits<std::uint32_t>::max()) {
      record_error(error, "physical lowering found an invalid CVU batch contract");
      return std::nullopt;
    }
    member.processor = first.processor;
    implementation_flags(plan, chain, &member.align_c16, &member.cblock);
    result.push_back(std::move(member));
    index += chain.size();
  }
  return result;
}

using CohortKey =
    std::tuple<std::size_t, std::uint32_t, std::int64_t, std::string, bool, bool>;

bool add_aligned_cohorts(const ModelExecutionPlan& plan, std::vector<Lane> lanes,
                         const bool align_from_boundary_end,
                         const PhysicalCommandRole role,
                         std::unordered_set<OpId>* claimed,
                         std::vector<DraftCommand>* drafts,
                         PhysicalCohortId* next_cohort_id, std::string* error) {
  std::map<CohortKey, std::vector<ReducedMember>> cohorts;
  for (const auto& lane : lanes) {
    auto reduced = reduce_lane(plan, lane, error);
    if (!reduced) {
      return false;
    }
    for (std::size_t index = 0; index < reduced->size(); ++index) {
      auto& member = (*reduced)[index];
      const auto boundary_distance =
          align_from_boundary_end ? reduced->size() - index - 1U : index;
      cohorts[{boundary_distance, member.capability.graph_id, member.batch,
               member.processor, member.align_c16, member.cblock}]
          .push_back(std::move(member));
    }
  }

  for (auto& [key, cohort] : cohorts) {
    (void)key;
    std::sort(cohort.begin(), cohort.end(), [](const auto& left, const auto& right) {
      return left.physical.ordinal < right.physical.ordinal;
    });
    const auto& capability = cohort.front().capability;
    const auto cohort_id = (*next_cohort_id)++;
    for (std::size_t begin = 0; begin < cohort.size(); begin += capability.maximum_members) {
      const auto end = std::min(cohort.size(),
                                begin + static_cast<std::size_t>(capability.maximum_members));
      DraftCommand draft;
      draft.cohort_id = cohort_id;
      draft.engine = PhysicalEngine::Cvu;
      draft.role = role;
      draft.graph_id = capability.graph_id;
      draft.batch_size = static_cast<std::uint32_t>(cohort.front().batch);
      draft.implementation_id = cvu_implementation_id(capability);
      draft.maximum_members = capability.maximum_members;
      for (std::size_t index = begin; index < end; ++index) {
        auto& member = cohort[index].physical;
        const auto already_claimed = std::count_if(
            member.semantic_chain.begin(), member.semantic_chain.end(),
            [&](const auto op_id) { return claimed->contains(op_id); });
        if (already_claimed == static_cast<std::ptrdiff_t>(member.semantic_chain.size())) {
          continue;
        }
        if (already_claimed != 0) {
          return record_error(error, "physical lowering found partially overlapping fused chains");
        }
        for (const auto op_id : member.semantic_chain) {
          claimed->emplace(op_id);
          if (draft.members.empty() && op_id == member.semantic_chain.front()) {
            draft.stable_order = plan.ops()[op_id].sequence;
          } else {
            draft.stable_order = std::min(draft.stable_order, plan.ops()[op_id].sequence);
          }
        }
        draft.members.push_back(std::move(member));
      }
      if (!draft.members.empty()) {
        drafts->push_back(std::move(draft));
      }
    }
  }
  return true;
}

} // namespace

bool is_address_relation_op(const OpSpec& op) noexcept {
  return is_relation(op);
}

bool resolve_exact_private_ordered_relation_path(const ModelExecutionPlan& plan,
                                                 const OpId first_id,
                                                 const OpId second_id,
                                                 std::vector<ValueId>* internal_values) {
  if (internal_values) {
    internal_values->clear();
  }
  if (first_id >= plan.ops().size() || second_id >= plan.ops().size() ||
      first_id == second_id) {
    return false;
  }
  const auto& first = plan.ops()[first_id];
  const auto& second = plan.ops()[second_id];
  if (first.outputs.size() != 1U || first.sequence >= second.sequence) {
    return false;
  }
  const auto root_id = first.outputs.front();
  const auto* root = plan.value(root_id);
  if (!root || root->read_expression || !root->storage_binding) {
    return false;
  }

  const auto is_public = [&](const ValueId value_id) {
    return std::any_of(plan.model_outputs().begin(), plan.model_outputs().end(),
                       [&](const auto& output) { return output.value_id == value_id; });
  };
  const auto unique_consumer = [&](const ValueId value_id) -> std::optional<OpId> {
    std::optional<OpId> result;
    for (const auto& op : plan.ops()) {
      for (const auto input : op.inputs) {
        if (input != value_id) {
          continue;
        }
        if (result.has_value()) {
          return std::nullopt;
        }
        result = op.id;
      }
    }
    return result;
  };

  std::vector<ValueId> path{root_id};
  ValueId cursor = root_id;
  for (std::size_t traversed = 0U; traversed <= plan.ops().size(); ++traversed) {
    if (is_public(cursor)) {
      return false;
    }
    const auto consumer_id = unique_consumer(cursor);
    if (!consumer_id) {
      return false;
    }
    if (*consumer_id == second_id) {
      if (std::count(second.inputs.begin(), second.inputs.end(), cursor) != 1) {
        return false;
      }
      if (internal_values) {
        *internal_values = std::move(path);
      }
      return true;
    }
    if (*consumer_id >= plan.ops().size()) {
      return false;
    }
    const auto& relation = plan.ops()[*consumer_id];
    if (!is_relation(relation) || relation.inputs.size() != 1U ||
        relation.outputs.size() != 1U || relation.inputs.front() != cursor ||
        !relation.dependencies.empty() || relation.sequence <= first.sequence ||
        relation.sequence >= second.sequence) {
      return false;
    }
    const auto next_id = relation.outputs.front();
    const auto* previous = plan.value(cursor);
    const auto* next = plan.value(next_id);
    if (!previous || !next || !is_dense_order_preserving_view(*root, *previous, *next)) {
      return false;
    }
    path.push_back(next_id);
    cursor = next_id;
  }
  return false;
}

std::optional<PhysicalExecutionPlan>
PhysicalExecutionLowerer::lower(const ModelExecutionPlan& semantic, std::string* error) {
  const auto& ops = semantic.ops();
  ProducerTable producers(semantic.values().size());
  ConsumerTable consumers(semantic.values().size());
  for (const auto& op : ops) {
    for (const auto output : op.outputs) {
      if (output >= producers.size() || producers[output]) {
        record_error(error, "physical lowering found a missing or duplicate semantic producer");
        return std::nullopt;
      }
      producers[output] = op.id;
    }
    for (const auto input : op.inputs) {
      if (input >= consumers.size()) {
        record_error(error, "physical lowering found an out-of-range semantic input");
        return std::nullopt;
      }
      consumers[input].push_back(op.id);
    }
  }
  std::unordered_set<ValueId> public_values;
  for (const auto& output : semantic.model_outputs()) {
    public_values.emplace(output.value_id);
  }

  std::unordered_set<OpId> claimed;
  std::vector<DraftCommand> drafts;
  PhysicalCohortId next_cohort_id = 0U;
  const auto mla_count = static_cast<std::size_t>(
      std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return op.kind == OpKind::Mla;
      }));
  std::size_t mla_ordinal = 0U;
  for (const auto& mla : ops) {
    if (mla.kind != OpKind::Mla) {
      continue;
    }
    const auto ingress_role = mla_ordinal == 0U ? PhysicalCommandRole::Ingress
                                                : PhysicalCommandRole::Interstitial;
    const auto egress_role = mla_ordinal + 1U == mla_count
                                 ? PhysicalCommandRole::Egress
                                 : PhysicalCommandRole::Interstitial;
    std::vector<Lane> ingress;
    std::vector<IngressFrontierValue> ingress_frontier;
    for (const auto input : mla.inputs) {
      std::unordered_set<ValueId> visiting;
      expand_ingress_relation_frontier(semantic, producers, consumers, input, mla.id,
                                       public_values, &visiting, &ingress_frontier);
    }
    ingress.reserve(ingress_frontier.size());
    for (std::size_t ordinal = 0; ordinal < ingress_frontier.size(); ++ordinal) {
      ingress.push_back(
          {static_cast<std::uint32_t>(ordinal),
           ingress_lane(semantic, producers, consumers,
                        ingress_frontier[ordinal].value_id,
                        ingress_frontier[ordinal].expected_consumer,
                        public_values)});
    }
    if (!add_aligned_cohorts(semantic, std::move(ingress), true, ingress_role,
                             &claimed, &drafts, &next_cohort_id, error)) {
      return std::nullopt;
    }

    std::vector<ValueId> egress_frontier;
    for (const auto output : mla.outputs) {
      std::unordered_set<ValueId> visiting;
      expand_egress_relation_frontier(semantic, consumers, output, public_values, &visiting,
                                      &egress_frontier);
    }
    std::vector<Lane> egress;
    egress.reserve(egress_frontier.size());
    for (std::size_t ordinal = 0; ordinal < egress_frontier.size(); ++ordinal) {
      egress.push_back(
          {static_cast<std::uint32_t>(ordinal),
           egress_lane(semantic, consumers, egress_frontier[ordinal], public_values)});
    }
    if (!add_aligned_cohorts(semantic, std::move(egress), false, egress_role,
                             &claimed, &drafts, &next_cohort_id, error)) {
      return std::nullopt;
    }
    ++mla_ordinal;
  }

  // Registered fusion is a graph-wide rule, not an MLA-name heuristic.  The
  // boundary walks above provide compiler-authored member ordinals and broad
  // horizontal grouping.  Fuse any remaining exact pair as a singleton so an
  // eligible A65/interstitial boundary can never fall through to two legacy
  // standalone commands.
  std::vector<std::pair<OpId, OpId>> residual_pairs;
  std::unordered_set<OpId> residual_pair_ops;
  for (const auto& first : ops) {
    if (claimed.contains(first.id) || !is_groupable_cvu(first, semantic) ||
        first.outputs.size() != 1U || public_values.contains(first.outputs.front())) {
      continue;
    }
    const auto second_id = next_groupable_cvu_consumer(
        semantic, consumers, public_values, first.outputs.front());
    if (!second_id || *second_id >= ops.size() || claimed.contains(*second_id) ||
        !resolve_exact_private_ordered_relation_path(semantic, first.id, *second_id) ||
        !cvu_capability(std::vector<OpId>{first.id, *second_id}, semantic)) {
      continue;
    }
    if (!residual_pair_ops.emplace(first.id).second ||
        !residual_pair_ops.emplace(*second_id).second) {
      record_error(error, "overlapping mandatory CVU patterns require a registered longer implementation");
      return std::nullopt;
    }
    residual_pairs.emplace_back(first.id, *second_id);
  }
  for (const auto& [first, second] : residual_pairs) {
    if (!add_aligned_cohorts(semantic, {{0U, {first, second}}}, false,
                             PhysicalCommandRole::Interstitial, &claimed, &drafts,
                             &next_cohort_id, error)) {
      return std::nullopt;
    }
  }

  // Every executable operation not owned by a proved aligned lane remains an
  // exact singleton. Safety is preferred over speculative grouping at branch,
  // join, public-intermediate, materialization, or opaque backend barriers.
  for (const auto& op : ops) {
    if (is_relation(op) || claimed.contains(op.id)) {
      continue;
    }
    const auto identity = implementation_id(op, semantic);
    if (identity.empty()) {
      record_error(error, "physical lowering has no implementation for semantic operation '" + op.name +
                      "'");
      return std::nullopt;
    }
    DraftCommand draft;
    draft.cohort_id = next_cohort_id++;
    draft.stable_order = op.sequence;
    draft.engine = engine_for(op);
    draft.role = draft.engine == PhysicalEngine::Cvu
                     ? PhysicalCommandRole::Interstitial
                     : PhysicalCommandRole::NonCvu;
    draft.implementation_id = identity;
    const auto batch = batch_for(op);
    if (batch <= 0 || static_cast<std::uint64_t>(batch) >
                          std::numeric_limits<std::uint32_t>::max()) {
      record_error(error, "physical lowering found an invalid command batch contract");
      return std::nullopt;
    }
    draft.batch_size = static_cast<std::uint32_t>(batch);
    if (const auto capability = cvu_capability(std::vector<OpId>{op.id}, semantic)) {
      draft.graph_id = capability->graph_id;
      draft.maximum_members = capability->maximum_members;
    }
    draft.members = {{0U, {op.id}, op.inputs, op.outputs}};
    claimed.emplace(op.id);
    drafts.push_back(std::move(draft));
  }

  std::vector<std::size_t> draft_for_op(ops.size(), drafts.size());
  for (std::size_t draft_id = 0; draft_id < drafts.size(); ++draft_id) {
    for (const auto& member : drafts[draft_id].members) {
      for (const auto op_id : member.semantic_chain) {
        if (op_id >= draft_for_op.size() || draft_for_op[op_id] != drafts.size()) {
          record_error(error, "physical lowering assigned a semantic operation more than once");
          return std::nullopt;
        }
        draft_for_op[op_id] = draft_id;
      }
    }
  }

  std::function<void(ValueId, std::set<std::size_t>*, std::unordered_set<ValueId>*)>
      collect_predecessors;
  collect_predecessors = [&](const ValueId value_id, std::set<std::size_t>* result,
                             std::unordered_set<ValueId>* visiting) {
    if (value_id >= producers.size() || !visiting->emplace(value_id).second) {
      return;
    }
    if (producers[value_id]) {
      const auto producer_id = *producers[value_id];
      if (draft_for_op[producer_id] != drafts.size()) {
        result->emplace(draft_for_op[producer_id]);
        return;
      }
      for (const auto input : ops[producer_id].inputs) {
        collect_predecessors(input, result, visiting);
      }
      return;
    }
    const auto* value = semantic.value(value_id);
    if (value && value->read_expression) {
      collect_predecessors(value->read_expression->source_value_id, result, visiting);
    }
  };

  for (std::size_t draft_id = 0; draft_id < drafts.size(); ++draft_id) {
    auto& draft = drafts[draft_id];
    for (const auto& member : draft.members) {
      for (const auto input : member.outer_inputs) {
        std::unordered_set<ValueId> visiting;
        collect_predecessors(input, &draft.predecessor_drafts, &visiting);
      }
      // Preserve compiler-authored control dependencies in addition to value
      // provenance. A dependency on a relation-only operation resolves through
      // that relation's inputs to the nearest executable producer.
      for (const auto op_id : member.semantic_chain) {
        for (const auto dependency : ops[op_id].dependencies) {
          if (dependency >= ops.size()) {
            record_error(error, "physical lowering found an out-of-range semantic dependency");
            return std::nullopt;
          }
          if (draft_for_op[dependency] != drafts.size()) {
            draft.predecessor_drafts.emplace(draft_for_op[dependency]);
            continue;
          }
          for (const auto input : ops[dependency].inputs) {
            std::unordered_set<ValueId> visiting;
            collect_predecessors(input, &draft.predecessor_drafts, &visiting);
          }
        }
      }
    }
    draft.predecessor_drafts.erase(draft_id);
  }

  std::vector<std::vector<std::size_t>> successors(drafts.size());
  std::vector<std::size_t> indegree(drafts.size(), 0U);
  for (std::size_t id = 0; id < drafts.size(); ++id) {
    indegree[id] = drafts[id].predecessor_drafts.size();
    for (const auto predecessor : drafts[id].predecessor_drafts) {
      successors[predecessor].push_back(id);
    }
  }
  using ReadyKey = std::tuple<std::uint64_t, std::string, std::size_t>;
  std::priority_queue<ReadyKey, std::vector<ReadyKey>, std::greater<>> ready;
  for (std::size_t id = 0; id < drafts.size(); ++id) {
    if (indegree[id] == 0U) {
      ready.emplace(drafts[id].stable_order, drafts[id].implementation_id, id);
    }
  }
  std::vector<std::size_t> topological;
  while (!ready.empty()) {
    const auto id = std::get<2>(ready.top());
    ready.pop();
    topological.push_back(id);
    for (const auto successor : successors[id]) {
      if (--indegree[successor] == 0U) {
        ready.emplace(drafts[successor].stable_order, drafts[successor].implementation_id,
                      successor);
      }
    }
  }
  if (topological.size() != drafts.size()) {
    record_error(error, "physical lowering reconstructed a dependency cycle");
    return std::nullopt;
  }

  PhysicalExecutionPlan result;
  result.command_for_semantic_op.resize(ops.size());
  result.commands.resize(drafts.size());
  std::vector<PhysicalCommandId> command_id_for_draft(drafts.size());
  for (std::size_t index = 0; index < topological.size(); ++index) {
    command_id_for_draft[topological[index]] = static_cast<PhysicalCommandId>(index);
  }
  std::ostringstream digest;
  for (std::size_t index = 0; index < topological.size(); ++index) {
    const auto draft_id = topological[index];
    const auto& draft = drafts[draft_id];
    auto& command = result.commands[index];
    command.id = static_cast<PhysicalCommandId>(index);
    command.cohort_id = draft.cohort_id;
    command.topological_rank = index;
    command.engine = draft.engine;
    command.role = draft.role;
    command.implementation_id = draft.implementation_id;
    command.graph_id = draft.graph_id;
    command.batch_size = draft.batch_size;
    command.maximum_members = draft.maximum_members;
    command.members = draft.members;
    for (const auto& member : command.members) {
      command.inputs.insert(command.inputs.end(), member.outer_inputs.begin(),
                            member.outer_inputs.end());
      command.outputs.insert(command.outputs.end(), member.outer_outputs.begin(),
                             member.outer_outputs.end());
      for (const auto op_id : member.semantic_chain) {
        result.command_for_semantic_op[op_id] = command.id;
      }
    }
    for (const auto predecessor : draft.predecessor_drafts) {
      command.predecessors.push_back(command_id_for_draft[predecessor]);
    }
    std::sort(command.predecessors.begin(), command.predecessors.end());
    for (const auto successor : successors[draft_id]) {
      command.successors.push_back(command_id_for_draft[successor]);
    }
    std::sort(command.successors.begin(), command.successors.end());
    digest << command.id << '@' << command.cohort_id << ':'
           << static_cast<unsigned int>(command.role) << ':' << command.graph_id << ':'
           << command.batch_size << ':'
           << command.implementation_id << ':'
           << command.maximum_members << ':';
    for (const auto& member : command.members) {
      digest << '[' << member.ordinal << ':';
      for (const auto origin : member.semantic_chain) {
        digest << origin << ',';
      }
      digest << '|';
      for (const auto input : member.outer_inputs) {
        digest << input << ',';
      }
      digest << "->";
      for (const auto output : member.outer_outputs) {
        digest << output << ',';
      }
      digest << ']';
    }
    digest << '<';
    for (const auto predecessor : command.predecessors) {
      digest << predecessor << ',';
    }
    digest << ">;";
  }
  result.deterministic_digest_material = digest.str();
  if (error) {
    error->clear();
  }
  return result;
}

std::optional<std::uint32_t>
minimum_cvu_member_capacity(const PhysicalExecutionPlan& plan) noexcept {
  std::optional<std::uint32_t> result;
  for (const auto& command : plan.commands) {
    if (command.engine != PhysicalEngine::Cvu || command.maximum_members == 0U) {
      continue;
    }
    result = result ? std::min(*result, command.maximum_members)
                    : command.maximum_members;
  }
  return result;
}

PhysicalExecutionTracker::PhysicalExecutionTracker(const PhysicalExecutionPlan* plan)
    : plan_(plan), states_(plan ? plan->commands.size() : 0U,
                          PhysicalCommandState::Pending) {}

std::optional<PhysicalExecutionTracker>
PhysicalExecutionTracker::create(const PhysicalExecutionPlan& plan, std::string* error) {
  if (plan.commands.empty()) {
    record_error(error, "physical execution tracker requires at least one command");
    return std::nullopt;
  }
  using CvuCohortContract =
      std::tuple<PhysicalCommandRole, std::uint32_t, std::uint32_t, std::string,
                 std::uint32_t>;
  std::unordered_map<PhysicalCohortId, CvuCohortContract> cvu_cohorts;
  for (std::size_t index = 0; index < plan.commands.size(); ++index) {
    const auto& command = plan.commands[index];
    if (command.id != index || command.topological_rank != index) {
      record_error(error, "physical execution tracker requires dense topological command ids");
      return std::nullopt;
    }
    SimaCvuCapabilityAbiRecord capability{};
    const bool valid_cvu_identity =
        command.engine == PhysicalEngine::Cvu && command.graph_id != 0U &&
        sima_cvu_capability_abi_lookup(command.graph_id, &capability) &&
        command.maximum_members == capability.maximum_members &&
        command.implementation_id == cvu_implementation_id(capability);
    if (command.batch_size == 0U || command.members.empty() ||
        (command.engine == PhysicalEngine::Cvu &&
         (command.role == PhysicalCommandRole::NonCvu || !valid_cvu_identity ||
          command.members.size() > command.maximum_members)) ||
        (command.engine != PhysicalEngine::Cvu &&
         (command.role != PhysicalCommandRole::NonCvu || command.graph_id != 0U ||
          command.maximum_members != 0U))) {
      record_error(error, "physical execution tracker found invalid command membership");
      return std::nullopt;
    }
    if (command.engine == PhysicalEngine::Cvu) {
      const CvuCohortContract contract{command.role, command.graph_id, command.batch_size,
                                       command.implementation_id,
                                       command.maximum_members};
      const auto [found, inserted] = cvu_cohorts.emplace(command.cohort_id, contract);
      if (!inserted && found->second != contract) {
        record_error(error, "physical execution tracker found an inconsistent CVU cohort");
        return std::nullopt;
      }
    }
    std::vector<ValueId> flattened_inputs;
    std::vector<ValueId> flattened_outputs;
    std::unordered_set<OpId> semantic_origins;
    std::optional<std::uint32_t> previous_ordinal;
    for (const auto& member : command.members) {
      if (member.semantic_chain.empty() ||
          (previous_ordinal && member.ordinal <= *previous_ordinal) ||
          (command.engine == PhysicalEngine::Cvu &&
           (member.outer_inputs.size() != 1U || member.outer_outputs.size() != 1U ||
            member.semantic_chain.size() != capability.semantic_pattern_length))) {
        record_error(error, "physical execution tracker found an invalid physical member");
        return std::nullopt;
      }
      previous_ordinal = member.ordinal;
      for (const auto op_id : member.semantic_chain) {
        if (!semantic_origins.emplace(op_id).second) {
          record_error(error, "physical execution tracker found duplicate semantic provenance");
          return std::nullopt;
        }
      }
      flattened_inputs.insert(flattened_inputs.end(), member.outer_inputs.begin(),
                              member.outer_inputs.end());
      flattened_outputs.insert(flattened_outputs.end(), member.outer_outputs.begin(),
                               member.outer_outputs.end());
    }
    if (flattened_inputs != command.inputs || flattened_outputs != command.outputs) {
      record_error(error, "physical execution tracker found inconsistent outer bindings");
      return std::nullopt;
    }
    for (const auto predecessor : command.predecessors) {
      if (predecessor >= index ||
          std::find(plan.commands[predecessor].successors.begin(),
                    plan.commands[predecessor].successors.end(), command.id) ==
              plan.commands[predecessor].successors.end()) {
        record_error(error, "physical execution tracker found a non-topological predecessor");
        return std::nullopt;
      }
    }
    for (const auto successor : command.successors) {
      if (successor <= index || successor >= plan.commands.size() ||
          std::find(plan.commands[successor].predecessors.begin(),
                    plan.commands[successor].predecessors.end(), command.id) ==
              plan.commands[successor].predecessors.end()) {
        record_error(error, "physical execution tracker found an invalid successor");
        return std::nullopt;
      }
    }
    if (!std::is_sorted(command.predecessors.begin(), command.predecessors.end()) ||
        !std::is_sorted(command.successors.begin(), command.successors.end()) ||
        std::adjacent_find(command.predecessors.begin(), command.predecessors.end()) !=
            command.predecessors.end() ||
        std::adjacent_find(command.successors.begin(), command.successors.end()) !=
            command.successors.end()) {
      record_error(error, "physical execution tracker found a duplicate dependency");
      return std::nullopt;
    }
  }
  if (error) error->clear();
  return PhysicalExecutionTracker(&plan);
}

std::optional<PhysicalCommandId> PhysicalExecutionTracker::next_ready() const noexcept {
  if (!plan_) return std::nullopt;
  for (const auto& command : plan_->commands) {
    if (ready(command.id)) return command.id;
  }
  return std::nullopt;
}

bool PhysicalExecutionTracker::ready(const PhysicalCommandId id) const noexcept {
  if (!plan_ || id >= states_.size() ||
      states_[id] != PhysicalCommandState::Pending) {
    return false;
  }
  const auto& command = plan_->commands[id];
  return std::all_of(command.predecessors.begin(), command.predecessors.end(),
                     [&](const PhysicalCommandId predecessor) {
                       return states_[predecessor] == PhysicalCommandState::Completed;
                     });
}

bool PhysicalExecutionTracker::claim(const PhysicalCommandId id,
                                     std::string* error) noexcept {
  if (!ready(id)) {
    return record_error(error, "physical command is not ready");
  }
  states_[id] = PhysicalCommandState::Submitted;
  if (error) error->clear();
  return true;
}

bool PhysicalExecutionTracker::complete(const PhysicalCommandId id,
                                        std::string* error) noexcept {
  if (id >= states_.size() || states_[id] != PhysicalCommandState::Submitted) {
    return record_error(error, "physical completion does not name a submitted command");
  }
  states_[id] = PhysicalCommandState::Completed;
  if (error) error->clear();
  return true;
}

bool PhysicalExecutionTracker::fail(const PhysicalCommandId id,
                                    std::string* error) noexcept {
  if (!plan_ || id >= states_.size() || states_[id] != PhysicalCommandState::Submitted) {
    return record_error(error,
                                 "physical failure does not name a submitted command");
  }
  states_[id] = PhysicalCommandState::Failed;
  // A frame is atomic at publication. Once any command fails, starting more
  // work cannot restore success and only wastes scarce device slots. Block all
  // not-yet-submitted commands, while already-submitted independent commands
  // remain owned and must be reaped to a terminal state.
  for (auto& state : states_) {
    if (state == PhysicalCommandState::Pending) {
      state = PhysicalCommandState::Blocked;
    }
  }
  if (error) error->clear();
  return true;
}

PhysicalCommandState
PhysicalExecutionTracker::state(const PhysicalCommandId id) const noexcept {
  return id < states_.size() ? states_[id] : PhysicalCommandState::Blocked;
}

bool PhysicalExecutionTracker::succeeded() const noexcept {
  return !states_.empty() &&
         std::all_of(states_.begin(), states_.end(), [](const auto state) {
           return state == PhysicalCommandState::Completed;
         });
}

bool PhysicalExecutionTracker::terminal() const noexcept {
  return std::all_of(states_.begin(), states_.end(), [](const auto state) {
    return state == PhysicalCommandState::Completed || state == PhysicalCommandState::Failed ||
           state == PhysicalCommandState::Blocked;
  });
}

void PhysicalExecutionTracker::reset() noexcept {
  std::fill(states_.begin(), states_.end(), PhysicalCommandState::Pending);
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

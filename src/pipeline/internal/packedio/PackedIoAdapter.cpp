#include "pipeline/internal/packedio/PackedIoAdapter.h"

#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::packedio {
namespace {

using pipeline_internal::sima::PhysicalBufferStaticSpec;

std::uint64_t span_end_bytes(std::int64_t offset, std::uint64_t size) {
  if (offset < 0) {
    return 0U;
  }
  if (static_cast<std::uint64_t>(offset) >
      std::numeric_limits<std::uint64_t>::max() - size) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(offset) + size;
}

const PhysicalBufferStaticSpec* find_physical(const std::vector<PhysicalBufferStaticSpec>& physicals,
                                              int physical_index) {
  auto it = std::find_if(physicals.begin(), physicals.end(),
                         [&](const PhysicalBufferStaticSpec& physical) {
                           return physical.physical_index == physical_index;
                         });
  return it == physicals.end() ? nullptr : &(*it);
}

std::uint64_t physical_span_bytes_for_tensor(const std::vector<std::int64_t>& shape,
                                             const std::vector<std::int64_t>& stride_bytes,
                                             const std::uint64_t logical_size_bytes) {
  if (shape.empty() || stride_bytes.empty()) {
    return logical_size_bytes;
  }
  const auto normalized_stride_bytes =
      pipeline_internal::normalize_strides_rank_to_shape(stride_bytes, {}, shape, true);
  const auto* strides = &normalized_stride_bytes;
  if (strides->size() != shape.size()) {
    return logical_size_bytes;
  }

  std::uint64_t max_offset = 0U;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || (*strides)[i] < 0) {
      return logical_size_bytes;
    }
    if (shape[i] == 1) {
      continue;
    }
    const auto dim = static_cast<std::uint64_t>(shape[i] - 1);
    const auto stride = static_cast<std::uint64_t>((*strides)[i]);
    const std::uint64_t delta = dim * stride;
    if (dim > 0U && delta / dim != stride) {
      return logical_size_bytes;
    }
    if (max_offset > (std::numeric_limits<std::uint64_t>::max() - delta)) {
      return logical_size_bytes;
    }
    max_offset += delta;
  }
  const std::uint64_t elem_bytes =
      strides->empty() ? 1U : static_cast<std::uint64_t>(strides->back());
  if (elem_bytes == 0U ||
      max_offset > (std::numeric_limits<std::uint64_t>::max() - elem_bytes)) {
    return logical_size_bytes;
  }
  return max_offset + elem_bytes;
}

} // namespace

bool validate_packed_contract(const CompiledRuntimeContract& contract, std::string* err) {
  auto fail = [&](const std::string& reason) {
    if (err) {
      *err = reason;
    }
    return false;
  };

  for (const auto& logical : contract.logical_inputs) {
    const auto* physical = find_physical(contract.physical_inputs, logical.physical_index);
    if (!physical) {
      std::ostringstream oss;
      oss << "packed input contract missing physical input index=" << logical.physical_index;
      return fail(oss.str());
    }
    const std::uint64_t physical_span_bytes =
        physical_span_bytes_for_tensor(logical.shape, logical.stride_bytes, logical.size_bytes);
    if (span_end_bytes(logical.byte_offset, physical_span_bytes) > physical->size_bytes) {
      std::ostringstream oss;
      oss << "packed input logical span exceeds physical input index=" << logical.physical_index;
      return fail(oss.str());
    }
  }

  for (const auto& logical : contract.logical_outputs) {
    const auto* physical = find_physical(contract.physical_outputs, logical.physical_index);
    if (!physical) {
      std::ostringstream oss;
      oss << "packed output contract missing physical output index=" << logical.physical_index;
      return fail(oss.str());
    }
    const std::uint64_t physical_span_bytes =
        physical_span_bytes_for_tensor(logical.shape, logical.stride_bytes, logical.size_bytes);
    if (span_end_bytes(logical.byte_offset, physical_span_bytes) > physical->size_bytes) {
      std::ostringstream oss;
      oss << "packed output logical span exceeds physical output index=" << logical.physical_index;
      return fail(oss.str());
    }
  }

  return true;
}

bool normalize_shared_parent_input_views(CompiledRuntimeContract* contract, std::string* err) {
  auto fail = [&](const std::string& reason) {
    if (err) {
      *err = reason;
    }
    return false;
  };
  if (err) {
    err->clear();
  }
  if (!contract) {
    return fail("shared parent input normalization requires a contract");
  }
  if (contract->logical_inputs.size() <= 1U || contract->physical_inputs.size() <= 1U) {
    return true;
  }

  struct Member {
    std::size_t logical_pos = 0U;
    std::size_t binding_pos = std::numeric_limits<std::size_t>::max();
    int parent_source_physical_index = -1;
    std::uint64_t absolute_byte_offset = 0U;
    std::uint64_t physical_span_bytes = 0U;
    const PhysicalBufferStaticSpec* physical = nullptr;
  };
  struct Group {
    int parent_source_physical_index = -1;
    std::string segment_name;
    pipeline_internal::sima::DeviceKind device_kind =
        pipeline_internal::sima::DeviceKind::Unknown;
    std::uint64_t size_bytes = 0U;
    std::uint64_t base_byte_offset = 0U;
    bool have_base_byte_offset = false;
    std::vector<std::size_t> members;
  };

  std::vector<Member> members;
  members.reserve(contract->logical_inputs.size());
  std::vector<Group> groups;
  groups.reserve(contract->physical_inputs.size());
  std::vector<int> group_for_logical(contract->logical_inputs.size(), -1);
  bool saw_parent_view = false;

  const auto binding_index_for_logical = [&](int logical_index,
                                             std::size_t fallback) -> std::size_t {
    const auto by_logical = std::find_if(
        contract->input_bindings.begin(), contract->input_bindings.end(),
        [&](const pipeline_internal::sima::InputBindingStaticSpec& binding) {
          return binding.local_logical_input_index == logical_index;
        });
    if (by_logical != contract->input_bindings.end()) {
      return static_cast<std::size_t>(
          std::distance(contract->input_bindings.begin(), by_logical));
    }
    return fallback < contract->input_bindings.size()
               ? fallback
               : std::numeric_limits<std::size_t>::max();
  };

  for (std::size_t i = 0; i < contract->logical_inputs.size(); ++i) {
    const auto& logical = contract->logical_inputs[i];
    const auto* physical = find_physical(contract->physical_inputs, logical.physical_index);
    if (!physical) {
      return true;
    }
    const std::size_t binding_pos = binding_index_for_logical(logical.logical_index, i);
    const auto* binding =
        binding_pos < contract->input_bindings.size() ? &contract->input_bindings[binding_pos]
                                                      : nullptr;

    int parent_source_physical_index = -1;
    if (physical->source_physical_index >= 0) {
      parent_source_physical_index = physical->source_physical_index;
    } else if (binding && binding->src_physical_output_index >= 0) {
      parent_source_physical_index = binding->src_physical_output_index;
    } else {
      parent_source_physical_index = physical->physical_index;
    }
    if (parent_source_physical_index < 0) {
      return true;
    }

    std::uint64_t absolute_byte_offset = 0U;
    if (physical->source_physical_index >= 0 && physical->source_byte_offset > 0) {
      absolute_byte_offset = static_cast<std::uint64_t>(physical->source_byte_offset);
      saw_parent_view = true;
    } else if (binding && binding->src_physical_byte_offset > 0) {
      absolute_byte_offset = static_cast<std::uint64_t>(binding->src_physical_byte_offset);
    }
    if (logical.byte_offset > 0) {
      const auto logical_offset = static_cast<std::uint64_t>(logical.byte_offset);
      if (absolute_byte_offset > std::numeric_limits<std::uint64_t>::max() - logical_offset) {
        return fail("shared parent input view offset overflow");
      }
      absolute_byte_offset += logical_offset;
    }

    const std::uint64_t span_bytes =
        physical_span_bytes_for_tensor(logical.shape, logical.stride_bytes, logical.size_bytes);

    int group_index = -1;
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      if (groups[gi].parent_source_physical_index == parent_source_physical_index) {
        group_index = static_cast<int>(gi);
        break;
      }
    }
    if (group_index < 0) {
      Group group;
      group.parent_source_physical_index = parent_source_physical_index;
      group.segment_name = physical->segment_name;
      group.device_kind = physical->device_kind;
      groups.push_back(std::move(group));
      group_index = static_cast<int>(groups.size() - 1U);
    }

    auto& group = groups[static_cast<std::size_t>(group_index)];
    if ((absolute_byte_offset == 0U || !group.have_base_byte_offset ||
         absolute_byte_offset < group.base_byte_offset)) {
      group.base_byte_offset = absolute_byte_offset;
      group.have_base_byte_offset = true;
      if (!physical->segment_name.empty()) {
        group.segment_name = physical->segment_name;
      }
    } else if (group.segment_name.empty() && !physical->segment_name.empty()) {
      group.segment_name = physical->segment_name;
    }
    if (group.device_kind == pipeline_internal::sima::DeviceKind::Unknown) {
      group.device_kind = physical->device_kind;
    }

    if (absolute_byte_offset > std::numeric_limits<std::uint64_t>::max() - span_bytes) {
      return fail("shared parent input view span overflow");
    }
    const std::uint64_t member_end = absolute_byte_offset + span_bytes;
    group.size_bytes = std::max(group.size_bytes, member_end);
    if (physical->source_byte_offset == 0 && physical->size_bytes > group.size_bytes) {
      group.size_bytes = physical->size_bytes;
    }

    group.members.push_back(members.size());
    group_for_logical[i] = group_index;
    members.push_back(Member{i, binding_pos, parent_source_physical_index, absolute_byte_offset,
                             span_bytes, physical});
    if (physical->source_physical_index >= 0 &&
        physical->source_physical_index != physical->physical_index) {
      saw_parent_view = true;
    }
  }

  if (!saw_parent_view || groups.size() >= contract->physical_inputs.size()) {
    return true;
  }

  std::vector<PhysicalBufferStaticSpec> normalized_physical_inputs;
  normalized_physical_inputs.reserve(groups.size());
  for (std::size_t gi = 0; gi < groups.size(); ++gi) {
    auto& group = groups[gi];
    const std::uint64_t base = group.have_base_byte_offset ? group.base_byte_offset : 0U;
    const std::uint64_t size_bytes =
        group.size_bytes > base ? group.size_bytes - base : group.size_bytes;
    if (group.segment_name.empty()) {
      group.segment_name = groups.size() == 1U ? "input_tensor"
                                               : "input_tensor_" + std::to_string(gi);
    }
    normalized_physical_inputs.push_back(
        pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
            static_cast<int>(gi), static_cast<int>(gi), size_bytes, group.device_kind,
            group.segment_name, group.parent_source_physical_index,
            base <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                ? static_cast<std::int64_t>(base)
                : std::numeric_limits<std::int64_t>::max()));
  }

  for (const auto& member : members) {
    const int group_index = group_for_logical[member.logical_pos];
    if (group_index < 0) {
      continue;
    }
    const auto& group = groups[static_cast<std::size_t>(group_index)];
    const std::uint64_t base = group.have_base_byte_offset ? group.base_byte_offset : 0U;
    if (member.absolute_byte_offset < base) {
      return fail("shared parent input view offset precedes group base");
    }
    const std::uint64_t local_offset = member.absolute_byte_offset - base;
    if (local_offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return fail("shared parent input view local offset exceeds int64");
    }

    auto& logical = contract->logical_inputs[member.logical_pos];
    logical.physical_index = group_index;
    logical.byte_offset = static_cast<std::int64_t>(local_offset);
    logical.segment_name = group.segment_name;

    if (member.binding_pos < contract->input_bindings.size()) {
      auto& binding = contract->input_bindings[member.binding_pos];
      binding.src_physical_output_index = group.parent_source_physical_index;
      binding.src_physical_byte_offset = static_cast<std::int64_t>(local_offset);
      binding.src_physical_size_bytes = member.physical_span_bytes;
      binding.source_segment_name = group.segment_name;
    }
  }

  contract->physical_inputs = std::move(normalized_physical_inputs);
  return validate_packed_contract(*contract, err);
}

std::vector<simaai::gst::TensorBufferSelector>
selectors_for_logical_inputs(const CompiledRuntimeContract& contract) {
  std::vector<simaai::gst::TensorBufferSelector> selectors;
  selectors.reserve(contract.logical_inputs.size());
  for (const auto& logical : contract.logical_inputs) {
    simaai::gst::TensorBufferSelector selector;
    selector.logical_index = logical.logical_index;
    selector.physical_index = logical.physical_index;
    selector.logical_name = logical.logical_name;
    selector.segment_name = logical.segment_name;
    selector.byte_offset = logical.byte_offset;
    const auto binding_it =
        std::find_if(contract.input_bindings.begin(), contract.input_bindings.end(),
                     [&](const pipeline_internal::sima::InputBindingStaticSpec& binding) {
                       return binding.local_logical_input_index == logical.logical_index;
                     });
    if (binding_it != contract.input_bindings.end()) {
      selector.logical_index = binding_it->src_logical_output_index >= 0
                                   ? binding_it->src_logical_output_index
                                   : selector.logical_index;
      selector.route_slot =
          binding_it->src_output_slot >= 0 ? binding_it->src_output_slot : selector.route_slot;
      selector.physical_index = binding_it->src_physical_output_index >= 0
                                    ? binding_it->src_physical_output_index
                                    : selector.physical_index;
      selector.byte_offset = binding_it->src_physical_byte_offset;
      if (!binding_it->source_segment_name.empty()) {
        selector.segment_name = binding_it->source_segment_name;
      }
    }
    selectors.push_back(std::move(selector));
  }
  return selectors;
}

bool prepare_physical_inputs(const CompiledRuntimeContract& contract,
                             const simaai::gst::TensorBufferView& upstream_view,
                             std::vector<std::uint8_t>* out_bytes,
                             std::string* err) {
  if (!out_bytes) {
    if (err) {
      *err = "packed input preparation requires output byte storage";
    }
    return false;
  }
  if (!validate_packed_contract(contract, err)) {
    return false;
  }
  return simaai::gst::tensor_buffer_materialize(
      upstream_view, selectors_for_logical_inputs(contract), out_bytes, err);
}

bool publish_logical_outputs(const CompiledRuntimeContract& contract,
                             const std::string& stage_key,
                             simaai::gst::TensorBufferView* out,
                             std::string* err) {
  if (!out) {
    if (err) {
      *err = "packed output publication requires output view";
    }
    return false;
  }
  if (!validate_packed_contract(contract, err)) {
    return false;
  }

  out->stage_key = stage_key;
  out->segments.clear();
  out->tensors.clear();
  out->segments.reserve(contract.physical_outputs.size());
  out->tensors.reserve(contract.logical_outputs.size());

  for (const auto& physical : contract.physical_outputs) {
    simaai::gst::TensorBufferSegmentView segment;
    segment.name = physical.segment_name;
    segment.size_bytes = static_cast<std::size_t>(physical.size_bytes);
    out->segments.push_back(std::move(segment));
  }
  for (const auto& logical : contract.logical_outputs) {
    simaai::gst::TensorBufferTensorView tensor;
    tensor.logical_index = logical.logical_index;
    tensor.physical_index = logical.physical_index;
    tensor.backend_output_index = logical.backend_output_index;
    tensor.route_slot = logical.output_slot;
    tensor.memory_index = logical.physical_index;
    tensor.logical_name = logical.logical_name;
    tensor.backend_name = logical.backend_name;
    tensor.segment_name = logical.segment_name;
    tensor.byte_offset = logical.byte_offset;
    tensor.size_bytes = logical.size_bytes;
    tensor.shape = logical.shape;
    tensor.stride_bytes = logical.stride_bytes;
    tensor.physical_span_bytes =
        physical_span_bytes_for_tensor(tensor.shape, tensor.stride_bytes, tensor.size_bytes);
    tensor.dtype = SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1;
    tensor.layout = SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1;
    out->tensors.push_back(std::move(tensor));
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::packedio

#include "pipeline/internal/OrderedPublicInputContract.h"

#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/TensorMath.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace simaai::neat::pipeline_internal {
namespace {
[[noreturn]] void reject(const std::string& detail) {
  error_util::throw_session_error(error_codes::kInputShape,
                                  "ordered tensor input contract: " + detail);
}

void validate_member(const PublicInputTensorSpec& expected, const Tensor& tensor,
                     std::size_t slot) {
  const auto fail = [&](const std::string& detail) {
    reject("slot " + std::to_string(slot) + " ('" + expected.name + "'): " + detail);
  };
  if (!tensor.route.name.empty() && tensor.route.name != expected.name)
    fail("logical name '" + tensor.route.name + "' does not match expected role");
  if (tensor.dtype != expected.dtype)
    fail("dtype mismatch (expected " + std::to_string(static_cast<int>(expected.dtype)) + ", got " +
         std::to_string(static_cast<int>(tensor.dtype)) + ")");
  if (!tensor.planes.empty() || tensor.semantic.encoded || tensor.semantic.byte_stream ||
      tensor.semantic.tess)
    fail("expected one dense logical tensor, not an opaque or composite carrier");
  InputOptions options;
  options.payload_type = PayloadType::Tensor;
  InputGeometry geometry;
  try {
    // Metadata-only bounds validation. Never map a GstBuffer or inspect payload.
    geometry = describe_tensor_geometry_or_throw(tensor, options, "tensor storage");
  } catch (const std::invalid_argument& error) {
    fail(error.what());
  }
  // Unknown preserves the public from_vector convenience. Explicit layouts
  // must agree, except the existing one-channel image projection to HW.
  bool layout_ok = tensor.layout == TensorLayout::Unknown ||
                   expected.layout == TensorLayout::Unknown || tensor.layout == expected.layout;
  if (!layout_ok && expected.layout == TensorLayout::HW &&
      (tensor.layout == TensorLayout::HWC || tensor.layout == TensorLayout::CHW) &&
      tensor.shape.size() >= 3 && tensor.shape.size() <= 4) {
    const auto channel =
        tensor.layout == TensorLayout::HWC ? tensor.shape.size() - 1 : tensor.shape.size() - 3;
    layout_ok = tensor.shape[channel] == 1;
  }
  if (!layout_ok ||
      !input_geometry_matches_logical_shape(geometry, expected.shape, expected.layout)) {
    const auto shape = [](const auto& dimensions) {
      std::ostringstream text;
      text << '[';
      for (std::size_t i = 0; i < dimensions.size(); ++i) {
        if (i)
          text << ',';
        text << dimensions[i];
      }
      return text.str() + ']';
    };
    fail("logical shape/layout mismatch (expected " + shape(expected.shape) + " layout " +
         std::to_string(static_cast<int>(expected.layout)) + ", got " + shape(tensor.shape) +
         " layout " + std::to_string(static_cast<int>(tensor.layout)) + ")");
  }
  if (static_cast<std::size_t>(tensor.byte_offset) % dtype_bytes(tensor.dtype) != 0)
    fail("byte offset is not aligned to the element type");
  // describe_tensor_geometry checked positive dimensions and overflow first.
  if (!tensor.strides_bytes.empty()) {
    std::size_t stride = dtype_bytes(tensor.dtype);
    for (std::size_t axis = tensor.shape.size(); axis-- > 0;) {
      if (static_cast<std::size_t>(tensor.strides_bytes[axis]) != stride)
        fail("noncontiguous strides are unsupported");
      stride *= static_cast<std::size_t>(tensor.shape[axis]);
    }
  }
}
} // namespace

void validate_ordered_public_input(const OrderedPublicInputContract& contract,
                                   const Tensor& tensor) {
  if (contract.inputs.size() != 1U)
    reject("expected " + std::to_string(contract.inputs.size()) + " tensors, got 1");
  validate_member(contract.inputs.front(), tensor, 0);
}

void validate_ordered_public_input(const OrderedPublicInputContract& contract,
                                   const Sample& sample) {
  // Match tensors_from_sample without allocating or modifying any route/owner.
  const auto visit = [&](auto&& self, const Sample& value, auto&& member) -> void {
    if (sample_has_tensor_list(value) && !value.tensors.empty()) {
      for (const auto& tensor : value.tensors)
        member(tensor);
    } else if (value.kind == SampleKind::Tensor && value.tensor) {
      member(*value.tensor);
    } else if (value.kind == SampleKind::Bundle && !value.fields.empty()) {
      for (const auto& field : value.fields)
        self(self, field, member);
    } else {
      reject("sample has no tensor metadata");
    }
  };
  std::size_t count = 0;
  visit(visit, sample, [&](const Tensor&) { ++count; });
  if (count != contract.inputs.size())
    reject("expected " + std::to_string(contract.inputs.size()) + " tensors, got " +
           std::to_string(count));
  std::size_t slot = 0;
  visit(visit, sample, [&](const Tensor& tensor) {
    validate_member(contract.inputs[slot], tensor, slot);
    ++slot;
  });
}
} // namespace simaai::neat::pipeline_internal

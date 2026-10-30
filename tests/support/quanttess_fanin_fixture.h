#pragma once

#include "support/processcvu_plan_fixture.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace sima_test {

// Explicit compiler facts for fan-in tests. The fixture describes either
// separate public tensors or views into a caller-owned packed input. Pack
// placement is authored here, not reconstructed by a second MPK interpreter.
inline auto quanttess_fanin_plan_data(const std::vector<std::int64_t>& sides,
                                      const std::vector<std::size_t>& boundary_order,
                                      bool packed_ingress = false) {
  namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  auto add_value = [&](std::string name, std::uint64_t bytes, std::string dtype,
                       sc::TensorShape shape, sc::ValueRepresentation representation) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    sc::ValueSpec value;
    value.id = id;
    value.name = std::move(name);
    value.required_bytes = bytes;
    value.logical_dtype = std::move(dtype);
    value.logical_shape = std::move(shape);
    value.logical_layout = "NHWC";
    value.representation = representation;
    data.values.push_back(std::move(value));
    return id;
  };
  std::uint64_t ingress_bytes = 0;
  for (auto side : sides)
    ingress_bytes += static_cast<std::uint64_t>(side * side) * 4U;
  if (packed_ingress) {
    add_value("public_packed_input", ingress_bytes, "FP32",
              {static_cast<std::int64_t>(ingress_bytes / 4U)}, sc::ValueRepresentation::Dense);
    data.values.front().storage_binding =
        sc::StorageBinding{sc::StorageBindingKind::External, 0U,          0U, ingress_bytes, {},
                           sc::StorageAccess::ReadOnly,      std::nullopt};
    data.model_inputs = {0U};
  }
  std::vector<sc::ValueId> inputs, scalar, tiled;
  std::uint64_t offset = 0;
  for (std::size_t lane = 0; lane < sides.size(); ++lane) {
    const auto bytes = static_cast<std::uint64_t>(sides[lane] * sides[lane]) * 4U;
    const auto id = add_value("input_" + std::to_string(lane), bytes, "FP32",
                              {1, sides[lane], sides[lane], 1}, sc::ValueRepresentation::Dense);
    inputs.push_back(id);
    if (packed_ingress) {
      data.values[id].storage_binding =
          sc::StorageBinding{sc::StorageBindingKind::View,
                             0U,
                             offset,
                             bytes,
                             {static_cast<std::int64_t>(bytes), sides[lane] * 4, 4, 4},
                             sc::StorageAccess::ReadOnly,
                             0U};
    } else {
      data.model_inputs.push_back(id);
      data.values[id].storage_binding =
          sc::StorageBinding{sc::StorageBindingKind::External, id,          0U, bytes, {},
                             sc::StorageAccess::ReadOnly,      std::nullopt};
    }
    offset += bytes;
  }
  auto add_op = [&](sc::OpKind kind, std::string name, std::vector<sc::ValueId> in,
                    std::vector<sc::ValueId> out, sc::OpConfig config) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = std::move(name);
    op.kind = kind;
    op.processor = kind == sc::OpKind::Mla ? "MLA" : "EV74";
    op.inputs = std::move(in);
    op.outputs = std::move(out);
    for (auto id : op.inputs)
      op.input_shapes.push_back(*data.values[id].logical_shape);
    for (auto id : op.outputs)
      op.output_shapes.push_back(*data.values[id].logical_shape);
    op.config = std::move(config);
    data.ops.push_back(std::move(op));
  };
  for (std::size_t lane = 0; lane < sides.size(); ++lane) {
    const auto side = sides[lane];
    const auto bytes = static_cast<std::uint64_t>(side * side);
    scalar.push_back(add_value("quant_" + std::to_string(lane), bytes, "INT8", {1, side, side, 1},
                               sc::ValueRepresentation::Dense));
    // These tiny fixture tiles all fit one canonical 16-byte storage quantum.
    if (bytes > 16U)
      throw std::invalid_argument("fan-in fixture tile exceeds its explicit 16-byte extent");
    tiled.push_back(add_value("tess_" + std::to_string(lane), 16U, "INT8", {1, side, side, 1},
                              sc::ValueRepresentation::Tessellated));
    add_op(sc::OpKind::Quantize, "quant_" + std::to_string(lane), {inputs[lane]}, {scalar[lane]},
           sc::QuantizeOpConfig{"INT8", 8, "TONEAREST", {{0.25, -7}}});
    add_op(sc::OpKind::Tessellate, "tess_" + std::to_string(lane), {scalar[lane]}, {tiled[lane]},
           sc::TessellateOpConfig{{side, side, 1}, false, false, "INT8"});
  }
  const auto packed_bytes = 16U * sides.size();
  const auto packed =
      add_value("ifm_pack", packed_bytes, "INT8", {1, static_cast<std::int64_t>(packed_bytes)},
                sc::ValueRepresentation::Packed);
  sc::PackOpConfig pack;
  std::vector<sc::ValueId> pack_inputs;
  for (std::size_t position = 0; position < boundary_order.size(); ++position) {
    const auto id = tiled.at(boundary_order[position]);
    pack_inputs.push_back(id);
    pack.components.push_back({id, position * 16U, 16U});
  }
  add_op(sc::OpKind::Pack, "ifm_pack", pack_inputs, {packed}, std::move(pack));
  const auto output =
      add_value("mla_output", 16U, "INT8", {1, 16}, sc::ValueRepresentation::BackendNative);
  add_op(sc::OpKind::Mla, "MLA_0", {packed}, {output}, sc::MlaOpConfig{"fixture.elf", 1});
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", packed, packed_bytes, 4096U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", output, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly}};
  data.model_outputs = {{0U, "mla_output", output}};
  return data;
}

} // namespace sima_test

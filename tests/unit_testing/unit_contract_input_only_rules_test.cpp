#include "contracts/NodeContractDefinition.h"
#include "nodes/sima/Preproc.h"
#include "test_main.h"

RUN_TEST("unit_contract_input_only_rules_test", ([] {
  using namespace simaai::neat;

  const PreprocOptions opt;
  Preproc node(opt);
  const NodeContractDefinition def = node.contract_definition();

  auto find_field = [&](const std::string& field_id) -> const ContractFieldSpec& {
    for (const auto& field : def.fields) {
      if (field.field_id == field_id) {
        return field;
      }
    }
    throw std::runtime_error("missing field spec: " + field_id);
  };

  const auto& input_shape = find_field("input_shape");
  const auto& input_img_type = find_field("input_img_type");
  const auto& output_img_type = find_field("output_img_type");
  const auto& tessellate = find_field("tessellate");

  require(input_shape.source == ContractFieldSource::InputOnly && input_shape.required,
          "input_shape should be required InputOnly");
  require(input_img_type.source == ContractFieldSource::InputOnly && input_img_type.required,
          "input_img_type should be required InputOnly");
  require(output_img_type.source == ContractFieldSource::BuilderOption &&
              output_img_type.override_policy == ContractOverridePolicy::BuilderOnly,
          "output_img_type should remain a builder-only field");
  require(tessellate.source == ContractFieldSource::BuilderOption &&
              tessellate.override_policy == ContractOverridePolicy::BuilderOnly,
          "tessellate should remain a builder-only field");
}));

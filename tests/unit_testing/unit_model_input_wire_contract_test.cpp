#include "pipeline/internal/PublicInputContract.h"
#include "test_main.h"

#include <cstdint>
#include <vector>

namespace {
using namespace simaai::neat::pipeline_internal;

void checked_alias_contract() {
  sima::MpkTensorContract authored;
  authored.shape_semantics = sima::MpkShapeSemantics::Geometry;
  authored.mpk_shape = {1, 36, 36, 256};
  authored.logical_shape = {36, 36, 256};
  sima::LogicalInputStaticSpec consumer;
  consumer.shape = authored.logical_shape;
  require(model_input_wire_shape(authored, consumer) == consumer.shape,
          "checked authored/logical alias must select the consumer wire shape");

  // No axis guess is required: the model has already checked the relationship.
  require(authored.mpk_shape == std::vector<std::int64_t>({1, 36, 36, 256}) &&
              authored.logical_shape == consumer.shape && consumer.layout.empty(),
          "wire selection must not rewrite authored shape, logical shape or layout");

  const auto rejects = [&](sima::MpkTensorContract input, sima::LogicalInputStaticSpec bound) {
    require(model_input_wire_shape(input, bound).empty(),
            "an unproved relationship must not authorize a wire alias");
  };
  auto changed = authored;
  changed.shape_semantics = sima::MpkShapeSemantics::PackedExtent;
  rejects(changed, consumer);
  changed.shape_semantics = sima::MpkShapeSemantics::Unknown;
  rejects(changed, consumer);
  changed = authored;
  changed.mpk_shape.front() = 2;
  rejects(changed, consumer);
  changed = authored;
  changed.mpk_shape.insert(changed.mpk_shape.begin(), 1);
  rejects(changed, consumer);
  changed = authored;
  changed.mpk_shape[1] = 37;
  rejects(changed, consumer);
  changed = authored;
  changed.logical_shape = changed.mpk_shape;
  rejects(changed, consumer);
  auto wrong_consumer = consumer;
  wrong_consumer.shape = {36, 36, 255};
  rejects(authored, wrong_consumer);
  changed = authored;
  changed.mpk_shape[1] = 0;
  changed.logical_shape[0] = 0;
  wrong_consumer.shape = changed.logical_shape;
  rejects(changed, wrong_consumer);

  // RF-DETR proposals are rank three already, not another batch-one alias.
  changed = authored;
  changed.mpk_shape = {1, 200, 4};
  changed.logical_shape = changed.mpk_shape;
  wrong_consumer.shape = changed.logical_shape;
  rejects(changed, wrong_consumer);
}
} // namespace

RUN_TEST("unit_model_input_wire_contract_test", checked_alias_contract);

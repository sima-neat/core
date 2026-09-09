#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

constexpr int kClassDepth = 37; // Concat(objectness[1], classes[36])
constexpr int kExpectedClasses = kClassDepth - 1;

// Thirteen grouped-by-role heads: [bbox x3][class x3][mask_coeff x3][kpt x3][proto].
// class_token names the class heads only; every other head keeps a role name that declares
// no score domain.
TensorList make_heads(const std::string& class_token) {
  const std::vector<int> grid = {80, 40, 20};
  TensorList tensors;
  auto add = [&](int h, int w, int channels, const std::string& name) {
    Tensor tensor;
    tensor.dtype = TensorDType::BFloat16;
    tensor.layout = TensorLayout::HWC;
    tensor.shape = {h, w, channels};
    tensor.storage = make_cpu_owned_storage(static_cast<std::size_t>(h) * w * channels * 2U);
    tensor.route.logical_index = static_cast<int>(tensors.size());
    tensor.route.physical_index = tensor.route.logical_index;
    tensor.route.route_slot = tensor.route.logical_index;
    tensor.route.name = name;
    tensors.push_back(std::move(tensor));
  };
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], 4, "bbox_" + std::to_string(i));
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], kClassDepth, class_token + "_" + std::to_string(i));
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], 32, "mask_coeff_" + std::to_string(i));
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], 39, "kpt_" + std::to_string(i));
  }
  add(160, 160, 32, "mask_proto");
  return tensors;
}

// Heads carrying no role hint at all, the shape a raw export takes.
TensorList make_generic_heads() {
  TensorList tensors = make_heads("class");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    tensors[i].route.name = "output_" + std::to_string(i);
  }
  return tensors;
}

BoxDecodeStaticContract extract(const TensorList& tensors, const std::string& label) {
  BoxDecodeStandaloneContractOverrides overrides;
  overrides.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
  std::string error;
  auto extracted = build_boxdecode_static_contract_from_sample(
      sample_from_tensors(tensors), BoxDecodeType::YoloXSegPose, std::nullopt, overrides, &error);
  require(extracted.has_value(), label + " must extract a standalone contract: " + error);
  return *extracted;
}

// The standalone node route always passes num_classes=0 (SimaBoxDecode.cpp).
BoxDecodeStaticContract finalize(const BoxDecodeStaticContract& contract) {
  return finalize_boxdecode_static_contract(contract, BoxDecodeType::YoloXSegPose, std::nullopt,
                                            std::nullopt, BoxDecodeTypeOption::Auto, 0.40, 0.45,
                                            100, /*num_classes=*/0, {});
}

} // namespace

RUN_TEST("unit_boxdecode_yolox_seg_pose_standalone_contract_test", ([] {
           // Names that encode no activation must still construct: this family is always raw
           // logits, so there is nothing for a name to disambiguate.
           for (const char* class_token : {"class_score", "cls_score", "class_logit"}) {
             const auto extracted = extract(make_heads(class_token), class_token);
             require(extracted.score_activation == BoxDecodeScoreActivation::Sigmoid,
                     std::string(class_token) + " heads must resolve to sigmoid");
             const auto finalized = finalize(extracted);
             require(finalized.score_activation == BoxDecodeScoreActivation::Sigmoid,
                     std::string(class_token) + " heads must stay sigmoid after finalization");
             require(finalized.num_classes == kExpectedClasses,
                     std::string(class_token) + " heads must resolve num_classes=36");
           }

           // The same export with no role hints at all.
           const auto generic = extract(make_generic_heads(), "output_N");
           require(generic.score_activation == BoxDecodeScoreActivation::Sigmoid,
                   "unnamed heads must resolve to sigmoid rather than fail extraction");
           const auto finalized = finalize(generic);
           require(finalized.score_activation == BoxDecodeScoreActivation::Sigmoid,
                   "unnamed heads must stay sigmoid after finalization");
           require(finalized.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
                   "unnamed heads must normalize to the grouped-by-role layout");
           require(finalized.num_classes == kExpectedClasses,
                   "unnamed heads must resolve num_classes=36");

           // A declared probability domain contradicts the family contract and must be rejected
           // rather than silently reinterpreted as logits.
           const auto declared_prob = extract(make_heads("class_prob"), "class_prob");
           require(declared_prob.score_activation == BoxDecodeScoreActivation::Identity,
                   "class_prob names must still declare the probability domain");
           bool rejected = false;
           try {
             finalize(declared_prob);
           } catch (const std::invalid_argument&) {
             rejected = true;
           }
           require(rejected, "a declared probability domain must be rejected, not overwritten");
         }));

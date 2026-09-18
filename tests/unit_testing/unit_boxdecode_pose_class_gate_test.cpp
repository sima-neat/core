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

BoxDecodeTensorStaticContract make_tensor(std::string name, int h, int w, int depth) {
  BoxDecodeTensorStaticContract out;
  out.logical_name = std::move(name);
  out.data_type = "BF16";
  out.layout = "HWC";
  out.input_shape = {h, w, depth};
  out.slice_shape = {h, w, depth};
  out.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
  return out;
}

BoxDecodeStaticContract make_contract(std::vector<int> pose_classes) {
  const std::vector<int> grid = {80, 40, 20};
  const std::vector<std::pair<const char*, int>> roles = {
      {"bbox", 4}, {"cls_score", kClassDepth}, {"mask_coeff", 32}, {"kpt", 39}};

  BoxDecodeStaticContract contract;
  contract.decode_type = BoxDecodeType::YoloXSegPose;
  contract.pose_classes = std::move(pose_classes);
  for (const auto& [role, depth] : roles) {
    for (std::size_t head = 0; head < grid.size(); ++head) {
      contract.tensors.push_back(make_tensor(std::string(role) + "_" + std::to_string(head),
                                             grid[head], grid[head], depth));
    }
  }
  contract.tensors.push_back(make_tensor("mask_proto", 160, 160, 32));
  for (const auto& tensor : contract.tensors) {
    contract.tensor_names.push_back(tensor.logical_name);
  }
  return contract;
}

BoxDecodeStaticContract finalize(const BoxDecodeStaticContract& contract,
                                 BoxDecodeType decode_type = BoxDecodeType::YoloXSegPose) {
  return finalize_boxdecode_static_contract(contract, decode_type, std::nullopt, std::nullopt,
                                            BoxDecodeTypeOption::Auto, 0.40, 0.45, 100,
                                            /*num_classes=*/0, {});
}

bool rejects(const BoxDecodeStaticContract& contract,
             BoxDecodeType decode_type = BoxDecodeType::YoloXSegPose) {
  try {
    finalize(contract, decode_type);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

} // namespace

RUN_TEST("unit_boxdecode_pose_class_gate_test", ([] {
           // An empty gate must stay empty: the backend reads a present-but-empty list as
           // "no class is pose-bearing" and would zero every keypoint record.
           const auto ungated = finalize(make_contract({}));
           require(ungated.num_classes == kExpectedClasses, "the fixture must resolve 36 classes");
           require(ungated.pose_classes.empty(), "an unset gate must not become a populated list");

           // A valid gate survives finalization, sorted and deduplicated to a canonical form.
           const auto gated = finalize(make_contract({7, 0, 3}));
           require((gated.pose_classes == std::vector<int>{0, 3, 7}),
                   "a valid gate must be carried through sorted");

           // Boundary classes are legal; one past the end is not.
           require(!rejects(make_contract({0, kExpectedClasses - 1})),
                   "the first and last class indices must be accepted");
           require(rejects(make_contract({kExpectedClasses})),
                   "a class index equal to num_classes must be rejected");
           require(rejects(make_contract({-1})), "a negative class index must be rejected");
           require(rejects(make_contract({3, 3})), "a repeated class index must be rejected");

           // The gate is meaningless where classes cannot be mixed, so it is refused rather
           // than silently dropped on the way to the backend. Class 0 is in range for a pose
           // family (which resolves num_classes=1), so only the decode-type rule can reject it.
           auto wrong_family = make_contract({0});
           wrong_family.decode_type = BoxDecodeType::YoloV8Pose;
           require(rejects(wrong_family, BoxDecodeType::YoloV8Pose),
                   "pose_classes must be rejected for a decode type that cannot gate by class");

           // normalize_boxdecode_pose_classes is the single validation point both routes share.
           require(
               normalize_boxdecode_pose_classes(BoxDecodeType::YoloXSegPose, {}, 0, "test").empty(),
               "an empty gate must not require a resolved num_classes");
           bool threw = false;
           try {
             normalize_boxdecode_pose_classes(BoxDecodeType::YoloXSegPose, {0}, 0, "test");
           } catch (const std::invalid_argument&) {
             threw = true;
           }
           require(threw, "a non-empty gate must require a resolved num_classes");
         }));

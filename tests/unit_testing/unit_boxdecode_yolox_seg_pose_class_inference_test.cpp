#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <array>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;

constexpr int kGridH = 80;
constexpr int kGridW = 80;
constexpr int kProtoExtent = 160;
constexpr int kClassDepth = 37; // Concat(objectness[1], classes[36])
constexpr int kKptDepth = 39;   // 13 keypoints x 3
constexpr int kExpectedClasses = kClassDepth - 1;

// Channel storage is padded to 16-channel stripes, so the padded extent is what a real
// contract carries in input_shape while slice_shape keeps the logical depth.
int align_c16(int depth) {
  return ((depth + 15) / 16) * 16;
}

BoxDecodeTensorStaticContract make_tensor(std::string name, int h, int w, int depth) {
  BoxDecodeTensorStaticContract out;
  out.logical_name = std::move(name);
  out.data_type = "BF16";
  out.layout = "HWC";
  out.input_shape = {h, w, align_c16(depth)};
  out.slice_shape = {h, w, depth};
  out.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
  return out;
}

struct Roles {
  int bbox = 4;
  int cls = kClassDepth;
  int mask_coeff = 32;
  int kpt = kKptDepth;
};

// role_major: [bbox x3][class x3][mask_coeff x3][kpt x3][proto]
// head_major: [bbox, class, mask_coeff, kpt] x3, then [proto]
BoxDecodeStaticContract make_contract(bool role_major, bool descriptive_names,
                                      Roles roles = Roles{}) {
  const std::array<int, 4> depths{roles.bbox, roles.cls, roles.mask_coeff, roles.kpt};
  const std::array<const char*, 4> role_names{"bbox", "cls_score", "mask_coeff", "kpt"};
  const std::array<int, 3> grid_h{kGridH, kGridH / 2, kGridH / 4};
  const std::array<int, 3> grid_w{kGridW, kGridW / 2, kGridW / 4};

  BoxDecodeStaticContract contract;
  contract.decode_type = BoxDecodeType::YoloXSegPose;
  contract.tensors.resize(13);
  for (int role = 0; role < 4; ++role) {
    for (int head = 0; head < 3; ++head) {
      const std::size_t index =
          static_cast<std::size_t>(role_major ? role * 3 + head : head * 4 + role);
      const std::string name = descriptive_names
                                   ? std::string(role_names[role]) + "_" + std::to_string(head)
                                   : "output_" + std::to_string(index);
      contract.tensors[index] = make_tensor(name, grid_h[head], grid_w[head], depths[role]);
    }
  }
  contract.tensors[12] =
      make_tensor(descriptive_names ? "mask_proto" : "output_12", kProtoExtent, kProtoExtent, 32);
  for (const auto& tensor : contract.tensors) {
    contract.tensor_names.push_back(tensor.logical_name);
  }
  return contract;
}

int resolve(BoxDecodeStaticContract contract) {
  stagesemantics::apply_yolox_seg_pose_model_managed_contract_defaults(&contract);
  return contract.num_classes;
}

bool rejects(BoxDecodeStaticContract contract) {
  try {
    stagesemantics::apply_yolox_seg_pose_model_managed_contract_defaults(&contract);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_boxdecode_yolox_seg_pose_class_inference_test", ([] {
      // Renaming otherwise identical inputs must not change the resolved class count.
      require(resolve(make_contract(/*role_major=*/true, /*descriptive_names=*/true)) ==
                  kExpectedClasses,
              "role-major named heads must resolve num_classes=36");
      require(resolve(make_contract(/*role_major=*/true, /*descriptive_names=*/false)) ==
                  kExpectedClasses,
              "role-major generic heads must resolve num_classes=36");
      require(resolve(make_contract(/*role_major=*/false, /*descriptive_names=*/false)) ==
                  kExpectedClasses,
              "head-major generic heads must resolve num_classes=36");

      // The padded 48-channel storage must never be mistaken for the class depth.
      require(align_c16(kClassDepth) == 48, "class head must be stored padded to 48 channels");

      // An explicit count still has to agree with the derived one, on either name set.
      for (const bool descriptive : {true, false}) {
        auto contract = make_contract(/*role_major=*/true, descriptive);
        contract.num_classes = kExpectedClasses;
        require(resolve(contract) == kExpectedClasses,
                "a matching explicit num_classes must be accepted");
        contract.num_classes = kExpectedClasses + 1;
        require(rejects(contract), "a mismatched explicit num_classes must be rejected");
      }

      // Positional derivation is declined when the geometry is not the documented one, so
      // a malformed export still fails loudly instead of resolving a plausible wrong count.
      Roles bad_mask_coeff;
      bad_mask_coeff.mask_coeff = 31;
      require(
          rejects(make_contract(/*role_major=*/true, /*descriptive_names=*/false, bad_mask_coeff)),
          "a non-32 mask-coefficient depth must not yield a derived class count");

      Roles bad_kpt;
      bad_kpt.kpt = 40; // not a multiple of 3
      require(rejects(make_contract(/*role_major=*/true, /*descriptive_names=*/false, bad_kpt)),
              "a keypoint depth that is not 3*K must not yield a derived class count");

      Roles bad_bbox;
      bad_bbox.bbox = 6;
      require(rejects(make_contract(/*role_major=*/true, /*descriptive_names=*/false, bad_bbox)),
              "a non-4 bbox depth must not yield a derived class count");

      auto short_contract = make_contract(/*role_major=*/true, /*descriptive_names=*/false);
      short_contract.tensors.pop_back();
      require(rejects(short_contract), "a contract without the mask prototype must be "
                                       "rejected rather than derived positionally");
    }));

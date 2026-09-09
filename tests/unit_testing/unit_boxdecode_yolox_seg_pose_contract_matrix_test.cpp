#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example yolox_seg_pose_contract_matrix
 * Device-free coverage of the yolox-seg-pose raw-head contract path, from the head
 * geometry a model exports down to the compiled payload the on-device decoder consumes.
 *
 * Core performs no decode arithmetic - the box/mask/keypoint math lives in the a65
 * backend - so what is verifiable here is everything that decides how the backend will
 * read those heads: class count, layout, activation, dtype and the pose-class gate.
 * Numeric decode output is covered against a byte-layout reference by
 * unit_detection_types_boxdecode_payloads_test.
 *
 * Needs no model pack or dispatcher, so it must fail rather than skip.
 */
#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <optional>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

// Logical head depths. The class head is Concat(objectness[1], classes[N]), so its depth
// is classes + 1; the keypoint head is 3 channels per point.
struct HeadDepths {
  int bbox = 4;
  int classes = 36;
  int mask_coeff = 32;
  int keypoints = 13;
  int proto = 32;
};

struct Case {
  std::string name;
  HeadDepths heads;
  bool role_major = true;
  bool descriptive_names = true;
  bool pad_c16 = false;
  bool model_managed = false;
  std::string dtype = "BF16";
  BoxDecodeSourceStorageKind storage = BoxDecodeSourceStorageKind::DenseHwcPhysical;
  std::vector<int> grid_h = {80, 40, 20};
  std::vector<int> grid_w = {80, 40, 20};
  std::vector<int> pose_classes;
  std::vector<int> expect_pose_classes; // empty: expect pose_classes verbatim
  int tensor_count_override = 0;        // 0 keeps the full 13
  std::string expect_error;             // non-empty: must throw with this substring
};

int align_c16(int depth) {
  return ((depth + 15) / 16) * 16;
}

BoxDecodeStaticContract build_contract(const Case& c) {
  const int class_depth = c.heads.classes + 1;
  const int kpt_depth = c.heads.keypoints * 3;
  const std::vector<std::pair<const char*, int>> roles = {{"bbox", c.heads.bbox},
                                                          {"cls_score", class_depth},
                                                          {"mask_coeff", c.heads.mask_coeff},
                                                          {"kpt", kpt_depth}};
  const std::size_t levels = c.grid_h.size();

  BoxDecodeStaticContract contract;
  contract.decode_type = BoxDecodeType::YoloXSegPose;
  contract.pose_classes = c.pose_classes;

  auto add = [&](const std::string& name, int h, int w, int depth) {
    BoxDecodeTensorStaticContract t;
    t.logical_name = name;
    t.backend_name = name;
    t.data_type = c.dtype;
    t.layout = "HWC";
    // Padding models align_channels(): storage rounds up to a 16-channel stripe while
    // slice_shape keeps the logical depth the decoder must stride by.
    t.input_shape = {h, w, c.pad_c16 ? align_c16(depth) : depth};
    t.slice_shape = {h, w, depth};
    t.source_storage_kind = c.storage;
    t.source_logical_output_index = static_cast<int>(contract.tensors.size());
    t.source_output_slot = t.source_logical_output_index;
    t.source_physical_index = t.source_logical_output_index;
    contract.tensors.push_back(std::move(t));
  };

  auto head_name = [&](const char* role, std::size_t level, std::size_t index) {
    return c.descriptive_names ? std::string(role) + "_" + std::to_string(level)
                               : "output_" + std::to_string(index);
  };

  if (c.role_major) {
    for (std::size_t role = 0; role < roles.size(); ++role) {
      for (std::size_t level = 0; level < levels; ++level) {
        add(head_name(roles[role].first, level, contract.tensors.size()), c.grid_h[level],
            c.grid_w[level], roles[role].second);
      }
    }
  } else {
    for (std::size_t level = 0; level < levels; ++level) {
      for (std::size_t role = 0; role < roles.size(); ++role) {
        add(head_name(roles[role].first, level, contract.tensors.size()), c.grid_h[level],
            c.grid_w[level], roles[role].second);
      }
    }
  }
  add(c.descriptive_names ? "mask_proto" : "output_" + std::to_string(contract.tensors.size()), 160,
      160, c.heads.proto);

  if (c.tensor_count_override > 0 &&
      static_cast<std::size_t>(c.tensor_count_override) < contract.tensors.size()) {
    contract.tensors.resize(static_cast<std::size_t>(c.tensor_count_override));
  }
  for (const auto& tensor : contract.tensors) {
    contract.tensor_names.push_back(tensor.logical_name);
  }
  return contract;
}

// Drive one case to the compiled payload the runtime consumes, on the requested route.
void check(const Case& c) {
  const BoxDecodeStaticContract contract = build_contract(c);
  const std::string label = "[" + c.name + "] ";

  if (!c.expect_error.empty()) {
    std::string what;
    try {
      if (c.model_managed) {
        BoxDecodeStaticContract managed = contract;
        apply_yolox_seg_pose_model_managed_contract_defaults(&managed);
      } else {
        finalize_boxdecode_static_contract(contract, BoxDecodeType::YoloXSegPose, std::nullopt,
                                           std::nullopt, BoxDecodeTypeOption::Auto, 0.30, 0.60, 100,
                                           /*num_classes=*/0, {});
      }
    } catch (const std::invalid_argument& e) {
      what = e.what();
    }
    require(!what.empty(), label + "must be rejected before lowering");
    require_contains(what, c.expect_error, label + "rejected for the wrong reason");
    return;
  }

  BoxDecodeStaticContract resolved;
  if (c.model_managed) {
    resolved = contract;
    apply_yolox_seg_pose_model_managed_contract_defaults(&resolved);
  } else {
    resolved =
        finalize_boxdecode_static_contract(contract, BoxDecodeType::YoloXSegPose, std::nullopt,
                                           std::nullopt, BoxDecodeTypeOption::Auto, 0.30, 0.60, 100,
                                           /*num_classes=*/0, {});
  }

  require(resolved.decode_type == BoxDecodeType::YoloXSegPose,
          label + "family type must be stable");
  require(resolved.num_classes == c.heads.classes, label + "class count must resolve to " +
                                                       std::to_string(c.heads.classes) + ", got " +
                                                       std::to_string(resolved.num_classes));
  require(resolved.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
          label + "layout must normalize to grouped-by-role-logit");
  require(resolved.score_activation == BoxDecodeScoreActivation::Sigmoid,
          label + "activation must normalize to sigmoid");

  // The compiled payload is what reaches the on-device decoder.
  const auto compiled = build_boxdecode_compiled_contract(resolved);
  require(compiled.payload.decode_type == BoxDecodeType::YoloXSegPose,
          label + "compiled family type mismatch");
  require(compiled.payload.num_classes == c.heads.classes, label + "compiled class count mismatch");
  require(compiled.payload.decode_type_option.has_value() &&
              *compiled.payload.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
          label + "compiled layout mismatch");
  require(compiled.payload.score_activation == BoxDecodeScoreActivation::Sigmoid,
          label + "compiled activation mismatch");
  require(compiled.payload.input_dtype == c.dtype, label + "compiled input dtype must be " +
                                                       c.dtype + ", got " +
                                                       compiled.payload.input_dtype);
  const std::vector<int>& expected_gate =
      c.expect_pose_classes.empty() ? c.pose_classes : c.expect_pose_classes;
  require(compiled.payload.pose_classes == expected_gate,
          label + "compiled pose-class gate mismatch");
  require(compiled.payload.slice_shapes.size() == contract.tensors.size(),
          label + "compiled payload must carry one slice shape per head");
}

} // namespace

RUN_TEST(
    "unit_boxdecode_yolox_seg_pose_contract_matrix_test", ([] {
      std::vector<Case> cases;

      // Baseline, then the same geometry with every head name stripped of its role.
      cases.push_back({.name = "role-major/named"});
      cases.push_back({.name = "role-major/generic", .descriptive_names = false});
      cases.push_back({.name = "head-major/named", .role_major = false});
      cases.push_back(
          {.name = "head-major/generic", .role_major = false, .descriptive_names = false});

      // Supported MLA dtypes and source storage kinds.
      for (const char* dtype : {"BF16", "INT8", "FP32", "UINT8"}) {
        cases.push_back({.name = std::string("dtype/") + dtype, .dtype = dtype});
      }
      cases.push_back(
          {.name = "storage/packed-cblock", .storage = BoxDecodeSourceStorageKind::PackedCBlock});
      cases.push_back(
          {.name = "storage/packed-hwc-c16", .storage = BoxDecodeSourceStorageKind::PackedHwcC16});

      // C16 padding: the padded extent must never be mistaken for the logical depth,
      // on either name set, since only the generic one reads depths positionally.
      cases.push_back({.name = "padding/c16-named", .pad_c16 = true});
      cases.push_back({.name = "padding/c16-generic", .descriptive_names = false, .pad_c16 = true});

      // Channel-slice boundaries around a 16-channel stripe. classes+1 lands exactly on
      // 16 and 32, and one past each.
      for (const int classes : {15, 16, 31, 32}) {
        cases.push_back({.name = "stripe-boundary/classes-" + std::to_string(classes),
                         .heads = {.classes = classes},
                         .descriptive_names = false,
                         .pad_c16 = true});
      }

      // Class and keypoint counts, including the single-class edge.
      for (const int classes : {1, 36, 79}) {
        cases.push_back(
            {.name = "classes/" + std::to_string(classes), .heads = {.classes = classes}});
      }
      for (const int keypoints : {13, 17}) {
        cases.push_back(
            {.name = "keypoints/" + std::to_string(keypoints), .heads = {.keypoints = keypoints}});
      }

      // Non-square feature levels.
      cases.push_back(
          {.name = "geometry/non-square", .grid_h = {80, 40, 20}, .grid_w = {60, 30, 15}});
      cases.push_back({.name = "geometry/non-square-generic",
                       .descriptive_names = false,
                       .grid_h = {80, 40, 20},
                       .grid_w = {60, 30, 15}});

      // Mixed pose/non-pose classes, on both routes.
      cases.push_back({.name = "gate/subset", .pose_classes = {0, 5, 35}});
      cases.push_back({.name = "gate/single", .pose_classes = {7}});
      // An unsorted gate must reach the backend ascending, as the ABI field documents.
      cases.push_back(
          {.name = "gate/unsorted", .pose_classes = {35, 0, 5}, .expect_pose_classes = {0, 5, 35}});

      // The model-managed route must land on the same answers as the standalone one.
      cases.push_back({.name = "model-managed/named", .model_managed = true});
      cases.push_back(
          {.name = "model-managed/generic", .descriptive_names = false, .model_managed = true});
      cases.push_back(
          {.name = "model-managed/gate", .model_managed = true, .pose_classes = {2, 9}});

      // Malformed contracts must fail before lowering rather than reach the backend.
      // Geometry that defeats positional inference must surface the explicit-num_classes
      // diagnostic, not a downstream symptom.
      const std::string kNeedsCount = "requires an explicit num_classes";
      cases.push_back({.name = "malformed/short-tensor-count",
                       .descriptive_names = false,
                       .tensor_count_override = 12,
                       .expect_error = kNeedsCount});
      cases.push_back({.name = "malformed/mask-coeff-depth",
                       .heads = {.mask_coeff = 31},
                       .descriptive_names = false,
                       .expect_error = kNeedsCount});
      cases.push_back({.name = "malformed/proto-depth",
                       .heads = {.proto = 31},
                       .descriptive_names = false,
                       .expect_error = kNeedsCount});
      cases.push_back({.name = "malformed/bbox-depth",
                       .heads = {.bbox = 6},
                       .descriptive_names = false,
                       .expect_error = kNeedsCount});
      cases.push_back({.name = "malformed/kpt-depth",
                       .heads = {.keypoints = 0},
                       .descriptive_names = false,
                       .expect_error = kNeedsCount});
      cases.push_back({.name = "malformed/gate-out-of-range",
                       .pose_classes = {36},
                       .expect_error = "must lie in [0, 36)"});
      cases.push_back({.name = "malformed/gate-duplicate",
                       .pose_classes = {3, 3},
                       .expect_error = "must not repeat a class index"});
      cases.push_back({.name = "malformed/gate-negative",
                       .pose_classes = {-1},
                       .expect_error = "must lie in [0, 36)"});

      for (const auto& c : cases) {
        check(c);
      }
      std::printf("[yolox-seg-pose-matrix] %zu cases OK\n", cases.size());
    }));

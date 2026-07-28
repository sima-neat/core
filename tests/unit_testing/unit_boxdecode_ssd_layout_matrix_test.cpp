#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Finalize an SSD300 contract whose loc heads match the recipe exactly, with the given
// conf-head depths, layout option and caller num_classes, and require it to fail fast.
void require_ssd_contract_rejected(const std::vector<int>& conf_channels,
                                   simaai::neat::BoxDecodeTypeOption option, int num_classes,
                                   const std::string& why) {
  using namespace simaai::neat;
  using namespace simaai::neat::pipeline_internal::sima;
  using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

  const std::vector<int> feat = {38, 19, 10, 5, 3, 1};
  const std::vector<int> priors = {4, 6, 6, 6, 4, 4};
  auto head = [](int side, int c) {
    BoxDecodeTensorStaticContract t;
    t.input_shape = {side, side, c};
    t.data_type = "BF16";
    t.layout = "HWC";
    return t;
  };

  BoxDecodeStaticContract contract;
  for (std::size_t i = 0; i < feat.size(); ++i) {
    contract.tensors.push_back(head(feat[i], 4 * priors[i]));
  }
  for (std::size_t i = 0; i < feat.size(); ++i) {
    contract.tensors.push_back(head(feat[i], conf_channels[i]));
  }

  bool threw = false;
  try {
    (void)finalize_boxdecode_static_contract(contract, BoxDecodeType::Ssd, std::nullopt,
                                             std::nullopt, option, 0.40, 0.45, 200, num_classes,
                                             {"orig_width", "orig_height"});
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, why);
}

// Conf depths for a well-formed SSD300 contract (81 classes).
const std::vector<int> kValidSsd300Conf = {324, 486, 486, 486, 324, 324};

} // namespace

// Exercises the SSD box-decode contract layer: two recipes only, validated against
// the grouped loc/conf head geometry with a recipe-specific activation matching the
// internals runtime. Covers class-count inference, softmax vs sigmoid, the
// grouped-by-role default, and fail-fast rejection of other geometry.
//   SSD300          -- feats {38,19,10,5,3,1}, priors {4,6,6,6,4,4}, softmax
//   SSD-MobileNetV2 -- feats {19,10,5,3,2,1}, priors {3,6,6,6,6,6}, sigmoid
RUN_TEST("unit_boxdecode_ssd_layout_matrix_test", ([] {
           using namespace simaai::neat;
           using namespace simaai::neat::pipeline_internal::sima;
           using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

           // One detection head described purely by its HWC geometry; SSD inference is
           // name-independent so logical/backend names are intentionally left blank.
           auto head = [](int h, int w, int c) {
             BoxDecodeTensorStaticContract t;
             t.input_shape = {h, w, c};
             t.data_type = "BF16";
             t.layout = "HWC";
             return t;
           };

           struct Case {
             const char* name;
             std::vector<int> feat;   // per-level feature-map side
             std::vector<int> priors; // per-level priors-per-cell
             int num_classes;         // foreground + background
             BoxDecodeScoreActivation activation;
           };

           const std::vector<Case> cases = {
               // SSD300 (dboxes300_coco): mixed priors-per-cell, softmax scores.
               {"ssd300",
                {38, 19, 10, 5, 3, 1},
                {4, 6, 6, 6, 4, 4},
                81,
                BoxDecodeScoreActivation::Softmax},
               // SSD-MobileNetV2-COCO: reduced first level (3 priors), sigmoid scores.
               {"ssd_mobilenet_v2",
                {19, 10, 5, 3, 2, 1},
                {3, 6, 6, 6, 6, 6},
                91,
                BoxDecodeScoreActivation::Sigmoid},
           };

           for (const auto& c : cases) {
             BoxDecodeStaticContract contract;
             // Grouped by role: all localization heads first, then all confidence heads.
             for (std::size_t i = 0; i < c.feat.size(); ++i) {
               contract.tensors.push_back(head(c.feat[i], c.feat[i], 4 * c.priors[i]));
             }
             for (std::size_t i = 0; i < c.feat.size(); ++i) {
               contract.tensors.push_back(head(c.feat[i], c.feat[i], c.num_classes * c.priors[i]));
             }

             auto finalized = finalize_boxdecode_static_contract(
                 contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt,
                 BoxDecodeTypeOption::Auto, 0.40, 0.45, 200, /*num_classes=*/0,
                 {"orig_width", "orig_height"});

             require(finalized.decode_type == BoxDecodeType::Ssd,
                     std::string(c.name) + ": decode type not preserved");
             require(finalized.num_classes == c.num_classes,
                     std::string(c.name) + ": ssd num_classes inference mismatch (got " +
                         std::to_string(finalized.num_classes) + ")");
             require(finalized.score_activation == c.activation,
                     std::string(c.name) + ": ssd score activation must match the recipe");
             require(finalized.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                     std::string(c.name) + ": ssd default layout must be grouped-by-role");
             require(finalized.detection_threshold == 0.40,
                     std::string(c.name) + ": detection threshold not carried");
             require(finalized.nms_iou_threshold == 0.45,
                     std::string(c.name) + ": nms iou threshold not carried");
             require(finalized.topk == 200, std::string(c.name) + ": topk not carried");
           }

           // Fail-fast: a "generic" SSD head set that matches neither recipe must be
           // rejected at contract finalization rather than silently decoded.
           {
             BoxDecodeStaticContract contract;
             const std::vector<int> feat = {64, 32, 16, 8};
             const int priors = 6;
             const int classes = 21;
             for (int side : feat) {
               contract.tensors.push_back(head(side, side, 4 * priors));
             }
             for (int side : feat) {
               contract.tensors.push_back(head(side, side, classes * priors));
             }
             bool threw = false;
             try {
               (void)finalize_boxdecode_static_contract(contract, BoxDecodeType::Ssd, std::nullopt,
                                                        std::nullopt, BoxDecodeTypeOption::Auto,
                                                        0.40, 0.45, 200, /*num_classes=*/0,
                                                        {"orig_width", "orig_height"});
             } catch (const std::exception&) {
               threw = true;
             }
             require(threw, "unsupported/generic SSD geometry must be rejected (fail fast)");
           }

           // Right feature sizes but wrong priors-per-cell (SSD300 feats with a uniform
           // 6 priors, which is not the SSD300 {4,6,6,6,4,4} signature) must also reject.
           {
             BoxDecodeStaticContract contract;
             const std::vector<int> feat = {38, 19, 10, 5, 3, 1};
             const int priors = 6;
             const int classes = 81;
             for (int side : feat) {
               contract.tensors.push_back(head(side, side, 4 * priors));
             }
             for (int side : feat) {
               contract.tensors.push_back(head(side, side, classes * priors));
             }
             bool threw = false;
             try {
               (void)finalize_boxdecode_static_contract(contract, BoxDecodeType::Ssd, std::nullopt,
                                                        std::nullopt, BoxDecodeTypeOption::Auto,
                                                        0.40, 0.45, 200, /*num_classes=*/0,
                                                        {"orig_width", "orig_height"});
             } catch (const std::exception&) {
               threw = true;
             }
             require(threw, "SSD300 feats with wrong priors-per-cell must be rejected");
           }

           // Localization heads alone must not decide the recipe: a contract whose loc
           // heads match SSD300 but whose confidence heads carry channel counts that are
           // not num_classes * priors-per-cell has no inferable class depth, so it must
           // fail fast rather than resolve to a recipe and compile with num_classes=0.
           // One conf head not divisible by that level's priors-per-cell.
           require_ssd_contract_rejected({325, 486, 486, 486, 324, 324}, BoxDecodeTypeOption::Auto,
                                         0, "ssd conf head geometry must be validated");
           // Conf heads that disagree on the class count across levels (91 vs 81).
           require_ssd_contract_rejected({324, 486, 486, 546, 324, 324}, BoxDecodeTypeOption::Auto,
                                         0, "ssd conf class counts must agree across levels");

           // SSD pairs tensors[i] with tensors[i+levels], so only a grouped-by-role
           // layout is valid. Any other token must be rejected, not carried into the
           // payload for the runtime to act on.
           for (const auto option :
                {BoxDecodeTypeOption::InterleavedByHead,
                 BoxDecodeTypeOption::InterleavedByHeadLogit, BoxDecodeTypeOption::PackedPerHead,
                 BoxDecodeTypeOption::Split3Interleaved, BoxDecodeTypeOption::GroupedByRoleLogit}) {
             require_ssd_contract_rejected(kValidSsd300Conf, option, 0,
                                           "ssd must reject non-grouped-by-role layout options");
           }

           // The conf heads encode 81 classes: a caller may restrict below that (the
           // runtime clamps the reported range) but must not declare more.
           require_ssd_contract_rejected(kValidSsd300Conf, BoxDecodeTypeOption::Auto, 91,
                                         "ssd num_classes above the encoded depth must reject");
           {
             BoxDecodeStaticContract contract;
             const std::vector<int> feat = {38, 19, 10, 5, 3, 1};
             const std::vector<int> priors = {4, 6, 6, 6, 4, 4};
             for (std::size_t i = 0; i < feat.size(); ++i) {
               contract.tensors.push_back(head(feat[i], feat[i], 4 * priors[i]));
             }
             for (std::size_t i = 0; i < feat.size(); ++i) {
               contract.tensors.push_back(head(feat[i], feat[i], 81 * priors[i])); // infers 81
             }
             auto finalized = finalize_boxdecode_static_contract(
                 contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt,
                 BoxDecodeTypeOption::Auto, 0.40, 0.45, 200, /*num_classes=*/21,
                 {"orig_width", "orig_height"});
             require(finalized.num_classes == 21,
                     "explicit ssd num_classes below the encoded depth must be honored");
           }

           // Model-managed (subset) route: a subset that declares decode_type=ssd but
           // leaves activation/layout unset must normalize to grouped-by-role and default
           // to SSD300 softmax, guarding the MPK route that bypasses finalize.
           {
             auto ssd_head = [](int side, int channels) {
               BoxDecodeTensorStaticContract t;
               t.input_shape = {side, side, channels};
               t.slice_shape = {side, side, channels};
               t.data_type = "BF16";
               t.layout = "HWC";
               t.source_storage_kind = BoxDecodeSourceStorageKind::PackedCBlock;
               return t;
             };
             BoxDecodeStaticContract contract;
             contract.decode_type = BoxDecodeType::Ssd; // score_activation / layout left unset
             const std::vector<int> feat = {38, 19, 10, 5, 3, 1};
             const std::vector<int> priors = {4, 6, 6, 6, 4, 4};
             const int classes = 81;
             for (std::size_t i = 0; i < feat.size(); ++i) {
               contract.tensors.push_back(ssd_head(feat[i], 4 * priors[i]));
             }
             for (std::size_t i = 0; i < feat.size(); ++i) {
               contract.tensors.push_back(ssd_head(feat[i], classes * priors[i]));
             }

             const auto subset =
                 plugin_contracts::extract_boxdecode_contract_subset_from_static_contract(contract);
             require(subset.decode_type == BoxDecodeType::Ssd, "subset: decode type not preserved");
             require(subset.score_activation == BoxDecodeScoreActivation::Unknown,
                     "subset precondition: extracted score activation should be unset");
             require(!subset.decode_type_option.has_value(),
                     "subset precondition: extracted head layout should be unset");

             // Options carry only the decode type, mirroring the MPK-managed direct route.
             BoxDecodeCompiledContractOptions options;
             options.decode_type = BoxDecodeType::Ssd;
             const auto compiled = build_boxdecode_compiled_contract_from_subset(subset, options);
             require(compiled.payload.score_activation == BoxDecodeScoreActivation::Softmax,
                     "subset ssd score activation must default to softmax when unresolved");
             require(compiled.payload.decode_type_option.has_value() &&
                         *compiled.payload.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                     "subset ssd head layout must default to grouped-by-role");
           }

           // Model-managed entry point (MPK subset extractor before lowering) must apply
           // the full SSD defaults -- recipe activation + class-count inference. Uses the
           // SSD-MobileNetV2 recipe to assert the sigmoid path.
           {
             BoxDecodeStaticContract contract;
             contract.decode_type = BoxDecodeType::Ssd;
             const std::vector<int> feat = {19, 10, 5, 3, 2, 1};
             const std::vector<int> priors = {3, 6, 6, 6, 6, 6};
             const int classes = 91;
             for (std::size_t i = 0; i < feat.size(); ++i) {
               contract.tensors.push_back(head(feat[i], feat[i], 4 * priors[i]));
             }
             for (std::size_t i = 0; i < feat.size(); ++i) {
               contract.tensors.push_back(head(feat[i], feat[i], classes * priors[i]));
             }
             apply_ssd_model_managed_contract_defaults(&contract);
             require(contract.score_activation == BoxDecodeScoreActivation::Sigmoid,
                     "model-managed SSD-MobileNetV2 score activation must be sigmoid");
             require(contract.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                     "model-managed ssd layout must be grouped-by-role");
             require(contract.num_classes == classes,
                     "model-managed ssd must infer class count from loc/conf head geometry");
           }

           // Token round-trip and family classification for the SSD type.
           require(parse_box_decode_type_token("ssd").value() == BoxDecodeType::Ssd,
                   "parse_box_decode_type_token(ssd) failed");
           require(parse_box_decode_type_token("SSD").value() == BoxDecodeType::Ssd,
                   "parse_box_decode_type_token is not case-insensitive for ssd");
           require(box_decode_type_token_string(BoxDecodeType::Ssd) == "ssd",
                   "box_decode_type_token_string(Ssd) mismatch");
           require(!box_decode_type_is_yolo_family(BoxDecodeType::Ssd),
                   "Ssd must not be a yolo family");
           require(!box_decode_type_is_segmentation(BoxDecodeType::Ssd), "Ssd is not segmentation");
           require(!box_decode_type_is_pose(BoxDecodeType::Ssd), "Ssd is not pose");
           require(std::string(box_decode_type_contract_summary(BoxDecodeType::Ssd)).find("SSD") !=
                       std::string::npos,
                   "Ssd contract summary missing SSD description");
         }));

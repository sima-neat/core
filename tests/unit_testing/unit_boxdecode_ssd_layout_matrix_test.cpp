#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>
#include <vector>

// Exercises the SSD-family box-decode semantics that join YOLO/DETR/etc. in the
// BoxDecode contract layer: geometric class-count inference from grouped loc/conf
// heads, softmax score activation, and the grouped-by-role layout default. The decode
// is generic across SSD variants (feature-map count, input size, and per-level
// priors-per-cell all vary), so the matrix covers a multi-level model with mixed
// priors-per-cell and a uniform-prior model.
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
           };

           const std::vector<Case> cases = {
               // Classic 6-level SSD300-style layout with mixed priors-per-cell.
               {"ssd_six_level_mixed_priors", {38, 19, 10, 5, 3, 1}, {4, 6, 6, 6, 4, 4}, 81},
               // A different SSD variant: fewer levels, uniform priors, different class count.
               {"ssd_four_level_uniform_priors", {64, 32, 16, 8}, {6, 6, 6, 6}, 21},
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
             require(finalized.score_activation == BoxDecodeScoreActivation::Softmax,
                     std::string(c.name) + ": ssd score activation must be softmax");
             require(finalized.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                     std::string(c.name) + ": ssd default layout must be grouped-by-role");
             require(finalized.detection_threshold == 0.40,
                     std::string(c.name) + ": detection threshold not carried");
             require(finalized.nms_iou_threshold == 0.45,
                     std::string(c.name) + ": nms iou threshold not carried");
             require(finalized.topk == 200, std::string(c.name) + ": topk not carried");
           }

           // An explicit num_classes from the caller must win over geometric inference.
           {
             BoxDecodeStaticContract contract;
             contract.tensors.push_back(head(10, 10, 4 * 6));  // loc head
             contract.tensors.push_back(head(10, 10, 21 * 6)); // conf head -> infers 21
             auto finalized = finalize_boxdecode_static_contract(
                 contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt,
                 BoxDecodeTypeOption::Auto, 0.40, 0.45, 200, /*num_classes=*/21,
                 {"orig_width", "orig_height"});
             require(finalized.num_classes == 21, "explicit ssd num_classes not honored");
           }

           // Token round-trip and family classification for the new SSD type.
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

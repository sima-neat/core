#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/SsdDecodeContract.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
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
RUN_TEST(
    "unit_boxdecode_ssd_layout_matrix_test", ([] {
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
        SsdRecipeId recipe_id;
        BoxDecodeScoreActivation activation;
        SsdConfidenceChannelOrder confidence_order;
      };

      const std::vector<Case> cases = {
          // SSD300 (dboxes300_coco): mixed priors-per-cell, softmax scores.
          {"ssd300",
           {38, 19, 10, 5, 3, 1},
           {4, 6, 6, 6, 4, 4},
           81,
           SsdRecipeId::Ssd300V1,
           BoxDecodeScoreActivation::Softmax,
           SsdConfidenceChannelOrder::ClassMajorAnchors},
          // SSD-MobileNetV2-COCO: reduced first level (3 priors), sigmoid scores.
          {"ssd_mobilenet_v2",
           {19, 10, 5, 3, 2, 1},
           {3, 6, 6, 6, 6, 6},
           91,
           SsdRecipeId::SsdMobile300V1,
           BoxDecodeScoreActivation::Sigmoid,
           SsdConfidenceChannelOrder::AnchorMajorClasses},
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
            contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
            0.40, 0.45, 200, /*num_classes=*/0, {"orig_width", "orig_height"});

        require(finalized.decode_type == BoxDecodeType::Ssd,
                std::string(c.name) + ": stable SSD family token not preserved");
        require(finalized.ssd_recipe_id == c.recipe_id,
                std::string(c.name) + ": exact SSD recipe not resolved");
        require(finalized.num_classes == c.num_classes,
                std::string(c.name) + ": ssd num_classes inference mismatch (got " +
                    std::to_string(finalized.num_classes) + ")");
        require(finalized.ssd_class_selection.encoded_count == c.num_classes &&
                    finalized.ssd_class_selection.selected_count == c.num_classes,
                std::string(c.name) + ": encoded/selected class state mismatch");
        require(finalized.score_activation == c.activation,
                std::string(c.name) + ": ssd score activation must match the recipe");
        require(finalized.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                std::string(c.name) + ": ssd default layout must be grouped-by-role");
        require(finalized.detection_threshold == 0.40,
                std::string(c.name) + ": detection threshold not carried");
        require(finalized.nms_iou_threshold == 0.45,
                std::string(c.name) + ": nms iou threshold not carried");
        require(finalized.topk == 200, std::string(c.name) + ": topk not carried");

        const auto* descriptor = find_ssd_recipe_descriptor(c.recipe_id);
        require(descriptor != nullptr, std::string(c.name) + ": descriptor missing");
        require(descriptor->activation == c.activation,
                std::string(c.name) + ": descriptor activation mismatch");
        require(descriptor->required_resize == ResizeMode::Stretch,
                std::string(c.name) + ": descriptor must require stretch resize");
        require(descriptor->background_class == 0,
                std::string(c.name) + ": descriptor background class mismatch");
        require(descriptor->confidence_order == c.confidence_order,
                std::string(c.name) + ": descriptor confidence order mismatch");
        require(descriptor->encoded_class_count == c.num_classes,
                std::string(c.name) + ": descriptor encoded class count mismatch");
      }

      // Prepared MLA tensors may have padded physical channel storage. Resolution must use
      // the logical sliced depth while still requiring the exact logical H/W/C signature.
      {
        auto padded_head = [](int side, int physical_channels, int logical_channels) {
          BoxDecodeTensorStaticContract tensor;
          tensor.input_shape = {side, side, physical_channels};
          tensor.slice_shape = {side, side, logical_channels};
          tensor.data_type = "BF16";
          tensor.layout = "HWC";
          return tensor;
        };
        BoxDecodeStaticContract contract;
        const std::vector<int> feat = {38, 19, 10, 5, 3, 1};
        const std::vector<int> priors = {4, 6, 6, 6, 4, 4};
        for (std::size_t i = 0; i < feat.size(); ++i) {
          contract.tensors.push_back(padded_head(feat[i], 32, 4 * priors[i]));
        }
        for (std::size_t i = 0; i < feat.size(); ++i) {
          const int logical_channels = 81 * priors[i];
          const int physical_channels = logical_channels == 324 ? 384 : 512;
          contract.tensors.push_back(padded_head(feat[i], physical_channels, logical_channels));
        }
        const auto finalized = finalize_boxdecode_static_contract(
            contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
            0.40, 0.45, 200, /*num_classes=*/0, {"orig_width", "orig_height"});
        require(finalized.decode_type == BoxDecodeType::Ssd &&
                    finalized.ssd_recipe_id == SsdRecipeId::Ssd300V1,
                "logical sliced channels must resolve padded SSD300 storage");

        // Sliced H/W are packed-storage tile dimensions, not the logical feature grid. Mirror the
        // captured SSD300 stripe layout and prove exact matching still uses input_shape H/W while
        // slice_shape C removes physical channel padding.
        auto striped_storage = contract;
        for (auto& tensor : striped_storage.tensors) {
          tensor.slice_shape[0] = 1;
          tensor.slice_shape[1] = std::min(tensor.input_shape[1], 19);
        }
        const auto striped = finalize_boxdecode_static_contract(
            striped_storage, BoxDecodeType::Ssd, std::nullopt, std::nullopt,
            BoxDecodeTypeOption::Auto, 0.40, 0.45, 200, /*num_classes=*/0,
            {"orig_width", "orig_height"});
        require(striped.ssd_recipe_id == SsdRecipeId::Ssd300V1,
                "captured stripe H/W must not replace the full SSD feature grid");

        auto invalid_sliced_depth = contract;
        invalid_sliced_depth.tensors[0].slice_shape.back() =
            invalid_sliced_depth.tensors[0].input_shape.back() + 1;
        bool invalid_depth_rejected = false;
        try {
          (void)finalize_boxdecode_static_contract(
              invalid_sliced_depth, BoxDecodeType::Ssd, std::nullopt, std::nullopt,
              BoxDecodeTypeOption::Auto, 0.40, 0.45, 200, /*num_classes=*/0,
              {"orig_width", "orig_height"});
        } catch (const std::exception&) {
          invalid_depth_rejected = true;
        }
        require(invalid_depth_rejected, "sliced channel depth beyond physical storage must reject");
      }

      // An MPK cannot override a profile's fixed score domain. In particular, an SSD300
      // signature paired with sigmoid must fail instead of being silently rewritten.
      {
        BoxDecodeStaticContract contract;
        contract.score_activation = BoxDecodeScoreActivation::Sigmoid;
        const std::vector<int> feat = {38, 19, 10, 5, 3, 1};
        const std::vector<int> priors = {4, 6, 6, 6, 4, 4};
        for (std::size_t i = 0; i < feat.size(); ++i) {
          contract.tensors.push_back(head(feat[i], feat[i], 4 * priors[i]));
        }
        for (std::size_t i = 0; i < feat.size(); ++i) {
          contract.tensors.push_back(head(feat[i], feat[i], 81 * priors[i]));
        }
        bool threw = false;
        try {
          (void)finalize_boxdecode_static_contract(
              contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
              0.40, 0.45, 200, /*num_classes=*/0, {"orig_width", "orig_height"});
        } catch (const std::exception& e) {
          threw = true;
          require(std::string(e.what()).find("activation conflicts") != std::string::npos,
                  "fixed SSD score-domain conflict must be explicit");
        }
        require(threw, "SSD300 must reject an explicit sigmoid score domain");
      }

      // A concrete profile request validates against only that profile; it cannot silently
      // resolve to another SSD member even when the observed heads are otherwise supported.
      {
        BoxDecodeStaticContract contract;
        const std::vector<int> feat = {38, 19, 10, 5, 3, 1};
        const std::vector<int> priors = {4, 6, 6, 6, 4, 4};
        for (std::size_t i = 0; i < feat.size(); ++i) {
          contract.tensors.push_back(head(feat[i], feat[i], 4 * priors[i]));
        }
        for (std::size_t i = 0; i < feat.size(); ++i) {
          contract.tensors.push_back(head(feat[i], feat[i], 81 * priors[i]));
        }
        bool threw = false;
        contract.ssd_recipe_id = SsdRecipeId::SsdMobile300V1;
        try {
          (void)finalize_boxdecode_static_contract(
              contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
              0.40, 0.45, 200, /*num_classes=*/0, {"orig_width", "orig_height"});
        } catch (const std::exception& e) {
          threw = true;
          require(std::string(e.what()).find("does not match requested profile") !=
                      std::string::npos,
                  "concrete-profile mismatch must name the requested-profile conflict");
        }
        require(threw, "SSD300 heads must not satisfy an explicit Mobile300 request");
      }

      // Exact order is part of the runtime contract. Swapping two complete levels must fail;
      // validation must never sort a correct set into a supported-looking signature.
      {
        BoxDecodeStaticContract contract;
        const std::vector<int> feat = {19, 38, 10, 5, 3, 1};
        const std::vector<int> priors = {6, 4, 6, 6, 4, 4};
        for (std::size_t i = 0; i < feat.size(); ++i) {
          contract.tensors.push_back(head(feat[i], feat[i], 4 * priors[i]));
        }
        for (std::size_t i = 0; i < feat.size(); ++i) {
          contract.tensors.push_back(head(feat[i], feat[i], 81 * priors[i]));
        }
        bool threw = false;
        std::string message;
        try {
          (void)finalize_boxdecode_static_contract(
              contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
              0.40, 0.45, 200, /*num_classes=*/0, {"orig_width", "orig_height"});
        } catch (const std::exception& e) {
          threw = true;
          message = e.what();
        }
        require(threw, "permuted SSD levels must be rejected");
        require(message.find("unsupported ordered head signature") != std::string::npos,
                "permuted SSD diagnostic must identify an unsupported ordered signature");
        require(message.find("levels are not reordered") != std::string::npos,
                "permuted SSD diagnostic must explain that order is authoritative");
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
          (void)finalize_boxdecode_static_contract(
              contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
              0.40, 0.45, 200, /*num_classes=*/0, {"orig_width", "orig_height"});
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
          (void)finalize_boxdecode_static_contract(
              contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
              0.40, 0.45, 200, /*num_classes=*/0, {"orig_width", "orig_height"});
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
      require_ssd_contract_rejected({325, 486, 486, 486, 324, 324}, BoxDecodeTypeOption::Auto, 0,
                                    "ssd conf head geometry must be validated");
      // Conf heads that disagree on the class count across levels (91 vs 81).
      require_ssd_contract_rejected({324, 486, 486, 546, 324, 324}, BoxDecodeTypeOption::Auto, 0,
                                    "ssd conf class counts must agree across levels");

      // SSD pairs tensors[i] with tensors[i+levels], so only a grouped-by-role
      // layout is valid. Any other token must be rejected, not carried into the
      // payload for the runtime to act on.
      for (const auto option :
           {BoxDecodeTypeOption::InterleavedByHead, BoxDecodeTypeOption::InterleavedByHeadLogit,
            BoxDecodeTypeOption::PackedPerHead, BoxDecodeTypeOption::Split3Interleaved,
            BoxDecodeTypeOption::GroupedByRoleLogit}) {
        require_ssd_contract_rejected(kValidSsd300Conf, option, 0,
                                      "ssd must reject non-grouped-by-role layout options");
      }

      // The SSD300 profile explicitly permits a contiguous prefix of its 81 encoded classes,
      // but a caller must not declare more channels than the heads encode.
      require_ssd_contract_rejected(kValidSsd300Conf, BoxDecodeTypeOption::Auto, 91,
                                    "ssd num_classes above the encoded depth must reject");
      require(ssd_encoded_num_classes(
                  BoxDecodeType::Ssd,
                  SsdClassSelection{81, 8, SsdClassSelectionKind::PrefixFromZero}, 8) == 81,
              "a second SSD prefix selection must validate against encoded depth, not the prior "
              "selected prefix");
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
            contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto,
            0.40, 0.45, 200, /*num_classes=*/21, {"orig_width", "orig_height"});
        require(finalized.num_classes == 21,
                "explicit ssd num_classes below the encoded depth must be honored");
        require(finalized.ssd_class_selection.encoded_count == 81 &&
                    finalized.ssd_class_selection.selected_count == 21 &&
                    finalized.ssd_class_selection.kind == SsdClassSelectionKind::PrefixFromZero,
                "SSD300 narrowing must preserve encoded depth and prefix selection explicitly");
      }

      // Model-managed (subset) route: normalize the complete static contract before
      // extraction so the subset carries the resolved type and activation authoritatively.
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

        apply_ssd_model_managed_contract_defaults(&contract);
        const auto subset =
            plugin_contracts::extract_boxdecode_contract_subset_from_static_contract(contract);
        require(subset.decode_type == BoxDecodeType::Ssd &&
                    subset.ssd_recipe_id == SsdRecipeId::Ssd300V1,
                "subset: resolved SSD300 recipe not preserved");
        require(subset.score_activation == BoxDecodeScoreActivation::Softmax,
                "subset: resolved SSD300 activation not preserved");
        require(subset.decode_type_option.has_value() &&
                    *subset.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                "subset: resolved SSD300 layout not preserved");

        // Options carry only the decode type, mirroring the MPK-managed direct route.
        BoxDecodeCompiledContractOptions options;
        options.decode_type = BoxDecodeType::Ssd;
        const auto compiled = build_boxdecode_compiled_contract_from_subset(subset, options);
        require(compiled.payload.score_activation == BoxDecodeScoreActivation::Softmax,
                "subset SSD300 score activation must remain softmax");
        require(compiled.payload.decode_type == BoxDecodeType::Ssd &&
                    compiled.payload.ssd_recipe_id == SsdRecipeId::Ssd300V1,
                "subset lowering must preserve the resolved SSD recipe");
        require(compiled.payload.ssd_class_selection.encoded_count == 81 &&
                    compiled.payload.ssd_class_selection.selected_count == 81,
                "subset lowering must preserve SSD encoded/selected class state");
        require(compiled.payload.decode_type_option.has_value() &&
                    *compiled.payload.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                "subset ssd head layout must default to grouped-by-role");

        auto unresolved_subset = subset;
        unresolved_subset.ssd_recipe_id = SsdRecipeId::Unknown;
        bool unresolved_rejected = false;
        try {
          (void)build_boxdecode_compiled_contract_from_subset(unresolved_subset, options);
        } catch (const std::exception& e) {
          unresolved_rejected = true;
          require(std::string(e.what()).find("unresolved") != std::string::npos,
                  "unresolved SSD subset rejection must explain the missing resolution");
        }
        require(unresolved_rejected,
                "generic SSD subset must not bypass complete shape resolution");
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
        for (auto& tensor : contract.tensors) {
          tensor.slice_shape = tensor.input_shape;
          tensor.source_storage_kind = BoxDecodeSourceStorageKind::PackedCBlock;
        }
        apply_ssd_model_managed_contract_defaults(&contract);
        require(contract.decode_type == BoxDecodeType::Ssd &&
                    contract.ssd_recipe_id == SsdRecipeId::SsdMobile300V1,
                "model-managed Mobile300 recipe must resolve");
        require(contract.score_activation == BoxDecodeScoreActivation::Sigmoid,
                "model-managed SSD-MobileNetV2 score activation must be sigmoid");
        require(contract.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
                "model-managed ssd layout must be grouped-by-role");
        require(contract.num_classes == classes,
                "model-managed ssd must infer class count from loc/conf head geometry");

        const auto subset =
            plugin_contracts::extract_boxdecode_contract_subset_from_static_contract(contract);
        BoxDecodeCompiledContractOptions narrowed;
        narrowed.decode_type = BoxDecodeType::Ssd;
        narrowed.num_classes = 8;
        bool rejected_narrowing = false;
        try {
          (void)build_boxdecode_compiled_contract_from_subset(subset, narrowed);
        } catch (const std::exception& e) {
          rejected_narrowing = true;
          require(std::string(e.what()).find("requires exactly 91") != std::string::npos,
                  "Mobile300 class-policy error must report the exact encoded count");
        }
        require(rejected_narrowing,
                "Mobile300 must reject class narrowing that has not been verified");
      }

      // Token round-trip and family classification for the SSD type.
      require(parse_box_decode_type_token("ssd").value() == BoxDecodeType::Ssd,
              "parse_box_decode_type_token(ssd) failed");
      require(parse_box_decode_type_token("SSD").value() == BoxDecodeType::Ssd,
              "parse_box_decode_type_token is not case-insensitive for ssd");
      require(box_decode_type_token_string(BoxDecodeType::Ssd) == "ssd",
              "box_decode_type_token_string(Ssd) mismatch");
      require(!parse_box_decode_type_token("ssd300-v1").has_value() &&
                  !parse_box_decode_type_token("ssd-mobile-300-v1").has_value(),
              "Core-side recipe IDs must not be accepted as runtime decode-family tokens");
      require(box_decode_type_is_ssd_family(BoxDecodeType::Ssd), "Ssd must be in the SSD family");
      require(!box_decode_type_is_yolo_family(BoxDecodeType::Ssd), "Ssd must not be a yolo family");
      require(!box_decode_type_is_segmentation(BoxDecodeType::Ssd), "Ssd is not segmentation");
      require(!box_decode_type_is_pose(BoxDecodeType::Ssd), "Ssd is not pose");
      require(std::string(box_decode_type_contract_summary(BoxDecodeType::Ssd)).find("SSD") !=
                  std::string::npos,
              "Ssd contract summary missing SSD description");
    }));

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example ssd300_contract_e2e_test.cpp
 * Device-free coverage of the SSD box-decode contract path down to the compiled
 * payload: both supported recipes and fail-fast rejection of everything else.
 *   SSD300          -- feats {38,19,10,5,3,1}, priors {4,6,6,6,4,4} -> softmax
 *   SSD-MobileNetV2 -- feats {19,10,5,3,2,1}, priors {3,6,6,6,6,6} -> sigmoid
 * Needs no model pack or dispatcher, so it is registered STRICT and must fail
 * (not skip) on any regression.
 */
#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

BoxDecodeTensorStaticContract head(int side, int channels) {
  BoxDecodeTensorStaticContract t;
  t.input_shape = {side, side, channels};
  t.slice_shape = {side, side, channels};
  t.data_type = "BF16";
  t.layout = "HWC";
  t.source_storage_kind = BoxDecodeSourceStorageKind::PackedCBlock;
  return t;
}

BoxDecodeStaticContract grouped_ssd_contract(const std::vector<int>& feats,
                                             const std::vector<int>& priors, int num_classes) {
  BoxDecodeStaticContract contract;
  contract.decode_type = BoxDecodeType::Ssd;
  for (std::size_t i = 0; i < feats.size(); ++i) {
    contract.tensors.push_back(head(feats[i], 4 * priors[i]));
  }
  for (std::size_t i = 0; i < feats.size(); ++i) {
    contract.tensors.push_back(head(feats[i], num_classes * priors[i]));
  }
  return contract;
}

// Drive one recipe all the way to the compiled payload the runtime consumes.
void check_recipe_compiles(const std::string& name, const std::vector<int>& feats,
                           const std::vector<int>& priors, int num_classes,
                           BoxDecodeScoreActivation expected_activation) {
  const BoxDecodeStaticContract contract = grouped_ssd_contract(feats, priors, num_classes);
  const auto finalized = finalize_boxdecode_static_contract(
      contract, BoxDecodeType::Ssd, std::nullopt, std::nullopt, BoxDecodeTypeOption::Auto, 0.30,
      0.60, 100, /*num_classes=*/0, {"orig_width", "orig_height"});
  require(finalized.decode_type == BoxDecodeType::Ssd, name + ": decode type must be SSD");
  require(finalized.score_activation == expected_activation,
          name + ": score activation must match the recipe contract");
  require(finalized.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
          name + ": layout must default to grouped-by-role");
  require(finalized.num_classes == num_classes, name + ": class count must be inferred");

  // The compiled payload is what reaches the on-device decoder.
  const auto compiled = build_boxdecode_compiled_contract(finalized);
  require(compiled.payload.decode_type == BoxDecodeType::Ssd,
          name + ": compiled decode type must be SSD");
  require(compiled.payload.score_activation == expected_activation,
          name + ": compiled score activation must match the recipe");
  require(compiled.payload.decode_type_option.has_value() &&
              *compiled.payload.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
          name + ": compiled layout must be grouped-by-role");
  require(compiled.payload.num_classes == num_classes,
          name + ": compiled class count must be carried");
  std::cout << "[ssd300-contract] " << name << " OK: activation="
            << (expected_activation == BoxDecodeScoreActivation::Softmax ? "softmax" : "sigmoid")
            << " classes=" << compiled.payload.num_classes << "\n";
}

void expect_rejected(const std::string& name, const std::vector<int>& feats,
                     const std::vector<int>& priors, int num_classes) {
  const BoxDecodeStaticContract contract = grouped_ssd_contract(feats, priors, num_classes);
  bool threw = false;
  try {
    (void)finalize_boxdecode_static_contract(contract, BoxDecodeType::Ssd, std::nullopt,
                                             std::nullopt, BoxDecodeTypeOption::Auto, 0.30, 0.60,
                                             100,
                                             /*num_classes=*/0, {"orig_width", "orig_height"});
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, name + ": non-recipe SSD geometry must be rejected (fail fast)");
  std::cout << "[ssd300-contract] " << name << " correctly rejected\n";
}

// Loc heads match a recipe but the paired conf heads do not carry num_classes*A.
void expect_conf_geometry_rejected(const std::string& name, const std::vector<int>& feats,
                                   const std::vector<int>& priors,
                                   const std::vector<int>& conf_channels) {
  BoxDecodeStaticContract contract;
  contract.decode_type = BoxDecodeType::Ssd;
  for (std::size_t i = 0; i < feats.size(); ++i) {
    contract.tensors.push_back(head(feats[i], 4 * priors[i]));
  }
  for (std::size_t i = 0; i < feats.size(); ++i) {
    contract.tensors.push_back(head(feats[i], conf_channels[i]));
  }
  bool threw = false;
  try {
    (void)finalize_boxdecode_static_contract(contract, BoxDecodeType::Ssd, std::nullopt,
                                             std::nullopt, BoxDecodeTypeOption::Auto, 0.30, 0.60,
                                             100,
                                             /*num_classes=*/0, {"orig_width", "orig_height"});
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, name + ": SSD conf head geometry must be validated, not assumed");
  std::cout << "[ssd300-contract] " << name << " correctly rejected\n";
}

} // namespace

int main() {
  try {
    // Both recipes must compile with their recipe-specific activation.
    check_recipe_compiles("ssd300", {38, 19, 10, 5, 3, 1}, {4, 6, 6, 6, 4, 4}, 81,
                          BoxDecodeScoreActivation::Softmax);
    check_recipe_compiles("ssd_mobilenet_v2", {19, 10, 5, 3, 2, 1}, {3, 6, 6, 6, 6, 6}, 91,
                          BoxDecodeScoreActivation::Sigmoid);

    // Model-managed (MPK subset) route must carry the same SSD300 contract.
    {
      BoxDecodeStaticContract contract =
          grouped_ssd_contract({38, 19, 10, 5, 3, 1}, {4, 6, 6, 6, 4, 4}, 81);
      apply_ssd_model_managed_contract_defaults(&contract);
      require(contract.score_activation == BoxDecodeScoreActivation::Softmax,
              "model-managed SSD300 must resolve to softmax");
      require(contract.num_classes == 81, "model-managed SSD300 must infer 81 classes");
      const auto subset =
          plugin_contracts::extract_boxdecode_contract_subset_from_static_contract(contract);
      BoxDecodeCompiledContractOptions options;
      options.decode_type = BoxDecodeType::Ssd;
      const auto compiled = build_boxdecode_compiled_contract_from_subset(subset, options);
      require(compiled.payload.decode_type_option.has_value() &&
                  *compiled.payload.decode_type_option == BoxDecodeTypeOption::GroupedByRole,
              "model-managed SSD subset layout must be grouped-by-role");
      std::cout << "[ssd300-contract] model-managed subset OK\n";
    }

    // Fail fast: generic / wrong-prior SSD head sets are rejected, not decoded.
    expect_rejected("generic_4_level", {64, 32, 16, 8}, {6, 6, 6, 6}, 21);
    expect_rejected("ssd300_wrong_priors", {38, 19, 10, 5, 3, 1}, {6, 6, 6, 6, 6, 6}, 81);

    // A recipe-shaped loc signature is not sufficient; without valid conf heads the
    // payload would reach the runtime with num_classes=0.
    expect_conf_geometry_rejected("ssd300_conf_indivisible", {38, 19, 10, 5, 3, 1},
                                  {4, 6, 6, 6, 4, 4},
                                  {81 * 4 + 1, 81 * 6, 81 * 6, 81 * 6, 81 * 4, 81 * 4});
    expect_conf_geometry_rejected("ssd300_conf_class_count_mismatch", {38, 19, 10, 5, 3, 1},
                                  {4, 6, 6, 6, 4, 4},
                                  {81 * 4, 81 * 6, 81 * 6, 91 * 6, 81 * 4, 81 * 4});

    std::cout << "[OK] ssd300_contract_e2e_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    // Strict, resource-free test: any failure is a real failure, never a skip.
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

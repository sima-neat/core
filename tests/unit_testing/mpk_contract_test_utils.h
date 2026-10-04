#pragma once

#include "pipeline/internal/sima/static_contract/MpkDecoder.h"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>

// Loads a fixture package through the production MPK entry point and keeps its contract.
inline std::optional<simaai::neat::pipeline_internal::sima::MpkContract>
load_test_mpk_contract(const std::filesystem::path& package_root, std::string* error) {
  auto loaded =
      simaai::neat::pipeline_internal::sima::static_contract::MpkDecoder::load_package_root(
          package_root, error);
  if (!loaded) {
    return std::nullopt;
  }
  return std::move(loaded->contract);
}

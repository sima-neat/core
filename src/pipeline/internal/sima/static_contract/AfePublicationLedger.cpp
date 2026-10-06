#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/AfePublicationLedger.h"

#include <array>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {

#include "pipeline/internal/sima/static_contract/generated/AfePublicationLedger.inc"

} // namespace

std::optional<AfePublicationContract>
lookup_afe_publication_contract(const std::string_view manifest_sha256) noexcept {
  for (const auto& item : kGeneratedPublicationContracts) {
    if (item.manifest_sha256 == manifest_sha256) {
      return item;
    }
  }
  return std::nullopt;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract

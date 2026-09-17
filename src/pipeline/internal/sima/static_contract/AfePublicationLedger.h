#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <optional>
#include <span>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima::static_contract {

struct AfePublicationContract {
  std::string_view manifest_sha256;
  std::span<const std::string_view> ordered_value_names;
};

// Release-owned exception ledger for stock AFE cohorts that omit both a
// PassThrough and an ordered public-output field. Generated from the checked-
// in machine-readable ledger; a digest change fails closed.
[[nodiscard]] std::optional<AfePublicationContract>
lookup_afe_publication_contract(std::string_view manifest_sha256) noexcept;

} // namespace simaai::neat::pipeline_internal::sima::static_contract

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstdint>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

/// Provenance for a tensor dtype token. Public terminal publication may trust
/// explicit/model/framework sources, but must not trust element-size-only
/// inference as semantic dtype truth.
enum class DTypeSource : std::uint8_t {
  Unknown = 0,
  ExplicitMpk,
  TypedObject,
  Alias,
  FallbackAlias,
  InferredFromSize,
  InternalContract,
};

inline const char* dtype_source_name(const DTypeSource source) {
  switch (source) {
  case DTypeSource::Unknown:
    return "unknown";
  case DTypeSource::ExplicitMpk:
    return "explicit_mpk";
  case DTypeSource::TypedObject:
    return "typed_object";
  case DTypeSource::Alias:
    return "alias";
  case DTypeSource::FallbackAlias:
    return "fallback_alias";
  case DTypeSource::InferredFromSize:
    return "inferred_from_size";
  case DTypeSource::InternalContract:
    return "internal_contract";
  }
  return "unknown";
}

inline DTypeSource dtype_source_from_name(std::string value) {
  for (char& c : value) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    } else if (c == '-') {
      c = '_';
    }
  }
  if (value == "explicit_mpk") {
    return DTypeSource::ExplicitMpk;
  }
  if (value == "typed_object") {
    return DTypeSource::TypedObject;
  }
  if (value == "alias") {
    return DTypeSource::Alias;
  }
  if (value == "fallback_alias") {
    return DTypeSource::FallbackAlias;
  }
  if (value == "inferred_from_size") {
    return DTypeSource::InferredFromSize;
  }
  if (value == "internal_contract") {
    return DTypeSource::InternalContract;
  }
  return DTypeSource::Unknown;
}

inline bool dtype_source_is_inferred_only(const DTypeSource source) {
  return source == DTypeSource::InferredFromSize;
}

} // namespace simaai::neat::pipeline_internal::sima

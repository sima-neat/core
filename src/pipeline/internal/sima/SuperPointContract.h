#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SuperPointTypes.h"

#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

enum class BoxDecodeTensorRole : int {
  Unknown = 0,
  DetectorLogits = 1,
  DescriptorGrid = 2,
};

struct SuperPointStaticContract {
  SuperPointProfile profile = SuperPointProfile::Auto;
  SuperPointOutputFormat output_format = SuperPointOutputFormat::FeaturePointsV1;
  TensorDType descriptor_output_dtype = TensorDType::Float32;
  int schema_version = 0;
  int nms_radius = -1;
  int border_margin = -1;
  int cell_stride = 8;
  int descriptor_stride = 8;
  int descriptor_dim = 0;
  std::string profile_fingerprint;
  std::string detector_tensor_id;
  std::string descriptor_tensor_id;
  std::string detector_representation;
  std::string descriptor_representation;
  bool profile_from_mpk = false;
  /// Preserve explicitly authored spatial controls when an API profile override is applied.
  bool nms_radius_from_mpk = false;
  bool border_margin_from_mpk = false;
  // Profile named by the MPK record whose fingerprint is carried above. This remains unchanged
  // when an API profile override is applied, so stale/mismatched provenance fails closed.
  SuperPointProfile fingerprint_profile = SuperPointProfile::Auto;
  /// True when a schema-0/manual contract omitted representation tokens and Neat applied the
  /// canonical raw SuperPoint input representations required by the runtime.
  bool representations_defaulted = false;
};

/** Apply an explicit profile over a resolved contract.
 *
 * A compiled contract may already contain defaults materialized for the old
 * profile. Reset only those derived values; explicitly authored MPK controls
 * retain their higher-precedence values.
 */
inline bool apply_superpoint_profile_override(SuperPointStaticContract* sp,
                                              SuperPointProfile profile) {
  if (!sp || profile == SuperPointProfile::Auto) {
    return false;
  }
  const bool changed = sp->profile != profile;
  sp->profile = profile;
  sp->profile_from_mpk = false;
  if (changed) {
    if (!sp->nms_radius_from_mpk) {
      sp->nms_radius = -1;
    }
    if (!sp->border_margin_from_mpk) {
      sp->border_margin = -1;
    }
  }
  return changed;
}

inline void apply_superpoint_spatial_overrides(SuperPointStaticContract* sp,
                                               const SuperPointOptions& requested) {
  if (!sp) {
    return;
  }
  if (requested.nms_radius >= 0) {
    sp->nms_radius = requested.nms_radius;
    sp->nms_radius_from_mpk = false;
  }
  if (requested.border_margin >= 0) {
    sp->border_margin = requested.border_margin;
    sp->border_margin_from_mpk = false;
  }
}

inline void canonicalize_schema0_superpoint_representations(SuperPointStaticContract* sp) {
  if (!sp || sp->schema_version != 0) {
    return;
  }
  if (sp->detector_representation.empty()) {
    sp->detector_representation = "raw-logits-65";
    sp->representations_defaulted = true;
  }
  if (sp->descriptor_representation.empty()) {
    sp->descriptor_representation = "coarse-pre-l2";
    sp->representations_defaulted = true;
  }
}

inline bool is_known_superpoint_profile(SuperPointProfile profile) {
  switch (profile) {
  case SuperPointProfile::Auto:
  case SuperPointProfile::LightGlueV1:
  case SuperPointProfile::MagicLeapDemoV1:
  case SuperPointProfile::PaperBicubicV1:
  case SuperPointProfile::A65V1:
    return true;
  }
  return false;
}

inline bool is_known_superpoint_output_format(SuperPointOutputFormat format) {
  switch (format) {
  case SuperPointOutputFormat::FeaturePointsV1:
  case SuperPointOutputFormat::LegacyA65InterleavedV0:
    return true;
  }
  return false;
}

inline bool is_supported_superpoint_descriptor_dtype(TensorDType dtype) {
  return dtype == TensorDType::Int8 || dtype == TensorDType::BFloat16 ||
         dtype == TensorDType::Float32;
}

/** Resolve the construction-time sentinel without guessing from tensor data.
 * Authoritative MPK metadata is applied before this helper is called, so an
 * unresolved schema-0/manual contract means that no profile was authored at
 * any higher precedence. A65V1 is the product default in that case. Schema v1
 * remains fail-closed and must carry an explicit profile.
 */
inline void resolve_default_superpoint_profile(SuperPointStaticContract* sp) {
  if (sp && sp->schema_version == 0 && sp->profile == SuperPointProfile::Auto) {
    sp->profile = SuperPointProfile::A65V1;
    sp->profile_from_mpk = false;
  }
}

inline double superpoint_default_detection_threshold(SuperPointProfile profile) {
  switch (profile) {
  case SuperPointProfile::LightGlueV1:
    return 5.0e-4;
  case SuperPointProfile::MagicLeapDemoV1:
    return 0.015;
  case SuperPointProfile::A65V1:
    return 0.1;
  case SuperPointProfile::Auto:
  case SuperPointProfile::PaperBicubicV1:
    return 0.0;
  }
  return 0.0;
}

inline double rebase_superpoint_detection_threshold(bool profile_changed, SuperPointProfile profile,
                                                    double requested_threshold,
                                                    double resolved_threshold) {
  return profile_changed && requested_threshold == 0.0
             ? superpoint_default_detection_threshold(profile)
             : resolved_threshold;
}

inline constexpr int kSuperPointDefaultTopK = 600;

inline bool is_sha256_fingerprint(std::string_view fingerprint) {
  constexpr std::string_view prefix = "sha256:";
  if (!fingerprint.starts_with(prefix) || fingerprint.size() != prefix.size() + 64U) {
    return false;
  }
  for (const unsigned char c : fingerprint.substr(prefix.size())) {
    if (std::isxdigit(c) == 0) {
      return false;
    }
  }
  return true;
}

/** Validate only authoritative metadata and provenance. Geometry and per-input facts are checked
 * by BoxDecode stage semantics after role binding. Schema 0 intentionally remains tolerant for
 * migration; schema 1 is a complete, fail-closed record. */
inline std::optional<std::string>
validate_superpoint_static_metadata(const SuperPointStaticContract& sp,
                                    bool require_resolved_profile) {
  constexpr const char* profiles = "lightglue-v1, magic-leap-demo-v1, a65-v1 "
                                   "(paper-bicubic-v1 is reserved)";
  if (!is_known_superpoint_profile(sp.profile)) {
    return std::string("unknown profile enum; supported production profiles: ") + profiles;
  }
  if (require_resolved_profile && sp.profile == SuperPointProfile::Auto) {
    return std::string("profile is unresolved; supported production profiles: ") + profiles;
  }
  if (!is_known_superpoint_output_format(sp.output_format)) {
    return "unknown output format; supported formats: feature-points-v1, "
           "legacy-a65-interleaved-v0";
  }
  if (!is_supported_superpoint_descriptor_dtype(sp.descriptor_output_dtype)) {
    return "descriptor output dtype must be INT8, BF16, or FP32";
  }
  if (sp.schema_version < 0 || sp.schema_version > 1) {
    return "unsupported MPK superpoint.schema_version; supported versions are 0 (migration) "
           "and 1";
  }
  if (!sp.profile_fingerprint.empty() && !is_sha256_fingerprint(sp.profile_fingerprint)) {
    return "profile_fingerprint must be 'sha256:' followed by exactly 64 hexadecimal digits";
  }
  if (!sp.detector_representation.empty() && sp.detector_representation != "raw-logits-65") {
    return "unsupported detector_representation '" + sp.detector_representation +
           "'; supported value is raw-logits-65";
  }
  if (!sp.descriptor_representation.empty() && sp.descriptor_representation != "coarse-pre-l2") {
    return "unsupported descriptor_representation '" + sp.descriptor_representation +
           "'; supported value is coarse-pre-l2";
  }
  if (!sp.profile_fingerprint.empty() && sp.fingerprint_profile != SuperPointProfile::Auto &&
      sp.fingerprint_profile != sp.profile) {
    return "selected profile '" + std::string(superpoint_profile_token(sp.profile)) +
           "' conflicts with an MPK fingerprint stamped for '" +
           superpoint_profile_token(sp.fingerprint_profile) +
           "'; re-stamp the MPK fingerprint for the selected profile";
  }
  if (sp.schema_version == 1) {
    if (sp.profile == SuperPointProfile::Auto) {
      return "schema v1 requires an explicit profile";
    }
    if (sp.profile_fingerprint.empty()) {
      return "schema v1 requires profile_fingerprint";
    }
    if (sp.detector_tensor_id.empty() || sp.descriptor_tensor_id.empty()) {
      return "schema v1 requires detector_tensor_id and descriptor_tensor_id";
    }
    if (sp.detector_tensor_id == sp.descriptor_tensor_id) {
      return "schema v1 detector_tensor_id and descriptor_tensor_id must be distinct";
    }
    if (sp.detector_representation.empty() || sp.descriptor_representation.empty()) {
      return "schema v1 requires detector_representation and descriptor_representation";
    }
  }
  return std::nullopt;
}

/** Merge node options over inherited Model options without treating unset spatial/profile
 * sentinels as explicit overrides. Output dtype and format have no unset sentinel and therefore
 * remain concrete node selections. */
inline SuperPointOptions merge_superpoint_node_options(const SuperPointOptions& inherited,
                                                       const SuperPointOptions& node) {
  SuperPointOptions merged = inherited;
  if (node.profile != SuperPointProfile::Auto) {
    merged.profile = node.profile;
  }
  if (node.nms_radius >= 0) {
    merged.nms_radius = node.nms_radius;
  }
  if (node.border_margin >= 0) {
    merged.border_margin = node.border_margin;
  }
  merged.descriptor_output_dtype = node.descriptor_output_dtype;
  merged.output_format = node.output_format;
  return merged;
}

} // namespace simaai::neat::pipeline_internal::sima

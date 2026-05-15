/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Subsets / projections of a SiMa plugin's
 *        full contract.
 *
 * For each per-stage kernel family (Quantize / Cast / Tessellate / QuantTess / ProcessMla /
 * Detessellate / Dequantize / DetessDequant / BoxDecode), this header exposes a small "subset"
 * struct carrying just the fields that family needs, plus extractors that pull each subset out
 * of the parent `MpkContract` (or a single stage's `MpkPluginIoContract`). The subsets feed
 * into the runtime-config builders that produce `CompiledProcessCvuRuntimeConfig` records.
 *
 * A few public wrappers (`semantic_shape_without_batch_public`, `derive_per_frame_rank_public`,
 * `inferred_batch_size_from_shape_public`) expose the per-frame normalization helpers used by
 * the extractors so out-of-TU assemblers can apply the same convention.
 *
 * @see MpkContract
 * @see SimaPluginStaticManifest
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {
struct BoxDecodeStaticContract;
struct ModelManagedRouteFlags;
} // namespace simaai::neat::pipeline_internal::sima

namespace simaai::neat::pipeline_internal::sima::plugin_contracts {

/**
 * @brief Stable identifier for one field of a plugin contract.
 *
 * Used by the contract-family declaration tables to enumerate which fields are required vs.
 * optional for a given family without baking the field name strings throughout the code.
 */
enum class PluginContractFieldKey : std::uint8_t {
  QuantParams = 0,
  InputShape,
  OutputShape,
  InputDtype,
  OutputDtype,
  RoundOff,
  SliceShape,
  FrameType,
  AlignC16,
  Cblock,
  ModelPath,
  BatchSize,
  BatchSzModel,
  DispatcherOutputSizes,
  DispatcherOutputNames,
  PerHeadInputShape,
  PerHeadQuantParams,
  FrameShape,
  LogicalInputs,
  InputBindings,
  SliceGeometry,
  RouteFlags,
  DecodeTypeOption,
  ScoreActivation,
};

/**
 * @brief Declares which fields a plugin contract family expects.
 *
 * Supplies the `family` token (e.g., `"quanttess"`), spans of required and optional fields, and
 * a flag for whether the family is per-head (e.g., DetessDequant has one entry per output head).
 */
struct PluginContractFamilyDeclaration {
  std::string_view family;                                 ///< Family identifier.
  std::span<const PluginContractFieldKey> required_fields; ///< Fields the family requires.
  std::span<const PluginContractFieldKey> optional_fields; ///< Fields the family may carry.
  bool per_head = false;                                   ///< True for per-head families.
};

/**
 * @brief QuantTess contract subset.
 *
 * @note `input_shape` and `output_shape` preserve the authored MPK/runtime tensor shapes.
 * `batch_size` is the explicit stage batch count when present; descriptor builders must not
 * reconstruct it from shape rank.
 */
struct QuantTessContractSubset {
  MpkQuantContract quant_params;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
  std::string input_layout;
  std::string input_dtype;
  std::string output_dtype;
  int round_off = -1;
  // Authored tile geometry; descriptor builders normalize rank.
  std::vector<std::int64_t> slice_shape;
  std::string frame_type;
  bool align_c16 = false;
  bool cblock = false;
  std::uint64_t output_size_bytes = 0U;
  int batch_size = 1;
};

/// Quantize stage contract subset.
struct QuantizeContractSubset {
  MpkQuantContract quant_params;
  std::vector<std::int64_t> input_shape;
  std::string input_layout;
  std::string input_dtype;
  std::string output_dtype;
  int round_off = -1;
};

/// Cast stage contract subset (dtype-only conversion).
struct CastContractSubset {
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
  std::string input_layout;
  std::string input_dtype;
  std::string output_dtype;
};

/// Tessellate stage contract subset.
struct TessellateContractSubset {
  std::vector<std::int64_t> input_shape;
  std::string input_layout;
  std::string frame_type;
  std::vector<std::int64_t> slice_shape;
  bool align_c16 = false;
  bool cblock = false;
  std::uint64_t output_size_bytes = 0U;
  int batch_size = 1;
};

/// ProcessMla stage contract subset (carries model path + dispatcher fan-out).
struct ProcessMlaContractSubset {
  std::string model_path;
  int batch_size = 0;
  int batch_sz_model = 0;
  std::vector<std::uint64_t> dispatcher_output_sizes;
  std::vector<std::string> dispatcher_output_names;
};

/// Detessellate stage contract subset.
struct DetessellateContractSubset {
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> input_transport_shape;
  std::uint64_t input_transport_size_bytes = 0U;
  std::vector<std::int64_t> frame_shape;
  std::string frame_type;
  std::vector<std::int64_t> slice_shape;
  bool align_c16 = false;
  bool cblock = false;
};

/// Dequantize stage contract subset.
struct DequantizeContractSubset {
  MpkQuantContract quant_params;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
  std::string input_dtype;
  std::string output_dtype;
};

/// Per-head subset for the fused DetessDequant contract.
struct DetessDequantHeadContractSubset {
  std::vector<std::int64_t> per_head_input_shape;
  std::vector<std::int64_t> input_transport_shape;
  std::uint64_t input_transport_size_bytes = 0U;
  MpkQuantContract per_head_quant_params;
  std::vector<std::int64_t> frame_shape;
  std::string frame_type;
  std::vector<std::int64_t> slice_shape;
  bool align_c16 = false;
  bool cblock = false;
  std::string output_dtype;
};

/// Fused DetessDequant contract subset — one entry per output head.
struct DetessDequantContractSubset {
  std::vector<DetessDequantHeadContractSubset> heads;
};

/// Pointer pair tying a Detess stage to its companion Dequant stage in the MPK.
struct DetessDequantStagePair {
  const MpkPluginIoContract* detess = nullptr;
  const MpkPluginIoContract* dequant = nullptr;
};

/// BoxDecode stage contract subset.
struct BoxDecodeContractSubset {
  std::vector<LogicalInputStaticSpec> logical_inputs;
  std::vector<InputBindingStaticSpec> input_bindings;
  std::vector<sima_ev_shape_desc> slice_shapes;
  BoxDecodeType decode_type = BoxDecodeType::Unspecified;
  bool tess_needed = false;
  bool quant_needed = false;
  std::optional<BoxDecodeTypeOption> decode_type_option;
  BoxDecodeScoreActivation score_activation = BoxDecodeScoreActivation::Unknown;
  int num_classes = 0;
};

// Phase 3a (Option A++): public wrappers around the per-frame normalization
// helpers used by all extractors in PluginContractSubsets.cpp. Out-of-TU
// subset assemblers (e.g. ProcessCvuStageSemantics.cpp) call these to apply
// the same convention: leading dims beyond the per-frame rank fold into an
// explicit batch_size scalar; the residual shape is per-frame.

/// Strip leading batch-only dims from `shape`; residual is per-frame at `per_frame_rank`.
std::vector<std::int64_t> semantic_shape_without_batch_public(std::vector<std::int64_t> shape,
                                                              int per_frame_rank = 3);

/// Derive the per-frame rank from a slice-shape hint and a peer's per-frame shape.
int derive_per_frame_rank_public(const std::vector<std::int64_t>& slice_shape_hint,
                                 const std::vector<std::int64_t>& peer_per_frame_shape);

/// Infer the implicit batch size by collapsing leading dims beyond `per_frame_rank` of `shape`.
int inferred_batch_size_from_shape_public(const std::vector<std::int64_t>& shape,
                                          int per_frame_rank = 3);

/// True when two shapes describe the same element order and differ only by size-1 axes.
bool unit_axis_shape_alias_public(const std::vector<std::int64_t>& lhs,
                                  const std::vector<std::int64_t>& rhs);

/// Pick the runtime descriptor shape for elementwise/value transforms.
std::vector<std::int64_t> canonical_value_transform_shape_public(
    std::string_view family, const std::vector<std::int64_t>& input_shape,
    const std::vector<std::int64_t>& output_shape,
    const std::vector<std::int64_t>& preferred_semantic_shape = {});

/// Stable string name for a `PluginContractFieldKey` (e.g., `"input_shape"`).
std::string_view plugin_contract_field_key_name(PluginContractFieldKey key);

/// Parse a string back into a `PluginContractFieldKey`; nullopt if unknown.
std::optional<PluginContractFieldKey> plugin_contract_field_key_from_name(std::string_view name);

/// Look up the family declaration table entry for `family`.
const PluginContractFamilyDeclaration& plugin_contract_family_declaration(std::string_view family);

/// Extract the (single) Quantize subset from an MPK that contains a quantize stage.
QuantizeContractSubset extract_quantize_contract_subset_from_mpk(const MpkContract& contract);
/// Extract a Quantize subset directly from one stage's I/O contract.
QuantizeContractSubset
extract_quantize_contract_subset_from_stage(const MpkPluginIoContract& stage);

/// Extract all Cast subsets from an MPK (one per cast stage).
std::vector<CastContractSubset> extract_cast_contract_subsets_from_mpk(const MpkContract& contract);
/// Extract a Cast subset from one stage's I/O contract.
CastContractSubset extract_cast_contract_subset_from_stage(const MpkPluginIoContract& stage);

/// Extract the Tessellate subset from an MPK that contains a tessellate stage.
TessellateContractSubset extract_tessellate_contract_subset_from_mpk(const MpkContract& contract);
/// Extract a Tessellate subset from one stage's I/O contract.
TessellateContractSubset
extract_tessellate_contract_subset_from_stage(const MpkPluginIoContract& stage);

/// Extract the QuantTess subset from an MPK with a fused quanttess stage.
QuantTessContractSubset extract_quanttess_contract_subset_from_mpk(const MpkContract& contract);
/// Extract a QuantTess subset from one stage's I/O contract.
QuantTessContractSubset
extract_quanttess_contract_subset_from_stage(const MpkPluginIoContract& stage);

/**
 * @brief Extract a ProcessMla subset from an `MlaStaticContract`.
 *
 * @param contract                       Source MLA contract.
 * @param include_dispatcher_output_names When true, also fills `dispatcher_output_names`.
 */
ProcessMlaContractSubset extract_processmla_contract_subset_from_static_contract(
    const MlaStaticContract& contract, bool include_dispatcher_output_names = false);

/// Extract all Detessellate subsets from an MPK (one per detess stage).
std::vector<DetessellateContractSubset>
extract_detessellate_contract_subsets_from_mpk(const MpkContract& contract);

/// Extract all Dequantize subsets from an MPK (one per dequant stage).
std::vector<DequantizeContractSubset>
extract_dequantize_contract_subsets_from_mpk(const MpkContract& contract);

/// Resolve all Detess+Dequant pairs in an MPK that participate in a fused detessdequant.
std::vector<DetessDequantStagePair>
resolve_detessdequant_stage_pairs_from_mpk(const MpkContract& contract);

/// Extract the fused DetessDequant subset (per-head) from an MPK.
DetessDequantContractSubset
extract_detessdequant_contract_subset_from_mpk(const MpkContract& contract);

/// Build a BoxDecode subset from an already-built `BoxDecodeStaticContract`.
BoxDecodeContractSubset
extract_boxdecode_contract_subset_from_static_contract(const BoxDecodeStaticContract& contract);

/**
 * @brief Build a BoxDecode subset directly from MPK + route flags.
 *
 * @param contract       Parsed MPK contract.
 * @param route_flags    Route flags computed from the MPK (see
 * `BoxDecodeStaticContractExtractor.h`).
 * @param terminal_stage Optional explicit box-decode terminal stage.
 * @param error_message  Optional out-parameter for failure diagnostics.
 */
std::optional<BoxDecodeContractSubset> extract_boxdecode_contract_subset_from_mpk(
    const MpkContract& contract, const ModelManagedRouteFlags& route_flags,
    const MpkPluginIoContract* terminal_stage = nullptr, std::string* error_message = nullptr);

/// Validate a BoxDecode subset; throws `std::runtime_error` on contract violations.
void validate_boxdecode_contract_subset(const BoxDecodeContractSubset& subset,
                                        const std::string& stage_name = "boxdecode");

/// Build a runtime config for a standalone Quantize stage.
stagesemantics::CompiledProcessCvuRuntimeConfig build_quantize_runtime_config_from_subset(
    const QuantizeContractSubset& subset, const std::string& physical_output_name,
    const std::string& published_output_name = "output_tensor");

/// Build a runtime config for the Cast+Tessellate fused path.
stagesemantics::CompiledProcessCvuRuntimeConfig build_tessellate_runtime_config_from_subsets(
    const CastContractSubset& cast_subset, const TessellateContractSubset& tess_subset,
    const std::string& physical_output_name,
    const std::string& published_output_name = "output_tensor");

/// Build a runtime config for a standalone Cast stage.
stagesemantics::CompiledProcessCvuRuntimeConfig
build_cast_runtime_config_from_subset(const CastContractSubset& subset,
                                      const std::string& physical_output_name,
                                      const std::string& published_output_name = "output_tensor");

/// Build a runtime config for one or more standalone Cast output heads.
stagesemantics::CompiledProcessCvuRuntimeConfig
build_cast_runtime_config_from_subsets(const std::vector<CastContractSubset>& subsets,
                                       const std::vector<std::string>& published_output_names = {});

/// Build a runtime config for the fused QuantTess stage.
stagesemantics::CompiledProcessCvuRuntimeConfig build_quanttess_runtime_config_from_subset(
    const QuantTessContractSubset& subset,
    const std::string& published_output_name = "output_tensor");

/// Construct a `ProcessMlaStagePayload` from a ProcessMla subset.
ProcessMlaStagePayload build_processmla_payload_from_subset(const ProcessMlaContractSubset& subset);

/// Build a runtime config for a Detessellate stage (one or more output heads).
stagesemantics::CompiledProcessCvuRuntimeConfig build_detessellate_runtime_config_from_subsets(
    const std::vector<DetessellateContractSubset>& subsets,
    const std::vector<std::string>& runtime_output_names = {},
    const std::vector<std::string>& published_output_names = {});

/// Build a runtime config for a Dequantize stage (one or more output heads).
stagesemantics::CompiledProcessCvuRuntimeConfig build_dequantize_runtime_config_from_subsets(
    const std::vector<DequantizeContractSubset>& subsets,
    const std::vector<std::string>& published_output_names = {});

/// Build a runtime config for the fused DetessDequant stage.
stagesemantics::CompiledProcessCvuRuntimeConfig build_detessdequant_runtime_config_from_subset(
    const DetessDequantContractSubset& subset,
    const std::vector<std::string>& runtime_output_names = {},
    const std::vector<std::string>& published_output_names = {});

} // namespace simaai::neat::pipeline_internal::sima::plugin_contracts

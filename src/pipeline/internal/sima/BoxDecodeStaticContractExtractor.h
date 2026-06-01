/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Static-contract extractor for the
 *        SimaBoxDecode postprocess stage.
 *
 * Pulls box-decode kernel metadata out of various sources (parsed MPK, raw tensor list,
 * `OutputSpec`, `Sample`, or a compiled upstream contract) and produces a typed
 * `BoxDecodeStaticContract` plus the `ModelManagedRouteFlags` describing what pre-stages
 * (quant/tess/cast) the route planner must include around the box-decode terminal.
 *
 * @see BoxDecodeType (public enum)
 * @see MpkContract
 * @see MlaStaticContractExtractor.h (companion for the MLA stage)
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/BoxDecodeType.h"
#include "pipeline/GraphOptions.h"
#include "builder/InputContractConfigurable.h"
#include "builder/OutputSpec.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

/**
 * @brief One input tensor seen by the box-decode stage.
 *
 * Carries the shape/dtype/layout the box-decoder expects plus the upstream identity
 * (logical/backend names, source segment, slot, physical offset/size) needed to bind to it.
 */
struct BoxDecodeTensorStaticContract {
  std::vector<int> input_shape;         ///< Full per-frame input tensor shape.
  std::vector<int> slice_shape;         ///< Tile/slice shape if the tensor is tessellated.
  std::string data_type;                ///< Tensor dtype token.
  std::string layout;                   ///< Tensor layout token (CHW / HWC / HW).
  std::string logical_name;             ///< Logical tensor name.
  std::string backend_name;             ///< Backend (compiled artifact) tensor name.
  std::string source_segment_name;      ///< Source plugin's segment name.
  int source_logical_output_index = -1; ///< Source plugin's logical-output index.
  int source_output_slot = -1;          ///< Source plugin's output slot.
  int source_physical_index = -1;       ///< Source plugin's physical-output index.
  std::int64_t source_byte_offset = 0;  ///< Byte offset within the source physical buffer.
  std::uint64_t source_size_bytes = 0;  ///< Size in bytes within the source physical buffer.
};

/// Description of one physical input buffer feeding the box-decode stage.
struct BoxDecodePhysicalInputStaticContract {
  std::string name;             ///< Physical buffer / segment name.
  int physical_index = -1;      ///< Physical-input index on the box-decode stage.
  std::int64_t byte_offset = 0; ///< Byte offset within that buffer.
  std::uint64_t size_bytes = 0; ///< Size in bytes.
};

/**
 * @brief Full static contract for one box-decode stage.
 *
 * Aggregates the box-decode kernel parameters (decode type, score activation, NMS thresholds,
 * topK, num_classes) together with all tensor inputs, physical inputs, and the dequantization
 * params propagated from upstream when applicable.
 */
struct BoxDecodeStaticContract {
  BoxDecodeType decode_type = BoxDecodeType::Unspecified; ///< Box-decode flavor (YOLO/SSD/etc.).
  BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto; ///< Decode-type sub-option.
  BoxDecodeScoreActivation score_activation =
      BoxDecodeScoreActivation::Unknown; ///< Score activation kind.
  std::string input_dtype;               ///< Dtype the kernel reads its inputs as.
  bool tess_needed = false;       ///< True if a tessellate stage must precede this box-decode.
  bool quant_needed = false;      ///< True if a quantize stage must precede.
  bool model_owned_flags = false; ///< True if route flags came from the MPK (vs. user override).
  bool quant_contract_required = false; ///< True if dq_scale / dq_zp must be present.
  int topk = 0;                         ///< Max detections retained.
  double detection_threshold = 0.0;     ///< Score cutoff before NMS.
  double nms_iou_threshold = 0.0;       ///< IoU threshold used by NMS.
  int num_classes = 0;                  ///< Number of class scores per anchor.

  std::vector<BoxDecodeTensorStaticContract> tensors; ///< Per-input tensor specs.
  std::vector<std::string> tensor_names; ///< Logical tensor names (parallel to `tensors`).
  std::vector<BoxDecodePhysicalInputStaticContract> physical_inputs; ///< Physical input buffers.
  std::vector<double> dq_scale;    ///< Dequant scale per output (if quant).
  std::vector<std::int64_t> dq_zp; ///< Dequant zero-point per output (if quant).
  std::vector<std::string>
      required_preprocess_meta_fields; ///< Preprocess meta fields the box-decoder needs.
};

/**
 * @brief Coarse semantics extracted directly from MPK box-decode metadata.
 *
 * Lighter-weight than `BoxDecodeStaticContract`; used as input when computing the route flags.
 */
struct ModelBoxdecodeSemantics {
  bool tess_needed = false;             ///< Box-decode needs a tessellate predecessor.
  bool quant_needed = false;            ///< Box-decode needs a quantize predecessor.
  bool quant_contract_required = false; ///< Box-decode needs explicit quant params from upstream.
};

/**
 * @brief Flags describing what model-managed pre-stages the planner must insert.
 *
 * Used by the route planner to materialize the right sequence of cast/quant/tess stages
 * before a box-decode terminal (or before the MLA when applicable).
 */
struct ModelManagedRouteFlags {
  bool quant_needed = false;            ///< Must include a quantize stage.
  bool tess_needed = false;             ///< Must include a tessellate stage.
  bool pre_cast_needed = false;         ///< Must include a pre-cast stage.
  bool quant_contract_required = false; ///< Quant params must be propagated through.
  bool include_pre_stage =
      false; ///< Synthesize at least one pre-stage even if individually disabled.
  bool boxdecode_selected = false; ///< Indicates a box-decode terminal is part of the route.
};

/// Derive the model-managed route flags from a fully-built `BoxDecodeStaticContract`.
ModelManagedRouteFlags
model_route_flags_from_boxdecode_contract(const BoxDecodeStaticContract& contract);

/// Derive the model-managed route flags from coarse box-decode semantics.
ModelManagedRouteFlags
model_route_flags_from_boxdecode_semantics(const ModelBoxdecodeSemantics& semantics);

/**
 * @brief Resolve route flags by inspecting the MPK contract.
 *
 * @param contract       Parsed MPK contract.
 * @param terminal_stage Optional explicit terminal stage; defaults to the auto-detected one.
 * @param error_message  Optional out-parameter populated on failure.
 * @return Resolved flags, or `std::nullopt` if the MPK lacks the required metadata.
 */
std::optional<ModelManagedRouteFlags> resolve_model_managed_boxdecode_route_flags_from_mpk(
    const MpkContract& contract, const MpkPluginIoContract* terminal_stage = nullptr,
    std::string* error_message = nullptr);

/// Build a box-decode static contract from an MPK with auto-detected terminal stage.
std::optional<BoxDecodeStaticContract>
build_boxdecode_static_contract_from_mpk(const MpkContract& contract,
                                         const ModelManagedRouteFlags& route_flags,
                                         std::string* error_message = nullptr);

/// Build a box-decode static contract from an MPK with an explicit terminal stage.
std::optional<BoxDecodeStaticContract> build_boxdecode_static_contract_from_mpk(
    const MpkContract& contract, const ModelManagedRouteFlags& route_flags,
    const MpkPluginIoContract* terminal_stage, std::string* error_message);

/// Build a box-decode static contract from a raw `TensorList` plus optional input contract.
std::optional<BoxDecodeStaticContract> build_boxdecode_static_contract_from_tensors(
    const TensorList& tensors, BoxDecodeType decode_type,
    const std::optional<InputContract>& input_contract = {}, std::string* error_message = nullptr);

/// Build a box-decode static contract from a builder-side `OutputSpec`.
std::optional<BoxDecodeStaticContract> build_boxdecode_static_contract_from_output_spec(
    const OutputSpec& spec, BoxDecodeType decode_type,
    const std::optional<InputContract>& input_contract = {}, std::string* error_message = nullptr);

/// Build a box-decode static contract from a `Sample` (e.g., dry-run output).
std::optional<BoxDecodeStaticContract>
build_boxdecode_static_contract_from_sample(const Sample& sample, BoxDecodeType decode_type,
                                            const std::optional<InputContract>& input_contract,
                                            std::string* error_message = nullptr);

/// Build a box-decode static contract from a compiled upstream node contract.
std::optional<BoxDecodeStaticContract>
build_boxdecode_static_contract_from_compiled_upstream(const CompiledNodeContract& upstream_stage,
                                                       BoxDecodeType decode_type,
                                                       std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

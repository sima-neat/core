/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Static-contract extractor for the MLA
 *        inference stage.
 *
 * Builds an `MlaStaticContract` from a parsed MPK plugin entry plus the surrounding logical /
 * physical output contracts. The resulting struct carries everything the planner and runtime
 * bridge need to bind the MLA stage: stage identity, model path, batch sizes, I/O specs, and
 * (when the compiled .elf is available) the actual placeholder symbol names baked into the
 * binary.
 *
 * @see MpkContract
 * @see BoxDecodeStaticContractExtractor.h (companion for the box-decode stage)
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

/**
 * @brief Static contract for one MLA stage, ready for manifest assembly.
 *
 * Captures the MLA's identity, batch geometry, and all input/output specs (logical and physical)
 * including the optional .elf-derived placeholder symbol names. The
 * `consumer_keeps_distinct_physical_inputs` flag signals to downstream coalescers whether the
 * stage's IFM segments must remain separate (multi-IFM dispatch) or can be packed.
 */
struct MlaStaticContract {
  std::string stage_id;                 ///< Stable stage identifier from the MPK.
  std::string node_name;                ///< Display node name (often derived from a hint).
  std::string model_path;               ///< Path to the MLA model artifact (.elf / package).
  int batch_size = 0;                   ///< Effective batch size at this stage.
  int batch_sz_model = 0;               ///< Batch size baked into the model itself.
  std::vector<TensorStaticSpec> inputs; ///< Input tensor specs (post-binding).
  std::vector<TensorStaticSpec> logical_inputs;       ///< Logical input specs (pre-binding view).
  std::vector<InputBindingStaticSpec> input_bindings; ///< Bindings from upstream outputs.
  std::vector<PhysicalBufferStaticSpec> physical_inputs; ///< Physical IFM buffers.
  std::vector<PhysicalBufferStaticSpec>
      dispatcher_physical_outputs;                        ///< Dispatcher OFMs (pre-publish).
  std::vector<PhysicalBufferStaticSpec> physical_outputs; ///< Published physical outputs.
  std::vector<LogicalTensorStaticSpec> logical_outputs;   ///< Logical output specs.
  std::vector<QuantStaticSpec> output_quant;              ///< Per-output quant params, if any.
  // True when this MLA stage's compiled .elf expects N>1 distinct physical
  // input segments (native multi-IFM dispatch, e.g. data.ifm.persistent.input_00
  // / input_01 placeholders). Derived from the absence of an upstream
  // canonical_op == "pack" producer combined with logical_inputs.size() > 1,
  // or — when .elf topology is available — directly from the IFM section
  // symbol count (preferred when populated).
  bool consumer_keeps_distinct_physical_inputs = false;
  // Optional .elf-derived I/O symbol names. When populated, the runtime can
  // bind cm_input_name / cm_output_name slots to the actual placeholder
  // symbols compiled into the binary instead of synthesized "ifmN" / "ofmN"
  // strings. Empty = no .elf topology available; runtime falls back to the
  // legacy synthesized naming.
  //   ifm_symbol_names[i] e.g. "data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0"
  //   ofm_symbol_names[i] e.g. "data.ofm.persistent.output_00/MLA_0/sigmoid_64.b0"
  std::vector<std::string> elf_ifm_symbol_names; ///< IFM placeholder symbols from the .elf.
  std::vector<std::string> elf_ofm_symbol_names; ///< OFM placeholder symbols from the .elf.
};

/**
 * @brief Build an `MlaStaticContract` from MPK stage data.
 *
 * @param mla                       The MLA plugin's I/O contract from the MPK.
 * @param logical_outputs           Logical output tensor contracts for the stage.
 * @param physical_outputs          Physical output tensor contracts for the stage.
 * @param node_name_hint            Optional friendly node name; falls back to MPK identity.
 * @param boundary_inputs_override  Optional override for the boundary input tensor list (e.g.,
 *                                  when adapter stages have rewritten the upstream contract).
 * @return Populated `MlaStaticContract`.
 */
MlaStaticContract build_mla_static_contract_from_mpk_stage(
    const MpkPluginIoContract& mla, const std::vector<MpkTensorContract>& logical_outputs,
    const std::vector<MpkTensorContract>& physical_outputs, const std::string& node_name_hint = {},
    const std::vector<MpkTensorContract>* boundary_inputs_override = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

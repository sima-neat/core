/**
 * @file
 * @ingroup mpk
 * @brief Helpers for parsing model pipeline_sequence.json.
 */
#pragma once

#include <string>
#include <vector>

namespace simaai::neat::mpk {

/**
 * @brief Single stage within a parsed pipeline sequence entry.
 *
 * Captures the identity of one stage (sequence id, name, plugin id, config path,
 * processor, kernel) as parsed from the MPK manifest's pipeline_sequence section.
 *
 * @ingroup mpk
 */
struct SequenceEntry {
  int sequence_id = 0;       ///< Ordinal id of the entry within the sequence.
  std::string name;          ///< Human-readable stage name.
  std::string plugin_id;     ///< Identifier of the plugin/element implementing the stage.
  std::string config_path;   ///< Path to the stage's configuration (relative to the model pack).
  std::string processor;     ///< Target processor (e.g., MLA, EV74).
  std::string kernel;        ///< Kernel label (used to identify pre/post adapters).
};

/**
 * @brief Pipeline sequence partitioned into pre / infer / post sections.
 *
 * Produced by `split_sequence_for_infer`. The `pre` and `post` lists hold user-controlled
 * adapter stages flanking the model-owned `infer` block.
 *
 * @ingroup mpk
 */
struct SequenceSplit {
  std::vector<SequenceEntry> pre;   ///< User-controlled pre adapters (leading).
  std::vector<SequenceEntry> infer; ///< Model-owned inference block.
  std::vector<SequenceEntry> post;  ///< User-controlled post adapters (trailing).
};

/**
 * @brief Indicates which source produced a `SequenceLoadResult`.
 *
 * `Strict` means a manifest sequence was found and parsed cleanly.
 * `FallbackSynthesized` means the loader synthesised a sequence after a recoverable error.
 * `MissingSequenceAssumeMlaOnly` means no sequence was present and an MLA-only stub was assumed.
 *
 * @ingroup mpk
 */
enum class SequenceLoadSource {
  Strict = 0,                       ///< Strict parse from a present manifest sequence.
  FallbackSynthesized = 1,          ///< Synthesised after a recoverable parse error.
  MissingSequenceAssumeMlaOnly = 2, ///< No sequence present; assume MLA-only.
};

/**
 * @brief Result wrapper carrying a parsed sequence plus diagnostics.
 *
 * Holds the parsed entries, the load source, and (in non-strict cases) the error string
 * surfaced by the strict parse path.
 *
 * @see load_pipeline_sequence_with_source
 * @ingroup mpk
 */
struct SequenceLoadResult {
  std::vector<SequenceEntry> sequence;                          ///< Parsed sequence entries.
  SequenceLoadSource source = SequenceLoadSource::Strict;       ///< How the sequence was obtained.
  std::string strict_error;                                     ///< Error from the strict parse, if any.
};

/**
 * @brief Parse `pipeline_sequence.json` (first pipeline) into ordered entries.
 *
 * @param etc_dir Directory containing the manifest json files.
 * @return Ordered sequence entries.
 */
std::vector<SequenceEntry> load_pipeline_sequence(const std::string& etc_dir);

/**
 * @brief Parse the pipeline sequence and return both the entries and the source diagnostic.
 *
 * @param etc_dir Directory containing the manifest json files.
 * @return `SequenceLoadResult` with the parsed sequence and load-source classification.
 */
SequenceLoadResult load_pipeline_sequence_with_source(const std::string& etc_dir);

/**
 * @brief Split a sequence into pre/infer/post sections for inference execution.
 *
 * Primary behavior: anchor infer on the first..last MLA stage and treat everything
 * before/after as pre/post (regardless of kernel labels). Legacy fallback (when no MLA
 * stage is present): strip known pre/post adapter kernels from the ends and keep the
 * middle as infer.
 *
 * @param seq Ordered pipeline sequence to split.
 * @return The partitioned `SequenceSplit`.
 */
SequenceSplit split_sequence_for_infer(const std::vector<SequenceEntry>& seq);

/// Returns true iff `kernel` is a known pre-adapter kernel name.
bool is_pre_adapter_kernel(const std::string& kernel);
/// Returns true iff `kernel` is a known post-adapter kernel name.
bool is_post_adapter_kernel(const std::string& kernel);

} // namespace simaai::neat::mpk

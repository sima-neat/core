/**
 * @file
 * @ingroup tensors
 * @brief Tensor conversion vocabulary, policies, and tracing.
 *
 * When a tensor needs to change form mid-pipeline (different dtype, layout, device, or memory
 * layout), the framework classifies the change as one of five `ConversionKind`s and applies
 * the configured `ConversionPolicy` to decide whether to allow it silently, allow it but
 * record a trace entry, or refuse it. Used by validation, by the planner when inserting
 * adapter stages, and by tools that audit pipeline conversions.
 */
#pragma once

#include "pipeline/TensorCore.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief What kind of transformation a tensor conversion represents.
 *
 * Ordered roughly cheapest-to-most-expensive: a `Reinterpret` is free (just a type/dtype
 * relabel), a `View` is free (sub-tensor pointer arithmetic), a `Pack` rearranges bytes
 * in-place, a `Convert` does math (e.g., FP32 → BF16), and a `Transfer` moves data across
 * device boundaries (e.g., CPU → MLA scratchpad).
 * @ingroup tensors
 */
enum class ConversionKind {
  Reinterpret = 0, ///< Same bytes, different dtype/layout label. No memory traffic.
  View,            ///< Pointer-arithmetic view (sub-tensor or strided slice). No copy.
  Pack,            ///< Byte-level rearrangement in place (e.g., layout swap, packing).
  Convert,         ///< Element-wise math conversion (e.g., FP32 → BF16, INT8 → FP32).
  Transfer,        ///< Cross-device transfer (e.g., CPU → accelerator scratch memory).
};

/**
 * @brief How strict the framework is about implicit tensor conversions.
 *
 * `Strict` (default for many internal validations) refuses any implicit conversion;
 * `AllowWithTrace` permits the conversion but records it to a `ConversionTraceCollector`
 * for later auditing; `AllowSilent` permits without recording.
 * @ingroup tensors
 */
enum class ConversionPolicy {
  Strict = 0,        ///< Refuse implicit conversions; require explicit user code.
  AllowWithTrace,    ///< Allow but record each conversion to a trace collector.
  AllowSilent,       ///< Allow without tracking.
};

/// Cost estimate for a conversion: bytes moved + a coarse compute-class bucket (low/med/high).
struct ConversionCost {
  std::uint64_t bytes_copied = 0; ///< Bytes that would be moved/written by this conversion.
  int compute_class = 0;          ///< Coarse compute cost: 0 = low, 1 = medium, 2 = high.
};

/**
 * @brief Single audit-log entry for a conversion that occurred.
 *
 * Collected into a `ConversionTraceCollector` when policy is `AllowWithTrace`. Useful for
 * tools that want to verify a pipeline doesn't introduce hidden expensive conversions.
 * @ingroup tensors
 */
struct ConversionTrace {
  std::string stage;                                 ///< Which stage/Node performed the conversion.
  ConversionKind kind = ConversionKind::Reinterpret; ///< What kind of conversion occurred.
  std::string src_desc;                              ///< Human description of the source tensor (e.g., `"FP32 HWC 1×640×640×3 CPU"`).
  std::string dst_desc;                              ///< Human description of the destination tensor.
  std::uint64_t bytes_copied = 0;                    ///< Bytes actually moved.
  std::uint64_t elapsed_us = 0;                      ///< Wall-clock time spent on the conversion.
  ConversionPolicy policy = ConversionPolicy::Strict; ///< Policy that authorized the conversion.
};

/// Collects `ConversionTrace` entries for post-mortem auditing of a pipeline run.
struct ConversionTraceCollector {
  std::vector<ConversionTrace> traces; ///< Accumulated trace entries, in order of occurrence.

  /// Append a trace entry.
  void add(ConversionTrace trace) {
    traces.push_back(std::move(trace));
  }
  /// Drop all collected entries.
  void clear() {
    traces.clear();
  }
};

/// Estimate the cost (bytes + compute class) of a conversion of the given kind moving `bytes_copied` bytes.
ConversionCost estimate_conversion_cost(ConversionKind kind, std::uint64_t bytes_copied);
/// Returns `true` if the policy allows a conversion of this kind (without performing it).
bool conversion_allowed(ConversionPolicy policy, ConversionKind kind);

} // namespace simaai::neat

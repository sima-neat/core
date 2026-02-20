/**
 * @file TensorConversion.cpp
 * @brief Lightweight policy + cost heuristics for Tensor conversions.
 *
 * This file defines:
 *  - estimate_conversion_cost(): assigns a rough "compute class" and tracks bytes copied
 *  - conversion_allowed(): checks whether a given conversion kind is allowed under a policy
 *
 * These utilities are intentionally simple and are meant for:
 *  - tracing / diagnostics (e.g., “did we copy?”, “did we do a device transfer?”)
 *  - high-level decisions (e.g., prefer View/Reinterpret over Convert/Transfer)
 *
 * TODO(repo-context):
 *  - Confirm the intended meaning of `ConversionCost::compute_class`:
 *      0 = memory-only (no compute) / packing
 *      1 = compute-heavy conversion (format/type conversion)
 *      2 = device transfer / DMA / IPC
 *    If you have a more granular model (e.g., separate “pack” vs “memcpy”), extend this mapping.
 *  - Confirm whether `ConversionPolicy::Strict` is meant to block *all* conversions
 *    or only specific kinds (e.g., allow View but block Pack/Convert/Transfer).
 *    Current implementation blocks everything.
 */

#include "pipeline/TensorConversion.h"

namespace simaai::neat {

ConversionCost estimate_conversion_cost(ConversionKind kind, std::uint64_t bytes_copied) {
  ConversionCost cost{};
  cost.bytes_copied = bytes_copied;

  switch (kind) {
  // Pure metadata operations: no compute, no copy.
  case ConversionKind::Reinterpret:
  case ConversionKind::View:
    cost.compute_class = 0;
    cost.bytes_copied = 0;
    break;

  // Packing implies at least a copy / memmove into a tight buffer, but no "compute" beyond that.
  // (We keep compute_class=0; bytes_copied stays as provided.)
  case ConversionKind::Pack:
    cost.compute_class = 0;
    break;

  // True conversion (dtype/format/layout transform): compute involved.
  case ConversionKind::Convert:
    cost.compute_class = 1;
    break;

  // Transfer (CPU<->device, device<->device, IPC): treated as highest class.
  case ConversionKind::Transfer:
    cost.compute_class = 2;
    break;
  }

  return cost;
}

/**
 * Decide whether a conversion kind is allowed under a policy.
 *
 * Current policy mapping:
 *  - Strict: disallow all conversions
 *  - AllowWithTrace / AllowSilent: allow all conversions
 *
 * TODO(repo-policy): If "Strict" should still allow View/Reinterpret (zero-copy),
 * update this function to check `kind` and allow only those.
 */
bool conversion_allowed(ConversionPolicy policy, ConversionKind /*kind*/) {
  switch (policy) {
  case ConversionPolicy::Strict:
    return false;
  case ConversionPolicy::AllowWithTrace:
  case ConversionPolicy::AllowSilent:
    return true;
  }
  return false;
}

} // namespace simaai::neat

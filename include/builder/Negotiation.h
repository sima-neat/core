/**
 * @file
 * @ingroup builder
 * @brief Tensor-pipeline caps negotiation entry point.
 *
 * `validate_tensor_pipeline()` walks a NodeGroup, asks each `SpecProvider`
 * for its input/output `TensorConstraint`, and threads the constraints
 * end-to-end. The result includes any conversion/coercion trace that
 * downstream tooling needs (e.g., layout transposes inserted, dtype casts,
 * quantization adjustments).
 */
#pragma once

#include "builder/SpecProvider.h"
#include "builder/NodeGroup.h"
#include "pipeline/TensorConversion.h"

#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Outcome of `validate_tensor_pipeline()`.
 *
 * Carries the boolean verdict, an error diagnostic on failure, and the
 * per-step conversion trace (zero-copy / coercion / rejection records) the
 * negotiation produced.
 *
 * @ingroup builder
 */
struct NegotiationResult {
  bool ok = true;                         ///< True if negotiation succeeded end-to-end.
  std::string error;                      ///< Human-readable diagnostic on failure.
  std::vector<ConversionTrace> trace;     ///< Per-edge conversion record (in pipeline order).
};

/**
 * @brief Validate a NodeGroup as a tensor pipeline against an external input.
 *
 * Walks the group, propagating tensor constraints and applying the supplied
 * `ConversionPolicy` at each boundary. Returns a `NegotiationResult` with
 * either a success trace or an error message.
 *
 * @param group  The NodeGroup to validate.
 * @param input  External input constraint (the pipeline's first stage sees this).
 * @param policy Conversion policy to apply at each Node boundary.
 * @return Outcome of negotiation.
 */
NegotiationResult validate_tensor_pipeline(const simaai::neat::NodeGroup& group,
                                           const TensorConstraint& input, ConversionPolicy policy);

} // namespace simaai::neat

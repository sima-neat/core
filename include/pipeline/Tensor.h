/**
 * @file
 * @ingroup tensors
 * @brief Umbrella header — the stable include surface for `simaai::neat::Tensor` users.
 *
 * Pulls in `TensorCore.h` (the type definition and methods) and `TensorSpec.h` (the
 * `TensorConstraint`/`TensorSpec` shape contract used in introspection APIs). Most
 * application code only needs to `#include "pipeline/Tensor.h"` to get the full Tensor
 * surface plus the `TensorList` alias.
 *
 * @see TensorCore.h for the Tensor struct itself
 * @see TensorSpec.h for the constraint/spec types
 * @see "Tensors: hiding the memory mess" (§0.10 of the design deep dive)
 */
#pragma once

#include "pipeline/TensorCore.h"
#include "pipeline/TensorSpec.h"

namespace simaai::neat {

/// Convenience alias for an ordered list of Tensors (multi-input/multi-output models).
using TensorList = std::vector<Tensor>;

} // namespace simaai::neat

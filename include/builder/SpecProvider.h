/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that publish a `TensorConstraint` input/output spec.
 *
 * `SpecProvider` is the tensor-shape counterpart of `OutputSpecProvider`:
 * where `OutputSpec` describes a boundary (gst caps + dtype + memory),
 * `TensorConstraint` describes a Node's tensor-level expectations
 * (shape/layout/dtype constraints used by `validate_tensor_pipeline()`).
 */
#pragma once

#include "pipeline/TensorSpec.h"

namespace simaai::neat {

/**
 * @brief Mixin interface implemented by Nodes that participate in tensor-level negotiation.
 *
 * Each Node is asked first what it expects on the input side
 * (`expected_input_spec()`); the Builder then walks the pipeline calling
 * `output_spec(input)` on each Node to thread the constraints through. Used
 * by `validate_tensor_pipeline()`.
 *
 * @ingroup builder
 * @see OutputSpecProvider
 * @see Negotiation
 */
class SpecProvider {
public:
  virtual ~SpecProvider() = default;

  /// @brief Return the tensor constraint this Node expects on its input.
  virtual TensorConstraint expected_input_spec() const = 0;

  /// @brief Return this Node's output tensor constraint given an upstream `input` constraint.
  virtual TensorConstraint output_spec(const TensorConstraint& input) const = 0;
};

} // namespace simaai::neat

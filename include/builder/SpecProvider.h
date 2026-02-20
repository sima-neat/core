#pragma once

#include "pipeline/TensorSpec.h"

namespace simaai::neat {

class SpecProvider {
public:
  virtual ~SpecProvider() = default;
  virtual TensorConstraint expected_input_spec() const = 0;
  virtual TensorConstraint output_spec(const TensorConstraint& input) const = 0;
};

} // namespace simaai::neat

#pragma once

#include "model/Model.h"

#include <memory>

namespace simaai::neat::pipeline_internal {

struct InputRouteProcessor {
  virtual ~InputRouteProcessor() = default;

  virtual Sample seed_tensors(const TensorList& inputs, const char* where) const {
    return process_tensors(inputs, where);
  }
  virtual Sample seed_samples(const SampleList& inputs, const char* where) const {
    return process_samples(inputs, where);
  }
  virtual Sample process_tensors(const TensorList& inputs, const char* where) const = 0;
  virtual Sample process_samples(const SampleList& inputs, const char* where) const = 0;
};

using InputRouteProcessorPtr = std::shared_ptr<const InputRouteProcessor>;

} // namespace simaai::neat::pipeline_internal

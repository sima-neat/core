#pragma once

#include "model/Model.h"

#include <memory>

namespace simaai::neat::pipeline_internal {

struct InputRouteProcessor {
  virtual ~InputRouteProcessor() = default;

  // One logical inference item: reject original image metadata before any
  // image-to-device conversion or branch execution. No transactional execution
  // guarantee is made for failures after this metadata-only phase.
  virtual void validate_images(const std::vector<cv::Mat>& inputs, const char* where) const = 0;

  // Cold, optional wire policy for the route's representative input. The caller
  // supplies the actual consumer so a later Model cannot authorize an earlier
  // custom stage. This never changes the retained Tensor's logical descriptor.
  virtual std::vector<std::int64_t> model_input_shape(const std::shared_ptr<Node>&) const {
    return {};
  }

  virtual Sample seed_tensors(const TensorList& inputs, const char* where) const {
    return process_tensors(inputs, where);
  }
  virtual Sample seed_samples(const Sample& inputs, const char* where) const {
    return process_samples(inputs, where);
  }
  virtual Sample process_tensors(const TensorList& inputs, const char* where) const = 0;
  virtual Sample process_samples(const Sample& inputs, const char* where) const = 0;
};

using InputRouteProcessorPtr = std::shared_ptr<const InputRouteProcessor>;

} // namespace simaai::neat::pipeline_internal

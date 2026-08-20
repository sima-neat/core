#pragma once

#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/PayloadType.h"

#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace sima_test {

/// ResNet50 emits one score per ImageNet class.
constexpr std::size_t kResNet50Classes = 1000;

/**
 * @brief Assert that @p sample is a well-formed terminal ResNet50 result.
 *
 * Checks the whole contract a caller pulling from a ResNet50 graph relies on:
 * the payload is a tensor, it is the single expected output, it landed in owned
 * CPU memory the app may read after the run, and it holds exactly
 * `kResNet50Classes` finite Float32 scores. A non-finite score means the model
 * ran but produced garbage, which no shape or type check would catch.
 *
 * @param context Caller-supplied prefix for every diagnostic, so a failure
 *        identifies which frame or run mode produced the bad sample.
 */
inline void require_valid_resnet50_output(const simaai::neat::Sample& sample,
                                          const std::string& context) {
  require(simaai::neat::sample_payload_type(sample) == simaai::neat::PayloadType::Tensor,
          context + ": model output payload is not a Tensor");
  require(sample.media_type == "application/vnd.simaai.tensor",
          context + ": model output media type mismatch");

  const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
  require(tensors.size() == 1U, context + ": expected one ResNet50 output tensor");
  const simaai::neat::Tensor& tensor = tensors.front();
  require(tensor.storage != nullptr, context + ": output tensor has no storage");
  require(tensor.storage->kind == simaai::neat::StorageKind::CpuOwned,
          context + ": terminal model output is not stored in owned CPU memory");
  require(tensor.dtype == simaai::neat::TensorDType::Float32,
          context + ": output tensor is not Float32");
  require(tensor.is_dense(), context + ": output tensor is not dense");
  require(tensor.dense_bytes_tight() == kResNet50Classes * sizeof(float),
          context + ": output tensor is not exactly 1000 Float32 scores");

  const simaai::neat::Mapping mapping = tensor.map(simaai::neat::MapMode::Read);
  require(mapping.data != nullptr, context + ": output tensor is not CPU-readable");
  require(mapping.size_bytes >= kResNet50Classes * sizeof(float),
          context + ": output tensor contains fewer than 1000 scores");
  const auto* scores = static_cast<const float*>(mapping.data);
  require(std::all_of(scores, scores + kResNet50Classes,
                      [](float value) { return std::isfinite(value); }),
          context + ": output tensor contains a non-finite score");
}

} // namespace sima_test

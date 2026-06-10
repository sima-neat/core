/**
 * @file
 * @ingroup tensors
 * @brief `TensorConstraint` (alias `TensorSpec`) — declarative shape/dtype/device contract for
 * tensors.
 *
 * `TensorConstraint` (typedef'd as `TensorSpec` from `Model.h`) is the declarative description
 * of *what kind of tensor a function expects or produces*. Used by `Model::input_specs()` /
 * `Model::output_specs()` for introspection and by validation code to assert that a tensor
 * meets the model's contract. Supports flexible matching: dtype lists, dynamic shape dims
 * (`-1` = any), device requirements, image-format hints, and segment requirements.
 *
 * @see Tensor (the value type) in TensorCore.h
 * @see Model::input_specs, Model::output_specs
 */
#pragma once

#include "pipeline/TensorCore.h"

#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Declarative tensor contract — describes the shape/dtype/device/format a tensor must
 * satisfy.
 *
 * Used by `Model::input_specs()`/`output_specs()` to advertise what the model expects/produces,
 * and by validation code to verify a tensor meets the contract via `matches()`. Empty fields
 * mean "no constraint on this dimension." Use `-1` in `shape` for dynamic axes.
 *
 * @code
 *   sima::TensorConstraint c;
 *   c.dtypes = { sima::TensorDType::Float32, sima::TensorDType::BFloat16 };
 *   c.rank = 4;
 *   c.shape = { 1, 3, -1, -1 };  // batch=1, channels=3, H/W dynamic
 *   if (!c.matches(my_tensor)) { ... }
 * @endcode
 *
 * @ingroup tensors
 */
struct TensorConstraint {
  std::vector<simaai::neat::TensorDType> dtypes; ///< Acceptable dtypes (empty = any).
  int rank = -1;                                 ///< Required rank (-1 = any).
  std::vector<int64_t> shape;             ///< Required shape; `-1` in any position means dynamic.
  std::optional<Device> device;           ///< Required device (empty = any).
  std::vector<Device> allowed_devices;    ///< Acceptable devices (empty = any).
  std::optional<Device> preferred_device; ///< Preferred device for placement (informational).

  std::optional<ImageSpec::PixelFormat>
      image_format; ///< Required image pixel format (only meaningful for image tensors).
  std::vector<Segment> required_segments; ///< Exact required memory-segment layout (advanced).
  std::vector<std::string>
      required_segment_names;  ///< Required memory-segment names (must all be present).
  bool allow_composite = true; ///< If false, reject composite (multi-plane) tensors like NV12.

  /**
   * @brief Returns `true` if `t` satisfies every non-empty constraint in this spec.
   *
   * Empty fields are skipped (treated as "no constraint"). Useful for inline validation in
   * application code and as the underlying check used by built-in contracts.
   */
  bool matches(const Tensor& t) const {
    if (rank >= 0 && static_cast<int>(t.shape.size()) != rank)
      return false;
    if (!shape.empty() && shape.size() == t.shape.size()) {
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] >= 0 && t.shape[i] != shape[i])
          return false;
      }
    }
    if (!dtypes.empty()) {
      bool ok = false;
      for (auto dt : dtypes) {
        if (dt == t.dtype) {
          ok = true;
          break;
        }
      }
      if (!ok)
        return false;
    }
    if (device.has_value()) {
      if (t.device.type != device->type || t.device.id != device->id)
        return false;
    }
    if (!allowed_devices.empty()) {
      bool ok = false;
      for (const auto& allowed : allowed_devices) {
        if (t.device.type == allowed.type && t.device.id == allowed.id) {
          ok = true;
          break;
        }
      }
      if (!ok)
        return false;
    }
    if (image_format.has_value()) {
      if (!t.semantic.image.has_value())
        return false;
      if (t.semantic.image->format != *image_format)
        return false;
    }
    if (!required_segments.empty()) {
      if (!t.storage || t.storage->sima_segments.empty())
        return false;
      if (t.storage->sima_segments.size() != required_segments.size())
        return false;
      for (size_t i = 0; i < required_segments.size(); ++i) {
        if (t.storage->sima_segments[i].name != required_segments[i].name)
          return false;
        if (t.storage->sima_segments[i].size_bytes != required_segments[i].size_bytes) {
          return false;
        }
      }
    }
    if (!required_segment_names.empty()) {
      if (!t.storage || t.storage->sima_segments.empty())
        return false;
      for (const auto& name : required_segment_names) {
        bool found = false;
        for (const auto& seg : t.storage->sima_segments) {
          if (seg.name == name) {
            found = true;
            break;
          }
        }
        if (!found)
          return false;
      }
    }
    if (!allow_composite && t.is_composite())
      return false;
    return true;
  }
};

} // namespace simaai::neat

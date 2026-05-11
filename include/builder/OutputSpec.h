/**
 * @file
 * @ingroup builder
 * @brief Boundary-only caps/shape projection between nodes.
 *
 * `OutputSpec` is the lightweight, builder-time description of what a Node
 * commits to emit at its output boundary — media type, pixel/element format,
 * shape, framerate, memory class, dtype. It is *not* the authoritative tensor
 * model (that lives in `Tensor` / `TensorBufferTensorView` /
 * `sima_ev_tensor_desc`); it exists so the Builder can plan caps negotiation
 * and the Session can stamp gst caps without consulting runtime objects.
 */
#pragma once

#include <cstddef>
#include <string>

namespace simaai::neat {

/**
 * @brief How confident the source of an `OutputSpec` is.
 * @ingroup builder
 */
enum class SpecCertainty {
  Unknown = 0,        ///< No information (default).
  Hint,               ///< Educated guess — may be wrong.
  Derived,            ///< Computed from inputs and Node logic.
  Authoritative,      ///< Source-of-truth (e.g., set by an Output sink itself).
};

/**
 * @brief Boundary/caps metadata for the data flowing out of a Node.
 *
 * This structure intentionally only describes the inter-Node boundary, not
 * the full internal tensor semantics. Nodes report it via
 * `OutputSpecProvider`; the framework uses it for caps negotiation, byte-size
 * estimation, and diagnostic reports. Defaults represent "unknown".
 *
 * @ingroup builder
 * @see OutputSpecProvider
 */
struct OutputSpec {
  // Boundary/caps metadata only. OutputSpec is not the authoritative generic
  // tensor semantic model; internal tensor truth must live in Tensor,
  // TensorBufferTensorView, or sima_ev_tensor_desc.
  std::string media_type;       ///< e.g. `"video/x-raw"`, `"application/vnd.simaai.tensor"`.
  std::string format;           ///< e.g. `"NV12"`, `"RGB"`, `"FP32"`.
  int width = -1;               ///< Width at the boundary; -1 if unknown.
  int height = -1;              ///< Height at the boundary; -1 if unknown.
  int depth = -1;               ///< Depth/channels at the boundary; -1 if unknown.
  int fps_num = 0;              ///< Framerate numerator.
  int fps_den = 1;              ///< Framerate denominator.
  std::string memory;           ///< e.g. `"SystemMemory"`, `"SimaAI"`, `"Unknown"`.
  std::string layout;           ///< Layout label (compatibility/boundary projection only).
  std::string dtype;            ///< e.g. `"UInt8"`, `"Float32"`.
  std::size_t byte_size = 0;    ///< Buffer byte-size estimate; 0 if unknown.
  SpecCertainty certainty = SpecCertainty::Unknown; ///< How trustworthy this spec is.
  std::string note;             ///< Free-form note (used in diagnostic reports).

  /// @brief True if every meaningful field is at its default ("nothing known").
  bool is_unknown() const {
    return media_type.empty() && format.empty() && width <= 0 && height <= 0 && depth <= 0 &&
           byte_size == 0 && certainty == SpecCertainty::Unknown;
  }

  /// @brief True if both `width` and `height` are positive.
  bool has_shape() const {
    return width > 0 && height > 0;
  }
};

/**
 * @brief Mixin interface implemented by Nodes that publish a boundary `OutputSpec`.
 *
 * The Builder calls `output_spec()` with the upstream's `OutputSpec` and uses
 * the returned spec as input to the next Node. Nodes that don't transform
 * the boundary description can return `input` unchanged.
 *
 * @ingroup builder
 * @see OutputSpec
 */
class OutputSpecProvider {
public:
  virtual ~OutputSpecProvider() = default;

  /// @brief Return this Node's output spec given the upstream's spec.
  virtual OutputSpec output_spec(const OutputSpec& input) const = 0;
};

/// @brief Compute the boundary buffer's byte size from a spec (best effort, may be 0).
std::size_t expected_byte_size(const OutputSpec& spec);

/// @brief Walk a NodeGroup propagating `OutputSpec` from `input` through each Node.
OutputSpec derive_output_spec(const class NodeGroup& group, const OutputSpec& input = {});

} // namespace simaai::neat

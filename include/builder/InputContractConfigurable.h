/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes whose input contract can be configured at build time.
 *
 * Some Nodes (e.g., a generic preprocess) accept any input shape/format until
 * the Builder pins them down via an `InputContract`. Such Nodes implement
 * `InputContractConfigurable` so the Builder's contract negotiation pass can
 * push the resolved input description into them.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace simaai::neat {

/**
 * @brief Concrete description of a Node's expected input.
 *
 * Populated by the Builder during contract negotiation and handed to a Node
 * via `InputContractConfigurable::apply_input_contract()`. Fields left at
 * their default (empty string / 0 / nullopt) indicate "unspecified".
 *
 * @ingroup builder
 */
struct InputContract {
  std::string media_type;           ///< e.g., `"video/x-raw"`, `"application/vnd.simaai.tensor"`.
  std::string format;               ///< e.g., `"NV12"`, `"RGB"`, `"FP32"`.
  std::string dtype;                ///< e.g., `"UInt8"`, `"Float32"`.
  std::string layout;               ///< Layout label (e.g., `"NHWC"`).
  int width = 0;                    ///< Width in pixels / elements; 0 if unspecified.
  int height = 0;                   ///< Height in pixels / elements; 0 if unspecified.
  int depth = 0;                    ///< Depth / channels; 0 if unspecified.
  std::optional<double> q_scale;    ///< Quantization scale, if applicable.
  std::optional<std::int64_t> q_zp; ///< Quantization zero-point, if applicable.
};

/**
 * @brief Mixin interface implemented by Nodes whose input contract is set at build time.
 *
 * The Builder calls `apply_input_contract()` once during contract negotiation;
 * Nodes use the values to specialize their gst fragment, internal config JSON,
 * or compiled-stage definitions. On error, the Node should populate `*err`.
 *
 * @ingroup builder
 * @see InputContract
 * @see NodeContractConfigurable
 */
class InputContractConfigurable {
public:
  virtual ~InputContractConfigurable() = default;

  /**
   * @brief Apply the resolved input contract to this Node.
   *
   * @param contract Concrete input description from the Builder.
   * @param err On failure, populated with a human-readable diagnostic.
   */
  virtual void apply_input_contract(const InputContract& contract, std::string* err) = 0;
};

} // namespace simaai::neat

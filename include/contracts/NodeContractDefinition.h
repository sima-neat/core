/**
 * @file
 * @ingroup contracts
 * @brief Structural definition of a Node's contract (ports, fields, override policies).
 *
 * `NodeContractDefinition` is the *static, declarative* shape of a Node's
 * contract: which input/output ports it has, what fields are configurable,
 * where each field's value comes from, and whether the Builder can override
 * it. The Builder consumes these definitions during contract resolution; the
 * resolved/compiled output is `CompiledNodeContract` (not in this header).
 *
 * @see NodeContractProvider
 * @see ContractRegistry
 */
#pragma once

#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Where the value of a contract field originates.
 *
 * Used by the Builder to decide whether a field is fully bound at definition
 * time, supplied by a builder option, derived from upstream caps, baked
 * into a model file, or owned by the Graph as a whole.
 *
 * @ingroup contracts
 */
enum class ContractFieldSource {
  Fixed,         ///< Value is hard-coded in the contract definition.
  BuilderOption, ///< Value comes from a builder-level option/argument.
  ModelOnly,     ///< Value is supplied by the model file (e.g., MPK contract).
  InputOnly,     ///< Value is determined by external input only.
  UpstreamOnly,  ///< Value is derived from the upstream Node's output spec.
  GraphOwned,    ///< Value is owned by the enclosing Graph (cross-Node).
};

/**
 * @brief Whether the Builder may override a contract field at build time.
 * @ingroup contracts
 */
enum class ContractOverridePolicy {
  Forbidden,   ///< Field cannot be overridden; attempts are validation errors.
  BuilderOnly, ///< Builder code may override; user-facing API may not.
};

/**
 * @brief Per-port contract: the shape/type/segment requirements for one input or output.
 *
 * A Node may declare multiple ports; each is identified by `port_id`. Empty
 * string fields mean "unconstrained at this stage" — the Builder fills them
 * in during contract resolution.
 *
 * @ingroup contracts
 */
struct ContractPortSpec {
  std::string port_id;                             ///< Stable port identifier.
  std::string media_type;                          ///< Constraint on media type (may be empty).
  std::string format;                              ///< Constraint on format (may be empty).
  std::string dtype;                               ///< Constraint on dtype (may be empty).
  std::string layout;                              ///< Constraint on layout (may be empty).
  std::vector<std::string> required_segment_names; ///< Segment names that must be present.
  std::vector<std::string>
      required_preprocess_meta_fields; ///< Preprocess-meta fields that must be present.
  bool require_quant = false;          ///< If true, port requires quantization metadata.
};

/**
 * @brief Per-field contract: where the field value comes from and the override policy.
 * @ingroup contracts
 */
struct ContractFieldSpec {
  std::string field_id;                                    ///< Stable field identifier.
  ContractFieldSource source = ContractFieldSource::Fixed; ///< Source of the value.
  ContractOverridePolicy override_policy = ContractOverridePolicy::Forbidden; ///< Override policy.
  bool required = false; ///< If true, field must be set after resolution.
};

/**
 * @brief Bundle of port and field specs that fully describes a Node's contract.
 *
 * Returned by `NodeContractProvider::contract_definition()` and consumed by
 * the Builder during contract negotiation.
 *
 * @ingroup contracts
 * @see NodeContractProvider
 */
struct NodeContractDefinition {
  std::string node_kind;                 ///< Node kind label (matches `Node::kind()`).
  std::string plugin_kind;               ///< Backing plugin/element kind (for diagnostics).
  std::vector<ContractPortSpec> inputs;  ///< Per-input-port contracts.
  std::vector<ContractPortSpec> outputs; ///< Per-output-port contracts.
  std::vector<ContractFieldSpec> fields; ///< Per-field contracts.
};

} // namespace simaai::neat

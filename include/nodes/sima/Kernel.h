/**
 * @file
 * @ingroup nodes_sima
 * @brief `Kernel` Node — any kernels-library operation by name, built in or loaded at runtime.
 *
 * Runs one named kernels-library operation on A65 or EV74. The operation may be built into the
 * installed kernels library or come from a provider library loaded at runtime with
 * `load_kernel_library()` or `KernelOptions::library`. A provider must come from the same
 * kernels build as the library that loads it.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"
#include "nodes/sima/SimaKernelTypes.h"
#include "pipeline/TensorTypes.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace simaai::neat {

/**
 * @brief One named operation parameter, as the kernels library reads it.
 *
 * @ingroup nodes_sima
 */
struct KernelParameter {
  /// Value types the kernels library accepts for a named parameter.
  using Value = std::variant<bool, std::int64_t, double, std::string, std::vector<std::int64_t>,
                             std::vector<double>>;
  std::string name; ///< Parameter name, as the operation documents it.
  Value value;      ///< Parameter value.
};

/**
 * @brief One dense output tensor of a `Kernel` Node.
 *
 * @ingroup nodes_sima
 */
struct KernelOutputSpec {
  std::vector<std::int64_t> shape; ///< Logical shape; every extent positive.
  TensorDType dtype;               ///< Element type; any except `Float64`.
};

/**
 * @brief Construction options for a `Kernel` Node.
 *
 * The input Sample's tensors are the operation's inputs, in order, dense or strided. The output
 * Sample holds one dense tensor per operation output, named `output0`, `output1`, ...
 *
 * `outputs` states the output tensors. A provider operation with output inference may omit
 * them; when both are present they must agree. Built-in operations have no output inference, so
 * they need `outputs`. Every mismatch fails when the graph is built.
 *
 * @ingroup nodes_sima
 */
struct KernelOptions {
  /// Operation name, for example `argmax`.
  std::string operation;
  /// Execution target; no fallback.
  SimaKernelTarget target = SimaKernelTarget::A65;
  /// Named operation parameters.
  std::vector<KernelParameter> parameters;
  /// Output tensors; required unless the operation's provider infers them.
  std::optional<std::vector<KernelOutputSpec>> outputs;
  /// Provider library to load before the operation is resolved. Loading is process-wide and
  /// idempotent: the library is never unloaded.
  std::optional<std::string> library;
};

/**
 * @brief Kernel Node that runs one named kernels-library operation.
 *
 * @ingroup nodes_sima
 */
class Kernel final : public Node, public OutputSpecProvider, public NodeContractProvider {
public:
  /// Construct with `KernelOptions`; throws `std::invalid_argument` on invalid options.
  explicit Kernel(KernelOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Kernel";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Structural contract definition; the ports follow the upstream tensors and the request.
  NodeContractDefinition contract_definition() const override;
  /// Prepare the checked kernel request from the upstream tensor contract.
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* error) const override;

  /// Inspect the Node's options.
  const KernelOptions& options() const {
    return opt_;
  }

private:
  KernelOptions opt_;
};

/**
 * @brief Load a kernels provider library for the whole process.
 *
 * Returns the names of its operations, which `Kernel` Nodes then accept. Loading the same file
 * again returns the same names. The library is never unloaded. Throws `std::runtime_error` with
 * the kernels library's message when the file cannot be loaded, comes from another kernels build
 * or reuses a taken operation name.
 *
 * @ingroup nodes_sima
 */
std::vector<std::string> load_kernel_library(const std::string& path);

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `Kernel` Node.
std::shared_ptr<simaai::neat::Node> Kernel(KernelOptions opt);
} // namespace simaai::neat::nodes

#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "test_main.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kAdmissionFailure =
    "test admission: authoritative physical input carriers rejected";
constexpr const char* kSecondaryApplyFailure = "test apply: incomplete rejected contract";

class RejectedAdmissionNode final : public simaai::neat::Node,
                                    public simaai::neat::NodeContractProvider,
                                    public simaai::neat::NodeContractConfigurable {
public:
  std::string kind() const override {
    return "RejectedAdmissionRegression";
  }

  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }

  std::string backend_fragment(int) const override {
    return "identity name=admission_regression";
  }

  std::vector<std::string> element_names(int) const override {
    return {"admission_regression"};
  }

  simaai::neat::NodeContractDefinition contract_definition() const override {
    simaai::neat::NodeContractDefinition definition;
    definition.node_kind = kind();
    return definition;
  }

  bool compile_node_contract(const simaai::neat::ContractCompileInput& input,
                             simaai::neat::CompiledNodeContract*,
                             std::string* error) const override {
    ++compile_count;
    received_complete_seed = input.ingress.ingress_sample.has_value() &&
                             input.ingress.ingress_sample->tensors.size() == 6U;
    if (received_complete_seed) {
      const auto& tensors = input.ingress.ingress_sample->tensors;
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        received_complete_seed = received_complete_seed &&
                                 tensors[i].route.name == "head_" + std::to_string(i) &&
                                 tensors[i].route.segment_name == "MLA_0" &&
                                 tensors[i].byte_offset == static_cast<std::int64_t>(i * 16U) &&
                                 tensors[i].storage == tensors.front().storage;
      }
    }
    if (error) {
      *error = kAdmissionFailure;
    }
    return false;
  }

  void apply_compiled_contract(const simaai::neat::CompiledNodeContract&,
                               std::string* error) override {
    ++apply_count;
    if (error) {
      *error = kSecondaryApplyFailure;
    }
  }

  mutable int compile_count = 0;
  mutable bool received_complete_seed = false;
  int apply_count = 0;
};

} // namespace

RUN_TEST("unit_graph_compile_diagnostic_preservation_test", ([] {
           using namespace simaai::neat;

           Sample seed;
           seed.kind = SampleKind::TensorSet;
           auto storage = std::make_shared<TensorBuffer>();
           storage->size_bytes = 96U;
           storage->sima_segments = {{"MLA_0", 96U}};
           for (int i = 0; i < 6; ++i) {
             Tensor tensor;
             tensor.storage = storage;
             tensor.dtype = TensorDType::Int8;
             tensor.layout = TensorLayout::HWC;
             tensor.shape = {1, 1, 16};
             tensor.strides_bytes = {16, 16, 1};
             tensor.byte_offset = i * 16;
             tensor.route.logical_index = i;
             tensor.route.memory_index = 0;
             tensor.route.physical_index = 0;
             tensor.route.name = "head_" + std::to_string(i);
             tensor.route.segment_name = "MLA_0";
             seed.tensors.push_back(std::move(tensor));
           }

           auto node = std::make_shared<RejectedAdmissionNode>();
           std::vector<std::shared_ptr<Node>> nodes{node};
           ContractCompileInput input;
           input.ingress.ingress_sample = seed;
           BuildResult build;
           // This deliberately invalid plugin must never reach gst_parse_launch
           // after a provider has already rejected the executable contract.
           build.pipeline_string = "missing_admission_regression_plugin ! fakesink";
           std::string diagnostic;
           session_build_compile_contracts(&build, nodes, input,
                                           "diagnostic preservation regression", &nodes);
           for (const auto& error : build.manifest_diagnostics.errors) {
             diagnostic += error + "\n";
           }

           require(node->compile_count == 1 && node->received_complete_seed,
                   "cold admission must receive the complete six-head seeded Sample once");
           require(diagnostic.find(kAdmissionFailure) != std::string::npos,
                   "the original provider admission failure must reach the build boundary");
           require(diagnostic.find(kSecondaryApplyFailure) == std::string::npos &&
                       node->apply_count == 0,
                   "a rejected contract must not be applied or mask the provider diagnostic");
           require(!build.rendered_manifest.has_value(),
                   "a rejected contract must not render a partial executable manifest");

           bool rejected_before_parse = false;
           try {
             GstElement* pipeline = session_build_parse_pipeline_or_throw(
                 build, "diagnostic preservation parse regression");
             if (pipeline) {
               gst_object_unref(pipeline);
             }
           } catch (const NeatError& error) {
             rejected_before_parse = true;
             require(error.report().error_code == error_codes::kPipelineShape,
                     "provider rejection must become a structured contract/build failure");
             require(std::string(error.what()).find(kAdmissionFailure) != std::string::npos,
                     "the parse boundary must preserve the provider admission diagnostic");
           }
           require(rejected_before_parse,
                   "the parse boundary must fail before trying to instantiate plugins");
         }));

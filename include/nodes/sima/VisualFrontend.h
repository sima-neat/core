/**
 * @file
 * @ingroup nodes_sima
 * @brief EV74 visual-frontend processcvu Nodes (`FeatureHistogram`, `GriderFast`,
 *        `TrackDescriptor`, `TrackKLT`).
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct VisualFrontendCommonOptions {
  int width = 0;
  int height = 0;
  int batch_size = 1;
  int debug = 0;
  int num_buffers = 0;
  std::string element_name;
};

struct FeatureHistogramOptions : public VisualFrontendCommonOptions {
  std::string input_name = "input_image";
  std::string output_name = "output_hist";
};

struct GriderFastOptions : public VisualFrontendCommonOptions {
  int threshold = 30;
  int max_features = 500;
  int grid_x = 8;
  int grid_y = 6;
  int min_px_dist = 10;
  std::string input_name = "input_image";
  std::string output_name = "output_features";
};

struct TrackDescriptorOptions : public VisualFrontendCommonOptions {
  int threshold = 30;
  int max_features = 500;
  int grid_x = 8;
  int grid_y = 6;
  int min_px_dist = 10;
  int descriptor_words = 8;
  std::string input_name = "input_image";
  std::string features_output_name = "output_features";
  std::string descriptors_output_name = "output_descriptors";
};

struct TrackKLTOptions {
  int width = 0;
  int height = 0;
  int num_points = 0;
  int win_half = 10;
  int max_iters = 30;
  int max_level = 3;
  int detect_new_features = 0;
  int fast_threshold = 30;
  int max_features = 500;
  int grid_x = 8;
  int grid_y = 6;
  int min_px_dist = 10;
  int debug = 0;
  int num_buffers = 0;
  std::string element_name;
  std::string prev_image_name = "prev_image";
  std::string cur_image_name = "cur_image";
  std::string input_points_name = "input_points";
  std::string output_points_name = "output_points";
  std::string output_status_name = "output_status";
  std::string output_features_name = "output_features";
};

class FeatureHistogram final : public Node,
                               public OutputSpecProvider,
                               public NodeContractProvider,
                               public NodeContractConfigurable {
public:
  explicit FeatureHistogram(FeatureHistogramOptions opt = {});
  std::string kind() const override { return "FeatureHistogram"; }
  NodeCapsBehavior caps_behavior() const override { return NodeCapsBehavior::Static; }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const FeatureHistogramOptions& options() const { return opt_; }

private:
  FeatureHistogramOptions opt_;
};

class GriderFast final : public Node,
                         public OutputSpecProvider,
                         public NodeContractProvider,
                         public NodeContractConfigurable {
public:
  explicit GriderFast(GriderFastOptions opt = {});
  std::string kind() const override { return "GriderFast"; }
  NodeCapsBehavior caps_behavior() const override { return NodeCapsBehavior::Static; }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const GriderFastOptions& options() const { return opt_; }

private:
  GriderFastOptions opt_;
};

class TrackDescriptor final : public Node,
                              public OutputSpecProvider,
                              public NodeContractProvider,
                              public NodeContractConfigurable {
public:
  explicit TrackDescriptor(TrackDescriptorOptions opt = {});
  std::string kind() const override { return "TrackDescriptor"; }
  NodeCapsBehavior caps_behavior() const override { return NodeCapsBehavior::Static; }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const TrackDescriptorOptions& options() const { return opt_; }

private:
  TrackDescriptorOptions opt_;
};

class TrackKLT final : public Node,
                       public OutputSpecProvider,
                       public NodeContractProvider,
                       public NodeContractConfigurable {
public:
  explicit TrackKLT(TrackKLTOptions opt = {});
  std::string kind() const override { return "TrackKLT"; }
  NodeCapsBehavior caps_behavior() const override { return NodeCapsBehavior::Static; }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const TrackKLTOptions& options() const { return opt_; }

private:
  TrackKLTOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> FeatureHistogram(FeatureHistogramOptions opt = {});
std::shared_ptr<simaai::neat::Node> GriderFast(GriderFastOptions opt = {});
std::shared_ptr<simaai::neat::Node> TrackDescriptor(TrackDescriptorOptions opt = {});
std::shared_ptr<simaai::neat::Node> TrackKLT(TrackKLTOptions opt = {});
} // namespace simaai::neat::nodes

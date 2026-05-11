#pragma once

#include "graph/Graph.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "test_utils.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sima_test {

class NoopStage final : public simaai::neat::graph::StageExecutor {
public:
  void on_input(simaai::neat::graph::StageMsg&&,
                std::vector<simaai::neat::graph::StageOutMsg>&) override {}
};

class PassThroughStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    const simaai::neat::graph::PortId only = ports.only_output();
    if (only != simaai::neat::graph::kInvalidPort) {
      out_port_ = only;
    }
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    out.push_back(simaai::neat::graph::StageOutMsg{
        .out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                        ? simaai::neat::graph::kInvalidPort
                        : out_port_,
        .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

inline bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

inline std::shared_ptr<simaai::neat::graph::Node>
make_stage_node(const std::string& kind, std::vector<simaai::neat::graph::PortDesc> inputs,
                std::vector<simaai::neat::graph::PortDesc> outputs,
                simaai::neat::graph::nodes::StageNode::OutputSpecFn out_fn = {},
                simaai::neat::graph::nodes::StageNode::StageExecutorFactory factory = {}) {
  using simaai::neat::graph::nodes::StageNode;
  if (!factory) {
    factory = []() { return std::make_unique<NoopStage>(); };
  }

  return std::make_shared<StageNode>(kind, std::move(factory), std::move(inputs),
                                     std::move(outputs), kind, std::move(out_fn));
}

inline std::shared_ptr<simaai::neat::graph::Node>
make_stage_source(const std::string& kind, const simaai::neat::OutputSpec& spec,
                  const std::string& out_name = "out") {
  using simaai::neat::graph::PortDesc;
  const auto out_fn = [spec](const std::vector<simaai::neat::OutputSpec>&,
                             simaai::neat::graph::PortId) { return spec; };
  std::vector<PortDesc> inputs;
  std::vector<PortDesc> outputs = {PortDesc{.name = out_name, .spec = spec}};
  return make_stage_node(kind, std::move(inputs), std::move(outputs), out_fn);
}

inline std::shared_ptr<simaai::neat::graph::Node>
make_stage_passthrough(const std::string& kind, int max_in_edges = 1,
                       const std::string& in_name = "in", const std::string& out_name = "out") {
  using simaai::neat::graph::PortDesc;
  const auto out_fn = [](const std::vector<simaai::neat::OutputSpec>& in,
                         simaai::neat::graph::PortId) {
    if (!in.empty())
      return in.front();
    return simaai::neat::OutputSpec{};
  };
  std::vector<PortDesc> inputs = {
      PortDesc{.name = in_name, .spec = simaai::neat::OutputSpec{}, .max_in_edges = max_in_edges}};
  std::vector<PortDesc> outputs = {PortDesc{.name = out_name, .spec = simaai::neat::OutputSpec{}}};
  return make_stage_node(kind, std::move(inputs), std::move(outputs), out_fn);
}

inline std::shared_ptr<simaai::neat::graph::Node>
make_pass_stage_node(const std::string& kind, const std::string& in_name = "in",
                     const std::string& out_name = "out") {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = in_name, .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = out_name, .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>(kind, std::move(factory), std::move(inputs),
                                     std::move(outputs), kind);
}

inline simaai::neat::Sample make_tensor_sample(int frame_id, const std::string& stream_id,
                                               int64_t pts_ns = -1, uint8_t fill = 0x17) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(12, 8, simaai::neat::ImageSpec::PixelFormat::RGB, fill);
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  sample.pts_ns = pts_ns;
  return sample;
}

inline bool has_field_name(const simaai::neat::Sample& bundle, const std::string& name) {
  for (const auto& field : bundle.fields) {
    if (field.stream_label == name || field.port_name == name)
      return true;
  }
  return false;
}

} // namespace sima_test

#include "graph/nodes/FanOut.h"

#include <stdexcept>
#include <unordered_set>

namespace simaai::neat::graph::nodes {

FanOut::FanOut(FanOutOptions opt) : opt_(std::move(opt)) {}

void FanOut::validate_outputs_(const StagePorts& ports) {
  out_ports_.clear();

  if (opt_.outputs.empty()) {
    const PortId only = ports.only_output();
    if (only == kInvalidPort) {
      throw std::runtime_error("FanOut: outputs not specified and no single output port");
    }
    out_ports_.push_back(only);
    validated_ = true;
    return;
  }

  std::unordered_set<std::string> seen;
  for (const auto& name : opt_.outputs) {
    if (name.empty()) {
      throw std::runtime_error("FanOut: empty output name");
    }
    if (!seen.insert(name).second) {
      throw std::runtime_error("FanOut: duplicate output name: " + name);
    }
    const PortId pid = ports.out_port(name);
    if (pid == kInvalidPort) {
      throw std::runtime_error("FanOut: unknown output port: " + name);
    }
    out_ports_.push_back(pid);
  }
  validated_ = true;
}

void FanOut::set_ports(const StagePorts& ports) {
  validate_outputs_(ports);
}

void FanOut::on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) {
  if (!validated_ || out_ports_.empty())
    return;

  const std::size_t fanout_count = out_ports_.size();
  out.reserve(out.size() + fanout_count);

  for (std::size_t i = 0; i + 1 < fanout_count; ++i) {
    out.push_back(StageOutMsg{.out_port = out_ports_[i], .sample = msg.sample});
  }

  out.push_back(StageOutMsg{.out_port = out_ports_.back(), .sample = std::move(msg.sample)});
}

std::shared_ptr<simaai::neat::graph::Node> FanOutNode(std::vector<std::string> outputs,
                                                      std::string label, std::string input) {
  FanOutOptions opt;
  opt.outputs = std::move(outputs);

  StageNode::StageExecutorFactory factory = [opt]() mutable {
    return std::make_unique<FanOut>(opt);
  };

  std::vector<PortDesc> inputs = {PortDesc{.name = std::move(input), .spec = OutputSpec{}}};
  std::vector<PortDesc> out_ports;
  out_ports.reserve(opt.outputs.size());
  for (const auto& name : opt.outputs) {
    out_ports.push_back(PortDesc{.name = name, .spec = OutputSpec{}});
  }

  StageNode::OutputSpecFn out_fn = [](const std::vector<OutputSpec>& in, PortId) {
    if (!in.empty())
      return in.front();
    return OutputSpec{};
  };

  return std::make_shared<StageNode>("FanOut", std::move(factory), std::move(inputs),
                                     std::move(out_ports), std::move(label), std::move(out_fn));
}

} // namespace simaai::neat::graph::nodes

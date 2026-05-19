#include "graph/nodes/JoinEncodedWithMeta.h"

#include "pipeline/TensorCore.h"

#include <sstream>
#include <stdexcept>

namespace simaai::neat::graph::nodes {

JoinEncodedWithMeta::JoinEncodedWithMeta(JoinEncodedWithMetaOptions opt) : opt_(std::move(opt)) {}

void JoinEncodedWithMeta::set_ports(const StagePorts& ports) {
  const PortId only = ports.only_output();
  if (only != kInvalidPort) {
    out_port_ = only;
  }
}

std::string JoinEncodedWithMeta::make_key_(const Sample& sample) const {
  std::string stream = sample.stream_id.empty() ? "stream0" : sample.stream_id;
  if (sample.pts_ns >= 0) {
    return stream + "|pts|" + std::to_string(sample.pts_ns);
  }
  if (sample.frame_id >= 0) {
    return stream + "|fid|" + std::to_string(sample.frame_id);
  }
  throw std::invalid_argument("JoinEncodedWithMeta: missing pts/frame_id (add StampFrameId)");
}

bool JoinEncodedWithMeta::is_encoded_(PortId port, const Sample& sample) const {
  if (opt_.encoded_port != kInvalidPort)
    return port == opt_.encoded_port;
  if (sample_has_tensor_list(sample)) {
    for (const auto& tensor : sample.tensors) {
      if (tensor.semantic.encoded.has_value())
        return true;
    }
  }
  return false;
}

std::string JoinEncodedWithMeta::field_name_(PortId port, const Sample& sample,
                                             bool encoded) const {
  if (encoded)
    return opt_.encoded_name;
  auto it = opt_.port_names.find(port);
  if (it != opt_.port_names.end() && !it->second.empty())
    return it->second;
  if (!sample.stream_label.empty())
    return sample.stream_label;
  if (!sample.port_name.empty())
    return sample.port_name;
  std::ostringstream oss;
  oss << "port" << port;
  return oss.str();
}

void JoinEncodedWithMeta::evict_if_needed_() {
  if (opt_.max_pending == 0)
    return;
  while (order_.size() > opt_.max_pending) {
    const std::string key = order_.front();
    order_.pop_front();
    pending_.erase(key);
  }
}

void JoinEncodedWithMeta::on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) {
  const Sample& in = msg.sample;
  const std::string key = make_key_(in);

  auto it = pending_.find(key);
  if (it == pending_.end()) {
    pending_[key] = {};
    order_.push_back(key);
    it = pending_.find(key);
  }
  it->second[msg.in_port] = msg.sample;

  evict_if_needed_();

  const bool encoded = is_encoded_(msg.in_port, msg.sample);
  if (!encoded && !opt_.emit_partial)
    return;

  if (!encoded) {
    // If we don't have an encoded sample yet, don't emit.
    bool has_encoded = false;
    for (const auto& [port, sample] : it->second) {
      if (is_encoded_(port, sample)) {
        has_encoded = true;
        break;
      }
    }
    if (!has_encoded)
      return;
  }

  // Build bundle
  Sample bundle;
  bundle.kind = SampleKind::Bundle;
  bundle.stream_id = in.stream_id;
  bundle.frame_id = in.frame_id;
  bundle.pts_ns = in.pts_ns;
  bundle.dts_ns = in.dts_ns;
  bundle.duration_ns = in.duration_ns;
  bundle.input_seq = in.input_seq;
  bundle.orig_input_seq = in.orig_input_seq;

  for (auto& [port, sample] : it->second) {
    const bool is_enc = is_encoded_(port, sample);
    Sample field = sample;
    field.stream_label = field_name_(port, sample, is_enc);
    bundle.fields.emplace_back(std::move(field));
  }

  pending_.erase(key);

  const PortId out_port = (out_port_ == kInvalidPort) ? kInvalidPort : out_port_;
  out.push_back(StageOutMsg{.out_port = out_port, .sample = std::move(bundle)});
}

void JoinEncodedWithMeta::on_tick(std::int64_t /*now_ns*/, std::vector<StageOutMsg>& /*out*/) {
  // Placeholder for timeout/TTL flushing if needed.
}

std::shared_ptr<simaai::neat::graph::Node>
JoinEncodedWithMetaNode(std::vector<std::string> inputs, std::string label, std::string output) {
  JoinEncodedWithMetaOptions opt;
  StageNode::StageExecutorFactory factory = [opt]() mutable {
    return std::make_unique<JoinEncodedWithMeta>(opt);
  };

  std::vector<PortDesc> in_ports;
  in_ports.reserve(inputs.size());
  for (const auto& name : inputs) {
    in_ports.push_back(PortDesc{.name = name, .spec = OutputSpec{}});
  }
  std::vector<PortDesc> out_ports = {PortDesc{.name = output, .spec = OutputSpec{}}};

  return std::make_shared<StageNode>("JoinEncodedWithMeta", std::move(factory), std::move(in_ports),
                                     std::move(out_ports), std::move(label));
}

} // namespace simaai::neat::graph::nodes

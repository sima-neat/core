#include "graph/nodes/StreamScheduler.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace simaai::neat::graph::nodes {
namespace {

bool tensor_is_zero_copy(const simaai::neat::Tensor& t) {
  return t.storage && t.storage->kind == simaai::neat::StorageKind::GstSample;
}

int zero_copy_backpressure_cap() {
  const char* env = std::getenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP");
  if (env && *env)
    return std::max(0, std::atoi(env));
  env = std::getenv("SIMA_GRAPH_ZERO_COPY_MAX_INFLIGHT");
  if (env && *env)
    return std::max(0, std::atoi(env));
  return 1;
}

void force_copy_tensor_if_zero_copy(simaai::neat::Tensor& t) {
  if (!tensor_is_zero_copy(t))
    return;
  t = t.clone();
  t.read_only = false;
}

void force_copy_sample_if_zero_copy(Sample& sample) {
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    force_copy_tensor_if_zero_copy(*sample.tensor);
    sample.owned = true;
    return;
  }
  if (sample.kind != SampleKind::Bundle)
    return;
  for (auto& field : sample.fields) {
    if (field.kind == SampleKind::Tensor && field.tensor.has_value()) {
      force_copy_tensor_if_zero_copy(*field.tensor);
      field.owned = true;
    }
  }
  sample.owned = true;
}

bool sample_has_zero_copy_tensor(const Sample& sample) {
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    return tensor_is_zero_copy(*sample.tensor);
  }
  if (sample.kind != SampleKind::Bundle)
    return false;
  for (const auto& field : sample.fields) {
    if (field.kind == SampleKind::Tensor && field.tensor.has_value()) {
      if (tensor_is_zero_copy(*field.tensor))
        return true;
    }
  }
  return false;
}

void maybe_force_copy_for_backpressure(Sample& sample, std::size_t qsize) {
  if (!sample_has_zero_copy_tensor(sample))
    return;
  const int cap = zero_copy_backpressure_cap();
  if (cap <= 0 || qsize < static_cast<std::size_t>(cap))
    return;
  force_copy_sample_if_zero_copy(sample);
}

} // namespace

StreamScheduler::StreamScheduler(StreamSchedulerOptions opt) : opt_(std::move(opt)) {
  if (opt_.max_batch < 1)
    opt_.max_batch = 1;
}

void StreamScheduler::set_ports(const StagePorts& ports) {
  const PortId only = ports.only_output();
  if (only != kInvalidPort) {
    out_port_ = only;
  }
}

void StreamScheduler::ensure_stream_(const std::string& stream_id) {
  if (stream_id.empty())
    return;
  if (active_.insert(stream_id).second) {
    rr_order_.push_back(stream_id);
  }
}

bool StreamScheduler::emit_one_(std::vector<StageOutMsg>& out) {
  if (rr_order_.empty())
    return false;

  std::size_t attempts = rr_order_.size();
  while (attempts-- > 0 && !rr_order_.empty()) {
    std::string stream_id = rr_order_.front();
    rr_order_.pop_front();

    auto it = queues_.find(stream_id);
    if (it == queues_.end() || it->second.empty()) {
      active_.erase(stream_id);
      continue;
    }

    Sample sample = std::move(it->second.front());
    it->second.pop_front();

    if (!it->second.empty()) {
      rr_order_.push_back(stream_id);
    } else {
      active_.erase(stream_id);
    }

    out.push_back(StageOutMsg{.out_port = out_port_, .sample = std::move(sample)});
    return true;
  }

  return false;
}

void StreamScheduler::on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) {
  Sample sample = std::move(msg.sample);
  std::string stream_id = sample.stream_id.empty() ? "stream0" : sample.stream_id;

  auto& q = queues_[stream_id];
  const bool has_limit = opt_.per_stream_queue > 0;
  if (has_limit && q.size() >= opt_.per_stream_queue) {
    if (opt_.drop_policy == StreamDropPolicy::DropOldest) {
      q.pop_front();
    } else {
      // Drop newest (incoming) sample.
      return;
    }
  }

  maybe_force_copy_for_backpressure(sample, q.size());
  q.push_back(std::move(sample));
  ensure_stream_(stream_id);

  const int max_emit = opt_.max_batch > 0 ? opt_.max_batch : 1;
  for (int i = 0; i < max_emit; ++i) {
    if (!emit_one_(out))
      break;
  }
}

std::shared_ptr<simaai::neat::graph::Node> StreamSchedulerNode(StreamSchedulerOptions opt,
                                                               std::string label, std::string input,
                                                               std::string output) {
  StageNode::StageExecutorFactory factory = [opt]() mutable {
    return std::make_unique<StreamScheduler>(opt);
  };

  std::vector<PortDesc> inputs = {
      PortDesc{.name = std::move(input), .spec = OutputSpec{}, .max_in_edges = 0}};
  std::vector<PortDesc> outputs = {PortDesc{.name = std::move(output), .spec = OutputSpec{}}};

  StageNode::OutputSpecFn out_fn = [](const std::vector<OutputSpec>& in, PortId) {
    if (!in.empty())
      return in.front();
    return OutputSpec{};
  };

  return std::make_shared<StageNode>("StreamScheduler", std::move(factory), std::move(inputs),
                                     std::move(outputs), std::move(label), std::move(out_fn));
}

} // namespace simaai::neat::graph::nodes

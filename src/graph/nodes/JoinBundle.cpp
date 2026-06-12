#include "graph/nodes/JoinBundle.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace simaai::neat::graph::nodes {
namespace {

std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool join_bundle_debug_enabled() {
  const char* raw = std::getenv("SIMA_JOIN_BUNDLE_DEBUG");
  return raw && *raw && std::string(raw) != "0";
}

} // namespace

JoinBundle::JoinBundle(JoinBundleOptions opt) : opt_(std::move(opt)) {}

void JoinBundle::set_ports(const StagePorts& ports) {
  input_names_ = opt_.inputs;
  if (input_names_.empty()) {
    if (ports.in.empty()) {
      throw std::runtime_error("JoinBundle: no input ports defined");
    }
    for (const auto& [name, _] : ports.in) {
      input_names_.push_back(name);
    }
    std::sort(input_names_.begin(), input_names_.end());
  }

  port_names_.clear();
  name_to_port_.clear();
  required_ports_.clear();

  for (const auto& name : input_names_) {
    const PortId pid = ports.in_port(name);
    if (pid == kInvalidPort) {
      throw std::runtime_error("JoinBundle: unknown input port: " + name);
    }
    port_names_[pid] = name;
    name_to_port_[name] = pid;
  }

  if (opt_.required.empty()) {
    if (!opt_.emit_partial) {
      for (const auto& [pid, name] : port_names_) {
        (void)name;
        required_ports_.insert(pid);
      }
    }
  } else {
    for (const auto& name : opt_.required) {
      const PortId pid = ports.in_port(name);
      if (pid == kInvalidPort) {
        throw std::runtime_error("JoinBundle: unknown required port: " + name);
      }
      required_ports_.insert(pid);
    }
  }

  out_port_ = ports.only_output();
  if (out_port_ == kInvalidPort) {
    throw std::runtime_error("JoinBundle: expected a single output port");
  }
}

std::string JoinBundle::make_key_(const Sample& sample) const {
  const std::string stream = sample.stream_id.empty() ? "stream0" : sample.stream_id;
  const std::string prefix = opt_.include_stream_id_in_key ? stream + "|" : std::string{};
  if (opt_.key_policy == JoinKeyPolicy::StreamPts) {
    if (sample.pts_ns >= 0) {
      const std::string key = prefix + "pts|" + std::to_string(sample.pts_ns);
      if (join_bundle_debug_enabled()) {
        std::fprintf(stderr,
                     "[JOIN_BUNDLE] make_key policy=ByPts key=%s stream=%s frame_id=%lld "
                     "pts_ns=%lld dts_ns=%lld duration_ns=%lld include_stream=%d fallback=%d\n",
                     key.c_str(), stream.c_str(), static_cast<long long>(sample.frame_id),
                     static_cast<long long>(sample.pts_ns), static_cast<long long>(sample.dts_ns),
                     static_cast<long long>(sample.duration_ns),
                     opt_.include_stream_id_in_key ? 1 : 0, opt_.allow_key_fallback ? 1 : 0);
      }
      return key;
    }
    if (opt_.allow_key_fallback && sample.frame_id >= 0)
      return prefix + "fid|" + std::to_string(sample.frame_id);
    throw std::runtime_error(opt_.allow_key_fallback
                                 ? "JoinBundle: missing pts/frame_id (add StampFrameId)"
                                 : "JoinBundle: missing pts_ns for strict ByPts combine");
  }

  if (sample.frame_id >= 0)
    return prefix + "fid|" + std::to_string(sample.frame_id);
  if (opt_.allow_key_fallback && sample.pts_ns >= 0)
    return prefix + "pts|" + std::to_string(sample.pts_ns);
  throw std::runtime_error(opt_.allow_key_fallback
                               ? "JoinBundle: missing frame_id/pts (add StampFrameId)"
                               : "JoinBundle: missing frame_id for strict ByFrame combine");
}

void JoinBundle::touch_key_(const std::string& key) {
  for (auto it = order_.begin(); it != order_.end(); ++it) {
    if (*it == key) {
      order_.erase(it);
      break;
    }
  }
  order_.push_back(key);
}

void JoinBundle::evict_expired_(std::int64_t now) {
  if (opt_.timeout_ms <= 0)
    return;
  const std::int64_t deadline = now - static_cast<std::int64_t>(opt_.timeout_ms) * 1000000LL;
  while (!order_.empty()) {
    const std::string& key = order_.front();
    auto it = pending_.find(key);
    if (it == pending_.end()) {
      order_.pop_front();
      continue;
    }
    if (it->second.last_seen_ns >= deadline)
      break;
    pending_.erase(it);
    order_.pop_front();
  }
}

void JoinBundle::evict_oldest_() {
  if (opt_.max_pending_keys == 0)
    return;
  while (pending_.size() > opt_.max_pending_keys && !order_.empty()) {
    const std::string key = order_.front();
    order_.pop_front();
    pending_.erase(key);
  }
}

bool JoinBundle::ready_(const Pending& pending) const {
  if (required_ports_.empty())
    return opt_.emit_partial;
  for (const auto& pid : required_ports_) {
    if (pending.samples.find(pid) == pending.samples.end())
      return false;
  }
  return true;
}

void JoinBundle::erase_key_(const std::string& key) {
  pending_.erase(key);
  for (auto it = order_.begin(); it != order_.end(); ++it) {
    if (*it == key) {
      order_.erase(it);
      break;
    }
  }
}

void JoinBundle::on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) {
  const std::int64_t ts = now_ns();
  const std::string key = make_key_(msg.sample);

  auto& entry = pending_[key];
  entry.samples[msg.in_port] = std::move(msg.sample);
  entry.last_seen_ns = ts;
  touch_key_(key);

  if (join_bundle_debug_enabled()) {
    std::fprintf(stderr,
                 "[JOIN_BUNDLE] on_input port=%u key=%s pending_keys=%zu fields_for_key=%zu "
                 "required=%zu out_before=%zu\n",
                 static_cast<unsigned>(msg.in_port), key.c_str(), pending_.size(),
                 entry.samples.size(), required_ports_.size(), out.size());
  }

  evict_expired_(ts);
  evict_oldest_();

  if (!ready_(entry)) {
    if (join_bundle_debug_enabled()) {
      std::fprintf(stderr,
                   "[JOIN_BUNDLE] not_ready key=%s fields_for_key=%zu required=%zu "
                   "pending_keys=%zu\n",
                   key.c_str(), entry.samples.size(), required_ports_.size(), pending_.size());
    }
    return;
  }

  Sample bundle;
  bundle.kind = SampleKind::Bundle;
  if (!entry.samples.empty()) {
    const auto& seed = entry.samples.begin()->second;
    bundle.stream_id = seed.stream_id;
    bundle.frame_id = seed.frame_id;
    bundle.pts_ns = seed.pts_ns;
    bundle.dts_ns = seed.dts_ns;
    bundle.duration_ns = seed.duration_ns;
    bundle.input_seq = seed.input_seq;
    bundle.orig_input_seq = seed.orig_input_seq;
  }

  for (const auto& name : input_names_) {
    auto it_port = name_to_port_.find(name);
    if (it_port == name_to_port_.end())
      continue;
    const PortId port = it_port->second;

    auto it = entry.samples.find(port);
    if (it == entry.samples.end())
      continue;
    Sample field = std::move(it->second);
    field.stream_label = name;
    bundle.fields.emplace_back(std::move(field));
  }

  erase_key_(key);

  out.push_back(StageOutMsg{.out_port = out_port_, .sample = std::move(bundle)});
  if (join_bundle_debug_enabled()) {
    std::fprintf(stderr, "[JOIN_BUNDLE] emitted key=%s out_port=%u out_after=%zu\n", key.c_str(),
                 static_cast<unsigned>(out_port_), out.size());
  }
}

void JoinBundle::on_tick(std::int64_t now_ns, std::vector<StageOutMsg>& /*out*/) {
  evict_expired_(now_ns);
  evict_oldest_();
}

std::shared_ptr<simaai::neat::graph::Node> JoinBundleNode(std::vector<std::string> inputs,
                                                          std::string label, std::string output,
                                                          JoinBundleOptions opt) {
  opt.inputs = inputs;

  StageNode::StageExecutorFactory factory = [opt]() mutable {
    return std::make_unique<JoinBundle>(opt);
  };

  std::vector<PortDesc> in_ports;
  in_ports.reserve(inputs.size());
  for (const auto& name : inputs) {
    in_ports.push_back(PortDesc{.name = name, .spec = OutputSpec{}});
  }
  std::vector<PortDesc> out_ports = {PortDesc{.name = std::move(output), .spec = OutputSpec{}}};

  StageNode::OutputSpecFn out_fn = [](const std::vector<OutputSpec>& in, PortId) {
    if (!in.empty())
      return in.front();
    return OutputSpec{};
  };

  return std::make_shared<StageNode>("JoinBundle", std::move(factory), std::move(in_ports),
                                     std::move(out_ports), std::move(label), std::move(out_fn));
}

} // namespace simaai::neat::graph::nodes

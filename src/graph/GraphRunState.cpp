#include "internal/GraphRunState.h"

namespace simaai::neat::graph {
void GraphRun::State::signal_stop() {
  if (graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] signal_stop\n");
  }
  stop.store(true);

  for (auto& st : stages) {
    if (st) {
      if (st->exec) {
        try {
          st->exec->request_stop();
        } catch (const std::exception& e) {
          if (graph_debug_enabled()) {
            std::fprintf(stderr, "[GRAPH] stage_request_stop_error node=%zu err=%s\n",
                         static_cast<std::size_t>(st->node_id), e.what());
          }
        }
      }
      st->mailbox.inbox.close();
    }
  }
  for (auto& sink : sinks) {
    if (sink.second)
      sink.second->close();
  }
  for (auto& pipe : pipelines) {
    if (pipe && pipe->input_queue)
      pipe->input_queue->close();
    if (pipe)
      pipe->cv.notify_all();
  }
}

void GraphRun::State::request_stop(const std::string& err) {
  if (graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] request_stop err=%s\n", err.empty() ? "<empty>" : err.c_str());
  }
  {
    std::lock_guard<std::mutex> lock(error_mu);
    if (error.empty())
      error = err;
  }
  signal_stop();
}

bool GraphRun::State::route_stage_output(NodeId node_id, const std::vector<PortId>& output_ports,
                                         StageOutMsg&& out_msg) {
  if (stop.load(std::memory_order_relaxed)) {
    return false;
  }

  PortId out_port = out_msg.out_port;
  if (out_port == kInvalidPort && output_ports.size() == 1) {
    out_port = output_ports[0];
  }

  const std::uint64_t key = port_key(node_id, out_port);
  const auto it = adjacency.find(key);
  if (it == adjacency.end() || it->second.empty()) {
    auto sink_it = sinks.find(node_id);
    if (sink_it == sinks.end()) {
      return true;
    }

    Sample sample_move = std::move(out_msg.sample);
    const std::size_t qsize = sink_it->second->size();
    maybe_force_copy_for_backpressure(sample_move, qsize, "sink_queue", node_id);
    if (!sink_it->second->push(std::move(sample_move), opt.push_timeout_ms)) {
      if (!stop.load(std::memory_order_relaxed)) {
        std::ostringstream msg;
        msg << "GraphRun: sink backpressure timeout (node=" << static_cast<std::size_t>(node_id)
            << ", edge_queue=" << opt.edge_queue << ", push_timeout_ms=" << opt.push_timeout_ms
            << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
        request_stop(msg.str());
      }
      return false;
    }
    return true;
  }

  auto dispatch_target = [&](const DownstreamTarget& target, Sample&& sample) -> bool {
    if (target.kind == DownstreamTarget::Kind::StageGroup) {
      return dispatch_to_stage_group(target.index, target.port, std::move(sample));
    }

    if (target.kind == DownstreamTarget::Kind::PipelineInput) {
      std::string build_err;
      if (!ensure_pipeline_built(target.index, sample, &build_err)) {
        request_stop(build_err.empty() ? "GraphRun: pipeline build failed" : build_err);
        return false;
      }
      sanitize_sample_for_pipeline_input(*pipelines[target.index], sample);
      auto& input_queue = pipelines[target.index]->input_queue;
      if (!input_queue) {
        return true;
      }
      if (!input_queue->push(std::move(sample), opt.push_timeout_ms)) {
        if (!stop.load(std::memory_order_relaxed)) {
          std::ostringstream msg;
          msg << "GraphRun: pipeline input backpressure timeout (seg="
              << static_cast<std::size_t>(pipelines[target.index]->seg.id)
              << ", edge_queue=" << opt.edge_queue << ", push_timeout_ms=" << opt.push_timeout_ms
              << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
          request_stop(msg.str());
        }
        return false;
      }
      return true;
    }

    const NodeId sink_node = static_cast<NodeId>(target.index);
    auto sink_it = sinks.find(sink_node);
    if (sink_it == sinks.end()) {
      return true;
    }
    const std::size_t qsize = sink_it->second->size();
    maybe_force_copy_for_backpressure(sample, qsize, "sink_queue", target.index);
    if (!sink_it->second->push(std::move(sample), opt.push_timeout_ms)) {
      if (!stop.load(std::memory_order_relaxed)) {
        std::ostringstream msg;
        msg << "GraphRun: sink backpressure timeout (node=" << static_cast<std::size_t>(sink_node)
            << ", edge_queue=" << opt.edge_queue << ", push_timeout_ms=" << opt.push_timeout_ms
            << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
        request_stop(msg.str());
      }
      return false;
    }
    return true;
  };

  const auto& targets = it->second;
  bool ok = true;
  if (targets.size() == 1) {
    ok = dispatch_target(targets.front(), std::move(out_msg.sample));
  } else {
    for (const auto& target : targets) {
      Sample sample_copy = out_msg.sample;
      ok = dispatch_target(target, std::move(sample_copy)) && ok;
    }
  }
  return ok;
}

bool GraphRun::State::ensure_pipeline_built(std::size_t index, const Sample& sample,
                                            std::string* err) {
  if (index >= pipelines.size()) {
    if (err)
      *err = "GraphRun: pipeline index out of range";
    return false;
  }
  auto& pipe = *pipelines[index];
  if (pipe.built.load(std::memory_order_acquire))
    return true;
  if (stop.load()) {
    if (err)
      *err = "GraphRun: stopped";
    return false;
  }

  {
    std::unique_lock<std::mutex> lock(pipe.mu);
    if (pipe.built.load(std::memory_order_acquire))
      return true;
    if (pipe.building) {
      pipe.cv.wait(lock, [&] { return pipe.built.load(std::memory_order_acquire) || stop.load(); });
      if (pipe.built.load(std::memory_order_acquire))
        return true;
      if (err)
        *err = "GraphRun: stopped";
      return false;
    }
    pipe.building = true;
  }

  Sample build_sample =
      simaai::neat::pipeline_internal::canonicalize_tensor_transport_sample(sample);

  try {
    if (graph_debug_enabled()) {
      std::fprintf(stderr, "[GRAPH] pipeline_build seg=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id));
    }
    std::unique_lock<std::mutex> serial_lock;
    if (graph_serial_pipeline_build_enabled()) {
      if (graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] pipeline_build_lock_wait seg=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id));
      }
      serial_lock = std::unique_lock<std::mutex>(graph_pipeline_build_mu());
      if (graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] pipeline_build_lock_acquired seg=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id));
      }
    }
    Run run = pipe.session.build(SampleList{build_sample}, RunMode::Async, pipe.run_options);
    {
      std::lock_guard<std::mutex> lock(pipe.mu);
      pipe.run = std::move(run);
      pipe.built.store(true, std::memory_order_release);
      pipe.building = false;
    }
    if (graph_debug_enabled()) {
      std::fprintf(stderr, "[GRAPH] pipeline_built seg=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id));
    }
    pipe.cv.notify_all();
    return true;
  } catch (const std::exception& e) {
    {
      std::lock_guard<std::mutex> lock(pipe.mu);
      pipe.building = false;
    }
    pipe.cv.notify_all();
    if (err)
      *err = e.what();
    return false;
  }
}
} // namespace simaai::neat::graph

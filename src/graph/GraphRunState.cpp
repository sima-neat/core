#include "internal/GraphRunState.h"

namespace simaai::neat::graph {
void GraphRun::State::signal_stop() {
  if (graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] signal_stop\n");
  }
  stop.store(true);

  for (auto& st : stages) {
    if (st)
      st->mailbox.inbox.close();
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

  Sample build_sample = sample;
  if (sample.kind == SampleKind::Bundle && !pipe.seg.input_complete) {
    build_sample = make_bundle_carrier_sample();
  }

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
    Run run = pipe.session.build(build_sample, RunMode::Async, opt.pipeline);
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

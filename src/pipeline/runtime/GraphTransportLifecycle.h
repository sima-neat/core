#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <atomic>
#include <mutex>
#include <utility>

namespace simaai::neat::runtime {

// These operations are shared by the graph's public pull, transport workers and stop paths.
// The small transport parameter also permits lock-order tests without scheduling-dependent
// sleeps. They do not introduce a second lifecycle or replace the transport's existing state.
template <class Transport> void publish_graph_transport_completion(Transport& transport) {
  {
    std::lock_guard lock(transport.mu);
    transport.completion_forwarded.store(true, std::memory_order_release);
  }
  transport.cv.notify_all();
}

template <class Transport> void notify_graph_transport_external_state(Transport& transport) {
  // The stop/ingress-close predicate was published under its own authority. Synchronize with
  // the waiter mutex before notifying so a false predicate check cannot lose this notification.
  // In particular, the caller must have released public_ingress_mu before entering this function.
  { std::lock_guard lock(transport.mu); }
  transport.cv.notify_all();
}

template <class Pipeline, class Construct>
void construct_graph_source_pipeline(Pipeline& pipeline, Construct&& construct) {
  // Requires an empty/unpublished child and a single startup owner. This is not a concurrent
  // rebuild/close helper; callers retain lifecycle ownership until construction is published.
  // GStreamer construction can synchronously invoke dispatch/stop callbacks. Never hold the
  // transport waiter mutex across it. Keep the returned child owned even if stop was requested
  // during construction: publishing it lets the normal graph teardown find and close it.
  auto child = std::forward<Construct>(construct)();
  {
    std::lock_guard lock(pipeline.transport.mu);
    pipeline.run_core = std::move(child);
    pipeline.transport.built.store(true, std::memory_order_release);
  }
  pipeline.transport.cv.notify_all();
}

} // namespace simaai::neat::runtime

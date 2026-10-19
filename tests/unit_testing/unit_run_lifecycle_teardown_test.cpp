#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "gst/GstInit.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/GraphTransportLifecycle.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class EnvVarGuard {
public:
  EnvVarGuard(const char* key, const char* value) : key_(key), had_(false) {
    const char* cur = std::getenv(key_);
    if (cur && *cur) {
      had_ = true;
      old_ = cur;
    }
    ::setenv(key_, value, 1);
  }

  ~EnvVarGuard() {
    if (had_) {
      ::setenv(key_, old_.c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_;
  std::string old_;
};

bool is_expected_teardown_error(const std::string& msg) {
  return msg.find("stream is stopping") != std::string::npos ||
         msg.find("EOS has been reached") != std::string::npos ||
         msg.find("gst_app_src_end_of_stream failed (flow=-2:flushing)") != std::string::npos ||
         msg.find("stream closed") != std::string::npos;
}

struct BlockingProbe {
  std::mutex mu;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;
};

GstPadProbeReturn block_pipeline_output(GstPad*, GstPadProbeInfo*, gpointer user_data) {
  auto* probe = static_cast<BlockingProbe*>(user_data);
  std::unique_lock<std::mutex> lock(probe->mu);
  probe->entered = true;
  probe->cv.notify_all();
  probe->cv.wait(lock, [&] { return probe->released; });
  return GST_PAD_PROBE_REMOVE;
}

GstElement* find_appsink(GstElement* pipeline) {
  GstIterator* iterator = gst_bin_iterate_sinks(GST_BIN(pipeline));
  require(iterator != nullptr, "run lifecycle teardown: pipeline has no sink iterator");

  GstElement* appsink = nullptr;
  GValue item = G_VALUE_INIT;
  while (!appsink) {
    const GstIteratorResult result = gst_iterator_next(iterator, &item);
    if (result == GST_ITERATOR_OK) {
      auto* candidate = GST_ELEMENT(g_value_get_object(&item));
      if (GST_IS_APP_SINK(candidate)) {
        appsink = GST_ELEMENT(gst_object_ref(candidate));
      }
      g_value_reset(&item);
      continue;
    }
    if (result == GST_ITERATOR_RESYNC) {
      gst_iterator_resync(iterator);
      continue;
    }
    break;
  }
  g_value_unset(&item);
  gst_iterator_free(iterator);
  return appsink;
}

bool input_dequeued_after(simaai::neat::runtime::RunCore& core, std::uint64_t enqueued_before) {
  if (core.inputs_enqueued.load(std::memory_order_acquire) <= enqueued_before) {
    return false;
  }
  std::lock_guard<std::mutex> lock(core.pipeline.in_mu);
  return core.pipeline.in_queue.empty();
}

void graph_completion_propagates_before_locked_publication() {
  using namespace simaai::neat;
  auto core = runtime::RunCore::create_graph_compat();
  auto& execution = core->graph_execution();
  auto segment = std::make_unique<runtime::PipelineSegmentRuntime>();
  constexpr graph::NodeId kOutput = 47;
  segment->seg.node_ids = {kOutput};
  execution.pipelines.push_back(std::move(segment));
  auto sink = std::make_shared<runtime::GraphSinkQueue>();
  sink->set_producer_count(1);
  execution.sinks.emplace(kOutput, sink);
  auto& transport = execution.pipelines.front()->transport;

  // This is the actual RunCore producer-completion path, not a replacement implementation.
  // Holding its publication mutex must not prevent downstream closure from reaching the sink.
  std::unique_lock lock(transport.mu);
  std::thread complete([&] { core->graph_pipeline_completed(0); });
  runtime::RuntimeSinkQueueMsg message;
  const bool received_sample = sink->pop(message, 2000);
  const bool propagated = !received_sample && sink->closed();
  const bool published_while_locked =
      transport.completion_forwarded.load(std::memory_order_acquire);
  lock.unlock();
  complete.join();
  require(propagated, "completion held the transport mutex across downstream propagation");
  require(!published_while_locked, "completion publication bypassed the transport waiter mutex");
  require(transport.completion_forwarded.load(std::memory_order_acquire),
          "completion was not published after downstream closure");
  core->close();
}

void stopped_source_construction_publishes_child_for_normal_teardown() {
  using namespace simaai::neat;
  EnvVarGuard synchronous("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0");
  EnvVarGuard no_async("SIMA_GST_TEARDOWN_ASYNC", "0");
  Graph graph;
  graph.add(nodes::Custom("videotestsrc num-buffers=1 pattern=black ! "
                          "video/x-raw,format=RGB,width=32,height=24,framerate=30/1",
                          InputRole::Source));
  graph.add(nodes::Output("output", OutputOptions::EveryFrame(2)));
  RunOptions options;
  options.queue_depth = 2;
  const auto plan = runtime::compile_public_graph(graph, options, std::nullopt);
  require(plan.pipeline_segments.size() == 1, "source ownership fixture must compile one segment");
  auto parent = runtime::RunCore::create_graph_compat();
  auto segment = std::make_unique<runtime::PipelineSegmentRuntime>();
  segment->seg = plan.pipeline_segments.front();
  parent->graph_execution().pipelines.push_back(std::move(segment));
  auto& pipeline = *parent->graph_execution().pipelines.front();
  std::weak_ptr<runtime::RunCore> observed;

  runtime::construct_graph_source_pipeline(pipeline, [&] {
    runtime::RunCoreStartOptions start;
    start.run_options = options;
    start.mode = RunMode::Async;
    auto child = runtime::RunCore::start_pipeline_segment(pipeline.seg, std::move(start));
    observed = child;
    // As in a streaming callback during startup, stop is requested before the just-created
    // child's ownership has been published. The normal parent close must still find it.
    parent->graph_signal_stop();
    return child;
  });
  require(parent->graph_stop_requested() && pipeline.transport.built.load() && pipeline.run_core,
          "construction-time stop left a running child outside graph ownership");
  auto child = observed.lock();
  require(child && child == pipeline.run_core, "constructed child identity changed at publication");
  parent->close();
  require(child->closed.load() &&
              child->stream_close_state.load() == runtime::InputStreamCloseState::Closed,
          "normal graph teardown did not close the child created during stop");
}

template <class Predicate>
bool wait_until(Predicate&& predicate, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return predicate();
}

void failed_start_releases_runtime_guard(bool connected) {
  using namespace simaai::neat;

  Graph graph;
  if (connected) {
    Graph source("image");
    source.add(nodes::Input("image"));
    Graph sink("output");
    sink.add(nodes::Output("output", OutputOptions::EveryFrame(4)));
    graph.connect(source, sink);
  } else {
    graph.add(nodes::Input("image"));
    graph.add(nodes::Output("output", OutputOptions::EveryFrame(4)));
  }

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x29);
  const Sample seed_sample = sample_from_tensors(TensorList{seed});
  RunOptions run_options;
  run_options.queue_depth = 4;
  runtime::ExecutionGraphPlan plan = runtime::compile_public_graph(graph, run_options, seed_sample);

  runtime::RunCoreStartOptions start_options;
  start_options.run_options = run_options;
  start_options.mode = RunMode::Async;
  start_options.seed = seed_sample;
  start_options.graph_options = runtime::graph_runtime_options_from_run_options(run_options);
  // Exercise the existing external-resource lifetime guard, not a decoder lease.
  start_options.guard = std::make_shared<int>(0);
  start_options.graph_verbose_guard = start_options.guard;
  const std::weak_ptr<void> resource = start_options.guard;
  start_options.after_pipeline_start_for_test = [] {
    throw std::runtime_error("injected failure after pipeline start");
  };

  bool threw = false;
  try {
    (void)runtime::RunCore::start(std::move(plan), std::move(start_options));
  } catch (const NeatError& error) {
    threw = true;
    require(std::string(error.what()).find("injected failure after pipeline start") !=
                std::string::npos,
            "startup rollback should preserve the original failure; actual=" +
                std::string(error.what()));
    require(error.report().error_code == error_codes::kPipelineShape &&
                error.report().repro_note.find("injected failure after pipeline start") !=
                    std::string::npos &&
                std::string(error.what()).find(error.report().error_code) != std::string::npos,
            "startup rollback must preserve the same code and detail in its structured report");
  }
  require(threw, "startup failure injection did not throw");
  require(wait_until([&] { return resource.expired(); }, std::chrono::seconds(3)),
          "failed startup did not release its runtime resource guard after teardown");
}

// CPU-only lifecycle proof. This does not qualify accelerator retirement;
// real device acceptance must additionally verify the provider's completed work.
void lazy_failure_after_completed_run_closes_on_assertion(bool explicit_close) {
  using namespace simaai::neat;
  EnvVarGuard synchronous("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0");
  EnvVarGuard no_async("SIMA_GST_TEARDOWN_ASYNC", "0");
  const Tensor seed = make_color_tensor(32, 24, ImageSpec::PixelFormat::RGB, 0x37);
  {
    // A failed second graph must not make qualification forget work completed
    // earlier in the same process (the original test runs ResNet before MNIST).
    Run prior = sima_test::make_async_rgb_run(seed, 2, 2);
    require(prior.try_push(TensorList{seed}), "prior CPU sub-run must accept work");
    const auto output = prior.pull(2000);
    require(output.has_value(), "prior CPU sub-run must complete before lazy rejection");
    auto observed = run_internal::core(prior);
    prior.close();
    prior.close();
    require(observed && observed->closed.load() &&
                observed->stream_close_state.load() == runtime::InputStreamCloseState::Closed,
            "prior sub-run was not synchronously closed");
  }

  Graph source("image");
  InputOptions input;
  input.payload_type = PayloadType::Image;
  input.format = FormatTag::RGB;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  source.add(nodes::Input("image", input));
  Graph output("output");
  output.add(nodes::Output("output", OutputOptions::EveryFrame(2)));
  Graph graph;
  graph.connect(source, output);
  std::shared_ptr<const runtime::RunCore> observed;
  struct ClearHook {
    ~ClearHook() {
      session_test::clear_composition_failure_for_test();
    }
  } clear;
  bool caught_assertion = false;
  try {
    Run run = graph.build(); // Deliberately no seed: compile on the first push.
    observed = run_internal::core(run);
    require(observed != nullptr, "lazy build must return its retained RunCore");
    session_test::arm_composition_failure_for_test(
        session_test::CompositionFailurePoint::PipelinePrepared);
    const bool accepted = run.push("image", TensorList{seed});
    const std::string diagnostic = run.last_error();
    session_test::clear_composition_failure_for_test();
    require(!accepted && diagnostic.find("injected Graph composition failure") != std::string::npos,
            "first lazy push must expose the original preparation failure: " + diagnostic);
    if (explicit_close) {
      run.close();
      run.close();
    }
    // Exercise actual Run stack unwinding, not merely a successful explicit close.
    throw std::runtime_error("reviewed post-failure assertion");
  } catch (const std::runtime_error& error) {
    require(std::string(error.what()) == "reviewed post-failure assertion",
            "unexpected lazy failure lifecycle diagnostic: " + std::string(error.what()));
    caught_assertion = true;
  }
  require(caught_assertion && observed && observed->closed.load() &&
              observed->stream_close_state.load() == runtime::InputStreamCloseState::Closed,
          "destructor/assertion route must close the actual lazy RunCore");
  require(observed->pipeline.stream.pipeline_handle() == nullptr,
          "failed lazy RunCore must not retain a live pipeline");
}

void detached_stream_stop_retains_runtime_until_cleanup() {
  using namespace simaai::neat;

  EnvVarGuard stream_stop_timeout("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", "100");
  EnvVarGuard input_stop_timeout("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "100");
  EnvVarGuard stop_flush("SIMA_INPUTSTREAM_STOP_FLUSH", "0");
  EnvVarGuard synchronous_teardown("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0");

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x4B);
  Run run = sima_test::make_async_rgb_run(seed, 8, 8);
  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));
  GstElement* pipeline = core->pipeline.stream.pipeline_handle();
  require(pipeline != nullptr, "run lifecycle teardown: missing pipeline");
  GstElement* appsink = find_appsink(pipeline);
  require(appsink != nullptr, "run lifecycle teardown: missing appsink");
  GstPad* sink_pad = gst_element_get_static_pad(appsink, "sink");
  require(sink_pad != nullptr, "run lifecycle teardown: missing appsink sink pad");

  BlockingProbe probe;
  gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, block_pipeline_output, &probe, nullptr);
  require(run.try_push(TensorList{seed}), "run lifecycle teardown: push failed");
  {
    std::unique_lock<std::mutex> lock(probe.mu);
    require(probe.cv.wait_for(lock, std::chrono::seconds(2), [&] { return probe.entered; }),
            "run lifecycle teardown: blocking probe was not reached");
  }

  std::weak_ptr<const runtime::RunCore> weak_core = core;
  core.reset();
  const auto close_started_at = std::chrono::steady_clock::now();
  run.close();
  const int close_ms = sima_test::elapsed_ms(close_started_at, std::chrono::steady_clock::now());
  auto retained_core = weak_core.lock();
  // The run's handle is gone, yet the RunCore lives: cleanup was deferred, not inline.
  const bool runtime_retained_past_close = retained_core != nullptr;
  retained_core.reset();

  {
    std::lock_guard<std::mutex> lock(probe.mu);
    probe.released = true;
  }
  probe.cv.notify_all();

  const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!weak_core.expired() && std::chrono::steady_clock::now() < cleanup_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  gst_object_unref(sink_pad);
  gst_object_unref(appsink);

  require(close_ms < 1000, "Run::close must remain bounded when stream teardown blocks");
  require(runtime_retained_past_close,
          "detached stream teardown must retain its RunCore until the blocked work exits");
  require(weak_core.expired(), "retained RunCore was not released after teardown completed");
}

void detached_stream_close_keeps_measurement_reads_safe() {
  using namespace simaai::neat;

  EnvVarGuard stream_stop_timeout("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", "100");
  EnvVarGuard input_stop_timeout("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "100");
  EnvVarGuard stop_flush("SIMA_INPUTSTREAM_STOP_FLUSH", "0");
  EnvVarGuard synchronous_teardown("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0");

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x5C);
  Run run = sima_test::make_async_rgb_run(seed, 8, 8);
  MeasureScope measurement = run.start_measurement();
  auto core = run_internal::core(run);
  GstElement* appsink = find_appsink(core->pipeline.stream.pipeline_handle());
  require(appsink != nullptr, "measurement teardown: missing appsink");
  GstPad* sink_pad = gst_element_get_static_pad(appsink, "sink");
  require(sink_pad != nullptr, "measurement teardown: missing appsink sink pad");

  BlockingProbe probe;
  gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, block_pipeline_output, &probe, nullptr);
  require(run.try_push(TensorList{seed}), "measurement teardown: push failed");
  {
    std::unique_lock<std::mutex> lock(probe.mu);
    require(probe.cv.wait_for(lock, std::chrono::seconds(2), [&] { return probe.entered; }),
            "measurement teardown: blocking probe was not reached");
  }

  run.close();
  // The concurrency assertions below are vacuous unless teardown deferred, so pin the owner.
  require(core->stream_close_state.load(std::memory_order_acquire) ==
              runtime::InputStreamCloseState::StreamStopThreadOwns,
          "measurement teardown: stream stop was not detached");

  std::atomic<bool> start{false};
  std::atomic<bool> keep_reading{true};
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (keep_reading.load(std::memory_order_acquire)) {
        (void)core->input_stats();
        (void)core->diag_snapshot();
      }
    });
  }
  std::exception_ptr measurement_error;
  std::thread stop_measurement([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    try {
      (void)measurement.stop();
    } catch (...) {
      measurement_error = std::current_exception();
    }
  });

  start.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(probe.mu);
    probe.released = true;
  }
  probe.cv.notify_all();

  stop_measurement.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  keep_reading.store(false, std::memory_order_release);
  for (auto& reader : readers) {
    reader.join();
  }
  gst_object_unref(sink_pad);
  gst_object_unref(appsink);
  if (measurement_error) {
    std::rethrow_exception(measurement_error);
  }
}

// Force the ownership handoff without blocking stream stop: latency_mu holds the input thread
// after dequeue and before stream.push(), while the stop path never takes that mutex. Drain
// first because on_output shares latency_mu and would stall stream stop instead.
void input_thread_timeout_hands_off_stream_close() {
  using namespace simaai::neat;

  EnvVarGuard input_stop_timeout("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "50");

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x6D);
  Run run = sima_test::make_async_rgb_run(seed, 8, 8);
  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));

  require(run.try_push(TensorList{seed}), "input thread handoff: warmup push failed");
  (void)run.pull(2000);

  std::unique_lock<std::mutex> timing_lock(core->latency_mu);
  const std::uint64_t enqueued_before = core->inputs_enqueued.load(std::memory_order_acquire);
  require(run.try_push(TensorList{seed}), "input thread handoff: wedging push failed");
  require(wait_until([&] { return input_dequeued_after(*core, enqueued_before); },
                     std::chrono::seconds(3)),
          "input thread handoff: input was not dequeued before teardown");

  const auto close_started_at = std::chrono::steady_clock::now();
  run.close();
  const int close_ms = sima_test::elapsed_ms(close_started_at, std::chrono::steady_clock::now());

  require(close_ms < 2000, "Run::close must stay bounded when the input thread is detached");
  // This state is the proof that close() skipped teardown; the pipeline handle is not,
  // because InputStream::stop() clears it on the normal path too.
  require(core->stream_close_state.load(std::memory_order_acquire) ==
              runtime::InputStreamCloseState::InputThreadOwns,
          "close ownership was not transferred to the detached input thread");
  require(core->pipeline.stream.can_push(),
          "RunCore closed the InputStream after handing ownership to the input thread");

  timing_lock.unlock();

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (core->stream_close_state.load(std::memory_order_acquire) !=
             runtime::InputStreamCloseState::Closed &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  require(core->stream_close_state.load(std::memory_order_acquire) ==
              runtime::InputStreamCloseState::Closed,
          "the detached input thread never completed the close it owned");
  // Race-free only because the acquire load above pairs with the closer's release store.
  require(core->pipeline.stream.pipeline_handle() == nullptr,
          "InputStream close state reached Closed without releasing the GStreamer pipeline");
}

void concurrent_stop_has_one_thread_owner() {
  using namespace simaai::neat;

  EnvVarGuard stream_stop_timeout("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", "0");
  EnvVarGuard input_stop_timeout("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "50");

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x71);
  Run run = sima_test::make_async_rgb_run(seed, 8, 8);
  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));

  require(run.try_push(TensorList{seed}), "concurrent stop: warmup push failed");
  (void)run.pull(2000);

  std::unique_lock<std::mutex> timing_lock(core->latency_mu);
  const std::uint64_t enqueued_before = core->inputs_enqueued.load(std::memory_order_acquire);
  require(run.try_push(TensorList{seed}), "concurrent stop: wedging push failed");
  require(wait_until([&] { return input_dequeued_after(*core, enqueued_before); },
                     std::chrono::seconds(3)),
          "concurrent stop: input was not dequeued before teardown");

  std::atomic<bool> start{false};
  std::exception_ptr stop_errors[2];
  std::thread stoppers[2];
  for (std::size_t i = 0; i < 2; ++i) {
    stoppers[i] = std::thread([&, i] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        core->stop();
      } catch (...) {
        stop_errors[i] = std::current_exception();
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& stopper : stoppers) {
    stopper.join();
  }
  timing_lock.unlock();

  for (const auto& error : stop_errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  require(wait_until(
              [&] {
                return core->stream_close_state.load(std::memory_order_acquire) ==
                       runtime::InputStreamCloseState::Closed;
              },
              std::chrono::seconds(3)),
          "concurrent stop: detached input thread did not finish closing the stream");
}

// Same handoff under an active MeasureScope: a child that detaches closes its own stream
// during stop_graph(), so its diag context must already have been snapshotted or the
// measurement loses every per-node metric.
void composite_child_handoff_preserves_measurement_node_metrics() {
  using namespace simaai::neat;

  EnvVarGuard input_stop_timeout("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "50");

  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x7E);
  Graph source("image");
  source.add(nodes::Input("image"));
  Graph sink("output");
  sink.add(nodes::Output("output", OutputOptions::EveryFrame(8)));
  Graph graph;
  graph.connect(source, sink);

  Run run = graph.build(TensorList{seed});
  MeasureScope scope = run.start_measurement();
  require(run.push("image", TensorList{seed}), "composite handoff: measured push failed");
  require(run.pull("output", 4000).has_value(), "composite handoff: measured pull failed");

  auto core = std::const_pointer_cast<runtime::RunCore>(run_internal::core(run));
  require(core->graph_execution_ != nullptr, "composite handoff: run is not composite");

  // Wedge every child after dequeue and before stream.push().
  std::vector<std::unique_lock<std::mutex>> child_locks;
  std::vector<std::shared_ptr<runtime::RunCore>> children;
  std::vector<std::uint64_t> enqueued_before;
  for (auto& pipe : core->graph_execution_->pipelines) {
    if (pipe && pipe->run_core) {
      children.push_back(pipe->run_core);
      enqueued_before.push_back(pipe->run_core->inputs_enqueued.load(std::memory_order_acquire));
      child_locks.emplace_back(pipe->run_core->latency_mu);
    }
  }
  require(!children.empty(), "composite handoff: no built child pipelines");
  require(run.push("image", TensorList{seed}), "composite handoff: wedging push failed");
  require(wait_until(
              [&] {
                for (std::size_t i = 0; i < children.size(); ++i) {
                  if (input_dequeued_after(*children[i], enqueued_before[i])) {
                    return true;
                  }
                }
                return false;
              },
              std::chrono::seconds(3)),
          "composite handoff: no child dequeued the wedged input");

  run.close();

  const bool any_child_detached =
      std::any_of(children.begin(), children.end(), [](const auto& child) {
        return child->stream_close_state.load(std::memory_order_acquire) ==
               runtime::InputStreamCloseState::InputThreadOwns;
      });
  child_locks.clear();

  require(wait_until(
              [&] {
                return std::all_of(children.begin(), children.end(), [](const auto& child) {
                  return child->stream_close_state.load(std::memory_order_acquire) ==
                         runtime::InputStreamCloseState::Closed;
                });
              },
              std::chrono::seconds(3)),
          "graph child did not close its stream after detached teardown completed");

  const MeasureReport report = scope.stop();
  require(any_child_detached,
          "composite handoff: no child input thread was detached, so the metric-retention "
          "path was never exercised");
  require(!report.node_metrics.empty(),
          "a child that detached its input thread must not cost the active MeasureScope its "
          "per-node metrics");
}

} // namespace

RUN_TEST("unit_run_lifecycle_teardown_test", ([] {
           using namespace simaai::neat;

           // CPU lifecycle is independent of the full-production MLA startup
           // prerequisite. The isolated installed-startup consumers exercise
           // that policy without this supported manual-initialization profile.
           EnvVarGuard manual_init("SIMA_ALLOW_GST_INIT", "1");
           gst_init(nullptr, nullptr);
           gst_init_once();
           // CTest can suppress initial system discovery. Load only Core's
           // selected real system-plugin cohort after its path setup.
           if (const char* paths = g_getenv("GST_PLUGIN_SYSTEM_PATH_1_0")) {
             gchar** directories = g_strsplit(paths, G_SEARCHPATH_SEPARATOR_S, -1);
             for (gchar** directory = directories; *directory; ++directory) {
               if (**directory)
                 gst_registry_scan_path(gst_registry_get(), *directory);
             }
             g_strfreev(directories);
           }

           EnvVarGuard input_stop_1("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "200");

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x3A);
           Run run = sima_test::make_async_rgb_run(seed, 128, 128);

           std::atomic<bool> keep_running{true};
           std::atomic<bool> teardown_started{false};
           std::atomic<int> pushes{0};
           std::atomic<int> pulls{0};
           std::mutex err_mu;
           std::string producer_err;
           std::string consumer_err;

           std::thread producer([&] {
             try {
               while (keep_running.load(std::memory_order_relaxed)) {
                 if (run.try_push(TensorList{seed})) {
                   pushes.fetch_add(1, std::memory_order_relaxed);
                 } else {
                   std::this_thread::sleep_for(std::chrono::milliseconds(1));
                 }
               }
             } catch (const std::exception& e) {
               if (teardown_started.load(std::memory_order_acquire) &&
                   is_expected_teardown_error(e.what())) {
                 return;
               }
               std::lock_guard<std::mutex> lock(err_mu);
               producer_err = e.what();
             }
           });

           std::thread consumer([&] {
             try {
               while (keep_running.load(std::memory_order_relaxed)) {
                 auto out = run.pull(20);
                 if (out.has_value()) {
                   pulls.fetch_add(1, std::memory_order_relaxed);
                 }
               }
             } catch (const std::exception& e) {
               if (teardown_started.load(std::memory_order_acquire) &&
                   is_expected_teardown_error(e.what())) {
                 return;
               }
               std::lock_guard<std::mutex> lock(err_mu);
               consumer_err = e.what();
             }
           });

           std::this_thread::sleep_for(std::chrono::milliseconds(200));

           const auto t0 = std::chrono::steady_clock::now();
           teardown_started.store(true, std::memory_order_release);
           run.close_input();
           run.stop();
           run.stop(); // idempotent stop under prior teardown
           const auto t1 = std::chrono::steady_clock::now();

           keep_running.store(false, std::memory_order_relaxed);
           producer.join();
           consumer.join();

           {
             std::lock_guard<std::mutex> lock(err_mu);
             if (!producer_err.empty()) {
               throw std::runtime_error("run lifecycle producer failed: " + producer_err);
             }
             if (!consumer_err.empty()) {
               throw std::runtime_error("run lifecycle consumer failed: " + consumer_err);
             }
           }

           const int stop_ms = sima_test::elapsed_ms(t0, t1);
           require(stop_ms < 5000, "Run::stop teardown exceeded expected bound");

           require(!run.try_push(TensorList{seed}), "Run::try_push should fail after stop()");
           (void)pushes.load(std::memory_order_relaxed);
           (void)pulls.load(std::memory_order_relaxed);

           detached_stream_stop_retains_runtime_until_cleanup();
           detached_stream_close_keeps_measurement_reads_safe();
           input_thread_timeout_hands_off_stream_close();
           concurrent_stop_has_one_thread_owner();
           composite_child_handoff_preserves_measurement_node_metrics();
           failed_start_releases_runtime_guard(false);
           failed_start_releases_runtime_guard(true);
           graph_completion_propagates_before_locked_publication();
           stopped_source_construction_publishes_child_for_normal_teardown();
           lazy_failure_after_completed_run_closes_on_assertion(false);
           lazy_failure_after_completed_run_closes_on_assertion(true);
         }));

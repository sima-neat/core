#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/EdgeRouter.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"
#include "nodes/common/Output.h"
#include "graphs/Fragments.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/HttpSource.h"
#include "nodes/io/Input.h"

#include <gst/gst.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::shared_ptr<simaai::neat::runtime::RunCore>
make_balanced_zero_copy_core(bool public_output_contract) {
  simaai::neat::RunOptions opt;
  opt.preset = simaai::neat::RunPreset::Balanced;
  opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;

  simaai::neat::InputStreamOptions stream_opt;
  stream_opt.copy_output = false;
  stream_opt.public_output_contract = public_output_contract;

  return simaai::neat::runtime::RunCore::start_single_pipeline(
      simaai::neat::InputStream{}, opt, stream_opt, simaai::neat::RunMode::Async);
}

simaai::neat::Sample make_fake_device_gst_sample(int id) {
  simaai::neat::Tensor tensor;
  tensor.device.type = simaai::neat::DeviceType::SIMA_CVU;
  tensor.storage = std::make_shared<simaai::neat::TensorBuffer>();
  tensor.storage->kind = simaai::neat::StorageKind::GstSample;
  tensor.storage->device.type = simaai::neat::DeviceType::SIMA_CVU;
  tensor.storage->size_bytes = 1;
  tensor.storage->holder = std::make_shared<int>(id);

  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::TensorSet;
  sample.tensors.push_back(std::move(tensor));
  return sample;
}

simaai::neat::Sample make_device_gst_sample_with_external_ref(GstSample** external_ref) {
  gst_init(nullptr, nullptr);
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 1, nullptr);
  GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "test GstSample allocation should succeed");
  if (external_ref) {
    *external_ref = gst_sample_ref(sample);
  }

  auto storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  require(storage != nullptr, "test GstSample storage should be created");
  storage->device.type = simaai::neat::DeviceType::SIMA_CVU;

  simaai::neat::Tensor tensor;
  tensor.device.type = simaai::neat::DeviceType::SIMA_CVU;
  tensor.storage = std::move(storage);

  simaai::neat::Sample out;
  out.kind = simaai::neat::SampleKind::TensorSet;
  out.tensors.push_back(std::move(tensor));
  return out;
}

} // namespace

RUN_TEST(
    "unit_graph_internal_zero_copy_fallback_test", ([] {
      {
        simaai::neat::runtime::ExecutionGraphPlan plan;

        simaai::neat::RunOptions run_opt;
        run_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;

        simaai::neat::runtime::RunCoreStartOptions start_opt;
        start_opt.run_options = run_opt;
        start_opt.mode = simaai::neat::RunMode::Async;
        start_opt.graph_options =
            simaai::neat::runtime::graph_runtime_options_from_run_options(run_opt);

        auto core = simaai::neat::runtime::RunCore::start(std::move(plan), std::move(start_opt));
        require(core != nullptr, "graph RunCore should start");
        require(core->holder_loan_gate != nullptr,
                "public graph zero-copy outputs should initialize a holder loan gate");
        require(core->holder_loan_gate->enabled(),
                "public graph zero-copy holder loan gate should be enabled");
        require(!core->pipeline.stream_opt.copy_output,
                "public graph zero-copy stream options should preserve zero-copy output");
        require(core->pipeline.stream_opt.public_output_contract,
                "public graph output loans should use public output contract semantics");
        require(core->pipeline.stream_opt.holder_loan_credits > 0,
                "public graph zero-copy holder loan credits should be configured");
        core->stop();
      }

      {
        simaai::neat::RunOptions run_opt;
        run_opt.queue_depth = 7;
        const auto graph_opt =
            simaai::neat::runtime::graph_runtime_options_from_run_options(run_opt);
        require(graph_opt.edge_queue == 7U,
                "public Graph::build should use RunOptions::queue_depth for graph edge queues");

        run_opt.queue_depth = 0;
        const auto unbounded_graph_opt =
            simaai::neat::runtime::graph_runtime_options_from_run_options(run_opt);
        require(unbounded_graph_opt.edge_queue == 0U,
                "queue_depth=0 should preserve the graph runtime's unbounded queue convention");
      }

      auto public_core = make_balanced_zero_copy_core(true);
      require(public_core->pipeline.zero_copy_fallback_enabled,
              "public balanced zero-copy output should keep the copy fallback enabled");
      require(!public_core->pipeline.copy_output_latched.load(std::memory_order_relaxed),
              "public zero-copy output should start unlatched");

      auto graph_internal_core = make_balanced_zero_copy_core(false);
      require(!graph_internal_core->pipeline.zero_copy_fallback_enabled,
              "graph-internal zero-copy transport must not clone tensors back to CPU under "
              "queue pressure");
      require(!graph_internal_core->pipeline.copy_output_latched.load(std::memory_order_relaxed),
              "graph-internal zero-copy transport should start unlatched");

      auto loan_core = std::make_shared<simaai::neat::runtime::RunCore>();
      loan_core->pipeline.supports_pull = true;
      loan_core->holder_loan_gate =
          std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(1);
      {
        std::lock_guard<std::mutex> lock(loan_core->pipeline.out_mu);
        loan_core->pipeline.out_queue.push_back(make_fake_device_gst_sample(1));
        loan_core->pipeline.out_queue.push_back(make_fake_device_gst_sample(2));
      }

      simaai::neat::Sample first;
      simaai::neat::PullError err;
      require(loan_core->pull(0, first, &err) == simaai::neat::PullStatus::Ok,
              "first zero-copy output should acquire the only holder loan");
      require(loan_core->holder_loan_gate->inflight() == 1,
              "first output should hold one loan credit");
      require(loan_core->pipeline.out_queue.size() == 1U,
              "first pull should remove exactly one queued output");

      simaai::neat::Sample blocked;
      const auto blocked_status = loan_core->pull(0, blocked, &err);
      require(blocked_status == simaai::neat::PullStatus::Timeout,
              "loan exhaustion should backpressure as timeout, not consume the output");
      require(loan_core->pipeline.out_queue.size() == 1U,
              "loan exhaustion must leave the queued output available for retry");
      require(blocked.tensors.empty() && !blocked.tensor.has_value(),
              "failed loan acquisition must not publish an unloaned sample");

      first = simaai::neat::Sample{};
      simaai::neat::Sample second;
      require(loan_core->pull(1000, second, &err) == simaai::neat::PullStatus::Ok,
              "retry after releasing the older output should recover the queued frame");
      require(loan_core->pipeline.out_queue.empty(),
              "successful retry should consume the preserved queued output");

      {
        simaai::neat::graph::runtime::BlockingQueue<int> bounded_queue(1);
        require(bounded_queue.try_push(1), "bounded queue should accept first item");
        int restored_candidate = 0;
        require(bounded_queue.pop_with_restore_reservation(restored_candidate, 0),
                "reserved pop should return the first bounded queue item");
        require(restored_candidate == 1, "reserved pop should return the first item");
        require(!bounded_queue.try_push(2),
                "producer must not see a free capacity slot while restore is reserved");
        require(bounded_queue.restore_reserved_front(std::move(restored_candidate)),
                "reserved restore must put back the popped item even for capacity-one queues");
        require(bounded_queue.size() == 1U,
                "reserved restore must leave bounded queue at capacity, not above it");
        const auto restored_stats = bounded_queue.stats();
        require(restored_stats.high_watermark == 1U,
                "reserved restore must not raise the bounded queue high-watermark");
        require(restored_stats.current_size == 1U,
                "reserved restore should account exactly one occupied bounded slot");
        int remaining = 0;
        require(bounded_queue.pop(remaining, 0), "bounded queue should still contain one item");
        require(remaining == 1, "reserved restore should preserve the original item ordering");

        require(bounded_queue.try_push(3), "bounded queue should accept another item");
        int consumed = 0;
        require(bounded_queue.pop_with_restore_reservation(consumed, 0),
                "reserved pop should be available for successful-consume path");
        require(!bounded_queue.try_push(4),
                "reserved consume path should also keep capacity hidden until released");
        require(bounded_queue.release_restore_reservation(),
                "successful consume path should release the reserved bounded slot");
        require(bounded_queue.try_push(4),
                "producer should see capacity only after reserved slot release");
      }

      constexpr simaai::neat::graph::NodeId kSinkNode = 7;
      auto graph_core = std::make_shared<simaai::neat::runtime::RunCore>();
      graph_core->graph_execution_ =
          std::make_unique<simaai::neat::runtime::ExecutionGraphRuntime>();
      simaai::neat::runtime::Endpoint endpoint;
      endpoint.kind = simaai::neat::runtime::Endpoint::Kind::GraphSink;
      endpoint.node = kSinkNode;
      graph_core->graph_execution_->plan.default_output = endpoint;
      graph_core->graph_execution_->sinks[kSinkNode] =
          std::make_shared<simaai::neat::runtime::GraphSinkQueue>();
      graph_core->holder_loan_gate =
          std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(1);
      graph_core->graph_execution_->sinks[kSinkNode]->push(
          simaai::neat::runtime::RuntimeSinkQueueMsg{make_fake_device_gst_sample(3)});
      graph_core->graph_execution_->sinks[kSinkNode]->push(
          simaai::neat::runtime::RuntimeSinkQueueMsg{make_fake_device_gst_sample(4)});

      simaai::neat::Sample graph_first;
      require(graph_core->pull(0, graph_first, &err) == simaai::neat::PullStatus::Ok,
              "first graph output should acquire the only holder loan");
      require(graph_core->graph_execution_->sinks[kSinkNode]->size() == 1U,
              "first graph pull should remove exactly one queued output");

      simaai::neat::Sample graph_blocked;
      const auto graph_blocked_status = graph_core->pull(0, graph_blocked, &err);
      require(graph_blocked_status == simaai::neat::PullStatus::Timeout,
              "graph loan exhaustion should backpressure as timeout");
      require(graph_core->graph_execution_->sinks[kSinkNode]->size() == 1U,
              "graph loan exhaustion must restore the queued output for retry");
      require(graph_blocked.tensors.empty() && !graph_blocked.tensor.has_value(),
              "graph loan exhaustion must not publish an unloaned sample");

      {
        GstSample* external_sample = nullptr;
        simaai::neat::Sample exported = make_device_gst_sample_with_external_ref(&external_sample);
        auto gate = std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(1);
        std::string loan_error;
        require(simaai::neat::pipeline_internal::attach_zero_copy_loan_to_sample(exported, gate,
                                                                                 &loan_error),
                loan_error.empty() ? "test sample should acquire zero-copy holder loan"
                                   : loan_error.c_str());
        require(gate->inflight() == 1, "exported zero-copy sample should hold one loan credit");

        GstBuffer* source_buffer = gst_sample_get_buffer(external_sample);
        require(source_buffer != nullptr, "external test GstSample should carry a GstBuffer");
        GstBuffer* transfer_buffer = gst_buffer_new();
        require(transfer_buffer != nullptr, "test transfer GstBuffer should allocate");
        const GstBufferCopyFlags copy_flags =
            static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS |
                                            GST_BUFFER_COPY_META | GST_BUFFER_COPY_MEMORY);
        require(gst_buffer_copy_into(transfer_buffer, source_buffer, copy_flags, 0, -1),
                "test transfer GstBuffer should wrap source memories");
        simaai::neat::pipeline_internal::attach_zero_copy_loans_to_gst_buffer(transfer_buffer,
                                                                              exported);

        exported = simaai::neat::Sample{};
        require(gate->inflight() == 1,
                "downstream transfer buffer should own the loan after exported Sample release");
        gst_buffer_unref(transfer_buffer);
        require(gate->inflight() == 0,
                "source GstSample weak loan marker must not pin the holder loan after transfer "
                "buffer release");
        gst_sample_unref(external_sample);
      }

      constexpr simaai::neat::graph::NodeId kBoundedSinkNode = 8;
      auto bounded_graph_core = std::make_shared<simaai::neat::runtime::RunCore>();
      bounded_graph_core->graph_execution_ =
          std::make_unique<simaai::neat::runtime::ExecutionGraphRuntime>();
      simaai::neat::runtime::Endpoint bounded_endpoint;
      bounded_endpoint.kind = simaai::neat::runtime::Endpoint::Kind::GraphSink;
      bounded_endpoint.node = kBoundedSinkNode;
      bounded_graph_core->graph_execution_->plan.default_output = bounded_endpoint;
      bounded_graph_core->graph_execution_->sinks[kBoundedSinkNode] =
          std::make_shared<simaai::neat::runtime::GraphSinkQueue>(1);
      bounded_graph_core->holder_loan_gate =
          std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(1);
      require(bounded_graph_core->holder_loan_gate->try_acquire(),
              "test should exhaust the bounded graph output loan gate");
      require(bounded_graph_core->graph_execution_->sinks[kBoundedSinkNode]->push(
                  simaai::neat::runtime::RuntimeSinkQueueMsg{make_fake_device_gst_sample(5)}, 0),
              "bounded graph sink should accept one queued output");
      simaai::neat::Sample bounded_blocked;
      const auto bounded_status = bounded_graph_core->pull(0, bounded_blocked, &err);
      require(bounded_status == simaai::neat::PullStatus::Timeout,
              "bounded graph loan exhaustion should be reported as retryable backpressure");
      require(bounded_graph_core->graph_execution_->sinks[kBoundedSinkNode]->size() == 1U,
              "bounded graph loan exhaustion must restore without dropping the popped output");
      require(!bounded_graph_core->graph_execution_->sinks[kBoundedSinkNode]->try_push(
                  simaai::neat::runtime::RuntimeSinkQueueMsg{make_fake_device_gst_sample(6)}),
              "bounded graph sink should still respect capacity after restoring the output");
      bounded_graph_core->holder_loan_gate->release();

      simaai::neat::Sample graph_second;
      std::thread release_graph_first_after_delay([&graph_first] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        graph_first = simaai::neat::Sample{};
      });
      const auto graph_waited_status = graph_core->pull(1000, graph_second, &err);
      release_graph_first_after_delay.join();
      require(graph_waited_status == simaai::neat::PullStatus::Ok,
              "graph positive-timeout pull should wait for loan release and recover the "
              "queued frame");
      require(graph_core->graph_execution_->sinks[kSinkNode]->size() == 0U,
              "successful graph retry should consume the restored queued output");

      auto qdata_core = std::make_shared<simaai::neat::runtime::RunCore>();
      qdata_core->pipeline.supports_pull = true;
      qdata_core->holder_loan_gate =
          std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(1);
      GstSample* external_sample_ref = nullptr;
      {
        std::lock_guard<std::mutex> lock(qdata_core->pipeline.out_mu);
        qdata_core->pipeline.out_queue.push_back(
            make_device_gst_sample_with_external_ref(&external_sample_ref));
      }
      require(external_sample_ref != nullptr, "test should retain an external GstSample ref");
      simaai::neat::Sample qdata_output;
      require(qdata_core->pull(0, qdata_output, &err) == simaai::neat::PullStatus::Ok,
              "GstSample-backed output should acquire a loan");
      require(qdata_core->holder_loan_gate->inflight() == 1,
              "GstSample-backed output should hold one loan while exported");
      auto qdata_holder = qdata_output.tensors.front().storage->holder;
      require(simaai::neat::pipeline_internal::holder_has_zero_copy_loans(qdata_holder),
              "holder-only pushes should be able to discover the live loan");
      GstBuffer* transfer_buffer = gst_buffer_new_allocate(nullptr, 1, nullptr);
      require(transfer_buffer != nullptr, "test transfer GstBuffer allocation should succeed");
      require(simaai::neat::pipeline_internal::attach_zero_copy_loans_from_holder_to_gst_buffer(
                  transfer_buffer, qdata_holder),
              "holder-only push should transfer the live loan to the downstream buffer");
      qdata_output = simaai::neat::Sample{};
      qdata_holder.reset();
      require(qdata_core->holder_loan_gate->inflight() == 1,
              "downstream transfer buffer should pin the loan after the exported holder is "
              "released");
      gst_buffer_unref(transfer_buffer);
      require(qdata_core->holder_loan_gate->inflight() == 0,
              "loan release must follow exported holders and downstream transfer buffers, not "
              "extra GstSample refs");
      gst_sample_unref(external_sample_ref);

      auto live_source = [](const std::string& name) {
        simaai::neat::CameraInputOptions opt;
        opt.buffer_name = name;
        simaai::neat::Graph graph(name);
        graph.add(simaai::neat::nodes::CameraInput(opt));
        return graph;
      };
      auto app_sink = [](const std::string& name) {
        simaai::neat::Graph graph(name);
        graph.add(simaai::neat::nodes::Output(name));
        return graph;
      };
      auto finite_http_source_with_live_text = [](const std::string& name) {
        simaai::neat::HttpSourceOptions opt;
        opt.location = "https://example.invalid/video?is-live=true";
        opt.is_live = false;
        simaai::neat::Graph graph(name);
        graph.add(simaai::neat::nodes::HttpSource(std::move(opt)));
        return graph;
      };
      auto consumer = [] {
        simaai::neat::Graph graph("consumer");
        graph.add(simaai::neat::nodes::Input("image"));
        graph.add(simaai::neat::nodes::Output("detections"));
        return graph;
      };

      simaai::neat::Graph finite_http_app("finite_http_fan_in");
      auto finite_http0 = finite_http_source_with_live_text("finite_http0");
      auto finite_http1 = finite_http_source_with_live_text("finite_http1");
      auto finite_http_detector = consumer();
      finite_http_app.connect(finite_http0, finite_http_detector);
      bool finite_http_rejected = false;
      try {
        finite_http_app.connect(finite_http1, finite_http_detector);
      } catch (const std::exception& e) {
        finite_http_rejected = std::string(e.what()).find("already connected") != std::string::npos;
      }
      require(finite_http_rejected,
              "finite HttpSource URLs containing is-live=true text must not auto-promote fan-in");

      simaai::neat::Graph mixed_policy_app("mixed_policy_live_fan_in");
      auto cam0 = live_source("cam0");
      auto cam1 = live_source("cam1");
      auto preview = app_sink("preview");
      auto detector = consumer();
      mixed_policy_app.connect(cam0, preview);
      mixed_policy_app.connect(cam0, detector);
      mixed_policy_app.connect(cam1, detector);
      const auto mixed_plan =
          simaai::neat::runtime::compile_public_graph(mixed_policy_app, simaai::neat::RunOptions{});
      bool saw_shared_fanout_trunk = false;
      bool saw_default_fanout_branch = false;
      bool saw_realtime_fanout_branch = false;
      for (const auto& edge : mixed_plan.edges) {
        const bool from_fanout = edge.from < mixed_plan.node_labels.size() &&
                                 mixed_plan.node_labels[edge.from].rfind("fanout", 0) == 0;
        const bool to_fanout = edge.to < mixed_plan.node_labels.size() &&
                               mixed_plan.node_labels[edge.to].rfind("fanout", 0) == 0;
        if (to_fanout) {
          saw_shared_fanout_trunk = true;
          require(edge.link_options.policy == simaai::neat::GraphLinkPolicy::Default,
                  "realtime fan-in policy must not attach to a shared FanOut trunk");
        }
        if (from_fanout && edge.link_options.policy == simaai::neat::GraphLinkPolicy::Default) {
          saw_default_fanout_branch = true;
        }
        if (from_fanout &&
            edge.link_options.policy == simaai::neat::GraphLinkPolicy::RealtimeLatestByStream) {
          saw_realtime_fanout_branch = true;
        }
      }
      require(saw_shared_fanout_trunk, "test graph should lower shared producer through FanOut");
      require(saw_default_fanout_branch,
              "default branch from shared FanOut should keep default policy");
      require(saw_realtime_fanout_branch,
              "fan-in branch from shared FanOut should keep realtime policy");

      simaai::neat::Graph redundant_branch_fan_in_app("redundant_branch_live_fan_in");
      auto redundant_detector = [] {
        simaai::neat::Graph graph("detector");
        graph.add(simaai::neat::nodes::Input("detector_frame"));
        graph.add(simaai::neat::nodes::Output("detections"));
        return graph;
      }();
      for (int stream = 0; stream < 2; ++stream) {
        auto source = live_source("redundant_cam" + std::to_string(stream));
        auto one_output_branch = simaai::neat::graphs::Branch("source", {"detector_frame"});

        simaai::neat::RealtimeGraphLinkOptions decoded_link;
        decoded_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        decoded_link.queue_depth = 1;
        decoded_link.stream_id = "redundant_stream" + std::to_string(stream);
        redundant_branch_fan_in_app.connect(source, one_output_branch, decoded_link);

        simaai::neat::RealtimeGraphLinkOptions detector_link;
        detector_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        detector_link.queue_depth = 16;
        detector_link.stream_id = "redundant_stream" + std::to_string(stream);
        redundant_branch_fan_in_app.connect(one_output_branch, redundant_detector, detector_link);
      }
      const auto redundant_branch_plan = simaai::neat::runtime::compile_public_graph(
          redundant_branch_fan_in_app, simaai::neat::RunOptions{});
      int redundant_realtime_edges = 0;
      bool saw_redundant_stream0 = false;
      bool saw_redundant_stream1 = false;
      for (const auto& edge : redundant_branch_plan.edges) {
        if (edge.link_options.policy != simaai::neat::GraphLinkPolicy::RealtimeLatestByStream) {
          continue;
        }
        ++redundant_realtime_edges;
        if (edge.stream_id == "redundant_stream0") {
          saw_redundant_stream0 = true;
        }
        if (edge.stream_id == "redundant_stream1") {
          saw_redundant_stream1 = true;
        }
      }
      require(redundant_realtime_edges >= 2,
              "one-output Branch elision must preserve realtime fan-in runtime edges");
      require(saw_redundant_stream0 && saw_redundant_stream1,
              "one-output Branch elision must preserve per-stream realtime identities");

      simaai::neat::RealtimeGraphLinkOptions stream_link;
      stream_link.stream_id = "compat_stream";

      simaai::neat::Graph default_stream_one_to_one_app("default_link_stream_id_one_to_one");
      auto one_to_one_src = live_source("one_to_one_src");
      auto one_to_one_sink = app_sink("one_to_one_sink");
      default_stream_one_to_one_app.connect(one_to_one_src, one_to_one_sink, stream_link);
      const auto default_stream_one_to_one_plan = simaai::neat::runtime::compile_public_graph(
          default_stream_one_to_one_app, simaai::neat::RunOptions{});
      bool saw_one_to_one_stream_id = false;
      for (const auto& edge : default_stream_one_to_one_plan.edges) {
        if (edge.stream_id == "compat_stream") {
          saw_one_to_one_stream_id = true;
        }
      }
      require(saw_one_to_one_stream_id,
              "default one-to-one stream_id link should remain explicit for runtime stamping");

      simaai::neat::Graph default_stream_fanout_app("default_link_stream_id_fanout");
      auto fanout_src = live_source("fanout_src");
      auto fanout_stream_sink = app_sink("fanout_stream_sink");
      auto fanout_plain_sink = app_sink("fanout_plain_sink");
      default_stream_fanout_app.connect(fanout_src, fanout_stream_sink, stream_link);
      default_stream_fanout_app.connect(fanout_src, fanout_plain_sink);
      const auto default_stream_fanout_plan = simaai::neat::runtime::compile_public_graph(
          default_stream_fanout_app, simaai::neat::RunOptions{});
      bool saw_stream_id_fanout_trunk = false;
      bool saw_stream_id_branch = false;
      bool saw_empty_default_branch = false;
      for (const auto& edge : default_stream_fanout_plan.edges) {
        const bool from_fanout =
            edge.from < default_stream_fanout_plan.node_labels.size() &&
            default_stream_fanout_plan.node_labels[edge.from].rfind("fanout", 0) == 0;
        const bool to_fanout =
            edge.to < default_stream_fanout_plan.node_labels.size() &&
            default_stream_fanout_plan.node_labels[edge.to].rfind("fanout", 0) == 0;
        if (to_fanout) {
          saw_stream_id_fanout_trunk = true;
          require(edge.stream_id.empty(),
                  "stream_id-only branch must not stamp the shared FanOut trunk");
        }
        if (from_fanout && edge.stream_id == "compat_stream") {
          saw_stream_id_branch = true;
        }
        if (from_fanout && edge.stream_id.empty() &&
            edge.link_options.policy == simaai::neat::GraphLinkPolicy::Default) {
          saw_empty_default_branch = true;
        }
      }
      require(saw_stream_id_fanout_trunk,
              "default stream-id test graph should lower shared producer through FanOut");
      require(saw_stream_id_branch,
              "default-link stream_id should stamp only the selected FanOut branch");
      require(saw_empty_default_branch, "other FanOut default branches should remain unstamped");

      constexpr simaai::neat::graph::NodeId kStreamSinkNode = 42;
      simaai::neat::runtime::ExecutionGraphRuntime stream_runtime;
      simaai::neat::runtime::EdgePlan stream_edge;
      stream_edge.to = kStreamSinkNode;
      stream_edge.stream_id = "runtime_stream";
      stream_runtime.plan.edges.push_back(std::move(stream_edge));
      stream_runtime.sinks[kStreamSinkNode] =
          std::make_shared<simaai::neat::runtime::GraphSinkQueue>();
      simaai::neat::runtime::EdgeRouter stream_router(stream_runtime);
      simaai::neat::runtime::EdgeRouterCallbacks stream_callbacks;
      stream_callbacks.stop_requested = [] { return false; };
      simaai::neat::runtime::EdgeRouterOptions stream_router_options;
      simaai::neat::Sample routed_sample;
      routed_sample.stream_id = "input_stream";
      simaai::neat::runtime::DownstreamTarget stream_target{
          simaai::neat::runtime::DownstreamTarget::Kind::GraphSink,
          static_cast<std::size_t>(kStreamSinkNode),
          simaai::neat::graph::kInvalidPort,
          0U,
      };
      require(stream_router.dispatch_to_target(stream_target, std::move(routed_sample),
                                               stream_router_options, stream_callbacks),
              "EdgeRouter should route default-link stream-id samples to graph sinks");
      simaai::neat::runtime::RuntimeSinkQueueMsg routed_out;
      require(stream_runtime.sinks[kStreamSinkNode]->pop(routed_out, 0),
              "stream-id router test should enqueue one sink sample");
      require(routed_out.sample.stream_id == "runtime_stream",
              "default runtime edges should stamp samples with their link stream_id");
      require(routed_out.sample.stream_label == "runtime_stream",
              "default runtime edge stream_id should provide a stream label when missing");

      simaai::neat::runtime::DownstreamTarget realtime_target{
          simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
          static_cast<std::size_t>(kStreamSinkNode),
          simaai::neat::graph::kInvalidPort,
          0U,
      };
      simaai::neat::RealtimeGraphLinkOptions realtime_credit_options;
      realtime_credit_options.max_inflight_per_stream = 1;
      simaai::neat::runtime::RealtimeLatestLink realtime_link(
          realtime_target, realtime_credit_options, "credit_stream");
      std::mutex realtime_mu;
      std::condition_variable realtime_cv;
      std::vector<std::int64_t> dispatched_frames;
      std::string realtime_error;
      realtime_link.start(
          [&](const simaai::neat::runtime::DownstreamTarget&, simaai::neat::Sample&& sample,
              std::size_t) {
            {
              std::lock_guard<std::mutex> lock(realtime_mu);
              dispatched_frames.push_back(sample.frame_id);
            }
            realtime_cv.notify_all();
            return true;
          },
          [] { return false; },
          [&](const std::string& msg) {
            std::lock_guard<std::mutex> lock(realtime_mu);
            realtime_error = msg;
            realtime_cv.notify_all();
          });

      simaai::neat::Sample credit_first = make_fake_device_gst_sample(7);
      credit_first.stream_id = "credit_stream";
      credit_first.frame_id = 1;
      require(realtime_link.offer(std::move(credit_first), 0U),
              "realtime credit link should accept the first frame");
      {
        std::unique_lock<std::mutex> lock(realtime_mu);
        require(realtime_cv.wait_for(lock, std::chrono::seconds(1),
                                     [&] { return dispatched_frames.size() == 1U; }),
                "first realtime credit frame should dispatch promptly");
        require(dispatched_frames.front() == 1, "first dispatched realtime frame should match");
      }

      simaai::neat::Sample credit_second = make_fake_device_gst_sample(8);
      credit_second.stream_id = "credit_stream";
      credit_second.frame_id = 2;
      require(realtime_link.offer(std::move(credit_second), 0U),
              "realtime credit link should accept the second frame");
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      {
        std::lock_guard<std::mutex> lock(realtime_mu);
        require(dispatched_frames.size() == 1U,
                "second realtime frame must wait while the first frame credit is in-flight");
      }
      simaai::neat::pipeline_internal::release_realtime_frame_credits(
          {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "credit_stream", 1}},
          "unit-graph-realtime-output");
      {
        std::unique_lock<std::mutex> lock(realtime_mu);
        require(realtime_cv.wait_for(lock, std::chrono::seconds(1),
                                     [&] { return dispatched_frames.size() == 2U; }),
                "second realtime credit frame should dispatch after output releases credit");
        require(dispatched_frames.back() == 2, "second dispatched realtime frame should match");
        require(realtime_error.empty(), "realtime credit link should not report an error");
      }
      realtime_link.close();
      realtime_link.join();
      const auto realtime_stats = realtime_link.stats();
      require(realtime_stats.credit_registered >= 2U,
              "decoder-backed PipelineInput realtime links should register graph-owned frame "
              "credits");
      require(realtime_stats.credit_released_by_output >= 1U,
              "realtime link should observe downstream credit release");
      require(realtime_stats.no_credit_skips > 0U,
              "realtime link should account credit-throttled scheduling attempts");
    }))

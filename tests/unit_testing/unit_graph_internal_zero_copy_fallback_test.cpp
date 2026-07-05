#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/EdgeRouter.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/Input.h"

#include <gst/gst.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

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
      auto consumer = [] {
        simaai::neat::Graph graph("consumer");
        graph.add(simaai::neat::nodes::Input("image"));
        graph.add(simaai::neat::nodes::Output("detections"));
        return graph;
      };

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

      simaai::neat::GraphLinkOptions stream_link;
      stream_link.stream_id = "compat_stream";

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
    }))

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/internal/InputStream.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

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

} // namespace

RUN_TEST("unit_graph_internal_zero_copy_fallback_test", ([] {
           {
             simaai::neat::runtime::ExecutionGraphPlan plan;

             simaai::neat::RunOptions run_opt;
             run_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;

             simaai::neat::runtime::RunCoreStartOptions start_opt;
             start_opt.run_options = run_opt;
             start_opt.mode = simaai::neat::RunMode::Async;
             start_opt.graph_options =
                 simaai::neat::runtime::graph_runtime_options_from_run_options(run_opt);

             auto core =
                 simaai::neat::runtime::RunCore::start(std::move(plan), std::move(start_opt));
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
           require(
               !graph_internal_core->pipeline.copy_output_latched.load(std::memory_order_relaxed),
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
         }))

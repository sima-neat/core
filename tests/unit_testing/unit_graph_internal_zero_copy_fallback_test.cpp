#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/runtime/RunInternal.h"
#include "graph/internal/GraphRunState.h"
#include "dmabuf_test_utils.h"
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
#include <optional>
#include <stdexcept>
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

simaai::neat::Sample sample_from_buffer(GstBuffer* buffer) {
  require(buffer != nullptr, "test GstBuffer allocation should succeed");
  GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
  require(sample != nullptr, "test GstSample allocation should succeed");
  simaai::neat::Tensor tensor;
  tensor.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.shape = {static_cast<std::int64_t>(gst_buffer_get_size(buffer))};
  tensor.read_only = true;

  simaai::neat::Sample out;
  out.kind = simaai::neat::SampleKind::TensorSet;
  out.tensors.push_back(std::move(tensor));
  return out;
}

simaai::neat::Sample make_legacy_device_gst_sample(int id) {
  gst_init(nullptr, nullptr);
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 1, nullptr);
  auto sample = sample_from_buffer(buffer);
  gst_buffer_unref(buffer);
  sample.frame_id = id;
  sample.tensors.front().device.type = simaai::neat::DeviceType::SIMA_CVU;
  sample.tensors.front().storage->device.type = simaai::neat::DeviceType::SIMA_CVU;
  return sample;
}

class ScopedEnv {
public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* prior = std::getenv(name)) {
      prior_ = std::string(prior);
    }
    if (value) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (prior_) {
      setenv(name_.c_str(), prior_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  std::optional<std::string> prior_;
};

// Six logical edges are six regions of one retained native frame, not six
// independently budgeted output carriers. Names deliberately do not identify
// the allocation role; the container node name is not needed by this policy.
simaai::neat::BuildResult make_retention_build() {
  using namespace simaai::neat;
  namespace sima = pipeline_internal::sima;
  BuildResult build;
  build.appsink_name = "mysink";
  build.pipeline_string = "appsrc name=input ! neatprocesscvu name=actual_preproc num-buffers=1 ! "
                          "neatprocessmla name=actual_mla num-buffers=1 multi-pipeline=false ! "
                          "neatprocesscvu name=actual_detess num-buffers=1 ! "
                          "identity name=adapter ! queue max-size-buffers=1 ! appsink name=mysink";
  build.rendered_manifest.emplace();
  for (int i = 0; i < 3; ++i) {
    sima::StageStaticSpec stage;
    stage.element_name = i == 0 ? "actual_preproc" : i == 1 ? "actual_mla" : "actual_detess";
    stage.logical_stage_id = "logical_" + std::to_string(i);
    stage.plugin_kind = i == 1 ? "neatprocessmla" : "neatprocesscvu";
    stage.frame_arena_size_bytes = 4096U;
    stage.frame_arena_role =
        i == 0 ? sima::FrameArenaRole::Allocate : sima::FrameArenaRole::ReuseInput;
    stage.frame_arena_provenance = sima::static_contract::ArenaAllocationProvenance::CoreAllocated;
    for (int head = 0; head < 6; ++head) {
      sima::PhysicalBufferStaticSpec physical;
      physical.physical_index = head;
      physical.size_bytes = 64U;
      stage.physical_outputs.push_back(physical);
      sima::LogicalTensorStaticSpec output;
      output.logical_index = head;
      output.physical_index = head;
      output.size_bytes = 64U;
      stage.logical_outputs.push_back(output);
      if (i > 0) {
        stage.physical_inputs.push_back(physical);
        sima::LogicalInputStaticSpec input;
        input.logical_index = head;
        input.physical_index = head;
        stage.logical_inputs.push_back(input);
        sima::InputBindingStaticSpec binding;
        binding.local_logical_input_index = head;
        binding.src_stage_index = i - 1;
        binding.src_stage_id = "logical_" + std::to_string(i - 1);
        binding.src_logical_output_index = head;
        binding.src_physical_output_index = head;
        stage.input_bindings.push_back(binding);
      }
    }
    build.rendered_manifest->stages.push_back(std::move(stage));
  }
  return build;
}

// Match the selector-free compiler projection: one preproc handoff, six MLA
// logical heads over physical parent zero, then six independently named CVU
// regions. Local source physical ordinals are not global producer identities.
simaai::neat::BuildResult make_value_bound_retention_build() {
  auto build = make_retention_build();
  auto& stages = build.rendered_manifest->stages;
  auto& preproc = stages[0];
  preproc.logical_outputs.resize(1U);
  preproc.physical_outputs.resize(1U);
  preproc.logical_outputs[0].logical_name = "output_tessellated_image";
  preproc.logical_outputs[0].segment_name = "tessellate_quantize_0/transform";
  preproc.logical_outputs[0].shape = {1, 640, 640, 3};
  auto& mla = stages[1];
  mla.logical_inputs.resize(1U);
  mla.physical_inputs.resize(1U);
  mla.input_bindings.resize(1U);
  mla.logical_inputs[0].shape = {640, 640, 3};
  mla.physical_outputs.resize(1U);
  mla.physical_outputs[0].segment_name = "MLA_0";
  mla.physical_outputs[0].size_bytes = 6U * 64U;
  for (std::size_t head = 0; head < mla.logical_outputs.size(); ++head) {
    auto& output = mla.logical_outputs[head];
    output.logical_name = "MLA_0_ofm_unpack_transform_" + std::to_string(head);
    output.backend_name = output.logical_name;
    output.segment_name = "MLA_0";
    output.physical_index = 0;
    output.byte_offset = head * 64U;
    auto& binding = stages[2].input_bindings[head];
    binding.cm_input_name = output.logical_name;
    binding.source_segment_name = "value_" + std::to_string(head + 4U);
  }
  mla.input_bindings[0].cm_input_name = preproc.logical_outputs[0].segment_name;
  mla.input_bindings[0].source_segment_name = preproc.logical_outputs[0].segment_name;
  for (auto& stage : stages) {
    for (auto& binding : stage.input_bindings) {
      binding.src_stage_index = -1;
      binding.src_stage_id.clear();
    }
  }
  return build;
}

simaai::neat::BuildResult make_ingress_retention_build() {
  auto build = make_value_bound_retention_build();
  build.pipeline_string =
      "appsrc name=input ! identity ! application/vnd.simaai.tensor ! queue max-size-buffers=1 ! "
      "neatprocessmla name=actual_mla num-buffers=1 ! "
      "neatprocesscvu name=actual_detess num-buffers=1 ! appsink name=mysink";
  build.rendered_manifest->stages.erase(build.rendered_manifest->stages.begin());
  auto& mla = build.rendered_manifest->stages.front();
  auto& binding = mla.input_bindings.front();
  binding.cm_input_name = "cast_0";
  binding.source_segment_name = "cast_0";
  binding.src_physical_size_bytes = mla.physical_inputs.front().size_bytes;
  binding.src_physical_byte_offset = 0;
  binding.required = true;
  mla.logical_inputs.front().logical_name = "cast_0";
  mla.logical_inputs.front().segment_name = "cast_0";
  mla.physical_inputs.front().segment_name = "cast_0";
  return build;
}

void require_native_retention_pool_policy() {
  using namespace simaai::neat;
  namespace sima = pipeline_internal::sima;
  const auto check = [](const std::string& pipeline, const std::string& owner, int depth = 1,
                        int floor = 2) {
    std::size_t floors = 0U;
    for (const auto& element : sima::parse_pipeline_elements(pipeline)) {
      if (element.plugin == "neatprocesscvu" || element.plugin == "neatprocessmla") {
        require(element.fragment.find("num-buffers=" + std::to_string(depth)) != std::string::npos,
                "retention storage must not change execution depth or trigger sync prefill");
      }
      if (element.fragment.find("output-pool-min-buffers=") != std::string::npos) {
        ++floors;
        require(element.element_name == owner &&
                    element.fragment.find("output-pool-min-buffers=" + std::to_string(floor)) !=
                        std::string::npos,
                "only the exact typed Allocate owner should receive a pool-only floor");
      }
    }
    require(floors == (owner.empty() ? 0U : 1U), "one cached carrier must be budgeted only once");
    require(pipeline.find("max-size-buffers=1") != std::string::npos,
            "retention policy must preserve the serial queue limit");
  };
  const auto reject = [](const BuildResult& build, const std::string& reason) {
    bool failed = false;
    try {
      (void)session_build_clamp_sync_build_result(build, 1);
    } catch (const std::exception& error) {
      failed = std::string(error.what()).find(reason) != std::string::npos;
    }
    require(failed, "ambiguous native carrier ownership must fail with an actionable diagnostic");
  };

  auto value_bound = make_value_bound_retention_build();
  check(session_build_clamp_sync_build_result(value_bound, -1), "actual_preproc");
  auto caps_bound = make_value_bound_retention_build();
  caps_bound.pipeline_string.insert(caps_bound.pipeline_string.find("identity name=adapter"),
                                    "video/x-raw(memory:DMABuf),format=(string)BGR ! ");
  check(session_build_clamp_sync_build_result(caps_bound, -1), "actual_preproc");
  std::string edge_error;
  const auto edges = sima::edgecontract::resolve_consumer_edge_contracts_exact(
      *value_bound.rendered_manifest, 2U, &edge_error);
  require(edge_error.empty() && edges.size() == 6U,
          "authored head identities must resolve six selector-free packed-parent edges");
  for (std::size_t head = 0; head < edges.size(); ++head) {
    require(
        edges[head].producer_stage_index == 1U &&
            edges[head].producer_logical_output->logical_index == static_cast<int>(head) &&
            edges[head].producer_physical_output->physical_index == 0 &&
            edges[head].binding == &value_bound.rendered_manifest->stages[2].input_bindings[head],
        "exact producer resolution must preserve each logical head, backing parent and binding");
  }
  std::swap(value_bound.rendered_manifest->stages[0], value_bound.rendered_manifest->stages[2]);
  check(session_build_clamp_sync_build_result(value_bound, -1), "actual_preproc");
  value_bound = make_value_bound_retention_build();
  value_bound.rendered_manifest->stages[0].logical_outputs[0].logical_name =
      "MLA_0_ofm_unpack_transform_0";
  reject(value_bound, "ambiguous authored producer value");
  value_bound = make_value_bound_retention_build();
  value_bound.rendered_manifest->stages[2].input_bindings[0].cm_input_name = "missing_value";
  reject(value_bound, "no producer for authored value");

  auto ingress = make_ingress_retention_build();
  const auto ingress_manifest = sima::serialize_manifest_json(*ingress.rendered_manifest);
  check(session_build_clamp_sync_build_result(ingress, 1), "");
  require(
      sima::serialize_manifest_json(*ingress.rendered_manifest) == ingress_manifest,
      "graph ingress must preserve authored bindings, arena geometry and CoreAllocated provenance");
  std::swap(ingress.rendered_manifest->stages[0], ingress.rendered_manifest->stages[1]);
  check(session_build_clamp_sync_build_result(ingress, 1), "");
  ingress = make_ingress_retention_build();
  ingress.rendered_manifest->stages.front().frame_arena_provenance =
      sima::static_contract::ArenaAllocationProvenance::ExternalAdopted;
  check(session_build_clamp_sync_build_result(ingress, 1), "");
  ingress.rendered_manifest->stages.front().input_bindings.front().src_stage_index = 1;
  reject(ingress, "graph ingress binding names a local producer");
  ingress = make_ingress_retention_build();
  ingress.rendered_manifest->stages.front().input_bindings.front().src_stage_id = "missing_stage";
  reject(ingress, "graph ingress binding names a local producer");
  ingress = make_ingress_retention_build();
  ingress.pipeline_string.replace(0, std::string("appsrc").size(), "videotestsrc");
  reject(ingress, "no producer for authored value");
  ingress = make_ingress_retention_build();
  ingress.pipeline_string.insert(ingress.pipeline_string.find("neatprocessmla"), "videoconvert ! ");
  reject(ingress, "no producer for authored value");
  ingress = make_ingress_retention_build();
  ingress.pipeline_string.replace(ingress.pipeline_string.find("neatprocessmla"),
                                  std::string("neatprocessmla").size(), "identity");
  reject(ingress, "no producer for authored value");

  auto build = make_retention_build();
  const auto clamped = session_build_clamp_sync_build_result(build, 1);
  check(clamped, "actual_preproc");
  build.pipeline_string = clamped;
  require(session_build_clamp_sync_build_result(build, 1) == clamped,
          "retention floor and both legacy clamp exclusions must be idempotent");
  build.pipeline_string.replace(build.pipeline_string.find("output-pool-min-buffers=2"),
                                std::string("output-pool-min-buffers=2").size(),
                                "output-pool-min-buffers=7");
  check(session_build_clamp_sync_build_result(build, 1), "actual_preproc", 1, 7);
  build = make_retention_build();
  check(session_build_clamp_sync_build_result(build, 4), "", 4);
  build.rendered_manifest->stages[2].input_bindings[0].src_stage_index = 0;
  build.rendered_manifest->stages[2].input_bindings[0].src_stage_id = "logical_0";
  check(session_build_clamp_sync_build_result(build, 1), "actual_preproc");

  build = make_retention_build();
  build.name_transform.prefix = "branch_";
  build.appsink_name = "branch_mysink";
  build.pipeline_string =
      "appsrc name=branch_input ! neatprocesscvu name=branch_actual_preproc num-buffers=1 ! "
      "neatprocessmla name=branch_actual_mla num-buffers=1 ! "
      "neatprocesscvu name=branch_actual_detess num-buffers=1 ! "
      "identity name=adapter ! queue max-size-buffers=1 ! appsink name=branch_mysink";
  check(session_build_clamp_sync_build_result(build, 1), "branch_actual_preproc");

  build = make_retention_build();
  build.rendered_manifest->stages[1].frame_arena_role = sima::FrameArenaRole::Allocate;
  check(session_build_clamp_sync_build_result(build, 1), "actual_mla");
  build.rendered_manifest->stages[2].frame_arena_role = sima::FrameArenaRole::Allocate;
  check(session_build_clamp_sync_build_result(build, 1), "actual_detess");
  build.pipeline_string = "appsrc name=input ! neatprocesscvu name=actual_preproc num-buffers=1 ! "
                          "neatprocessmla name=actual_mla num-buffers=1 ! "
                          "queue max-size-buffers=1 ! appsink name=mysink";
  build.rendered_manifest->stages.pop_back();
  check(session_build_clamp_sync_build_result(build, 1), "actual_mla");

  build = make_retention_build();
  build.rendered_manifest->stages[0].frame_arena_role = sima::FrameArenaRole::ReuseInput;
  build.rendered_manifest->stages[0].frame_arena_provenance =
      sima::static_contract::ArenaAllocationProvenance::ExternalAdopted;
  check(session_build_clamp_sync_build_result(build, 1), "");

  build = make_retention_build();
  build.rendered_manifest->stages[1].frame_arena_provenance =
      sima::static_contract::ArenaAllocationProvenance::ExternalAdopted;
  build.rendered_manifest->stages[1].input_bindings.clear();
  reject(build, "cannot resolve reused carrier");

  build = make_retention_build();
  build.rendered_manifest->stages[2].input_bindings[0].src_stage_index = -1;
  build.rendered_manifest->stages[2].input_bindings[0].src_stage_id.clear();
  reject(build, "omits an explicit producer");
  build = make_retention_build();
  build.rendered_manifest->stages[2].input_bindings[0].src_stage_id = "logical_0";
  reject(build, "conflicting producer");
  build = make_retention_build();
  build.rendered_manifest->stages[0].logical_stage_id = "logical_1";
  reject(build, "ambiguous producer");
  build = make_retention_build();
  build.rendered_manifest->stages[0].element_name = "actual_mla";
  reject(build, "duplicate typed element");
  build = make_retention_build();
  build.pipeline_string.insert(0, "identity name=actual_preproc ! ");
  reject(build, "not uniquely rendered");
  build = make_retention_build();
  for (auto& binding : build.rendered_manifest->stages[1].input_bindings) {
    binding.src_stage_index = 2;
    binding.src_stage_id = "logical_2";
  }
  reject(build, "cycle at stage");
  build = make_retention_build();
  build.rendered_manifest->stages[1].frame_arena_role = sima::FrameArenaRole::Allocate;
  auto& split_binding = build.rendered_manifest->stages[2].input_bindings[0];
  split_binding.src_stage_index = 0;
  split_binding.src_stage_id = "logical_0";
  reject(build, "different carrier origins");
  build = make_retention_build();
  build.pipeline_string += " ! appsink name=second_sink";
  reject(build, "expected exactly");
  build = make_retention_build();
  build.pipeline_string.insert(0, "tee name=t ! t.src_0 ! ");
  reject(build, "nonlinear");
  build = make_retention_build();
  build.pipeline_string.insert(0, "( identity name=in_bin ) ! ");
  reject(build, "nonlinear");
  build = make_retention_build();
  build.pipeline_string.insert(0, "identity name=other_chain ; ");
  reject(build, "nonlinear");
  build = make_retention_build();
  build.pipeline_string.replace(build.pipeline_string.find("appsink name=mysink"),
                                std::string("appsink name=mysink").size(),
                                "tee name=t ! t.src_0 ! neatboxdecode name=decoded num-buffers=1 ! "
                                "appsink name=mysink");
  const auto materialized = session_build_clamp_sync_build_result(build, 1);
  require(materialized.find("decoded num-buffers=2") != std::string::npos &&
              materialized.find("output-pool-min-buffers=") == std::string::npos,
          "unrelated native topology must not reject a non-native terminal materializer");
  build = make_retention_build();
  build.rendered_manifest.reset();
  const auto legacy = session_build_clamp_sync_build_result(build, 1);
  require(legacy.find("actual_detess num-buffers=2") != std::string::npos &&
              legacy.find("output-pool-min-buffers=") == std::string::npos,
          "untyped routes must retain the existing legacy terminal behavior");
}

void require_output_policy_matrix() {
  using namespace simaai::neat;
  ScopedEnv detess("SIMA_DETESS_ZERO_COPY", nullptr);
  for (const char* override : {static_cast<const char*>(nullptr), "owned", "zerocopy"}) {
    ScopedEnv output_default("SIMA_OUTPUT_MEMORY_DEFAULT", override);
    for (const auto mode : {RunMode::Sync, RunMode::Async}) {
      for (const auto preset : {RunPreset::Reliable, RunPreset::Balanced, RunPreset::Realtime}) {
        for (const auto memory :
             {OutputMemory::Auto, OutputMemory::ZeroCopy, OutputMemory::Owned}) {
          RunOptions options;
          options.preset = preset;
          options.output_memory = memory;
          const auto resolved = session_build_resolve_build_opt(mode, options);
          require(resolved.output_memory == memory,
                  "build resolution must preserve caller output-memory intent");
          auto stream = session_build_make_stream_options(resolved, mode);
          const bool auto_owned = override && std::string(override) == "owned";
          const bool auto_zero_copy = override && std::string(override) == "zerocopy";
          const bool preserve_dma = memory == OutputMemory::Auto && !auto_owned;
          const bool zero_copy = memory == OutputMemory::ZeroCopy ||
                                 (memory == OutputMemory::Auto && !auto_owned &&
                                  (auto_zero_copy || (mode != RunMode::Sync &&
                                                      resolved.preset != RunPreset::Reliable)));
          require(stream.preserve_dmabuf_output == preserve_dma,
                  "Auto must retain DMA unless the caller deliberately selects owned output");
          require(stream.copy_output == !zero_copy,
                  "ordinary CPU and legacy output must keep their preset/mode copy policy");
          session_build_finalize_public_zero_copy_holder_loan_credits(stream);
          require((stream.holder_loan_credits > 0) == (zero_copy || preserve_dma),
                  "DMA-preserving Auto must reserve public loan credits in every mode");
        }
      }
    }
  }
  for (const auto mode : {RunMode::Sync, RunMode::Async}) {
    ScopedEnv output_default("SIMA_OUTPUT_MEMORY_DEFAULT", nullptr);
    RunOptions options;
    options.preset = RunPreset::Reliable;
    options.output_memory = OutputMemory::Auto;
    runtime::RunCoreStartOptions start;
    start.run_options = options;
    start.mode = mode;
    start.graph_options = runtime::graph_runtime_options_from_run_options(options);
    auto core = runtime::RunCore::start(runtime::ExecutionGraphPlan{}, std::move(start));
    require(core->pipeline.stream_opt.copy_output &&
                core->pipeline.stream_opt.preserve_dmabuf_output && core->holder_loan_gate &&
                core->holder_loan_gate->enabled(),
            "graph Auto must retain a loan gate even when ordinary output defaults to owned");
    core->stop();
  }
  ScopedEnv detess_override("SIMA_DETESS_ZERO_COPY", "1");
  const auto detess_options = session_build_resolve_build_opt(RunMode::Sync, RunOptions{});
  require(detess_options.output_memory == OutputMemory::ZeroCopy,
          "existing synchronous detess zero-copy override must remain explicit");
}

void require_dmabuf_loans_and_pressure_preserve_storage() {
  using namespace simaai::neat;
  gst_init(nullptr, nullptr);
  // memfd wrapped as standard DMA-BUF proves metadata/ownership only. These
  // cases must not map or submit it to a device; either would be a test error.
  GstBuffer* buffer = sima_test::make_bookkeeping_dmabuf(128);
  const auto identity = sima_test::dmabuf_span(buffer);
  auto sample = sample_from_buffer(buffer);
  gst_buffer_unref(buffer);
  require(sample.tensors.front().device.type == DeviceType::CPU &&
              sample.tensors.front().storage->device.type == DeviceType::CPU &&
              sample.tensors.front().storage->sima_mem_target_flags == 0,
          "standard DMA fixture must not depend on legacy placement flags");
  require(pipeline_internal::sample_has_dmabuf_memory(sample) &&
              pipeline_internal::sample_has_device_gstsample_holder(sample),
          "standard DMA-BUF must be recognized from retained memory");
  // Multiple views of the same retained storage consume one credit, not one
  // credit per plane or output field.
  sample.tensors.push_back(sample.tensors.front());
  require(pipeline_internal::count_distinct_device_gstsample_holders(sample) == 1,
          "shared DMA views must count as one retained holder");
  auto gate = std::make_shared<pipeline_internal::HolderLoanGate>(1);
  std::string error;
  require(pipeline_internal::attach_zero_copy_loan_to_sample(sample, gate, &error),
          "DMA sample must acquire one public loan");
  require(gate->inflight() == 1, "shared DMA views consumed more than one loan");

  for (const char* cap_name :
       {"SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP", "SIMA_GRAPH_ZERO_COPY_MAX_INFLIGHT"}) {
    ScopedEnv cap("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP", nullptr);
    ScopedEnv legacy_cap("SIMA_GRAPH_ZERO_COPY_MAX_INFLIGHT", nullptr);
    ScopedEnv enabled(cap_name, "1");
    const auto before = pipeline_internal::snapshot_tensor_io_stats();
    run_internal::maybe_force_copy_for_backpressure(sample, 2, "dma-test", false);
    run_internal::maybe_force_copy_for_backpressure(sample.tensors.front(), 2, "dma-test", false);
    graph::maybe_force_copy_for_backpressure(sample, 2, "dma-test", 0U);
    const auto after = pipeline_internal::snapshot_tensor_io_stats();
    require(after.tensor_copy_count == before.tensor_copy_count &&
                after.gst_memory_map_calls == before.gst_memory_map_calls,
            "optional pressure helpers must not copy or map standard DMA storage");
    GstBuffer* retained =
        pipeline_internal::buffer_from_tensor_holder(sample.tensors.front().storage->holder);
    require(retained != nullptr, "pressure helpers lost the original DMA holder");
    const auto retained_identity = sima_test::dmabuf_span(retained);
    gst_buffer_unref(retained);
    require(retained_identity == identity, "pressure helpers replaced the DMA allocation/view");
  }
  sample = Sample{};
  require(gate->inflight() == 0, "final DMA view release must return its one credit");

  auto core = std::make_shared<runtime::RunCore>();
  core->pipeline.supports_pull = true;
  core->holder_loan_gate = std::make_shared<pipeline_internal::HolderLoanGate>(1);
  std::vector<sima_test::DmaBufSpan> queued_identities;
  for (int i = 0; i < 2; ++i) {
    GstBuffer* queued_buffer = sima_test::make_bookkeeping_dmabuf(128);
    queued_identities.push_back(sima_test::dmabuf_span(queued_buffer));
    core->pipeline.out_queue.push_back(sample_from_buffer(queued_buffer));
    gst_buffer_unref(queued_buffer);
  }
  Sample first;
  PullError pull_error;
  require(core->pull(0, first, &pull_error) == PullStatus::Ok &&
              core->holder_loan_gate->inflight() == 1,
          "standard DMA public pull must acquire the available loan");
  Sample next;
  require(core->pull(0, next, &pull_error) == PullStatus::Timeout &&
              core->pipeline.out_queue.size() == 1U && next.tensors.empty(),
          "DMA loan exhaustion must preserve the pending allocation for retry");
  first = Sample{};
  require(core->pull(1000, next, &pull_error) == PullStatus::Ok,
          "releasing an older DMA view must allow retry without copying");
  GstBuffer* retried =
      pipeline_internal::buffer_from_tensor_holder(next.tensors.front().storage->holder);
  require(retried != nullptr, "retried DMA sample lost its original holder");
  const auto retried_identity = sima_test::dmabuf_span(retried);
  gst_buffer_unref(retried);
  require(retried_identity == queued_identities.back(),
          "loan retry changed the queued DMA allocation or view");
  next = Sample{};
  require(core->holder_loan_gate->inflight() == 0, "DMA retry left an outstanding loan");

  GstBuffer* cpu_buffer = gst_buffer_new_allocate(nullptr, 128, nullptr);
  auto cpu_sample = sample_from_buffer(cpu_buffer);
  gst_buffer_unref(cpu_buffer);
  require(!pipeline_internal::sample_has_dmabuf_memory(cpu_sample) &&
              !pipeline_internal::sample_has_device_gstsample_holder(cpu_sample),
          "ordinary CPU storage must not consume a DMA holder credit");
  auto legacy_sample = make_legacy_device_gst_sample(1);
  require(!pipeline_internal::sample_has_dmabuf_memory(legacy_sample) &&
              pipeline_internal::sample_has_device_gstsample_holder(legacy_sample),
          "legacy device classification must remain distinct from standard DMA");
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
      require_native_retention_pool_policy();
      require_output_policy_matrix();
      require_dmabuf_loans_and_pressure_preserve_storage();
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
      require(!public_core->pipeline.zero_copy_fallback_enabled,
              "explicit public ZeroCopy must not allow a Balanced pressure-copy fallback");
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
        loan_core->pipeline.out_queue.push_back(make_legacy_device_gst_sample(1));
        loan_core->pipeline.out_queue.push_back(make_legacy_device_gst_sample(2));
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

      {
        simaai::neat::graph::runtime::BlockingQueue<int> cancellation_queue(1);
        require(cancellation_queue.push_interruptible(1, 0,
                                                      [&] { return cancellation_queue.closed(); }),
                "interruptible push cancellation predicates may safely inspect their queue");
        int queued = 0;
        require(cancellation_queue.pop(queued, 0) && queued == 1,
                "interruptible push should preserve the item after a queue-inspecting predicate");
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
          simaai::neat::runtime::RuntimeSinkQueueMsg{make_legacy_device_gst_sample(3)});
      graph_core->graph_execution_->sinks[kSinkNode]->push(
          simaai::neat::runtime::RuntimeSinkQueueMsg{make_legacy_device_gst_sample(4)});

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
                  simaai::neat::runtime::RuntimeSinkQueueMsg{make_legacy_device_gst_sample(5)}, 0),
              "bounded graph sink should accept one queued output");
      simaai::neat::Sample bounded_blocked;
      const auto bounded_status = bounded_graph_core->pull(0, bounded_blocked, &err);
      require(bounded_status == simaai::neat::PullStatus::Timeout,
              "bounded graph loan exhaustion should be reported as retryable backpressure");
      require(bounded_graph_core->graph_execution_->sinks[kBoundedSinkNode]->size() == 1U,
              "bounded graph loan exhaustion must restore without dropping the popped output");
      require(!bounded_graph_core->graph_execution_->sinks[kBoundedSinkNode]->try_push(
                  simaai::neat::runtime::RuntimeSinkQueueMsg{make_legacy_device_gst_sample(6)}),
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

        simaai::neat::GraphLinkOptions decoded_link;
        decoded_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        decoded_link.queue_depth = 1;
        decoded_link.stream_id = "redundant_stream" + std::to_string(stream);
        redundant_branch_fan_in_app.connect(source, one_output_branch, decoded_link);

        simaai::neat::GraphLinkOptions detector_link;
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

      simaai::neat::GraphLinkOptions stream_link;
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
      simaai::neat::GraphLinkOptions realtime_credit_options;
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

      simaai::neat::Sample credit_first = make_legacy_device_gst_sample(7);
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

      simaai::neat::Sample credit_second = make_legacy_device_gst_sample(8);
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

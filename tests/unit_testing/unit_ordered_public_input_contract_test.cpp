#include "pipeline/internal/OrderedPublicInputContract.h"
#include "pipeline/internal/PublicInputContract.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include <simaai/scheduler/client/frame_layout.h>
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/NeatError.h"
#include "pipeline/Graph.h"
#include "pipeline/ErrorCodes.h"
#include "nodes/sima/VisualFrontend.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <limits>

namespace {
using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal;

OrderedPublicInputContract contract() {
  return {{{"previous", TensorDType::UInt8, TensorLayout::HW, {1, 8, 16}},
           {"current", TensorDType::UInt8, TensorLayout::HW, {1, 8, 16}},
           {"points", TensorDType::Int32, TensorLayout::HW, {1, 4, 2}}}};
}

Sample input(int generation = 0) {
  // One shared allocation with three nonzero subviews is intentional. Physical
  // segment/name/index metadata belongs to transport and must remain untouched.
  auto owner = make_cpu_owned_storage(512);
  Tensor image;
  image.storage = owner;
  image.dtype = TensorDType::UInt8;
  image.layout = TensorLayout::HW;
  image.shape = {8, 16};
  image.byte_offset = 16 + generation;
  image.route.name = "previous";
  image.route.segment_name = "retained_output";
  image.route.memory_index = 7;
  Tensor current = image;
  current.route.name = "current";
  current.byte_offset = 160 + generation;
  Tensor points = image;
  points.route.name = "points";
  points.layout = TensorLayout::Unknown;
  points.dtype = TensorDType::Int32;
  points.shape = {4, 2};
  points.byte_offset = 320 + 4 * generation;
  return sample_from_tensors({image, current, points});
}

template <class Fn> void rejects(Fn&& fn, const char* detail = nullptr) {
  try {
    fn();
  } catch (const NeatError& error) {
    require(error.report().error_code == error_codes::kInputShape, "wrong input-contract category");
    const std::string text = error.what();
    require(text.find("ordered tensor input contract") != std::string::npos,
            "missing stable input-contract diagnostic");
    if (detail)
      require(text.find(detail) != std::string::npos, "missing slot-specific diagnostic");
    return;
  }
  throw std::runtime_error("malformed ordered input was accepted");
}

void metadata_contract() {
  const auto expected = contract();
  auto valid = input();
  unsigned maps = 0;
  valid.tensors.front().storage->map_fn = [&](MapMode) {
    ++maps;
    throw std::runtime_error("metadata preflight mapped payload");
    return Mapping{};
  };
  validate_ordered_public_input(expected, valid);
  std::array<unsigned, 3> order{0, 1, 2};
  while (std::next_permutation(order.begin(), order.end())) {
    auto wrong = valid;
    for (std::size_t i = 0; i < order.size(); ++i)
      wrong.tensors[i] = valid.tensors[order[i]];
    rejects([&] { validate_ordered_public_input(expected, wrong); }, "slot");
  }
  auto check = [&](auto mutate, const char* detail = nullptr) {
    auto wrong = valid;
    mutate(wrong);
    rejects([&] { validate_ordered_public_input(expected, wrong); }, detail);
    validate_ordered_public_input(expected, valid);
  };
  check([](Sample& s) { s.tensors.pop_back(); }, "expected 3 tensors");
  check([](Sample& s) { s.tensors.push_back(s.tensors.back()); }, "expected 3 tensors");
  check([](Sample& s) { s.tensors[1].route.name = "previous"; }, "slot 1");
  check([](Sample& s) { s.tensors[2].route.name = "other"; }, "slot 2");
  check([](Sample& s) { s.tensors[2].dtype = TensorDType::Float32; }, "dtype mismatch");
  check([](Sample& s) { s.tensors[1].shape = {16, 8}; }, "shape/layout");
  check([](Sample& s) { s.tensors[1].strides_bytes = {17, 1}; }, "noncontiguous");
  check([](Sample& s) { s.tensors[2].byte_offset = 500; }, "storage too small");
  check([](Sample& s) { s.tensors[2].byte_offset = -1; }, "negative");
  check([](Sample& s) { s.tensors[2].byte_offset = 321; }, "aligned");
  auto gray_channel = valid;
  gray_channel.tensors[0].shape = {1, 8, 16, 1};
  gray_channel.tensors[0].layout = TensorLayout::HWC;
  validate_ordered_public_input(expected, gray_channel);
  check([](Sample& s) { s.tensors[2].shape = {std::numeric_limits<std::int64_t>::max(), 2}; });
  check([](Sample& s) { s.tensors[2].strides_bytes = {4}; }, "stride rank");
  for (auto& tensor : valid.tensors)
    tensor.route.name.clear();
  validate_ordered_public_input(expected, valid);
  require(valid.tensors.front().route.segment_name == "retained_output" &&
              valid.tensors.front().route.memory_index == 7 && maps == 0,
          "validation changed physical identity or mapped payload");
  auto rebound = valid;
  rebound.tensors.front().route.name = "previous";
  validate_ordered_public_input(expected, rebound);
  require(valid.tensors.front().route.name.empty() &&
              rebound.tensors.front().storage == valid.tensors.front().storage,
          "shallow logical rebinding changed original metadata or owner");
  validate_ordered_public_input(expected, input(1));
  auto batched = expected;
  auto batch_sample = input();
  for (std::size_t i = 0; i < batched.inputs.size(); ++i) {
    batched.inputs[i].shape.front() = 2;
    batch_sample.tensors[i].shape = batched.inputs[i].shape;
    batch_sample.tensors[i].storage = make_cpu_owned_storage(1024);
  }
  validate_ordered_public_input(batched, batch_sample);
  rejects([&] { validate_ordered_public_input(expected, batch_sample); }, "shape/layout");
}

void queue_contract() {
  auto core = runtime::RunCore::create_graph_compat();
  core->graph_execution_.reset();
  core->mode = RunMode::Async;
  core->pipeline.supports_push = true;
  core->opt.queue_depth = 1;
  core->opt.overflow_policy = OverflowPolicy::KeepLatest;
  core->push_sample_policy = runtime::PushSamplePolicy::PreserveSample;
  core->pipeline.public_input_contract = contract();
  InputOptions options;
  options.payload_type = PayloadType::Tensor;
  options.memory_policy = InputMemoryPolicy::SystemMemory;
  core->pipeline.tensor_input_opt_for_cv = options;
  auto valid = input();
  valid.frame_id = 1;
  require(core->push_samples(Sample{valid}, false), "initial input not admitted");
  const auto submitted = core->inputs_enqueued.load();
  auto wrong = valid;
  std::swap(wrong.tensors[0], wrong.tensors[1]);
  rejects([&] { core->push_samples(Sample{wrong}, false); }, "slot 0");
  require(core->inputs_enqueued.load() == submitted && core->inputs_dropped.load() == 0 &&
              !core->stop_requested.load() && core->pipeline.in_queue.size() == 1 &&
              core->pipeline.in_queue.front().msg.frame_id == 1,
          "invalid input consumed admission, evicted pending work or poisoned Run");
  valid = input(1);
  valid.frame_id = 2;
  require(core->push_samples(Sample{valid}, false), "same Run failed valid-after-invalid");
  const auto& queued = core->pipeline.in_queue.front().msg;
  require(queued.frame_id == 2 && queued.tensors.front().storage == valid.tensors.front().storage &&
              queued.tensors.front().byte_offset == valid.tensors.front().byte_offset,
          "current generation owner/offset was not retained");
  core->close();
}

// A custom context-dependent author must not receive a speculative empty
// compile call simply because its public structural definition has two ports.
class ContextualNode final : public Node, public NodeContractProvider {
public:
  mutable int compilations = 0;
  std::string kind() const override {
    return "ContextualTest";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int) const override {
    return "identity";
  }
  std::vector<std::string> element_names(int) const override {
    return {};
  }
  NodeContractDefinition contract_definition() const override {
    NodeContractDefinition definition;
    definition.inputs.resize(2);
    definition.fields.push_back(
        {"shape", ContractFieldSource::InputOnly, ContractOverridePolicy::Forbidden, true});
    return definition;
  }
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* output,
                             std::string*) const override {
    ++compilations;
    if (!input.ingress.ingress_sample)
      throw std::runtime_error("contextual provider requires its original input facts");
    output->renderable = true;
    return true;
  }
};

void boundary_policy_contract() {
  // A minimal public view exercises topology and opt-in policy without
  // preparing a GStreamer/device session.
  struct Edge {
    std::size_t from, to;
  };
  struct View {
    std::vector<std::shared_ptr<Node>> vertices;
    std::vector<Edge> edges;
    std::vector<runtime::FragmentPlan> fragments;
    GraphOptions options;
  } view;
  TrackKLTOptions options;
  options.width = 64;
  options.height = 64;
  options.win_half = 3;
  options.max_level = 1;
  options.num_points = 4;
  view.vertices = {std::make_shared<Input>(InputOptions{}), std::make_shared<TrackKLT>(options)};
  view.edges = {{0, 1}};
  const auto authored = ordered_public_input_contract(view, 0);
  require(authored && authored->inputs.size() == 3 &&
              authored->inputs[2].dtype == TensorDType::Int32,
          "direct public TrackKLT did not retain explicit ordered contract");
  auto malformed = input();
  // Discovery took only configured options. A malformed first tensor cannot
  // redefine the expected role/geometry or disable strict policy.
  for (std::size_t i = 0; i < malformed.tensors.size(); ++i)
    malformed.tensors[i].route.name = authored->inputs[i].name;
  malformed.tensors[0].shape = {1};
  rejects([&] { validate_ordered_public_input(*authored, malformed); }, "shape/layout");
  view.edges.push_back({0, 1});
  require(!ordered_public_input_contract(view, 0), "fanout guessed an input-role contract");
  view.edges = {{1, 0}, {0, 1}};
  require(!ordered_public_input_contract(view, 0), "internal edge reinterpreted producer labels");
  view.edges = {{0, 1}};
  GriderFastOptions fast;
  fast.width = fast.height = 64;
  view.vertices[1] = std::make_shared<GriderFast>(fast);
  require(!ordered_public_input_contract(view, 0), "single-input path acquired strict role names");
  const auto contextual = std::make_shared<ContextualNode>();
  view.vertices[1] = contextual;
  require(!ordered_public_input_contract(view, 0) && contextual->compilations == 0,
          "optional policy discovery invoked contextual provider without facts");
  ContractCompileInput normal;
  normal.ingress.ingress_sample = input();
  sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled = compile_node_contracts(view.vertices, normal, &diagnostics);
  require(diagnostics.errors.empty() && compiled.fully_renderable && contextual->compilations == 1,
          "normal contextual compilation was changed");

  Graph graph;
  graph.add(nodes::Input(InputOptions{}));
  graph.add(nodes::TrackKLT(options));
  graph.add(nodes::Output());
  const auto plan = runtime::compile_public_graph(graph, RunOptions{}, std::nullopt);
  require(plan.default_input &&
              plan.pipeline_segments.at(plan.default_input->segment).public_input_contract,
          "cold graph compilation lost the endpoint's immutable expected contract");
  rejects([&] { (void)runtime::compile_public_graph(graph, RunOptions{}, malformed); },
          "shape/layout");
}
// Address authority is a transport contract, not a fabricated numerical CVU
// operation. Actual visual operations are checked/prepared separately below.
class PhysicalAuthorityNode final : public Node, public NodeContractProvider {
public:
  sima::PhysicalAddressSource address = sima::PhysicalAddressSource::FrameSpan;
  bool producer = false;
  bool shared_views = false;
  std::string kind() const override {
    return "PhysicalAuthorityTransportTest";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  std::string backend_fragment(int) const override {
    return "identity";
  }
  std::vector<std::string> element_names(int) const override {
    return {};
  }
  NodeContractDefinition contract_definition() const override {
    return {};
  }
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* output,
                             std::string*) const override {
    output->transport.emplace();
    output->transport->payload_kind = sima::StagePayloadKind::None;
    output->transport->plugin_kind = "identity";
    output->element_name = output->logical_stage_id =
        "physical-authority-" + std::to_string(input.node_index);
    output->renderable = true;
    auto& runtime = output->transport->runtime_contract;
    namespace client = simaai::scheduler::client;
    runtime.allocation_policy = client::AllocationPolicy::RequireCma;
    // Imported frame storage is authored as an explicit retained FrameLayout.
    // Ordinary output carriers are placed by the canonical packing pass.
    if (address == sima::PhysicalAddressSource::FrameSpan && !producer) {
      auto layout = std::make_shared<client::FrameLayout>();
      layout->ownership = client::FrameOwnership::ReuseInput;
      layout->allocations = {
          {512, client::kAccessCpuA65, client::AllocationPolicy::PreserveImported}};
      layout->outputs = {{0, {0, 128, 128}}};
      layout->inputs = {{client::KnownFrameSpan{0, {0, 24, 64}}, 4}};
      if (shared_views)
        layout->inputs.push_back({client::KnownFrameSpan{0, {0, 24, 64}}, 20});
      runtime.frame_layout = std::move(layout);
      runtime.reuse_input_pad = 0;
    }
    sima::LogicalInputStaticSpec logical;
    logical.logical_index = logical.physical_index = logical.backend_input_index = 0;
    logical.byte_offset = 4;
    logical.size_bytes = 32;
    logical.dtype = "UINT8";
    logical.shape = {32};
    runtime.logical_inputs.push_back(logical);
    sima::PhysicalBufferStaticSpec physical;
    physical.physical_index = 0;
    physical.size_bytes = 64;
    // FrameLayout authoring uses normalized carrier-local spans. Retain the
    // same nonzero offsets: external 16+8, or producer output 128+8.
    physical.source_byte_offset =
        producer && address != sima::PhysicalAddressSource::RuntimePhysicalBinding ? 136 : 24;
    physical.address_source = address;
    runtime.physical_inputs.push_back(physical);
    sima::InputBindingStaticSpec binding;
    binding.local_logical_input_index = binding.src_physical_output_index = 0;
    binding.sink_pad_index = 0;
    binding.src_physical_byte_offset = 0;
    binding.src_physical_size_bytes = 64;
    binding.src_stage_index = producer ? 0 : -1;
    runtime.input_bindings.push_back(binding);
    if (shared_views) {
      logical.logical_index = logical.backend_input_index = 1;
      logical.byte_offset = 20;
      runtime.logical_inputs.push_back(logical);
      binding.local_logical_input_index = 1;
      runtime.input_bindings.push_back(binding);
    }
    sima::PhysicalBufferStaticSpec result;
    result.physical_index = result.allocator_index = 0;
    result.source_byte_offset = 128;
    result.size_bytes = 128;
    runtime.physical_outputs.push_back(result);
    sima::LogicalTensorStaticSpec view;
    view.logical_index = view.physical_index = view.backend_output_index = 0;
    view.dtype = "UINT8";
    view.shape = {128};
    view.size_bytes = 128;
    runtime.logical_outputs.push_back(view);
    return true;
  }
};

void check_final_authority(const simaai::scheduler::client::FrameLayout& layout,
                           sima::PhysicalAddressSource expected, bool producer, bool shared) {
  namespace client = simaai::scheduler::client;
  require(layout.inputs.size() == (shared ? 2U : 1U), "final shared-view cardinality changed");
  for (std::size_t i = 0; i < layout.inputs.size(); ++i) {
    const auto& input = layout.inputs[i];
    require(input.view_offset == (i == 0 ? 4U : 20U),
            "final layout lost or doubled a logical subview offset");
    if (expected == sima::PhysicalAddressSource::RuntimePhysicalBinding) {
      const auto* source = std::get_if<client::PhysicalBinding>(&input.source);
      require(source && source->pad == 0 && source->binding_index == 0 &&
                  source->byte_offset == 24 && source->size_bytes == 64,
              "physical binding must preserve carrier16+relative8 exactly once");
    } else if (expected == sima::PhysicalAddressSource::DirectCarrierSpan && !producer) {
      const auto* source = std::get_if<client::DirectMemory>(&input.source);
      require(source && source->pad == 0 && source->memory_index == 0 &&
                  source->byte_offset == 24 && source->size_bytes == 64,
              "direct carrier must not turn into binding-table or frame authority");
    } else {
      const auto* source = std::get_if<client::KnownFrameSpan>(&input.source);
      // An internal frame uses its actual producer's output base128, not the
      // consumer's independent carrier base16. External reuse owns its own base.
      require(source && source->pad == 0 && source->span.allocation_index == 0 &&
                  source->span.byte_offset == (producer ? 136U : 24U) &&
                  source->span.size_bytes == 64,
              "known frame must preserve its real owner plus relative8 exactly once");
    }
  }
}

void physical_authority_contract() {
  using sima::PhysicalAddressSource;
  TrackKLTOptions tracking;
  tracking.width = tracking.height = 64;
  tracking.num_points = 4;
  tracking.win_half = 3;
  tracking.max_level = 1;
  GriderFastOptions fast;
  fast.width = fast.height = 64;
  TrackDescriptorOptions descriptor;
  descriptor.width = descriptor.height = 64;
  const std::vector<std::shared_ptr<Node>> consumers{std::make_shared<TrackKLT>(tracking),
                                                     std::make_shared<GriderFast>(fast),
                                                     std::make_shared<TrackDescriptor>(descriptor)};
  InputOptions public_input;
  public_input.payload_type = PayloadType::Tensor;
  public_input.memory_policy = InputMemoryPolicy::Ev74;
  for (const auto& consumer : consumers) {
    const std::vector<std::shared_ptr<Node>> nodes{std::make_shared<Input>(public_input), consumer};
    const auto context_name = consumer->kind() + " public input: ";
    ContractCompileInput context;
    sima::ManifestBuildDiagnostics diagnostics;
    const auto compiled = compile_node_contracts(nodes, context, &diagnostics);
    require(diagnostics.errors.empty() && compiled.stages.size() == 1 &&
                compiled.stages.front().processcvu.has_value(),
            context_name + "visual contract compilation failed before final placement");
    const auto stage = project_processcvu_stage_contract(*compiled.stages.front().processcvu);
    const auto expected = PhysicalAddressSource::DirectCarrierSpan;
    require(stage.physical_inputs.size() == (consumer->kind() == "TrackKLT" ? 3U : 1U),
            context_name + "compilation changed physical input cardinality");
    require(stage.input_bindings.size() == stage.physical_inputs.size() &&
                stage.logical_inputs.size() == stage.physical_inputs.size(),
            context_name + "compilation changed input binding/logical cardinality");
    for (std::size_t i = 0; i < stage.physical_inputs.size(); ++i) {
      const auto& physical = stage.physical_inputs[i];
      const auto& binding = stage.input_bindings[i];
      const auto label = context_name + "input " + std::to_string(i) + ": ";
      require(physical.address_source == expected, label + "incorrect address authority");
      require(physical.physical_index == int(i), label + "incorrect physical input order");
      require(physical.source_byte_offset == 0 && physical.size_bytes > 0,
              label + "incorrect physical span");
      require(binding.sink_pad_index == 0 && binding.local_logical_input_index == int(i) &&
                  binding.src_stage_index < 0 && binding.src_stage_id.empty() &&
                  binding.src_physical_byte_offset == 0,
              label + "incorrect public-root route");
      if (consumer->kind() == "GriderFast") {
        // Single-input/single-output facts leave upstream selectors unset.
        // Final placement resolves this public root from physical_index; it
        // must not fabricate an upstream producer to satisfy an early check.
        require(binding.src_logical_output_index == -1 && binding.src_output_slot == -1 &&
                    binding.src_physical_output_index == -1 && binding.src_physical_size_bytes == 0,
                label + "incorrect implicit single-IO root contract");
      } else {
        require(binding.src_logical_output_index == int(i) && binding.src_output_slot == int(i) &&
                    binding.src_physical_output_index == int(i) &&
                    binding.src_physical_size_bytes == physical.size_bytes,
                label + "incorrect explicit multi-IO root contract");
      }
      require(stage.logical_inputs[i].backend_input_index == int(i),
              label + "incorrect backend input order");
    }
    auto manifest = render_manifest_from_compiled_contracts(compiled, context, &diagnostics);
    require(manifest && diagnostics.errors.empty(), context_name + "visual contract render failed");
    const auto failure = sima::prepare_kernel_stage_recipes(&*manifest, {}, {}, {}, {});
    require(!failure, context_name + "visual checked recipe preparation failed");
    sima::finalize_pipeline_storage(*manifest);
    const auto& final = manifest->stages.front();
    require(final.native_kernel && final.frame_layout &&
                final.frame_layout->inputs.size() == stage.physical_inputs.size(),
            context_name + "inputs did not survive checked recipe/final allocation placement");
    namespace client = simaai::scheduler::client;
    for (std::size_t i = 0; i < final.frame_layout->inputs.size(); ++i) {
      const auto& input = final.frame_layout->inputs[i];
      const auto label = context_name + "final input " + std::to_string(i) + ": ";
      require(input.view_offset == 0, label + "spurious view offset");
      // Final placement resolves standalone public tensor views after producer
      // discovery. Their ordinal selects this
      // frame's binding table, not an allocation index.
      const auto* source = std::get_if<client::PhysicalBinding>(&input.source);
      require(source && source->pad == 0 && source->binding_index == i &&
                  source->byte_offset == 0 &&
                  source->size_bytes == stage.physical_inputs[i].size_bytes,
              label + "public tensor did not become an exact PhysicalBinding");
    }
  }
  for (const auto address :
       {PhysicalAddressSource::DirectCarrierSpan, PhysicalAddressSource::RuntimePhysicalBinding,
        PhysicalAddressSource::FrameSpan}) {
    for (const bool producer : {false, true})
      for (const bool shared : {false, true}) {
        const auto node = std::make_shared<PhysicalAuthorityNode>();
        node->address = address;
        node->producer = producer;
        node->shared_views = shared;
        std::vector<std::shared_ptr<Node>> nodes{std::make_shared<Input>(InputOptions{})};
        if (producer) {
          const auto upstream = std::make_shared<PhysicalAuthorityNode>();
          upstream->address = PhysicalAddressSource::DirectCarrierSpan;
          nodes.push_back(upstream);
        }
        nodes.push_back(node);
        ContractCompileInput context;
        // Transport retains its explicitly authored physical authority.
        sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled = compile_node_contracts(nodes, context, &diagnostics);
        require(diagnostics.errors.empty() && compiled.fully_renderable &&
                    compiled.stages.size() == (producer ? 2U : 1U) &&
                    compiled.stages.back().transport && !compiled.stages.back().processcvu &&
                    !compiled.stages.back().resolved_executor,
                "non-executing transport authority fixture acquired a numerical executor");
        const auto& runtime = compiled.stages.back().transport->runtime_contract;
        const auto expected = address;
        require(runtime.physical_inputs.front().address_source == expected &&
                    runtime.physical_inputs.front().source_byte_offset ==
                        (producer && address != PhysicalAddressSource::RuntimePhysicalBinding
                             ? 136
                             : 24) &&
                    runtime.input_bindings.front().src_physical_byte_offset == 0 &&
                    runtime.input_bindings.front().src_physical_size_bytes == 64,
                "compilation changed a frame/producer authority or authored relative span");
        auto manifest = render_manifest_from_compiled_contracts(compiled, context, &diagnostics);
        require(manifest && diagnostics.errors.empty(), "transport authority render failed");
        sima::finalize_pipeline_storage(*manifest);
        require(bool(manifest->stages.back().frame_layout), "transport final FrameLayout missing");
        check_final_authority(*manifest->stages.back().frame_layout, expected, producer, shared);
      }
  }
  // Contract compilation must preserve transitive consumers' authored authority.
  const std::vector<std::shared_ptr<Node>> transitive{std::make_shared<Input>(InputOptions{}),
                                                      std::make_shared<GriderFast>(fast),
                                                      std::make_shared<TrackKLT>(tracking)};
  ContractCompileInput context;
  sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled = compile_node_contracts(transitive, context, &diagnostics);
  require(diagnostics.errors.empty() && compiled.stages.size() == 2,
          "transitive contract fixture failed");
  require(
      compiled.stages.back().processcvu->runtime_contract.physical_inputs.front().address_source ==
          PhysicalAddressSource::DirectCarrierSpan,
      "contract compilation changed a transitive consumer's address authority");
}

} // namespace
RUN_TEST("unit_ordered_public_input_contract_test", ([] {
           metadata_contract();
           queue_contract();
           boundary_policy_contract();
           physical_authority_contract();
         }));

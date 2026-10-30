#include "../asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.h"

#include <simaai/scheduler/client/kernel.h>
#include <simaai/scheduler/testing/route_probe.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace simaai::neat;
namespace ps = pipeline_internal::sima;
namespace semantics = ps::stagesemantics;
namespace client = simaai::scheduler::client;
namespace test = simaai::scheduler::testing;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Independent expectations from the four registry MPKs, not the production SSD
// recipe table. BF16 is linear HWC with C8-aligned pitch, NOT twice INT8 C16.
constexpr std::array<int, 6> kSides{19, 10, 5, 3, 2, 1};
constexpr std::array<int, 12> kLogicalChannels{12,  24,  24,  24,  24,  24,
                                               273, 546, 546, 546, 546, 546};
constexpr std::array<int, 12> kInt8Pitch{16, 32, 32, 32, 32, 32, 288, 560, 560, 560, 560, 560};
constexpr std::array<int, 12> kBf16Pitch{16, 24, 24, 24, 24, 24, 280, 552, 552, 552, 552, 552};
constexpr std::array<std::uint64_t, 12> kInt8Bytes{5776,   3200,  800,   288,  128,  32,
                                                   103968, 56000, 14000, 5040, 2240, 560};
constexpr std::array<std::uint64_t, 12> kBf16Bytes{11552,  4800,   1200,  432,  192,  48,
                                                   202160, 110400, 27600, 9936, 4416, 1104};

ps::BoxDecodeStaticContract extract(const ps::MpkContract& mpk, bool integer,
                                    const std::string& context) {
  std::string error;
  const auto flags = ps::resolve_model_managed_boxdecode_route_flags_from_mpk(mpk, nullptr, &error);
  require(flags.has_value(), context + ": route facts: " + error);
  require(!flags->tess_needed && flags->quant_needed == integer,
          context + ": linear padded heads must not become tessellated or change dtype");
  const auto contract = ps::build_boxdecode_static_contract_from_mpk(mpk, *flags, &error);
  require(contract.has_value(), context + ": BoxDecode extraction: " + error);
  return semantics::finalize_boxdecode_static_contract(*contract, BoxDecodeType::Ssd, std::nullopt,
                                                       flags, BoxDecodeTypeOption::Auto, .2, .6,
                                                       100, 0, {});
}

void check_one(const std::string& model, bool integer, bool permuted = false,
               std::size_t prefix = 0) {
  const auto name = model + (integer ? "-w8a8" : "-w16a16");
  const auto path =
      sima_test::test_shared_asset_path("tests/assets/ssd-contract/" + name + "-mpk.json");
  std::string error;
  const auto loaded = ps::static_contract::MpkDecoder::load(path, &error);
  require(loaded.has_value(), name + ": actual MPK load: " + error);
  const auto& mpk = loaded->contract;
  const auto* mla = ps::get_mla_stage_io_contract(mpk);
  require(mla && mla->output_tensors.size() == 12, name + ": expected twelve MLA roots");
  const auto& pitch = integer ? kInt8Pitch : kBf16Pitch;
  const auto& bytes = integer ? kInt8Bytes : kBf16Bytes;
  const std::int64_t scalar = integer ? 1 : 2;
  const auto expected_dtype = integer ? client::DType::Int8 : client::DType::BFloat16;
  for (std::size_t i = 0; i < bytes.size(); ++i)
    require(mla->output_tensors[i].size_bytes == bytes[i],
            name + ": unexpected authoritative root extent at " + std::to_string(i));

  const auto extracted = extract(mpk, integer, name);
  require(extracted.ssd_recipe_id == semantics::SsdRecipeId::SsdMobile300V1 &&
              extracted.tensors.size() == 12,
          name + ": wrong resolved SSD signature");

  // The compiler's slice names contain "unpack", but their typed kernel is
  // slice_transform. Names must not select another physical-storage contract.
  auto renamed = mpk;
  for (std::size_t i = 0; i < renamed.plugins.size(); ++i) {
    auto& plugin = renamed.plugins[i];
    if (plugin.kernel == "slice_transform") {
      plugin.name = "renamed_slice_" + std::to_string(i);
      plugin.plugin_id = plugin.name;
    }
  }
  const auto renamed_contract = extract(renamed, integer, name + " renamed slices");
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto& head = extracted.tensors[i];
    const int side = kSides[i % kSides.size()];
    require(head.source_physical_index == static_cast<int>(i) && head.source_byte_offset == 0 &&
                head.source_size_bytes == bytes[i],
            name + ": lost distinct root/extent/relative offset at " + std::to_string(i));
    require(head.source_storage_kind == ps::BoxDecodeSourceStorageKind::DenseHwcPhysical &&
                head.input_shape == std::vector<int>({side, side, pitch[i]}) &&
                head.slice_shape == std::vector<int>({side, side, kLogicalChannels[i]}),
            name + ": physical pitch or semantic shape changed at " + std::to_string(i));
    const auto& control = renamed_contract.tensors[i];
    require(control.source_size_bytes == head.source_size_bytes &&
                control.input_shape == head.input_shape &&
                control.slice_shape == head.slice_shape &&
                control.source_storage_kind == head.source_storage_kind,
            name + ": slice spelling changed physical interpretation");
  }

  // Build the same upstream/static and native BoxDecode contracts as production;
  // no ELF is read and no MLA or numerical operation is submitted by this test.
  CompiledNodeContract producer;
  producer.element_name = producer.logical_stage_id = "ssd_mla";
  producer.plugin_kind = "processmla";
  producer.renderable = true;
  producer.processmla =
      semantics::build_mla_compiled_contract(ps::build_mla_static_contract_from_mpk_stage(
          *mla, ps::get_mla_logical_outputs_contract(mpk),
          ps::get_mla_boundary_physical_outputs_contract(mpk), producer.element_name));
  auto compiled = semantics::build_boxdecode_compiled_contract(extracted);
  // Do not repair source selectors in the fixture: production admission must
  // preserve the true producer even when the MPK head uses a Slice alias.
  for (const auto& binding : compiled.runtime_contract.input_bindings)
    require(binding.src_stage_index == -1 && binding.src_stage_id.empty(),
            name + ": fixture no longer exercises unprojected model head aliases");
  if (permuted) {
    // Physical IDs are identities, not vector positions, logical indices, or
    // allocation ordinals. Reverse and sparsify them without changing tensors.
    const auto remap = [](int index) { return 100 + 3 * (11 - index); };
    for (auto& physical : producer.processmla->runtime_contract.physical_outputs)
      physical.physical_index = remap(physical.physical_index);
    for (auto& logical : producer.processmla->runtime_contract.logical_outputs)
      logical.physical_index = remap(logical.physical_index);
    for (auto& binding : compiled.runtime_contract.input_bindings)
      binding.src_physical_output_index = remap(binding.src_physical_output_index);
    std::reverse(producer.processmla->runtime_contract.physical_outputs.begin(),
                 producer.processmla->runtime_contract.physical_outputs.end());
  }
  if (prefix) {
    for (auto& physical : producer.processmla->runtime_contract.physical_outputs)
      physical.size_bytes += prefix;
    for (auto& logical : producer.processmla->runtime_contract.logical_outputs)
      logical.byte_offset += prefix;
    for (auto& logical : compiled.runtime_contract.logical_inputs)
      logical.byte_offset += prefix;
  }
  ContractCompileInput context;
  context.immediate_upstream = &producer;
  const auto expect_rejected = [&](const auto& changed, const auto& changed_context,
                                   const std::string& expected) {
    try {
      (void)ps::compile_boxdecode_kernel(
          changed, changed_context, {300, 300, 300, 300, ResizeMode::Stretch}, ResizeMode::Stretch);
    } catch (const std::invalid_argument& error) {
      require(std::string(error.what()).find(expected) != std::string::npos,
              name + ": wrong rejection: " + error.what());
      return;
    }
    require(false, name + ": accepted " + expected);
  };
  auto missing = compiled;
  missing.runtime_contract.input_bindings.front().src_physical_output_index = 999;
  expect_rejected(missing, context, "no authoritative upstream physical carrier");
  auto conflicting = compiled;
  conflicting.runtime_contract.input_bindings.front().src_stage_id = "another_producer";
  expect_rejected(conflicting, context, "conflicting upstream carrier identity");
  auto unnamed = producer;
  unnamed.element_name.clear();
  unnamed.logical_stage_id.clear();
  auto unnamed_context = context;
  unnamed_context.immediate_upstream = &unnamed;
  expect_rejected(compiled, unnamed_context, "upstream carrier has no stage identity");
  auto ambiguous = producer;
  ambiguous.processmla->runtime_contract.physical_outputs.push_back(
      ambiguous.processmla->runtime_contract.physical_outputs.front());
  auto ambiguous_context = context;
  ambiguous_context.immediate_upstream = &ambiguous;
  expect_rejected(compiled, ambiguous_context, "ambiguous upstream physical carrier");
  auto native = ps::compile_boxdecode_kernel(
      compiled, context, {300, 300, 300, 300, ResizeMode::Stretch}, ResizeMode::Stretch);
  for (const auto& binding : native.runtime_contract.input_bindings)
    require(binding.src_stage_id == producer.logical_stage_id && binding.src_stage_index == -1,
            name + ": admission lost the actual producer behind a Slice alias");
  require(native.recipe && native.recipe->members.size() == 2,
          name + ": unexpected detess/cast/dequant staging member");
  auto inspected = client::inspect_kernel(native.recipe->members.front().executable);
  require(std::holds_alternative<client::KernelInspection>(inspected),
          name + ": native recipe was not admitted");
  const auto inspection = std::get<client::KernelInspection>(std::move(inspected));
  const auto specs = inspection.inputs();
  const auto requirements = inspection.input_requirements();
  const auto& ports = native.recipe->members.front().inputs;
  require(specs.size() == 12 && requirements.size() == 12 && ports.size() == 12,
          name + ": native port count changed");
  for (std::size_t i = 0; i < specs.size(); ++i) {
    require(ports[i].source == client::TensorStagePortSource::Input && ports[i].tensor == i &&
                ports[i].byte_offset == 0,
            name + ": native port lost its distinct root at " + std::to_string(i));
    const auto side = kSides[i % kSides.size()];
    require(specs[i].dtype == expected_dtype &&
                specs[i].shape == std::vector<std::int64_t>({side, side, kLogicalChannels[i]}),
            name + ": native dtype or logical shape changed at " + std::to_string(i));
    if (pitch[i] != kLogicalChannels[i]) {
      const auto* strided = std::get_if<client::Strided>(&specs[i].layout);
      require(strided && strided->strides_bytes ==
                             std::array<std::int64_t, 6>{side * pitch[i] * scalar,
                                                         pitch[i] * scalar, scalar, 0, 0, 0},
              name + ": padded HWC pitch was discarded at " + std::to_string(i));
    }
    const auto span = ((side * side - 1) * pitch[i] + kLogicalChannels[i]) * scalar;
    // Native dense-HWC readers retain the complete physical pixel pitch,
    // including the final pixel's padding. The last logical element alone is
    // not their storage contract (for example BF16 head 0: 11544 vs 11552).
    require(requirements[i].storage_bytes == bytes[i] &&
                requirements[i].storage_bytes >= static_cast<std::uint64_t>(span),
            name + ": native storage mismatch at " + std::to_string(i) +
                ": required=" + std::to_string(requirements[i].storage_bytes) +
                " logical_span=" + std::to_string(span) + " root=" + std::to_string(bytes[i]));
  }

  CompiledNodeContract consumer;
  consumer.element_name = consumer.logical_stage_id = "ssd_boxdecode";
  consumer.plugin_kind = "kernel";
  consumer.renderable = true;
  consumer.kernel = std::move(native);
  // This is a manifest-only carrier regression, not an MLA executable fixture.
  // Preserve the real compiled MLA I/O snapshot and MLA memory-access policy
  // for placement, without fabricating ELF port names or alignment evidence.
  // Actual Model tests cover executable-aware MLA rendering separately.
  ps::resolve_compiled_execution_targets(producer, {});
  CompiledTransportContract carrier_snapshot;
  carrier_snapshot.plugin_kind = "processmla";
  carrier_snapshot.kernel_kind = "mla";
  carrier_snapshot.payload_kind = ps::StagePayloadKind::ProcessMla;
  carrier_snapshot.model_managed_stage = true;
  carrier_snapshot.runtime_contract = std::move(producer.processmla->runtime_contract);
  producer.processmla.reset();
  producer.transport = std::move(carrier_snapshot);
  CompiledPipelineContracts pipeline;
  pipeline.fully_renderable = true;
  pipeline.stages.push_back(std::move(producer));
  pipeline.stages.push_back(std::move(consumer));
  for (auto& stage : pipeline.stages)
    ps::resolve_compiled_execution_targets(stage, {});
  ps::ManifestBuildDiagnostics diagnostics;
  auto manifest = render_manifest_from_compiled_contracts(pipeline, {}, &diagnostics);
  std::string render_error = name + ": render failed";
  for (const auto& diagnostic : diagnostics.errors)
    render_error += ": " + diagnostic;
  require(manifest && diagnostics.errors.empty(), render_error);
  ps::finalize_pipeline_storage(*manifest);
  require(manifest->stages.size() == 2 && manifest->stages[1].frame_layout,
          name + ": missing final placed carrier layout");
  require(bool(manifest->stages[0].frame_layout), name + ": missing producer layout");
  const auto& produced = *manifest->stages[0].frame_layout;
  auto& stage = manifest->stages[1];
  const auto& layout = *stage.frame_layout;
  require(layout.inputs.size() == 12, name + ": placed input count changed");
  std::set<std::pair<std::size_t, std::size_t>> addresses;
  for (std::size_t i = 0; i < layout.inputs.size(); ++i) {
    const auto& input = layout.inputs[i];
    const auto* source = std::get_if<client::KnownFrameSpan>(&input.source);
    require(source && input.view_offset == prefix && source->span.size_bytes == bytes[i] + prefix &&
                requirements[i].storage_bytes <= source->span.size_bytes - input.view_offset,
            name + ": placed head exceeds carrier or doubles offset at " + std::to_string(i));
    const auto source_id = compiled.runtime_contract.input_bindings[i].src_physical_output_index;
    const auto output = std::find_if(produced.outputs.begin(), produced.outputs.end(),
                                     [&](const auto& value) { return value.tensor == source_id; });
    require(output != produced.outputs.end() &&
                source->span.allocation_index == output->span.allocation_index &&
                source->span.byte_offset == output->span.byte_offset &&
                source->span.size_bytes == output->span.size_bytes,
            name + ": head was not projected onto its actual published producer span");
    require(addresses.emplace(source->span.allocation_index, source->span.byte_offset).second,
            name + ": two independent MLA heads address the same placed root");
  }

  client::StageRequest request;
  request.layout = stage.frame_layout;
  require(ps::build_kernel_stage_publication(stage, &request.publication, &error),
          name + ": publication: " + error);
  client::KernelProgram program;
  program.target = stage.native_kernel->target;
  program.members = stage.native_kernel->members;
  program.constant_data = stage.native_kernel->constant_data;
  const auto factory_error = ps::wire_kernel_stage_factories(*stage.native_kernel, program);
  require(!factory_error,
          name + ": factories: " + (factory_error ? factory_error->message : std::string()));
  request.program = std::move(program);
  test::RouteProbe probe;
  const auto failure = probe.prepare(request);
  require(!failure, name + ": real Scheduler cold preparation: " +
                        (failure ? failure->field + ": " + failure->message : std::string()));
  require(probe.member_count() == 2, name + ": Scheduler changed the checked member count");

  // A one-byte-short authored view must still be rejected by the real Scheduler
  // capacity check. The positive path above must not be repaired by disabling it.
  auto short_layout = std::make_shared<client::FrameLayout>(layout);
  auto& last = std::get<client::KnownFrameSpan>(short_layout->inputs.back().source);
  last.span.size_bytes =
      short_layout->inputs.back().view_offset + requirements.back().storage_bytes - 1;
  request.layout = std::move(short_layout);
  test::RouteProbe short_probe;
  const auto rejected = short_probe.prepare(request);
  require(rejected && rejected->field == "input.routing" &&
              rejected->message.find("kernel input exceeds authored carrier span") !=
                  std::string::npos,
          name + ": one-byte-short input no longer fails closed");
  std::cout << name << ": exact MPK -> 12 strided native heads -> placed carrier admission PASS\n";
}
} // namespace

void check_ssd_mpk_carrier_contracts() {
  for (const auto* model : {"ssd_mobilenet_v1", "ssd_mobilenet_v2"})
    for (const bool integer : {false, true})
      check_one(model, integer);
  check_one("ssd_mobilenet_v1", true, true);
  check_one("ssd_mobilenet_v2", false, true, 64);
}

// Optional artifact-backed branch of the existing strict test executable. The
// caller supplies an authentic SSD MPK with its actual ELF; missing artifacts
// are errors, not skips. This compiles/places the public Model route without
// constructing a Run or invoking any numerical member or MLA executable.
void check_ssd_model_carrier_contract(const std::string& archive) {
  Model::Options options;
  options.preprocess.kind = InputKind::Image;
  options.preprocess.enable = AutoFlag::On;
  options.preprocess.resize.enable = AutoFlag::On;
  options.preprocess.resize.mode = ResizeMode::Stretch;
  options.preprocess.normalize.enable = AutoFlag::On;
  options.preprocess.normalize.mean = {.5f, .5f, .5f};
  options.preprocess.normalize.stddev = {.5f, .5f, .5f};
  options.preprocess.normalize.has_explicit_stats = true;
  options.decode_type = BoxDecodeType::Ssd;
  options.score_threshold = .2f;
  options.nms_iou_threshold = .6f;
  options.top_k = 100;
  Model model(archive, options);
  auto input = model.input_appsrc_options(false);
  input.width = 1280;
  input.height = 720;
  input.format = "RGB";
  std::vector<std::shared_ptr<Node>> nodes{nodes::Input(input)};
  auto route = internal::ModelAccess::build_public_route_nodes(model, {});
  nodes.insert(nodes.end(), route.begin(), route.end());
  nodes.push_back(nodes::Output());
  nodes = session_build_materialize_model_bound_nodes(nodes, false);
  session_build_apply_derived_input_contracts(&nodes);
  auto build = build_pipeline_full(nodes, false, "mysink", false, {});
  session_build_compile_contracts(&build, nodes, ContractCompileInput{}, "ssd_carrier_test",
                                  &nodes);
  require(build.manifest_diagnostics.errors.empty(), "real SSD Model contract compilation failed");
  session_build_prepare_stage_contracts(build, "ssd_carrier_test");
  require(bool(build.prepared_manifest), "real SSD Model placement missing");
  const auto& stages = build.prepared_manifest->stages;
  const auto producer = std::find_if(stages.begin(), stages.end(), [](const auto& stage) {
    return stage.payload_kind == ps::StagePayloadKind::ProcessMla;
  });
  const auto consumer = std::find_if(stages.begin(), stages.end(), [](const auto& stage) {
    return stage.native_kernel && stage.native_kernel->target == client::Hardware::A65 &&
           stage.logical_inputs.size() == 12;
  });
  require(producer != stages.end() && consumer != stages.end() && producer->frame_layout &&
              consumer->frame_layout,
          "real SSD Model lost MLA or native BoxDecode layout");
  const auto& published = *producer->frame_layout;
  const auto& consumed = *consumer->frame_layout;
  require(published.allocations.size() == 1 && published.outputs.size() == 12 &&
              consumed.inputs.size() == 12,
          "real SSD Model must pack twelve MLA roots into one published allocation");
  const bool integer = consumer->logical_inputs.front().dtype == "INT8";
  const auto& bytes = integer ? kInt8Bytes : kBf16Bytes;
  constexpr std::array<std::size_t, 12> int8_offsets{
      180224, 196608, 204800, 212992, 217088, 221184, 0, 106496, 163840, 188416, 200704, 208896};
  if (integer)
    require(published.allocations.front().size_bytes == 225280,
            "actual SSD INT8 packed arena size changed");
  std::set<std::size_t> offsets;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto output = std::find_if(published.outputs.begin(), published.outputs.end(),
                                     [&](const auto& value) { return value.tensor == i; });
    const auto* source = std::get_if<client::KnownFrameSpan>(&consumed.inputs[i].source);
    require(output != published.outputs.end() && source && source->pad == 0 &&
                source->span.allocation_index == output->span.allocation_index &&
                source->span.byte_offset == output->span.byte_offset &&
                source->span.size_bytes == bytes[i] && output->span.size_bytes == bytes[i] &&
                consumed.inputs[i].view_offset == 0,
            "real SSD head " + std::to_string(i) + " did not use its published MLA span");
    require(source->span.allocation_index == 0 && offsets.insert(source->span.byte_offset).second,
            "real SSD independent heads alias or use an unpublished allocation");
    if (integer)
      require(source->span.byte_offset == int8_offsets[i],
              "real SSD physical-to-packed projection changed at " + std::to_string(i));
  }
  client::StageRequest request;
  request.layout = consumer->frame_layout;
  std::string error;
  require(ps::build_kernel_stage_publication(*consumer, &request.publication, &error),
          "real SSD publication: " + error);
  client::KernelProgram program;
  program.target = consumer->native_kernel->target;
  program.members = consumer->native_kernel->members;
  program.constant_data = consumer->native_kernel->constant_data;
  const auto factory_error = ps::wire_kernel_stage_factories(*consumer->native_kernel, program);
  require(!factory_error,
          "real SSD factory: " + (factory_error ? factory_error->message : std::string()));
  // The actual dynamic recipe and its final carrier placements are asserted
  // above. RouteProbe intentionally cannot own frame-dependent resolvers.
  // Admit an unchanged checked seed copy for static capacity validation only;
  // dynamic execution remains a separately guarded public Model test.
  bool dynamic_recipe = bool(program.frame_adapter_factory);
  for (const auto& member : program.members)
    dynamic_recipe = dynamic_recipe || bool(member.resolve_factory);
  require(dynamic_recipe, "real SSD must retain its production dynamic recipe");
  program.frame_adapter_factory = {};
  for (auto& member : program.members)
    member.resolve_factory = {};
  std::cout << "SSD_STATIC_SEED_CAPACITY_ONLY_NOT_DYNAMIC_QUALIFICATION\n";
  request.program = std::move(program);
  test::RouteProbe probe;
  const auto failure = probe.prepare(request);
  require(!failure, "real SSD placed Scheduler admission: " +
                        (failure ? failure->field + ": " + failure->message : std::string()));
  std::cout << "SSD_MODEL_PACKED_CARRIER_PASS_NO_RUN_NO_MEMBER_SUBMISSION\n";
}

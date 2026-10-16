#include "support/processcvu_plan_fixture.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/KernelRunnerIntegration.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "test_main.h"

#include <array>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gstsimaaitensorbuffer.h>
#include <simaai/neat/internal/dmabuf/DmaBufPool.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gst/gst.h>
#include <nlohmann/json.hpp>
#include <optional>
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <simaai/scheduler/kernel_preparation.h>
#include <string>
#include <vector>

namespace {

namespace ps = simaai::neat::pipeline_internal::sima;
namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;
namespace pss = simaai::neat::pipeline_internal::sima::stagesemantics;
namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
using json = nlohmann::json;

enum class ExecutionMode { None, Preflight, EV74 };

void ensure_gst_ready() {
  static bool ready = false;
  if (!ready) {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
    require(simaai::gst::tensor_buffer_register_meta(), "canonical tensor metadata registration");
    ready = true;
  }
}

std::filesystem::path write_mpk_fixture(const std::string& name, const bool align_c16,
                                        const bool cblock, const std::size_t transport_bytes) {
  const auto root =
      std::filesystem::temp_directory_path() / ("sima_processcvu_v2_encoding_" + name);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  require(!ec, "failed to create ProcessCVU V2 fixture root");

  json plugins = json::array();
  plugins.push_back({{"name", "MLA_0"},
                     {"sequence", 1},
                     {"processor", "MLA"},
                     {"config_params", {{"desired_batch_size", 1}, {"actual_batch_size", 1}}},
                     {"resources", {{"executable", "fixture.elf"}}},
                     {"input_nodes", {{{"name", "model_input"}, {"type", "buffer"}, {"size", 1}}}},
                     {"output_nodes",
                      {{{"name", "descriptor"},
                        {"type", "buffer"},
                        {"size", transport_bytes},
                        {"dtype", "bfloat16"},
                        {"shape", json::array({1, transport_bytes})}}}}});
  plugins.push_back(
      {{"name", "detess_descriptor"},
       {"sequence", 2},
       {"processor", "EV74"},
       {"config_params",
        {{"desired_batch_size", 1},
         {"actual_batch_size", 1},
         {"kernel", "detessellation_transform"},
         {"params",
          {{"slice_shape", json::array({2, 3, 33})},
           {"align_c16", align_c16},
           {"cblock", cblock},
           {"frame_type", "bfloat16"},
           {"frame_shape", json::array({2, 3, 33})},
           {"input_shapes", json::array({json::array({1, transport_bytes})})},
           {"output_shapes", json::array({json::array({2, 3, 33})})}}}}},
       {"input_nodes", {{{"name", "descriptor"}, {"type", "buffer"}, {"size", transport_bytes}}}},
       {"output_nodes",
        {{{"name", "detess_output"},
          {"type", "buffer"},
          {"size", 396},
          {"dtype", "bfloat16"},
          {"shape", json::array({2, 3, 33})}}}}});

  std::ofstream out(root / "mpk.json");
  require(out.is_open(), "failed to open ProcessCVU V2 fixture manifest");
  out << json{{"name", "processcvu_v2_encoding_fixture"},
              {"model_path", "fixture.elf"},
              {"input_nodes", {{{"name", "model_input"}, {"type", "buffer"}, {"size", 1}}}},
              {"plugins", std::move(plugins)}}
             .dump(2);
  out.close();
  require(out.good(), "failed to finalize ProcessCVU V2 fixture manifest");
  return root;
}

ps::StageStaticSpec make_stage(const simaai::neat::CompiledProcessCvuContract& compiled,
                               const std::string& name) {

  ps::StageStaticSpec stage;
  stage.element_name = name;
  stage.logical_stage_id = name;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = compiled.payload.graph_name;
  stage.payload_kind = ps::StagePayloadKind::ProcessCvu;
  stage.processcvu = compiled.payload;
  stage.processcvu.exact_stage_name_or_id = name;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.output_quant = compiled.runtime_contract.output_quant;

  return stage;
}

ps::StageStaticSpec make_stage(const pss::CompiledProcessCvuRuntimeConfig& runtime,
                               const std::string& name) {
  return make_stage(pss::build_processcvu_compiled_contract_from_runtime_config(runtime), name);
}

ps::SimaPluginStaticManifest make_manifest(const ps::MpkContract& contract) {
  require(contract.plugins.size() == 2U, "fixture should contain MLA and detess stages");
  const auto& detess = contract.plugins[1];
  const auto subsets = pcs::extract_detessellate_contract_subsets_from_mpk(contract);
  require(subsets.size() == 1U, "fixture should produce one detess subset");
  const std::string output_name = detess.output_tensors.front().name;
  const auto runtime =
      pcs::build_detessellate_runtime_config_from_subsets(subsets, {output_name}, {output_name});
  auto stage = make_stage(runtime, detess.name);
  stage.kernel_kind = detess.kernel;
  ps::SimaPluginStaticManifest manifest;
  manifest.model_id = contract.model_name;
  manifest.stages.push_back(std::move(stage));
  return manifest;
}

void lower_recipes(ps::SimaPluginStaticManifest& manifest,
                   const std::vector<ps::PipelineElementSpec>& elements = {}) {
  const auto failure = ps::prepare_kernel_stage_recipes(&manifest, std::nullopt, elements, {},
                                                        simaai::neat::NameTransform{});
  require(!failure, failure ? failure->field + ": " + failure->message : "");
  for (const auto& stage : manifest.stages)
    require(stage.native_kernel && !stage.native_kernel->members.empty(),
            "Core did not emit a canonical numerical recipe");
}

simaai::runtime::KernelInspection inspect_first(const ps::SimaPluginStaticManifest& manifest) {
  const auto checked = simaai::scheduler::inspect_kernel(
      manifest.stages.front().native_kernel->members.front().executable);
  const auto* value = std::get_if<simaai::runtime::KernelInspection>(&checked);
  require(value, "canonical checked executable inspection failed");
  return *value;
}

void check_metadata_encoding(const std::filesystem::path& root, const bool padded) {
  std::string error;
  const auto contract = ps::load_mpk_contract_from_pack_root(root.string(), &error);
  require(contract.has_value(), "MPK encoding fixture should load: " + error);
  require(contract->plugins[1].align_c16 == padded && !contract->plugins[1].cblock,
          "MPK padded/compact flags must remain distinct from CBlock16");
  const auto manifest = make_manifest(*contract);
  const auto& cfg = manifest.stages.front().processcvu;
  require(cfg.input_tensors.size() == 1U && cfg.output_tensors.size() == 1U,
          "metadata fixture must retain one input and one output descriptor");
  const auto& input = cfg.input_tensors.front();
  const auto& output = cfg.output_tensors.front();
  require(input.layout_kind == SIMA_EV_LAYOUT_TILED &&
              input.layout.tiled.flags == (padded ? SIMA_EV_TILED_FLAG_PADDED_HWC_C16
                                                  : SIMA_EV_TILED_FLAG_COMPACT_CHANNELS),
          "metadata projection must preserve the exact padded/compact encoding, not CBlock");
  // Dense descriptor storage is intentionally unspecified in this metadata
  // helper; the published logical span, not a fabricated descriptor extent,
  // carries the exact 396-byte output contract.
  require(input.storage.nbytes == (padded ? 576U : 396U) && output.storage.nbytes == 0U &&
              manifest.stages.front().logical_outputs.size() == 1U &&
              manifest.stages.front().logical_outputs.front().size_bytes == 396U &&
              input.dtype == SIMA_EV_DTYPE_BF16 && output.dtype == SIMA_EV_DTYPE_BF16,
          "metadata projection must retain exact MPK capacities and BF16 dtype");
  require(input.shape.rank == 3U && output.shape.rank == 3U,
          "metadata descriptors must retain the MPK tensor rank");
  for (std::size_t axis = 0; axis < 3U; ++axis) {
    constexpr std::array<std::uint32_t, 3U> shape{2U, 3U, 33U};
    require(input.shape.sizes[axis] == shape[axis] && output.shape.sizes[axis] == shape[axis],
            "metadata descriptors must retain the MPK logical shape");
  }
  require(!manifest.stages.front().native_kernel,
          "metadata-only fixtures must not be presented as prepared executables");
}

// Execute the same compiler-produced canonical recipe checked below. The test
// never constructs numerical requests or substitutes its layouts/placement.
void execute_compiler_contract(const ps::SimaPluginStaticManifest& manifest, bool bf16,
                               unsigned batches, ExecutionMode mode, bool blocked = true) {
  namespace dma = simaai::neat::internal::dmabuf;
  namespace tr = simaai::gst;
  namespace gs = simaai::runtime::gstreamer;
  require(gs::register_element(), "native kernel element registration failed");
  require(manifest.stages.size() == 1, "one compiled fixture stage required");
  const auto& authored = manifest.stages.front();
  const auto& recipe = *authored.native_kernel;
  simaai::runtime::gstreamer::TensorStagePublication publication;
  std::string publication_error;
  require(ps::build_kernel_stage_publication(authored, &publication, &publication_error),
          publication_error);
  const auto phase = [&](const char* name) {
    std::cerr << "Core EV fixture " << authored.element_name << " batch=" << batches
              << " phase=" << name << std::endl;
  };
  phase("construct pipeline");
  require(authored.processcvu.requested_run_target == "EV74" &&
              recipe.target == simaai::runtime::Hardware::EV74,
          "fixture must explicitly request the EV backend through Core policy");
  auto pipeline = std::shared_ptr<GstElement>(gst_pipeline_new("core-blocked-contract"),
                                              [](GstElement* element) {
                                                if (element) {
                                                  gst_element_set_state(element, GST_STATE_NULL);
                                                  gst_object_unref(element);
                                                }
                                              });
  auto* source = gst_element_factory_make("appsrc", "producer");
  auto* runner = gst_element_factory_make("neatkernelrunner", authored.element_name.c_str());
  auto* sink = gst_element_factory_make("appsink", "collector");
  require(pipeline && source && runner && sink, "fixture pipeline factories are unavailable");
  gst_bin_add_many(GST_BIN(pipeline.get()), source, runner, sink, nullptr);
  g_object_set(source, "caps", publication.input_caps.get(), "format", GST_FORMAT_TIME, nullptr);
  g_object_set(sink, "sync", FALSE, "async", FALSE, "enable-last-sample", FALSE, nullptr);
  require(gst_element_link(source, runner) && gst_element_link(runner, sink),
          "fixture links failed");
  std::string issue;
  phase("attach manifest");
  require(ps::attach_manifest_context(pipeline.get(), manifest, &issue), issue);

  // Independent HWC oracle for two spatial rows, three columns and 33 channels:
  // three C16 blocks, each containing six pixel records. BF16 stores separate
  // low/high 16-byte lanes in each record. No production packing helper is used.
  const std::size_t input_bytes = batches * (blocked ? 288U : 198U) * (bf16 ? 2U : 1U);
  std::vector<std::byte> packed(input_bytes, std::byte{0xa5});
  std::vector<float> expected(batches * 198U);
  for (unsigned batch = 0; batch < batches; ++batch)
    for (unsigned pixel = 0; pixel < 6; ++pixel)
      for (unsigned channel = 0; channel < 33; ++channel) {
        const auto logical = (batch * 6U + pixel) * 33U + channel;
        const auto record = (channel / 16U) * 6U + pixel;
        const auto lane = channel % 16U;
        if (bf16) {
          // Exact BF16 finite values vary in both byte planes.
          const auto bits = std::uint16_t(0x3e00U + logical * 13U);
          const auto at = blocked ? batch * 576U + record * 32U + lane : logical * 2U;
          packed[at] = std::byte(bits & 255U);
          packed[at + (blocked ? 16U : 1U)] = std::byte(bits >> 8U);
          expected[logical] = std::bit_cast<float>(std::uint32_t(bits) << 16U);
        } else {
          const auto code = std::bit_cast<std::int8_t>(std::uint8_t(logical * 73U));
          packed[batch * 288U + record * 16U + lane] = std::byte(std::uint8_t(code));
          // Compiler q_scale is a public multiplier: dequant lowers its reciprocal.
          expected[logical] = (float(code) + 7.0F) * 4.0F;
        }
      }
  require(authored.physical_inputs.size() == 1, "one physical fixture input required");
  const auto& input_contract = authored.physical_inputs.front();
  require(input_contract.source_byte_offset >= 0 && input_contract.size_bytes >= input_bytes,
          "compiled input carrier does not cover the numerical payload");
  const bool uses_binding =
      input_contract.address_source == ps::PhysicalAddressSource::RuntimePhysicalBinding;
  // A physical-command input refers to its upstream carrier binding, not to
  // GstMemory directly. Model that real producer contract, including a nonzero
  // aligned carrier base; never rewrite the compiler's address-source policy.
  const std::size_t carrier_base = uses_binding ? 4096 : 0;
  const auto carrier_extent =
      std::size_t(input_contract.source_byte_offset) + input_contract.size_bytes;
  const auto input_offset = carrier_base + std::size_t(input_contract.source_byte_offset);
  const auto allocation_bytes = std::max<std::size_t>(
      4096, std::max(authored.frame_arena_size_bytes, carrier_base + carrier_extent + 64));
  std::optional<tr::TensorBufferPreparedMetaTemplate> input_publication;
  if (uses_binding) {
    require(input_contract.physical_index == 0 && authored.logical_inputs.size() == 1,
            "fixture expects one producer physical/logical carrier");
    const auto& logical = authored.logical_inputs.front();
    const auto expected_dtype = bf16 ? "BF16" : "INT8";
    require(logical.dtype == expected_dtype &&
                ps::tensorsemantics::normalize_layout_token(logical.layout) == "HWC",
            "fixture producer must retain its independently authored dtype and channel order");
    tr::TensorBufferPublishContract producer;
    producer.stage_key = "compiler-fixture-input";
    producer.physical_outputs.push_back({0, carrier_extent, input_contract.segment_name});
    tr::TensorBufferPublishLogicalOutput tensor;
    tensor.logical_index = logical.logical_index;
    tensor.backend_output_index = logical.backend_input_index;
    tensor.physical_index = logical.physical_index;
    tensor.memory_index = 0;
    tensor.route_slot = 0;
    tensor.logical_name = logical.logical_name;
    tensor.backend_name = logical.backend_name;
    tensor.segment_name = logical.segment_name;
    tensor.byte_offset = logical.byte_offset;
    tensor.size_bytes = logical.size_bytes;
    tensor.dtype = bf16 ? SIMA_TENSOR_SET_DTYPE_BF16_V1 : SIMA_TENSOR_SET_DTYPE_INT8_V1;
    // Tensor-set metadata expresses channel order separately from full shape;
    // explicit NHWC keeps N in tensor.shape, not in a second layout enum.
    tensor.layout = SIMA_TENSOR_SET_LAYOUT_HWC_V1;
    tensor.shape = logical.shape;
    tensor.stride_bytes = logical.stride_bytes;
    producer.logical_outputs.push_back(std::move(tensor));
    tr::TensorBufferPublishContract projected;
    std::vector<tr::TensorBufferPhysicalBinding> physical_bindings;
    require(tr::tensor_buffer_project_single_memory_arena_with_physical_bindings(
                producer, {{carrier_base, carrier_extent}}, allocation_bytes, "fixture-arena",
                &projected, &physical_bindings, &issue),
            issue);
    input_publication.emplace();
    require(tr::tensor_buffer_prepare_meta_template_from_contract(projected, physical_bindings,
                                                                  &*input_publication, &issue),
            issue);
  }
  const auto attach_input_publication = [&](GstBuffer* buffer) {
    if (!input_publication)
      return;
    require(tr::tensor_buffer_attach_prepared_meta(buffer, *input_publication, &issue), issue);
    tr::TensorBufferPhysicalBinding binding;
    require(tr::tensor_buffer_physical_binding_at(buffer, 0, &binding, &issue), issue);
    require(binding.memory_index == 0 && binding.byte_offset == carrier_base &&
                binding.size_bytes == carrier_extent,
            "producer physical binding differs from compiler input carrier");
  };
  if (mode == ExecutionMode::Preflight) {
    // Exercise the exact publisher on ordinary memory as well as topology;
    // metadata prerequisites must fail here, not during a firmware trial.
    gs::Buffer host(gst_buffer_new_allocate(nullptr, allocation_bytes, nullptr));
    require(bool(host), "cold fixture carrier allocation failed");
    attach_input_publication(host.get());
    const auto rejected = ps::attach_kernel_runner_stages(pipeline.get(), {});
    require(rejected && rejected->field == "kernel-runner.element" &&
                rejected->message.find("no prepared stage") != std::string::npos,
            "Core candidate preflight must reject an unbound numerical element");
    phase("cold preflight passed");
    return;
  }
  phase("prepare execution");
  const std::array requests{ps::KernelRunnerStageRequest{
      authored.element_name, authored.logical_stage_id, 1, 2, {}, &authored}};
  const auto attach = ps::prepare_and_attach_kernel_runners(pipeline.get(), requests);
  require(!attach, attach ? attach->field + ": " + attach->message : "");
  auto context = std::shared_ptr<GstContext>(gst_element_get_context(runner, gs::context_type),
                                             gst_context_unref);
  const auto execution = gs::stage_from_context(context.get());
  require(execution && !execution->members.empty(), "Core attachment did not prepare execution");
  for (const auto& member : execution->members)
    require(member.operation && member.operation->hardware() == simaai::runtime::Hardware::EV74,
            "fixture must qualify actual EV74 execution, not an A65 fallback");

  phase("allocate input");
  dma::Error error;
  gs::Buffer input(dma::allocateDmaBufBuffer(dma::HeapKind::Cma, allocation_bytes, {}, &error));
  require(bool(input), error.message());
  auto input_view = dma::DmaBufView::fromGstMemory(gst_buffer_peek_memory(input.get(), 0), 0,
                                                   allocation_bytes, &error);
  auto mapped = input_view ? input_view->map(dma::CpuAccess::Write, &error) : std::nullopt;
  require(bool(mapped), error.message());
  std::vector<std::byte> initial(allocation_bytes, std::byte{0xd3});
  std::copy(packed.begin(), packed.end(), initial.begin() + input_offset);
  std::memcpy(mapped->data(), initial.data(), initial.size());
  require(mapped->finish(&error), error.message());
  mapped.reset();
  auto* meta = gst_buffer_add_custom_meta(input.get(), "GstSimaMeta");
  require(meta, "fixture correlation metadata is unavailable");
  gst_structure_set(gst_custom_meta_get_structure(meta), "frame-id", G_TYPE_INT64, gint64(41),
                    "timestamp", G_TYPE_UINT64, guint64(812), "stream-id", G_TYPE_STRING,
                    "core-blocked", nullptr);
  attach_input_publication(input.get());
  GST_BUFFER_PTS(input.get()) = 1234;
  phase("start pipeline");
  require(gst_element_set_state(pipeline.get(), GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "fixture pipeline failed to start");
  phase("submit");
  if (gst_app_src_push_buffer(GST_APP_SRC(source), input.release()) != GST_FLOW_OK) {
    std::cerr << "UNRETIRED: fixture submission acceptance uncertain; guarded recovery required\n"
              << std::flush;
    for (;;)
      std::this_thread::sleep_for(std::chrono::hours(24));
  }
  auto output = std::shared_ptr<GstSample>(
      gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 10 * GST_SECOND), gst_sample_unref);
  if (!output) {
    auto bus = std::shared_ptr<GstBus>(gst_element_get_bus(pipeline.get()), gst_object_unref);
    while (auto* message = gst_bus_pop_filtered(bus.get(), GST_MESSAGE_ERROR)) {
      GError* failure = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(message, &failure, &debug);
      std::cerr << "fixture device error: " << (failure ? failure->message : "unknown") << " "
                << (debug ? debug : "") << '\n';
      g_clear_error(&failure);
      g_free(debug);
      gst_message_unref(message);
    }
    // Keep the process and its DMA owners alive until the outer guard has
    // stopped EV. Exiting here could release buffers while hardware uses them.
    std::cerr << "UNRETIRED: Core-produced EV contract did not retire\n" << std::flush;
    for (;;)
      std::this_thread::sleep_for(std::chrono::hours(24));
  }
  phase("retired output");
  auto* buffer = gst_sample_get_buffer(output.get());
  tr::TensorBufferView tensors;
  require(tr::tensor_buffer_create_view(buffer, nullptr, &tensors, &issue), issue);
  require(tensors.tensors.size() == 1, "compiled output publication arity changed");
  const auto& tensor = tensors.tensors.front();
  require(tensor.byte_offset >= 0 && tensor.size_bytes == expected.size() * sizeof(float),
          "compiled output publication extent changed");
  auto* memory = gst_buffer_peek_memory(buffer, tensor.memory_index);
  auto result_view = dma::DmaBufView::fromGstMemory(memory, std::size_t(tensor.byte_offset),
                                                    tensor.size_bytes, &error);
  auto result = result_view ? result_view->map(dma::CpuAccess::Read, &error) : std::nullopt;
  require(bool(result), error.message());
  require(std::memcmp(result->data(), expected.data(), tensor.size_bytes) == 0,
          "Core-produced blocked contract differs from independent HWC oracle");
  require(result->finish(&error), error.message());
  result.reset();
  require(GST_BUFFER_PTS(buffer) == 1234, "output correlation timestamp changed");
  auto* output_meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  gint64 output_frame = -1;
  guint64 output_timestamp = 0;
  require(output_meta &&
              gst_structure_get_int64(gst_custom_meta_get_structure(output_meta), "frame-id",
                                      &output_frame) &&
              output_frame == 41 &&
              gst_structure_get_uint64(gst_custom_meta_get_structure(output_meta), "timestamp",
                                       &output_timestamp) &&
              output_timestamp == 812,
          "Core correlation metadata changed across numerical execution");
  phase("stop retired pipeline");
  gst_app_src_end_of_stream(GST_APP_SRC(source));
  require(gst_element_set_state(pipeline.get(), GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
          "retired fixture did not stop");
  mapped = input_view->map(dma::CpuAccess::Read, &error);
  require(bool(mapped), error.message());
  require(std::memcmp(mapped->data(), initial.data(), initial.size()) == 0,
          "kernel modified input or surrounding guards");
  require(mapped->finish(&error), error.message());
  std::cout << "PASS: Core compiler -> plugin -> Runner -> EV74 "
            << (!blocked ? "BF16 dense Cast"
                : bf16   ? "BF16 blocked DetessCast"
                         : "INT8 CBlock DetessDequant")
            << " batch=" << batches << " values=" << expected.size() << '\n';
}

ps::SimaPluginStaticManifest make_detessdequant_cblock_manifest(unsigned batches = 1) {
  const auto batch = static_cast<std::int64_t>(batches);
  const auto input_bytes = std::size_t(batches) * 288U;
  const auto dense_bytes = std::size_t(batches) * 198U;
  const auto output_bytes = dense_bytes * 4U;
  pcs::DetessDequantHeadContractSubset head;
  head.per_head_input_shape = {batch, 2, 3, 33};
  head.input_transport_shape = {batch, 2, 3, 48};
  head.input_transport_size_bytes = input_bytes;
  head.per_head_quant_params.scales = {0.25};
  head.per_head_quant_params.zero_points = {-7};
  head.frame_shape = {batch, 2, 3, 33};
  head.frame_type = "INT8";
  head.slice_shape = {2, 3, 33};
  head.align_c16 = true;
  head.cblock = true;
  head.output_dtype = "FP32";
  const auto contract = pcs::build_detessdequant_tensor_descriptor_contract(head, head.frame_shape);
  require(contract.input.storage.nbytes == 288U && contract.batch_size == int(batches),
          "authoritative DetessDequant factory must preserve CBlock physical capacity: batch=" +
              std::to_string(batches) + " expected=" + std::to_string(input_bytes) +
              " actual=" + std::to_string(contract.input.storage.nbytes));
  // Exercise the actual physical compiler boundary. The subset helper above
  // preserves compiler layout facts, but does not select native dispatch or
  // its final descriptor alignment; those belong to command projection.
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const sc::TensorShape shape{batch, 2, 3, 33};
  const sc::TensorShape tile{2, 3, 33};
  data.carriers = {
      {0U, input_bytes, 64U, sc::ValueRepresentation::Tessellated},
      {1U, dense_bytes, 64U, sc::ValueRepresentation::Dense},
      {2U, output_bytes, 64U, sc::ValueRepresentation::Dense},
  };
  const std::array<std::uint64_t, 3U> spans{input_bytes, dense_bytes, output_bytes};
  for (std::uint32_t index = 0; index < spans.size(); ++index) {
    sc::ValueSpec value;
    value.id = index;
    value.name = index == 0U ? "blocked_input" : index == 1U ? "detess_output" : "dequant_output";
    value.required_bytes = index == 2U ? output_bytes : dense_bytes;
    value.logical_dtype = index == 2U ? "FP32" : "INT8";
    value.logical_shape = shape;
    value.logical_layout = "NHWC";
    value.representation =
        index == 0U ? sc::ValueRepresentation::Tessellated : sc::ValueRepresentation::Dense;
    value.storage_binding = sc::StorageBinding{
        index == 0U ? sc::StorageBindingKind::External : sc::StorageBindingKind::Root,
        index,
        0U,
        spans[index],
        {},
        index == 0U ? sc::StorageAccess::ReadOnly : sc::StorageAccess::ReadWrite,
        std::nullopt};
    data.values.push_back(std::move(value));
  }
  data.model_inputs = {0U};
  data.model_outputs = {{0U, "dequant_output", 2U}};
  for (std::uint32_t index = 0; index < 2U; ++index) {
    sc::OpSpec op;
    op.id = index;
    op.sequence = index + 1U;
    op.name = index == 0U ? "detess" : "dequant";
    op.kind = index == 0U ? sc::OpKind::Detessellate : sc::OpKind::Dequantize;
    op.processor = "EV74";
    op.inputs = {index};
    op.outputs = {index + 1U};
    op.input_shapes = {shape};
    op.output_shapes = {shape};
    op.config = index == 0U
                    ? sc::OpConfig{sc::DetessellateOpConfig{tile, shape, true, true, "INT8"}}
                    : sc::OpConfig{sc::DequantizeOpConfig{"INT8", {{0.25, -7}}}};
    data.ops.push_back(std::move(op));
  }
  std::string error;
  const auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "CBlock semantic plan must validate: " + error);
  const auto physical = sc::PhysicalExecutionLowerer::lower(*plan, &error);
  require(physical.has_value(), "CBlock physical plan must lower: " + error);
  std::vector<sc::PhysicalCommandId> commands;
  for (const auto& command : physical->commands) {
    if (command.engine == sc::PhysicalEngine::Cvu)
      commands.push_back(command.id);
  }
  require(commands.size() == 1U, "CBlock semantic chain must fuse into one command");
  const auto arena =
      sc::FrameSlotArenaPlan::compile(*plan, *physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
                                      sc::kLegacyEvoCmaRegionAlignmentBytes, &error);
  require(arena.has_value(), "CBlock frame arena must compile: " + error);
  const auto compiled =
      sc::build_dmabuf_plan_processcvu_command_contract(*plan, *physical, commands, *arena, &error);
  require(compiled.has_value(), "CBlock native command contract must compile: " + error);
  simaai::neat::CompiledNodeContract node;
  node.element_name = node.logical_stage_id = "detessdequant_cblock";
  node.processcvu = *compiled;
  node.renderable = true;
  simaai::neat::CompiledPipelineContracts pipeline;
  pipeline.stages.push_back(std::move(node));
  pipeline.fully_renderable = true;
  ps::ManifestBuildDiagnostics diagnostics;
  simaai::neat::ContractCompileInput compile_input;
  compile_input.processcvu_requested_run_target = "EV74";
  auto manifest =
      simaai::neat::render_manifest_from_compiled_contracts(pipeline, compile_input, &diagnostics);
  require(manifest.has_value(), "CBlock production manifest rendering must succeed");
  return std::move(*manifest);
}

void reject_malformed_canonical_alignment() {
  auto valid = make_detessdequant_cblock_manifest();
  // Keep an unprepared copy of the production-projected contract. Neither its
  // operation identity nor its physical capacities are manufactured by this test.
  auto malformed = valid;
  ensure_gst_ready();
  lower_recipes(valid);
  require(valid.stages.front().native_kernel->members.size() == 1U,
          "unaltered fused fixture must prepare before testing malformed alignment");
  malformed.stages.front().processcvu.input_tensors.front().layout.tiled.tile_align_bytes = 0;
  const auto failure = ps::prepare_kernel_stage_recipes(&malformed, std::nullopt, {}, {},
                                                        simaai::neat::NameTransform{});
  require(failure && failure->code == simaai::runtime::ErrorCode::InvalidRequest &&
              failure->field == "inputs[0].tile_align_bytes",
          "malformed canonical input must report its exact alignment field: " +
              (failure ? failure->field + ": " + failure->message : "unexpected admission"));
  require(!malformed.stages.front().native_kernel,
          "failed admission must not publish a partial canonical recipe");
}

simaai::kernels::Layout
authoritative_detessdequant_cblock_layout(ExecutionMode mode = ExecutionMode::None, unsigned batches = 1) {
  const auto input_bytes = std::size_t(batches) * 288U;
  const auto output_bytes = std::size_t(batches) * 792U;
  auto manifest = make_detessdequant_cblock_manifest(batches);
  ensure_gst_ready();
  lower_recipes(manifest);
  const auto& authored = manifest.stages.front();
  const auto& recipe = *authored.native_kernel;
  const auto inspection = inspect_first(manifest);
  require(authored.processcvu.graph_name == "detessdequant" && inspection.inputs().size() == 1U &&
              inspection.outputs().size() == 1U,
          "physical compilation must author one canonical DetessDequant operation");
  require(inspection.input_requirements().front().storage_bytes == input_bytes &&
              inspection.output_requirements().front().storage_bytes == output_bytes,
          "canonical CBlock must preserve physical input and dense output capacities");
  require(authored.processcvu.q_scale_list == std::vector<double>{0.25} &&
              authored.processcvu.q_zp_list == std::vector<int>{-7},
          "Core physical-command qparams changed before canonical lowering");
  require(recipe.members.size() == 1U && recipe.members.front().inputs.front().byte_offset == 0U &&
              recipe.members.front().outputs.front().byte_offset == 0U,
          "CBlock must create one checked member with exact offsets without device access");
  if (mode != ExecutionMode::None)
    execute_compiler_contract(manifest, false, batches, mode);
  return inspection.inputs().front().layout;
}

void verify_tiled_explicit_batch_axes() {
  struct Case {
    const char* layout;
    std::vector<int> shape;
    std::vector<int> tile;
    std::array<std::uint8_t, 4> axes;
  };
  const std::array<Case, 3> cases{{
      {"NHWC", {2, 4, 5, 3}, {1, 2, 5, 3},
       {SIMA_EV_AXIS_N, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W, SIMA_EV_AXIS_C}},
      {"NCHW", {2, 3, 4, 5}, {1, 3, 2, 5},
       {SIMA_EV_AXIS_N, SIMA_EV_AXIS_C, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W}},
      // Do not invent N for an ambiguous four-axis HWC declaration. This
      // retains the established D/H/W/C interpretation, not a rank heuristic.
      {"HWC", {2, 4, 5, 3}, {1, 2, 5, 3},
       {SIMA_EV_AXIS_D, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W, SIMA_EV_AXIS_C}},
  }};
  for (const auto& test : cases) {
    sima_ev_tensor_desc descriptor{};
    std::string error;
    require(ps::tensorsemantics::build_tiled_tensor_desc(
                test.shape, test.tile, "INT8", test.layout, 16U, &descriptor, &error,
                "missing descriptor", "invalid rank", "invalid dimension", "invalid dtype",
                "tile rank mismatch", "invalid tile dimension"),
            std::string(test.layout) + ": " + error);
    require(descriptor.shape.rank == 4U, "explicit-layout test lost tensor rank");
    for (std::size_t axis = 0; axis < test.axes.size(); ++axis) {
      require(descriptor.shape.axis_semantics[axis] == test.axes[axis] &&
                  descriptor.shape.sizes[axis] == test.shape[axis] &&
                  descriptor.layout.tiled.tile_sizes[axis] == test.tile[axis],
              std::string(test.layout) + " must retain exact authored axes, shape and tiles");
    }
    sima_ev_tensor_desc dense{};
    const bool dense_built = ps::tensorsemantics::build_dense_tensor_desc(
        test.shape, "INT8", test.layout, &dense, &error, "missing descriptor", "invalid rank",
        "invalid dimension", "invalid dtype", "missing strides");
    require(dense_built, std::string(test.layout) + ": " + error);
    for (std::size_t axis = 0; axis < test.axes.size(); ++axis)
      require(dense.shape.axis_semantics[axis] == test.axes[axis],
              "dense fixture must retain the same explicit semantic axes");

    // The production runtime adapter admits explicit descriptors; it does not
    // reconstruct them from layout/shape strings. Preserve that authority and
    // its fail-closed missing-descriptor contract, including full N=2 axes.
    pss::CompiledProcessCvuRuntimeConfig config;
    config.graph_family = "detess";
    config.graph_name = "detessellate";
    config.default_input_name = "input";
    config.runtime_input_names = {"input"};
    config.runtime_output_names = config.published_output_names = {"output"};
    config.primary_output_name = "output";
    config.input_dtype = config.output_dtype = "INT8";
    config.input_shapes = config.output_shapes = {test.shape};
    config.slice_shapes = {test.tile};
    config.input_tensors = {descriptor};
    config.output_tensors = {dense};
    const auto adapted = pss::build_processcvu_payload_from_runtime_config_internal(config);
    require(adapted.input_tensors.size() == 1U && adapted.output_tensors.size() == 1U &&
                std::memcmp(&adapted.input_tensors.front(), &descriptor, sizeof(descriptor)) == 0 &&
                std::memcmp(&adapted.output_tensors.front(), &dense, sizeof(dense)) == 0,
            "runtime adapter must preserve complete explicit dense and tiled descriptors");
    config.input_tensors.clear();
    config.output_tensors.clear();
    bool rejected_missing_descriptors = false;
    try {
      (void)pss::build_processcvu_payload_from_runtime_config_internal(config);
    } catch (const std::invalid_argument& failure) {
      rejected_missing_descriptors =
          std::string(failure.what()).find("requires explicit typed input_tensors/output_tensors") !=
          std::string::npos;
    }
    require(rejected_missing_descriptors,
            "runtime adapter must not infer missing typed descriptors from shape strings");
  }
}

void authoritative_bf16_contract(ExecutionMode mode = ExecutionMode::None, bool blocked = true,
                                 unsigned batches = 1) {
  // The fixture authors exact semantic values/storage, not bridge flags or a
  // legacy MPK reconstruction. Production physical lowering must fuse DetessCast
  // and derive the blocked-BF16 reader without an exposed intermediate buffer.
  const std::vector<std::int64_t> logical{batches, 2, 3, 33};
  const auto input_bytes = batches * (blocked ? 576U : 396U);
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const auto add_value = [&](const std::string& name, const std::string& dtype,
                             std::uint64_t bytes, bool tiled) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    sc::ValueSpec value;
    value.id = id;
    value.name = name;
    value.required_bytes = bytes;
    value.logical_dtype = dtype;
    value.logical_shape = logical;
    value.logical_layout = "NHWC";
    value.representation = tiled ? sc::ValueRepresentation::Tessellated
                                 : sc::ValueRepresentation::Dense;
    value.storage_binding = sc::StorageBinding{
        id == 0U ? sc::StorageBindingKind::External : sc::StorageBindingKind::Root,
        id, 0U, bytes, {},
        id == 0U ? sc::StorageAccess::ReadOnly : sc::StorageAccess::ReadWrite, std::nullopt};
    data.carriers.push_back({id, bytes, 4096U, value.representation});
    data.values.push_back(std::move(value));
    return id;
  };
  const auto input = add_value("ofm0", "BF16", input_bytes, blocked);
  data.model_inputs = {input};
  auto scalar_input = input;
  if (blocked) {
    scalar_input = add_value("detess_private", "BF16", batches * 396U, false);
    sc::OpSpec detess;
    detess.id = 0U;
    detess.sequence = 1U;
    detess.name = "detess";
    detess.kind = sc::OpKind::Detessellate;
    detess.processor = "EV74";
    detess.inputs = {input};
    detess.outputs = {scalar_input};
    detess.input_shapes = detess.output_shapes = {logical};
    detess.config = sc::DetessellateOpConfig{{2, 3, 33}, logical, true, true, "BF16"};
    data.ops.push_back(std::move(detess));
  }
  const auto output_id = add_value("head0_fp32", "FP32", batches * 792U, false);
  sc::OpSpec cast;
  cast.id = static_cast<sc::OpId>(data.ops.size());
  cast.sequence = data.ops.size() + 1U;
  cast.name = "cast";
  cast.kind = sc::OpKind::Cast;
  cast.processor = "EV74";
  cast.inputs = {scalar_input};
  cast.outputs = {output_id};
  cast.input_shapes = cast.output_shapes = {logical};
  cast.config = sc::CastOpConfig{"FP32"};
  data.ops.push_back(std::move(cast));
  data.model_outputs = {{0U, "head0_fp32", output_id}};
  const auto fixture =
      sima_test::compile_processcvu_fixture(std::move(data), blocked ? "detesscast" : "cast");
  const auto& compiled = fixture.contract;
  require(fixture.physical.commands.front().batch_size == batches,
          "typed physical cohort must retain the authored semantic batch");
  require(compiled.payload.input_tensors.size() == 1U &&
              compiled.payload.output_tensors.size() == 1U,
          "production BF16 lowering must expose only the exact outer boundaries");
  if (blocked) {
    require(fixture.physical.commands.size() == 1U &&
                fixture.physical.commands.front().members.front().semantic_chain.size() == 2U &&
                !fixture.arena.region(scalar_input),
            "DetessCast must remain fused without materializing its private intermediate");
  }
  if (blocked)
    require((compiled.payload.opt_flags & (1U << 6)) != 0U,
            "MPK blocked BF16 must select the compiler lane-split reader");
  simaai::neat::CompiledNodeContract node;
  node.element_name = node.logical_stage_id = "detesscast_cblock";
  node.processcvu = compiled;
  node.renderable = true;
  simaai::neat::CompiledPipelineContracts pipeline;
  pipeline.stages.push_back(std::move(node));
  pipeline.fully_renderable = true;
  ps::ManifestBuildDiagnostics diagnostics;
  simaai::neat::ContractCompileInput compile_input;
  compile_input.processcvu_requested_run_target = "EV74";
  auto rendered =
      simaai::neat::render_manifest_from_compiled_contracts(pipeline, compile_input, &diagnostics);
  require(rendered.has_value(), "BF16 production manifest rendering must succeed");
  auto& manifest = *rendered;
  ensure_gst_ready();
  std::string error;
  lower_recipes(manifest);
  const auto& authored = manifest.stages.front();
  const auto& recipe = *authored.native_kernel;
  const auto inspection = inspect_first(manifest);
  simaai::runtime::gstreamer::TensorStagePublication publication;
  require(ps::build_kernel_stage_publication(authored, &publication, &error), error);
  if (!blocked) {
    // The typed cohort records N=2. The canonical executable must still cover
    // that complete tensor once, not multiply N into a second repetition.
    require(authored.processcvu.graph_name == "cast" && authored.processcvu.batch_size == 1 &&
                recipe.members.size() == 1U,
            "Core must retain the full MPK Cast tensor without external batch repetition");
    require(inspection.inputs().size() == 1U && inspection.outputs().size() == 1U &&
                inspection.input_requirements().front().storage_bytes == input_bytes &&
                inspection.output_requirements().front().storage_bytes == batches * 792U,
            "canonical Cast must cover every input and output element in both images");
    const auto check_dense_tensor = [&](const auto& spec, auto dtype, std::int64_t width) {
      require(spec.dtype == dtype && spec.shape.size() == logical.size(),
              "canonical Cast changed the full tensor dtype or rank");
      const auto* strides = std::get_if<simaai::kernels::Strided>(&spec.layout);
      require(strides || std::holds_alternative<simaai::kernels::Contiguous>(spec.layout),
              "canonical Cast must preserve dense tensor storage");
      std::int64_t stride = width;
      for (std::size_t axis = logical.size(); axis-- > 0;) {
        require(spec.shape.dimensions()[axis] == logical[axis] &&
                    (!strides || strides->strides_bytes[axis] == stride),
                "canonical Cast changed a full-N tensor dimension or stride");
        stride *= logical[axis];
      }
    };
    check_dense_tensor(inspection.inputs().front(), simaai::runtime::DType::BFloat16, 2);
    check_dense_tensor(inspection.outputs().front(), simaai::runtime::DType::Float32, 4);
    const auto& member = recipe.members.front();
    require(member.inputs.size() == 1U && member.outputs.size() == 1U &&
                member.inputs.front().byte_offset == 0U &&
                member.outputs.front().byte_offset == 0U && member.inputs.front().tensor == 0U &&
                member.outputs.front().tensor == 0U,
            "full-tensor Cast placement must cover its exact input and output from offset zero");
    const auto& outputs = publication.contract->logical_outputs;
    require(outputs.size() == 1 && outputs.front().size_bytes == batches * 792U &&
                outputs.front().shape == logical && outputs.front().byte_offset == 0 &&
                outputs.front().dtype == SIMA_TENSOR_SET_DTYPE_FP32_V1 &&
                outputs.front().backend_output_index == 0,
            "Core Cast publication must retain the complete typed N=2 tensor");
    const std::vector<std::int64_t> output_strides{792, 396, 132, 4};
    require(outputs.front().stride_bytes.empty() || outputs.front().stride_bytes == output_strides,
            "Core Cast publication must not stride across or omit an image");
    require(publication.contract->physical_outputs.size() == 1U &&
                publication.contract->physical_outputs.front().size_bytes == batches * 792U,
            "Core Cast physical publication must reserve the complete output span");
    if (mode != ExecutionMode::None)
      execute_compiler_contract(manifest, true, batches, mode, false);
    return;
  }
  require(authored.processcvu.graph_name == "detesscast" &&
              (authored.processcvu.opt_flags & (1U << 6)) != 0U &&
              inspection.inputs().size() == 1U && inspection.outputs().size() == 1U,
          "Core BF16 must retain compiler lane-split authority and one tensor pair");
  require(std::holds_alternative<simaai::kernels::MlaCBlocked>(inspection.inputs().front().layout),
          "canonical BF16 input must retain lane-split CBlocked storage, not a host repack");
  const auto& output = inspection.outputs().front();
  require(inspection.input_requirements().front().storage_bytes == batches * 576U &&
              inspection.output_requirements().front().storage_bytes == batches * 792U &&
              output.dtype == simaai::runtime::DType::Float32 && output.shape.size() == 4U,
          "canonical BF16 must preserve every 576-byte input and 792-byte dense FP32 output batch");
  const std::array<std::int64_t, 4> dense_strides{792, 396, 132, 4};
  const auto* strides = std::get_if<simaai::kernels::Strided>(&output.layout);
  require(strides || std::holds_alternative<simaai::kernels::Contiguous>(output.layout),
          "canonical output must retain dense storage");
  for (std::size_t axis = 0; axis < logical.size(); ++axis) {
    require(output.shape.dimensions()[axis] == logical[axis] &&
                (!strides || strides->strides_bytes[axis] == dense_strides[axis]),
            "canonical BF16 output must retain exact dense FP32 shape and strides");
  }
  require(publication.contract->physical_outputs.size() == 1U &&
              publication.contract->physical_outputs.front().size_bytes == batches * 792U,
          "BF16 publication must reserve exactly 792 dense FP32 bytes per batch");
  require(recipe.members.size() == 1U, "BF16 must create exactly one checked native transform");
  if (mode != ExecutionMode::None)
    execute_compiler_contract(manifest, true, batches, mode);
}

} // namespace

int main(int argc, char** argv) {
  const bool device = argc == 2 && std::strcmp(argv[1], "--ev74") == 0;
  const bool preflight = argc == 2 && std::strcmp(argv[1], "--preflight-only") == 0;
  const auto mode = device      ? ExecutionMode::EV74
                    : preflight ? ExecutionMode::Preflight
                                : ExecutionMode::None;
  return sima_test::run_test("unit_processcvu_v2_mpk_encoding_test", [&] {
    require(argc == 1 || device || preflight,
            "usage: unit_processcvu_v2_mpk_encoding_test [--ev74|--preflight-only]");
    ps::GraphTiledChannelEncoding encoding;
    std::string encoding_error;
    require(!ps::resolve_graph_processcvu_tiled_channel_encoding("detessellate", true, true,
                                                                 &encoding, &encoding_error) &&
                encoding_error.find("supported only for detessdequant") != std::string::npos,
            "unsupported model CBlock family must retain an actionable diagnostic");
    require(ps::resolve_graph_processcvu_tiled_channel_encoding("detessdequant", true, true,
                                                                &encoding, &encoding_error) &&
                encoding == ps::GraphTiledChannelEncoding::CBlock16,
            "model CBlock storage must remain distinct from padded HWC");
    require(ps::resolve_graph_processcvu_tiled_channel_encoding("detessdequant", true, false,
                                                                &encoding, &encoding_error) &&
                encoding == ps::GraphTiledChannelEncoding::PaddedHwcC16,
            "model align_c16 without cblock must select padded HWC");
    require(ps::resolve_graph_processcvu_tiled_channel_encoding("detessdequant", false, false,
                                                                &encoding, &encoding_error) &&
                encoding == ps::GraphTiledChannelEncoding::CompactPixelMajor,
            "model compact storage must remain explicit");
    require(!ps::resolve_graph_processcvu_tiled_channel_encoding("detessdequant", false, false,
                                                                 nullptr, &encoding_error),
            "model encoding query requires an output destination");

    verify_tiled_explicit_batch_axes();
    authoritative_bf16_contract(mode);
    authoritative_bf16_contract(mode, true, 2);
    authoritative_bf16_contract(mode, false, 2);
    const auto cblock_n2 = authoritative_detessdequant_cblock_layout(mode, 2);
    require(std::holds_alternative<simaai::kernels::MlaCBlocked>(cblock_n2),
            "N2 DetessDequant must preserve its full CBlock tensor in one executable");
    const auto cblock = authoritative_detessdequant_cblock_layout(mode);
    require(std::holds_alternative<simaai::kernels::MlaCBlocked>(cblock),
            "authoritative DetessDequant must retain canonical CBlock16");
    reject_malformed_canonical_alignment();
    check_metadata_encoding(write_mpk_fixture("padded_hwc", true, false, 576U), true);
    check_metadata_encoding(write_mpk_fixture("compact", false, false, 396U), false);
  });
}

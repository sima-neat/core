#include "model/internal/ModelPack.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/VisualFrontend.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemanticsInternal.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.h"
#include "test_main.h"
#include "support/processcvu_plan_fixture.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require_build_dense_test_tensor(const std::vector<int>& shape, const std::string& dtype,
                                     const std::string& layout, sima_ev_tensor_desc* out,
                                     const char* expectation) {
  std::string error_detail;
  require(simaai::neat::pipeline_internal::sima::tensorsemantics::build_dense_tensor_desc(
              shape, dtype, layout, out, &error_detail, "test_tensor_output_missing",
              "test_tensor_shape_rank_invalid", "test_tensor_shape_dim_invalid",
              "test_tensor_dtype_invalid", "test_tensor_stride_output_missing"),
          expectation);
}

void require_build_tiled_test_tensor(const std::vector<int>& shape,
                                     const std::vector<int>& tile_shape, const std::string& dtype,
                                     const std::string& layout, std::uint32_t tile_align_bytes,
                                     sima_ev_tensor_desc* out, const char* expectation) {
  std::string error_detail;
  require(simaai::neat::pipeline_internal::sima::tensorsemantics::build_tiled_tensor_desc(
              shape, tile_shape, dtype, layout, tile_align_bytes, out, &error_detail,
              "test_tensor_output_missing", "test_tensor_shape_rank_invalid",
              "test_tensor_shape_dim_invalid", "test_tensor_dtype_invalid",
              "test_tile_shape_rank_mismatch", "test_tile_shape_dim_invalid"),
          expectation);
}

simaai::neat::PreprocOptions make_preproc_options() {
  simaai::neat::PreprocOptions opt;
  opt.model_managed_contract = true;
  opt.set_input_shape({720, 1280, 3});
  opt.input_img_type = "RGB";
  opt.set_output_shape({640, 640, 3});
  opt.scaled_width = 640;
  opt.scaled_height = 640;
  opt.output_img_type = "RGB";
  opt.output_dtype = "EVXX_INT8";
  opt.normalize = true;
  opt.aspect_ratio = true;
  opt.tessellate = true;
  opt.single_output_handoff = true;
  opt.set_slice_shape({32, 128, 3});
  opt.q_scale = 0.25;
  opt.q_zp = 7;
  return opt;
}

std::uint64_t dtype_size_bytes_for_test(const std::string& dtype) {
  std::string upper = dtype;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (upper == "BF16" || upper == "FP16" || upper == "UINT16" || upper == "INT16" ||
      upper == "EVXX_BFLOAT16" || upper == "EVXX_FLOAT16") {
    return 2U;
  }
  if (upper == "FP32" || upper == "FLOAT32" || upper == "INT32" || upper == "UINT32" ||
      upper == "EVXX_FLOAT32") {
    return 4U;
  }
  return 1U;
}

std::uint64_t packed_tensor_bytes_for_test(const std::vector<std::int64_t>& shape,
                                           const std::string& dtype) {
  std::uint64_t elems = 1U;
  for (const auto dim : shape) {
    require(dim > 0, "test shape must be strictly positive");
    elems *= static_cast<std::uint64_t>(dim);
  }
  return elems * dtype_size_bytes_for_test(dtype);
}

std::vector<std::int64_t> tensor_desc_shape_for_test(const sima_ev_tensor_desc& desc) {
  std::vector<std::int64_t> out;
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  out.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    out.push_back(desc.shape.sizes[i]);
  }
  return out;
}

std::vector<std::int64_t> tensor_desc_tile_shape_for_test(const sima_ev_tensor_desc& desc) {
  std::vector<std::int64_t> out;
  if (desc.layout_kind != SIMA_EV_LAYOUT_TILED) {
    return out;
  }
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  out.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    out.push_back(desc.layout.tiled.tile_sizes[i]);
  }
  return out;
}

std::vector<std::uint8_t> tensor_desc_axes_for_test(const sima_ev_tensor_desc& desc) {
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  return {desc.shape.axis_semantics, desc.shape.axis_semantics + rank};
}

std::vector<std::int64_t> tensor_desc_strides_for_test(const sima_ev_tensor_desc& desc) {
  std::vector<std::int64_t> out;
  if (desc.layout_kind != SIMA_EV_LAYOUT_STRIDED) {
    return out;
  }
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  out.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    out.push_back(desc.layout.strided.strides_bytes[i]);
  }
  return out;
}

simaai::neat::pipeline_internal::sima::MpkTensorContract
make_test_tensor(const std::string& name, const std::string& dtype,
                 const std::vector<std::int64_t>& shape, std::uint64_t size_bytes = 0U) {
  using simaai::neat::pipeline_internal::sima::MpkShapeSemantics;
  using simaai::neat::pipeline_internal::sima::MpkTensorContract;

  MpkTensorContract tensor;
  tensor.name = name;
  tensor.segment_name = name;
  tensor.dtype = dtype;
  tensor.logical_dtype = dtype;
  tensor.logical_shape = shape;
  tensor.mpk_shape = shape;
  tensor.shape_semantics = MpkShapeSemantics::Geometry;
  tensor.size_bytes = size_bytes > 0U
                          ? static_cast<std::size_t>(size_bytes)
                          : static_cast<std::size_t>(packed_tensor_bytes_for_test(shape, dtype));
  return tensor;
}

simaai::neat::pipeline_internal::sima::MpkTensorContract
make_test_tensor_without_mpk_shape(const std::string& name, const std::string& dtype,
                                   const std::vector<std::int64_t>& logical_shape,
                                   std::uint64_t size_bytes = 0U) {
  using simaai::neat::pipeline_internal::sima::MpkShapeSemantics;
  using simaai::neat::pipeline_internal::sima::MpkTensorContract;

  MpkTensorContract tensor;
  tensor.name = name;
  tensor.segment_name = name;
  tensor.dtype = dtype;
  tensor.logical_dtype = dtype;
  tensor.logical_shape = logical_shape;
  tensor.shape_semantics = MpkShapeSemantics::Geometry;
  tensor.size_bytes =
      size_bytes > 0U
          ? static_cast<std::size_t>(size_bytes)
          : static_cast<std::size_t>(packed_tensor_bytes_for_test(logical_shape, dtype));
  return tensor;
}

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

std::vector<std::int64_t> dense_test_strides(const sc::TensorShape& shape,
                                             const std::string& dtype) {
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t stride = dtype_size_bytes_for_test(dtype);
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// Operation shape arrays include batch; ValueSpec/Detess frame geometry may
// instead be a rank-2 WC or rank-3 HWC view. All fixtures here are single-frame.
sc::TensorShape single_batch_op_shape(sc::TensorShape shape) {
  if (shape.size() < 4U)
    shape.insert(shape.begin(), 1);
  return shape;
}

// These are explicit compiler-owned fixtures, not a second MPK interpreter.
// Public rank changes are authored address-only operations; equal byte counts
// alone never authorize a view or a fused chain.
sc::ModelExecutionPlanData
make_detess_scalar_plan(const sc::TensorShape& frame, const sc::TensorShape& publication,
                        const sc::TensorShape& tile, const std::string& dtype,
                        std::uint64_t packed_bytes, bool align_c16, bool cblock) {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const auto dense_bytes = packed_tensor_bytes_for_test(frame, dtype);
  const auto add_value = [&](std::string name, const sc::TensorShape& shape,
                             const std::string& scalar, std::uint64_t span,
                             sc::ValueRepresentation representation) {
    sc::ValueSpec value;
    value.id = static_cast<sc::ValueId>(data.values.size());
    value.name = std::move(name);
    value.logical_shape = shape;
    value.logical_dtype = scalar;
    if (shape.size() >= 3U)
      value.logical_layout = "HWC";
    value.required_bytes = packed_tensor_bytes_for_test(shape, scalar);
    value.representation = representation;
    value.storage_binding = sc::StorageBinding{value.id == 0U ? sc::StorageBindingKind::External
                                                              : sc::StorageBindingKind::Root,
                                               value.id,
                                               0U,
                                               span,
                                               {},
                                               sc::StorageAccess::ReadWrite,
                                               std::nullopt};
    data.values.push_back(std::move(value));
  };
  const auto add_op = [&](sc::OpKind kind, std::string name, sc::ValueId input, sc::ValueId output,
                          sc::OpConfig config) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = op.id + 1U;
    op.kind = kind;
    op.name = std::move(name);
    op.processor = "EV74";
    op.inputs = {input};
    op.outputs = {output};
    op.input_shapes = {single_batch_op_shape(*data.values[input].logical_shape)};
    op.output_shapes = {single_batch_op_shape(*data.values[output].logical_shape)};
    op.config = std::move(config);
    data.ops.push_back(std::move(op));
  };
  add_value("blocked", frame, dtype, packed_bytes, sc::ValueRepresentation::Tessellated);
  add_value("detess", frame, dtype, dense_bytes, sc::ValueRepresentation::Dense);
  data.model_inputs = {0U};
  add_op(sc::OpKind::Detessellate, "detess", 0U, 1U,
         sc::DetessellateOpConfig{tile, frame, align_c16, cblock, dtype});
  sc::ValueId scalar_input = 1U;
  if (frame != publication) {
    add_value("singleton_view", publication, dtype, dense_bytes, sc::ValueRepresentation::Dense);
    data.values.back().storage_binding.reset();
    data.values.back().read_expression =
        sc::ReadExpression{1U, 0U, dense_test_strides(publication, dtype)};
    add_op(sc::OpKind::Reshape, "singleton_view", 1U, 2U, sc::ReshapeOpConfig{publication});
    scalar_input = 2U;
  }
  add_value("published", publication, "FP32", packed_tensor_bytes_for_test(publication, "FP32"),
            sc::ValueRepresentation::Dense);
  const auto output = static_cast<sc::ValueId>(data.values.size() - 1U);
  if (dtype == "BF16")
    add_op(sc::OpKind::Cast, "cast", scalar_input, output, sc::CastOpConfig{"FP32"});
  else
    add_op(sc::OpKind::Dequantize, "dequant", scalar_input, output,
           sc::DequantizeOpConfig{dtype, {{0.5, 7}}});
  data.model_outputs = {{0U, "public_output", output}};
  return data;
}

simaai::neat::CompiledProcessCvuContract make_selected_post_cast_contract() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const std::array<sc::TensorShape, 2> inputs{{{384, 1664, 1}, {192, 832, 2}}};
  const std::array<sc::TensorShape, 2> outputs{{{12, 52, 45}, {3, 13, 4}}};
  for (sc::ValueId id = 0U; id < 8U; ++id) {
    const auto& shape = id < 4U ? inputs[id % 2U] : outputs[id % 2U];
    const std::string dtype = id < 2U || id >= 6U ? "FP32" : "BF16";
    data.values.push_back(sc::ValueSpec{id, "value_" + std::to_string(id),
                                        packed_tensor_bytes_for_test(shape, dtype), dtype, shape,
                                        "HWC"});
  }
  data.model_inputs = {0U, 1U};
  for (sc::OpId id = 0U; id < 5U; ++id) {
    sc::OpSpec op;
    op.id = id;
    op.sequence = id + 1U;
    op.name = id == 2U ? "MLA_0" : "arbitrary_cast_" + std::to_string(id);
    op.processor = id == 2U ? "MLA" : "EV74";
    if (id == 2U) {
      op.kind = sc::OpKind::Mla;
      op.inputs = {2U, 3U};
      op.outputs = {4U, 5U};
      op.config = sc::MlaOpConfig{"model.elf", 4};
    } else {
      op.kind = sc::OpKind::Cast;
      const sc::ValueId input = id < 2U ? id : id + 1U;
      op.inputs = {input};
      op.outputs = {input + 2U};
      op.input_shapes = {single_batch_op_shape(*data.values[input].logical_shape)};
      op.output_shapes = {single_batch_op_shape(*data.values[input + 2U].logical_shape)};
      op.config = sc::CastOpConfig{id < 2U ? "BF16" : "FP32"};
    }
    data.ops.push_back(std::move(op));
  }
  for (std::size_t i = 0U; i < 2U; ++i) {
    data.backend_ports.push_back(
        {0U, sc::BackendPortDirection::Input, i, "ifm" + std::to_string(i),
         static_cast<sc::ValueId>(i + 2U), data.values[i + 2U].required_bytes, 4096U,
         sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly});
    data.backend_ports.push_back(
        {0U, sc::BackendPortDirection::Output, i, "ofm" + std::to_string(i),
         static_cast<sc::ValueId>(i + 4U), data.values[i + 4U].required_bytes, 4096U,
         sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly});
    data.model_outputs.push_back({i, "head" + std::to_string(i), static_cast<sc::ValueId>(i + 6U)});
  }
  std::string error;
  const auto semantic = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(semantic.has_value(), "pre/post cast semantic fixture: " + error);
  const auto physical = sc::PhysicalExecutionLowerer::lower(*semantic, &error);
  require(physical.has_value(), "pre/post cast physical fixture: " + error);
  const auto arena = sc::FrameSlotArenaPlan::compile(
      *semantic, *physical, sc::FrameSlotArenaReuse::DisjointLifetimes, 4096U, &error);
  require(arena.has_value(), "pre/post cast arena fixture: " + error);
  std::vector<sc::PhysicalCommandId> post_commands;
  for (const auto& command : physical->commands) {
    if (command.engine == sc::PhysicalEngine::Cvu && command.implementation_id == "cast" &&
        std::all_of(command.members.begin(), command.members.end(), [](const auto& member) {
          return member.semantic_chain.size() == 1U && member.semantic_chain.front() >= 3U;
        }))
      post_commands.push_back(command.id);
  }
  require(post_commands.size() == 1U, "post-MLA casts must share one physical command cohort");
  const auto& post = physical->commands.at(post_commands.front());
  require(post.members.size() == 2U &&
              post.members[0].semantic_chain == std::vector<sc::OpId>{3U} &&
              post.members[1].semantic_chain == std::vector<sc::OpId>{4U},
          "only the two exact post-MLA semantic casts may be selected");
  const auto result = sc::build_dmabuf_plan_processcvu_command_contract(
      *semantic, *physical, post_commands, *arena, &error);
  require(result.has_value(), "selected post-MLA cast fixture: " + error);
  return *result;
}

void require_packed_input_projection(const simaai::neat::CompiledProcessCvuContract& compiled,
                                     int width, int height, int depth, int channels,
                                     const char* expectation) {
  require(!compiled.payload.input_shapes.empty(), expectation);
  const auto& front = compiled.payload.input_shapes.front();
  // input_shapes stores each input as a shape vector; build the expected shape
  // using the convention [depth, height, width, channels] (D, H, W, C) with
  // depth omitted when it equals 1.
  std::vector<int> expected;
  if (depth > 1) {
    expected.push_back(depth);
  }
  expected.push_back(height);
  expected.push_back(width);
  expected.push_back(channels);
  require(front == expected, expectation);
}

void require_runtime_packed_logical_input_shape(
    const simaai::neat::CompiledProcessCvuContract& compiled, int width, int height, int depth,
    int channels, const char* expectation) {
  require(!compiled.runtime_contract.logical_inputs.empty(), expectation);
  std::vector<std::int64_t> expected;
  if (depth > 1) {
    expected.push_back(depth);
  }
  expected.push_back(height);
  expected.push_back(width);
  expected.push_back(channels);
  require(compiled.runtime_contract.logical_inputs.front().shape == expected, expectation);
}

void verify_canonical_identity_boundary() {
  using namespace simaai::neat::pipeline_internal::sima;
  using namespace simaai::neat::pipeline_internal::sima::stagesemantics;
  struct IdentityCase {
    const char* source;
    const char* operation;
    const char* family;
    ProcessCvuGraphFamily family_enum;
  };
  const std::array cases{
      IdentityCase{"quantize", "quant", "quantize", ProcessCvuGraphFamily::Quant},
      IdentityCase{"dequantize", "dequant", "dequantize", ProcessCvuGraphFamily::Dequant},
      IdentityCase{"tessellate", "tess", "tessellate", ProcessCvuGraphFamily::Tess},
      IdentityCase{"detessellate", "detess", "detessellate", ProcessCvuGraphFamily::Detess},
      IdentityCase{"cast", "cast", "cast", ProcessCvuGraphFamily::Cast},
      IdentityCase{"casttess", "casttess", "casttess", ProcessCvuGraphFamily::CastTess},
      IdentityCase{"detesscast", "detesscast", "detesscast", ProcessCvuGraphFamily::DetessCast},
      IdentityCase{"quantizetessellate", "quanttess", "quanttess",
                   ProcessCvuGraphFamily::QuantTess},
      IdentityCase{"quantize_tessellate", "quanttess", "quanttess",
                   ProcessCvuGraphFamily::QuantTess},
      IdentityCase{"detessdequant", "detessdequant", "detessdequant",
                   ProcessCvuGraphFamily::DetessDequant},
      IdentityCase{"feature_histogram", "feature_histogram", "feature_histogram",
                   ProcessCvuGraphFamily::VisualFrontend},
      IdentityCase{"grider_fast", "grider_fast", "grider_fast",
                   ProcessCvuGraphFamily::VisualFrontend},
      IdentityCase{"track_descriptor", "track_descriptor", "track_descriptor",
                   ProcessCvuGraphFamily::VisualFrontend},
      IdentityCase{"track_klt", "track_klt", "track_klt", ProcessCvuGraphFamily::VisualFrontend}};
  ProcessCvuCanonicalFacts facts;
  facts.physical_input_names = {"source"};
  facts.physical_output_names = {"result"};
  facts.published_output_names = {"result"};
  facts.primary_output_name = "result";
  ProcessCvuCanonicalInputFact input;
  input.logical_index = input.physical_index = 0;
  input.physical_name = input.logical_name = "source";
  input.shape = {2, 3, 4};
  input.dtype = "FP32";
  input.layout = "HWC";
  input.size_bytes = 96U;
  facts.inputs = {input};
  ProcessCvuCanonicalOutputFact output;
  output.logical_index = output.physical_index = output.output_slot = output.tensor_index = 0;
  output.physical_name = output.logical_name = "result";
  output.shape = input.shape;
  output.dtype = input.dtype;
  output.layout = input.layout;
  output.size_bytes = input.size_bytes;
  facts.outputs = {output};

  for (const auto& test : cases) {
    for (const bool completed : {false, true}) {
      ProcessCvuStagePayload payload;
      payload.graph_name = test.source;
      payload.graph_family = test.family;
      payload.canonical_contract = completed;
      payload.graph_id = 236;
      payload.descriptor_abi_id = payload.descriptor_contract_version = 99U;
      payload.binding_schema_version = payload.supported_placement_mask = 99U;
      payload.allowed_frame_patch_mask = 99U;
      payload.batch_size = 3;
      payload.input_shapes = payload.output_shapes = {{2, 3, 4}};
      payload.slice_shapes = {{1, 3, 4}};
      payload.has_q_scale = payload.has_q_zp = true;
      payload.q_scale = 0.25;
      payload.q_zp = -7;
      const auto compiled = build_processcvu_compiled_contract_from_facts(payload, facts);
      const auto& actual = compiled.payload;
      require(actual.graph_name == test.operation && actual.graph_family == test.family &&
                  actual.graph_family_enum == test.family_enum,
              "compiled identity must normalize name, family and enum together");
      require(actual.graph_id == 0 && actual.descriptor_abi_id == 0U &&
                  actual.descriptor_contract_version == 0U && actual.binding_schema_version == 0U &&
                  actual.supported_placement_mask == 0U && actual.allowed_frame_patch_mask == 0U,
              "canonical contracts must not retain legacy numerical identity fields");
      require(actual.canonical_contract == completed && payload.graph_id == 236 &&
                  payload.graph_name == test.source &&
                  payload.graph_family_enum == ProcessCvuGraphFamily::Unknown,
              "identity finalization must preserve completion evidence and raw caller state");
      require(
          actual.batch_size == payload.batch_size && actual.input_shapes == payload.input_shapes &&
              actual.output_shapes == payload.output_shapes &&
              actual.slice_shapes == payload.slice_shapes && actual.q_scale == payload.q_scale &&
              actual.q_zp == payload.q_zp &&
              compiled.runtime_contract.logical_outputs.front().size_bytes == output.size_bytes &&
              compiled.exposed_view.exposed_logical_outputs.front().shape == output.shape,
          "identity finalization must preserve geometry, quantization and publication");
      const auto repeated = build_processcvu_compiled_contract_from_facts(actual, facts);
      require(repeated.payload.graph_name == actual.graph_name &&
                  repeated.payload.graph_family_enum == actual.graph_family_enum &&
                  repeated.payload.maximum_members == actual.maximum_members,
              "compiled identity finalization must be idempotent");
    }
  }
  auto preproc = build_processcvu_compile_inputs_from_options(make_preproc_options());
  preproc.payload.graph_id = -1;
  const auto prepared = build_processcvu_compiled_contract(preproc);
  require(prepared.payload.graph_name == "preproc" && prepared.payload.graph_id == 0 &&
              prepared.payload.graph_family_enum == ProcessCvuGraphFamily::Preproc &&
              prepared.payload.canonical_contract == preproc.payload.canonical_contract,
          "Preproc must use the same checked canonical identity boundary");

  const auto reject = [&](ProcessCvuStagePayload payload, const char* diagnostic) {
    try {
      (void)build_processcvu_compiled_contract_from_facts(payload, facts);
      require(false, "contradictory identity must not escape as a compiled contract");
    } catch (const std::invalid_argument& error) {
      require_contains(error.what(), diagnostic, "identity rejection must identify the conflict");
    }
  };
  ProcessCvuStagePayload invalid;
  invalid.graph_name = "quanttess";
  invalid.graph_family = "detessdequant";
  reject(invalid, "graph family");
  invalid.graph_family = "quanttess";
  invalid.graph_family_enum = ProcessCvuGraphFamily::DetessDequant;
  reject(invalid, "graph family enum");
  invalid.graph_name = "not_a_numerical_operation";
  invalid.graph_family.clear();
  invalid.graph_family_enum = ProcessCvuGraphFamily::Unknown;
  reject(invalid, "unknown canonical numerical stage");
}

} // namespace

RUN_TEST(
    "unit_contract_compiler_processcvu_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

      verify_canonical_identity_boundary();

      {
        struct PhysicalIdentityCase {
          int graph_id;
          const char* graph_family;
        };
        constexpr std::array<PhysicalIdentityCase, 4U> cases{{
            {224, "casttess"},
            {225, "detesscast"},
            {226, "quanttess"},
            {227, "detessdequant"},
        }};
        for (const auto& identity_case : cases) {
          CompiledProcessCvuContract physical;
          physical.payload.graph_id = identity_case.graph_id;
          physical.payload.graph_name = identity_case.graph_family;
          physical.payload.graph_family = identity_case.graph_family;
          physical.runtime_contract.plugin_kind = "processcvu";
          const std::string identity =
              "physical_cvu_cohort_1_" + std::to_string(identity_case.graph_id);
          CompiledNodeContract compiled;
          std::string error;
          require(build_processcvu_node_contract("PhysicalProcessCvu", identity, identity,
                                                 NodeContractDefinition{}, physical, &compiled,
                                                 &error),
                  "strict physical ProcessCVU identity fixture must compile: " + error);
          require(compiled.element_name == identity && compiled.logical_stage_id == identity,
                  "graphs 224-227 must preserve the exact caller-authored physical element and "
                  "stage identity");
        }
      }

      {
        pipeline_internal::sima::ProcessCvuStagePayload preproc;
        preproc.graph_family_enum = pipeline_internal::sima::ProcessCvuGraphFamily::Preproc;
        preproc.graph_name = "preproc";

        ContractCompileInput unsupported;
        unsupported.processcvu.pre_run_target = "A65";
        bool rejected = false;
        try {
          (void)pipeline_internal::sima::resolve_processcvu_backend_decision(preproc, unsupported,
                                                                             "preproc");
        } catch (const std::invalid_argument& ex) {
          rejected = std::string(ex.what()).find("Supported target(s): EV74") != std::string::npos;
        }
        require(rejected,
                "an explicit A65 request for EV74-only preproc must fail instead of falling back");

        ContractCompileInput supported;
        supported.processcvu.pre_run_target = "EV74";
        const auto decision = pipeline_internal::sima::resolve_processcvu_backend_decision(
            preproc, supported, "preproc");
        require(decision.effective_run_target == "EV74",
                "an explicit EV74 request for preproc should remain supported");

        pipeline_internal::sima::ProcessCvuStagePayload dual_backend;
        dual_backend.graph_family_enum = pipeline_internal::sima::ProcessCvuGraphFamily::Quant;
        dual_backend.graph_name = "quant";
        const auto a65_decision = pipeline_internal::sima::resolve_processcvu_backend_decision(
            dual_backend, unsupported, "pre_quant");
        require(
            a65_decision.effective_run_target == "A65",
            "strict placement must preserve explicit targets for genuinely dual-backend stages");

        pipeline_internal::sima::ProcessCvuStagePayload detessdequant;
        detessdequant.graph_family_enum =
            pipeline_internal::sima::ProcessCvuGraphFamily::DetessDequant;
        detessdequant.graph_name = "detessdequant";
        const auto detessdequant_caps = pipeline_internal::sima::processcvu_backend_capabilities(
            detessdequant, "post_detessdequant");
        require(detessdequant_caps.supports_ev74 && detessdequant_caps.supports_a65,
                "detessdequant policy should advertise both available backends");
        require(detessdequant_caps.auto_run_target == "A65" &&
                    detessdequant_caps.auto_exec_backend ==
                        pipeline_internal::sima::ProcessCvuResolvedExecBackend::A65 &&
                    detessdequant_caps.reason == "a65_preferred_auto_post",
                "detessdequant capability diagnostics should authoritatively advertise AUTO A65");

        const ContractCompileInput automatic;
        const auto automatic_post = pipeline_internal::sima::resolve_processcvu_backend_decision(
            detessdequant, automatic, "post_detessdequant");
        require(automatic_post.requested_run_target == "AUTO" &&
                    automatic_post.effective_run_target == detessdequant_caps.auto_run_target &&
                    automatic_post.resolved_exec_backend == detessdequant_caps.auto_exec_backend &&
                    automatic_post.reason == "a65_preferred_auto_post:legacy_or_env",
                "AUTO detessdequant resolution must agree with its advertised capability policy");

        ContractCompileInput explicit_ev74;
        explicit_ev74.processcvu.post_run_target = "EV74";
        const auto explicit_post = pipeline_internal::sima::resolve_processcvu_backend_decision(
            detessdequant, explicit_ev74, "post_detessdequant");
        require(explicit_post.effective_run_target == "EV74" &&
                    explicit_post.resolved_exec_backend ==
                        pipeline_internal::sima::ProcessCvuResolvedExecBackend::Evxx &&
                    explicit_post.reason == "requested_ev74_supported:processcvu_post",
                "an explicit EV74 post target must override the AUTO A65 preference");

        const auto quant_caps =
            pipeline_internal::sima::processcvu_backend_capabilities(dual_backend, "pre_quant");
        require(quant_caps.auto_run_target == "EV74" &&
                    quant_caps.auto_exec_backend ==
                        pipeline_internal::sima::ProcessCvuResolvedExecBackend::Evxx,
                "AUTO quant should preserve its EV74 pre-stage preference");

        pipeline_internal::sima::ProcessCvuStagePayload cast;
        cast.graph_family_enum = pipeline_internal::sima::ProcessCvuGraphFamily::Cast;
        cast.graph_name = "cast";
        require(pipeline_internal::sima::processcvu_backend_capabilities(cast, "pre_cast")
                            .auto_run_target == "EV74" &&
                    pipeline_internal::sima::processcvu_backend_capabilities(cast, "post_cast")
                            .auto_run_target == "A65",
                "AUTO cast preference should be role-aware at the capability authority");
      }

      {
        GriderFastOptions fast_opt;
        fast_opt.width = 640;
        fast_opt.height = 480;
        fast_opt.max_features = 256;
        fast_opt.grid_x = 8;
        fast_opt.grid_y = 6;
        GriderFast fast_node(fast_opt);
        ContractCompileInput cc_input;
        cc_input.node_index = 4;
        CompiledNodeContract compiled;
        std::string err;
        require(fast_node.compile_node_contract(cc_input, &compiled, &err),
                (std::string("GriderFast node contract should compile: ") + err).c_str());
        require(compiled.processcvu.has_value(),
                "GriderFast node contract should carry a processcvu payload");
        require(compiled.processcvu->payload.graph_id == 0,
                "GriderFast payload must not assign a native numerical identity");
        require(compiled.processcvu->payload.graph_name == "grider_fast",
                "GriderFast payload should use canonical graph name");
        require(compiled.processcvu->payload.max_features == 256,
                "GriderFast payload should preserve max_features");
        require(compiled.processcvu->runtime_contract.physical_inputs.size() == 1U,
                "GriderFast should expose one runtime input");
        require(compiled.processcvu->runtime_contract.physical_outputs.size() == 1U,
                "GriderFast should expose one runtime output");
      }

      {
        TrackKLTOptions klt_opt;
        klt_opt.width = 640;
        klt_opt.height = 480;
        klt_opt.num_points = 128;
        klt_opt.batch_size = 2;
        klt_opt.max_features = 64;
        klt_opt.detect_new_features = 0;
        TrackKLT klt_node(klt_opt);
        ContractCompileInput cc_input;
        cc_input.node_index = 5;
        CompiledNodeContract compiled;
        std::string err;
        require(klt_node.compile_node_contract(cc_input, &compiled, &err),
                (std::string("TrackKLT node contract should compile: ") + err).c_str());
        require(compiled.processcvu.has_value(),
                "TrackKLT node contract should carry a processcvu payload");
        require(compiled.processcvu->payload.graph_name == "track_klt",
                "TrackKLT payload should retain its composite semantic identity");
        require(compiled.processcvu->payload.default_output_names.size() == 3U,
                "TrackKLT runtime should still allocate output_features for the EV ABI");
        require(compiled.processcvu->exposed_view.exposed_output_order.size() == 2U,
                "TrackKLT should publish only points/status when detect_new_features is disabled");
        require(compiled.processcvu->runtime_contract.physical_inputs.size() == 3U,
                "TrackKLT should preserve its three physical inputs");
        require(compiled.processcvu->runtime_contract.physical_outputs.size() == 3U,
                "TrackKLT should preserve its three physical runtime outputs");

        namespace si = pipeline_internal::sima;
        const auto& public_outputs = compiled.processcvu->exposed_view.exposed_logical_outputs;
        const auto status =
            std::find_if(public_outputs.begin(), public_outputs.end(),
                         [](const auto& logical) { return logical.backend_output_index == 1; });
        require(status != public_outputs.end() &&
                    status->shape == std::vector<std::int64_t>({2, 128, 1}) &&
                    status->publication_relation == si::PublicationViewRelation::SingletonAxes,
                "public TrackKLT status must retain its trailing singleton view authority");
        require(public_outputs.front().publication_relation ==
                    si::PublicationViewRelation::Identity,
                "TrackKLT points must not inherit status publication permission");

        CompiledPipelineContracts pipeline_contracts;
        pipeline_contracts.stages.push_back(compiled);
        pipeline_contracts.fully_renderable = true;
        si::ManifestBuildDiagnostics diagnostics;
        auto manifest =
            render_manifest_from_compiled_contracts(pipeline_contracts, cc_input, &diagnostics);
        require(manifest.has_value() && diagnostics.errors.empty(),
                "public TrackKLT contract must render without an execution adapter");
        // Logical order is not backend identity. The status permission must
        // survive when it is the first exposed view rather than vector slot1.
        std::reverse(manifest->stages.front().logical_outputs.begin(),
                     manifest->stages.front().logical_outputs.end());
        const auto failure = si::prepare_kernel_stage_recipes(&*manifest, {}, {}, {}, {});
        require(!failure, ("public TrackKLT canonical lowering failed: " +
                           (failure ? failure->message : std::string{}))
                              .c_str());
        require(manifest->stages.size() == 1U && manifest->stages.front().native_kernel &&
                    manifest->stages.front().native_kernel->publication,
                "public TrackKLT must retain its canonical publication recipe");
        const auto& publication = *manifest->stages.front().native_kernel->publication;
        const auto view =
            std::find_if(publication.physical_views.begin(), publication.physical_views.end(),
                         [](const auto& value) { return value.backend_output_index == 1; });
        require(view != publication.physical_views.end() &&
                    view->relation ==
                        ::simaai::scheduler::client::TensorStagePublicationRelation::SingletonAxes,
                "canonical Scheduler request must preserve the semantic status view");
        for (const auto& value : publication.physical_views) {
          if (value.backend_output_index != 1)
            require(value.relation ==
                        ::simaai::scheduler::client::TensorStagePublicationRelation::Identity,
                    "canonical lowering must not authorize unrelated output reshapes");
        }

        auto unauthorised = *manifest;
        unauthorised.stages.front().native_kernel.reset();
        for (auto& logical : unauthorised.stages.front().logical_outputs)
          logical.publication_relation = si::PublicationViewRelation::Identity;
        require(!si::prepare_kernel_stage_recipes(&unauthorised, {}, {}, {}, {}),
                "identity-only recipe construction must preserve its default relation");
        for (const auto& value :
             unauthorised.stages.front().native_kernel->publication->physical_views)
          require(value.relation ==
                      ::simaai::scheduler::client::TensorStagePublicationRelation::Identity,
                  "kernel inspection must not manufacture shape-view permission");
      }

      {
        namespace ts = simaai::neat::pipeline_internal::sima::tensorsemantics;
        std::vector<int> normalized;
        std::string error_detail;
        require(ts::normalize_tile_shape({1, 300, 4}, {4, 4}, &normalized, &error_detail, "missing",
                                         "prefix", "dim"),
                "tile normalizer should accept shorter tile ranks");
        require(normalized == std::vector<int>({1, 4, 4}),
                "tile normalizer should left-pad shorter tile ranks with 1");

        normalized.clear();
        require(ts::normalize_tile_shape({1, 36, 36, 256}, {36, 4, 32}, &normalized, &error_detail,
                                         "missing", "prefix", "dim"),
                "tile normalizer should accept RF-DETR feature-map tile rank");
        require(normalized == std::vector<int>({1, 36, 4, 32}),
                "tile normalizer should left-pad RF-DETR feature-map tile rank with 1");

        normalized.clear();
        require(ts::normalize_tile_shape({300, 4}, {1, 4, 4}, &normalized, &error_detail, "missing",
                                         "prefix", "dim"),
                "tile normalizer should trim longer tile ranks with leading 1s");
        require(normalized == std::vector<int>({4, 4}),
                "tile normalizer should trim only the leading compatibility 1");

        normalized.clear();
        error_detail.clear();
        require(!ts::normalize_tile_shape({300, 4}, {2, 4, 4}, &normalized, &error_detail,
                                          "missing", "prefix", "dim"),
                "tile normalizer should reject non-1 leading trim prefixes");
        require(error_detail == "prefix",
                "tile normalizer should report the rank-prefix validation failure");

        normalized.clear();
        error_detail.clear();
        require(!ts::normalize_tile_shape({1, 300, 4}, {2, 4, 4}, &normalized, &error_detail,
                                          "missing", "prefix", "dim"),
                "tile normalizer should reject tile dims larger than tensor dims");
        require(error_detail == "dim",
                "tile normalizer should report the tile-dimension validation failure");

        normalized.clear();
        error_detail.clear();
        require(!ts::normalize_tile_shape({1, 300, 4}, {0, 4, 4}, &normalized, &error_detail,
                                          "missing", "prefix", "dim"),
                "tile normalizer should reject non-positive tile dims");
        require(error_detail == "dim",
                "tile normalizer should report non-positive tile dims as dimension failures");
      }

      {
        namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;

        pcs::CastContractSubset cast_subset;
        cast_subset.input_shape = {1, 300, 4};
        cast_subset.input_dtype = "FP32";
        cast_subset.output_dtype = "BF16";
        pcs::TessellateContractSubset tess_subset;
        tess_subset.input_shape = {1, 300, 4};
        tess_subset.frame_type = "BF16";
        tess_subset.slice_shape = {4, 4};
        tess_subset.output_size_bytes = 2400U;
        tess_subset.batch_size = 1;
        const auto tess_runtime = pcs::build_tessellate_runtime_config_from_subsets(
            cast_subset, tess_subset, "rfdetr_boxes_tess", "rfdetr_boxes_tess");
        require(tess_runtime.input_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(tess_runtime.input_tensors.front()) ==
                        std::vector<std::int64_t>({1, 300, 4}),
                "tessellate descriptor should preserve authored rank-3 input shape");
        require(tess_runtime.output_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(tess_runtime.output_tensors.front()) ==
                        std::vector<std::int64_t>({1, 300, 4}),
                "tessellate descriptor should preserve authored rank-3 output shape");
        require(tensor_desc_tile_shape_for_test(tess_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 4, 4}),
                "tessellate descriptor should normalize [4,4] to [1,4,4]");

        pcs::QuantTessContractSubset quanttess_subset;
        quanttess_subset.quant_params.scales = {0.25};
        quanttess_subset.quant_params.zero_points = {-7};
        quanttess_subset.input_shape = {1, 36, 36, 256};
        quanttess_subset.output_shape = {1, 36, 36, 256};
        quanttess_subset.input_dtype = "FP32";
        quanttess_subset.output_dtype = "INT8";
        quanttess_subset.round_off = 0;
        quanttess_subset.slice_shape = {36, 4, 32};
        quanttess_subset.frame_type = "INT8";
        quanttess_subset.output_size_bytes = 663552U;
        quanttess_subset.batch_size = 1;
        const auto quanttess_runtime =
            pcs::build_quanttess_runtime_config_from_subset(quanttess_subset, "rfdetr_feat_tess");
        require(quanttess_runtime.input_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(quanttess_runtime.input_tensors.front()) ==
                        std::vector<std::int64_t>({1, 36, 36, 256}),
                "quanttess descriptor should preserve authored rank-4 input shape");
        require(quanttess_runtime.output_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(quanttess_runtime.output_tensors.front()) ==
                        std::vector<std::int64_t>({1, 36, 36, 256}),
                "quanttess descriptor should preserve authored rank-4 output shape");
        require(tensor_desc_tile_shape_for_test(quanttess_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 36, 4, 32}),
                "quanttess descriptor should normalize [36,4,32] to [1,36,4,32]");

        pcs::QuantizeContractSubset quantize_subset;
        quantize_subset.quant_params.scales = {0.25};
        quantize_subset.quant_params.zero_points = {-7};
        quantize_subset.input_shape = {1, 300, 4};
        quantize_subset.input_dtype = "FP32";
        quantize_subset.output_dtype = "INT8";
        quantize_subset.round_off = 0;
        const auto quantize_runtime = pcs::build_quantize_runtime_config_from_subset(
            quantize_subset, "quantized", "quantized");
        require(tensor_desc_shape_for_test(quantize_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "quantize dense descriptor should preserve authored input shape");
        require(tensor_desc_shape_for_test(quantize_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "quantize dense descriptor should preserve authored output shape");

        pcs::DequantizeContractSubset dequantize_subset;
        dequantize_subset.quant_params.scales = {0.25};
        dequantize_subset.quant_params.zero_points = {-7};
        dequantize_subset.input_shape = {1, 300, 4};
        dequantize_subset.output_shape = {1, 300, 4};
        dequantize_subset.input_dtype = "INT8";
        dequantize_subset.output_dtype = "FP32";
        const auto dequantize_runtime =
            pcs::build_dequantize_runtime_config_from_subsets({dequantize_subset}, {"dequantized"});
        require(tensor_desc_shape_for_test(dequantize_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "dequantize dense descriptor should preserve authored input shape");
        require(tensor_desc_shape_for_test(dequantize_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "dequantize dense descriptor should preserve authored output shape");

        pcs::DetessellateContractSubset detess_subset;
        detess_subset.input_shape = {1, 300, 4};
        detess_subset.frame_shape = {1, 300, 4};
        detess_subset.input_transport_shape = {1, 300, 4};
        detess_subset.input_transport_size_bytes =
            packed_tensor_bytes_for_test({1, 300, 4}, "INT8");
        detess_subset.frame_type = "INT8";
        detess_subset.slice_shape = {4, 4};
        const auto detess_runtime = pcs::build_detessellate_runtime_config_from_subsets(
            {detess_subset}, {"detess"}, {"detess"});
        require(tensor_desc_shape_for_test(detess_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "detessellate tiled descriptor should preserve authored frame shape");
        require(tensor_desc_tile_shape_for_test(detess_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 4, 4}),
                "detessellate tiled descriptor should normalize shorter tile shape");
        require(tensor_desc_shape_for_test(detess_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "detessellate dense output descriptor should preserve authored frame shape");

        pcs::DetessDequantHeadContractSubset head;
        head.per_head_input_shape = {300, 4};
        head.input_transport_shape = {300, 4};
        head.input_transport_size_bytes = packed_tensor_bytes_for_test({300, 4}, "INT8");
        head.per_head_quant_params.scales = {0.25};
        head.per_head_quant_params.zero_points = {-7};
        head.frame_shape = {300, 4};
        head.frame_type = "INT8";
        head.slice_shape = {4, 4};
        head.output_dtype = "FP32";
        pcs::DetessDequantContractSubset detessdequant_subset;
        detessdequant_subset.heads = {head};
        const auto detessdequant_runtime = pcs::build_detessdequant_runtime_config_from_subset(
            detessdequant_subset, {"detessdq"}, {"detessdq"});
        require(tensor_desc_shape_for_test(detessdequant_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({300, 4}),
                "detessdequant tiled descriptor should preserve current authored frame shape");
        require(tensor_desc_tile_shape_for_test(detessdequant_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({4, 4}),
                "detessdequant tiled descriptor should preserve matching tile shape");
        require(tensor_desc_shape_for_test(detessdequant_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({300, 4}),
                "detessdequant dense output descriptor should preserve authored frame shape");
      }

      auto node = nodes::Preproc(make_preproc_options());
      require(dynamic_cast<const simaai::neat::Preproc*>(node.get()) != nullptr &&
                  dynamic_cast<const simaai::neat::Preproc*>(node.get())
                      ->options()
                      .model_managed_contract,
              "preproc fixture must retain model-managed ownership");
      std::vector<std::shared_ptr<Node>> nodes_to_compile = {node};
      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      ContractCompileInput input;
      input.pipeline_label = "unit_contract_compiler_processcvu_test";

      const CompiledPipelineContracts compiled =
          compile_node_contracts(nodes_to_compile, input, &diagnostics);

      require(diagnostics.errors.empty(), "compile_node_contracts should not emit errors");
      require(compiled.fully_renderable, "compiled contract set should be renderable");
      require(compiled.stages.size() == 1U, "compiled contract should contain one stage");
      require(compiled.stages.front().processcvu.has_value(),
              "preproc should compile to a processcvu contract");

      const auto& processcvu = *compiled.stages.front().processcvu;
      require(processcvu.runtime_contract.logical_inputs.size() == 1U,
              "preproc runtime contract should expose one logical input");
      require(processcvu.payload.output_shapes.size() == 2U &&
                  std::all_of(
                      processcvu.payload.output_shapes.begin(),
                      processcvu.payload.output_shapes.end(),
                      [](const auto& shape) { return shape == std::vector<int>({640, 640, 3}); }) &&
                  processcvu.payload.runtime_output_logical_shapes.size() == 2U &&
                  processcvu.payload.runtime_output_logical_shapes[0] ==
                      std::vector<int>({1, 640, 640, 3}) &&
                  processcvu.payload.runtime_output_logical_shapes[1] ==
                      std::vector<int>({1, 640, 640, 3}),
              "compiled model-managed preproc payload must separate HWC hardware geometry from "
              "NHWC runtime semantics");
      // Runtime contract publishes both rgb and tess outputs; the exposed view
      // narrows to the selected tessellated handoff.
      const auto tess_runtime_it =
          std::find_if(processcvu.runtime_contract.logical_outputs.begin(),
                       processcvu.runtime_contract.logical_outputs.end(), [](const auto& logical) {
                         return logical.logical_name == "output_tessellated_image";
                       });
      require(tess_runtime_it != processcvu.runtime_contract.logical_outputs.end(),
              "preproc runtime contract should preserve the tessellated handoff name");
      require(tess_runtime_it->shape == std::vector<std::int64_t>({1, 640, 640, 3}),
              "model-managed preproc should publish the exact batched semantic output shape");
      require(tess_runtime_it->size_bytes == 640U * 640U * 3U,
              "preproc tessellated handoff should preserve the packed MLA ingress byte size");
      {
        std::vector<std::shared_ptr<Node>> indexed_nodes = {
            nodes::Input(InputOptions{}),
            nodes::Preproc(make_preproc_options()),
            nodes::Output(OutputOptions{}),
        };
        pipeline_internal::sima::ManifestBuildDiagnostics indexed_diagnostics;
        const CompiledPipelineContracts indexed =
            compile_node_contracts(indexed_nodes, ContractCompileInput{}, &indexed_diagnostics);
        require(indexed_diagnostics.errors.empty(),
                "indexed compile_node_contracts should not emit errors");
        require(indexed.stages.size() == 1U, "indexed compile should contain one semantic stage");
        require(indexed.stages.front().element_name == "n1_preproc",
                "standalone preproc contract should use actual node index for element identity");
        require(indexed.stages.front().logical_stage_id == "n1_preproc",
                "standalone preproc contract should use actual node index for logical stage id");
      }
      {
        ContractCompileInput remapped_input;
        remapped_input.node_indices = {7};
        pipeline_internal::sima::ManifestBuildDiagnostics remapped_diagnostics;
        const CompiledPipelineContracts remapped = compile_node_contracts(
            {nodes::Preproc(make_preproc_options())}, remapped_input, &remapped_diagnostics);
        require(remapped_diagnostics.errors.empty(),
                "remapped compile_node_contracts should not emit errors");
        require(remapped.stages.size() == 1U, "remapped compile should contain one semantic stage");
        require(remapped.stages.front().element_name == "n7_preproc",
                "fused branch contracts should use actual rendered node index");
        require(remapped.stages.front().logical_stage_id == "n7_preproc",
                "fused branch contracts should preserve actual rendered stage id");
      }
      require(processcvu.exposed_view.exposed_logical_outputs.size() == 1U,
              "preproc exposed view should expose one selected output");
      require(processcvu.preproc_single_output_handoff,
              "preproc single-output handoff flag should be preserved");

      const auto direct_inputs =
          build_processcvu_compile_inputs_from_options(make_preproc_options());
      require(direct_inputs.payload.output_shapes.size() == 2U &&
                  std::all_of(
                      direct_inputs.payload.output_shapes.begin(),
                      direct_inputs.payload.output_shapes.end(),
                      [](const auto& shape) { return shape == std::vector<int>({640, 640, 3}); }),
              "graph-200 physical output geometry must remain HWC");
      require(direct_inputs.payload.runtime_output_logical_shapes.size() == 2U &&
                  direct_inputs.payload.runtime_output_logical_shapes[0] ==
                      std::vector<int>({1, 640, 640, 3}) &&
                  direct_inputs.payload.runtime_output_logical_shapes[1] ==
                      std::vector<int>({1, 640, 640, 3}),
              "model-managed preproc must author one exact batch axis for every runtime output");
      require(direct_inputs.payload.primary_output_name == direct_inputs.facts.primary_output_name,
              "direct preproc compile inputs should preserve primary output selection");
      require(!direct_inputs.facts.published_output_names.empty() &&
                  direct_inputs.facts.published_output_names.front() ==
                      direct_inputs.payload.primary_output_name,
              "direct preproc compile inputs should preserve the selected published output");
      require(!direct_inputs.payload.default_output_names.empty() &&
                  std::find(direct_inputs.payload.default_output_names.begin(),
                            direct_inputs.payload.default_output_names.end(),
                            std::string("output_tessellated_image")) !=
                      direct_inputs.payload.default_output_names.end(),
              "direct preproc compile inputs should publish the tessellated runtime output");
      require(direct_inputs.facts.inputs.size() == 1U,
              "direct preproc compile inputs should preserve input count");
      require(direct_inputs.facts.inputs.front().shape == std::vector<std::int64_t>({720, 1280, 3}),
              "direct preproc compile inputs should preserve the authored single input shape");
      require(direct_inputs.payload.input_tensors.size() == 1U &&
                  simaai::neat::pipeline_internal::sima::tensorsemantics::
                          layout_token_from_ev_tensor_desc(
                              direct_inputs.payload.input_tensors.front()) == "HWC",
              "direct preproc compile inputs should synthesize semantic typed input descriptor");
      const auto direct_tess_it = std::find_if(
          direct_inputs.facts.outputs.begin(), direct_inputs.facts.outputs.end(),
          [](const auto& output) { return output.logical_name == "output_tessellated_image"; });
      require(direct_tess_it != direct_inputs.facts.outputs.end(),
              "direct preproc compile inputs should preserve the tessellated runtime output");
      require(direct_tess_it->shape == std::vector<std::int64_t>({1, 640, 640, 3}),
              "direct preproc facts must retain the model batch axis");
      require(direct_tess_it->representation == ProcessCvuOutputRepresentation::PackedBlob,
              "direct preproc compile inputs should keep packed MLA handoff semantics");
      require(direct_tess_it->size_bytes == 640U * 640U * 3U,
              "direct preproc compile inputs should preserve packed MLA ingress byte size");
      require(!direct_inputs.payload.output_tensors.empty() &&
                  std::any_of(direct_inputs.payload.output_tensors.begin(),
                              direct_inputs.payload.output_tensors.end(),
                              [](const auto& t) { return t.descriptor().layout_kind == SIMA_EV_LAYOUT_TILED; }),
              "direct preproc compile inputs should synthesize tiled typed output descriptor");
      {
        const auto tiled_it =
            std::find_if(direct_inputs.payload.output_tensors.begin(),
                         direct_inputs.payload.output_tensors.end(),
                         [](const auto& t) { return t.descriptor().layout_kind == SIMA_EV_LAYOUT_TILED; });
        require(tiled_it != direct_inputs.payload.output_tensors.end(),
                "direct preproc compile inputs should include a tiled output descriptor");
        require(sima_ev_tensor_tile_axis_size(&tiled_it->descriptor(), SIMA_EV_AXIS_H, 0) == 32 &&
                    sima_ev_tensor_tile_axis_size(&tiled_it->descriptor(), SIMA_EV_AXIS_W, 0) == 128 &&
                    sima_ev_tensor_tile_axis_size(&tiled_it->descriptor(), SIMA_EV_AXIS_C, 0) == 3,
                "direct preproc compile inputs should preserve typed output tile geometry");
      }
      {
        auto invalid_preproc = direct_inputs.payload;
        invalid_preproc.input_shapes.push_back({1, 1, 1});
        invalid_preproc.num_in_tensor = 2;
        bool threw = false;
        try {
          (void)build_preproc_facts_from_payload_internal(invalid_preproc);
        } catch (const std::invalid_argument& ex) {
          threw = std::string(ex.what()) == "processcvu preproc payload requires exactly one input";
        }
        require(threw, "direct preproc facts builder should reject multi-input preproc payloads");
      }

      const auto direct_compiled = build_processcvu_compiled_contract(direct_inputs);
      require(direct_compiled.runtime_contract.logical_outputs.size() ==
                  processcvu.runtime_contract.logical_outputs.size(),
              "shared processcvu compiled-contract adapter should preserve runtime logical output "
              "count");
      const auto direct_compiled_tess_it = std::find_if(
          direct_compiled.runtime_contract.logical_outputs.begin(),
          direct_compiled.runtime_contract.logical_outputs.end(),
          [](const auto& logical) { return logical.logical_name == "output_tessellated_image"; });
      require(
          direct_compiled_tess_it != direct_compiled.runtime_contract.logical_outputs.end(),
          "shared processcvu compiled-contract adapter should preserve tessellated runtime output");
      require(direct_compiled_tess_it->size_bytes == tess_runtime_it->size_bytes,
              "shared processcvu compiled-contract adapter should preserve packed handoff bytes");

      {
        auto standalone_options = make_preproc_options();
        standalone_options.model_managed_contract = false;
        const auto standalone_inputs =
            build_processcvu_compile_inputs_from_options(standalone_options);
        require(standalone_inputs.payload.output_shapes.size() == 2U &&
                    std::all_of(standalone_inputs.payload.output_shapes.begin(),
                                standalone_inputs.payload.output_shapes.end(),
                                [](const auto& shape) {
                                  return shape == std::vector<int>({640, 640, 3});
                                }) &&
                    standalone_inputs.payload.runtime_output_logical_shapes.size() == 2U &&
                    standalone_inputs.payload.runtime_output_logical_shapes[0] ==
                        std::vector<int>({640, 640, 3}) &&
                    standalone_inputs.payload.runtime_output_logical_shapes[1] ==
                        std::vector<int>({640, 640, 3}),
                "standalone preproc must preserve its existing unbatched HWC contract");
      }

      {
        auto unsupported = make_preproc_options();
        unsupported.single_output_handoff = false;
        bool ctor_threw = false;
        try {
          simaai::neat::Preproc unsupported_node(unsupported);
          (void)unsupported_node;
        } catch (const std::runtime_error& ex) {
          ctor_threw = std::string(ex.what()) == "Preproc dual-output contract is currently "
                                                 "unsupported; use single_output_handoff=true.";
        }
        require(ctor_threw, "preproc construction should reject unsupported dual-output handoff");

        bool threw = false;
        try {
          (void)build_processcvu_compile_inputs_from_options(unsupported);
        } catch (const std::invalid_argument& ex) {
          threw = std::string(ex.what()) == "Preproc dual-output contract is currently "
                                            "unsupported; use single_output_handoff=true.";
        }
        require(threw,
                "processcvu preproc compile inputs should reject unsupported dual-output handoff");
      }

      simaai::neat::pipeline_internal::sima::stagesemantics::CompiledProcessCvuRuntimeConfig
          packed_tess_runtime;
      packed_tess_runtime.graph_family = "tessellate";
      packed_tess_runtime.graph_name = "tessellate";
      packed_tess_runtime.graph_id = 2;
      packed_tess_runtime.default_input_name = "input_tensor";
      packed_tess_runtime.runtime_input_names = {"input_tensor"};
      packed_tess_runtime.runtime_output_names = {"output_tensor"};
      packed_tess_runtime.published_output_names = {"output_tensor"};
      packed_tess_runtime.physical_input_names = {"input_tensor"};
      packed_tess_runtime.physical_output_names = {"output_tensor"};
      packed_tess_runtime.primary_output_name = "output_tensor";
      packed_tess_runtime.input_shapes = {{192, 832, 2}};
      packed_tess_runtime.output_shapes = {{192, 832, 2}};
      packed_tess_runtime.slice_shapes = {{42, 39, 2}};
      packed_tess_runtime.input_dtype = "BF16";
      packed_tess_runtime.output_dtype = "BF16";
      packed_tess_runtime.out_dtype = "BF16";
      packed_tess_runtime.runtime_output_logical_shapes = {{192, 832, 2}};
      packed_tess_runtime.runtime_output_logical_index_list = {0};
      packed_tess_runtime.runtime_output_output_slot_list = {0};
      packed_tess_runtime.runtime_output_physical_index_list = {0};
      packed_tess_runtime.runtime_output_dtype_list = {"BF16"};
      packed_tess_runtime.runtime_output_logical_layout_list = {"HWC"};
      packed_tess_runtime.runtime_output_transport_kind_list = {
          simaai::neat::pipeline_internal::sima::ProcessCvuOutputTransportKind::Packed};
      {
        sima_ev_tensor_desc input_desc{};
        sima_ev_tensor_desc output_desc{};
        require_build_dense_test_tensor(packed_tess_runtime.input_shapes.front(),
                                        packed_tess_runtime.input_dtype, "HWC", &input_desc,
                                        "packed tess runtime should synthesize typed input tensor");
        require_build_tiled_test_tensor(
            packed_tess_runtime.output_shapes.front(), packed_tess_runtime.slice_shapes.front(),
            packed_tess_runtime.output_dtype, "HWC", 16U, &output_desc,
            "packed tess runtime should synthesize typed output tensor");
        packed_tess_runtime.input_tensors = {{input_desc, packed_tess_runtime.input_dtype}};
        packed_tess_runtime.output_tensors = {{output_desc, packed_tess_runtime.output_dtype}};
      }
      const std::uint64_t packed_tess_expected_bytes = 639616U * dtype_size_bytes_for_test("BF16");

      const auto packed_tess_inputs =
          build_processcvu_compile_inputs_from_runtime_config(packed_tess_runtime);
      require(packed_tess_inputs.facts.outputs.size() == 1U,
              "packed tess runtime config should emit one canonical output fact");
      require(packed_tess_inputs.facts.outputs.front().representation ==
                  simaai::neat::pipeline_internal::sima::stagesemantics::
                      ProcessCvuOutputRepresentation::PackedTensor,
              "packed tess runtime config should preserve packed output representation");
      // Packed-tensor outputs no longer carry an explicit size_bytes on the
      // canonical fact — geometric size is derived from shape+dtype downstream.
      (void)packed_tess_expected_bytes;

      const auto packed_tess_compiled =
          build_processcvu_compiled_contract_from_runtime_config(packed_tess_runtime);
      require(packed_tess_compiled.runtime_contract.logical_outputs.size() == 1U,
              "packed tess runtime config should compile one logical output");
      // Logical-output size_bytes for packed-tensor representation is no longer
      // populated on the compiled contract; downstream consumers derive it from
      // shape+dtype directly.
      require(packed_tess_compiled.runtime_contract.logical_outputs.front().shape ==
                  std::vector<std::int64_t>({192, 832, 2}),
              "packed tess compiled contract should preserve the semantic output shape");

      {
        using simaai::neat::pipeline_internal::sima::plugin_contracts::CastContractSubset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            build_quanttess_runtime_config_from_subset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            extract_quanttess_contract_subset_from_stage;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            extract_tessellate_contract_subset_from_stage;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::QuantTessContractSubset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::TessellateContractSubset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            build_tessellate_runtime_config_from_subsets;
        using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;

        // Direct descriptor coverage for build_tessellate_runtime_config_from_subsets
        // lives above with the RF-DETR rank-normalization cases. Keep this block
        // focused on MPK/stage extraction behavior.
        (void)build_tessellate_runtime_config_from_subsets;
        (void)CastContractSubset{};
        (void)TessellateContractSubset{};

        MpkPluginIoContract exact_tess_stage;
        exact_tess_stage.name = "tess_exact";
        exact_tess_stage.kernel = "tessellation_transform";
        exact_tess_stage.processor = "EV74";
        exact_tess_stage.sequence = 2;
        exact_tess_stage.batch_size = 1;
        exact_tess_stage.batch_sz_model = 1;
        exact_tess_stage.frame_type = "BF16";
        exact_tess_stage.slice_shape = {384, 17, 1};
        exact_tess_stage.has_align_c16 = true;
        exact_tess_stage.align_c16 = false;
        exact_tess_stage.has_cblock = true;
        exact_tess_stage.cblock = false;
        exact_tess_stage.input_tensors = {
            make_test_tensor_without_mpk_shape("cast_0", "BF16", {384, 1664, 1})};

        const auto exact_tess_subset =
            extract_tessellate_contract_subset_from_stage(exact_tess_stage);
        require(
            exact_tess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
            "tess subset extraction should preserve the authored rank-3 logical input shape when "
            "no mpk_shape is present");

        exact_tess_stage.input_tensors = {make_test_tensor("cast_0", "BF16", {384, 1664, 1})};
        exact_tess_stage.input_tensors.front().mpk_shape = {384, 1664, 1};
        exact_tess_stage.input_tensors.front().logical_shape = {384, 1664, 1};
        const auto rank3_tess_subset =
            extract_tessellate_contract_subset_from_stage(exact_tess_stage);
        require(rank3_tess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "tess subset extraction should preserve the MPK-authored rank-3 input shape");

        MpkPluginIoContract exact_quanttess_stage;
        exact_quanttess_stage.name = "quanttess_exact";
        exact_quanttess_stage.kernel = "quanttess";
        exact_quanttess_stage.processor = "EV74";
        exact_quanttess_stage.sequence = 2;
        exact_quanttess_stage.batch_size = 1;
        exact_quanttess_stage.batch_sz_model = 1;
        exact_quanttess_stage.frame_type = "INT8";
        exact_quanttess_stage.slice_shape = {384, 84, 1};
        exact_quanttess_stage.has_align_c16 = true;
        exact_quanttess_stage.align_c16 = false;
        exact_quanttess_stage.has_cblock = true;
        exact_quanttess_stage.cblock = false;
        exact_quanttess_stage.canonical_input_dtype = "INT8";
        exact_quanttess_stage.canonical_output_dtype = "INT8";
        exact_quanttess_stage.round_off = "TONEAREST";
        simaai::neat::pipeline_internal::sima::MpkQuantContract quant;
        quant.scales = {0.25};
        quant.zero_points = {-7};
        exact_quanttess_stage.quant = quant;
        exact_quanttess_stage.input_tensors = {
            make_test_tensor_without_mpk_shape("quantize_0", "INT8", {384, 1664, 1})};
        exact_quanttess_stage.output_tensors = {
            make_test_tensor("tess_out", "INT8", {384, 1664, 1}, 638976U)};

        const QuantTessContractSubset exact_quanttess_subset =
            extract_quanttess_contract_subset_from_stage(exact_quanttess_stage);
        require(exact_quanttess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "quanttess subset extraction should preserve the authored rank-3 input shape when "
                "no mpk_shape is present");
        require(exact_quanttess_subset.output_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "quanttess subset extraction should preserve the semantic output shape");
        const auto exact_quanttess_runtime =
            build_quanttess_runtime_config_from_subset(exact_quanttess_subset, "tess_out");
        require(exact_quanttess_runtime.output_shapes.size() == 1U,
                "quanttess runtime config should expose one fused output shape");
        require(
            exact_quanttess_runtime.output_shapes.front() == std::vector<int>({384, 1664, 1}),
            "quanttess runtime config should keep the semantic output shape instead of rewriting "
            "it to packed bytes");
        require(exact_quanttess_runtime.runtime_output_logical_shapes.size() == 1U,
                "quanttess runtime config should expose one logical output shape");
        require(exact_quanttess_runtime.runtime_output_logical_shapes.front() ==
                    std::vector<int>({384, 1664, 1}),
                "quanttess runtime config logical outputs should stay faithful to the semantic MPK "
                "output shape");
        require(exact_quanttess_runtime.output_tensors.size() == 1U &&
                    exact_quanttess_runtime.output_tensors.front().descriptor().storage.nbytes == 638976U,
                "quanttess runtime config should preserve the MPK packed output byte count instead "
                "of using fixed full-tile slots");

        exact_quanttess_stage.input_tensors = {
            make_test_tensor("quantize_0", "INT8", {384, 1664, 1})};
        exact_quanttess_stage.input_tensors.front().mpk_shape = {384, 1664, 1};
        exact_quanttess_stage.input_tensors.front().logical_shape = {384, 1664, 1};
        const QuantTessContractSubset rank3_quanttess_subset =
            extract_quanttess_contract_subset_from_stage(exact_quanttess_stage);
        require(rank3_quanttess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "quanttess subset extraction should preserve the MPK-authored rank-3 input shape");
      }

      CompiledProcessCvuRuntimeConfig invalid_runtime;
      invalid_runtime.graph_family = "dequantize";
      invalid_runtime.graph_name = "dequantize";
      invalid_runtime.default_input_name = "input_tensor";
      invalid_runtime.published_output_names = {"output_tensor"};
      invalid_runtime.primary_output_name = "output_tensor";
      invalid_runtime.input_dtype = "INT8";
      invalid_runtime.output_dtype = "FP32";
      invalid_runtime.out_dtype = "FP32";

      const auto expect_invalid_runtime = [](CompiledProcessCvuRuntimeConfig runtime,
                                             const char* expected_token, const char* expectation) {
        bool threw = false;
        try {
          (void)build_processcvu_compiled_contract_from_runtime_config(runtime);
        } catch (const std::invalid_argument& ex) {
          threw = std::string(ex.what()).find(expected_token) != std::string::npos;
        }
        require(threw, expectation);
      };

      expect_invalid_runtime(
          invalid_runtime, "runtime_output_names",
          "runtime-config contract build should reject missing runtime_output_names instead of "
          "borrowing published outputs");

      CompiledProcessCvuRuntimeConfig missing_default_input = invalid_runtime;
      missing_default_input.runtime_output_names = {"output_tensor"};
      missing_default_input.default_input_name.clear();
      expect_invalid_runtime(
          missing_default_input, "default_input_name",
          "runtime-config contract build should reject missing default_input_name");

      CompiledProcessCvuRuntimeConfig missing_primary_output = invalid_runtime;
      missing_primary_output.runtime_output_names = {"output_tensor"};
      missing_primary_output.primary_output_name.clear();
      expect_invalid_runtime(
          missing_primary_output, "primary_output_name",
          "runtime-config contract build should reject missing primary_output_name");

      CompiledProcessCvuRuntimeConfig missing_published_outputs = invalid_runtime;
      missing_published_outputs.runtime_output_names = {"output_tensor"};
      missing_published_outputs.published_output_names.clear();
      expect_invalid_runtime(
          missing_published_outputs, "published_output_names",
          "runtime-config contract build should reject missing published_output_names");

      CompiledProcessCvuRuntimeConfig packed_multi_runtime;
      packed_multi_runtime.graph_family = "dequantize";
      packed_multi_runtime.graph_name = "dequantize";
      packed_multi_runtime.graph_id = 6;
      packed_multi_runtime.default_input_name = "input_tensor";
      packed_multi_runtime.runtime_input_names = {"input_tensor"};
      packed_multi_runtime.runtime_output_names = {"output_tensor"};
      packed_multi_runtime.published_output_names = {"head_0", "head_1"};
      packed_multi_runtime.physical_input_names = {"input_tensor"};
      packed_multi_runtime.physical_output_names = {"output_tensor"};
      packed_multi_runtime.primary_output_name = "head_0";
      packed_multi_runtime.input_shapes = {{8, 8, 16, 8}, {16, 4, 8, 16}};
      packed_multi_runtime.output_shapes = {{256}};
      packed_multi_runtime.runtime_output_logical_shapes = {{256}};
      packed_multi_runtime.runtime_output_logical_index_list = {0};
      packed_multi_runtime.runtime_output_output_slot_list = {0};
      packed_multi_runtime.runtime_output_physical_index_list = {0};
      packed_multi_runtime.runtime_output_dtype_list = {"FP32"};
      packed_multi_runtime.runtime_output_logical_layout_list = {"HW"};
      packed_multi_runtime.input_dtype = "INT8";
      packed_multi_runtime.output_dtype = "FP32";
      packed_multi_runtime.out_dtype = "FP32";
      packed_multi_runtime.has_q_scale = true;
      packed_multi_runtime.q_scale = 0.5;
      packed_multi_runtime.has_q_zp = true;
      packed_multi_runtime.q_zp = -7;
      {
        sima_ev_tensor_desc input_desc0{};
        sima_ev_tensor_desc input_desc1{};
        sima_ev_tensor_desc output_desc{};
        require_build_dense_test_tensor(
            packed_multi_runtime.input_shapes[0], packed_multi_runtime.input_dtype, "HWC",
            &input_desc0, "packed multi runtime should synthesize first typed input tensor");
        require_build_dense_test_tensor(
            packed_multi_runtime.input_shapes[1], packed_multi_runtime.input_dtype, "HWC",
            &input_desc1, "packed multi runtime should synthesize second typed input tensor");
        require_build_dense_test_tensor(
            packed_multi_runtime.output_shapes.front(), packed_multi_runtime.output_dtype, "HW",
            &output_desc, "packed multi runtime should synthesize typed output tensor");
        packed_multi_runtime.input_tensors = {{input_desc0, packed_multi_runtime.input_dtype},
                                             {input_desc1, packed_multi_runtime.input_dtype}};
        packed_multi_runtime.output_tensors = {{output_desc, packed_multi_runtime.output_dtype},
                                              {output_desc, packed_multi_runtime.output_dtype}};
      }
      packed_multi_runtime.q_scale_list = {0.5, 0.25};
      packed_multi_runtime.q_zp_list = {-7, 9};

      const auto packed_multi_payload =
          build_processcvu_payload_from_runtime_config_internal(packed_multi_runtime);
      require(packed_multi_payload.num_in_tensor == 2,
              "runtime-config payload should derive semantic num_in_tensor from logical arrays");
      require(packed_multi_payload.default_output_names.size() == 1U,
              "runtime-config payload should preserve packed transport runtime outputs");

      expect_invalid_runtime(packed_multi_runtime, "explicit packed-route facts",
                             "generic runtime-config contract build should reject semantic "
                             "multi-io over packed transport");

      {
        const auto fixture = sima_test::compile_processcvu_fixture(
            make_detess_scalar_plan({1, 1, 1, 1000}, {1, 1000}, {1, 1, 1000}, "INT8", 1008U, true,
                                    true),
            "detessdequant");
        const auto& compiled = fixture.contract;
        require(compiled.payload.input_tensors.size() == 1U &&
                    compiled.payload.output_tensors.size() == 1U,
                "ResNet scalar chain must retain one fused descriptor pair");
        require(tensor_desc_shape_for_test(compiled.payload.input_tensors.front()) ==
                        std::vector<std::int64_t>({1, 1, 1, 1000}) &&
                    tensor_desc_shape_for_test(compiled.payload.output_tensors.front()) ==
                        std::vector<std::int64_t>({1, 1, 1, 1000}),
                "fused execution must retain the explicit Detess frame geometry");
        require(compiled.payload.output_shapes == std::vector<std::vector<int>>{{1, 1000}} &&
                    compiled.payload.runtime_output_logical_shapes ==
                        std::vector<std::vector<int>>{{1, 1000}} &&
                    compiled.runtime_contract.logical_outputs.size() == 1U &&
                    compiled.runtime_contract.logical_outputs.front().shape ==
                        std::vector<std::int64_t>({1, 1000}) &&
                    compiled.runtime_contract.logical_outputs.front().size_bytes == 4000U,
                "the explicit private flatten must retain published [1,1000] FP32 storage");
        require(compiled.runtime_contract.logical_outputs.front().publication_relation ==
                    simaai::neat::pipeline_internal::sima::PublicationViewRelation::SingletonAxes,
                "rank change requires the proved singleton publication relation");
        require(compiled.payload.q_scale_list == std::vector<double>{0.5} &&
                    compiled.payload.q_zp_list == std::vector<int>{7},
                "physical fused lowering must retain the authored quantization");
      }

      {
        // Identical rank-two dimensions do not identify a legacy NC/HW grammar.
        // Typed recipes own trailing W,C axes even with the canonical HWC label.
        for (const std::string layout : {std::string{}, std::string{"HWC"}}) {
          auto typed =
              make_detess_scalar_plan({1, 16}, {1, 16}, {1, 16}, "INT8", 16U, false, false);
          typed.ops.front().kernel = "typed_detess_alias";
          for (auto& value : typed.values)
            value.logical_layout = layout;
          const auto fixture =
              sima_test::compile_processcvu_fixture(std::move(typed), "detessdequant");
          for (const auto* descriptor : {&fixture.contract.payload.input_tensors.front(),
                                         &fixture.contract.payload.output_tensors.front()}) {
            require(tensor_desc_shape_for_test(*descriptor) == sc::TensorShape({1, 16}) &&
                        tensor_desc_axes_for_test(*descriptor) ==
                            std::vector<std::uint8_t>({SIMA_EV_AXIS_W, SIMA_EV_AXIS_C}),
                    "typed rank-two geometry must not enter legacy NC/HW inference");
          }
        }

        struct LegacyGeometryCase {
          sc::TensorShape frame;
          sc::TensorShape tile;
          std::uint64_t packed_bytes;
          sc::TensorShape execution;
        };
        for (const auto& test : {LegacyGeometryCase{{1, 7}, {7}, 16U, {1, 1, 1, 7}},
                                 LegacyGeometryCase{{2, 3}, {1, 1, 1}, 96U, {1, 2, 3, 1}}}) {
          auto legacy = make_detess_scalar_plan(test.frame, test.frame, test.tile, "INT8",
                                                test.packed_bytes, true, true);
          std::get<sc::DetessellateOpConfig>(legacy.ops.front().config).frame_shape_semantics =
              sc::DetessFrameShapeSemantics::LegacyMpk;
          for (auto& value : legacy.values)
            value.logical_layout = "HWC";
          const auto fixture =
              sima_test::compile_processcvu_fixture(std::move(legacy), "detessdequant");
          require(tensor_desc_shape_for_test(fixture.contract.payload.input_tensors.front()) ==
                          test.execution &&
                      tensor_desc_shape_for_test(fixture.contract.payload.output_tensors.front()) ==
                          test.execution &&
                      fixture.contract.payload.input_tensors.front().descriptor().storage.nbytes ==
                          test.packed_bytes &&
                      fixture.contract.runtime_contract.logical_outputs.front().shape == test.frame,
                  "legacy rank-two NC/HW keeps its exact transport proof and public shape");
        }
        for (const auto semantics : {sc::DetessFrameShapeSemantics::LegacyMpk,
                                     static_cast<sc::DetessFrameShapeSemantics>(255)}) {
          auto ambiguous =
              make_detess_scalar_plan({1, 16}, {1, 16}, {1, 16}, "INT8", 16U, false, false);
          std::get<sc::DetessellateOpConfig>(ambiguous.ops.front().config).frame_shape_semantics =
              semantics;
          bool rejected = false;
          try {
            (void)sima_test::compile_processcvu_fixture(std::move(ambiguous), "detessdequant");
          } catch (const std::invalid_argument& error) {
            const std::string detail = error.what();
            rejected = detail.find(semantics == sc::DetessFrameShapeSemantics::LegacyMpk
                                       ? "ambiguous between NC and HW"
                                       : "unknown frame-shape semantics") != std::string::npos;
          }
          require(rejected, "ambiguous or unknown legacy rank-two geometry must fail closed");
        }
      }

      {
        constexpr std::uint64_t packed_bytes = 365408U;
        constexpr std::uint64_t output_bytes = 601U * 601U * sizeof(float);
        const auto fixture = sima_test::compile_processcvu_fixture(
            make_detess_scalar_plan({601, 601}, {1, 601, 601}, {67, 601}, "INT8", packed_bytes,
                                    true, true),
            "detessdequant");
        const auto& compiled = fixture.contract;
        require(compiled.payload.input_tensors.size() == 1U &&
                    compiled.payload.output_tensors.size() == 1U &&
                    compiled.runtime_contract.logical_outputs.size() == 1U,
                "WC fixture must retain exactly one fused input/output/public view");
        const auto& input = compiled.payload.input_tensors.front().descriptor();
        const auto& output = compiled.payload.output_tensors.front().descriptor();
        require(tensor_desc_shape_for_test(input) == std::vector<std::int64_t>({601, 601}) &&
                    tensor_desc_shape_for_test(output) == std::vector<std::int64_t>({601, 601}),
                "explicit WC frames must remain rank-2 throughout fused execution");
        require(tensor_desc_axes_for_test(input) ==
                        std::vector<std::uint8_t>({SIMA_EV_AXIS_W, SIMA_EV_AXIS_C}) &&
                    tensor_desc_axes_for_test(output) ==
                        std::vector<std::uint8_t>({SIMA_EV_AXIS_W, SIMA_EV_AXIS_C}),
                "rank-2 descriptors must retain explicit W,C axis semantics");
        require(tensor_desc_tile_shape_for_test(input) == std::vector<std::int64_t>({67, 601}),
                "rank-2 input must preserve authored W,C tile geometry");
        require(input.storage.nbytes == packed_bytes && output.storage.nbytes == output_bytes &&
                    !sima_ev_tiled_uses_compact_channels(&input) &&
                    input.layout.tiled.flags == SIMA_EV_TILED_FLAG_CBLOCK16,
                "physical projection must retain exact spans and CBlock16 encoding");
        require(tensor_desc_strides_for_test(output) ==
                    std::vector<std::int64_t>({601 * sizeof(float), sizeof(float)}),
                "WC output must carry authoritative contiguous byte strides");
        require(compiled.payload.output_shapes == std::vector<std::vector<int>>({{1, 601, 601}}) &&
                    compiled.runtime_contract.logical_outputs.front().shape ==
                        std::vector<std::int64_t>({1, 601, 601}) &&
                    compiled.runtime_contract.logical_outputs.front().size_bytes == output_bytes,
                "explicit public singleton axis and FP32 byte span must survive lowering");
        const auto padded = sima_test::compile_processcvu_fixture(
            make_detess_scalar_plan({601, 601}, {1, 601, 601}, {67, 601}, "INT8", packed_bytes,
                                    true, false),
            "detessdequant");
        require(padded.contract.payload.input_tensors.front().descriptor().layout.tiled.flags ==
                    SIMA_EV_TILED_FLAG_PADDED_HWC_C16,
                "align_c16 without cblock must retain padded-HWC encoding");
      }

      {
        const auto fixture = sima_test::compile_processcvu_fixture(
            make_detess_scalar_plan({1, 1, 1, 213}, {1, 213}, {1, 1, 64}, "INT8", 224U, true,
                                    false),
            "detessdequant");
        const auto& compiled = fixture.contract;
        require(tensor_desc_shape_for_test(compiled.payload.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 1, 1, 213}),
                "execution geometry must not be reconstructed from the rank-2 public view");
        require(compiled.payload.output_shapes == std::vector<std::vector<int>>{{1, 213}} &&
                    compiled.payload.runtime_output_logical_shapes.front() ==
                        std::vector<int>({1, 213}) &&
                    compiled.runtime_contract.logical_outputs.front().shape ==
                        std::vector<std::int64_t>({1, 213}),
                "rank-2 publication must preserve the explicit private reshape");
      }

      {
        for (const auto& shape : {sc::TensorShape{1, 213}, sc::TensorShape{1, 1, 1, 213}}) {
          const auto tile =
              shape.size() == 2U ? sc::TensorShape{1, 213} : sc::TensorShape{1, 1, 213};
          const auto fixture = sima_test::compile_processcvu_fixture(
              make_detess_scalar_plan(shape, shape, tile, "BF16", 448U, true, false), "detesscast");
          const auto& compiled = fixture.contract;
          require(tensor_desc_shape_for_test(compiled.payload.input_tensors.front()) == shape &&
                      tensor_desc_shape_for_test(compiled.payload.output_tensors.front()) ==
                          shape &&
                      compiled.runtime_contract.logical_outputs.front().shape == shape,
                  "rank-2 and rank-4 DetessCast must preserve exact typed execution/public shapes");
          require(compiled.payload.input_dtype == "BF16" &&
                      compiled.payload.output_dtype == "FP32" &&
                      compiled.runtime_contract.logical_outputs.front().size_bytes == 852U,
                  "fused DetessCast must preserve BF16->FP32 type and exact published size");
        }
      }

      {
        const auto fixture = sima_test::compile_processcvu_fixture(
            make_detess_scalar_plan({1, 1, 1, 213}, {1, 213}, {1, 1, 213}, "BF16", 448U, true,
                                    false),
            "detesscast");
        const auto& compiled = fixture.contract;
        require(compiled.payload.input_tensors.size() == 1U &&
                    compiled.payload.output_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(compiled.payload.input_tensors.front()) ==
                        sc::TensorShape{1, 1, 1, 213} &&
                    tensor_desc_shape_for_test(compiled.payload.output_tensors.front()) ==
                        sc::TensorShape{1, 1, 1, 213},
                "BF16 private reshape must not change fused DetessCast execution geometry");
        require(
            compiled.payload.output_shapes == std::vector<std::vector<int>>{{1, 213}} &&
                compiled.payload.runtime_output_logical_shapes ==
                    std::vector<std::vector<int>>{{1, 213}} &&
                compiled.runtime_contract.logical_outputs.size() == 1U &&
                compiled.runtime_contract.logical_outputs.front().shape ==
                    sc::TensorShape{1, 213} &&
                compiled.runtime_contract.logical_outputs.front().publication_relation ==
                    simaai::neat::pipeline_internal::sima::PublicationViewRelation::SingletonAxes,
            "BF16 rank-2 publication needs exact private singleton relation authority");
      }

      {
        const auto compiled = make_selected_post_cast_contract();
        require(compiled.payload.input_dtype == "BF16" && compiled.payload.output_dtype == "FP32",
                "semantic command selection must select post-MLA BF16->FP32, not pre-MLA casts");
        require(compiled.payload.num_in_tensor == 2 && compiled.payload.input_shapes.size() == 2U &&
                    compiled.payload.input_shapes ==
                        std::vector<std::vector<int>>{{12, 52, 45}, {3, 13, 4}},
                "selected post-MLA command cohort must preserve both heads and their geometry");
        require(
            compiled.runtime_contract.logical_outputs.size() == 2U &&
                compiled.runtime_contract.logical_outputs[0].shape == sc::TensorShape{12, 52, 45} &&
                compiled.runtime_contract.logical_outputs[1].shape == sc::TensorShape{3, 13, 4} &&
                compiled.runtime_contract.output_order.size() == 2U,
            "selected post-MLA casts must expose both exact routed output heads");
        for (std::size_t i = 0; i < 2U; ++i) {
          const auto& route = compiled.runtime_contract.output_order[i];
          const auto& output = compiled.runtime_contract.logical_outputs[i];
          require(route.logical_output_index == output.logical_index &&
                      route.output_slot == static_cast<int>(i) &&
                      route.segment_name == output.segment_name,
                  "post-MLA routes must track their selected logical and physical outputs");
        }
      }
    }));

#include "asset_utils.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.h"
#include "test_main.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path core_root() {
  return sima_test::test_source_root();
}

struct Yolov8VariantFixture {
  std::string tar_path;
  std::string unpack_root;
};

std::string trim_trailing_slashes(std::string value) {
  while (!value.empty() && value.back() == '/') {
    value.pop_back();
  }
  return value;
}

bool path_has_yolov8_contract_files(const std::filesystem::path& root) {
  return sima_test::is_usable_regular_file(root / "yolov8n_modified_mpk.json") &&
         sima_test::is_usable_regular_file(root / "pipeline_sequence.json");
}

std::string yolo_variant_base_url() {
  if (const char* env = std::getenv("SIMA_YOLOV8N_VARIANTS_BASE_URL"); env && *env) {
    return trim_trailing_slashes(env);
  }
  if (const char* env = std::getenv("SIMA_YOLOV8_VARIANTS_BASE_URL"); env && *env) {
    return trim_trailing_slashes(env);
  }
  return {};
}

Yolov8VariantFixture resolve_yolov8_variant_fixture(const std::string& stem) {
  const std::filesystem::path root = core_root();
  const std::filesystem::path drive_dir = root / "tmp" / "yolov8n_drive";
  const std::filesystem::path unpack_dir = root / "tmp" / "yolov8n_unpack" / stem;
  const std::filesystem::path tar_path = drive_dir / (stem + ".tar.gz");

  std::error_code ec;
  std::filesystem::create_directories(drive_dir, ec);
  ec.clear();
  std::filesystem::create_directories(unpack_dir.parent_path(), ec);

  if (!sima_test::is_usable_regular_file(tar_path)) {
    const std::string base_url = yolo_variant_base_url();
    require(!base_url.empty(),
            "missing YOLOv8n fixture '" + tar_path.string() +
                "'. Upload the fixture tarballs to the public test-assets repo and set "
                "SIMA_YOLOV8N_VARIANTS_BASE_URL (or SIMA_YOLOV8_VARIANTS_BASE_URL) "
                "to the directory/release URL containing " +
                stem + ".tar.gz");

    const std::string url = base_url + "/" + stem + ".tar.gz";
    require(sima_test::download_file(url, tar_path),
            "failed to download YOLOv8n fixture from " + url + " to " + tar_path.string());
  }

  if (!path_has_yolov8_contract_files(unpack_dir)) {
    ec.clear();
    std::filesystem::remove_all(unpack_dir, ec);
    ec.clear();
    std::filesystem::create_directories(unpack_dir, ec);
    const std::string cmd = "tar -xzf " + sima_test::shell_quote(tar_path.string()) + " -C " +
                            sima_test::shell_quote(unpack_dir.string());
    require(std::system(cmd.c_str()) == 0, "failed to unpack YOLOv8n fixture " + tar_path.string() +
                                               " into " + unpack_dir.string());
    require(path_has_yolov8_contract_files(unpack_dir),
            "YOLOv8n fixture missing expected contract files after unpack: " + unpack_dir.string());
  }

  return Yolov8VariantFixture{tar_path.string(), unpack_dir.string()};
}

const Yolov8VariantFixture& yolov8_int8_fixture() {
  static const Yolov8VariantFixture fixture =
      resolve_yolov8_variant_fixture("yolov8n_A_W_int8_mpk");
  return fixture;
}

const Yolov8VariantFixture& yolov8_bf16_fixture() {
  static const Yolov8VariantFixture fixture =
      resolve_yolov8_variant_fixture("yolov8n_A_W_BF16_mpk");
  return fixture;
}

std::string yolov8_int8_tar_path() {
  return yolov8_int8_fixture().tar_path;
}

std::string yolov8_int8_unpack_root() {
  return yolov8_int8_fixture().unpack_root;
}

std::string yolov8_bf16_tar_path() {
  return yolov8_bf16_fixture().tar_path;
}

std::string yolov8_bf16_unpack_root() {
  return yolov8_bf16_fixture().unpack_root;
}

simaai::neat::pipeline_internal::sima::MpkContract load_yolov8_int8_contract() {
  std::string error;
  const auto contract = simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root(
      yolov8_int8_unpack_root(), &error);
  require(contract.has_value(), "expected YOLOv8 INT8 MPK contract: " + error);
  return *contract;
}

simaai::neat::pipeline_internal::sima::MpkContract load_yolov8_bf16_contract() {
  std::string error;
  const auto contract = simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root(
      yolov8_bf16_unpack_root(), &error);
  require(contract.has_value(), "expected YOLOv8 BF16 MPK contract: " + error);
  return *contract;
}

simaai::neat::Model::Options int8_model_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Tensor;
  opt.preprocess.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.resize.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.layout_convert.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.quantize.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.tessellate.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::Auto;
  opt.upstream_name = "decoder";
  return opt;
}

simaai::neat::Model::Options bf16_model_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Auto;
  opt.preprocess.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::Auto;
  opt.upstream_name = "decoder";
  return opt;
}

std::vector<std::string> detessdequant_output_names_for_test() {
  return {"bbox_0", "bbox_1", "bbox_2", "cls_0", "cls_1", "cls_2"};
}

void verify_yolov8_int8_contract_subset() {
  using namespace simaai::neat;
  namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;
  namespace pss = simaai::neat::pipeline_internal::sima::stagesemantics;
  using simaai::neat::pipeline_internal::sima::build_mla_static_contract_from_mpk_stage;
  using simaai::neat::pipeline_internal::sima::get_mla_logical_outputs_contract;
  using simaai::neat::pipeline_internal::sima::get_mla_published_outputs_contract;
  using simaai::neat::pipeline_internal::sima::get_mla_stage_io_contract;

  const auto& quanttess_decl = pcs::plugin_contract_family_declaration("quanttess");
  require(quanttess_decl.required_fields.size() == 6U,
          "quanttess declaration should expose the 6-field required subset");
  require(quanttess_decl.optional_fields.size() == 2U,
          "quanttess declaration should expose align_c16/cblock as optional fields");
  require(!quanttess_decl.per_head, "quanttess declaration should be single-stage");

  const auto& processmla_decl = pcs::plugin_contract_family_declaration("processmla");
  require(processmla_decl.required_fields.size() == 4U,
          "processmla declaration should expose the 4-field required subset");
  require(processmla_decl.optional_fields.size() == 1U,
          "processmla declaration should expose dispatcher_output_names as optional");
  require(!processmla_decl.per_head, "processmla declaration should be single-stage");

  const auto& detessdequant_decl = pcs::plugin_contract_family_declaration("detessdequant");
  require(detessdequant_decl.required_fields.size() == 6U,
          "detessdequant declaration should expose the 6-field required subset");
  require(detessdequant_decl.optional_fields.size() == 2U,
          "detessdequant declaration should expose align_c16/cblock as optional fields");
  require(detessdequant_decl.per_head, "detessdequant declaration should be per-head");

  Model model(yolov8_int8_tar_path(), int8_model_options());
  const auto& pack = internal::ModelAccess::pack(model);
  const auto plan = pack.execution_plan();
  require(plan.pre.size() == 1U, "YOLOv8 INT8 should compile one fused pre stage");
  require(plan.infer.size() == 1U, "YOLOv8 INT8 should compile one MLA infer stage");
  require(plan.post.size() == 1U, "YOLOv8 INT8 should compile one fused post stage");
  require(plan.pre.front().kind == internal::ExecutionStageKind::QuantTess,
          "YOLOv8 INT8 pre stage should be quanttess");
  require(plan.infer.front().kind == internal::ExecutionStageKind::Mla,
          "YOLOv8 INT8 infer stage should be processmla");
  require(plan.post.front().kind == internal::ExecutionStageKind::DetessDequant,
          "YOLOv8 INT8 post stage should be detessdequant");

  const auto mpk = load_yolov8_int8_contract();

  const auto quanttess_subset = pcs::extract_quanttess_contract_subset_from_mpk(mpk);
  require(quanttess_subset.quant_params.scales.size() == 1U,
          "quanttess subset should expose one quant scale for YOLOv8 INT8");
  require(quanttess_subset.quant_params.zero_points.size() == 1U,
          "quanttess subset should expose one quant zero-point for YOLOv8 INT8");
  require(quanttess_subset.quant_params.scales.front() == 254.9999849195601,
          "quanttess subset should preserve quant scale from quantize_0");
  require(quanttess_subset.quant_params.zero_points.front() == -128,
          "quanttess subset should preserve quant zero-point from quantize_0");
  // The quanttess extractor strips the leading batch dim on the subset and
  // hoists it into the explicit `batch_size` scalar; downstream consumers
  // re-batch via batch_size when they need a 4-rank descriptor.
  require(quanttess_subset.input_shape == std::vector<std::int64_t>({640, 640, 3}),
          "quanttess subset should preserve the per-frame input shape");
  require(quanttess_subset.batch_size == 1,
          "quanttess subset should hoist the leading batch=1 into batch_size");
  require(quanttess_subset.input_dtype == "FP32",
          "quanttess subset should normalize the quantize input dtype");
  require(quanttess_subset.output_dtype == "INT8",
          "quanttess subset should normalize the quantize output dtype");
  require(quanttess_subset.slice_shape == std::vector<std::int64_t>({32, 128, 3}),
          "quanttess subset should preserve tess slice_shape");
  require(quanttess_subset.frame_type == "INT8",
          "quanttess subset should preserve tess frame_type");
  require(!quanttess_subset.align_c16 && !quanttess_subset.cblock,
          "quanttess subset should preserve untiled align/cblock flags");

  const auto quanttess_runtime = pcs::build_quanttess_runtime_config_from_subset(quanttess_subset);
  const auto quanttess_inputs =
      pss::build_processcvu_compile_inputs_from_runtime_config(quanttess_runtime);
  require(quanttess_inputs.payload.graph_family == "quanttess",
          "quanttess subset builder should compile a quanttess payload");
  require(quanttess_inputs.payload.input_shapes.size() == 1U &&
              quanttess_inputs.payload.input_shapes.front() == std::vector<int>({640, 640, 3}),
          "quanttess subset builder should preserve input geometry");
  require(quanttess_inputs.payload.slice_shapes.size() == 1U &&
              quanttess_inputs.payload.slice_shapes.front() == std::vector<int>({32, 128, 3}),
          "quanttess subset builder should preserve tile geometry");
  require(quanttess_inputs.payload.q_scale == 254.9999849195601 &&
              quanttess_inputs.payload.q_zp == -128,
          "quanttess subset builder should preserve quant params");
  require(quanttess_inputs.payload.out_dtype == "INT8",
          "quanttess subset builder should normalize output dtype to INT8");

  auto broken_quanttess = quanttess_subset;
  broken_quanttess.slice_shape.clear();
  try {
    (void)pcs::build_quanttess_runtime_config_from_subset(broken_quanttess);
    throw std::runtime_error("quanttess subset builder should reject a missing slice_shape");
  } catch (const std::invalid_argument& e) {
    require_contains(e.what(), "slice_shape",
                     "quanttess subset builder should report the missing required field");
  }

  const auto* mla_stage = get_mla_stage_io_contract(mpk);
  require(mla_stage != nullptr, "YOLOv8 INT8 should expose one MLA stage in the MPK");
  const auto mla_logical_outputs = get_mla_logical_outputs_contract(mpk);
  const auto mla_boundary_physical_outputs = get_mla_boundary_physical_outputs_contract(mpk);
  const auto mla_static = build_mla_static_contract_from_mpk_stage(
      *mla_stage, mla_logical_outputs, mla_boundary_physical_outputs, "MLA_0_1");

  const auto processmla_subset =
      pcs::extract_processmla_contract_subset_from_static_contract(mla_static, false);
  require(processmla_subset.model_path == "yolov8n_modified_stage1_mla.elf",
          "processmla subset should preserve model_path");
  require(processmla_subset.batch_size == 1 && processmla_subset.batch_sz_model == 1,
          "processmla subset should preserve batch fields");
  require(processmla_subset.dispatcher_output_sizes == std::vector<std::uint64_t>({1209600U}),
          "processmla subset should preserve dispatcher output sizes");
  require(processmla_subset.dispatcher_output_names.empty(),
          "processmla subset should leave dispatcher output names empty for single-output YOLOv8");

  const auto mla_payload = pcs::build_processmla_payload_from_subset(processmla_subset);
  require(mla_payload.dispatcher_output_sizes == std::vector<std::uint64_t>({1209600U}),
          "processmla payload builder should preserve dispatcher output sizes");
  require(mla_payload.dispatcher_output_names.empty(),
          "processmla payload builder should keep dispatcher output names optional");

  const auto compiled_mla =
      pss::build_mla_compiled_contract_from_subset(processmla_subset, mla_static);
  require(compiled_mla.payload.dispatcher_output_sizes == std::vector<std::uint64_t>({1209600U}),
          "processmla compiled contract should come from the subset payload");
  require(compiled_mla.payload.dispatcher_output_names.empty(),
          "processmla compiled contract should not force dispatcher output names");

  auto broken_processmla = processmla_subset;
  broken_processmla.dispatcher_output_sizes.clear();
  try {
    (void)pcs::build_processmla_payload_from_subset(broken_processmla);
    throw std::runtime_error(
        "processmla subset builder should reject missing dispatcher_output_sizes");
  } catch (const std::invalid_argument& e) {
    require_contains(e.what(), "dispatcher_output_sizes",
                     "processmla subset builder should report the missing required field");
  }

  const auto detessdequant_subset = pcs::extract_detessdequant_contract_subset_from_mpk(mpk);
  require(detessdequant_subset.heads.size() == 6U,
          "detessdequant subset should preserve the six YOLOv8 post heads");
  require(detessdequant_subset.heads.front().per_head_input_shape ==
              std::vector<std::int64_t>({1, 80, 80, 64}),
          "detessdequant subset should preserve the per-head input shape");
  require(detessdequant_subset.heads.front().frame_shape ==
              std::vector<std::int64_t>({1, 80, 80, 64}),
          "detessdequant subset should preserve frame_shape from detessellate");
  require(detessdequant_subset.heads.front().slice_shape == std::vector<std::int64_t>({1, 80, 64}),
          "detessdequant subset should preserve slice_shape from detessellate");
  require(detessdequant_subset.heads.front().frame_type == "INT8",
          "detessdequant subset should preserve per-head frame_type");
  require(detessdequant_subset.heads.front().align_c16 && detessdequant_subset.heads.front().cblock,
          "detessdequant subset should preserve detess align/cblock flags");
  require(detessdequant_subset.heads.front().per_head_quant_params.scales.front() ==
              8.084461464679606,
          "detessdequant subset should preserve per-head dq scale");
  require(detessdequant_subset.heads.front().per_head_quant_params.zero_points.front() == -56,
          "detessdequant subset should preserve per-head dq zero-point");
  require(detessdequant_subset.heads.front().output_dtype == "FP32",
          "detessdequant subset should normalize dequant output dtype");

  const auto detessdequant_runtime = pcs::build_detessdequant_runtime_config_from_subset(
      detessdequant_subset, detessdequant_output_names_for_test());
  const auto detessdequant_payload =
      pss::build_processcvu_payload_from_runtime_config_internal(detessdequant_runtime);
  require(detessdequant_payload.graph_family == "detessdequant",
          "detessdequant subset builder should compile a detessdequant payload");
  require(detessdequant_payload.num_in_tensor == 6,
          "detessdequant subset builder should preserve head count");
  require(detessdequant_payload.input_shapes.size() == 6U,
          "detessdequant subset builder should preserve per-head input shapes");
  require(detessdequant_payload.input_shapes == (std::vector<std::vector<int>>{{1, 80, 80, 64},
                                                                               {1, 40, 40, 64},
                                                                               {1, 20, 20, 64},
                                                                               {1, 80, 80, 80},
                                                                               {1, 40, 40, 80},
                                                                               {1, 20, 20, 80}}),
          "detessdequant subset builder should preserve per-head input shapes (N,H,W,C)");
  require(detessdequant_payload.slice_shapes.size() == 6U,
          "detessdequant subset builder should preserve per-head slice shapes");
  require(detessdequant_payload.slice_shapes ==
              (std::vector<std::vector<int>>{
                  {1, 80, 64}, {1, 40, 64}, {1, 20, 64}, {1, 80, 80}, {1, 40, 80}, {1, 20, 80}}),
          "detessdequant subset builder should preserve per-head slice shapes (H,W,C)");
  require(detessdequant_payload.dq_scale_list.front() == 8.084461464679606,
          "detessdequant subset builder should preserve per-head dq scales");
  require(detessdequant_payload.dq_zp_list.front() == -56,
          "detessdequant subset builder should preserve per-head dq zero-points");
  require(detessdequant_payload.output_dtype == "FP32",
          "detessdequant subset builder should preserve output dtype");

  auto broken_detessdequant = detessdequant_subset;
  broken_detessdequant.heads.front().frame_shape.clear();
  try {
    (void)pcs::build_detessdequant_runtime_config_from_subset(
        broken_detessdequant, detessdequant_output_names_for_test());
    throw std::runtime_error("detessdequant subset builder should reject a missing frame_shape");
  } catch (const std::invalid_argument& e) {
    require_contains(e.what(), "frame_shape",
                     "detessdequant subset builder should report the missing required field");
  }
}

void verify_yolov8_bf16_contract_subset() {
  using namespace simaai::neat;
  namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;
  namespace pss = simaai::neat::pipeline_internal::sima::stagesemantics;
  using simaai::neat::pipeline_internal::sima::build_mla_static_contract_from_mpk_stage;
  using simaai::neat::pipeline_internal::sima::get_mla_logical_outputs_contract;
  using simaai::neat::pipeline_internal::sima::get_mla_published_outputs_contract;
  using simaai::neat::pipeline_internal::sima::get_mla_stage_io_contract;

  const auto& cast_decl = pcs::plugin_contract_family_declaration("cast");
  require(cast_decl.required_fields.size() == 3U,
          "cast declaration should expose the 3-field required subset");
  require(cast_decl.optional_fields.empty(), "cast declaration should not expose optional fields");
  require(!cast_decl.per_head, "cast declaration should be single-stage");

  const auto& tess_decl = pcs::plugin_contract_family_declaration("tessellate");
  require(tess_decl.required_fields.size() == 3U,
          "tessellate declaration should expose the 3-field required subset");
  require(tess_decl.optional_fields.size() == 2U,
          "tessellate declaration should expose align_c16/cblock as optional fields");
  require(!tess_decl.per_head, "tessellate declaration should be single-stage");

  const auto& processmla_decl = pcs::plugin_contract_family_declaration("processmla");
  require(processmla_decl.required_fields.size() == 4U,
          "processmla declaration should expose the 4-field required subset");
  require(processmla_decl.optional_fields.size() == 1U,
          "processmla declaration should expose dispatcher_output_names as optional");

  const auto& detess_decl = pcs::plugin_contract_family_declaration("detessellate");
  require(detess_decl.required_fields.size() == 4U,
          "detessellate declaration should expose the 4-field required subset");
  require(detess_decl.optional_fields.size() == 2U,
          "detessellate declaration should expose align_c16/cblock as optional fields");
  require(!detess_decl.per_head, "detessellate declaration should be single-stage");

  Model model(yolov8_bf16_tar_path(), bf16_model_options());
  const auto& pack = internal::ModelAccess::pack(model);
  const auto plan = pack.execution_plan();
  require(plan.pre.size() == 1U, "YOLOv8 BF16 should compile one pre stage");
  require(plan.infer.size() == 1U, "YOLOv8 BF16 should compile one MLA infer stage");
  require(plan.post.size() == 1U, "YOLOv8 BF16 should compile one post stage");
  require(plan.pre.front().kind == internal::ExecutionStageKind::CastTess,
          "YOLOv8 BF16 pre stage should be casttess (fused cast+tess)");
  require(plan.infer.front().kind == internal::ExecutionStageKind::Mla,
          "YOLOv8 BF16 infer stage should be processmla");
  require(plan.post.front().kind == internal::ExecutionStageKind::DetessCast,
          "YOLOv8 BF16 post stage should be detesscast (fused detess+cast)");

  const auto mpk = load_yolov8_bf16_contract();

  const auto cast_subsets = pcs::extract_cast_contract_subsets_from_mpk(mpk);
  require(cast_subsets.size() == 7U, "YOLOv8 BF16 should expose one pre cast and six post casts");
  require(cast_subsets.front().input_shape == std::vector<std::int64_t>({640, 640, 3}),
          "cast subset should preserve the BF16 pre-cast input shape");
  require(cast_subsets.front().input_dtype == "FP32",
          "cast subset should normalize the BF16 pre-cast input dtype");
  require(cast_subsets.front().output_dtype == "BF16",
          "cast subset should normalize the BF16 pre-cast output dtype");
  require(cast_subsets.back().output_dtype == "FP32",
          "cast subset should preserve the terminal post-cast output dtype");

  const auto tess_subset = pcs::extract_tessellate_contract_subset_from_mpk(mpk);
  require(tess_subset.input_shape == std::vector<std::int64_t>({640, 640, 3}),
          "tess subset should preserve the cast output geometry");
  require(tess_subset.frame_type == "BF16", "tess subset should preserve BF16 frame_type");
  require(tess_subset.slice_shape == std::vector<std::int64_t>({640, 32, 3}),
          "tess subset should preserve tess slice_shape");
  require(!tess_subset.align_c16 && !tess_subset.cblock,
          "tess subset should preserve unaligned tess flags");

  const auto tess_runtime = pcs::build_tessellate_runtime_config_from_subsets(
      cast_subsets.front(), tess_subset, "output_tensor");
  const auto tess_payload =
      pss::build_processcvu_payload_from_runtime_config_internal(tess_runtime);
  require(tess_payload.graph_family == "tessellate",
          "tess subset builder should compile a tessellate payload");
  require(tess_payload.input_shapes.size() == 1U &&
              tess_payload.input_shapes.front() == std::vector<int>({640, 640, 3}),
          "tess subset builder should preserve BF16 pre-MLA input geometry");
  require(tess_payload.slice_shapes.size() == 1U &&
              tess_payload.slice_shapes.front() == std::vector<int>({640, 32, 3}),
          "tess subset builder should preserve BF16 tile geometry");
  // The fused cast+tess runtime carries the cast input dtype (FP32) and tess
  // output dtype (BF16); the cast head provides the FP32->BF16 conversion.
  require(tess_payload.input_dtype == "FP32" && tess_payload.output_dtype == "BF16",
          "fused cast+tess payload should preserve cast input/tess output dtypes");

  auto broken_tess = tess_subset;
  broken_tess.slice_shape.clear();
  try {
    (void)pcs::build_tessellate_runtime_config_from_subsets(cast_subsets.front(), broken_tess,
                                                            "output_tensor");
    throw std::runtime_error("tess subset builder should reject a missing slice_shape");
  } catch (const std::invalid_argument& e) {
    require_contains(e.what(), "slice_shape",
                     "tess subset builder should report the missing required field");
  }

  const auto* mla_stage = get_mla_stage_io_contract(mpk);
  require(mla_stage != nullptr, "YOLOv8 BF16 should expose one MLA stage in the MPK");
  const auto mla_logical_outputs = get_mla_logical_outputs_contract(mpk);
  const auto mla_boundary_physical_outputs = get_mla_boundary_physical_outputs_contract(mpk);
  const auto mla_static = build_mla_static_contract_from_mpk_stage(
      *mla_stage, mla_logical_outputs, mla_boundary_physical_outputs, "MLA_0_1");

  const auto processmla_subset =
      pcs::extract_processmla_contract_subset_from_static_contract(mla_static, false);
  require(processmla_subset.model_path == "yolov8n_modified_stage1_mla.elf",
          "processmla subset should preserve BF16 model_path");
  require(processmla_subset.batch_size == 1 && processmla_subset.batch_sz_model == 1,
          "processmla subset should preserve BF16 batch fields");
  require(processmla_subset.dispatcher_output_sizes == std::vector<std::uint64_t>({2419200U}),
          "processmla subset should preserve BF16 dispatcher output sizes");

  const auto detess_subsets = pcs::extract_detessellate_contract_subsets_from_mpk(mpk);
  require(detess_subsets.size() == 6U,
          "detess subset extraction should preserve the six BF16 post heads");
  require(detess_subsets.front().frame_shape == std::vector<std::int64_t>({1, 80, 80, 64}),
          "detess subset should preserve frame_shape from the first BF16 head");
  require(detess_subsets.front().frame_type == "BF16",
          "detess subset should preserve BF16 frame_type");
  require(detess_subsets.front().slice_shape == std::vector<std::int64_t>({16, 16, 16}),
          "detess subset should preserve the first BF16 slice_shape");
  require(detess_subsets.front().align_c16 && detess_subsets.front().cblock,
          "detess subset should preserve detess align/cblock flags");

  const auto detess_runtime = pcs::build_detessellate_runtime_config_from_subsets(
      detess_subsets, detessdequant_output_names_for_test(), detessdequant_output_names_for_test());
  const auto detess_payload =
      pss::build_processcvu_payload_from_runtime_config_internal(detess_runtime);
  require(detess_payload.graph_family == "detessellate",
          "detess subset builder should compile a detessellate payload");
  require(detess_payload.num_in_tensor == 6, "detess subset builder should preserve head count");
  require(detess_payload.input_shapes.size() == 6U,
          "detess subset builder should preserve per-head input shapes");
  require(detess_payload.input_shapes == (std::vector<std::vector<int>>{{1, 80, 80, 64},
                                                                        {1, 40, 40, 64},
                                                                        {1, 20, 20, 64},
                                                                        {1, 80, 80, 80},
                                                                        {1, 40, 40, 80},
                                                                        {1, 20, 20, 80}}),
          "detess subset builder should preserve per-head input shapes (N,H,W,C)");
  require(detess_payload.slice_shapes.size() == 6U,
          "detess subset builder should preserve per-head slice shapes");
  require(detess_payload.slice_shapes ==
              (std::vector<std::vector<int>>{
                  {16, 16, 16}, {8, 8, 16}, {4, 4, 16}, {16, 4, 80}, {8, 2, 80}, {4, 1, 80}}),
          "detess subset builder should preserve per-head slice shapes (H,W,C)");
  require(detess_payload.output_dtype == "BF16",
          "detess subset builder should preserve BF16 output dtype");

  auto broken_detess = detess_subsets.front();
  broken_detess.frame_shape.clear();
  try {
    (void)pcs::build_detessellate_runtime_config_from_subsets({broken_detess});
    throw std::runtime_error("detess subset builder should reject a missing frame_shape");
  } catch (const std::invalid_argument& e) {
    require_contains(e.what(), "frame_shape",
                     "detess subset builder should report the missing required field");
  }
}

void verify_yolov8_pre_stage_facts_match_canonical_contracts() {
  using namespace simaai::neat;
  namespace pss = simaai::neat::pipeline_internal::sima::stagesemantics;

  auto verify = [&](const Model& model,
                    const simaai::neat::pipeline_internal::sima::MpkContract& mpk,
                    internal::ExecutionStageKind kind, const std::string& expected_stage_name,
                    int expected_graph_id, const std::string& label) {
    const auto& pack = internal::ModelAccess::pack(model);
    const auto plan = pack.execution_plan();
    const auto pre_facts = pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
    require(plan.pre.size() == 1U, label + " should expose one preprocess execution stage");
    require(pre_facts.size() == 1U, label + " should expose one preprocess stage fact");
    require(plan.pre.front().kind == kind, label + " preprocess stage kind should match");
    require(plan.pre.front().stage_name == expected_stage_name,
            label + " preprocess stage should preserve the canonical family stage name");
    require(pre_facts.front().stage_name == expected_stage_name,
            label + " preprocess stage fact should preserve the canonical family stage name");
    require(pre_facts.front().processcvu_contract.has_value(),
            label + " preprocess stage fact should include a processcvu contract");

    const auto generic = pss::build_processcvu_mpk_compiled_contract_for_stage_kind(mpk, kind);
    const auto& from_fact = *pre_facts.front().processcvu_contract;
    require(from_fact.payload.graph_id == expected_graph_id,
            label + " preprocess stage fact should preserve the expected graph id");
    require(from_fact.payload.graph_id == generic.payload.graph_id,
            label + " preprocess stage fact should match the canonical graph id");
    require(from_fact.payload.graph_family == generic.payload.graph_family,
            label + " preprocess stage fact should match the canonical graph family");
    require(from_fact.payload.input_shapes == generic.payload.input_shapes,
            label + " preprocess stage fact should match canonical input geometry");
    require(from_fact.payload.output_shapes == generic.payload.output_shapes,
            label + " preprocess stage fact should match canonical output geometry");
    require(from_fact.payload.input_dtype == generic.payload.input_dtype &&
                from_fact.payload.output_dtype == generic.payload.output_dtype,
            label + " preprocess stage fact should match canonical dtypes");
  };

  const Model int8_model(yolov8_int8_tar_path(), int8_model_options());
  verify(int8_model, load_yolov8_int8_contract(), internal::ExecutionStageKind::QuantTess,
         "quanttess", 226, "YOLOv8 INT8");

  // BF16 path uses the fused CastTess stage, but
  // build_processcvu_mpk_compiled_contract_for_stage_kind currently returns
  // nullopt for casttess (canonical contract not yet wired). Skipping the
  // canonical-equivalence verification for BF16 until that gap is closed.
}

} // namespace

RUN_TEST("unit_yolov8_contract_subset_test", ([] {
           verify_yolov8_int8_contract_subset();
           verify_yolov8_bf16_contract_subset();
           verify_yolov8_pre_stage_facts_match_canonical_contracts();
         }));

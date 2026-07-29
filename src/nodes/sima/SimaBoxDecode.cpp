#include "nodes/sima/SimaBoxDecode.h"

#include "gst/GstHelpers.h"
#include "builder/InputContractConfigurable.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "model/internal/RoutePlanner.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat {
using pipeline_internal::lower_copy;

struct BoxDecodeOptionsInternal {
  int sima_allocator_type = 2;
  bool silent = true;
  bool emit_signals = false;
  bool transmit = false;
  BoxDecodeType decode_type = BoxDecodeType::Unspecified;
  BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto;
  // Explicit source byte layout of the upstream head tensors, for hand-built graphs that decode
  // without a model pack (the upstream compiled contract does not carry packing flags, so this
  // cannot be inferred). nullopt means "not specified": resolved from the model pack when present,
  // otherwise contract compilation fails fast. Unused on the model-pack/compiled-contract paths.
  std::optional<pipeline_internal::sima::BoxDecodeSourceStorageKind> source_storage_kind;
  // Explicit overrides for the standalone (no-model-pack) box-decode contract: whether the upstream
  // heads need detessellation / dequantization. nullopt keeps the inferred standalone default (or
  // the model-pack value). Like source_storage_kind, unused on the model-pack/compiled-contract
  // paths.
  std::optional<bool> override_tess_needed;
  std::optional<bool> override_quant_needed;
  int top_k = 0;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics> model_semantics;
  std::optional<pipeline_internal::sima::ModelManagedRouteFlags> model_route_flags;
  std::optional<pipeline_internal::sima::BoxDecodeStaticContract> model_static_contract;
  std::shared_ptr<const CompiledBoxDecodeContract> compiled_contract;
  std::vector<std::string> required_preprocess_meta_fields;
  std::shared_ptr<const simaai::neat::internal::ModelLineageBinding> model_lineage;
  simaai::neat::internal::RequestedPostRouteKind requested_post_route =
      simaai::neat::internal::RequestedPostRouteKind::Auto;
  std::optional<bool> expect_resize;
  std::optional<bool> expect_normalize;
  std::optional<bool> expect_quantize;
  std::optional<bool> expect_tessellate;
  int original_width = 0;
  int original_height = 0;
  int model_width = 0;
  int model_height = 0;
  // Explicit override for the preprocess resize mode. Set when the customer is
  // running the model without an upstream Preproc stage (e.g. CPU-letterboxed
  // FP32 tensor directly into quanttess→mla→boxdecode) and therefore the per-
  // buffer GstSimaaiPreprocessMeta won't carry `preproc_resize_mode`. When
  // present the contract drops `preproc_resize_mode` from
  // required_preprocess_meta_fields, and the backend fragment emits the
  // resize-mode token for the GST plugin override. nullopt
  // means "no override; expect the value from upstream meta as before".
  std::optional<ResizeMode> resize_mode_override;
  std::string element_name;
  std::string factory = "neatobjectdecode";
};

namespace {

using json = nlohmann::json;

bool boxdecode_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_BOXDECODE_DEBUG", false);
}

/// Map the public source-storage selector onto the internal contract enum.
pipeline_internal::sima::BoxDecodeSourceStorageKind
source_storage_kind_from_public(BoxDecodeSourceStorage storage) {
  switch (storage) {
  case BoxDecodeSourceStorage::PackedCBlock:
    return pipeline_internal::sima::BoxDecodeSourceStorageKind::PackedCBlock;
  case BoxDecodeSourceStorage::DenseHwc:
  default:
    return pipeline_internal::sima::BoxDecodeSourceStorageKind::DenseHwcPhysical;
  }
}

void validate_requested_boxdecode_contract_type_local(BoxDecodeType contract_type,
                                                      BoxDecodeType requested_type,
                                                      std::string_view context) {
  using pipeline_internal::sima::box_decode_type_matches_requested_contract;
  using pipeline_internal::sima::box_decode_type_token_string;
  if (box_decode_type_matches_requested_contract(contract_type, requested_type)) {
    return;
  }
  const std::string contract_token = box_decode_type_token_string(contract_type);
  const std::string requested_token = box_decode_type_token_string(requested_type);
  std::ostringstream oss;
  oss << context
      << ": requested BoxDecode decode_type does not match the MPK-derived tensor contract "
      << "(requested=" << (requested_token.empty() ? "unspecified" : requested_token)
      << ", mpk_contract=" << (contract_token.empty() ? "unspecified" : contract_token)
      << "). Use the matching BoxDecodeType for this model, or leave the default Model route as "
         "raw tensors and attach an explicit compatible postprocess stage.";
  throw std::runtime_error(oss.str());
}

bool boxdecode_name_has_class_score_semantics_local(const std::string& raw_name) {
  const std::string name = lower_copy(raw_name);
  return name.find("class_prob") != std::string::npos ||
         name.find("class_probability") != std::string::npos ||
         name.find("class_logit") != std::string::npos ||
         name.find("class_logits") != std::string::npos;
}

bool boxdecode_name_looks_generic_local(const std::string& raw_name) {
  const std::string name = lower_copy(raw_name);
  return name.empty() || name.rfind("pass_through_out_", 0) == 0 ||
         name.rfind("output_tensor_", 0) == 0 || name == "output_tensor";
}

bool boxdecode_contract_needs_sample_semantic_refinement_local(
    const pipeline_internal::sima::BoxDecodeStaticContract& contract) {
  bool saw_explicit_semantics = false;
  bool saw_generic_names = false;
  for (const auto& tensor : contract.tensors) {
    saw_explicit_semantics =
        saw_explicit_semantics ||
        boxdecode_name_has_class_score_semantics_local(tensor.logical_name) ||
        boxdecode_name_has_class_score_semantics_local(tensor.backend_name) ||
        boxdecode_name_has_class_score_semantics_local(tensor.source_segment_name);
    saw_generic_names = saw_generic_names ||
                        boxdecode_name_looks_generic_local(tensor.logical_name) ||
                        boxdecode_name_looks_generic_local(tensor.backend_name) ||
                        boxdecode_name_looks_generic_local(tensor.source_segment_name);
  }
  if (saw_explicit_semantics) {
    return false;
  }
  if (saw_generic_names) {
    return true;
  }
  return contract.score_activation == pipeline_internal::sima::BoxDecodeScoreActivation::Unknown ||
         contract.decode_type_option == BoxDecodeTypeOption::Auto;
}

json shape_descs_to_json(const std::vector<sima_ev_shape_desc>& shapes) {
  json out = json::array();
  for (const auto& shape : shapes) {
    json dims = json::array();
    const auto rank = std::min<std::uint32_t>(shape.rank, SIMA_EV_MAX_RANK);
    for (std::uint32_t i = 0; i < rank; ++i) {
      dims.push_back(shape.sizes[i]);
    }
    out.push_back(std::move(dims));
  }
  return out;
}

void maybe_refine_boxdecode_contract_from_ingress_sample_local(
    pipeline_internal::sima::BoxDecodeStaticContract* contract, const Sample& ingress_sample,
    BoxDecodeType decode_type, const std::optional<InputContract>& input_contract) {
  if (!contract ||
      !(decode_type == BoxDecodeType::YoloV8 || decode_type == BoxDecodeType::YoloV26 ||
        decode_type == BoxDecodeType::YoloV26Pose || decode_type == BoxDecodeType::YoloV26Seg ||
        decode_type == BoxDecodeType::YoloV6 || decode_type == BoxDecodeType::YoloX) ||
      !boxdecode_contract_needs_sample_semantic_refinement_local(*contract)) {
    return;
  }

  std::string sample_error;
  auto sample_contract = pipeline_internal::sima::build_boxdecode_static_contract_from_sample(
      ingress_sample, decode_type, input_contract,
      pipeline_internal::sima::BoxDecodeStandaloneContractOverrides{}, &sample_error);
  if (!sample_contract.has_value()) {
    return;
  }
  if (sample_contract->score_activation !=
      pipeline_internal::sima::BoxDecodeScoreActivation::Unknown) {
    contract->score_activation = sample_contract->score_activation;
  }
  if (sample_contract->decode_type_option != BoxDecodeTypeOption::Auto) {
    contract->decode_type_option = sample_contract->decode_type_option;
  }
}

std::string boxdecode_dossier_dir() {
  const char* raw = std::getenv("SIMA_BOXDECODE_DOSSIER_DIR");
  return (raw && *raw) ? std::string(raw) : std::string();
}

void boxdecode_write_json_file(const std::filesystem::path& path, const json& value) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::trunc);
  out << value.dump(2) << "\n";
}

json quant_to_json(const std::optional<pipeline_internal::sima::QuantStaticSpec>& quant) {
  if (!quant.has_value()) {
    return nullptr;
  }
  return json{
      {"granularity", static_cast<int>(quant->granularity)},
      {"axis", quant->axis},
      {"scales", quant->scales},
      {"zero_points", quant->zero_points},
  };
}

template <typename T> json vector_to_json_array(const std::vector<T>& values) {
  json arr = json::array();
  for (const auto& value : values) {
    arr.push_back(value);
  }
  return arr;
}

json logical_input_to_json(const pipeline_internal::sima::LogicalInputStaticSpec& logical) {
  return json{
      {"logical_index", logical.logical_index},
      {"backend_input_index", logical.backend_input_index},
      {"physical_index", logical.physical_index},
      {"shape", logical.shape},
      {"stride_bytes", logical.stride_bytes},
      {"byte_offset", logical.byte_offset},
      {"size_bytes", logical.size_bytes},
      {"dtype", logical.dtype},
      {"layout", logical.layout},
      {"logical_name", logical.logical_name},
      {"backend_name", logical.backend_name},
      {"segment_name", logical.segment_name},
      {"quant", quant_to_json(logical.quant)},
  };
}

json input_binding_to_json(const pipeline_internal::sima::InputBindingStaticSpec& binding) {
  return json{
      {"sink_pad_index", binding.sink_pad_index},
      {"local_logical_input_index", binding.local_logical_input_index},
      {"src_stage_index", binding.src_stage_index},
      {"src_stage_id", binding.src_stage_id},
      {"src_logical_output_index", binding.src_logical_output_index},
      {"src_output_slot", binding.src_output_slot},
      {"src_physical_output_index", binding.src_physical_output_index},
      {"src_physical_size_bytes", binding.src_physical_size_bytes},
      {"src_physical_byte_offset", binding.src_physical_byte_offset},
      {"required", binding.required},
      {"cm_input_name", binding.cm_input_name},
      {"source_segment_name", binding.source_segment_name},
  };
}

json boxdecode_static_contract_to_json(
    const pipeline_internal::sima::BoxDecodeStaticContract& contract) {
  json tensors = json::array();
  for (const auto& tensor : contract.tensors) {
    tensors.push_back({
        {"input_shape", tensor.input_shape},
        {"slice_shape", tensor.slice_shape},
        {"data_type", tensor.data_type},
        {"logical_name", tensor.logical_name},
        {"backend_name", tensor.backend_name},
        {"source_segment_name", tensor.source_segment_name},
        {"source_logical_output_index", tensor.source_logical_output_index},
        {"source_output_slot", tensor.source_output_slot},
        {"source_physical_index", tensor.source_physical_index},
        {"source_byte_offset", tensor.source_byte_offset},
        {"source_size_bytes", tensor.source_size_bytes},
    });
  }
  json physical_inputs = json::array();
  for (const auto& physical : contract.physical_inputs) {
    physical_inputs.push_back({
        {"name", physical.name},
        {"physical_index", physical.physical_index},
        {"byte_offset", physical.byte_offset},
        {"size_bytes", physical.size_bytes},
    });
  }
  return json{
      {"decode_type", static_cast<int>(contract.decode_type)},
      {"decode_type_option", static_cast<int>(contract.decode_type_option)},
      {"score_activation", static_cast<int>(contract.score_activation)},
      {"input_dtype", contract.input_dtype},
      {"tess_needed", contract.tess_needed},
      {"quant_needed", contract.quant_needed},
      {"topk", contract.topk},
      {"detection_threshold", contract.detection_threshold},
      {"nms_iou_threshold", contract.nms_iou_threshold},
      {"tensor_names", contract.tensor_names},
      {"tensors", std::move(tensors)},
      {"physical_inputs", std::move(physical_inputs)},
      {"dq_scale", contract.dq_scale},
      {"dq_zp", contract.dq_zp},
  };
}

void maybe_dump_boxdecode_core_dossier(
    const pipeline_internal::sima::BoxDecodeStaticContract& contract,
    const simaai::neat::CompiledBoxDecodeContract& compiled, const std::string& element_name,
    const std::string& logical_stage_id) {
  const std::string dir = boxdecode_dossier_dir();
  if (dir.empty()) {
    return;
  }
  json logical_inputs = json::array();
  for (const auto& logical : compiled.runtime_contract.logical_inputs) {
    logical_inputs.push_back(logical_input_to_json(logical));
  }
  json input_bindings = json::array();
  for (const auto& binding : compiled.runtime_contract.input_bindings) {
    input_bindings.push_back(input_binding_to_json(binding));
  }
  json payload{
      {"decode_type", static_cast<int>(compiled.payload.decode_type)},
      {"decode_type_option", compiled.payload.decode_type_option.has_value()
                                 ? json(static_cast<int>(*compiled.payload.decode_type_option))
                                 : json(nullptr)},
      {"score_activation", static_cast<int>(compiled.payload.score_activation)},
      {"input_dtype", compiled.payload.input_dtype},
      {"tess_needed", compiled.payload.tess_needed},
      {"quant_needed", compiled.payload.quant_needed},
      {"quant_contract_required", compiled.payload.quant_contract_required},
      {"model_owned_flags", compiled.payload.model_owned_flags},
      {"detection_threshold", compiled.payload.detection_threshold},
      {"nms_iou_threshold", compiled.payload.nms_iou_threshold},
      {"topk", compiled.payload.topk},
      {"slice_shapes", shape_descs_to_json(compiled.payload.slice_shapes)},
  };
  json root{
      {"element_name", element_name},
      {"logical_stage_id", logical_stage_id},
      {"static_contract", boxdecode_static_contract_to_json(contract)},
      {"payload", std::move(payload)},
      {"runtime_contract",
       {
           {"plugin_kind", compiled.runtime_contract.plugin_kind},
           {"required_preprocess_meta_fields",
            compiled.runtime_contract.required_preprocess_meta_fields},
           {"logical_inputs", std::move(logical_inputs)},
           {"input_bindings", std::move(input_bindings)},
       }},
  };
  boxdecode_write_json_file(std::filesystem::path(dir) / "core_boxdecode_contract.json", root);
}

std::string resolve_boxdecode_factory() {
  if (const char* forced = std::getenv("SIMA_BOXDECODE_FACTORY"); forced && *forced) {
    const std::string forced_name(forced);
    if (forced_name != "neatobjectdecode") {
      throw std::runtime_error("SimaBoxDecode: only neatobjectdecode is supported. "
                               "Remove SIMA_BOXDECODE_FACTORY or set it to neatobjectdecode.");
    }
  }
  if (!element_exists("neatobjectdecode")) {
    throw std::runtime_error(
        "SimaBoxDecode: required GStreamer element 'neatobjectdecode' is not available. "
        "Ensure the NEAT objectdecode plugin is installed and discoverable in the current "
        "GST plugin path.");
  }
  return "neatobjectdecode";
}

pipeline_internal::sima::ModelManagedRouteFlags
resolve_model_route_flags(const simaai::neat::Model& model,
                          const std::optional<bool>& route_tess_needed_override,
                          const std::optional<bool>& route_quant_needed_override) {
  auto flags = simaai::neat::internal::ModelAccess::model_managed_route_flags(model);
  if (route_tess_needed_override.has_value()) {
    flags.tess_needed = *route_tess_needed_override;
  }
  if (route_quant_needed_override.has_value()) {
    flags.quant_needed = *route_quant_needed_override;
  }
  flags.quant_contract_required = flags.quant_needed;
  flags.boxdecode_selected = true;
  return flags;
}

pipeline_internal::sima::ModelManagedRouteFlags merge_boxdecode_contract_route_flags(
    const pipeline_internal::sima::ModelManagedRouteFlags& model_flags,
    const pipeline_internal::sima::BoxDecodeStaticContract& contract) {
  auto flags = pipeline_internal::sima::model_route_flags_from_boxdecode_contract(contract);
  flags.pre_cast_needed = model_flags.pre_cast_needed;
  flags.include_pre_stage = model_flags.include_pre_stage;
  return flags;
}

bool has_explicit_dimension_pair(int width, int height) {
  return width > 0 && height > 0;
}

void validate_dimension_override_pair(int width, int height, const char* label,
                                      const char* context) {
  const bool width_set = width > 0;
  const bool height_set = height > 0;
  if (width_set == height_set) {
    return;
  }
  std::ostringstream oss;
  oss << context << ": explicit " << label << " requires both width and height";
  oss << " (width=" << width << ", height=" << height << ")";
  throw std::invalid_argument(oss.str());
}

std::vector<std::string>
filter_required_preprocess_meta_fields(const std::vector<std::string>& fields, int original_width,
                                       int original_height, int model_width, int model_height,
                                       bool has_resize_mode_override = false) {
  std::vector<std::string> filtered;
  filtered.reserve(fields.size());
  const bool has_original_override = has_explicit_dimension_pair(original_width, original_height);
  const bool has_model_override = has_explicit_dimension_pair(model_width, model_height);
  // resize_mode_override is the customer's explicit "I did my own preproc"
  // signal — by definition there is no upstream Preproc stage to emit
  // per-buffer GstSimaaiPreprocessMeta, so the *entire* preproc_* family of
  // required meta fields (color_in/out, axis_perm, normalize, quantize,
  // tessellate, affine_*) must be dropped together. Geometry fields are
  // still gated on the corresponding dimension overrides above so callers
  // can opt into per-field relaxation without flipping the whole switch.
  for (const auto& field : fields) {
    if (field == "preproc_original_width" && has_original_override) {
      continue;
    }
    if (field == "preproc_original_height" && has_original_override) {
      continue;
    }
    if (has_model_override &&
        (field == "preproc_resized_width" || field == "preproc_resized_height" ||
         field == "preproc_scaled_width" || field == "preproc_scaled_height" ||
         field == "preproc_pad_left" || field == "preproc_pad_right" ||
         field == "preproc_pad_top" || field == "preproc_pad_bottom")) {
      continue;
    }
    if (has_resize_mode_override) {
      // resize_mode lives in this bucket plus all the non-geometry semantic
      // fields below; they all originate from the same upstream Preproc
      // emitter, which is absent in the manual-preproc composition.
      if (field == "preproc_resize_mode" || field == "preproc_color_in" ||
          field == "preproc_color_out" || field == "preproc_axis_perm" ||
          field == "preproc_normalize" || field == "preproc_quantize" ||
          field == "preproc_tessellate" || field == "preproc_affine_m00" ||
          field == "preproc_affine_m01" || field == "preproc_affine_m02" ||
          field == "preproc_affine_m10" || field == "preproc_affine_m11" ||
          field == "preproc_affine_m12" || field == "preproc_affine_scale_x" ||
          field == "preproc_affine_scale_y" || field == "preproc_affine_offset_x" ||
          field == "preproc_affine_offset_y") {
        continue;
      }
    }
    filtered.push_back(field);
  }
  return filtered;
}

void apply_yolov26_compiled_payload_overrides(CompiledBoxDecodeContract* compiled) {
  if (!compiled || (compiled->payload.decode_type != BoxDecodeType::YoloV26 &&
                    compiled->payload.decode_type != BoxDecodeType::YoloV26Pose &&
                    compiled->payload.decode_type != BoxDecodeType::YoloV26Seg)) {
    return;
  }
  // YOLO26 score heads are logits by contract. Make model-managed overrides
  // robust even when an MPK route was auto-extracted through legacy YOLOv8
  // grouped-head heuristics.
  compiled->payload.decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
  compiled->payload.score_activation = pipeline_internal::sima::BoxDecodeScoreActivation::Sigmoid;
  if (compiled->payload.decode_type == BoxDecodeType::YoloV26Pose &&
      compiled->payload.num_classes <= 0) {
    compiled->payload.num_classes = 1;
  }
}

void apply_raw_yolov6_yolox_compiled_payload_overrides(CompiledBoxDecodeContract* compiled) {
  if (!compiled || (compiled->payload.decode_type != BoxDecodeType::YoloV6 &&
                    compiled->payload.decode_type != BoxDecodeType::YoloX)) {
    return;
  }
  compiled->payload.score_activation = pipeline_internal::sima::BoxDecodeScoreActivation::Sigmoid;
  if (!compiled->payload.decode_type_option.has_value() ||
      *compiled->payload.decode_type_option == BoxDecodeTypeOption::Auto) {
    compiled->payload.decode_type_option = compiled->payload.decode_type == BoxDecodeType::YoloX
                                               ? BoxDecodeTypeOption::Split3Interleaved
                                               : BoxDecodeTypeOption::InterleavedByHeadLogit;
  }
}

void apply_ssd_compiled_payload_overrides(CompiledBoxDecodeContract* compiled) {
  if (!compiled || compiled->payload.decode_type != BoxDecodeType::Ssd) {
    return;
  }
  // SSD confidence heads are raw class logits decoded with a softmax over the class
  // dimension. Keep the grouped-by-role head layout (all loc heads, then all conf
  // heads) robust even when a model-managed route was auto-extracted.
  compiled->payload.score_activation = pipeline_internal::sima::BoxDecodeScoreActivation::Softmax;
  if (!compiled->payload.decode_type_option.has_value() ||
      *compiled->payload.decode_type_option == BoxDecodeTypeOption::Auto) {
    compiled->payload.decode_type_option = BoxDecodeTypeOption::GroupedByRole;
  }
}

} // namespace

static BoxDecodeOptionsInternal options_from_model(
    const simaai::neat::internal::ModelPack& model,
    const pipeline_internal::sima::ModelManagedRouteFlags& resolved_route_flags,
    const CompiledBoxDecodeContract& compiled_contract,
    const std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics>& forced_model_semantics =
        std::nullopt) {
  BoxDecodeOptionsInternal opt;
  opt.factory = resolve_boxdecode_factory();
  if (forced_model_semantics.has_value()) {
    opt.model_semantics = *forced_model_semantics;
  } else {
    pipeline_internal::sima::ModelBoxdecodeSemantics boxdecode_semantics;
    boxdecode_semantics.tess_needed = resolved_route_flags.tess_needed;
    boxdecode_semantics.quant_needed = resolved_route_flags.quant_needed;
    boxdecode_semantics.quant_contract_required = resolved_route_flags.quant_contract_required;
    opt.model_semantics = boxdecode_semantics;
  }
  opt.model_route_flags = resolved_route_flags;
  opt.compiled_contract = std::make_shared<const CompiledBoxDecodeContract>(compiled_contract);
  opt.decode_type = compiled_contract.payload.decode_type;
  if (compiled_contract.payload.decode_type_option.has_value()) {
    opt.decode_type_option = *compiled_contract.payload.decode_type_option;
  }
  opt.top_k = compiled_contract.payload.topk;
  opt.detection_threshold = compiled_contract.payload.detection_threshold;
  opt.nms_iou_threshold = compiled_contract.payload.nms_iou_threshold;
  if (boxdecode_debug_enabled()) {
    const std::string decode_type_token = pipeline_internal::sima::box_decode_type_token_string(
        compiled_contract.payload.decode_type);
    std::fprintf(stderr, "[boxdecode-debug] etc_dir=%s factory=%s decode_type=%s\n",
                 model.etc_dir().c_str(), opt.factory.c_str(), decode_type_token.c_str());
  }
  return opt;
}

static BoxDecodeOptionsInternal
options_from_customer(BoxDecodeType decode_type, double detection_threshold,
                      double nms_iou_threshold, int top_k, const std::string& element_name,
                      int original_width, int original_height, int model_width, int model_height,
                      BoxDecodeTypeOption decode_type_option) {
  BoxDecodeOptionsInternal opt;
  opt.factory = resolve_boxdecode_factory();
  opt.element_name = element_name;
  opt.original_width = original_width;
  opt.original_height = original_height;
  opt.model_width = model_width;
  opt.model_height = model_height;
  if (decode_type != BoxDecodeType::Unspecified) {
    opt.decode_type = decode_type;
  }
  opt.decode_type_option = decode_type_option;
  if (detection_threshold > 0.0) {
    opt.detection_threshold = detection_threshold;
  }
  if (nms_iou_threshold > 0.0) {
    opt.nms_iou_threshold = nms_iou_threshold;
  }
  if (top_k > 0) {
    opt.top_k = top_k;
  }
  opt.required_preprocess_meta_fields = filter_required_preprocess_meta_fields(
      default_preprocess_meta_required_fields(), original_width, original_height, model_width,
      model_height);
  return opt;
}

static BoxDecodeOptionsInternal options_from_contract(
    const pipeline_internal::sima::BoxDecodeStaticContract& static_contract,
    BoxDecodeType decode_type, double detection_threshold, double nms_iou_threshold, int top_k,
    const std::string& element_name,
    const std::vector<std::string>& required_preprocess_meta_fields,
    const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>& route_flags,
    const std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics>& model_semantics,
    const std::optional<bool>& expect_resize, const std::optional<bool>& expect_normalize,
    const std::optional<bool>& expect_quantize, const std::optional<bool>& expect_tessellate,
    int original_width, int original_height, int model_width, int model_height,
    BoxDecodeTypeOption decode_type_option) {
  BoxDecodeOptionsInternal opt;
  opt.factory = resolve_boxdecode_factory();
  opt.element_name = element_name;
  opt.model_static_contract = static_contract;
  opt.model_route_flags = route_flags;
  opt.model_semantics = model_semantics;
  opt.required_preprocess_meta_fields = filter_required_preprocess_meta_fields(
      required_preprocess_meta_fields, original_width, original_height, model_width, model_height);
  opt.expect_resize = expect_resize;
  opt.expect_normalize = expect_normalize;
  opt.expect_quantize = expect_quantize;
  opt.expect_tessellate = expect_tessellate;
  opt.original_width = original_width;
  opt.original_height = original_height;
  opt.model_width = model_width;
  opt.model_height = model_height;
  if (decode_type != BoxDecodeType::Unspecified) {
    opt.decode_type = decode_type;
  }
  opt.decode_type_option = decode_type_option;
  if (detection_threshold > 0.0) {
    opt.detection_threshold = detection_threshold;
  }
  if (nms_iou_threshold > 0.0) {
    opt.nms_iou_threshold = nms_iou_threshold;
  }
  if (top_k > 0) {
    opt.top_k = top_k;
  }
  return opt;
}

SimaBoxDecode::SimaBoxDecode(BoxDecodeType decode_type, double detection_threshold,
                             double nms_iou_threshold, int top_k, const std::string& element_name,
                             int original_width, int original_height, int model_width,
                             int model_height, BoxDecodeTypeOption decode_type_option,
                             std::optional<BoxDecodeSourceStorage> source_storage,
                             std::optional<bool> detess, std::optional<bool> dequant) {
  validate_dimension_override_pair(original_width, original_height, "original dimensions",
                                   "SimaBoxDecode");
  validate_dimension_override_pair(model_width, model_height, "model dimensions", "SimaBoxDecode");
  auto opt = std::make_unique<BoxDecodeOptionsInternal>(options_from_customer(
      decode_type, detection_threshold, nms_iou_threshold, top_k, element_name, original_width,
      original_height, model_width, model_height, decode_type_option));
  if (!pipeline_internal::sima::is_box_decode_type_specified(opt->decode_type)) {
    throw std::invalid_argument(
        "SimaBoxDecode: decode_type is required and cannot be BoxDecodeType::Unspecified.");
  }
  if (source_storage.has_value()) {
    opt->source_storage_kind = source_storage_kind_from_public(*source_storage);
  }
  opt->override_tess_needed = detess;
  opt->override_quant_needed = dequant;
  opt_ = std::move(opt);
}

SimaBoxDecode::SimaBoxDecode(const simaai::neat::Model& model, BoxDecodeType decode_type,
                             double detection_threshold, double nms_iou_threshold, int top_k,
                             const std::string& element_name, std::optional<bool> route_tess_needed,
                             std::optional<bool> route_quant_needed, int original_width,
                             int original_height, int model_width, int model_height,
                             std::optional<ResizeMode> resize_mode_override,
                             BoxDecodeTypeOption decode_type_option) {
  validate_dimension_override_pair(original_width, original_height, "original dimensions",
                                   "SimaBoxDecode(Model)");
  // model_{width,height} are optional overrides on the model-managed path: by
  // default they're derived from the model pack at construction time, but the
  // caller may pass non-zero values to refine the decoder kernel's spatial
  // knobs (e.g. a YOLOv8 MPK trained at 640 driven with 320x320 frames). Both
  // must be set together so the kernel never sees a half-specified override.
  validate_dimension_override_pair(model_width, model_height, "model dimensions",
                                   "SimaBoxDecode(Model)");
  const auto& pack = simaai::neat::internal::ModelAccess::pack(model);
  int resolved_original_width = original_width;
  int resolved_original_height = original_height;
  int resolved_model_width = model_width;
  int resolved_model_height = model_height;
  const pipeline_internal::sima::ModelManagedRouteFlags resolved_route_flags =
      resolve_model_route_flags(model, route_tess_needed, route_quant_needed);
  auto effective_route_flags = resolved_route_flags;
  simaai::neat::internal::ModelAccess::require_model_managed_stage(
      model, simaai::neat::internal::StageNodeKind::BoxDecode, "SimaBoxDecode(Model)");
  CompiledBoxDecodeContract compiled_contract;
  try {
    compiled_contract =
        simaai::neat::internal::ModelAccess::build_boxdecode_stage_contract(model, false);
    effective_route_flags.tess_needed = compiled_contract.payload.tess_needed;
    effective_route_flags.quant_needed = compiled_contract.payload.quant_needed;
    effective_route_flags.quant_contract_required =
        compiled_contract.payload.quant_contract_required;
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "SimaBoxDecode(Model): failed to issue model-managed boxdecode contract: " +
        std::string(e.what()));
  }
  if (decode_type != BoxDecodeType::Unspecified) {
    validate_requested_boxdecode_contract_type_local(compiled_contract.payload.decode_type,
                                                     decode_type, "SimaBoxDecode(Model)");
    compiled_contract.payload.decode_type = decode_type;
  }
  if (decode_type_option != BoxDecodeTypeOption::Auto) {
    compiled_contract.payload.decode_type_option = decode_type_option;
    if (decode_type_option == BoxDecodeTypeOption::GroupedByRoleProbability ||
        decode_type_option == BoxDecodeTypeOption::InterleavedByHeadProbability) {
      compiled_contract.payload.score_activation =
          pipeline_internal::sima::BoxDecodeScoreActivation::Identity;
    } else if (decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit ||
               decode_type_option == BoxDecodeTypeOption::InterleavedByHeadLogit) {
      compiled_contract.payload.score_activation =
          pipeline_internal::sima::BoxDecodeScoreActivation::Sigmoid;
    }
  }
  apply_yolov26_compiled_payload_overrides(&compiled_contract);
  apply_raw_yolov6_yolox_compiled_payload_overrides(&compiled_contract);
  apply_ssd_compiled_payload_overrides(&compiled_contract);
  if (detection_threshold > 0.0) {
    compiled_contract.payload.detection_threshold = detection_threshold;
  }
  if (nms_iou_threshold > 0.0) {
    compiled_contract.payload.nms_iou_threshold = nms_iou_threshold;
  }
  if (top_k > 0) {
    compiled_contract.payload.topk = top_k;
  }
  auto opt = std::make_unique<BoxDecodeOptionsInternal>(
      options_from_model(pack, effective_route_flags, compiled_contract));
  opt->model_lineage = simaai::neat::internal::make_model_lineage_binding(
      model, simaai::neat::internal::ModelLineageStageRole::ManualPost,
      simaai::neat::internal::RequestedPostRouteKind::BoxDecode, "SimaBoxDecode");
  opt->requested_post_route = simaai::neat::internal::RequestedPostRouteKind::BoxDecode;
  opt->decode_type_option = decode_type_option;
  opt->element_name = element_name;
  const auto resolved = model.resolved_preprocess_plan();
  opt->resize_mode_override = resize_mode_override;
  opt->required_preprocess_meta_fields = filter_required_preprocess_meta_fields(
      resolved.meta_contract.required_fields, resolved_original_width, resolved_original_height,
      resolved_model_width, resolved_model_height, resize_mode_override.has_value());
  if (opt->compiled_contract) {
    auto updated = std::make_shared<CompiledBoxDecodeContract>(*opt->compiled_contract);
    updated->runtime_contract.required_preprocess_meta_fields =
        opt->required_preprocess_meta_fields;
    opt->compiled_contract = updated;
  }
  opt->expect_resize = (resolved.effective.resize.enable == AutoFlag::On);
  opt->expect_normalize = (resolved.effective.normalize.enable == AutoFlag::On);
  opt->expect_quantize = (resolved.effective.quantize.enable == AutoFlag::On);
  opt->expect_tessellate = (resolved.effective.tessellate.enable == AutoFlag::On);
  opt->original_width = resolved_original_width;
  opt->original_height = resolved_original_height;
  opt->model_width = resolved_model_width;
  opt->model_height = resolved_model_height;
  if (decode_type != BoxDecodeType::Unspecified)
    opt->decode_type = decode_type;
  if (decode_type_option != BoxDecodeTypeOption::Auto) {
    opt->decode_type_option = decode_type_option;
  }
  if (detection_threshold > 0.0)
    opt->detection_threshold = detection_threshold;
  if (nms_iou_threshold > 0.0)
    opt->nms_iou_threshold = nms_iou_threshold;
  if (top_k > 0)
    opt->top_k = top_k;
  if (!pipeline_internal::sima::is_box_decode_type_specified(opt->decode_type)) {
    throw std::invalid_argument(
        "SimaBoxDecode: decode_type is required and cannot be BoxDecodeType::Unspecified. "
        "Set Model::Options.decode_type or pass decode_type explicitly.");
  }

  opt_ = std::move(opt);
}

#ifdef SIMA_NEAT_INTERNAL
SimaBoxDecode::SimaBoxDecode(
    const pipeline_internal::sima::BoxDecodeStaticContract& contract, BoxDecodeType decode_type,
    double detection_threshold, double nms_iou_threshold, int top_k,
    const std::string& element_name,
    const std::vector<std::string>& required_preprocess_meta_fields,
    std::optional<pipeline_internal::sima::ModelManagedRouteFlags> route_flags,
    std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics> model_semantics,
    std::optional<bool> expect_resize, std::optional<bool> expect_normalize,
    std::optional<bool> expect_quantize, std::optional<bool> expect_tessellate, int original_width,
    int original_height, int model_width, int model_height,
    BoxDecodeTypeOption decode_type_option) {
  validate_dimension_override_pair(original_width, original_height, "original dimensions",
                                   "SimaBoxDecode");
  validate_dimension_override_pair(model_width, model_height, "model dimensions", "SimaBoxDecode");
  auto opt = std::make_unique<BoxDecodeOptionsInternal>(options_from_contract(
      contract, decode_type, detection_threshold, nms_iou_threshold, top_k, element_name,
      required_preprocess_meta_fields, route_flags, model_semantics, expect_resize,
      expect_normalize, expect_quantize, expect_tessellate, original_width, original_height,
      model_width, model_height, decode_type_option));
  if (!pipeline_internal::sima::is_box_decode_type_specified(opt->decode_type)) {
    throw std::invalid_argument(
        "SimaBoxDecode: decode_type is required and cannot be BoxDecodeType::Unspecified. "
        "Set decode_type explicitly for standalone boxdecode.");
  }
  opt_ = std::move(opt);
}
#endif

NodeContractDefinition SimaBoxDecode::contract_definition() const {
  NodeContractDefinition def;
  def.node_kind = kind();
  def.plugin_kind = "boxdecode";

  ContractPortSpec input;
  input.port_id = "input_tensor";
  input.media_type = "application/vnd.simaai.tensor";
  if (opt_) {
    input.required_preprocess_meta_fields = opt_->required_preprocess_meta_fields;
  }
  def.inputs.push_back(std::move(input));
  return def;
}

bool SimaBoxDecode::compile_node_contract(const ContractCompileInput& input,
                                          CompiledNodeContract* out, std::string* err) const {
  if (!opt_) {
    if (err) {
      *err = "SimaBoxDecode: missing node options";
    }
    return false;
  }
  const std::string element_name = element_names(input.node_index).empty()
                                       ? std::string("boxdecode")
                                       : element_names(input.node_index).front();
  try {
    if (opt_->compiled_contract) {
      return pipeline_internal::sima::stagesemantics::build_boxdecode_node_contract(
          kind(), factory_internal(), element_name,
          user_label().empty() ? element_name : user_label(), contract_definition(),
          *opt_->compiled_contract, out, err);
    }

    std::optional<pipeline_internal::sima::BoxDecodeStaticContract> contract =
        opt_->model_static_contract;
    // Hand-built (no-model-pack) decode paths cannot infer the source byte layout / detess /
    // dequant facts the model pack would normally provide; forward any explicit caller overrides.
    const pipeline_internal::sima::BoxDecodeStandaloneContractOverrides standalone_overrides{
        opt_->source_storage_kind, opt_->override_tess_needed, opt_->override_quant_needed};
    if (!contract.has_value()) {
      if (input.immediate_upstream != nullptr) {
        contract = pipeline_internal::sima::build_boxdecode_static_contract_from_compiled_upstream(
            *input.immediate_upstream, opt_->decode_type, standalone_overrides, err);
        if (!contract.has_value()) {
          // Helper may return nullopt without populating *err. Provide a
          // contextual default so the diagnostic message conveys *which*
          // path failed instead of bubbling up an empty string that gets
          // overwritten by a generic "no compiler path" message upstream.
          if (err && err->empty()) {
            *err = "SimaBoxDecode: failed to derive box-decode static contract from immediate "
                   "upstream stage '" +
                   input.immediate_upstream->node_kind +
                   "' (compiled upstream did not expose detess/dequant metadata required for "
                   "boxdecode)";
          }
          return false;
        }
        if (input.ingress.ingress_sample.has_value()) {
          maybe_refine_boxdecode_contract_from_ingress_sample_local(
              &*contract, *input.ingress.ingress_sample, opt_->decode_type, input_contract_);
        }
      } else if (!input.ingress.ingress_sample.has_value()) {
        if (err) {
          *err = "SimaBoxDecode: inferred standalone contract requires Graph::build/run input "
                 "sample";
        }
        return false;
      } else {
        contract = pipeline_internal::sima::build_boxdecode_static_contract_from_sample(
            *input.ingress.ingress_sample, opt_->decode_type, input_contract_, standalone_overrides,
            err);
        if (!contract.has_value()) {
          if (err && err->empty()) {
            *err = "SimaBoxDecode: failed to derive box-decode static contract from ingress "
                   "sample";
          }
          return false;
        }
      }
    }
    const auto finalized_contract =
        pipeline_internal::sima::stagesemantics::finalize_boxdecode_static_contract(
            *contract, opt_->decode_type, opt_->model_semantics, opt_->model_route_flags,
            opt_->decode_type_option, opt_->detection_threshold, opt_->nms_iou_threshold,
            opt_->top_k, /*num_classes=*/0, opt_->required_preprocess_meta_fields);
    const auto compiled =
        pipeline_internal::sima::stagesemantics::build_boxdecode_compiled_contract(
            finalized_contract);
    maybe_dump_boxdecode_core_dossier(finalized_contract, compiled, element_name,
                                      user_label().empty() ? element_name : user_label());
    return pipeline_internal::sima::stagesemantics::build_boxdecode_node_contract(
        kind(), factory_internal(), element_name,
        user_label().empty() ? element_name : user_label(), contract_definition(), compiled, out,
        err);
  } catch (const std::exception& ex) {
    if (err) {
      *err = ex.what();
    }
    return false;
  }
}

void SimaBoxDecode::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) {
    err->clear();
  }
}

void SimaBoxDecode::apply_input_contract(const InputContract& contract, std::string* err) {
  input_contract_ = contract;
  if (err) {
    err->clear();
  }
}

#ifdef SIMA_NEAT_INTERNAL
const std::string& SimaBoxDecode::factory_internal() const {
  static const std::string kDefaultFactory = "neatobjectdecode";
  return opt_ ? opt_->factory : kDefaultFactory;
}

BoxDecodeType SimaBoxDecode::decode_type_internal() const {
  return opt_ ? opt_->decode_type : BoxDecodeType::Unspecified;
}

double SimaBoxDecode::detection_threshold_internal() const {
  return opt_ ? opt_->detection_threshold : 0.0;
}

double SimaBoxDecode::nms_iou_threshold_internal() const {
  return opt_ ? opt_->nms_iou_threshold : 0.0;
}

int SimaBoxDecode::top_k_internal() const {
  return opt_ ? opt_->top_k : 0;
}

int SimaBoxDecode::original_width_internal() const {
  return opt_ ? opt_->original_width : 0;
}

int SimaBoxDecode::original_height_internal() const {
  return opt_ ? opt_->original_height : 0;
}

BoxDecodeTypeOption SimaBoxDecode::decode_type_option_internal() const {
  return opt_ ? opt_->decode_type_option : BoxDecodeTypeOption::Auto;
}

const std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics>&
SimaBoxDecode::model_semantics_internal() const {
  static const std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics> kNone;
  return opt_ ? opt_->model_semantics : kNone;
}

const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>&
SimaBoxDecode::model_route_flags_internal() const {
  static const std::optional<pipeline_internal::sima::ModelManagedRouteFlags> kNone;
  return opt_ ? opt_->model_route_flags : kNone;
}

const std::optional<pipeline_internal::sima::BoxDecodeStaticContract>&
SimaBoxDecode::model_static_contract_internal() const {
  static const std::optional<pipeline_internal::sima::BoxDecodeStaticContract> kNone;
  return opt_ ? opt_->model_static_contract : kNone;
}

const std::vector<std::string>& SimaBoxDecode::required_preprocess_meta_fields_internal() const {
  static const std::vector<std::string> kEmpty;
  return opt_ ? opt_->required_preprocess_meta_fields : kEmpty;
}

const std::shared_ptr<const internal::ModelLineageBinding>&
SimaBoxDecode::model_lineage_binding_internal() const {
  static const std::shared_ptr<const internal::ModelLineageBinding> kNone;
  return opt_ ? opt_->model_lineage : kNone;
}

internal::RequestedPostRouteKind SimaBoxDecode::requested_post_route_internal() const {
  return opt_ ? opt_->requested_post_route : internal::RequestedPostRouteKind::Auto;
}
#endif

std::string SimaBoxDecode::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element(opt_->factory.c_str(), "SimaBoxDecode::backend_fragment");
  const char* factory = opt_->factory.c_str();
  const std::string name =
      opt_->element_name.empty() ? std::string("boxdecode") : opt_->element_name;
  ss << factory << " name=" << name << " stage-id=" << name;

  ss << " silent=" << (opt_->silent ? "true" : "false");
  ss << " emit-signals=" << (opt_->emit_signals ? "true" : "false");
  if (opt_->sima_allocator_type > 0) {
    ss << " sima-allocator-type=" << opt_->sima_allocator_type;
  }
  if (!pipeline_internal::sima::is_box_decode_type_specified(opt_->decode_type)) {
    throw std::runtime_error(
        "SimaBoxDecode::backend_fragment: decode_type is required but unspecified.");
  }
  const std::string decode_type_token =
      pipeline_internal::sima::box_decode_type_token_string(opt_->decode_type);
  ss << " transmit=" << (opt_->transmit ? "true" : "false");
  if (boxdecode_debug_enabled()) {
    std::fprintf(stderr,
                 "[boxdecode-debug] backend_fragment stage=%s factory=%s decode_type=%s topk=%d "
                 "det=%.6f nms=%.6f metadata_only=1 contract_only=1\n",
                 name.c_str(), opt_->factory.c_str(), decode_type_token.c_str(), opt_->top_k,
                 opt_->detection_threshold, opt_->nms_iou_threshold);
  }
  if (opt_->original_width > 0) {
    ss << " original-width=" << opt_->original_width;
  }
  if (opt_->original_height > 0) {
    ss << " original-height=" << opt_->original_height;
  }
  if (opt_->model_width > 0) {
    ss << " model-width=" << opt_->model_width;
  }
  if (opt_->model_height > 0) {
    ss << " model-height=" << opt_->model_height;
  }
  if (opt_->resize_mode_override.has_value()) {
    const char* mode = "letterbox";
    if (*opt_->resize_mode_override == ResizeMode::Stretch)
      mode = "stretch";
    if (*opt_->resize_mode_override == ResizeMode::Crop)
      mode = "crop";
    ss << " resize-mode=" << mode;
  }
  return ss.str();
}

std::vector<std::string> SimaBoxDecode::element_names(int node_index) const {
  (void)node_index;
  if (opt_ && !opt_->element_name.empty()) {
    return {opt_->element_name};
  }
  return {"boxdecode"};
}

OutputSpec SimaBoxDecode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "BBOX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = opt_ ? opt_->factory : "neatobjectdecode";
  return out;
}

std::optional<PreprocessMetaRequirement> SimaBoxDecode::preprocess_meta_requirement() const {
  if (!opt_ || opt_->required_preprocess_meta_fields.empty()) {
    return std::nullopt;
  }
  PreprocessMetaRequirement req;
  req.stage_name = "boxdecode";
  req.plugin_name = opt_->factory;
  req.required_fields = opt_->required_preprocess_meta_fields;
  req.expect_resize = opt_->expect_resize;
  req.expect_normalize = opt_->expect_normalize;
  req.expect_quantize = opt_->expect_quantize;
  req.expect_tessellate = opt_->expect_tessellate;
  return req;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node>
SimaBoxDecode(BoxDecodeType decode_type, double detection_threshold, double nms_iou_threshold,
              int top_k, const std::string& element_name, int original_width, int original_height,
              int model_width, int model_height, BoxDecodeTypeOption decode_type_option,
              std::optional<BoxDecodeSourceStorage> source_storage, std::optional<bool> detess,
              std::optional<bool> dequant) {
  return std::make_shared<simaai::neat::SimaBoxDecode>(
      decode_type, detection_threshold, nms_iou_threshold, top_k, element_name, original_width,
      original_height, model_width, model_height, decode_type_option, source_storage, detess,
      dequant);
}

std::shared_ptr<simaai::neat::Node>
SimaBoxDecode(const simaai::neat::Model& model, BoxDecodeType decode_type,
              double detection_threshold, double nms_iou_threshold, int top_k,
              const std::string& element_name, std::optional<bool> route_tess_needed,
              std::optional<bool> route_quant_needed, int original_width, int original_height,
              int model_width, int model_height, std::optional<ResizeMode> resize_mode_override,
              BoxDecodeTypeOption decode_type_option) {
  return std::make_shared<simaai::neat::SimaBoxDecode>(
      model, decode_type, detection_threshold, nms_iou_threshold, top_k, element_name,
      route_tess_needed, route_quant_needed, original_width, original_height, model_width,
      model_height, resize_mode_override, decode_type_option);
}

#ifdef SIMA_NEAT_INTERNAL
std::shared_ptr<simaai::neat::Node>
SimaBoxDecode(const pipeline_internal::sima::BoxDecodeStaticContract& contract,
              BoxDecodeType decode_type, double detection_threshold, double nms_iou_threshold,
              int top_k, const std::string& element_name,
              const std::vector<std::string>& required_preprocess_meta_fields,
              std::optional<pipeline_internal::sima::ModelManagedRouteFlags> route_flags,
              std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics> model_semantics,
              std::optional<bool> expect_resize, std::optional<bool> expect_normalize,
              std::optional<bool> expect_quantize, std::optional<bool> expect_tessellate,
              int original_width, int original_height, int model_width, int model_height,
              BoxDecodeTypeOption decode_type_option) {
  return std::make_shared<simaai::neat::SimaBoxDecode>(
      contract, decode_type, detection_threshold, nms_iou_threshold, top_k, element_name,
      required_preprocess_meta_fields, route_flags, model_semantics, expect_resize,
      expect_normalize, expect_quantize, expect_tessellate, original_width, original_height,
      model_width, model_height, decode_type_option);
}
#endif

} // namespace simaai::neat::nodes

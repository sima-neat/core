#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <array>
#include <cctype>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace simaai::neat::pipeline_internal::sima {
namespace {

void set_error(std::string* error_message, const std::string& message);
bool quant_contract_complete(const std::optional<MpkQuantContract>& quant);
bool tensor_name_looks_class_prob_local(const std::string& raw_name);
bool tensor_name_looks_class_logit_local(const std::string& raw_name);
bool tensor_name_looks_dequant_stage_local(const std::string& raw_name);
bool tensor_name_declares_probability_domain_local(const std::string& raw_name);
struct BoxDecodeTensorLineageFactsLocal;
enum class BoxDecodeTensorRoleLocal;
std::optional<std::pair<BoxDecodeTensorRoleLocal, int>>
classify_boxdecode_tensor_semantics_from_name_local(const std::string& raw_name);
bool contract_looks_grouped_by_role_yolov8_local(const BoxDecodeStaticContract& contract);
std::string lower_copy_local(std::string value);
int to_non_negative_int_local(std::int64_t value);

std::string tensor_name_from_logical_output_local(const LogicalTensorStaticSpec& logical,
                                                  std::size_t index) {
  if (!logical.backend_name.empty()) {
    return logical.backend_name;
  }
  if (!logical.logical_name.empty()) {
    return logical.logical_name;
  }
  if (!logical.segment_name.empty()) {
    return logical.segment_name;
  }
  return "input_tensor_" + std::to_string(index);
}

bool dims_from_logical_output_local(const LogicalTensorStaticSpec& logical, int* out_h, int* out_w,
                                    int* out_c) {
  if (!out_h || !out_w || !out_c) {
    return false;
  }
  std::vector<std::int64_t> dims = logical.shape;
  if (dims.size() == 4U && dims.front() == 1) {
    dims.erase(dims.begin());
  }
  if (dims.size() != 2U && dims.size() != 3U) {
    return false;
  }
  const std::string layout = lower_copy_local(logical.layout);
  if (layout == "chw" && dims.size() == 3U) {
    *out_c = to_non_negative_int_local(dims[0]);
    *out_h = to_non_negative_int_local(dims[1]);
    *out_w = to_non_negative_int_local(dims[2]);
  } else if (layout == "hw" || dims.size() == 2U) {
    *out_h = to_non_negative_int_local(dims[0]);
    *out_w = to_non_negative_int_local(dims[1]);
    *out_c = 1;
  } else {
    *out_h = to_non_negative_int_local(dims[0]);
    *out_w = to_non_negative_int_local(dims[1]);
    *out_c = (dims.size() > 2U) ? to_non_negative_int_local(dims[2]) : 1;
  }
  return *out_h > 0 && *out_w > 0 && *out_c > 0;
}

std::vector<const LogicalTensorStaticSpec*>
ordered_logical_outputs_from_runtime_contract_local(const CompiledRuntimeContract& runtime) {
  std::vector<const LogicalTensorStaticSpec*> ordered;
  ordered.reserve(runtime.logical_outputs.size());
  std::unordered_set<int> seen_logical;
  if (!runtime.output_order.empty()) {
    for (const auto& route : runtime.output_order) {
      const LogicalTensorStaticSpec* match = nullptr;
      if (route.logical_output_index >= 0) {
        for (const auto& logical : runtime.logical_outputs) {
          if (logical.logical_index == route.logical_output_index) {
            match = &logical;
            break;
          }
        }
      }
      if (!match && route.tensor_index >= 0 &&
          static_cast<std::size_t>(route.tensor_index) < runtime.logical_outputs.size()) {
        match = &runtime.logical_outputs[static_cast<std::size_t>(route.tensor_index)];
      }
      if (!match) {
        continue;
      }
      if (match->logical_index >= 0 && !seen_logical.insert(match->logical_index).second) {
        continue;
      }
      ordered.push_back(match);
    }
  }
  if (ordered.empty()) {
    for (const auto& logical : runtime.logical_outputs) {
      ordered.push_back(&logical);
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const LogicalTensorStaticSpec* lhs, const LogicalTensorStaticSpec* rhs) {
                       return lhs->logical_index < rhs->logical_index;
                     });
  }
  return ordered;
}

std::string lower_copy_local(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

int to_non_negative_int_local(std::int64_t value) {
  if (value <= 0) {
    return 0;
  }
  if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

std::string normalize_mpk_dtype_token_local(std::string raw) {
  for (char& c : raw) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  constexpr const char* kPrefixes[] = {"EVXX_", "EV81_", "EV80_", "EV79_"};
  for (const char* prefix : kPrefixes) {
    if (raw.rfind(prefix, 0U) == 0U) {
      raw.erase(0U, std::char_traits<char>::length(prefix));
      break;
    }
  }
  if (raw.find("BFLOAT16") != std::string::npos || raw.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("FLOAT32") != std::string::npos || raw.find("FP32") != std::string::npos) {
    return "FP32";
  }
  if (raw.find("FLOAT16") != std::string::npos || raw.find("FP16") != std::string::npos) {
    return "FP16";
  }
  if (raw.find("UINT8") != std::string::npos) {
    return "UINT8";
  }
  if (raw.find("INT8") != std::string::npos) {
    return "INT8";
  }
  if (raw.find("UINT16") != std::string::npos) {
    return "UINT16";
  }
  if (raw.find("INT16") != std::string::npos) {
    return "INT16";
  }
  if (raw.find("UINT32") != std::string::npos) {
    return "UINT32";
  }
  if (raw.find("INT32") != std::string::npos) {
    return "INT32";
  }
  return raw;
}

bool dtype_is_quantized_like_local(std::string raw) {
  raw = normalize_mpk_dtype_token_local(std::move(raw));
  if (raw.empty()) {
    return false;
  }
  if (raw.find("BF16") != std::string::npos || raw.find("BFLOAT16") != std::string::npos ||
      raw.find("FP16") != std::string::npos || raw.find("FP32") != std::string::npos ||
      raw.find("FLOAT") != std::string::npos) {
    return false;
  }
  return raw.find("INT8") != std::string::npos || raw.find("UINT8") != std::string::npos ||
         raw.find("INT16") != std::string::npos || raw.find("UINT16") != std::string::npos ||
         raw.find("INT32") != std::string::npos || raw.find("UINT32") != std::string::npos;
}

std::optional<std::int64_t> quant_min_code_for_dtype_local(std::string raw) {
  raw = normalize_mpk_dtype_token_local(std::move(raw));
  if (raw == "INT8") {
    return static_cast<std::int64_t>(std::numeric_limits<std::int8_t>::min());
  }
  if (raw == "UINT8") {
    return 0;
  }
  if (raw == "INT16") {
    return static_cast<std::int64_t>(std::numeric_limits<std::int16_t>::min());
  }
  if (raw == "UINT16") {
    return 0;
  }
  if (raw == "INT32") {
    return static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min());
  }
  if (raw == "UINT32") {
    return 0;
  }
  return std::nullopt;
}

std::optional<std::int64_t> quant_max_code_for_dtype_local(std::string raw) {
  raw = normalize_mpk_dtype_token_local(std::move(raw));
  if (raw == "INT8") {
    return static_cast<std::int64_t>(std::numeric_limits<std::int8_t>::max());
  }
  if (raw == "UINT8") {
    return static_cast<std::int64_t>(std::numeric_limits<std::uint8_t>::max());
  }
  if (raw == "INT16") {
    return static_cast<std::int64_t>(std::numeric_limits<std::int16_t>::max());
  }
  if (raw == "UINT16") {
    return static_cast<std::int64_t>(std::numeric_limits<std::uint16_t>::max());
  }
  if (raw == "INT32") {
    return static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
  }
  if (raw == "UINT32") {
    return static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
  }
  return std::nullopt;
}

bool tensor_name_explicitly_declares_score_semantics_local(
    const BoxDecodeTensorStaticContract& tensor) {
  return tensor_name_looks_class_prob_local(tensor.logical_name) ||
         tensor_name_looks_class_logit_local(tensor.logical_name) ||
         tensor_name_looks_class_prob_local(tensor.backend_name) ||
         tensor_name_looks_class_logit_local(tensor.backend_name) ||
         tensor_name_looks_class_prob_local(tensor.source_segment_name) ||
         tensor_name_looks_class_logit_local(tensor.source_segment_name);
}

bool decode_type_is_yolov8_family_local(BoxDecodeType decode_type) {
  return decode_type == BoxDecodeType::YoloV8 || decode_type == BoxDecodeType::YoloV8Seg ||
         decode_type == BoxDecodeType::YoloV8Pose;
}

bool decode_type_is_yolov26_family_local(BoxDecodeType decode_type) {
  return decode_type == BoxDecodeType::YoloV26 || decode_type == BoxDecodeType::YoloV26Pose ||
         decode_type == BoxDecodeType::YoloV26Seg;
}

bool decode_type_is_raw_yolov6_yolox_family_local(BoxDecodeType decode_type) {
  return decode_type == BoxDecodeType::YoloV6 || decode_type == BoxDecodeType::YoloX;
}

int logical_channel_depth_local(const BoxDecodeTensorStaticContract& tensor) {
  if (tensor.slice_shape.size() >= 3U && tensor.slice_shape.back() > 0) {
    return tensor.slice_shape.back();
  }
  if (tensor.input_shape.size() >= 3U && tensor.input_shape.back() > 0) {
    return tensor.input_shape.back();
  }
  return 0;
}

bool tensor_name_looks_objectness_logit_local(const BoxDecodeTensorStaticContract& tensor) {
  auto looks = [](std::string raw) {
    raw = lower_copy_local(raw);
    return raw.find("obj_logit") != std::string::npos ||
           raw.find("objectness_logit") != std::string::npos ||
           raw.find("object_logit") != std::string::npos;
  };
  return looks(tensor.logical_name) || looks(tensor.backend_name) ||
         looks(tensor.source_segment_name);
}

int infer_raw_yolo_class_depth_local(const BoxDecodeStaticContract& contract) {
  int best = 0;
  for (const auto& tensor : contract.tensors) {
    const int c = logical_channel_depth_local(tensor);
    if (c <= 4) {
      continue;
    }
    if (tensor_name_looks_objectness_logit_local(tensor)) {
      continue;
    }
    best = std::max(best, c);
  }
  return best;
}

void apply_raw_yolov6_yolox_contract_overrides_local(BoxDecodeStaticContract* contract) {
  if (!contract || !decode_type_is_raw_yolov6_yolox_family_local(contract->decode_type)) {
    return;
  }
  // The raw-head surgeries deliberately cut before the final CPU-friendly
  // nonlinear decode: class/objectness tensors are logits, not probabilities.
  if (contract->score_activation == BoxDecodeScoreActivation::Unknown) {
    contract->score_activation = BoxDecodeScoreActivation::Sigmoid;
  }
  if (contract->decode_type_option == BoxDecodeTypeOption::Auto) {
    contract->decode_type_option = contract->decode_type == BoxDecodeType::YoloX
                                       ? BoxDecodeTypeOption::Split3Interleaved
                                       : BoxDecodeTypeOption::InterleavedByHeadLogit;
  }
  if (contract->num_classes <= 0) {
    contract->num_classes = infer_raw_yolo_class_depth_local(*contract);
  }
}

void maybe_infer_score_activation_from_boxdecode_contract_local(BoxDecodeStaticContract* contract) {
  if (!contract || contract->score_activation != BoxDecodeScoreActivation::Unknown) {
    return;
  }
  bool saw_prob_tensor = false;
  bool saw_logit_tensor = false;
  for (const auto& tensor : contract->tensors) {
    saw_prob_tensor = saw_prob_tensor ||
                      tensor_name_declares_probability_domain_local(tensor.logical_name) ||
                      tensor_name_declares_probability_domain_local(tensor.backend_name) ||
                      tensor_name_declares_probability_domain_local(tensor.source_segment_name);
    saw_logit_tensor = saw_logit_tensor ||
                       tensor_name_looks_class_logit_local(tensor.logical_name) ||
                       tensor_name_looks_class_logit_local(tensor.backend_name) ||
                       tensor_name_looks_class_logit_local(tensor.source_segment_name);
  }
  if (saw_prob_tensor && !saw_logit_tensor) {
    contract->score_activation = BoxDecodeScoreActivation::Identity;
    return;
  }
  if (saw_logit_tensor && !saw_prob_tensor) {
    contract->score_activation = BoxDecodeScoreActivation::Sigmoid;
    return;
  }
}

bool quantized_tensor_looks_probability_domain_local(const BoxDecodeTensorStaticContract& tensor,
                                                     double dq_scale, std::int64_t dq_zp) {
  if (!dtype_is_quantized_like_local(tensor.data_type) || dq_scale <= 0.0) {
    return false;
  }
  const auto qmin = quant_min_code_for_dtype_local(tensor.data_type);
  const auto qmax = quant_max_code_for_dtype_local(tensor.data_type);
  if (!qmin.has_value() || !qmax.has_value() || dq_zp != *qmin) {
    return false;
  }
  const double min_value = (static_cast<double>(*qmin) - static_cast<double>(dq_zp)) / dq_scale;
  const double max_value = (static_cast<double>(*qmax) - static_cast<double>(dq_zp)) / dq_scale;
  return std::abs(min_value) <= 1e-9 && max_value <= 1.05;
}

void maybe_override_quantized_yolov8_score_activation_local(BoxDecodeStaticContract* contract) {
  if (!contract || !decode_type_is_yolov8_family_local(contract->decode_type) ||
      !contract->quant_needed || contract->score_activation != BoxDecodeScoreActivation::Unknown ||
      contract->tensors.size() != contract->dq_scale.size() ||
      contract->tensors.size() != contract->dq_zp.size() || contract->tensors.empty()) {
    return;
  }
  if (std::any_of(contract->tensors.begin(), contract->tensors.end(), [](const auto& tensor) {
        return tensor_name_explicitly_declares_score_semantics_local(tensor);
      })) {
    return;
  }
  int max_depth = 0;
  for (const auto& tensor : contract->tensors) {
    max_depth = std::max(max_depth, tensor.input_shape.size() >= 3 ? tensor.input_shape[2] : 0);
  }
  if (max_depth <= 4) {
    return;
  }
  bool saw_score_tensor = false;
  for (std::size_t i = 0; i < contract->tensors.size(); ++i) {
    const auto& tensor = contract->tensors[i];
    if ((tensor.input_shape.size() >= 3 ? tensor.input_shape[2] : 0) != max_depth) {
      continue;
    }
    saw_score_tensor = true;
    if (!quantized_tensor_looks_probability_domain_local(tensor, contract->dq_scale[i],
                                                         contract->dq_zp[i])) {
      return;
    }
  }
  if (saw_score_tensor) {
    contract->score_activation = BoxDecodeScoreActivation::Identity;
  }
}

void maybe_default_float_yolo_score_activation_local(BoxDecodeStaticContract* contract) {
  if (!contract || contract->score_activation != BoxDecodeScoreActivation::Unknown ||
      contract->quant_needed) {
    return;
  }
  if (decode_type_is_yolov8_family_local(contract->decode_type) ||
      decode_type_is_yolov26_family_local(contract->decode_type) ||
      decode_type_is_raw_yolov6_yolox_family_local(contract->decode_type)) {
    contract->score_activation = BoxDecodeScoreActivation::Sigmoid;
  }
}

bool contract_looks_grouped_by_role_yolov8_local(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.size() < 2U || (contract.tensors.size() % 2U) != 0U) {
    return false;
  }
  const std::size_t heads = contract.tensors.size() / 2U;
  for (std::size_t i = 0; i < heads; ++i) {
    const auto& bbox = contract.tensors[i];
    const auto& score = contract.tensors[i + heads];
    const int bbox_w = bbox.input_shape.size() >= 2 ? bbox.input_shape[1] : 0;
    const int bbox_h = bbox.input_shape.size() >= 1 ? bbox.input_shape[0] : 0;
    const int bbox_d = bbox.input_shape.size() >= 3 ? bbox.input_shape[2] : 0;
    const int score_w = score.input_shape.size() >= 2 ? score.input_shape[1] : 0;
    const int score_h = score.input_shape.size() >= 1 ? score.input_shape[0] : 0;
    const int score_d = score.input_shape.size() >= 3 ? score.input_shape[2] : 0;
    if (bbox_w <= 0 || bbox_h <= 0 || bbox_d <= 0 || score_w <= 0 || score_h <= 0 || score_d <= 0) {
      return false;
    }
    if (bbox_w != score_w || bbox_h != score_h) {
      return false;
    }
  }
  return true;
}

void maybe_infer_yolov8_decode_type_option_local(BoxDecodeStaticContract* contract) {
  if (!contract ||
      (!decode_type_is_yolov8_family_local(contract->decode_type) &&
       !decode_type_is_yolov26_family_local(contract->decode_type)) ||
      contract->decode_type_option != BoxDecodeTypeOption::Auto ||
      !contract_looks_grouped_by_role_yolov8_local(*contract)) {
    return;
  }

  bool saw_probability_name = false;
  bool saw_logit_name = false;
  for (const auto& tensor : contract->tensors) {
    saw_probability_name =
        saw_probability_name ||
        tensor_name_declares_probability_domain_local(tensor.logical_name) ||
        tensor_name_declares_probability_domain_local(tensor.backend_name) ||
        tensor_name_declares_probability_domain_local(tensor.source_segment_name);
    saw_logit_name = saw_logit_name || tensor_name_looks_class_logit_local(tensor.logical_name) ||
                     tensor_name_looks_class_logit_local(tensor.backend_name) ||
                     tensor_name_looks_class_logit_local(tensor.source_segment_name);
  }

  if (saw_probability_name && !saw_logit_name) {
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleProbability;
    return;
  }
  if (saw_logit_name && !saw_probability_name) {
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
    return;
  }
  if (contract->score_activation == BoxDecodeScoreActivation::Sigmoid) {
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
    return;
  }
  if (contract->score_activation == BoxDecodeScoreActivation::Identity) {
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleProbability;
    return;
  }
}

std::string dtype_token_from_tensor_local(const TensorDType dtype) {
  switch (dtype) {
  case TensorDType::BFloat16:
    return "BF16";
  case TensorDType::Float32:
    return "FP32";
  case TensorDType::Float64:
    return "FP64";
  case TensorDType::Int8:
    return "INT8";
  case TensorDType::UInt8:
    return "UINT8";
  case TensorDType::Int16:
    return "INT16";
  case TensorDType::UInt16:
    return "UINT16";
  case TensorDType::Int32:
    return "INT32";
  default:
    return {};
  }
}

int dtype_size_bytes_local(std::string raw) {
  raw = normalize_mpk_dtype_token_local(std::move(raw));
  if (raw == "INT8" || raw == "UINT8") {
    return 1;
  }
  if (raw == "BF16" || raw == "FP16" || raw == "INT16" || raw == "UINT16") {
    return 2;
  }
  if (raw == "FP32" || raw == "INT32" || raw == "UINT32") {
    return 4;
  }
  if (raw == "FP64") {
    return 8;
  }
  return 0;
}

std::optional<TensorLayout> tensor_layout_from_contract_token_local(std::string raw) {
  raw = lower_copy_local(std::move(raw));
  if (raw == "chw") {
    return TensorLayout::CHW;
  }
  if (raw == "hwc") {
    return TensorLayout::HWC;
  }
  if (raw == "hw") {
    return TensorLayout::HW;
  }
  return std::nullopt;
}

TensorLayout resolve_tensor_layout_local(const Tensor& tensor,
                                         const std::optional<InputContract>& input_contract) {
  if (tensor.layout != TensorLayout::Unknown) {
    return tensor.layout;
  }
  if (input_contract.has_value() && !input_contract->layout.empty()) {
    if (const auto parsed = tensor_layout_from_contract_token_local(input_contract->layout)) {
      return *parsed;
    }
  }
  if (tensor.shape.size() == 2U) {
    return TensorLayout::HW;
  }
  if (tensor.shape.size() == 3U) {
    return TensorLayout::HWC;
  }
  if (tensor.shape.size() == 4U && tensor.shape.front() == 1) {
    return TensorLayout::HWC;
  }
  return TensorLayout::Unknown;
}

std::string layout_token_from_tensor_layout_local(TensorLayout layout) {
  switch (layout) {
  case TensorLayout::CHW:
    return "CHW";
  case TensorLayout::HWC:
    return "HWC";
  case TensorLayout::HW:
    return "HW";
  case TensorLayout::Unknown:
  default:
    return {};
  }
}

bool dims_from_tensor_local(const Tensor& tensor,
                            const std::optional<InputContract>& input_contract, int* out_h,
                            int* out_w, int* out_c) {
  if (!out_h || !out_w || !out_c) {
    return false;
  }
  std::vector<std::int64_t> dims = tensor.shape;
  if (dims.size() == 4U && dims.front() == 1) {
    dims.erase(dims.begin());
  }
  if (dims.size() != 2U && dims.size() != 3U) {
    return false;
  }
  const TensorLayout layout = resolve_tensor_layout_local(tensor, input_contract);
  if (layout == TensorLayout::CHW && dims.size() == 3U) {
    *out_c = to_non_negative_int_local(dims[0]);
    *out_h = to_non_negative_int_local(dims[1]);
    *out_w = to_non_negative_int_local(dims[2]);
  } else if (layout == TensorLayout::HW || dims.size() == 2U) {
    *out_h = to_non_negative_int_local(dims[0]);
    *out_w = to_non_negative_int_local(dims[1]);
    *out_c = 1;
  } else {
    *out_h = to_non_negative_int_local(dims[0]);
    *out_w = to_non_negative_int_local(dims[1]);
    *out_c = dims.size() > 2U ? to_non_negative_int_local(dims[2]) : 1;
  }
  return *out_h > 0 && *out_w > 0 && *out_c > 0;
}

std::string tensor_name_from_sample_local(const Tensor& tensor, std::size_t index) {
  if (!tensor.route.backend_name.empty()) {
    return tensor.route.backend_name;
  }
  if (!tensor.route.name.empty()) {
    return tensor.route.name;
  }
  if (!tensor.route.segment_name.empty()) {
    return tensor.route.segment_name;
  }
  return "input_tensor_" + std::to_string(index);
}

std::uint64_t dense_tensor_bytes_local(const Tensor& tensor,
                                       const std::string& dtype_token_override) {
  if (!dtype_token_override.empty()) {
    const int elem_bytes = dtype_size_bytes_local(dtype_token_override);
    if (elem_bytes > 0 && !tensor.shape.empty()) {
      std::uint64_t elements = 1;
      for (const auto dim : tensor.shape) {
        if (dim <= 0) {
          return 0;
        }
        elements *= static_cast<std::uint64_t>(dim);
      }
      return elements * static_cast<std::uint64_t>(elem_bytes);
    }
  }
  return static_cast<std::uint64_t>(tensor.cpu().contiguous().copy_payload_bytes().size());
}

bool dtype_is_float_like_local(std::string raw) {
  raw = normalize_mpk_dtype_token_local(std::move(raw));
  return raw == "BF16" || raw == "FP16" || raw == "FP32" || raw == "FP64";
}

float bfloat16_to_float_local(std::uint16_t raw) {
  const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16U;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float fp16_to_float_local(std::uint16_t raw) {
  const std::uint32_t sign = static_cast<std::uint32_t>(raw & 0x8000U) << 16U;
  const std::uint32_t exp = (raw >> 10U) & 0x1FU;
  const std::uint32_t mant = raw & 0x03FFU;

  std::uint32_t bits = 0U;
  if (exp == 0U) {
    if (mant != 0U) {
      std::uint32_t normalized = mant;
      int shift = 0;
      while ((normalized & 0x0400U) == 0U) {
        normalized <<= 1U;
        ++shift;
      }
      normalized &= 0x03FFU;
      const std::uint32_t fp32_exp = 127U - 15U - static_cast<std::uint32_t>(shift) + 1U;
      bits = sign | (fp32_exp << 23U) | (normalized << 13U);
    } else {
      bits = sign;
    }
  } else if (exp == 0x1FU) {
    bits = sign | 0x7F800000U | (mant << 13U);
  } else {
    const std::uint32_t fp32_exp = exp + (127U - 15U);
    bits = sign | (fp32_exp << 23U) | (mant << 13U);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

bool sample_float_value_at_local(const std::vector<std::uint8_t>& payload, const std::string& dtype,
                                 std::size_t elem_index, float* out_value) {
  if (!out_value) {
    return false;
  }
  const std::string normalized = normalize_mpk_dtype_token_local(dtype);
  if (normalized == "BF16") {
    const std::size_t byte_offset = elem_index * 2U;
    if (byte_offset + 1U >= payload.size()) {
      return false;
    }
    const std::uint16_t raw = static_cast<std::uint16_t>(payload[byte_offset]) |
                              (static_cast<std::uint16_t>(payload[byte_offset + 1U]) << 8U);
    *out_value = bfloat16_to_float_local(raw);
    return true;
  }
  if (normalized == "FP16") {
    const std::size_t byte_offset = elem_index * 2U;
    if (byte_offset + 1U >= payload.size()) {
      return false;
    }
    const std::uint16_t raw = static_cast<std::uint16_t>(payload[byte_offset]) |
                              (static_cast<std::uint16_t>(payload[byte_offset + 1U]) << 8U);
    *out_value = fp16_to_float_local(raw);
    return true;
  }
  if (normalized == "FP32") {
    const std::size_t byte_offset = elem_index * 4U;
    if (byte_offset + 3U >= payload.size()) {
      return false;
    }
    std::uint32_t raw = 0U;
    std::memcpy(&raw, payload.data() + byte_offset, sizeof(raw));
    std::memcpy(out_value, &raw, sizeof(raw));
    return true;
  }
  return false;
}

std::optional<BoxDecodeScoreActivation>
maybe_infer_float_yolov8_score_activation_from_sample_values_local(
    const TensorList& tensors, const std::optional<InputContract>& input_contract) {
  int max_depth = 0;
  std::vector<std::size_t> candidate_indices;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    int h = 0;
    int w = 0;
    int c = 0;
    if (!dims_from_tensor_local(tensors[i], input_contract, &h, &w, &c)) {
      continue;
    }
    if (c > max_depth) {
      max_depth = c;
      candidate_indices.clear();
      candidate_indices.push_back(i);
    } else if (c == max_depth) {
      candidate_indices.push_back(i);
    }
  }
  if (max_depth <= 4 || candidate_indices.empty()) {
    return std::nullopt;
  }

  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  std::size_t sampled_values = 0U;

  for (const std::size_t tensor_index : candidate_indices) {
    const auto& tensor = tensors[tensor_index];
    const std::string dtype =
        normalize_mpk_dtype_token_local(dtype_token_from_tensor_local(tensor.dtype));
    if (!dtype_is_float_like_local(dtype)) {
      return std::nullopt;
    }
    const int elem_bytes = dtype_size_bytes_local(dtype);
    if (elem_bytes <= 0) {
      return std::nullopt;
    }

    const std::vector<std::uint8_t> payload = tensor.cpu().contiguous().copy_payload_bytes();
    if (payload.empty() || (payload.size() % static_cast<std::size_t>(elem_bytes)) != 0U) {
      return std::nullopt;
    }

    const std::size_t elem_count = payload.size() / static_cast<std::size_t>(elem_bytes);
    const std::size_t sample_limit = 4096U;
    const std::size_t step = std::max<std::size_t>(1U, elem_count / sample_limit);
    for (std::size_t elem = 0U; elem < elem_count; elem += step) {
      float value = 0.0f;
      if (!sample_float_value_at_local(payload, dtype, elem, &value) || !std::isfinite(value)) {
        return std::nullopt;
      }
      min_value = std::min(min_value, static_cast<double>(value));
      max_value = std::max(max_value, static_cast<double>(value));
      ++sampled_values;
    }
  }

  if (sampled_values == 0U) {
    return std::nullopt;
  }
  if (min_value < -0.05 || max_value > 1.25) {
    return BoxDecodeScoreActivation::Sigmoid;
  }
  return std::nullopt;
}

std::optional<BoxDecodeScoreActivation> resolve_boxdecode_score_activation_from_sample_local(
    const TensorList& tensors, BoxDecodeType decode_type,
    const std::optional<InputContract>& input_contract, std::string* error_message) {
  bool saw_prob_tensor = false;
  bool saw_logit_tensor = false;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const std::string name = tensor_name_from_sample_local(tensors[i], i);
    saw_prob_tensor = saw_prob_tensor || tensor_name_declares_probability_domain_local(name);
    saw_logit_tensor = saw_logit_tensor || tensor_name_looks_class_logit_local(name);
  }
  if (saw_prob_tensor && saw_logit_tensor) {
    set_error(error_message,
              "boxdecode inferred input tensors expose both class_prob and class_logit semantics");
    return std::nullopt;
  }
  if (saw_prob_tensor) {
    return BoxDecodeScoreActivation::Identity;
  }
  if (saw_logit_tensor) {
    return BoxDecodeScoreActivation::Sigmoid;
  }
  if (decode_type_is_yolov8_family_local(decode_type) ||
      decode_type_is_yolov26_family_local(decode_type)) {
    if (const auto inferred = maybe_infer_float_yolov8_score_activation_from_sample_values_local(
            tensors, input_contract);
        inferred.has_value()) {
      return inferred;
    }
  }
  if (decode_type_is_yolov8_family_local(decode_type) ||
      decode_type_is_yolov26_family_local(decode_type)) {
    return BoxDecodeScoreActivation::Sigmoid;
  }
  set_error(error_message,
            "boxdecode inferred input tensors require explicit class score activation semantics");
  return std::nullopt;
}

bool tensor_name_looks_detess_stage_local(const std::string& raw_name) {
  const std::string name = lower_copy_local(raw_name);
  return name.find("detessellate_") != std::string::npos ||
         name.find("detessellation_transform") != std::string::npos ||
         name.find("tuple_get_item_") != std::string::npos;
}

bool tensor_name_looks_generic_output_local(const std::string& raw_name) {
  const std::string name = lower_copy_local(raw_name);
  return name.empty() || name.rfind("pass_through_out_", 0U) == 0U ||
         name.rfind("output_tensor_", 0U) == 0U || name == "output_tensor";
}

bool tensor_name_is_aux_parent_local(std::string raw_name) {
  raw_name = lower_copy_local(std::move(raw_name));
  if (raw_name.empty() || raw_name == "parent" || raw_name == "mla_output_tensor" ||
      raw_name == "__mla_parent__") {
    return true;
  }
  if (raw_name.rfind("mla_", 0U) != 0U) {
    return false;
  }
  const char* suffix = raw_name.c_str() + 4;
  if (!suffix || !*suffix) {
    return false;
  }
  for (const char* p = suffix; *p != '\0'; ++p) {
    if (std::isdigit(static_cast<unsigned char>(*p)) == 0) {
      return false;
    }
  }
  return true;
}

std::optional<int> tensor_name_head_index_local(std::string raw_name) {
  raw_name = lower_copy_local(std::move(raw_name));
  if (raw_name.empty()) {
    return std::nullopt;
  }

  const std::string tuple_token = "tuple_get_item_";
  const std::size_t tuple_pos = raw_name.find(tuple_token);
  if (tuple_pos != std::string::npos) {
    std::size_t idx_start = tuple_pos + tuple_token.size();
    std::size_t idx_end = idx_start;
    while (idx_end < raw_name.size() &&
           std::isdigit(static_cast<unsigned char>(raw_name[idx_end])) != 0) {
      ++idx_end;
    }
    if (idx_end > idx_start) {
      try {
        return std::stoi(raw_name.substr(idx_start, idx_end - idx_start));
      } catch (...) {
        return std::nullopt;
      }
    }
  }

  const std::size_t us = raw_name.rfind('_');
  if (us != std::string::npos && (us + 1U) < raw_name.size()) {
    std::size_t idx_end = us + 1U;
    while (idx_end < raw_name.size() &&
           std::isdigit(static_cast<unsigned char>(raw_name[idx_end])) != 0) {
      ++idx_end;
    }
    if (idx_end == raw_name.size() && idx_end > (us + 1U)) {
      try {
        return std::stoi(raw_name.substr(us + 1U));
      } catch (...) {
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

bool tensor_name_looks_raw_mla_head_local(std::string raw_name) {
  raw_name = lower_copy_local(std::move(raw_name));
  if (raw_name.empty() || tensor_name_is_aux_parent_local(raw_name)) {
    return false;
  }
  if (raw_name.find("mla_") == std::string::npos) {
    return false;
  }
  return tensor_name_head_index_local(raw_name).has_value();
}

bool sample_tensor_semantics_require_tess_local(
    const Tensor& tensor, const std::string& tensor_name, const std::string& input_dtype,
    const std::optional<InputContract>& input_contract) {
  if (!tensor.semantic.tess.has_value()) {
    return false;
  }
  if (tensor_name_looks_detess_stage_local(tensor_name) ||
      tensor_name_looks_generic_output_local(tensor_name)) {
    return false;
  }
  if (tensor_name_looks_raw_mla_head_local(tensor_name)) {
    return true;
  }

  int h = 0;
  int w = 0;
  int c = 0;
  const bool has_dense_geometry =
      dims_from_tensor_local(tensor, input_contract, &h, &w, &c) && h > 0 && w > 0 && c > 0;
  if (has_dense_geometry && dtype_is_float_like_local(input_dtype)) {
    return false;
  }
  return true;
}

void maybe_reorder_tensors_by_logical_index_local(TensorList* tensors) {
  if (!tensors || tensors->empty()) {
    return;
  }
  const bool all_have_logical_index =
      std::all_of(tensors->begin(), tensors->end(),
                  [](const auto& tensor) { return tensor.route.logical_index >= 0; });
  if (!all_have_logical_index) {
    return;
  }
  std::stable_sort(tensors->begin(), tensors->end(), [](const auto& a, const auto& b) {
    return a.route.logical_index < b.route.logical_index;
  });
}

bool dims_from_mpk_shape_for_input_nhwc_local(const std::vector<std::int64_t>& raw_shape,
                                              int* out_h, int* out_w, int* out_c) {
  if (!out_h || !out_w || !out_c || raw_shape.empty()) {
    return false;
  }
  int n = 1;
  int h = 0;
  int w = 0;
  int c = 0;
  if (raw_shape.size() >= 4U) {
    n = to_non_negative_int_local(raw_shape[raw_shape.size() - 4U]);
    h = to_non_negative_int_local(raw_shape[raw_shape.size() - 3U]);
    w = to_non_negative_int_local(raw_shape[raw_shape.size() - 2U]);
    c = to_non_negative_int_local(raw_shape[raw_shape.size() - 1U]);
  } else if (raw_shape.size() == 3U) {
    h = to_non_negative_int_local(raw_shape[0]);
    w = to_non_negative_int_local(raw_shape[1]);
    c = to_non_negative_int_local(raw_shape[2]);
  } else if (raw_shape.size() == 2U) {
    h = to_non_negative_int_local(raw_shape[0]);
    w = to_non_negative_int_local(raw_shape[1]);
    c = 1;
  } else {
    h = 1;
    w = to_non_negative_int_local(raw_shape[0]);
    c = 1;
  }
  if (n <= 0) {
    n = 1;
  }
  (void)n;
  if (h <= 0 || w <= 0 || c <= 0) {
    return false;
  }
  *out_h = h;
  *out_w = w;
  *out_c = c;
  return true;
}

bool dims_from_mpk_tess_slice_shape_local(const std::vector<std::int64_t>& slice_shape, int* out_h,
                                          int* out_w, int* out_c) {
  if (!out_h || !out_w || !out_c) {
    return false;
  }
  int d = 1;
  int h = 0;
  int w = 0;
  int c = 0;
  if (slice_shape.size() == 4U) {
    d = to_non_negative_int_local(slice_shape[0]);
    h = to_non_negative_int_local(slice_shape[1]);
    w = to_non_negative_int_local(slice_shape[2]);
    c = to_non_negative_int_local(slice_shape[3]);
  } else if (slice_shape.size() == 3U) {
    d = 1;
    h = to_non_negative_int_local(slice_shape[0]);
    w = to_non_negative_int_local(slice_shape[1]);
    c = to_non_negative_int_local(slice_shape[2]);
  } else {
    return false;
  }
  if (d <= 0) {
    d = 1;
  }
  (void)d;
  if (h <= 0 || w <= 0 || c <= 0) {
    return false;
  }
  *out_h = h;
  *out_w = w;
  *out_c = c;
  return true;
}

bool kernel_is_detess_like_local(const std::string& kernel_kind) {
  const std::string token = lower_copy_local(kernel_kind);
  if (token.find("detessdequant") != std::string::npos) {
    return false;
  }
  return token.find("detessellate") != std::string::npos || token == "detess" ||
         token.find("detess_") != std::string::npos;
}

bool kernel_is_dequant_like_local(const std::string& kernel_kind) {
  const std::string token = lower_copy_local(kernel_kind);
  return token.find("dequant") != std::string::npos;
}

std::string stage_identity_token_local(const MpkPluginIoContract& plugin) {
  std::string token = plugin.kernel;
  if (!plugin.name.empty()) {
    if (!token.empty()) {
      token.push_back('|');
    }
    token += plugin.name;
  }
  if (!plugin.plugin_id.empty()) {
    if (!token.empty()) {
      token.push_back('|');
    }
    token += plugin.plugin_id;
  }
  if (!plugin.executable.empty()) {
    if (!token.empty()) {
      token.push_back('|');
    }
    token += plugin.executable;
  }
  return token;
}

bool stage_is_detess_like_local(const MpkPluginIoContract& plugin) {
  return kernel_is_detess_like_local(stage_identity_token_local(plugin));
}

bool stage_is_dequant_like_local(const MpkPluginIoContract& plugin) {
  return kernel_is_dequant_like_local(stage_identity_token_local(plugin));
}

bool stage_has_explicit_dequant_quant_local(const MpkPluginIoContract& plugin) {
  const std::string token = lower_copy_local(stage_identity_token_local(plugin));
  return token.find("dequant") != std::string::npos &&
         token.find("detessdequant") == std::string::npos;
}

enum class BoxDecodeTensorRoleLocal {
  Unknown = 0,
  Regression,
  Score,
};

struct BoxDecodeTensorLineageFactsLocal {
  BoxDecodeTensorRoleLocal role = BoxDecodeTensorRoleLocal::Unknown;
  std::optional<int> head_index;
  std::optional<std::array<int, 3>> detess_slice;
  std::optional<std::array<int, 3>> detess_transport_input_hwc;
  std::optional<std::uint64_t> detess_transport_size_bytes;
  std::optional<double> dq_scale;
  std::optional<std::int64_t> dq_zp;
};

std::string
synthesize_boxdecode_tensor_name_from_lineage_local(const BoxDecodeTensorLineageFactsLocal& facts,
                                                    bool quant_needed) {
  if (!facts.head_index.has_value()) {
    return {};
  }
  switch (facts.role) {
  case BoxDecodeTensorRoleLocal::Regression:
    return "bbox_" + std::to_string(*facts.head_index);
  case BoxDecodeTensorRoleLocal::Score:
    return quant_needed ? ("class_prob_" + std::to_string(*facts.head_index)) : std::string{};
  case BoxDecodeTensorRoleLocal::Unknown:
  default:
    return {};
  }
}

bool boxdecode_tensor_has_semantic_name_local(const BoxDecodeTensorStaticContract& tensor) {
  return classify_boxdecode_tensor_semantics_from_name_local(tensor.logical_name).has_value() ||
         classify_boxdecode_tensor_semantics_from_name_local(tensor.backend_name).has_value() ||
         classify_boxdecode_tensor_semantics_from_name_local(tensor.source_segment_name)
             .has_value();
}

void maybe_restore_boxdecode_semantic_names_from_lineage_local(
    BoxDecodeStaticContract* contract,
    const std::vector<BoxDecodeTensorLineageFactsLocal>& lineage_facts) {
  if (!contract || contract->tensors.size() != lineage_facts.size()) {
    return;
  }
  for (std::size_t i = 0; i < contract->tensors.size(); ++i) {
    auto& tensor = contract->tensors[i];
    if (boxdecode_tensor_has_semantic_name_local(tensor)) {
      continue;
    }
    const std::string synthesized = synthesize_boxdecode_tensor_name_from_lineage_local(
        lineage_facts[i],
        contract->quant_needed && !decode_type_is_yolov26_family_local(contract->decode_type));
    if (synthesized.empty()) {
      continue;
    }
    tensor.logical_name = synthesized;
    tensor.backend_name = synthesized;
    if (i < contract->tensor_names.size()) {
      contract->tensor_names[i] = synthesized;
    }
  }
}

void maybe_restore_grouped_role_semantic_names_from_structure_local(
    BoxDecodeStaticContract* contract) {
  if (!contract ||
      (contract->decode_type != BoxDecodeType::YoloV8 &&
       !decode_type_is_yolov26_family_local(contract->decode_type)) ||
      !contract_looks_grouped_by_role_yolov8_local(*contract) || contract->tensors.empty() ||
      (contract->tensors.size() % 2U) != 0U) {
    return;
  }

  bool all_unclassified = true;
  for (const auto& tensor : contract->tensors) {
    if (boxdecode_tensor_has_semantic_name_local(tensor)) {
      all_unclassified = false;
      break;
    }
  }
  if (!all_unclassified) {
    return;
  }

  const std::size_t heads = contract->tensors.size() / 2U;
  for (std::size_t i = 0; i < heads; ++i) {
    const std::string bbox_name = "bbox_" + std::to_string(i);
    contract->tensors[i].logical_name = bbox_name;
    contract->tensors[i].backend_name = bbox_name;
    if (i < contract->tensor_names.size()) {
      contract->tensor_names[i] = bbox_name;
    }

    const bool score_is_probability =
        contract->quant_needed && !decode_type_is_yolov26_family_local(contract->decode_type);
    const std::string score_name =
        std::string(score_is_probability ? "class_prob_" : "class_logit_") + std::to_string(i);
    contract->tensors[i + heads].logical_name = score_name;
    contract->tensors[i + heads].backend_name = score_name;
    if ((i + heads) < contract->tensor_names.size()) {
      contract->tensor_names[i + heads] = score_name;
    }
  }
}

std::uint64_t output_key_local(std::size_t plugin_index, int output_index) {
  return (static_cast<std::uint64_t>(plugin_index) << 32U) |
         static_cast<std::uint32_t>(output_index + 1);
}

const MpkTensorContract* pick_stage_output_for_input_local(const MpkPluginIoContract& stage,
                                                           int input_index, int* output_index) {
  if (output_index) {
    *output_index = -1;
  }
  if (stage.output_tensors.empty()) {
    return nullptr;
  }
  if (stage.output_tensors.size() == 1U) {
    if (output_index) {
      *output_index = 0;
    }
    return &stage.output_tensors.front();
  }
  if (input_index >= 0 && static_cast<std::size_t>(input_index) < stage.output_tensors.size()) {
    if (output_index) {
      *output_index = input_index;
    }
    return &stage.output_tensors[static_cast<std::size_t>(input_index)];
  }
  if (output_index) {
    *output_index = 0;
  }
  return &stage.output_tensors.front();
}

const MpkTensorContract* pick_stage_input_for_index_local(const MpkPluginIoContract& stage,
                                                          int input_index) {
  if (stage.input_tensors.empty()) {
    return nullptr;
  }
  if (stage.input_tensors.size() == 1U) {
    return &stage.input_tensors.front();
  }
  if (input_index >= 0 && static_cast<std::size_t>(input_index) < stage.input_tensors.size()) {
    return &stage.input_tensors[static_cast<std::size_t>(input_index)];
  }
  return &stage.input_tensors.front();
}

bool assign_unique_int_local(std::optional<int>* slot, int value, std::string* error_message,
                             const std::string& message) {
  if (!slot) {
    return false;
  }
  if (!slot->has_value()) {
    *slot = value;
    return true;
  }
  if (*slot == value) {
    return true;
  }
  set_error(error_message, message);
  return false;
}

bool assign_unique_double_local(std::optional<double>* slot, double value,
                                std::string* error_message, const std::string& message) {
  if (!slot) {
    return false;
  }
  if (!slot->has_value()) {
    *slot = value;
    return true;
  }
  if (*slot == value) {
    return true;
  }
  set_error(error_message, message);
  return false;
}

bool assign_unique_int64_local(std::optional<std::int64_t>* slot, std::int64_t value,
                               std::string* error_message, const std::string& message) {
  if (!slot) {
    return false;
  }
  if (!slot->has_value()) {
    *slot = value;
    return true;
  }
  if (*slot == value) {
    return true;
  }
  set_error(error_message, message);
  return false;
}

bool assign_unique_role_local(BoxDecodeTensorRoleLocal* slot, BoxDecodeTensorRoleLocal value,
                              std::string* error_message, const std::string& message) {
  if (!slot || value == BoxDecodeTensorRoleLocal::Unknown) {
    return true;
  }
  if (*slot == BoxDecodeTensorRoleLocal::Unknown) {
    *slot = value;
    return true;
  }
  if (*slot == value) {
    return true;
  }
  set_error(error_message, message);
  return false;
}

bool maybe_record_boxdecode_tensor_semantics_from_name_local(
    BoxDecodeTensorLineageFactsLocal* facts, const std::string& raw_name,
    std::string* error_message, const std::string& conflict_prefix) {
  if (!facts || raw_name.empty()) {
    return true;
  }
  const auto classified = classify_boxdecode_tensor_semantics_from_name_local(raw_name);
  if (!classified.has_value()) {
    return true;
  }
  const auto [role, head_index] = *classified;
  if (!assign_unique_role_local(&facts->role, role, error_message,
                                conflict_prefix +
                                    " resolved conflicting tensor roles on one MPK branch")) {
    return false;
  }
  if (!assign_unique_int_local(&facts->head_index, head_index, error_message,
                               conflict_prefix +
                                   " resolved conflicting tensor head indices on one MPK branch")) {
    return false;
  }
  return true;
}

bool assign_unique_slice_local(std::optional<std::array<int, 3>>* slot,
                               const std::array<int, 3>& value, std::string* error_message,
                               const std::string& message) {
  if (!slot) {
    return false;
  }
  if (!slot->has_value()) {
    *slot = value;
    return true;
  }
  if (*slot == value) {
    return true;
  }
  set_error(error_message, message);
  return false;
}

bool assign_unique_uint64_local(std::optional<std::uint64_t>* slot, std::uint64_t value,
                                std::string* error_message, const std::string& message) {
  if (!slot) {
    return false;
  }
  if (!slot->has_value()) {
    *slot = value;
    return true;
  }
  if (*slot == value) {
    return true;
  }
  set_error(error_message, message);
  return false;
}

std::string detess_transport_dtype_token_local(const MpkPluginIoContract& stage,
                                               const MpkTensorContract* input_tensor) {
  if (!stage.frame_type.empty()) {
    return normalize_mpk_dtype_token_local(stage.frame_type);
  }
  if (!stage.canonical_input_dtype.empty()) {
    return normalize_mpk_dtype_token_local(stage.canonical_input_dtype);
  }
  if (input_tensor) {
    if (!input_tensor->logical_dtype.empty()) {
      return normalize_mpk_dtype_token_local(input_tensor->logical_dtype);
    }
    if (!input_tensor->dtype.empty()) {
      return normalize_mpk_dtype_token_local(input_tensor->dtype);
    }
  }
  if (!stage.canonical_output_dtype.empty()) {
    return normalize_mpk_dtype_token_local(stage.canonical_output_dtype);
  }
  return {};
}

std::optional<std::pair<std::array<int, 3>, std::uint64_t>>
detess_transport_input_hwc_from_mpk_local(const MpkPluginIoContract& stage,
                                          const MpkTensorContract* input_tensor,
                                          std::string* error_message) {
  if (!stage_is_detess_like_local(stage)) {
    return std::nullopt;
  }
  if (!input_tensor) {
    set_error(error_message,
              "boxdecode model-managed contract requires detess input tensor transport facts");
    return std::nullopt;
  }
  if (input_tensor->size_bytes == 0U) {
    set_error(error_message,
              "boxdecode model-managed contract requires non-zero detess input tensor size");
    return std::nullopt;
  }
  int h = 0;
  int w = 0;
  int logical_c = 0;
  if (!dims_from_mpk_shape_for_input_nhwc_local(stage.frame_shape, &h, &w, &logical_c)) {
    set_error(error_message,
              "boxdecode model-managed contract requires detess frame_shape transport facts");
    return std::nullopt;
  }
  const std::string dtype = detess_transport_dtype_token_local(stage, input_tensor);
  const int elem_bytes = dtype_size_bytes_local(dtype);
  if (elem_bytes <= 0) {
    set_error(error_message,
              "boxdecode model-managed contract requires detess transport dtype facts");
    return std::nullopt;
  }
  std::uint64_t spatial_bytes = 0U;
  const auto h64 = static_cast<std::uint64_t>(h);
  const auto w64 = static_cast<std::uint64_t>(w);
  const auto elem64 = static_cast<std::uint64_t>(elem_bytes);
  if (h64 == 0U || w64 == 0U || elem64 == 0U ||
      h64 > (std::numeric_limits<std::uint64_t>::max() / w64)) {
    set_error(error_message,
              "boxdecode model-managed contract detess transport byte span overflow");
    return std::nullopt;
  }
  spatial_bytes = h64 * w64;
  if (spatial_bytes > (std::numeric_limits<std::uint64_t>::max() / elem64)) {
    set_error(error_message,
              "boxdecode model-managed contract detess transport byte span overflow");
    return std::nullopt;
  }
  spatial_bytes *= elem64;
  const auto declared_bytes = static_cast<std::uint64_t>(input_tensor->size_bytes);
  if (spatial_bytes == 0U || (declared_bytes % spatial_bytes) != 0U) {
    set_error(error_message,
              "boxdecode model-managed contract detess transport byte span mismatch: MPK detess "
              "input declares " +
                  std::to_string(input_tensor->size_bytes) +
                  " bytes, which is not divisible by frame_shape*dtype bytes " +
                  std::to_string(spatial_bytes));
    return std::nullopt;
  }
  const std::uint64_t physical_c = declared_bytes / spatial_bytes;
  if (physical_c < static_cast<std::uint64_t>(logical_c)) {
    set_error(error_message,
              "boxdecode model-managed contract detess transport channel count is smaller than "
              "frame_shape logical channels");
    return std::nullopt;
  }
  const bool byte_granule_aligned =
      physical_c <= (std::numeric_limits<std::uint64_t>::max() / elem64) &&
      ((physical_c * elem64) % 16U) == 0U;
  if (((stage.has_align_c16 && stage.align_c16) || (stage.has_cblock && stage.cblock)) &&
      !byte_granule_aligned) {
    set_error(error_message,
              "boxdecode model-managed contract detess transport expected 16-byte aligned packed "
              "channel storage because MPK align_c16/cblock is set");
    return std::nullopt;
  }
  if (physical_c > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    set_error(error_message,
              "boxdecode model-managed contract detess transport channel count overflows int");
    return std::nullopt;
  }
  return std::make_pair(std::array<int, 3>{h, w, static_cast<int>(physical_c)}, declared_bytes);
}

std::optional<int> parse_head_index_after_token_local(const std::string& raw_name,
                                                      const char* token) {
  if (!token || !*token) {
    return std::nullopt;
  }
  const std::string lower = lower_copy_local(raw_name);
  const std::string needle = lower_copy_local(token);
  std::size_t pos = lower.find(needle);
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  pos += needle.size();
  if (pos >= lower.size() || !std::isdigit(static_cast<unsigned char>(lower[pos]))) {
    return std::nullopt;
  }
  int value = 0;
  while (pos < lower.size() && std::isdigit(static_cast<unsigned char>(lower[pos]))) {
    value = value * 10 + (lower[pos] - '0');
    ++pos;
  }
  return value;
}

std::optional<std::pair<BoxDecodeTensorRoleLocal, int>>
classify_boxdecode_tensor_semantics_from_name_local(const std::string& raw_name) {
  if (const auto idx = parse_head_index_after_token_local(raw_name, "class_prob_")) {
    return std::make_pair(BoxDecodeTensorRoleLocal::Score, *idx);
  }
  if (const auto idx = parse_head_index_after_token_local(raw_name, "class_probability_")) {
    return std::make_pair(BoxDecodeTensorRoleLocal::Score, *idx);
  }
  if (const auto idx = parse_head_index_after_token_local(raw_name, "class_logit_")) {
    return std::make_pair(BoxDecodeTensorRoleLocal::Score, *idx);
  }
  if (const auto idx = parse_head_index_after_token_local(raw_name, "class_logits_")) {
    return std::make_pair(BoxDecodeTensorRoleLocal::Score, *idx);
  }
  if (const auto idx = parse_head_index_after_token_local(raw_name, "bbox_")) {
    return std::make_pair(BoxDecodeTensorRoleLocal::Regression, *idx);
  }
  if (const auto idx = parse_head_index_after_token_local(raw_name, "reg_")) {
    return std::make_pair(BoxDecodeTensorRoleLocal::Regression, *idx);
  }
  return std::nullopt;
}

std::unordered_map<std::size_t, std::size_t>
build_execution_positions_local(const std::vector<std::size_t>& ordered) {
  std::unordered_map<std::size_t, std::size_t> out;
  out.reserve(ordered.size());
  for (std::size_t pos = 0; pos < ordered.size(); ++pos) {
    out.emplace(ordered[pos], pos);
  }
  return out;
}

std::unordered_map<std::uint64_t, std::vector<const MpkContractEdge*>>
build_outgoing_edges_local(const MpkContract& contract) {
  std::unordered_map<std::uint64_t, std::vector<const MpkContractEdge*>> outgoing;
  outgoing.reserve(contract.edges.size());
  for (const auto& edge : contract.edges) {
    outgoing[output_key_local(edge.src_plugin_index, edge.src_output_index)].push_back(&edge);
  }
  return outgoing;
}

std::optional<std::pair<std::size_t, int>> resolve_boxdecode_tensor_source_local(
    const MpkContract& contract,
    const std::unordered_map<std::size_t, std::size_t>& execution_positions,
    const MpkTensorContract& logical_output, std::size_t mla_pos, std::size_t terminal_pos) {
  const std::string want_name =
      !logical_output.name.empty() ? logical_output.name : logical_output.segment_name;
  const std::string want_segment =
      !logical_output.segment_name.empty() ? logical_output.segment_name : logical_output.name;
  auto find_best = [&](const std::string& needle) -> std::optional<std::pair<std::size_t, int>> {
    if (needle.empty()) {
      return std::nullopt;
    }
    std::optional<std::pair<std::size_t, int>> best;
    std::size_t best_pos = std::numeric_limits<std::size_t>::max();
    for (std::size_t plugin_index = 0; plugin_index < contract.plugins.size(); ++plugin_index) {
      const auto pos_it = execution_positions.find(plugin_index);
      if (pos_it == execution_positions.end()) {
        continue;
      }
      if (pos_it->second < mla_pos || pos_it->second > terminal_pos) {
        continue;
      }
      const auto& plugin = contract.plugins[plugin_index];
      for (std::size_t oi = 0; oi < plugin.output_tensors.size(); ++oi) {
        const auto& output = plugin.output_tensors[oi];
        const bool match = (!output.name.empty() && output.name == needle) ||
                           (!output.segment_name.empty() && output.segment_name == needle);
        if (!match) {
          continue;
        }
        if (!best.has_value() || pos_it->second < best_pos) {
          best = std::make_pair(plugin_index, static_cast<int>(oi));
          best_pos = pos_it->second;
        }
      }
    }
    return best;
  };

  if (const auto by_name = find_best(want_name)) {
    return by_name;
  }
  if (want_segment != want_name) {
    return find_best(want_segment);
  }
  return std::nullopt;
}

std::optional<BoxDecodeTensorLineageFactsLocal> collect_boxdecode_tensor_lineage_facts_local(
    const MpkContract& contract,
    const std::unordered_map<std::size_t, std::size_t>& execution_positions,
    const std::unordered_map<std::uint64_t, std::vector<const MpkContractEdge*>>& outgoing_edges,
    std::size_t source_plugin_index, int source_output_index, std::size_t terminal_pos,
    std::string* error_message) {
  BoxDecodeTensorLineageFactsLocal facts;
  auto inspect_stage_io = [&](const MpkPluginIoContract& stage,
                              const MpkTensorContract* input_tensor,
                              const MpkTensorContract* output_tensor)
      -> std::optional<BoxDecodeTensorLineageFactsLocal> {
    if (!maybe_record_boxdecode_tensor_semantics_from_name_local(
            &facts, stage.name, error_message, "boxdecode model-managed contract")) {
      return std::nullopt;
    }
    if (!maybe_record_boxdecode_tensor_semantics_from_name_local(
            &facts, stage.kernel, error_message, "boxdecode model-managed contract")) {
      return std::nullopt;
    }
    if (output_tensor) {
      if (!maybe_record_boxdecode_tensor_semantics_from_name_local(
              &facts, output_tensor->name, error_message, "boxdecode model-managed contract")) {
        return std::nullopt;
      }
      if (!maybe_record_boxdecode_tensor_semantics_from_name_local(
              &facts, output_tensor->segment_name, error_message,
              "boxdecode model-managed contract")) {
        return std::nullopt;
      }
    }

    if (stage_is_detess_like_local(stage)) {
      int h = 0;
      int w = 0;
      int c = 0;
      if (!dims_from_mpk_tess_slice_shape_local(stage.slice_shape, &h, &w, &c)) {
        set_error(error_message,
                  "boxdecode tessellated route requires explicit detess slice facts for every "
                  "upstream tensor");
        return std::nullopt;
      }
      if (!assign_unique_slice_local(
              &facts.detess_slice, {w, h, c}, error_message,
              "boxdecode model-managed contract resolved conflicting detess slice facts on one "
              "MPK branch")) {
        return std::nullopt;
      }
      if (input_tensor && input_tensor->size_bytes > 0U && !stage.frame_shape.empty()) {
        const auto transport =
            detess_transport_input_hwc_from_mpk_local(stage, input_tensor, error_message);
        if (!transport.has_value()) {
          return std::nullopt;
        }
        if (!assign_unique_slice_local(
                &facts.detess_transport_input_hwc, transport->first, error_message,
                "boxdecode model-managed contract resolved conflicting detess transport input "
                "shapes on one MPK branch")) {
          return std::nullopt;
        }
        if (!assign_unique_uint64_local(
                &facts.detess_transport_size_bytes, transport->second, error_message,
                "boxdecode model-managed contract resolved conflicting detess transport byte "
                "spans on one MPK branch")) {
          return std::nullopt;
        }
      }
    }

    if (stage_has_explicit_dequant_quant_local(stage) && quant_contract_complete(stage.quant)) {
      const double scale = stage.quant->scales.front();
      const std::int64_t zp = stage.quant->zero_points.front();
      if (!assign_unique_double_local(
              &facts.dq_scale, scale, error_message,
              "boxdecode model-managed contract resolved conflicting dq_scale facts on one MPK "
              "branch")) {
        return std::nullopt;
      }
      if (!assign_unique_int64_local(
              &facts.dq_zp, zp, error_message,
              "boxdecode model-managed contract resolved conflicting dq_zp facts on one MPK "
              "branch")) {
        return std::nullopt;
      }
    }
    return facts;
  };

  if (source_plugin_index >= contract.plugins.size()) {
    set_error(error_message, "boxdecode model-managed contract resolved an invalid source plugin");
    return std::nullopt;
  }
  int seed_output_index = source_output_index;
  const auto* seed_input =
      pick_stage_input_for_index_local(contract.plugins[source_plugin_index], source_output_index);
  const auto* seed_output = pick_stage_output_for_input_local(
      contract.plugins[source_plugin_index], source_output_index, &seed_output_index);
  if (!inspect_stage_io(contract.plugins[source_plugin_index], seed_input, seed_output)
           .has_value()) {
    return std::nullopt;
  }

  std::queue<std::pair<std::size_t, int>> pending;
  std::unordered_set<std::uint64_t> visited;
  pending.emplace(source_plugin_index, source_output_index);

  while (!pending.empty()) {
    const auto [plugin_index, output_index] = pending.front();
    pending.pop();
    const auto visit_key = output_key_local(plugin_index, output_index);
    if (!visited.insert(visit_key).second) {
      continue;
    }

    const auto outgoing_it = outgoing_edges.find(visit_key);
    if (outgoing_it == outgoing_edges.end()) {
      continue;
    }

    for (const auto* edge : outgoing_it->second) {
      if (!edge || edge->dst_plugin_index >= contract.plugins.size()) {
        continue;
      }
      if (!maybe_record_boxdecode_tensor_semantics_from_name_local(
              &facts, edge->tensor_name, error_message, "boxdecode model-managed contract")) {
        return std::nullopt;
      }
      const auto pos_it = execution_positions.find(edge->dst_plugin_index);
      if (pos_it == execution_positions.end() || pos_it->second > terminal_pos) {
        continue;
      }

      const auto& consumer = contract.plugins[edge->dst_plugin_index];
      int next_output_index = -1;
      const auto* input_tensor = pick_stage_input_for_index_local(consumer, edge->dst_input_index);
      const auto* output_tensor =
          pick_stage_output_for_input_local(consumer, edge->dst_input_index, &next_output_index);
      if (!inspect_stage_io(consumer, input_tensor, output_tensor).has_value()) {
        return std::nullopt;
      }

      if (output_tensor && next_output_index >= 0) {
        pending.emplace(edge->dst_plugin_index, next_output_index);
      }
    }
  }

  return facts;
}

bool boxdecode_has_repeated_spatial_groups_local(
    const std::vector<BoxDecodeTensorStaticContract>& tensors) {
  if (tensors.size() <= 2U) {
    return false;
  }
  std::unordered_map<std::uint64_t, std::size_t> counts;
  counts.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    const int tw = tensor.input_shape.size() >= 2 ? tensor.input_shape[1] : 0;
    const int th = tensor.input_shape.size() >= 1 ? tensor.input_shape[0] : 0;
    const std::uint64_t key =
        (static_cast<std::uint64_t>(tw) << 32U) | static_cast<std::uint32_t>(th);
    ++counts[key];
  }
  for (const auto& [_, count] : counts) {
    if (count > 1U) {
      return true;
    }
  }
  return false;
}

int boxdecode_role_sort_order_local(BoxDecodeTensorRoleLocal role) {
  switch (role) {
  case BoxDecodeTensorRoleLocal::Regression:
    return 0;
  case BoxDecodeTensorRoleLocal::Score:
    return 1;
  case BoxDecodeTensorRoleLocal::Unknown:
  default:
    return 2;
  }
}

template <typename T>
void apply_permutation_local(std::vector<T>* values, const std::vector<std::size_t>& order) {
  if (!values || values->size() != order.size()) {
    return;
  }
  std::vector<T> reordered;
  reordered.reserve(values->size());
  for (const auto index : order) {
    reordered.push_back(std::move((*values)[index]));
  }
  *values = std::move(reordered);
}

bool tensor_name_contains_token_local(const std::string& raw_name, const char* token) {
  if (!token || !*token) {
    return false;
  }
  return lower_copy_local(raw_name).find(token) != std::string::npos;
}

bool tensor_name_has_value_preserving_wrapper_prefix_local(const std::string& raw_name) {
  if (raw_name.empty()) {
    return false;
  }
  const std::string lower = lower_copy_local(raw_name);
  const std::string prefix = lower.substr(0, lower.find('/'));
  return prefix == "cast" || prefix.rfind("cast_", 0) == 0 || prefix == "dequant" ||
         prefix.rfind("dequant_", 0) == 0 || prefix.rfind("dequantize_", 0) == 0 ||
         prefix == "pass_through" || prefix.rfind("pass_through_", 0) == 0;
}

bool tensor_name_looks_class_prob_local(const std::string& raw_name) {
  return tensor_name_contains_token_local(raw_name, "class_prob") ||
         tensor_name_contains_token_local(raw_name, "class_probability");
}

bool tensor_name_looks_class_logit_local(const std::string& raw_name) {
  return tensor_name_contains_token_local(raw_name, "class_logit") ||
         tensor_name_contains_token_local(raw_name, "class_logits");
}

bool tensor_name_looks_dequant_stage_local(const std::string& raw_name) {
  return tensor_name_contains_token_local(raw_name, "dequant");
}

bool tensor_name_declares_probability_domain_local(const std::string& raw_name) {
  if (!tensor_name_looks_class_prob_local(raw_name) ||
      tensor_name_looks_dequant_stage_local(raw_name)) {
    return false;
  }

  if (!tensor_name_has_value_preserving_wrapper_prefix_local(raw_name)) {
    return true;
  }

  // Cast/pass-through wrappers preserve the value domain.  Keep class_prob
  // semantics when the producer name is wrapped as "cast_N/class_prob_M" or
  // "pass_through_N/class_prob_M"; only dequant wrappers are filtered above.
  const std::string lower = lower_copy_local(raw_name);
  const std::size_t slash = lower.find('/');
  return slash != std::string::npos && tensor_name_looks_class_prob_local(lower.substr(slash + 1U));
}

std::optional<BoxDecodeScoreActivation>
resolve_boxdecode_score_activation_local(const MpkContract& contract, std::size_t mla_pos,
                                         std::size_t terminal_pos, bool quant_needed,
                                         std::string* error_message) {
  const auto ordered = plugins_in_execution_order(contract);
  bool saw_prob_tensor_pre_dequant = false;
  bool saw_prob_tensor_post_dequant = false;
  bool saw_logit_tensor_pre_dequant = false;
  bool saw_logit_tensor_post_dequant = false;
  bool passed_dequant_stage = false;

  for (std::size_t pos = mla_pos + 1U; pos < terminal_pos; ++pos) {
    if (ordered[pos] >= contract.plugins.size()) {
      continue;
    }
    const auto& plugin = contract.plugins[ordered[pos]];
    passed_dequant_stage = passed_dequant_stage || stage_is_dequant_like_local(plugin);
    for (const auto& tensor : plugin.output_tensors) {
      const std::string name = !tensor.name.empty() ? tensor.name : tensor.segment_name;
      if (tensor_name_declares_probability_domain_local(name)) {
        if (passed_dequant_stage) {
          saw_prob_tensor_post_dequant = true;
        } else {
          saw_prob_tensor_pre_dequant = true;
        }
      }
      if (tensor_name_looks_class_logit_local(name)) {
        if (passed_dequant_stage) {
          saw_logit_tensor_post_dequant = true;
        } else {
          saw_logit_tensor_pre_dequant = true;
        }
      }
    }
  }

  const bool saw_prob_tensor = saw_prob_tensor_pre_dequant || saw_prob_tensor_post_dequant;
  const bool saw_logit_tensor = saw_logit_tensor_pre_dequant || saw_logit_tensor_post_dequant;
  if (saw_prob_tensor && saw_logit_tensor) {
    set_error(error_message,
              "boxdecode score activation is ambiguous: MPK post chain exposes both class_prob "
              "and class_logit tensors");
    return std::nullopt;
  }
  if (saw_logit_tensor) {
    return BoxDecodeScoreActivation::Sigmoid;
  }
  if (saw_prob_tensor_pre_dequant) {
    return BoxDecodeScoreActivation::Identity;
  }
  if (quant_needed && saw_prob_tensor_post_dequant) {
    // A downstream dequant stage does not apply sigmoid. When the selected external boxdecode
    // boundary is still quantized, class_prob naming that appears only after dequant refers to the
    // dequantized view, so the quantized boundary still requires sigmoid in boxdecode.
    return BoxDecodeScoreActivation::Sigmoid;
  }
  if (saw_prob_tensor_post_dequant) {
    return BoxDecodeScoreActivation::Identity;
  }
  if (quant_needed) {
    set_error(error_message,
              "boxdecode quantized route requires explicit class score activation semantics in "
              "the MPK tensor lineage");
    return std::nullopt;
  }
  set_error(error_message,
            "boxdecode route requires explicit class score activation semantics in the MPK tensor "
            "lineage");
  return std::nullopt;
}

std::optional<BoxDecodeScoreActivation>
resolve_boxdecode_score_activation_for_selected_tensors_local(
    const std::vector<MpkTensorContract>& tensors, std::string* error_message) {
  bool saw_prob_tensor = false;
  bool saw_logit_tensor = false;
  for (const auto& tensor : tensors) {
    const std::string name = !tensor.name.empty() ? tensor.name : tensor.segment_name;
    saw_prob_tensor = saw_prob_tensor || tensor_name_declares_probability_domain_local(name);
    saw_logit_tensor = saw_logit_tensor || tensor_name_looks_class_logit_local(name);
  }
  if (saw_prob_tensor && saw_logit_tensor) {
    set_error(error_message,
              "boxdecode score activation is ambiguous: selected input tensors expose both "
              "class_prob and class_logit semantics");
    return std::nullopt;
  }
  if (saw_prob_tensor) {
    return BoxDecodeScoreActivation::Identity;
  }
  if (saw_logit_tensor) {
    return BoxDecodeScoreActivation::Sigmoid;
  }
  return std::nullopt;
}

std::optional<bool> resolve_external_boxdecode_tess_needed_local(
    const MpkContract& contract, const MpkPluginIoContract& mla_stage,
    const std::vector<MpkTensorContract>& logical_outputs,
    const ModelManagedRouteFlags& route_flags, std::string* error_message) {
  auto bypass_mla_unpack_enabled = []() -> bool {
    const char* raw = std::getenv("SIMA_BOXDECODE_BYPASS_MLA_UNPACK");
    return raw && *raw && std::strcmp(raw, "0") != 0;
  };
  const auto* unpack_stage = get_mla_unpack_stage_io_contract(contract);
  if (bypass_mla_unpack_enabled()) {
    const bool packed_single_physical_mla =
        mla_stage.output_tensors.size() == 1U && logical_outputs.size() > 1U;
    if (!packed_single_physical_mla) {
      set_error(error_message,
                "external/raw boxdecode bypass requires one packed MLA parent and multiple "
                "logical outputs");
      return std::nullopt;
    }
    return true;
  }
  if (unpack_stage) {
    if (unpack_stage->output_tensors.empty()) {
      set_error(error_message,
                "external boxdecode requires explicit unpacked MLA logical outputs; unpack stage "
                "has no outputs");
      return std::nullopt;
    }
    if (logical_outputs.size() != unpack_stage->output_tensors.size()) {
      set_error(error_message,
                "external boxdecode requires one boundary tensor per MLA unpack output");
      return std::nullopt;
    }
    for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
      int h = 0;
      int w = 0;
      int c = 0;
      const auto& tensor = logical_outputs[i];
      const std::vector<std::int64_t>& shape =
          !tensor.logical_shape.empty() ? tensor.logical_shape : tensor.mpk_shape;
      if (!dims_from_mpk_shape_for_input_nhwc_local(shape, &h, &w, &c)) {
        set_error(error_message,
                  "external boxdecode requires structured unpacked MLA tensor geometry for every "
                  "selected input");
        return std::nullopt;
      }
      if (h <= 0 || w <= 0 || c <= 0) {
        set_error(error_message,
                  "external boxdecode requires non-zero unpacked MLA tensor geometry");
        return std::nullopt;
      }
    }
    // Preserve the typed route flags here. An explicit unpack boundary does not authorize
    // external boxdecode to reinterpret downstream detess lineage; unsupported MLA unpack
    // publication must fail in the MLA runtime instead of being masked by decode-mode coercion.
    return route_flags.tess_needed;
  }

  const bool packed_single_physical_mla =
      mla_stage.output_tensors.size() == 1U && logical_outputs.size() > 1U;
  if (packed_single_physical_mla || route_flags.tess_needed) {
    set_error(error_message,
              "external boxdecode requires explicit unpacked MLA logical outputs; raw "
              "tessellated/picked MLA exports are unsupported");
    return std::nullopt;
  }
  return false;
}

std::string upper_copy_local(std::string v) {
  for (char& c : v) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return v;
}

bool dtype_is_bf16_like(std::string raw) {
  raw = upper_copy_local(std::move(raw));
  return raw.find("BF16") != std::string::npos || raw.find("BFLOAT16") != std::string::npos;
}

bool dtype_is_quantized_like(std::string raw) {
  raw = upper_copy_local(std::move(raw));
  if (raw.empty() || dtype_is_bf16_like(raw)) {
    return false;
  }
  return raw.find("INT8") != std::string::npos || raw.find("UINT8") != std::string::npos ||
         raw.find("INT16") != std::string::npos || raw.find("UINT16") != std::string::npos ||
         raw.find("INT32") != std::string::npos || raw.find("UINT32") != std::string::npos;
}

bool quant_contract_complete(const std::optional<MpkQuantContract>& quant) {
  return quant.has_value() && !quant->scales.empty() && !quant->zero_points.empty();
}

std::optional<std::size_t> plugin_index_from_pointer(const MpkContract& contract,
                                                     const MpkPluginIoContract* stage) {
  if (!stage) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    if (&contract.plugins[i] == stage) {
      return i;
    }
  }
  return std::nullopt;
}

bool mla_outputs_are_quantized_like(const MpkContract& contract) {
  const auto* mla_stage = get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    return false;
  }
  for (const auto& tensor : get_mla_logical_outputs_contract(contract)) {
    const std::string dtype = !tensor.logical_dtype.empty() ? tensor.logical_dtype : tensor.dtype;
    if (dtype_is_quantized_like(normalize_mpk_dtype_token_local(dtype))) {
      return true;
    }
  }
  for (const auto& tensor : mla_stage->output_tensors) {
    const std::string dtype = !tensor.logical_dtype.empty() ? tensor.logical_dtype : tensor.dtype;
    if (dtype_is_quantized_like(normalize_mpk_dtype_token_local(dtype))) {
      return true;
    }
  }
  return dtype_is_quantized_like(
      normalize_mpk_dtype_token_local(mla_stage->canonical_output_dtype));
}

std::optional<std::pair<std::size_t, std::size_t>>
mla_and_terminal_positions(const MpkContract& contract, const MpkPluginIoContract* terminal_stage,
                           std::string* error_message) {
  const auto* mla_stage = get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    set_error(error_message,
              "boxdecode model-managed facts require an upstream MLA stage contract");
    return std::nullopt;
  }
  const auto mla_index = plugin_index_from_pointer(contract, mla_stage);
  if (!mla_index.has_value()) {
    set_error(error_message, "boxdecode model-managed facts could not resolve MLA stage index");
    return std::nullopt;
  }

  const auto ordered = plugins_in_execution_order(contract);
  auto find_position = [&](std::size_t plugin_index) -> std::optional<std::size_t> {
    for (std::size_t pos = 0; pos < ordered.size(); ++pos) {
      if (ordered[pos] == plugin_index) {
        return pos;
      }
    }
    return std::nullopt;
  };

  const auto mla_pos = find_position(*mla_index);
  if (!mla_pos.has_value()) {
    set_error(error_message, "boxdecode model-managed facts could not resolve MLA execution order");
    return std::nullopt;
  }

  if (!terminal_stage) {
    return std::make_pair(*mla_pos, ordered.size());
  }

  const auto terminal_index = plugin_index_from_pointer(contract, terminal_stage);
  if (!terminal_index.has_value()) {
    set_error(error_message,
              "boxdecode model-managed facts could not resolve terminal stage index");
    return std::nullopt;
  }
  const auto terminal_pos = find_position(*terminal_index);
  if (!terminal_pos.has_value()) {
    set_error(error_message,
              "boxdecode model-managed facts could not resolve terminal stage execution order");
    return std::nullopt;
  }
  if (*terminal_pos <= *mla_pos) {
    set_error(error_message, "boxdecode model-managed facts require a terminal stage after MLA");
    return std::nullopt;
  }
  return std::make_pair(*mla_pos, *terminal_pos);
}

void set_error(std::string* error_message, const std::string& message) {
  if (error_message) {
    *error_message = message;
  }
}

} // namespace

ModelManagedRouteFlags
model_route_flags_from_boxdecode_contract(const BoxDecodeStaticContract& contract) {
  ModelManagedRouteFlags flags;
  flags.quant_needed = contract.quant_needed;
  flags.tess_needed = contract.tess_needed;
  flags.quant_contract_required = contract.quant_needed;
  flags.boxdecode_selected = true;
  return flags;
}

ModelManagedRouteFlags
model_route_flags_from_boxdecode_semantics(const ModelBoxdecodeSemantics& semantics) {
  ModelManagedRouteFlags flags;
  flags.quant_needed = semantics.quant_needed;
  flags.tess_needed = semantics.tess_needed;
  flags.quant_contract_required = semantics.quant_contract_required;
  flags.boxdecode_selected = true;
  return flags;
}

std::optional<ModelManagedRouteFlags>
resolve_model_managed_boxdecode_route_flags_from_mpk(const MpkContract& contract,
                                                     const MpkPluginIoContract* terminal_stage,
                                                     std::string* error_message) {
  set_error(error_message, {});
  const auto positions = mla_and_terminal_positions(contract, terminal_stage, error_message);
  if (!positions.has_value()) {
    return std::nullopt;
  }

  const auto ordered = plugins_in_execution_order(contract);
  const auto [mla_pos, terminal_pos] = *positions;
  ModelManagedRouteFlags flags;
  flags.boxdecode_selected = true;
  for (std::size_t pos = mla_pos + 1U; pos < terminal_pos; ++pos) {
    if (ordered[pos] >= contract.plugins.size()) {
      continue;
    }
    const auto& plugin = contract.plugins[ordered[pos]];
    flags.tess_needed = flags.tess_needed || stage_is_detess_like_local(plugin);
    flags.quant_needed = flags.quant_needed || stage_is_dequant_like_local(plugin);
  }

  const auto* mla_stage = get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    set_error(error_message,
              "boxdecode model-managed facts require an upstream MLA stage contract");
    return std::nullopt;
  }
  if (!flags.quant_needed &&
      (quant_contract_complete(mla_stage->quant) || mla_outputs_are_quantized_like(contract))) {
    flags.quant_needed = true;
  }
  flags.quant_contract_required = flags.quant_needed;
  return flags;
}

std::optional<BoxDecodeStaticContract>
build_boxdecode_static_contract_from_mpk(const MpkContract& contract,
                                         const ModelManagedRouteFlags& route_flags,
                                         std::string* error_message) {
  return build_boxdecode_static_contract_from_mpk(contract, route_flags, nullptr, error_message);
}

std::optional<BoxDecodeStaticContract> build_boxdecode_static_contract_from_mpk(
    const MpkContract& contract, const ModelManagedRouteFlags& route_flags,
    const MpkPluginIoContract* terminal_stage, std::string* error_message) {
  auto bypass_mla_unpack_enabled = []() -> bool {
    const char* raw = std::getenv("SIMA_BOXDECODE_BYPASS_MLA_UNPACK");
    return raw && *raw && std::strcmp(raw, "0") != 0;
  };
  auto fail = [&](std::string message) -> std::optional<BoxDecodeStaticContract> {
    set_error(error_message, message);
    return std::nullopt;
  };
  set_error(error_message, {});

  const auto* mla_stage = get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    return fail("boxdecode model-managed contract requires an upstream MLA stage contract");
  }
  const auto positions = mla_and_terminal_positions(contract, terminal_stage, error_message);
  if (!positions.has_value()) {
    return std::nullopt;
  }
  const auto ordered = plugins_in_execution_order(contract);
  const auto [mla_pos, terminal_pos] = *positions;
  const auto mla_index = plugin_index_from_pointer(contract, mla_stage);
  if (!mla_index.has_value()) {
    return fail("boxdecode model-managed contract could not resolve MLA stage index");
  }
  const auto execution_positions = build_execution_positions_local(ordered);
  const auto outgoing_edges = build_outgoing_edges_local(contract);

  std::vector<MpkTensorContract> logical_outputs = get_mla_logical_outputs_contract(contract);
  if (logical_outputs.empty()) {
    return fail("boxdecode model-managed contract requires upstream MLA logical outputs");
  }

  BoxDecodeStaticContract out;
  out.score_activation = BoxDecodeScoreActivation::Unknown;
  const auto tess_needed =
      terminal_stage ? std::optional<bool>(route_flags.tess_needed)
                     : resolve_external_boxdecode_tess_needed_local(
                           contract, *mla_stage, logical_outputs, route_flags, error_message);
  if (!tess_needed.has_value()) {
    return std::nullopt;
  }
  out.tess_needed = *tess_needed;
  out.quant_needed = route_flags.quant_needed;
  out.topk = 0;
  out.detection_threshold = 0.0;
  out.nms_iou_threshold = 0.0;
  out.input_dtype = normalize_mpk_dtype_token_local(mla_stage->canonical_output_dtype);
  if (out.input_dtype.empty()) {
    out.input_dtype = normalize_mpk_dtype_token_local(mla_stage->frame_type);
  }
  const bool bypass_unpack_boundary = terminal_stage == nullptr && bypass_mla_unpack_enabled();
  const auto* unpack_stage = get_mla_unpack_stage_io_contract(contract);
  const bool explicit_unpack_boundary =
      !bypass_unpack_boundary && unpack_stage && !unpack_stage->output_tensors.empty() &&
      unpack_stage->output_tensors.size() == logical_outputs.size();
  const bool preserve_raw_packed_parent_source =
      terminal_stage == nullptr && (bypass_unpack_boundary || !explicit_unpack_boundary) &&
      mla_stage->output_tensors.size() == 1U;

  std::vector<std::pair<std::size_t, int>> lineage_roots;
  lineage_roots.reserve(logical_outputs.size());
  const auto* mla_fanout_edge = [&]() -> const MpkContractEdge* {
    const MpkContractEdge* best = nullptr;
    std::size_t best_order = std::numeric_limits<std::size_t>::max();
    for (const auto& edge : contract.edges) {
      if (edge.src_plugin_index != *mla_index || edge.src_output_index != 0) {
        continue;
      }
      if (edge.dst_plugin_index >= contract.plugins.size()) {
        continue;
      }
      const std::size_t order = execution_positions.count(edge.dst_plugin_index)
                                    ? execution_positions.at(edge.dst_plugin_index)
                                    : std::numeric_limits<std::size_t>::max();
      if (!best || order < best_order) {
        best = &edge;
        best_order = order;
      }
    }
    return best;
  }();
  if (!bypass_unpack_boundary && mla_fanout_edge &&
      mla_fanout_edge->dst_plugin_index < contract.plugins.size()) {
    const auto& downstream = contract.plugins[mla_fanout_edge->dst_plugin_index];
    if (lower_copy_local(downstream.kernel).find("unpack") != std::string::npos &&
        downstream.output_tensors.size() == logical_outputs.size()) {
      for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
        lineage_roots.emplace_back(mla_fanout_edge->dst_plugin_index, static_cast<int>(i));
      }
    }
  }
  if (lineage_roots.size() != logical_outputs.size()) {
    lineage_roots.clear();
    for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
      lineage_roots.emplace_back(*mla_index, static_cast<int>(i));
    }
  }
  for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
    if (const auto resolved = resolve_boxdecode_tensor_source_local(
            contract, execution_positions, logical_outputs[i], mla_pos, terminal_pos)) {
      lineage_roots[i] = *resolved;
    }
  }

  out.tensors.reserve(logical_outputs.size());
  out.tensor_names.reserve(logical_outputs.size());
  for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
    const auto& tensor = logical_outputs[i];
    const std::vector<std::int64_t> logical_shape =
        !tensor.logical_shape.empty() ? tensor.logical_shape : tensor.mpk_shape;
    int h = 0;
    int w = 0;
    int c = 0;
    if (!dims_from_mpk_shape_for_input_nhwc_local(logical_shape, &h, &w, &c)) {
      return fail("boxdecode model-managed contract requires explicit upstream tensor geometry");
    }
    const std::string entry_dtype = normalize_mpk_dtype_token_local(
        !tensor.logical_dtype.empty() ? tensor.logical_dtype : tensor.dtype);
    const int elem_bytes = dtype_size_bytes_local(entry_dtype);

    int physical_h = h;
    int physical_w = w;
    int physical_c = c;
    if (elem_bytes > 0 && tensor.stride_bytes.size() == logical_shape.size() &&
        tensor.stride_bytes.size() >= 3U) {
      // MLA boundary views published after an unpack+slice keep the logical shape
      // (for example [1,80,80,4]) but preserve the parent storage strides
      // (for example [102400,1280,16,2] for BF16 [1,80,80,8]).  Reuse those
      // existing MPK stride facts: input_shape is the physical/storage HWC view
      // consumed by objectdecode, slice_shape is the logical region to decode.
      const std::size_t rank = tensor.stride_bytes.size();
      const std::int64_t stride_w = tensor.stride_bytes[rank - 2U];
      const std::int64_t stride_c = tensor.stride_bytes[rank - 1U];
      if (stride_c == elem_bytes && stride_w > 0 &&
          (stride_w % static_cast<std::int64_t>(elem_bytes)) == 0) {
        const auto inferred_physical_c =
            static_cast<int>(stride_w / static_cast<std::int64_t>(elem_bytes));
        if (inferred_physical_c >= c) {
          physical_c = inferred_physical_c;
        }
      }
    }
    const std::uint64_t physical_size_bytes =
        elem_bytes > 0
            ? static_cast<std::uint64_t>(physical_h) * static_cast<std::uint64_t>(physical_w) *
                  static_cast<std::uint64_t>(physical_c) * static_cast<std::uint64_t>(elem_bytes)
            : tensor.size_bytes;

    BoxDecodeTensorStaticContract entry;
    entry.input_shape = {physical_h, physical_w, physical_c};
    entry.slice_shape = {h, w, c};
    entry.data_type = entry_dtype;
    entry.layout = "HWC";
    if (entry.data_type.empty()) {
      return fail("boxdecode model-managed contract requires explicit upstream tensor dtype");
    }
    if (tensor.name.empty() && tensor.segment_name.empty()) {
      return fail("boxdecode model-managed contract requires explicit upstream tensor name");
    }
    entry.logical_name = !tensor.name.empty() ? tensor.name : tensor.segment_name;
    entry.backend_name = entry.logical_name;
    entry.source_logical_output_index =
        tensor.tensor_index >= 0 ? tensor.tensor_index : static_cast<int>(i);
    entry.source_output_slot = tensor.tensor_index >= 0 ? tensor.tensor_index : static_cast<int>(i);
    const bool published_segment_distinct =
        !tensor.name.empty() && !tensor.segment_name.empty() && tensor.name != tensor.segment_name;
    const int source_physical_index =
        tensor.source_physical_index >= 0 ? tensor.source_physical_index : tensor.physical_index;
    std::string source_parent_segment_name;
    if (source_physical_index >= 0 && mla_index.has_value() &&
        static_cast<std::size_t>(source_physical_index) <
            contract.plugins[*mla_index].output_tensors.size()) {
      const auto& source_tensor =
          contract.plugins[*mla_index]
              .output_tensors[static_cast<std::size_t>(source_physical_index)];
      source_parent_segment_name =
          !source_tensor.segment_name.empty()
              ? source_tensor.segment_name
              : (!source_tensor.name.empty() ? source_tensor.name : std::string());
    }
    // Boxdecode consumes logical tensor views from the runtime TensorBuffer contract. The
    // byte offset that matters here is therefore the published view offset within the bound
    // source segment, not the parent-carrier base offset. Preserving the logical view offset is
    // what allows grouped MLA heads that share one parent segment to resolve each sub-view at the
    // correct position instead of re-reading every head from byte offset 0 of the parent.
    std::int64_t source_byte_offset = tensor.source_byte_offset;
    if (source_byte_offset == 0 && tensor.byte_offset > 0) {
      source_byte_offset = tensor.byte_offset;
    }
    if ((preserve_raw_packed_parent_source || explicit_unpack_boundary) &&
        !source_parent_segment_name.empty()) {
      entry.source_segment_name = source_parent_segment_name;
    } else {
      entry.source_segment_name =
          published_segment_distinct
              ? tensor.name
              : (!tensor.segment_name.empty() ? tensor.segment_name : entry.logical_name);
    }
    entry.source_physical_index = source_physical_index;
    entry.source_byte_offset = source_byte_offset;
    entry.source_size_bytes = physical_size_bytes > 0U ? physical_size_bytes : tensor.size_bytes;
    out.tensors.push_back(std::move(entry));
    out.tensor_names.push_back(out.tensors.back().logical_name);
    BoxDecodePhysicalInputStaticContract physical_input;
    if ((preserve_raw_packed_parent_source || explicit_unpack_boundary) &&
        !source_parent_segment_name.empty()) {
      physical_input.name = source_parent_segment_name;
    }
    if (physical_input.name.empty()) {
      physical_input.name =
          explicit_unpack_boundary
              ? out.tensors.back().source_segment_name
              : (!tensor.segment_name.empty() ? tensor.segment_name
                                              : out.tensors.back().source_segment_name);
    }
    physical_input.physical_index = out.tensors.back().source_physical_index;
    physical_input.byte_offset = out.tensors.back().source_byte_offset;
    physical_input.size_bytes = out.tensors.back().source_size_bytes;
    out.physical_inputs.push_back(std::move(physical_input));
  }

  if (out.tensors.empty()) {
    return fail("boxdecode model-managed contract resolved no upstream tensors");
  }
  if (out.input_dtype.empty()) {
    out.input_dtype = out.tensors.front().data_type;
  }

  std::vector<BoxDecodeTensorLineageFactsLocal> lineage_facts;
  lineage_facts.reserve(out.tensors.size());
  for (std::size_t i = 0; i < lineage_roots.size(); ++i) {
    const auto facts = collect_boxdecode_tensor_lineage_facts_local(
        contract, execution_positions, outgoing_edges, lineage_roots[i].first,
        lineage_roots[i].second, terminal_pos, error_message);
    if (!facts.has_value()) {
      return std::nullopt;
    }
    lineage_facts.push_back(*facts);
  }

  if (out.tess_needed) {
    for (std::size_t i = 0; i < out.tensors.size(); ++i) {
      if (!lineage_facts[i].detess_slice.has_value()) {
        return fail("boxdecode tessellated route requires upstream detess slice facts for every "
                    "input tensor");
      }
      // detess_slice is {w, h, c}; slice_shape convention is {h, w, c}.
      out.tensors[i].slice_shape = {static_cast<int>((*lineage_facts[i].detess_slice)[1]),
                                    static_cast<int>((*lineage_facts[i].detess_slice)[0]),
                                    static_cast<int>((*lineage_facts[i].detess_slice)[2])};
    }
    out.input_dtype = out.tensors.front().data_type;

    // A model-managed tessellated BoxDecode binds to the physical MPK transport buffer, not to the
    // semantic detess output view.  The MPK detess stage declares both sides of that contract:
    //   * slice_shape is the logical decode view (for example 80x80x4 boxes).
    //   * input tensor size/frame_shape/align_c16 describe the packed source view consumed by the
    //     next plugin (for example 80x80x16 boxes).
    // Use those explicit MPK facts instead of inferring padding heuristically so INT8/BF16 and
    // direct/tess routes share the same source-of-truth.
    const bool detess_transport_any =
        std::any_of(lineage_facts.begin(), lineage_facts.end(), [](const auto& facts) {
          return facts.detess_transport_input_hwc.has_value() ||
                 facts.detess_transport_size_bytes.has_value();
        });
    const bool detess_transport_all =
        !lineage_facts.empty() &&
        std::all_of(lineage_facts.begin(), lineage_facts.end(), [](const auto& facts) {
          return facts.detess_transport_input_hwc.has_value() &&
                 facts.detess_transport_size_bytes.has_value();
        });
    if ((preserve_raw_packed_parent_source || explicit_unpack_boundary) && detess_transport_any) {
      if (!detess_transport_all) {
        return fail("boxdecode tessellated route requires MPK detess transport facts for every "
                    "packed input tensor");
      }
      for (std::size_t i = 0; i < out.tensors.size(); ++i) {
        auto& tensor = out.tensors[i];
        const auto& physical_hwc = *lineage_facts[i].detess_transport_input_hwc;
        tensor.input_shape = {physical_hwc[0], physical_hwc[1], physical_hwc[2]};
        tensor.source_size_bytes = *lineage_facts[i].detess_transport_size_bytes;
        if (i < out.physical_inputs.size()) {
          out.physical_inputs[i].size_bytes = tensor.source_size_bytes;
        }
      }

      const bool packed_single_mla_parent =
          mla_stage->output_tensors.size() == 1U && out.tensors.size() > 1U;
      if (packed_single_mla_parent) {
        std::uint64_t source_byte_offset = 0;
        for (std::size_t i = 0; i < out.tensors.size(); ++i) {
          if (source_byte_offset >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return fail("boxdecode packed tensor source byte offset overflows int64");
          }
          out.tensors[i].source_byte_offset = static_cast<std::int64_t>(source_byte_offset);
          if (i < out.physical_inputs.size()) {
            out.physical_inputs[i].byte_offset = out.tensors[i].source_byte_offset;
          }
          source_byte_offset += out.tensors[i].source_size_bytes;
        }
        const auto parent_size =
            static_cast<std::uint64_t>(mla_stage->output_tensors.front().size_bytes);
        if (parent_size > 0U && source_byte_offset != parent_size) {
          return fail("boxdecode packed tensor byte spans do not sum to MPK MLA output size: "
                      "sum=" +
                      std::to_string(source_byte_offset) +
                      " parent=" + std::to_string(parent_size));
        }
      }
    }
  }

  if (route_flags.quant_needed) {
    std::vector<double> quant_scales;
    std::vector<std::int64_t> quant_zps;
    const bool branch_quant_complete =
        !lineage_facts.empty() &&
        std::all_of(lineage_facts.begin(), lineage_facts.end(), [](const auto& facts) {
          return facts.dq_scale.has_value() && facts.dq_zp.has_value();
        });
    const bool branch_quant_any =
        std::any_of(lineage_facts.begin(), lineage_facts.end(), [](const auto& facts) {
          return facts.dq_scale.has_value() || facts.dq_zp.has_value();
        });
    if (branch_quant_any && !branch_quant_complete) {
      return fail("boxdecode quantized route requires explicit per-branch quant facts for every "
                  "input tensor");
    }
    if (branch_quant_complete) {
      for (std::size_t i = 0; i < out.tensors.size(); ++i) {
        quant_scales.push_back(*lineage_facts[i].dq_scale);
        quant_zps.push_back(*lineage_facts[i].dq_zp);
      }
    } else if (quant_contract_complete(mla_stage->quant)) {
      for (std::size_t i = 0; i < out.tensors.size(); ++i) {
        const int raw_index = out.tensors[i].source_output_slot >= 0
                                  ? out.tensors[i].source_output_slot
                                  : out.tensors[i].source_logical_output_index;
        if (raw_index < 0 ||
            static_cast<std::size_t>(raw_index) >= mla_stage->quant->scales.size() ||
            static_cast<std::size_t>(raw_index) >= mla_stage->quant->zero_points.size()) {
          return fail("boxdecode quantized route requires upstream MLA quant facts for every "
                      "input tensor");
        }
        quant_scales.push_back(mla_stage->quant->scales[static_cast<std::size_t>(raw_index)]);
        quant_zps.push_back(mla_stage->quant->zero_points[static_cast<std::size_t>(raw_index)]);
      }
    }
    if (quant_scales.size() < out.tensors.size() || quant_zps.size() < out.tensors.size()) {
      return fail("boxdecode quantized route requires upstream quant contract facts");
    }
    for (std::size_t i = 0; i < out.tensors.size(); ++i) {
      if (quant_scales[i] == 0.0) {
        return fail("boxdecode quantized route requires non-zero upstream dq_scale facts");
      }
    }
    out.dq_scale = std::move(quant_scales);
    out.dq_zp = std::move(quant_zps);
  } else {
    out.dq_scale.assign(out.tensors.size(), 1.0);
    out.dq_zp.assign(out.tensors.size(), 0);
  }

  if (route_flags.quant_needed && !out.tensors.empty()) {
    std::string quant_tensor_dtype;
    for (const auto& tensor : out.tensors) {
      if (dtype_is_quantized_like_local(tensor.data_type)) {
        quant_tensor_dtype = normalize_mpk_dtype_token_local(tensor.data_type);
        break;
      }
    }
    if (quant_tensor_dtype.empty()) {
      if (dtype_is_quantized_like_local(out.input_dtype)) {
        quant_tensor_dtype = normalize_mpk_dtype_token_local(out.input_dtype);
      } else {
        return fail("boxdecode quantized route requires explicit upstream quantized tensor dtype");
      }
    }
    for (std::size_t i = 0; i < out.tensors.size(); ++i) {
      if (!dtype_is_quantized_like_local(out.tensors[i].data_type)) {
        return fail("boxdecode quantized route requires quantized upstream tensor dtype for tensor "
                    "index " +
                    std::to_string(i));
      }
    }
    if (!dtype_is_quantized_like_local(out.input_dtype)) {
      out.input_dtype = quant_tensor_dtype;
    }
  }

  maybe_restore_boxdecode_semantic_names_from_lineage_local(&out, lineage_facts);
  maybe_restore_grouped_role_semantic_names_from_structure_local(&out);
  maybe_infer_score_activation_from_boxdecode_contract_local(&out);
  maybe_override_quantized_yolov8_score_activation_local(&out);
  maybe_default_float_yolo_score_activation_local(&out);
  maybe_infer_yolov8_decode_type_option_local(&out);
  apply_raw_yolov6_yolox_contract_overrides_local(&out);

  return out;
}

std::optional<BoxDecodeStaticContract> build_boxdecode_static_contract_from_compiled_upstream(
    const simaai::neat::CompiledNodeContract& upstream_stage, BoxDecodeType decode_type,
    std::string* error_message) {
  auto fail = [&](std::string message) -> std::optional<BoxDecodeStaticContract> {
    set_error(error_message, std::move(message));
    return std::nullopt;
  };
  set_error(error_message, {});

  const auto* runtime = compiled_runtime_contract_from_stage(&upstream_stage);
  if (!runtime) {
    return fail("boxdecode inferred contract requires typed upstream runtime contract");
  }
  if (runtime->logical_outputs.empty()) {
    return fail("boxdecode inferred contract requires upstream logical output tensors");
  }

  const auto ordered = ordered_logical_outputs_from_runtime_contract_local(*runtime);
  if (ordered.empty()) {
    return fail("boxdecode inferred contract requires ordered upstream logical outputs");
  }

  BoxDecodeStaticContract out;
  out.decode_type = decode_type;
  out.score_activation = BoxDecodeScoreActivation::Unknown;

  out.input_dtype = normalize_mpk_dtype_token_local(ordered.front()->dtype);
  if (out.input_dtype.empty()) {
    return fail("boxdecode inferred contract requires explicit upstream tensor dtype");
  }
  out.quant_needed = dtype_is_quantized_like_local(out.input_dtype);
  out.tess_needed = false;

  out.tensors.reserve(ordered.size());
  out.tensor_names.reserve(ordered.size());
  out.physical_inputs.reserve(ordered.size());
  out.dq_scale.reserve(ordered.size());
  out.dq_zp.reserve(ordered.size());

  for (std::size_t i = 0; i < ordered.size(); ++i) {
    const auto& logical = *ordered[i];
    int h = 0;
    int w = 0;
    int c = 0;
    if (!dims_from_logical_output_local(logical, &h, &w, &c)) {
      return fail("boxdecode inferred contract requires explicit upstream tensor geometry for "
                  "tensor index " +
                  std::to_string(i));
    }

    const std::string tensor_name = tensor_name_from_logical_output_local(logical, i);
    const std::string dtype = normalize_mpk_dtype_token_local(logical.dtype);
    if (dtype.empty()) {
      return fail("boxdecode inferred contract requires explicit upstream tensor dtype for tensor "
                  "index " +
                  std::to_string(i));
    }

    int output_slot = static_cast<int>(i);
    for (const auto& route : runtime->output_order) {
      if (route.logical_output_index == logical.logical_index) {
        output_slot = route.output_slot;
        break;
      }
    }
    if (logical.output_slot >= 0) {
      output_slot = logical.output_slot;
    }

    BoxDecodeTensorStaticContract entry;
    entry.input_shape = {h, w, c};
    entry.slice_shape = {h, w, c};
    entry.data_type = dtype;
    entry.layout = logical.layout;
    entry.logical_name = !logical.logical_name.empty() ? logical.logical_name : tensor_name;
    entry.backend_name = !logical.backend_name.empty() ? logical.backend_name : tensor_name;
    entry.source_segment_name = !logical.segment_name.empty() ? logical.segment_name : tensor_name;
    entry.source_logical_output_index =
        logical.logical_index >= 0 ? logical.logical_index : static_cast<int>(i);
    entry.source_output_slot = output_slot;
    entry.source_physical_index = logical.physical_index;
    entry.source_byte_offset = logical.byte_offset;
    entry.source_size_bytes = logical.size_bytes;
    out.tensors.push_back(entry);
    out.tensor_names.push_back(entry.logical_name);

    BoxDecodePhysicalInputStaticContract physical;
    physical.name = entry.source_segment_name;
    physical.physical_index = entry.source_physical_index;
    physical.byte_offset = entry.source_byte_offset;
    physical.size_bytes = entry.source_size_bytes;
    out.physical_inputs.push_back(std::move(physical));

    if (logical.quant.has_value() && !logical.quant->scales.empty() &&
        !logical.quant->zero_points.empty()) {
      out.dq_scale.push_back(logical.quant->scales.front());
      out.dq_zp.push_back(logical.quant->zero_points.front());
    } else if (!out.quant_needed) {
      out.dq_scale.push_back(1.0);
      out.dq_zp.push_back(0);
    } else {
      return fail("boxdecode inferred quantized contract requires upstream q_scale/q_zp for "
                  "tensor index " +
                  std::to_string(i));
    }
  }

  maybe_infer_score_activation_from_boxdecode_contract_local(&out);
  maybe_override_quantized_yolov8_score_activation_local(&out);
  maybe_default_float_yolo_score_activation_local(&out);
  maybe_infer_yolov8_decode_type_option_local(&out);
  apply_raw_yolov6_yolox_contract_overrides_local(&out);

  return out;
}

std::optional<BoxDecodeStaticContract>
build_boxdecode_static_contract_from_sample(const Sample& sample, BoxDecodeType decode_type,
                                            const std::optional<InputContract>& input_contract,
                                            std::string* error_message) {
  auto fail = [&](std::string message) -> std::optional<BoxDecodeStaticContract> {
    set_error(error_message, std::move(message));
    return std::nullopt;
  };
  set_error(error_message, {});

  TensorList tensors = tensors_from_sample(sample, true);
  if (tensors.empty()) {
    return fail("boxdecode inferred contract requires at least one tensor input sample");
  }
  maybe_reorder_tensors_by_logical_index_local(&tensors);

  BoxDecodeStaticContract out;
  out.decode_type = decode_type;
  out.score_activation = BoxDecodeScoreActivation::Unknown;

  std::string input_dtype = input_contract.has_value()
                                ? normalize_mpk_dtype_token_local(input_contract->dtype)
                                : std::string();
  if (input_dtype.empty()) {
    input_dtype =
        normalize_mpk_dtype_token_local(dtype_token_from_tensor_local(tensors.front().dtype));
  }
  if (input_dtype.empty()) {
    return fail("boxdecode inferred contract requires explicit tensor dtype");
  }
  out.input_dtype = input_dtype;
  out.quant_needed = dtype_is_quantized_like_local(input_dtype);
  out.tess_needed = false;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const std::string tensor_name = tensor_name_from_sample_local(tensors[i], i);
    // Standalone boxdecode samples can carry stale tess lineage after detess/dequant boundaries.
    // Keep the tess route only when the published sample still looks like a raw MLA head export.
    if (sample_tensor_semantics_require_tess_local(tensors[i], tensor_name, input_dtype,
                                                   input_contract)) {
      out.tess_needed = true;
      break;
    }
  }

  out.tensors.reserve(tensors.size());
  out.tensor_names.reserve(tensors.size());
  out.physical_inputs.reserve(tensors.size());
  out.dq_scale.reserve(tensors.size());
  out.dq_zp.reserve(tensors.size());

  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    int h = 0;
    int w = 0;
    int c = 0;
    if (!dims_from_tensor_local(tensor, input_contract, &h, &w, &c)) {
      return fail(
          "boxdecode inferred contract requires explicit tensor geometry for tensor index " +
          std::to_string(i));
    }

    const std::string tensor_name = tensor_name_from_sample_local(tensor, i);
    const std::string dtype =
        input_dtype.empty()
            ? normalize_mpk_dtype_token_local(dtype_token_from_tensor_local(tensor.dtype))
            : input_dtype;
    if (dtype.empty()) {
      return fail("boxdecode inferred contract requires explicit tensor dtype for tensor index " +
                  std::to_string(i));
    }
    const std::uint64_t tensor_bytes = dense_tensor_bytes_local(tensor, dtype);
    if (tensor_bytes == 0U) {
      return fail("boxdecode inferred contract requires non-zero tensor span for tensor index " +
                  std::to_string(i));
    }

    BoxDecodeTensorStaticContract entry;
    entry.input_shape = {h, w, c};
    entry.slice_shape = {h, w, c};
    entry.data_type = dtype;
    entry.layout =
        layout_token_from_tensor_layout_local(resolve_tensor_layout_local(tensor, input_contract));
    entry.logical_name = tensor_name;
    entry.backend_name = tensor_name;
    entry.source_segment_name = tensor_name;
    entry.source_logical_output_index =
        tensor.route.logical_index >= 0 ? tensor.route.logical_index : static_cast<int>(i);
    entry.source_output_slot =
        tensor.route.route_slot >= 0 ? tensor.route.route_slot : static_cast<int>(i);
    entry.source_physical_index =
        tensor.route.physical_index >= 0 ? tensor.route.physical_index : static_cast<int>(i);
    entry.source_byte_offset = tensor.route.physical_byte_offset;
    entry.source_size_bytes = tensor_bytes;
    out.tensors.push_back(entry);
    out.tensor_names.push_back(tensor_name);

    BoxDecodePhysicalInputStaticContract physical;
    physical.name = tensor_name;
    physical.physical_index = entry.source_physical_index;
    physical.byte_offset = entry.source_byte_offset;
    physical.size_bytes = entry.source_size_bytes;
    out.physical_inputs.push_back(std::move(physical));

    if (tensor.semantic.quant.has_value() && !tensor.semantic.quant->scales.empty() &&
        !tensor.semantic.quant->zero_points.empty()) {
      out.dq_scale.push_back(tensor.semantic.quant->scales.front());
      out.dq_zp.push_back(tensor.semantic.quant->zero_points.front());
    } else if (input_contract.has_value() && input_contract->q_scale.has_value() &&
               input_contract->q_zp.has_value()) {
      out.dq_scale.push_back(*input_contract->q_scale);
      out.dq_zp.push_back(*input_contract->q_zp);
    } else if (!out.quant_needed) {
      out.dq_scale.push_back(1.0);
      out.dq_zp.push_back(0);
    } else {
      return fail("boxdecode inferred quantized contract requires q_scale/q_zp for tensor index " +
                  std::to_string(i));
    }
  }

  if (const auto inferred = resolve_boxdecode_score_activation_from_sample_local(
          tensors, decode_type, input_contract, error_message);
      inferred.has_value()) {
    out.score_activation = *inferred;
  } else if (error_message && !error_message->empty()) {
    return std::nullopt;
  }

  maybe_infer_score_activation_from_boxdecode_contract_local(&out);
  maybe_override_quantized_yolov8_score_activation_local(&out);
  maybe_default_float_yolo_score_activation_local(&out);
  maybe_infer_yolov8_decode_type_option_local(&out);
  apply_raw_yolov6_yolox_contract_overrides_local(&out);

  return out;
}

} // namespace simaai::neat::pipeline_internal::sima

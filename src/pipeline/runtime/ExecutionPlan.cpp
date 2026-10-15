#include "pipeline/runtime/ExecutionPlanAssembly.h"

#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

using json = nlohmann::ordered_json;
namespace sima = pipeline_internal::sima;

struct ExecutionPlan::Impl {
  json document;
};

namespace {

constexpr int kSchemaVersion = 1;
constexpr const char* kSchemaName = "sima.neat.execution_plan";

// One graph node of the Model and the compiled stage it runs, if any.
struct PlanNode {
  const NodeReport* report = nullptr;
  const sima::StageStaticSpec* stage = nullptr;
  std::size_t stage_index = 0U;
};

std::string node_id(const PlanNode& node) {
  return "p" + std::to_string(node.report->index);
}

// Stage specs spell dtypes differently ("INT8", "fp32"); the plan uses lowercase tokens.
std::string dtype_token(std::string dtype) {
  std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (dtype == "fp32") {
    return "float32";
  }
  if (dtype == "bf16") {
    return "bfloat16";
  }
  if (dtype == "fp16") {
    return "float16";
  }
  return dtype;
}

bool is_quantized_dtype(const std::string& dtype) {
  return dtype == "int8" || dtype == "int16";
}

std::vector<std::int64_t> batch_first(std::vector<std::int64_t> shape, int batch) {
  if (shape.empty() || shape.front() != batch) {
    shape.insert(shape.begin(), batch);
  }
  return shape;
}

const char* runs_on(const PlanNode& node) {
  if (node.stage == nullptr) {
    return "A65";
  }
  switch (node.stage->payload_kind) {
  case sima::StagePayloadKind::ProcessCvu:
    return node.stage->processcvu.resolved_exec_backend == "A65" ? "A65" : "EV74";
  case sima::StagePayloadKind::ProcessMla:
    return "MLA";
  default:
    return "A65";
  }
}

sima::QuantStaticSpec per_tensor_quant(double scale, std::int64_t zero_point) {
  sima::QuantStaticSpec quant;
  quant.scales = {scale};
  quant.zero_points = {zero_point};
  return quant;
}

// Dequantize kernels hold the parameters of the tensors they dequantize in their q lists.
bool dequantizes(const sima::ProcessCvuStagePayload& cvu) {
  return cvu.graph_family_enum == sima::ProcessCvuGraphFamily::Dequant ||
         cvu.graph_family_enum == sima::ProcessCvuGraphFamily::DetessDequant;
}

std::optional<sima::QuantStaticSpec> cvu_q_params(const sima::ProcessCvuStagePayload& cvu,
                                                  std::size_t index) {
  if (index < cvu.q_scale_list.size() && index < cvu.q_zp_list.size()) {
    return per_tensor_quant(cvu.q_scale_list[index], cvu.q_zp_list[index]);
  }
  if (cvu.has_q_scale) {
    return per_tensor_quant(cvu.q_scale, cvu.has_q_zp ? cvu.q_zp : 0);
  }
  return std::nullopt;
}

// Quantization a CVU stage applies to its `index`-th output.
std::optional<sima::QuantStaticSpec> cvu_output_quant(const sima::StageStaticSpec* stage,
                                                      std::size_t index) {
  if (stage == nullptr || stage->payload_kind != sima::StagePayloadKind::ProcessCvu ||
      dequantizes(stage->processcvu)) {
    return std::nullopt;
  }
  return cvu_q_params(stage->processcvu, index);
}

// Quantization a CVU stage removes from its `index`-th input.
std::optional<sima::QuantStaticSpec> cvu_input_dequant(const sima::StageStaticSpec* stage,
                                                       std::size_t index) {
  if (stage == nullptr || stage->payload_kind != sima::StagePayloadKind::ProcessCvu) {
    return std::nullopt;
  }
  const auto& cvu = stage->processcvu;
  if (index < cvu.dq_scale_list.size() && index < cvu.dq_zp_list.size()) {
    return per_tensor_quant(cvu.dq_scale_list[index], cvu.dq_zp_list[index]);
  }
  return dequantizes(cvu) ? cvu_q_params(cvu, index) : std::nullopt;
}

json quant_json(const sima::QuantStaticSpec& quant) {
  if (quant.granularity == sima::QuantGranularity::PerTensor && quant.scales.size() == 1U &&
      quant.zero_points.size() == 1U) {
    return {{"scale", quant.scales.front()}, {"zero_point", quant.zero_points.front()}};
  }
  return {{"axis", quant.axis}, {"scale", quant.scales}, {"zero_point", quant.zero_points}};
}

// One tensor on an edge. Quant candidates are tried in order; only quantized dtypes carry one.
json tensor_json(const std::string& logical_name, const std::string& backend_name,
                 const std::vector<std::int64_t>& shape, const std::string& dtype, int batch,
                 std::initializer_list<std::optional<sima::QuantStaticSpec>> quant) {
  const std::string token = dtype_token(dtype);
  json out = {{"name", logical_name.empty() ? backend_name : logical_name},
              {"shape", batch_first(shape, batch)},
              {"dtype", token}};
  if (is_quantized_dtype(token)) {
    for (const auto& candidate : quant) {
      if (candidate.has_value()) {
        out["quant"] = quant_json(*candidate);
        break;
      }
    }
  }
  return out;
}

json producer_output_json(const sima::StageStaticSpec& producer,
                          const sima::LogicalTensorStaticSpec& output, int batch,
                          const std::optional<sima::QuantStaticSpec>& consumer_quant) {
  const auto index = static_cast<std::size_t>(std::max(output.logical_index, 0));
  return tensor_json(output.logical_name, output.backend_name, output.shape, output.dtype, batch,
                     {output.quant, consumer_quant, cvu_output_quant(&producer, index)});
}

// Tensors from `from` to `to`: the consumer's resolved bindings when both run stages, otherwise
// the outputs or inputs of whichever side runs one.
json edge_tensors(const sima::SimaPluginStaticManifest& manifest, const PlanNode& from,
                  const PlanNode& to, int batch) {
  json tensors = json::array();
  if (from.stage != nullptr && to.stage != nullptr) {
    for (const auto& edge :
         sima::edgecontract::resolve_consumer_edge_contracts(manifest, to.stage_index)) {
      if (edge.producer_stage_index != from.stage_index ||
          edge.producer_logical_output == nullptr) {
        continue;
      }
      const auto* input = edge.consumer_logical_input;
      std::optional<sima::QuantStaticSpec> consumer_quant =
          input != nullptr ? input->quant : std::nullopt;
      if (!consumer_quant.has_value()) {
        consumer_quant = cvu_input_dequant(
            to.stage, static_cast<std::size_t>(input ? std::max(input->logical_index, 0) : 0));
      }
      tensors.push_back(
          producer_output_json(*from.stage, *edge.producer_logical_output, batch, consumer_quant));
    }
    if (!tensors.empty()) {
      return tensors;
    }
  }
  if (from.stage != nullptr) {
    for (const auto& output : from.stage->logical_outputs) {
      const auto index = static_cast<std::size_t>(std::max(output.logical_index, 0));
      tensors.push_back(
          producer_output_json(*from.stage, output, batch, cvu_input_dequant(to.stage, index)));
    }
    return tensors;
  }
  if (to.stage != nullptr) {
    for (const auto& input : to.stage->logical_inputs) {
      const auto index = static_cast<std::size_t>(std::max(input.logical_index, 0));
      tensors.push_back(tensor_json(input.logical_name, input.backend_name, input.shape,
                                    input.dtype, batch,
                                    {input.quant, cvu_input_dequant(to.stage, index)}));
    }
  }
  return tensors;
}

std::string kernel_name(const sima::StageStaticSpec& stage) {
  if (stage.payload_kind == sima::StagePayloadKind::ProcessCvu &&
      !stage.processcvu.graph_name.empty()) {
    return stage.processcvu.graph_name;
  }
  return stage.kernel_kind;
}

json preproc_settings(const sima::ProcessCvuStagePayload& cvu) {
  json out = {{"input_img_type", cvu.input_img_type},
              {"output_img_type", cvu.output_img_type},
              {"scaled_width", cvu.scaled_width},
              {"scaled_height", cvu.scaled_height},
              {"scaling_type", cvu.scaling_type},
              {"padding_type", cvu.padding_type},
              {"pad_value", cvu.pad_value}};
  if (cvu.aspect_ratio >= 0) {
    out["aspect_ratio"] = cvu.aspect_ratio != 0;
  }
  if (cvu.normalize >= 0) {
    out["normalize"] = cvu.normalize != 0;
  }
  if (!cvu.channel_mean.empty()) {
    out["channel_mean"] = cvu.channel_mean;
  }
  if (!cvu.channel_stddev.empty()) {
    out["channel_stddev"] = cvu.channel_stddev;
  }
  return out;
}

// Thresholds and top-k of 0 leave the plugin default in place, so only set values are reported.
json box_decode_settings(const sima::BoxDecodeStagePayload& box) {
  json out = {{"decode_type", box_decode_type_token(box.decode_type)},
              {"num_classes", box.num_classes}};
  if (box.detection_threshold > 0.0) {
    out["detection_threshold"] = box.detection_threshold;
  }
  if (box.nms_iou_threshold > 0.0) {
    out["nms_iou_threshold"] = box.nms_iou_threshold;
  }
  if (box.topk > 0) {
    out["top_k"] = box.topk;
  }
  return out;
}

json node_json(const PlanNode& node) {
  json out = {{"id", node_id(node)}, {"kind", node.report->kind}, {"runs_on", runs_on(node)}};
  if (node.stage == nullptr) {
    return out;
  }
  if (node.stage->payload_kind != sima::StagePayloadKind::ProcessMla) {
    out["kernel"] = kernel_name(*node.stage);
  }
  if (node.stage->payload_kind == sima::StagePayloadKind::ProcessCvu &&
      node.stage->processcvu.graph_family_enum == sima::ProcessCvuGraphFamily::Preproc) {
    out["settings"] = preproc_settings(node.stage->processcvu);
  } else if (node.stage->payload_kind == sima::StagePayloadKind::BoxDecode) {
    out["settings"] = box_decode_settings(node.stage->boxdecode);
  }
  return out;
}

// Pair each graph node, in pipeline order, with the compiled stage whose element it renders.
std::vector<PlanNode> plan_nodes(const pipeline_internal::SegmentBuildPlan& build,
                                 const sima::SimaPluginStaticManifest& manifest) {
  std::vector<PlanNode> nodes;
  for (const auto& report : build.node_reports) {
    nodes.push_back(PlanNode{&report});
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const PlanNode& a, const PlanNode& b) { return a.report->index < b.report->index; });
  for (std::size_t i = 0; i < manifest.stages.size(); ++i) {
    const auto& stage = manifest.stages[i];
    const std::string element = apply_name_transform(build.name_transform, stage.element_name);
    const auto owner = std::find_if(nodes.begin(), nodes.end(), [&](const PlanNode& node) {
      const auto& elements = node.report->elements;
      return std::find(elements.begin(), elements.end(), element) != elements.end();
    });
    if (owner == nodes.end()) {
      throw std::runtime_error("Model::execution_plan: stage '" + stage.element_name +
                               "' belongs to no graph node");
    }
    if (owner->stage != nullptr) {
      throw std::runtime_error("Model::execution_plan: node " + owner->report->kind +
                               " runs more than one stage");
    }
    owner->stage = &stage;
    owner->stage_index = i;
  }
  return nodes;
}

int model_batch(const sima::SimaPluginStaticManifest& manifest) {
  for (const auto& stage : manifest.stages) {
    if (stage.payload_kind == sima::StagePayloadKind::ProcessMla &&
        stage.processmla.batch_size > 0) {
      return stage.processmla.batch_size;
    }
  }
  return 1;
}

std::optional<json> model_memory(const sima::SimaPluginStaticManifest& manifest) {
  for (const auto& stage : manifest.stages) {
    if (stage.frame_arena_role == sima::FrameArenaRole::Allocate &&
        stage.frame_arena_size_bytes > 0U) {
      const bool dms =
          stage.frame_arena_storage_domain == sima::static_contract::ArenaStorageDomain::Dms;
      return json{{"per_frame_bytes", stage.frame_arena_size_bytes}, {"type", dms ? "DMS" : "CMA"}};
    }
  }
  return std::nullopt;
}

} // namespace

ExecutionPlan::ExecutionPlan() : impl_(std::make_shared<const Impl>()) {}
ExecutionPlan::~ExecutionPlan() = default;
ExecutionPlan::ExecutionPlan(const ExecutionPlan&) = default;
ExecutionPlan& ExecutionPlan::operator=(const ExecutionPlan&) = default;
ExecutionPlan::ExecutionPlan(ExecutionPlan&&) noexcept = default;
ExecutionPlan& ExecutionPlan::operator=(ExecutionPlan&&) noexcept = default;

int ExecutionPlan::schema_version() const {
  return kSchemaVersion;
}

std::string ExecutionPlan::to_json(int indent) const {
  if (!impl_ || impl_->document.is_null()) {
    return "{}";
  }
  return impl_->document.dump(indent < 0 ? -1 : indent);
}

ExecutionPlan internal::ExecutionPlanAccess::make(const pipeline_internal::SegmentBuildPlan& build,
                                                  const std::string& model_name) {
  if (!build.rendered_manifest.has_value()) {
    throw std::runtime_error("Model::execution_plan: the build rendered no stage manifest");
  }
  const auto& manifest = *build.rendered_manifest;
  const int batch = model_batch(manifest);
  const std::vector<PlanNode> nodes = plan_nodes(build, manifest);

  json model = {{"name", model_name}, {"batch", batch}};
  if (auto memory = model_memory(manifest)) {
    model["memory"] = std::move(*memory);
  }
  json node_list = json::array();
  json edge_list = json::array();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    node_list.push_back(node_json(nodes[i]));
    if (i + 1U == nodes.size()) {
      continue;
    }
    json edge = {{"from", node_id(nodes[i])}, {"to", node_id(nodes[i + 1U])}};
    json tensors = edge_tensors(manifest, nodes[i], nodes[i + 1U], batch);
    if (!tensors.empty()) {
      edge["tensors"] = std::move(tensors);
    }
    edge_list.push_back(std::move(edge));
  }

  auto impl = std::make_shared<ExecutionPlan::Impl>();
  impl->document = {{"schema", kSchemaName},
                    {"schema_version", kSchemaVersion},
                    {"model", std::move(model)},
                    {"nodes", std::move(node_list)},
                    {"edges", std::move(edge_list)}};
  ExecutionPlan plan;
  plan.impl_ = std::move(impl);
  return plan;
}

} // namespace simaai::neat

#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include "pipeline/internal/sima/BoxDecodePublication.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/stagesemantics/SsdDecodeContract.h"

#include <simaai/scheduler/client/kernel.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace k = ::simaai::scheduler::client;
namespace rt = ::simaai::scheduler::client;
namespace gs = ::simaai::scheduler::client;
namespace scheduler = ::simaai::scheduler::client;
namespace specs = specbuilders;
[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument("BoxDecode kernel recipe: " + message);
}
template <class T> T checked(rt::Result<T> value) {
  if (auto* error = std::get_if<rt::Error>(&value))
    invalid(std::string(error->field) + ": " + error->message);
  return std::get<T>(std::move(value));
}
k::DType dtype(const std::string& value) {
  if (value == "FP32" || value == "FLOAT32")
    return k::DType::Float32;
  if (value == "BF16" || value == "BFLOAT16")
    return k::DType::BFloat16;
  if (value == "INT8")
    return k::DType::Int8;
  invalid("unsupported head dtype '" + value + "'");
}
const char* dtype_name(k::DType type) {
  switch (type) {
  case k::DType::Float32:
    return "FP32";
  case k::DType::BFloat16:
    return "BF16";
  case k::DType::Int8:
    return "INT8";
  case k::DType::UInt8:
    return "UINT8";
  case k::DType::Int32:
    return "INT32";
  default:
    invalid("unsupported result dtype");
  }
}
std::size_t scalar_bytes(k::DType type) {
  if (type == k::DType::Int8 || type == k::DType::UInt8)
    return 1;
  if (type == k::DType::BFloat16)
    return 2;
  return 4;
}
std::size_t dense_bytes(const k::TensorSpec& tensor) {
  std::size_t bytes = scalar_bytes(tensor.dtype);
  for (auto dimension : tensor.shape) {
    if (dimension <= 0 || static_cast<std::uint64_t>(dimension) > SIZE_MAX / bytes)
      invalid("tensor byte extent overflows");
    bytes *= static_cast<std::size_t>(dimension);
  }
  return bytes;
}
std::array<std::int64_t, 3> hwc(std::span<const std::int64_t> shape, const std::string& layout,
                                bool bounded = true) {
  if (shape.size() < 3 || shape.size() > 6)
    invalid("head rank must be 3..6");
  if (bounded)
    for (std::size_t i = 0; i + 3 < shape.size(); ++i)
      if (shape[i] != 1)
        invalid("head outer dimensions must be singleton");
  const auto n = shape.size();
  std::array<std::int64_t, 3> result{shape[n - 3], shape[n - 2], shape[n - 1]};
  if (layout == "CHW" || layout == "NCHW")
    result = {shape[n - 2], shape[n - 1], shape[n - 3]};
  else if (layout != "HWC" && layout != "NHWC")
    invalid("head layout must explicitly identify HWC axes");
  for (auto value : result)
    if ((bounded ? value <= 0 : value < 0) || (bounded && value > 32767))
      invalid("head extent outside supported range");
  return result;
}
std::array<std::int64_t, 3> slice_hwc(const BoxDecodeStagePayload& payload, std::size_t input,
                                      const LogicalInputStaticSpec& logical) {
  if (input >= payload.slice_shapes.size())
    invalid("missing explicit head slice shape");
  const auto& slice = payload.slice_shapes[input];
  if (slice.rank > SIMA_EV_MAX_RANK)
    invalid("slice rank exceeds contract bound");
  std::array<std::int64_t, SIMA_EV_MAX_RANK> dimensions{};
  for (std::size_t i = 0; i < slice.rank; ++i)
    dimensions[i] = slice.sizes[i];
  // Slice shapes are already canonical HWC in the BoxDecode payload.
  return hwc({dimensions.data(), slice.rank}, "HWC");
}
struct Input {
  k::TensorSpec tensor;
  gs::TensorStagePort port;
  std::vector<std::size_t> dependencies;
};
struct Builder {
  CompiledKernelContract compiled;
  std::shared_ptr<KernelStageRecipe> recipe = std::make_shared<KernelStageRecipe>();
  std::size_t next_output = 1;
  gs::TensorStagePort scratch(const k::TensorSpec& tensor,
                              const k::TensorRequirements& requirements) {
    const auto index = next_output++;
    const auto bytes = requirements.storage_bytes;
    if (bytes != dense_bytes(tensor))
      invalid("private output projection differs from checked storage");
    const auto name = "kernel_private_" + std::to_string(index);
    const std::vector<std::int64_t> shape(tensor.shape.begin(), tensor.shape.end());
    recipe->private_outputs.push_back(specs::build_logical_output_static_spec(
        static_cast<int>(index), static_cast<int>(index), static_cast<int>(index),
        static_cast<int>(index), static_cast<int>(index), shape, dtype_name(tensor.dtype), "", name,
        name, name, 0, bytes));
    recipe->private_buffers.push_back(specs::build_physical_buffer_static_spec(
        static_cast<int>(index), static_cast<int>(index), bytes, DeviceKind::Cpu, name));
    return {index, 0, gs::TensorStagePortSource::Output};
  }
  void detess(Input& input, const k::TensorSpec& output) {
    auto executable =
        scheduler::check_kernel({"detess", k::Target::A65, {&input.tensor, 1}, {&output, 1}, {}});
    if (auto* error = std::get_if<rt::Error>(&executable))
      invalid(std::string(error->field) + ": " + error->message);
    const auto inspection =
        checked(scheduler::inspect_kernel(std::get<rt::KernelExecutable>(executable)));
    auto port = scratch(inspection.outputs().front(), inspection.output_requirements().front());
    const auto member = recipe->members.size();
    recipe->members.push_back({std::get<rt::KernelExecutable>(std::move(executable)),
                               {input.port},
                               {port},
                               input.dependencies,
                               {}});
    input = {output, port, {member}};
  }
  Input input(const CompiledBoxDecodeContract& source, std::size_t index, int logical_channels,
              bool box_operation) {
    auto& logical = compiled.runtime_contract.logical_inputs.at(index);
    const auto shape = hwc(logical.shape, logical.layout);
    if (logical_channels <= 0 || logical_channels > shape[2])
      invalid("logical channel count exceeds source storage");
    if (logical.byte_offset < 0)
      invalid("negative head byte offset");
    logical.backend_input_index = static_cast<int>(index);
    Input in{{std::vector<std::int64_t>(shape.begin(), shape.end()), dtype(logical.dtype)},
             {index, static_cast<std::size_t>(logical.byte_offset)},
             {}};
    const auto kind = source.payload.tensor_storage_kind.at(index);
    if (kind == 0)
      in.tensor.layout = k::MlaCBlocked{slice_hwc(source.payload, index, logical)};
    else if (kind == 2) {
      const auto tile = slice_hwc(source.payload, index, logical);
      in.tensor.layout = k::MlaTiled{
          {-1, 0, 1, 2}, {1, tile[0], tile[1], tile[2]}, k::MlaTiled::Channels::PadTo16};
    } else if (kind != 1)
      invalid("unknown explicit source storage kind");
    // Storage kind 1 is linear storage, not MLA-tiled. Logical layout locates
    // H/W/C axes; explicit byte strides remain authoritative, including planar
    // FP32 views. An absent stride vector retains the established physical-HWC
    // storage contract rather than inferring a new layout from the axis names.
    if (kind == 1 && !logical.stride_bytes.empty()) {
      if (logical.stride_bytes.size() != logical.shape.size())
        invalid("head stride rank does not match shape");
      const auto stride = hwc(logical.stride_bytes, logical.layout, false);
      const auto width = static_cast<std::int64_t>(scalar_bytes(in.tensor.dtype));
      const std::array<std::int64_t, 3> contiguous{shape[1] * shape[2] * width, shape[2] * width,
                                                   width};
      if (stride != contiguous) {
        k::Strided layout;
        std::copy(stride.begin(), stride.end(), layout.strides_bytes.begin());
        in.tensor.layout = layout;
      }
    }
    // BoxDecode needs explicit detessellation of pixel-major MLA storage;
    // SuperPoint consumes it natively. Neither format aliases C-block bytes.
    if (kind == 2 && box_operation) {
      const k::TensorSpec output{in.tensor.shape, in.tensor.dtype};
      detess(in, output);
    }
    // Preserve native FP32. A compulsory BF16 cast changes threshold/NMS decisions;
    // Kernels owns admission of the exact dtype and physical layout.
    auto narrowed = shape;
    narrowed[2] = logical_channels;
    in.tensor.shape.assign(narrowed.begin(), narrowed.end());
    if (std::holds_alternative<k::Contiguous>(in.tensor.layout) && shape[2] != logical_channels) {
      const auto width = static_cast<std::int64_t>(scalar_bytes(in.tensor.dtype));
      k::Strided strided;
      strided.strides_bytes = {shape[1] * shape[2] * width, shape[2] * width, width};
      in.tensor.layout = strided;
    }
    return in;
  }
};
std::string_view profile(BoxDecodeType type) {
  switch (type) {
  case BoxDecodeType::YoloV26:
  case BoxDecodeType::YoloV26Pose:
  case BoxDecodeType::YoloV26Seg:
    return "yolo26_v1";
  case BoxDecodeType::YoloV5:
  case BoxDecodeType::YoloV5Seg:
    return "yolov5_v1";
  case BoxDecodeType::YoloV7:
  case BoxDecodeType::YoloV7Seg:
    return "yolov7_v1";
  case BoxDecodeType::YoloV8:
  case BoxDecodeType::YoloV8Pose:
  case BoxDecodeType::YoloV8Seg:
  case BoxDecodeType::YoloV9:
  case BoxDecodeType::YoloV9Seg:
  case BoxDecodeType::YoloV10:
  case BoxDecodeType::YoloV10Seg:
    return "yolo_dfl_v1";
  case BoxDecodeType::YoloV6:
    return "yolov6_v1";
  case BoxDecodeType::YoloX:
    return "yolox_v1";
  case BoxDecodeType::YoloXSegPose:
    return "yolox_seg_pose_v1";
  case BoxDecodeType::Yolo:
    return "decoded_v1";
  case BoxDecodeType::EffDet:
    return "efficientdet_v1";
  case BoxDecodeType::Detr:
    return "detr_v1";
  case BoxDecodeType::Centernet:
    return "centernet_v1";
  case BoxDecodeType::Ssd:
    return "ssd_v1";
  case BoxDecodeType::RcnnStage1:
    return "rcnn_proposals_v1";
  default:
    invalid("decode profile is unspecified");
  }
}
std::string_view task(BoxDecodeType type) {
  switch (type) {
  case BoxDecodeType::YoloXSegPose:
    return "segmentation_pose";
  case BoxDecodeType::YoloV26Pose:
  case BoxDecodeType::YoloV8Pose:
    return "pose";
  case BoxDecodeType::YoloV26Seg:
  case BoxDecodeType::YoloV5Seg:
  case BoxDecodeType::YoloV7Seg:
  case BoxDecodeType::YoloV8Seg:
  case BoxDecodeType::YoloV9Seg:
  case BoxDecodeType::YoloV10Seg:
    return "segmentation";
  default:
    return "detection";
  }
}
bool grouped(BoxDecodeTypeOption option) {
  switch (option) {
  case BoxDecodeTypeOption::GroupedByRole:
  case BoxDecodeTypeOption::GroupedByRoleProbability:
  case BoxDecodeTypeOption::GroupedByRoleLogit:
  case BoxDecodeTypeOption::Split3Grouped:
    return true;
  case BoxDecodeTypeOption::InterleavedByHead:
  case BoxDecodeTypeOption::InterleavedByHeadProbability:
  case BoxDecodeTypeOption::InterleavedByHeadLogit:
  case BoxDecodeTypeOption::Split3Interleaved:
  case BoxDecodeTypeOption::PackedPerHead:
    return false;
  default:
    invalid("head grouping must be resolved before numerical preparation");
  }
}
const QuantStaticSpec& quantization(const LogicalInputStaticSpec& input) {
  if (!input.quant || input.quant->granularity != QuantGranularity::PerTensor ||
      input.quant->scales.size() != 1 || input.quant->zero_points.size() != 1 ||
      !std::isfinite(input.quant->scales[0]) || input.quant->scales[0] <= 0 ||
      input.quant->zero_points[0] < -128 || input.quant->zero_points[0] > 127)
    invalid("INT8 heads require positive finite per-tensor quantization and an INT8 zero point");
  return *input.quant;
}
} // namespace

BoxDecodeKernelRecipeResult prepare_boxdecode_kernel(const CompiledBoxDecodeContract& source,
                                                     const BoxDecodeGeometryOptions& geometry) {
  const auto& p = source.payload;
  const auto count = source.runtime_contract.logical_inputs.size();
  if (!count || p.tensor_storage_kind.size() != count)
    invalid("input/storage contract count mismatch");
  if (geometry.model_width <= 0 || geometry.model_height <= 0 || geometry.original_width <= 0 ||
      geometry.original_height <= 0)
    invalid("model and original image geometry must be resolved");
  if (geometry.resize == ResizeMode::Crop && p.decode_type != BoxDecodeType::SuperPoint &&
      !(p.decode_type == BoxDecodeType::YoloXSegPose && geometry.model_to_original))
    invalid("crop geometry is not supported by the numerical recipe");
  if (p.topk == 0)
    invalid("top_k is unresolved: set a positive output capacity for standalone bounding-box "
            "decoding, or inherit a positive Model capacity with a model-bound node");
  if (p.topk < 0 || p.topk > 65536)
    invalid("top_k output capacity must be 1..65536");
  if (p.decode_type != BoxDecodeType::SuperPoint && (p.num_classes < 1 || p.num_classes > 336))
    invalid("class count must be 1..336 before profile-specific validation");
  Builder build;
  build.compiled.runtime_contract = source.runtime_contract;
  auto& runtime = build.compiled.runtime_contract;
  runtime.plugin_kind = "kernel";
  std::vector<Input> inputs;
  std::vector<k::TensorSpec> outputs;
  k::KernelExecutable numerical;
  BoxDecodeWireFormat wire = BoxDecodeWireFormat::Boxes;
  if (p.decode_type == BoxDecodeType::SuperPoint) {
    if (count != 2 || p.tensor_roles.size() != 2)
      invalid("SuperPoint requires two explicit tensor roles");
    if (const auto error = validate_superpoint_static_metadata(p.superpoint, true))
      invalid(*error);
    const auto detector = std::find(p.tensor_roles.begin(), p.tensor_roles.end(),
                                    BoxDecodeTensorRole::DetectorLogits);
    const auto descriptor = std::find(p.tensor_roles.begin(), p.tensor_roles.end(),
                                      BoxDecodeTensorRole::DescriptorGrid);
    if (detector == p.tensor_roles.end() || descriptor == p.tensor_roles.end() ||
        detector == descriptor)
      invalid("SuperPoint roles are incomplete");
    const std::array indices{static_cast<std::size_t>(detector - p.tensor_roles.begin()),
                             static_cast<std::size_t>(descriptor - p.tensor_roles.begin())};
    std::string_view selected_profile;
    switch (p.superpoint.profile) {
    case SuperPointProfile::A65V1:
      selected_profile = "a65_v1";
      break;
    case SuperPointProfile::LightGlueV1:
      selected_profile = "lightglue_v1";
      break;
    case SuperPointProfile::MagicLeapDemoV1:
      selected_profile = "magicleap_demo_v1";
      break;
    default:
      invalid("SuperPoint profile is unresolved or unsupported");
    }
    if (p.superpoint.cell_stride != 8 || p.superpoint.descriptor_stride != 8)
      invalid("SuperPoint requires the supported stride-eight contract");
    std::vector<k::Parameter> parameters{
        {"profile", selected_profile},
        {"model_input_size", k::ParameterArray{geometry.model_width, geometry.model_height}},
        {"topk_capacity", p.topk},
        {"detection_threshold", static_cast<float>(p.detection_threshold)},
        {"nms_radius", p.superpoint.nms_radius},
        {"border_margin", p.superpoint.border_margin}};
    for (std::size_t role = 0; role < 2; ++role) {
      inputs.push_back(
          build.input(source, indices[role], role == 0 ? 65 : p.superpoint.descriptor_dim, false));
      if (inputs.back().tensor.dtype == k::DType::Int8) {
        const auto& q = quantization(runtime.logical_inputs[indices[role]]);
        const std::string prefix =
            role == 0 ? "detector_quantization_" : "descriptor_quantization_";
        parameters.push_back({prefix + "scale", q.scales[0]});
        parameters.push_back({prefix + "zero_point", q.zero_points[0]});
        parameters.push_back({prefix + "reciprocal", true});
      }
    }
    k::DType descriptors;
    switch (p.superpoint.descriptor_output_dtype) {
    case TensorDType::Int8:
      descriptors = k::DType::Int8;
      break;
    case TensorDType::BFloat16:
      descriptors = k::DType::BFloat16;
      break;
    case TensorDType::Float32:
      descriptors = k::DType::Float32;
      break;
    default:
      invalid("unsupported descriptor output dtype");
    }
    outputs = {{{p.topk, 2}, k::DType::Float32},
               {{p.topk}, k::DType::Float32},
               {{p.topk, p.superpoint.descriptor_dim}, descriptors},
               {{1}, k::DType::Int32}};
    const std::array tensors{inputs[0].tensor, inputs[1].tensor};
    numerical = checked(
        scheduler::check_kernel({"superpoint", k::Target::A65, tensors, outputs, parameters}));
    wire = p.superpoint.output_format == SuperPointOutputFormat::FeaturePointsV1
               ? BoxDecodeWireFormat::FeaturePointsV1
               : BoxDecodeWireFormat::LegacyFeaturePointsV0;
  } else {
    const auto selected_profile = profile(p.decode_type);
    const auto selected_task = task(p.decode_type);
    int model_width = geometry.model_width, model_height = geometry.model_height;
    const bool rcnn = selected_profile == "rcnn_proposals_v1";
    if (rcnn && p.topk > 16384)
      invalid("RCNN topk is per level and must not exceed 16384");
    const int capacity = rcnn ? 4 * p.topk : p.topk;
    std::string_view score_activation;
    switch (p.score_activation) {
    case BoxDecodeScoreActivation::Identity:
      score_activation = "identity";
      break;
    case BoxDecodeScoreActivation::Sigmoid:
      score_activation = "sigmoid";
      break;
    case BoxDecodeScoreActivation::Softmax:
      score_activation = "softmax";
      break;
    default:
      invalid("score activation must be resolved");
    }
    const bool combined = selected_profile == "yolov5_v1" || selected_profile == "yolov7_v1";
    const bool ssd = selected_profile == "ssd_v1";
    const bool decoded = selected_profile == "decoded_v1";
    const bool x = selected_profile == "yolox_v1";
    const bool center = selected_profile == "centernet_v1";
    const bool auxiliary = selected_task != "detection";
    const bool seg_pose = selected_task == "segmentation_pose";
    const bool masks = selected_task == "segmentation" || seg_pose;
    int keypoints = 17;
    const auto option = p.decode_type_option.value_or(BoxDecodeTypeOption::Auto);
    const bool split_extents = decoded && (option == BoxDecodeTypeOption::Split3Interleaved ||
                                           option == BoxDecodeTypeOption::Split3Grouped);
    const std::size_t per_head = (combined ? 1 : 2) + (x || center || split_extents ? 1 : 0) +
                                 (auxiliary ? 1 : 0) + (seg_pose ? 1 : 0);
    if (count < (masks ? 1U : 0U) || (count - (masks ? 1 : 0)) % per_head)
      invalid("head arity does not match profile/task");
    const std::size_t heads = (count - (masks ? 1 : 0)) / per_head;
    bool by_role = ssd;
    if (!ssd && !combined && heads > 1) {
      if (option != BoxDecodeTypeOption::Auto)
        by_role = grouped(option);
      else {
        // Preserve the legacy static Auto grouping rule, using authored tensor
        // geometry only. Kernel admission checks every resulting role/shape.
        const auto depth = [&](std::size_t i) {
          return hwc(runtime.logical_inputs[i].shape, runtime.logical_inputs[i].layout)[2];
        };
        by_role = true;
        for (std::size_t i = 1; i < heads; ++i)
          by_role = by_role && depth(i) == depth(0);
      }
    }
    auto port = [&](std::size_t head, std::size_t role) {
      return by_role ? role * heads + head : head * per_head + role;
    };
    const auto logical_depth = [&](std::size_t index) {
      const auto& input = runtime.logical_inputs[index];
      // Dense carriers may have channel padding; packed slice depth is a tile,
      // not the logical tensor depth.
      return p.tensor_storage_kind[index] == 1 ? slice_hwc(p, index, input)[2]
                                               : hwc(input.shape, input.layout)[2];
    };
    std::vector<int> channels(count);
    const stagesemantics::SsdRecipeDescriptor* ssd_descriptor = nullptr;
    std::string_view ssd_recipe = "none";
    int decoded_anchors = 1;
    bool decoded_objectness = false;
    k::ParameterRecords head_parameters, quantization_parameters;
    std::size_t first_scores_input = 0;
    if (ssd) {
      switch (p.ssd_recipe_id) {
      case stagesemantics::SsdRecipeId::Ssd300V1:
        ssd_recipe = "ssd300";
        break;
      case stagesemantics::SsdRecipeId::SsdMobile300V1:
        ssd_recipe = "mobilenet300";
        break;
      case stagesemantics::SsdRecipeId::SsdMobile320V1:
        ssd_recipe = "mobilenet320";
        break;
      case stagesemantics::SsdRecipeId::SsdLiteMobile320V1:
        ssd_recipe = "torchvision_lite320";
        break;
      default:
        invalid("SSD recipe is unresolved");
      }
      ssd_descriptor = stagesemantics::find_ssd_recipe_descriptor(p.ssd_recipe_id);
      if (!ssd_descriptor || heads != ssd_descriptor->ordered_levels.size())
        invalid("SSD head count does not match the selected recipe");
    }
    for (std::size_t h = 0; h < heads; ++h) {
      std::size_t boxes = port(h, 0), scores = combined ? boxes : port(h, x ? 2 : 1);
      std::optional<std::size_t> extra;
      int anchors = (combined || rcnn)                      ? 3
                    : selected_profile == "efficientdet_v1" ? 9
                    : ssd ? ssd_descriptor->ordered_levels[h].localization_channels / 4
                          : 1;
      if (center) {
        scores = port(h, 0);
        boxes = port(h, 1);
        extra = port(h, 2);
      } else if (x)
        extra = port(h, 1);
      else if (split_extents)
        extra = port(h, 1);
      if (decoded) {
        const auto c =
            hwc(runtime.logical_inputs[boxes].shape, runtime.logical_inputs[boxes].layout)[2];
        const int coordinate_channels = split_extents ? 2 : 4;
        if (c != coordinate_channels && c != coordinate_channels * 3) {
          scores = port(h, 0);
          boxes = port(h, 1);
          if (split_extents)
            extra = port(h, 2);
        } else
          scores = port(h, per_head - 1);
        const auto box_channels =
            hwc(runtime.logical_inputs[boxes].shape, runtime.logical_inputs[boxes].layout)[2];
        anchors = static_cast<int>(box_channels) / coordinate_channels;
        const auto score_channels =
            hwc(runtime.logical_inputs[scores].shape, runtime.logical_inputs[scores].layout)[2];
        const bool objectness = score_channels != anchors * p.num_classes;
        if (h && (decoded_anchors != anchors || decoded_objectness != objectness))
          invalid("decoded head options disagree");
        decoded_anchors = anchors;
        decoded_objectness = objectness;
      }
      if (selected_profile == "detr_v1") {
        if (hwc(runtime.logical_inputs[boxes].shape, runtime.logical_inputs[boxes].layout)[2] != 4)
          std::swap(boxes, scores);
      }
      channels[boxes] = combined                            ? 3 * (p.num_classes + 5)
                        : center                            ? 2
                        : selected_profile == "yolo_dfl_v1" ? 64
                                                            : (split_extents ? 2 : 4) * anchors;
      if (!combined)
        channels[scores] =
            ssd ? ssd_descriptor->ordered_levels[h].confidence_channels
            : selected_profile == "detr_v1"
                ? p.num_classes + 1
                : anchors * (p.num_classes + (seg_pose || (decoded && decoded_objectness) ? 1 : 0));
      if (extra)
        channels[*extra] = x ? 1 : center ? 2 : 2 * anchors;
      std::optional<std::size_t> aux;
      if (auxiliary) {
        aux = port(h, seg_pose ? 2 : per_head - 1);
        channels[*aux] = masks ? 32 * anchors : 51;
      }
      k::ParameterRecord head{{"boxes_input", boxes}, {"scores_input", scores}};
      if (extra)
        head.push_back({"extra_input", *extra});
      if (aux)
        head.push_back({"auxiliary_input", *aux});
      if (seg_pose) {
        const auto keypoints_input = port(h, 3);
        // Logical depth, never padded carrier channels, determines P.
        if (logical_depth(boxes) != 4 || logical_depth(scores) != p.num_classes + 1 ||
            logical_depth(*aux) != 32)
          invalid("combined logical roles require 4 box channels, packed objectness+classes and 32 "
                  "mask coefficients");
        const auto point_channels = logical_depth(keypoints_input);
        if (point_channels < 3 || point_channels > 51 || point_channels % 3)
          invalid("combined keypoint head requires 1..17 logical triplets");
        const auto points = static_cast<int>(point_channels / 3);
        if (h && points != keypoints)
          invalid("combined keypoint counts disagree across heads");
        keypoints = points;
        channels[keypoints_input] = static_cast<int>(point_channels);
        head.push_back({"keypoints_input", keypoints_input});
      }
      head_parameters.push_back(std::move(head));
      if (h == 0)
        first_scores_input = scores;
    }
    if (center) {
      // The legacy CenterNet contract decodes directly in score-grid units.
      // Keep that coordinate frame to preserve its intermediate rounding.
      const auto grid = hwc(runtime.logical_inputs[first_scores_input].shape,
                            runtime.logical_inputs[first_scores_input].layout);
      model_width = static_cast<int>(grid[1]);
      model_height = static_cast<int>(grid[0]);
    }
    if (masks) {
      if (seg_pose && logical_depth(count - 1) != 32)
        invalid("combined mask basis requires 32 logical channels");
      channels.back() = 32;
    }
    for (std::size_t i = 0; i < count; ++i) {
      inputs.push_back(build.input(source, i, channels[i], true));
      if (inputs.back().tensor.dtype == k::DType::Int8) {
        const auto& q = quantization(runtime.logical_inputs[i]);
        // Existing compiler contracts carry the quantization divisor. Preserve
        // the historical FP32 reciprocal, not an FP64 reciprocal rounded later.
        quantization_parameters.push_back(
            {{"input", i},
             {"dequant_scale", 1.0F / static_cast<float>(q.scales[0])},
             {"zero_point", static_cast<std::int32_t>(q.zero_points[0])}});
      }
    }
    outputs = {{{capacity, 4}, rcnn ? k::DType::Float32 : k::DType::Int32},
               {{capacity}, k::DType::Float32},
               {{capacity}, k::DType::Int32},
               {{1}, k::DType::Int32}};
    if (rcnn) {
      outputs.push_back({{capacity}, k::DType::Int32});
      wire = BoxDecodeWireFormat::RcnnProposals;
    }
    if (selected_task == "pose") {
      outputs.push_back({{p.topk, 17, 3}, k::DType::Float32});
      wire = BoxDecodeWireFormat::Pose;
    }
    if (masks) {
      outputs.push_back({{p.topk, 160, 160}, k::DType::UInt8});
      wire = BoxDecodeWireFormat::Masks;
    }
    if (seg_pose) {
      outputs.push_back({{p.topk, keypoints, 3}, k::DType::Float32});
      wire = BoxDecodeWireFormat::MasksPose;
    }
    std::vector<k::TensorSpec> tensors;
    for (const auto& in : inputs)
      tensors.push_back(in.tensor);
    std::vector<k::Parameter> parameters{
        {"profile", selected_profile},
        {"task", selected_task},
        {"model_input_size", k::ParameterArray{model_width, model_height}},
        {"original_image_size",
         k::ParameterArray{geometry.original_width, geometry.original_height}},
        {"resize", geometry.resize == ResizeMode::Stretch ? "stretch" : "centered_letterbox"},
        {"num_classes", p.num_classes},
        {"max_detections", capacity},
        {"rcnn_per_level_topk", rcnn ? p.topk : 0},
        {"detection_threshold", static_cast<float>(p.detection_threshold)},
        {"nms_iou_threshold", static_cast<float>(p.nms_iou_threshold)},
        {"score_activation", score_activation},
        {"ssd_recipe", ssd_recipe},
        {"decoded_anchors_per_cell", decoded_anchors},
        {"decoded_objectness", decoded_objectness},
        {"decoded_split_extents", split_extents},
        {"heads", std::move(head_parameters)},
        {"quantization", std::move(quantization_parameters)}};
    if (masks)
      parameters.push_back({"mask_basis_input", count - 1});
    if (seg_pose) {
      parameters.push_back({"keypoints", keypoints});
      k::ParameterArray pose_classes;
      for (const auto value : p.pose_classes)
        pose_classes.emplace_back(value);
      parameters.push_back({"pose_classes", std::move(pose_classes)});
      if (geometry.model_to_original) {
        const auto& t = *geometry.model_to_original;
        parameters.push_back({"model_to_original", k::ParameterArray{t[0], t[1], t[2], t[3]}});
      }
    } else if (geometry.model_to_original) {
      invalid("explicit affine requires a complete-geometry BoxDecode profile");
    }
    numerical = checked(
        scheduler::check_kernel({"boxdecode", k::Target::A65, tensors, outputs, parameters}));
  }
  gs::TensorStageMember decode;
  decode.executable = std::move(numerical);
  for (const auto& input : inputs) {
    decode.inputs.push_back(input.port);
    decode.dependencies.insert(decode.dependencies.end(), input.dependencies.begin(),
                               input.dependencies.end());
  }
  std::sort(decode.dependencies.begin(), decode.dependencies.end());
  decode.dependencies.erase(std::unique(decode.dependencies.begin(), decode.dependencies.end()),
                            decode.dependencies.end());
  const auto inspection = checked(scheduler::inspect_kernel(decode.executable));
  for (std::size_t i = 0; i < inspection.outputs().size(); ++i)
    decode.outputs.push_back(
        build.scratch(inspection.outputs()[i], inspection.output_requirements()[i]));
  const auto publication = prepare_boxdecode_publication(wire, outputs);
  const auto decode_member = build.recipe->members.size();
  auto publication_inputs = decode.outputs;
  build.recipe->members.push_back(std::move(decode));
  build.recipe->members.push_back(
      {publication.executable, std::move(publication_inputs), {{0, 0}}, {decode_member}, {}});
  const auto name = runtime.logical_outputs.empty() ? std::string("boxdecode")
                                                    : runtime.logical_outputs.front().logical_name;
  runtime.physical_outputs = {
      specs::build_physical_buffer_static_spec(0, 0, publication.bytes, DeviceKind::Cpu, name)};
  runtime.logical_outputs = {specs::build_logical_output_static_spec(
      0, 0, 0, 0, 0, {static_cast<std::int64_t>(publication.bytes)}, "UINT8", "", name, name, name,
      0, publication.bytes)};
  runtime.output_order = {specs::build_output_route_static_spec(0, 0, 0, name, name)};
  auto metrics = std::make_shared<gs::StageMetrics>();
  metrics->phase = "Exec";
  metrics->component = "boxdecode_stage_wall_v2";
  // The backend includes the dependent wire publication, not only the numerical decode.
  metrics->regions.push_back(
      {"Run", "boxdecode_backend_wall_v2", decode_member, decode_member + 1});
  build.recipe->metrics = std::move(metrics);
  build.compiled.recipe = std::move(build.recipe);
  return {std::move(build.compiled), decode_member};
}
} // namespace simaai::neat::pipeline_internal::sima

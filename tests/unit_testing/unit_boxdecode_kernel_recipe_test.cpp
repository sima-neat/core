#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include <simaai/kernels/authoring.h>
#include <iostream>
#include <stdexcept>

namespace {
using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
namespace gs = simaai::runtime::gstreamer;
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
CompiledBoxDecodeContract source(BoxDecodeType type, int classes = 2) {
  CompiledBoxDecodeContract s;
  s.payload.decode_type = type;
  s.payload.num_classes = classes;
  s.payload.topk = 4;
  s.payload.detection_threshold = .5;
  s.payload.nms_iou_threshold = .5;
  s.payload.score_activation = BoxDecodeScoreActivation::Sigmoid;
  s.payload.decode_type_option = BoxDecodeTypeOption::InterleavedByHead;
  return s;
}
void input(CompiledBoxDecodeContract& s, int h, int w, int c, const std::string& dtype = "BF16",
           int kind = 1, int offset = 0, bool chw = false) {
  const int i = s.runtime_contract.logical_inputs.size();
  auto shape = chw ? std::vector<std::int64_t>{c, h, w} : std::vector<std::int64_t>{h, w, c};
  auto logical = specbuilders::build_logical_input_static_spec(
      i, i, i, shape, dtype, chw ? "CHW" : "HWC", "input" + std::to_string(i),
      "input" + std::to_string(i), "input" + std::to_string(i), offset);
  if (dtype == "INT8")
    logical.quant = QuantStaticSpec{QuantGranularity::PerTensor, -1, {3.1}, {-3}};
  s.runtime_contract.logical_inputs.push_back(std::move(logical));
  s.payload.tensor_storage_kind.push_back(kind);
  sima_ev_shape_desc slice{};
  slice.rank = 3;
  slice.sizes[0] = h;
  slice.sizes[1] = w;
  slice.sizes[2] = c;
  s.payload.slice_shapes.push_back(slice);
}
BoxDecodeKernelRecipeResult check(const CompiledBoxDecodeContract& s,
                                  BoxDecodeGeometryOptions geometry = {64, 64, 64, 64,
                                                                       ResizeMode::Letterbox}) {
  auto r = prepare_boxdecode_kernel(s, geometry);
  const auto& recipe = *r.compiled.recipe;
  require(recipe.members.size() >= 2, "decode/publication members missing");
  require(recipe.members.back().dependencies == std::vector<std::size_t>{r.decode_member},
          "publication must depend on decode");
  require(recipe.metrics && recipe.metrics->version == 1 && recipe.metrics->phase == "Exec" &&
              recipe.metrics->component == "boxdecode_stage_wall_v2",
          "recipe must declare the versioned outer stage wall scope");
  require(recipe.metrics->regions.size() == 1, "one backend region per frame, not per member");
  const auto& region = recipe.metrics->regions.front();
  require(region.phase == "Run" && region.component == "boxdecode_backend_wall_v2" &&
              region.first_member == r.decode_member &&
              region.last_member == recipe.members.size() - 1,
          "backend timing must include wire publication and exclude preceding conversion helpers");
  require(r.compiled.runtime_contract.logical_outputs.size() == 1, "wire output count");
  for (auto port : recipe.members.back().inputs)
    require(port.source == gs::TensorStagePortSource::Output, "publication reads private outputs");
  return r;
}
void families() {
  auto rcnn = source(BoxDecodeType::RcnnStage1, 1);
  for (int side : {136, 68, 34, 17, 9}) {
    input(rcnn, side, side, 12);
    input(rcnn, side, side, 3);
  }
  const auto proposals = check(rcnn, {544, 544, 544, 544, ResizeMode::Letterbox});
  const auto& specs = proposals.compiled.recipe->members[proposals.decode_member]
                          .executable.state()
                          .contract->output_specs;
  require(specs.size() == 5 && specs[0].dtype == simaai::kernels::DType::Float32 &&
              specs[0].shape == simaai::kernels::Shape({16, 4}),
          "RCNN needs four per-level capacities and unrounded FP32 coordinates");

  for (auto type : {BoxDecodeType::YoloV26, BoxDecodeType::YoloV6}) {
    auto s = source(type);
    input(s, 8, 8, 4);
    input(s, 8, 8, 2);
    check(s);
  }
  for (auto type : {BoxDecodeType::YoloV5, BoxDecodeType::YoloV7}) {
    auto s = source(type);
    s.payload.decode_type_option = BoxDecodeTypeOption::Auto;
    for (int n : {8, 4, 2})
      input(s, n, n, 32);
    check(s);
  }
  for (auto type : {BoxDecodeType::YoloV8, BoxDecodeType::YoloV9, BoxDecodeType::YoloV10}) {
    auto s = source(type);
    for (int n : {8, 4, 2}) {
      input(s, n, n, 64);
      input(s, n, n, 2);
    }
    check(s);
  }
  auto x = source(BoxDecodeType::YoloX);
  x.payload.decode_type_option = BoxDecodeTypeOption::Split3Interleaved;
  for (int n : {8, 4, 2}) {
    input(x, n, n, 4);
    input(x, n, n, 1);
    input(x, n, n, 2);
  }
  check(x);
  auto decoded = source(BoxDecodeType::Yolo);
  input(decoded, 8, 8, 4);
  input(decoded, 8, 8, 2);
  check(decoded);
  auto det = source(BoxDecodeType::Detr);
  input(det, 1, 8, 4);
  input(det, 1, 8, 3);
  det.payload.score_activation = BoxDecodeScoreActivation::Softmax;
  check(det, {64, 64, 64, 64, ResizeMode::Stretch});
  auto center = source(BoxDecodeType::Centernet);
  center.payload.decode_type_option = BoxDecodeTypeOption::Split3Interleaved;
  input(center, 8, 8, 2);
  input(center, 8, 8, 2);
  input(center, 8, 8, 2);
  check(center);
  auto eff = source(BoxDecodeType::EffDet);
  for (int n : {32, 16, 8, 4, 2}) {
    input(eff, n, n, 36);
    input(eff, n, n, 32);
  }
  check(eff, {256, 256, 256, 256, ResizeMode::Letterbox});
  auto ssd = source(BoxDecodeType::Ssd, 81);
  ssd.payload.ssd_recipe_id = stagesemantics::SsdRecipeId::Ssd300V1;
  ssd.payload.score_activation = BoxDecodeScoreActivation::Softmax;
  const int dims[]{38, 19, 10, 5, 3, 1}, anchors[]{4, 6, 6, 6, 4, 4};
  for (int i = 0; i < 6; ++i)
    input(ssd, dims[i], dims[i], anchors[i] * 4);
  for (int i = 0; i < 6; ++i)
    input(ssd, dims[i], dims[i], anchors[i] * 81);
  check(ssd, {300, 300, 300, 300, ResizeMode::Stretch});
}
void auxiliary_tasks() {
  for (auto type : {BoxDecodeType::YoloV26Pose, BoxDecodeType::YoloV8Pose}) {
    auto s = source(type, 1);
    s.payload.decode_type_option = BoxDecodeTypeOption::InterleavedByHead;
    const bool dfl = type == BoxDecodeType::YoloV8Pose;
    for (int n : {8, 4, 2}) {
      input(s, n, n, dfl ? 64 : 4);
      input(s, n, n, 1);
      input(s, n, n, 51);
    }
    check(s);
  }
  for (auto type :
       {BoxDecodeType::YoloV26Seg, BoxDecodeType::YoloV5Seg, BoxDecodeType::YoloV7Seg,
        BoxDecodeType::YoloV8Seg, BoxDecodeType::YoloV9Seg, BoxDecodeType::YoloV10Seg}) {
    auto s = source(type);
    const bool combined = type == BoxDecodeType::YoloV5Seg || type == BoxDecodeType::YoloV7Seg;
    const bool raw = type == BoxDecodeType::YoloV26Seg;
    for (int n : {80, 40, 20}) {
      input(s, n, n, combined ? 32 : raw ? 4 : 64);
      if (!combined)
        input(s, n, n, 2);
      input(s, n, n, combined ? 96 : 32);
    }
    input(s, 160, 160, 32);
    check(s, {640, 640, 640, 640, ResizeMode::Letterbox});
  }
}

void normalization() {
  for (auto dtype : {"BF16", "INT8", "FP32"}) {
    auto s = source(BoxDecodeType::YoloV26);
    input(s, 8, 8, 4, dtype, 1, 64);
    input(s, 8, 8, 2, dtype);
    const auto r = check(s);
    require(r.compiled.recipe->members.front().inputs.front().byte_offset == 64,
            "source offset lost");
    require(r.decode_member == (std::string(dtype) == "FP32" ? 2 : 0), "unexpected conversion");
  }
  auto rank6 = source(BoxDecodeType::YoloV26);
  input(rank6, 8, 8, 4);
  input(rank6, 8, 8, 2);
  for (auto& logical : rank6.runtime_contract.logical_inputs) {
    const auto bytes = static_cast<std::int64_t>(logical.size_bytes);
    logical.shape.insert(logical.shape.begin(), 3, 1);
    logical.stride_bytes.insert(logical.stride_bytes.begin(), 3, bytes);
  }
  check(rank6);
  auto chw = source(BoxDecodeType::YoloV26);
  input(chw, 8, 8, 4, "FP32", 1, 0, true);
  input(chw, 8, 8, 2, "FP32", 1, 0, true);
  require(check(chw).decode_member == 2, "CHW FP32 normalization missing");
  auto packed = source(BoxDecodeType::YoloV26);
  input(packed, 8, 8, 16, "INT8", 0);
  input(packed, 8, 8, 16, "INT8", 0);
  require(check(packed).decode_member == 0, "C-block coordinates must not add a detess pass");
  auto hwc16 = source(BoxDecodeType::YoloV26);
  input(hwc16, 8, 8, 16, "BF16", 2);
  input(hwc16, 8, 8, 16, "BF16", 2);
  require(check(hwc16).decode_member == 2, "HWC-C16 normalization missing");
}
void superpoint() {
  for (auto profile : {SuperPointProfile::A65V1, SuperPointProfile::LightGlueV1,
                       SuperPointProfile::MagicLeapDemoV1}) {
    auto s = source(BoxDecodeType::SuperPoint);
    s.payload.superpoint.profile = profile;
    s.payload.superpoint.nms_radius = 4;
    s.payload.superpoint.border_margin = 4;
    s.payload.superpoint.descriptor_dim = 256;
    s.payload.tensor_roles = {BoxDecodeTensorRole::DescriptorGrid,
                              BoxDecodeTensorRole::DetectorLogits};
    input(s, 8, 8, 256, "FP32");
    input(s, 8, 8, 65, "FP32");
    const auto r = check(s);
    require(r.compiled.recipe->members[r.decode_member].inputs[0].tensor == 1,
            "SuperPoint explicit role ignored");
  }
  // MLA BF16 pixel-major storage is byte-planar, not dense padded BF16.
  // Keep the explicit native layout and avoid an unnecessary detess member.
  auto tiled = source(BoxDecodeType::SuperPoint);
  tiled.payload.superpoint.profile = SuperPointProfile::A65V1;
  tiled.payload.superpoint.nms_radius = 4;
  tiled.payload.superpoint.border_margin = 4;
  tiled.payload.superpoint.descriptor_dim = 256;
  tiled.payload.tensor_roles = {BoxDecodeTensorRole::DetectorLogits,
                                BoxDecodeTensorRole::DescriptorGrid};
  input(tiled, 8, 8, 65, "BF16", 2);
  input(tiled, 8, 8, 256, "BF16", 2);
  require(check(tiled).decode_member == 0, "native SuperPoint tiled storage was copied");
  require(check(tiled, {64, 64, 64, 64, ResizeMode::Crop}).decode_member == 0,
          "model-space SuperPoint inherited BoxDecode inverse-crop restrictions");
  for (const auto& dtype : {"BF16", "INT8", "FP32"}) {
    auto padded = tiled;
    padded.payload.tensor_storage_kind = {1, 1};
    auto& detector = padded.runtime_contract.logical_inputs[0];
    detector.shape = {8, 8, 80};
    detector.dtype = dtype;
    detector.stride_bytes.clear();
    if (detector.dtype == "INT8")
      detector.quant = QuantStaticSpec{QuantGranularity::PerTensor, -1, {3.1}, {-3}};
    const auto prepared = check(padded);
    require(prepared.decode_member == 0, "padded SuperPoint must use the native dense reader");
    const auto& numerical = prepared.compiled.recipe->members.front().executable.state();
    const auto expected = detector.dtype == "INT8"   ? simaai::kernels::DType::Int8
                          : detector.dtype == "BF16" ? simaai::kernels::DType::BFloat16
                                                     : simaai::kernels::DType::Float32;
    require(numerical.contract->input_specs[0].dtype == expected &&
                numerical.contract->input_specs[0].shape.dimensions().back() == 65 &&
                std::holds_alternative<simaai::kernels::Strided>(
                    numerical.contract->input_specs[0].layout),
            "padded SuperPoint must retain its scalar dtype and explicit pixel stride");
  }
}
void rejects() {
  auto s = source(BoxDecodeType::YoloV26);
  input(s, 8, 8, 4, "INT8");
  input(s, 8, 8, 2, "INT8");
  s.runtime_contract.logical_inputs[0].quant.reset();
  try {
    check(s);
    throw std::runtime_error("missing quantization accepted");
  } catch (const std::invalid_argument&) {
  }
  s = source(BoxDecodeType::YoloV26);
  input(s, 8, 8, 4);
  input(s, 8, 8, 2);
  try {
    check(s, {64, 64, 64, 64, ResizeMode::Crop});
    throw std::runtime_error("crop accepted");
  } catch (const std::invalid_argument&) {
  }
}
} // namespace
int main() {
  try {
    families();
    auxiliary_tasks();
    normalization();
    superpoint();
    rejects();
    std::cout << "BoxDecode recipe families, normalization, roles and rejection contracts passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

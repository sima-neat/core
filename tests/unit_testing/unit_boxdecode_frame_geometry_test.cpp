// boxdecode_geometry_for_frames against the per-frame resolver it restates
// (BoxDecodeGeometry.h): for a set of frame facts and of stage geometries, one buffer that carries
// the facts through the resolver, and the same facts through the function, give the same answer:
// the same refusal with the same text, the stage's cold kernel, or a variant. For a variant the
// function's request is one the Kernels package checks, and its values prepare the variant.
// No device: the A65 kernel is prepared, never run.

#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <simaai/scheduler/testing/probe.h>

#include <gstsimaaitensorbuffer_meta.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace k = simaai::scheduler::client;
namespace probe = simaai::scheduler::testing;
namespace core = simaai::neat::pipeline_internal::sima;

struct BufferFree {
  void operator()(GstBuffer* p) const {
    if (p)
      gst_buffer_unref(p);
  }
};
using Buffer = std::unique_ptr<GstBuffer, BufferFree>;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class T> T take(k::Result<T> result) {
  if (const auto* error = std::get_if<k::Error>(&result))
    throw std::runtime_error(error->field + ": " + error->message);
  return std::get<T>(std::move(result));
}

// A decode whose "image_geometry" variant takes an optional axis-aligned transform: the request
// states the cold seed, a 640x640 image and, when the stage has one, the seed's transform.
core::KernelStageRecipe::Request request(const std::optional<std::array<float, 4>>& transform) {
  std::vector<k::TensorSpec> inputs;
  k::ParameterRecords heads;
  for (int side : {80, 40, 20}) {
    const auto port = inputs.size();
    for (int channels : {4, 3, 32, 3})
      inputs.push_back({{side, side, channels}, k::DType::BFloat16});
    heads.push_back({{"boxes_input", port},
                     {"scores_input", port + 1},
                     {"auxiliary_input", port + 2},
                     {"keypoints_input", port + 3}});
  }
  inputs.push_back({{160, 160, 32}, k::DType::BFloat16});
  std::vector<k::TensorSpec> outputs{
      {{4, 4}, k::DType::Int32}, {{4}, k::DType::Float32},         {{4}, k::DType::Int32},
      {{1}, k::DType::Int32},    {{4, 160, 160}, k::DType::UInt8}, {{4, 1, 3}, k::DType::Float32}};
  std::vector<k::Parameter> parameters{{"profile", "yolox_seg_pose_v1"},
                                       {"task", "segmentation_pose"},
                                       {"heads", std::move(heads)},
                                       {"model_input_size", k::ParameterArray{640, 640}},
                                       {"original_image_size", k::ParameterArray{640, 640}},
                                       {"num_classes", 2},
                                       {"max_detections", 4},
                                       {"detection_threshold", .5f},
                                       {"score_activation", "sigmoid"},
                                       {"mask_basis_input", 12},
                                       {"keypoints", 1}};
  if (transform) {
    const auto& t = *transform;
    parameters.push_back({"model_to_original", k::ParameterArray{t[0], t[1], t[2], t[3]}});
  }
  return {"boxdecode", std::move(inputs), std::move(outputs), std::move(parameters)};
}

simaai::gst::TensorBufferPublicationOwner publication() {
  using namespace simaai::gst;
  TensorBufferPublishContract contract;
  contract.stage_key = "frame-geometry-fixture";
  contract.physical_outputs.push_back({0, 4, "arena"});
  TensorBufferPublishLogicalOutput tensor;
  tensor.logical_index = tensor.physical_index = tensor.memory_index = 0;
  tensor.backend_output_index = tensor.route_slot = 0;
  tensor.logical_name = tensor.backend_name = "output";
  tensor.segment_name = "arena";
  tensor.size_bytes = 4;
  tensor.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  tensor.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  tensor.shape = {2, 2};
  tensor.stride_bytes = {2, 1};
  contract.logical_outputs.push_back(std::move(tensor));
  TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require(tensor_buffer_prepare_meta_template_from_contract(contract, &prepared, &error),
          "publication preparation failed");
  auto result = tensor_buffer_freeze_publication(prepared, &error);
  require(bool(result), "publication freeze failed");
  return result;
}

// One frame's facts, as the image preproc publishes them.
struct Facts {
  const char* name;
  int width = 640, height = 640, resized_width = 640, resized_height = 640;
  const char* resize = "letterbox";
  std::array<double, 6> affine{2., 0., 10., 0., 2., -10.};
  bool roi_list = false;
  bool none = false; ///< a buffer without facts
};

// A buffer that carries `f` as compact frame metadata (none: a bare buffer). Null when the facts
// are ones no stage can publish (the publication refuses them).
Buffer buffer_of(const Facts& f) {
  Buffer buffer(gst_buffer_new_allocate(nullptr, 4, nullptr));
  if (f.none)
    return buffer;
  std::string error;
  require(simaai::gst::tensor_buffer_attach_publication(buffer.get(), publication(), &error),
          "attach publication failed");
  simaai::gst::TensorBufferPreprocessFacts facts;
  facts.original_width = f.width;
  facts.original_height = f.height;
  facts.resized_width = f.resized_width;
  facts.resized_height = f.resized_height;
  facts.primary = {0, 0, f.width, f.height};
  facts.placed = {0, 0, f.resized_width, f.resized_height};
  facts.affine = f.affine;
  facts.source_stride = static_cast<std::int64_t>(f.width) * 3;
  facts.resize_mode = f.resize;
  facts.color_in = facts.color_out = "RGB";
  if (f.roi_list) {
    facts.roi_list = true;
    facts.input_batch = facts.capacity = 1;
    facts.counts = {1, 1, 0, 0};
    facts.roi_batch_indices = {0};
    facts.roi_rects = {0, 0, f.width, f.height};
    facts.roi_affines.assign(f.affine.begin(), f.affine.end());
  }
  simaai::gst::TensorBufferFrameMetadata frame;
  frame.preprocess = simaai::gst::tensor_buffer_freeze_preprocess_facts(facts, &error);
  if (!frame.preprocess)
    return nullptr;
  require(simaai::gst::tensor_buffer_attach_frame_metadata(buffer.get(), std::move(frame), &error),
          "facts attach failed");
  return buffer;
}

core::BoxDecodeFrameFacts function_facts(const Facts& f) {
  core::BoxDecodeFrameFacts facts;
  if (f.none)
    return facts;
  facts.original_width = f.width;
  facts.original_height = f.height;
  facts.resized_width = f.resized_width;
  facts.resized_height = f.resized_height;
  facts.resize_mode = f.resize;
  facts.roi_list = f.roi_list;
  facts.affine = f.affine;
  return facts;
}

// A stage geometry: the resolver's options and its two flags.
struct Stage {
  const char* name;
  core::BoxDecodeGeometryOptions options;
  bool size_from_frame, resize_required;
};

std::size_t compare(const Stage& stage, const std::vector<Facts>& cases) {
  probe::OperationProbe executor;
  const core::KernelStageRecipe::Request cold = request(stage.options.model_to_original);
  const auto checked = take(k::check_kernel(
      {cold.operation, k::Target::A65, cold.inputs, cold.outputs, cold.parameters}));
  require(!executor.prepare(checked), "A65 preparation failed");
  const auto base = executor.operation();
  core::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = checked;
  recipe.boxdecode_program.emplace();
  recipe.boxdecode_program->request = cold;
  core::attach_boxdecode_geometry_resolver(recipe, 0, stage.options, stage.size_from_frame,
                                           stage.resize_required);
  require(recipe.resolver_factories.size() == 1, std::string(stage.name) + ": one resolver");
  auto resolve = recipe.resolver_factories.front().instantiate();
  const auto variant = take(k::bind_kernel_variant(checked, "image_geometry"));
  std::size_t refused = 0;
  for (const Facts& f : cases) {
    const std::string at = std::string(stage.name) + ", " + f.name + ": ";
    const Buffer buffer = buffer_of(f);
    if (!buffer) {
      std::cout << "  " << at << "not publishable, left out\n";
      continue;
    }
    const std::array<GstBuffer*, 1> buffers{buffer.get()};
    const auto resolved = resolve(executor.preparer(), buffers);
    const auto selected = core::boxdecode_geometry_for_frames(recipe, function_facts(f));
    const auto* resolver_error = std::get_if<k::Error>(&resolved);
    const auto* function_error = std::get_if<k::Error>(&selected);
    require(bool(resolver_error) == bool(function_error),
            at + "the resolver " +
                (resolver_error ? "refuses (" + resolver_error->message + ")"
                                : std::string("accepts")) +
                ", the function " +
                (function_error ? "refuses (" + function_error->message + ")"
                                : std::string("accepts")));
    if (resolver_error) {
      require(resolver_error->message == function_error->message,
              at + "the resolver says '" + resolver_error->message + "', the function '" +
                  function_error->message + "'");
      ++refused;
      std::cout << "  " << at << "refused: " << function_error->message << "\n";
      continue;
    }
    const auto& operation = std::get<k::PreparedSelection>(resolved);
    const auto& geometry = std::get<core::BoxDecodeFrameGeometry>(selected);
    require((operation == base) == geometry.cold,
            at + "the resolver selects " + (operation == base ? "the cold kernel" : "a variant") +
                ", the function says " + (geometry.cold ? "the cold kernel" : "a variant"));
    require(geometry.variant == "image_geometry", at + "the stage binds image_geometry");
    // The request with the frame's values is one the Kernels package checks, and the values
    // prepare the variant on the cold kernel's resources.
    (void)take(k::check_kernel({geometry.request.operation, k::Target::A65, geometry.request.inputs,
                                geometry.request.outputs, geometry.request.parameters}));
    std::array<std::optional<k::ParameterValue>, 2> values{
        k::ParameterValue{k::ParameterArray{geometry.image_size[0], geometry.image_size[1]}}, {}};
    if (geometry.model_to_original) {
      const auto& t = *geometry.model_to_original;
      values[1] = k::ParameterArray{t[0], t[1], t[2], t[3]};
    }
    const auto prepared = take(executor.preparer().prepare(variant, {{}, {}, {values.data(), 2U}}));
    require(k::same_kernel_contract_resources(prepared.executable(), checked),
            at + "the function's values prepare the variant on the cold kernel's resources");
    std::cout << "  " << at << (geometry.cold ? "the cold kernel" : "a variant") << " for "
              << geometry.image_size[0] << "x" << geometry.image_size[1] << "\n";
  }
  return refused;
}

} // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  try {
    using simaai::neat::ResizeMode;
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<Facts> cases{
        {.name = "the seed's geometry"},
        {.name = "another offset", .affine = {2., 0., 12., 0., 2., -10.}},
        {.name = "another image size", .width = 1280, .height = 720},
        {.name = "unequal scales", .affine = {2., 0., 10., 0., 3., -10.}},
        {.name = "no facts", .none = true},
        {.name = "an ROI list", .roi_list = true},
        {.name = "a shear", .affine = {2., .1, 10., 0., 2., -10.}},
        {.name = "a negative scale", .affine = {-2., 0., 10., 0., 2., -10.}},
        {.name = "an infinite scale", .affine = {inf, 0., 10., 0., 2., -10.}},
        {.name = "another canvas", .resized_width = 320},
        {.name = "another resize mode", .resize = "stretch"},
        {.name = "an image size out of range", .width = 2000000},
    };
    // The seed every stage below was checked for: 640x640, original = model * 2 + (10, -10).
    core::BoxDecodeGeometryOptions seed{640, 640, 640, 640, ResizeMode::Letterbox};
    seed.model_to_original = std::array<float, 4>{2.F, 2.F, 10.F, -10.F};
    seed.complete_geometry = true;
    core::BoxDecodeGeometryOptions oriented = seed;
    oriented.uniform_transform_required = true;
    core::BoxDecodeGeometryOptions required = oriented;
    required.require_frame_transform = true;
    core::BoxDecodeGeometryOptions unseeded = seed;
    unseeded.model_to_original.reset();
    const std::vector<Stage> stages{
        {"each frame's size and affine", seed, true, true},
        {"each frame's affine, the size explicit", seed, false, false},
        {"equal scales (oriented boxes)", oriented, true, true},
        {"equal scales, each frame must bring its affine", required, true, true},
        {"no seed transform, resize mode not required", unseeded, true, false},
    };
    std::size_t refused = 0, compared = 0;
    for (const Stage& stage : stages) {
      refused += compare(stage, cases);
      compared += cases.size();
    }
    require(refused > 0 && refused < compared, "the cases refuse some frames and take others");
    std::cout << "boxdecode_geometry_for_frames agrees with the resolver on " << compared
              << " frames of " << stages.size() << " stage geometries (" << refused
              << " refused)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

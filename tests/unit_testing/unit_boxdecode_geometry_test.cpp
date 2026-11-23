#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
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

namespace {
namespace k = simaai::scheduler::client;
namespace probe = simaai::scheduler::testing;
namespace rt = simaai::scheduler::client;
namespace core = simaai::neat::pipeline_internal::sima;
struct BufferFree {
  void operator()(GstBuffer* p) const {
    if (p)
      gst_buffer_unref(p);
  }
};
using Buffer = std::unique_ptr<GstBuffer, BufferFree>;
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class T> T take(rt::Result<T> result) {
  if (const auto* error = std::get_if<rt::Error>(&result))
    throw std::runtime_error(error->field + ": " + error->message);
  return std::get<T>(std::move(result));
}
k::KernelExecutable executable(bool crop = false) {
  if (crop) {
    const std::array inputs{k::TensorSpec{{1, 1, 4}, k::DType::Float32},
                            k::TensorSpec{{1, 1, 3}, k::DType::Float32}};
    const std::array outputs{
        k::TensorSpec{{4, 4}, k::DType::Int32}, k::TensorSpec{{4}, k::DType::Float32},
        k::TensorSpec{{4}, k::DType::Int32}, k::TensorSpec{{1}, k::DType::Int32}};
    const std::array parameters{
        k::Parameter{"profile", "detr_v1"},
        k::Parameter{"heads", k::ParameterRecords{{{"boxes_input", 0}, {"scores_input", 1}}}},
        k::Parameter{"model_input_size", k::ParameterArray{640, 640}},
        k::Parameter{"original_image_size", k::ParameterArray{640, 640}},
        k::Parameter{"model_to_original", k::ParameterArray{2.f, 3.f, 10.f, -10.f}},
        k::Parameter{"resize", "stretch"},
        k::Parameter{"score_activation", "softmax"},
        k::Parameter{"num_classes", 2},
        k::Parameter{"max_detections", 4}};
    return take(k::check_kernel({"boxdecode", k::Target::A65, inputs, outputs, parameters}));
  }

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
  const std::array<k::TensorSpec, 6> outputs{{{{4, 4}, k::DType::Int32},
                                              {{4}, k::DType::Float32},
                                              {{4}, k::DType::Int32},
                                              {{1}, k::DType::Int32},
                                              {{4, 160, 160}, k::DType::UInt8},
                                              {{4, 1, 3}, k::DType::Float32}}};
  const std::array parameters{k::Parameter{"profile", "yolox_seg_pose_v1"},
                              k::Parameter{"task", "segmentation_pose"},
                              k::Parameter{"heads", std::move(heads)},
                              k::Parameter{"model_input_size", k::ParameterArray{640, 640}},
                              k::Parameter{"original_image_size", k::ParameterArray{640, 640}},
                              k::Parameter{"num_classes", 2},
                              k::Parameter{"max_detections", 4},
                              k::Parameter{"detection_threshold", .5f},
                              k::Parameter{"score_activation", "sigmoid"},
                              k::Parameter{"mask_basis_input", 12},
                              k::Parameter{"keypoints", 1}};
  return take(k::check_kernel({"boxdecode", k::Target::A65, inputs, outputs, parameters}));
}
void set_geometry(GstStructure* meta, double offset = 10.) {
  gst_structure_remove_all_fields(meta);
  gst_structure_set(meta, "preproc_original_width", G_TYPE_INT, 640, "preproc_original_height",
                    G_TYPE_INT, 640, "preproc_resized_width", G_TYPE_INT, 640,
                    "preproc_resized_height", G_TYPE_INT, 640, "preproc_resize_mode", G_TYPE_STRING,
                    "letterbox", "preproc_affine_m00", G_TYPE_DOUBLE, 2., "preproc_affine_m01",
                    G_TYPE_DOUBLE, 0., "preproc_affine_m02", G_TYPE_DOUBLE, offset,
                    "preproc_affine_m10", G_TYPE_DOUBLE, 0., "preproc_affine_m11", G_TYPE_DOUBLE,
                    3., "preproc_affine_m12", G_TYPE_DOUBLE, -10., nullptr);
}
simaai::gst::TensorBufferPublicationOwner compact_publication() {
  using namespace simaai::gst;
  TensorBufferPublishContract contract;
  contract.stage_key = "combined-geometry-fixture";
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
          "compact publication preparation failed");
  auto result = tensor_buffer_freeze_publication(prepared, &error);
  require(bool(result), "compact publication freeze failed");
  return result;
}
void geometry(bool crop = false) {
  probe::OperationProbe executor;
  const auto checked = executable(crop);
  require(!executor.prepare(checked), "real A65 preparation failed");
  const auto base = executor.operation();
  core::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = checked;
  core::BoxDecodeGeometryOptions options{640, 640, 640, 640, simaai::neat::ResizeMode::Letterbox};
  options.complete_geometry = true;
  if (crop) {
    options.resize = simaai::neat::ResizeMode::Crop;
    options.require_frame_transform = true;
  }
  const auto set = [crop](GstStructure* meta, double offset = 10.) {
    set_geometry(meta, offset);
    if (crop)
      gst_structure_set(meta, "preproc_resize_mode", G_TYPE_STRING, "crop", nullptr);
  };
  core::attach_boxdecode_geometry_resolver(recipe, 0, options, true, true);
  require(recipe.resolver_factories.size() == 1 && !recipe.members.front().resolve_factory,
          "complete geometry recipe owns live mutable state");
  auto resolve = recipe.resolver_factories.front().instantiate();
  auto independent = recipe.resolver_factories.front().instantiate();
  const gchar* tags[] = {nullptr};
  if (!gst_meta_get_info("GstSimaMeta"))
    require(gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr),
            "metadata registration failed");
  Buffer first(gst_buffer_new()), second(gst_buffer_new());
  auto* a = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(first.get(), "GstSimaMeta"));
  auto* b = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(second.get(), "GstSimaMeta"));
  const std::array<GstBuffer*, 2> buffers{first.get(), second.get()};
  set(a);
  set(b);
  const auto first_variant = take(resolve(executor.preparer(), buffers));
  require(first_variant != base && take(resolve(executor.preparer(), buffers)) == first_variant,
          "complete geometry was not cached");
  require(k::same_kernel_contract_resources(first_variant.executable(), checked),
          "complete geometry rebuilt scratch or tensor contract");
  require(take(independent(executor.preparer(), buffers)) != first_variant,
          "independent preparation shared mutable geometry cache");
  set(a, 12.);
  set(b, 12.);
  const auto changed = take(resolve(executor.preparer(), buffers));
  require(changed != first_variant && changed != base,
          "same dimensions with changed affine reused stale geometry");
  set(a);
  set(b);
  require(take(resolve(executor.preparer(), buffers)) == first_variant,
          "immutable geometry snapshot was mutated");
  for (int failure = 0; failure < 8; ++failure) {
    set(a);
    set(b);
    switch (failure) {
    case 0:
      gst_structure_set(b, "preproc_affine_m02", G_TYPE_DOUBLE, 11., nullptr);
      break;
    case 1:
      gst_structure_remove_field(a, "preproc_affine_m11");
      break;
    case 2:
      gst_structure_set(a, "preproc_affine_m01", G_TYPE_DOUBLE, .1, nullptr);
      break;
    case 3:
      gst_structure_set(a, "preproc_affine_m00", G_TYPE_DOUBLE,
                        std::numeric_limits<double>::infinity(), nullptr);
      break;
    case 4:
      gst_structure_set(a, "preproc_affine_m00", G_TYPE_DOUBLE, -2., nullptr);
      break;
    case 5:
      gst_structure_set(a, "preproc_resized_width", G_TYPE_INT, 320, nullptr);
      break;
    case 6:
      gst_structure_set(a, "preproc_roi_list_enable", G_TYPE_BOOLEAN, TRUE, nullptr);
      break;
    case 7:
      gst_structure_set(a, "preproc_affine_scale_x", G_TYPE_DOUBLE, 3., "preproc_affine_scale_y",
                        G_TYPE_DOUBLE, 3., "preproc_affine_offset_x", G_TYPE_DOUBLE, 10.,
                        "preproc_affine_offset_y", G_TYPE_DOUBLE, -10., nullptr);
      break;
    }
    require(std::holds_alternative<rt::Error>(resolve(executor.preparer(), buffers)),
            "invalid complete geometry accepted");
  }
  set(a);
  set(b);
  require(take(resolve(executor.preparer(), buffers)) == first_variant,
          "invalid frame evicted a valid warmed snapshot");
  // Bound cache may evict, but old snapshots remain immutable and retain resources.
  for (int i = 0; i < 6; ++i) {
    set(a, 30. + i);
    set(b, 30. + i);
    require(k::same_kernel_contract_resources(
                take(resolve(executor.preparer(), buffers)).executable(), checked),
            "geometry cache eviction changed resource identity");
  }
  set(a);
  set(b);
  require(take(resolve(executor.preparer(), buffers)) != first_variant &&
              k::same_kernel_contract_resources(first_variant.executable(), checked),
          "eviction invalidated retained snapshot or cache was unbounded");
  // Compact producer facts must select exactly the same checked geometry as
  // the legacy boundary serializer, including the affine direction.
  Buffer compact(gst_buffer_new_allocate(nullptr, 4, nullptr));
  std::string error;
  require(
      simaai::gst::tensor_buffer_attach_publication(compact.get(), compact_publication(), &error),
      "attach compact publication failed");
  simaai::gst::TensorBufferPreprocessFacts facts;
  facts.original_width = facts.original_height = facts.resized_width = facts.resized_height = 640;
  facts.primary = {10, -10, 640, 639};
  facts.placed = {0, 0, 320, 213};
  facts.affine = {2., 0., 10., 0., 3., -10.};
  facts.source_stride = 1920;
  facts.resize_mode = crop ? "crop" : "letterbox";
  if (crop) {
    facts.primary = {0, 0, 640, 640};
    facts.placed = {0, 0, 640, 640};
  }
  facts.color_in = facts.color_out = "RGB";
  simaai::gst::TensorBufferFrameMetadata frame;
  frame.preprocess = simaai::gst::tensor_buffer_freeze_preprocess_facts(facts, &error);
  require(bool(frame.preprocess), "compact facts freeze failed");
  require(simaai::gst::tensor_buffer_attach_frame_metadata(compact.get(), std::move(frame), &error),
          "compact facts attach failed");
  const std::array<GstBuffer*, 2> mixed{first.get(), compact.get()};
  require(take(resolve(executor.preparer(), mixed)) == take(resolve(executor.preparer(), buffers)),
          "compact and legacy affine authorities differ");
  // Explicit extent does not suppress dynamic affine updates.
  core::KernelStageRecipe explicit_recipe;
  explicit_recipe.members = recipe.members;
  core::attach_boxdecode_geometry_resolver(explicit_recipe, 0, options, false, false);
  auto explicit_resolve = explicit_recipe.resolver_factories.front().instantiate();
  require(take(explicit_resolve(executor.preparer(), buffers)) != base,
          "explicit image dimensions suppressed actual preprocessing affine");
  gst_structure_set(a, "preproc_original_width", G_TYPE_INT, 800, nullptr);
  require(std::holds_alternative<rt::Error>(explicit_resolve(executor.preparer(), buffers)),
          "affine source extent conflict accepted");
}
// Placement is derived from published pads (or the Preproc policy before the
// first frame), never from tensor shapes. One odd pixel is centering remainder.
void letterbox_placement() {
  simaai::neat::PreprocessRuntimeMeta facts;
  facts.resize_mode = "letterbox";
  facts.original_width = 1280;
  facts.original_height = 720;
  facts.scaled_width = 1024;
  facts.scaled_height = 576;
  facts.pad_top = facts.pad_bottom = 224;
  require(!core::boxdecode_letterbox_is_off_center(&facts, nullptr),
          "centered letterbox reported as off-center");
  facts.pad_top = 0;
  facts.pad_bottom = 448;
  require(core::boxdecode_letterbox_is_off_center(&facts, nullptr),
          "SAM top-left letterbox (pad_bottom 448) was not detected");
  facts.pad_top = 1;
  facts.pad_bottom = 0;
  require(!core::boxdecode_letterbox_is_off_center(&facts, nullptr),
          "one-pixel centering remainder must stay centered");
  facts.pad_top = facts.pad_bottom = 0;
  facts.pad_left = 0;
  facts.pad_right = 2;
  require(core::boxdecode_letterbox_is_off_center(&facts, nullptr),
          "horizontal off-center placement was not detected");
  facts.resize_mode = "stretch";
  require(!core::boxdecode_letterbox_is_off_center(&facts, nullptr), "stretch has no placement");
  core::ProcessCvuStagePayload policy;
  policy.aspect_ratio = 1;
  for (const char* padding : {"TOPLEFT", "topleft", "TopRight", "BOTTOMLEFT", "bottomright"}) {
    policy.padding_type = padding;
    require(core::boxdecode_letterbox_is_off_center(nullptr, &policy),
            "Preproc corner placement was not detected");
  }
  for (const char* padding : {"CENTER", "NONE", ""}) {
    policy.padding_type = padding;
    require(!core::boxdecode_letterbox_is_off_center(nullptr, &policy),
            "centered or crop Preproc policy reported as off-center");
  }
  policy.padding_type = "TOPLEFT";
  policy.aspect_ratio = 0;
  require(!core::boxdecode_letterbox_is_off_center(nullptr, &policy),
          "a stretch Preproc policy has no placement");
  policy.aspect_ratio = 1;
  facts.resize_mode = "letterbox";
  facts.pad_left = facts.pad_right = 0;
  require(!core::boxdecode_letterbox_is_off_center(&facts, &policy),
          "published facts must take precedence over the Preproc policy");
  using simaai::neat::BoxDecodeType;
  for (const auto type : {BoxDecodeType::YoloV8, BoxDecodeType::YoloV26, BoxDecodeType::Detr,
                          BoxDecodeType::DetrFocal, BoxDecodeType::Ssd})
    require(core::boxdecode_supports_detection_transform(type),
            "detection profile lost its explicit transform");
  for (const auto type : {BoxDecodeType::YoloV8Seg, BoxDecodeType::YoloV8Pose,
                          BoxDecodeType::RcnnStage1, BoxDecodeType::SuperPoint})
    require(!core::boxdecode_supports_detection_transform(type),
            "auxiliary or proposal payload implicitly admitted an explicit transform");
}
// Segment Anything geometry comes only from published preprocess facts: a
// top-left letterbox (MobileSAM) or a stretch (EdgeTAM, EfficientSAM,
// EfficientSAM3); the model size is the canvas, never a tensor-shape ratio.
void sam_geometry() {
  using simaai::neat::BoxDecodeType;
  using simaai::neat::ResizeMode;
  // The error names the encoder preprocess setting the model needs.
  const auto fails = [](BoxDecodeType type, const simaai::neat::PreprocessRuntimeMeta& facts,
                        core::BoxDecodeGeometryOptions requested, const char* message = "") {
    try {
      (void)core::sam_boxdecode_geometry(type, facts, requested);
    } catch (const std::invalid_argument& error) {
      return std::string(error.what()).find(message) != std::string::npos;
    }
    return false;
  };
  // A Preproc publishes original = model * (original / scaled) - pad * that scale.
  const auto whole_image = [](simaai::neat::PreprocessRuntimeMeta& facts) {
    facts.affine_m00 = double(facts.original_width) / facts.scaled_width;
    facts.affine_m11 = double(facts.original_height) / facts.scaled_height;
    facts.affine_m02 = -facts.pad_left * facts.affine_m00;
    facts.affine_m12 = -facts.pad_top * facts.affine_m11;
  };
  simaai::neat::PreprocessRuntimeMeta letterbox;
  letterbox.resize_mode = "letterbox";
  letterbox.original_width = 1280;
  letterbox.original_height = 720;
  letterbox.resized_width = letterbox.resized_height = 1024;
  letterbox.scaled_width = 1024;
  letterbox.scaled_height = 576;
  letterbox.pad_bottom = 448;
  whole_image(letterbox);
  auto g = core::sam_boxdecode_geometry(BoxDecodeType::MobileSam, letterbox, {});
  require(g.resize == ResizeMode::Letterbox && g.model_width == 1024 && g.model_height == 1024 &&
              g.original_width == 1280 && g.original_height == 720 && !g.complete_geometry,
          "MobileSAM top-left letterbox geometry");
  constexpr const char* kTopLeft =
      "MobileSAM needs the image preprocessed with resize.mode=Letterbox and "
      "resize.placement=TopLeft on the Model that preprocesses it";
  auto centered = letterbox;
  centered.pad_top = centered.pad_bottom = 224;
  whole_image(centered);
  require(fails(BoxDecodeType::MobileSam, centered, {}, kTopLeft),
          "a centered letterbox must be rejected with the MobileSAM encoder setting");
  // The kernel crops int(extent * canvas / longest + 0.5) pixels: 1000x601 scales to
  // 1024x615. A standalone TOPLEFT Preproc rounds upward to 616.
  auto odd = letterbox;
  odd.original_width = 1000;
  odd.original_height = 601;
  odd.scaled_height = 615;
  odd.pad_bottom = 1024 - 615;
  whole_image(odd);
  g = core::sam_boxdecode_geometry(BoxDecodeType::MobileSam, odd, {});
  require(g.original_width == 1000 && g.original_height == 601 && g.model_height == 1024,
          "the SAM-rounded 1000x601 letterbox must be admitted");
  auto rounded_up = odd;
  rounded_up.scaled_height = 616;
  rounded_up.pad_bottom = 1024 - 616;
  whole_image(rounded_up);
  require(fails(BoxDecodeType::MobileSam, rounded_up, {},
                "masks need the 1000x601 image scaled to 1024x615 on its 1024x1024 canvas; its "
                "preprocess facts report 1024x616"),
          "an upward-rounded scaled extent must be rejected");
  auto wide = letterbox;
  wide.resized_height = 768;
  wide.scaled_height = 576;
  wide.pad_bottom = 192;
  whole_image(wide);
  require(fails(BoxDecodeType::MobileSam, wide, {}, "letterboxes onto a square canvas"),
          "a non-square letterbox canvas must be rejected");
  // A crop composed before the resize keeps the resize mode but maps the canvas onto
  // part of the image.
  auto cropped = letterbox;
  cropped.affine_m02 = 160.;
  require(fails(BoxDecodeType::MobileSam, cropped, {},
                "the preprocess affine must map the 1024x576 scaled image onto the whole "
                "1280x720 image (affine_m00 = 1280/1024, affine_m11 = 720/576"),
          "a letterbox of a cropped region must be rejected");
  auto identity = letterbox;
  identity.affine_m00 = identity.affine_m11 = 1.;
  identity.affine_m02 = identity.affine_m12 = 0.;
  require(fails(BoxDecodeType::MobileSam, identity, {}, "or facts without that affine"),
          "facts without their model-to-original affine must be rejected");
  core::BoxDecodeGeometryOptions transform;
  transform.model_to_original = std::array<float, 4>{1.25f, 1.25f, 0.f, 0.f};
  require(fails(BoxDecodeType::MobileSam, letterbox, transform, "explicit model-to-original"),
          "an explicit transform must be rejected");
  for (const auto& [type, model] : {std::pair{BoxDecodeType::EdgeTamSeg, "EdgeTAM"},
                                    std::pair{BoxDecodeType::EdgeTamTrack, "EdgeTAM"},
                                    std::pair{BoxDecodeType::EfficientSam, "EfficientSAM"},
                                    std::pair{BoxDecodeType::EfficientSam3, "EfficientSAM3"}})
    require(fails(type, letterbox, {},
                  (std::string(model) + " needs the image preprocessed with resize.mode=Stretch")
                      .c_str()),
            "a letterbox must be rejected with the stretch encoder setting");
  simaai::neat::PreprocessRuntimeMeta stretch;
  stretch.resize_mode = "stretch";
  stretch.original_width = 640;
  stretch.original_height = 480;
  stretch.resized_width = stretch.resized_height = 1008;
  stretch.scaled_width = stretch.scaled_height = 1008;
  whole_image(stretch);
  require(fails(BoxDecodeType::MobileSam, stretch, {}, kTopLeft),
          "a stretch must be rejected with the MobileSAM encoder setting");
  for (const auto type : {BoxDecodeType::EdgeTamSeg, BoxDecodeType::EdgeTamTrack,
                          BoxDecodeType::EfficientSam, BoxDecodeType::EfficientSam3}) {
    // A decoder Model's nominal Letterbox resize does not override the facts.
    core::BoxDecodeGeometryOptions nominal;
    nominal.resize = ResizeMode::Letterbox;
    g = core::sam_boxdecode_geometry(type, stretch, nominal);
    require(g.resize == ResizeMode::Stretch && g.model_width == 1008 && g.model_height == 1008 &&
                g.original_width == 640 && g.original_height == 480,
            "1008 stretch geometry");
  }
  core::BoxDecodeGeometryOptions requested;
  requested.model_width = requested.model_height = 1024;
  require(fails(BoxDecodeType::EdgeTamSeg, stretch, requested, "model size differs"),
          "a model size other than the published canvas must be rejected");
  requested = {};
  requested.original_width = 1920;
  requested.original_height = 1080;
  require(fails(BoxDecodeType::EdgeTamTrack, stretch, requested, "original image size conflicts"),
          "an original size other than the published one must be rejected");
  auto crop = stretch;
  crop.resize_mode = "crop";
  require(fails(BoxDecodeType::EfficientSam, crop, {}, "report resize_mode 'crop'"),
          "crop is not a SAM geometry");
  // Crop then Stretch: Preproc composes the crop into the affine and publishes the
  // last stage's stretch over the original 640x480 source.
  auto crop_then_stretch = stretch;
  crop_then_stretch.affine_m00 = 540. / 1008.;
  crop_then_stretch.affine_m02 = 100.;
  require(fails(BoxDecodeType::EfficientSam, crop_then_stretch, {}, "onto the whole 640x480 image"),
          "a stretch of a cropped region must be rejected");
  auto padded = stretch;
  padded.scaled_width = 1000;
  padded.pad_right = 8;
  whole_image(padded);
  require(fails(BoxDecodeType::EdgeTamSeg, padded, {}, "image scaled to 1008x1008"),
          "a stretch that leaves padding must be rejected");
}
// A prepared Segment Anything plan keeps the image and preprocessing it was
// admitted for. Every frame is checked against all of it, not only its size:
// a centered letterbox of the same image would otherwise get masks placed for
// the admitted top-left letterbox.
void sam_fixed_image() {
  probe::OperationProbe executor;
  const auto checked = executable();
  require(!executor.prepare(checked), "real A65 preparation failed");
  const auto base = executor.operation();
  core::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = checked;
  // MobileSAM: 1280x720 letterboxed top-left onto 1024x1024, scaled to 1024x576.
  const core::BoxDecodeFixedImage image{1280, 720, "letterbox", 1024, 1024, {0, 0, 1024, 576}};
  bool rejected = false;
  try {
    auto incomplete = image;
    incomplete.resize_mode.clear();
    core::attach_boxdecode_fixed_image_guard(recipe, 0, incomplete);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected && recipe.resolver_factories.empty(),
          "a fixed image without its preprocessing must be rejected");
  core::attach_boxdecode_fixed_image_guard(recipe, 0, image);
  auto resolve = recipe.resolver_factories.front().instantiate();
  const auto rejects = [&](GstBuffer* buffer, const char* message) {
    const std::array<GstBuffer*, 1> one{buffer};
    auto result = resolve(executor.preparer(), one);
    const auto* error = std::get_if<rt::Error>(&result);
    return error && error->message.find(message) != std::string::npos;
  };
  constexpr const char* kPreprocessing = "another resize mode, canvas or placement";
  const gchar* tags[] = {nullptr};
  if (!gst_meta_get_info("GstSimaMeta"))
    require(gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr),
            "metadata registration failed");
  Buffer legacy(gst_buffer_new()), prompt(gst_buffer_new());
  auto* meta =
      gst_custom_meta_get_structure(gst_buffer_add_custom_meta(legacy.get(), "GstSimaMeta"));
  const auto set = [meta](const char* mode, int original_width, int original_height, int canvas,
                          int scaled_width, int scaled_height, int top) {
    gst_structure_remove_all_fields(meta);
    gst_structure_set(
        meta, "preproc_original_width", G_TYPE_INT, original_width, "preproc_original_height",
        G_TYPE_INT, original_height, "preproc_resized_width", G_TYPE_INT, canvas,
        "preproc_resized_height", G_TYPE_INT, canvas, "preproc_scaled_width", G_TYPE_INT,
        scaled_width, "preproc_scaled_height", G_TYPE_INT, scaled_height, "preproc_pad_left",
        G_TYPE_INT, 0, "preproc_pad_right", G_TYPE_INT, canvas - scaled_width, "preproc_pad_top",
        G_TYPE_INT, top, "preproc_pad_bottom", G_TYPE_INT, canvas - scaled_height - top,
        "preproc_resize_mode", G_TYPE_STRING, mode, nullptr);
  };
  set("letterbox", 1280, 720, 1024, 1024, 576, 0);
  const std::array<GstBuffer*, 2> with_prompt{prompt.get(), legacy.get()};
  require(take(resolve(executor.preparer(), with_prompt)) == base,
          "the admitted frame and a prompt without facts must keep the prepared plan");
  // A later run whose inputs all lack facts must not reuse the previous image's geometry.
  require(rejects(prompt.get(), "semantic.preprocess facts on its inputs on every run"),
          "a run without any image facts must be rejected");
  set("letterbox", 1280, 720, 1024, 1024, 576, 0);
  gst_structure_set(meta, "preproc_roi_list_enable", G_TYPE_BOOLEAN, TRUE, nullptr);
  require(rejects(legacy.get(), "ROI-list"), "ROI-list facts must be rejected as at admission");
  set("letterbox", 1280, 720, 1024, 1024, 576, 224);
  require(rejects(legacy.get(), kPreprocessing), "a centered letterbox must be rejected");
  set("stretch", 1280, 720, 1024, 1024, 1024, 0);
  require(rejects(legacy.get(), kPreprocessing), "a stretch must be rejected");
  set("letterbox", 1280, 720, 1008, 1008, 567, 0);
  require(rejects(legacy.get(), kPreprocessing), "another canvas must be rejected");
  set("letterbox", 1280, 720, 1024, 1024, 577, 0);
  require(rejects(legacy.get(), kPreprocessing), "another scaled extent must be rejected");
  set("letterbox", 1920, 1080, 1024, 1024, 576, 0);
  require(rejects(legacy.get(), "sized for another original image"),
          "another image size must be rejected");
  // The kernel ignores the affine: a crop composed before the resize must fail every run.
  constexpr const char* kAffine = "does not map the scaled image onto the whole image";
  set("letterbox", 1280, 720, 1024, 1024, 576, 0);
  gst_structure_set(meta, "preproc_affine_m00", G_TYPE_DOUBLE, 1.25, "preproc_affine_m01",
                    G_TYPE_DOUBLE, 0., "preproc_affine_m02", G_TYPE_DOUBLE, 0.,
                    "preproc_affine_m10", G_TYPE_DOUBLE, 0., "preproc_affine_m11", G_TYPE_DOUBLE,
                    1.25, "preproc_affine_m12", G_TYPE_DOUBLE, 0., nullptr);
  require(take(resolve(executor.preparer(), with_prompt)) == base,
          "a whole-image legacy affine must keep the prepared plan");
  gst_structure_set(meta, "preproc_affine_m02", G_TYPE_DOUBLE, 160., nullptr);
  require(rejects(legacy.get(), kAffine), "a legacy affine of a cropped region must be rejected");
  set("letterbox", 1280, 720, 1024, 1024, 576, 0);
  gst_structure_set(meta, "preproc_affine_scale_x", G_TYPE_DOUBLE, 1.0, "preproc_affine_scale_y",
                    G_TYPE_DOUBLE, 1.0, "preproc_affine_offset_x", G_TYPE_DOUBLE, 0.,
                    "preproc_affine_offset_y", G_TYPE_DOUBLE, 0., nullptr);
  require(rejects(legacy.get(), kAffine), "legacy affine aliases must be checked too");
  // Compact producer facts are checked the same way.
  const auto compact = [](std::array<std::int64_t, 4> placed, double crop_x = 0.) {
    Buffer buffer(gst_buffer_new_allocate(nullptr, 4, nullptr));
    std::string error;
    require(
        simaai::gst::tensor_buffer_attach_publication(buffer.get(), compact_publication(), &error),
        "attach compact publication failed");
    simaai::gst::TensorBufferPreprocessFacts facts;
    facts.original_width = 1280;
    facts.original_height = 720;
    facts.resized_width = facts.resized_height = 1024;
    facts.primary = {0, 0, 1280, 720};
    facts.placed = placed;
    facts.affine = {1.25, 0.,   crop_x - 1.25 * double(placed[0]),
                    0.,   1.25, -1.25 * double(placed[1])};
    facts.source_stride = 3840;
    facts.resize_mode = "letterbox";
    facts.color_in = facts.color_out = "RGB";
    simaai::gst::TensorBufferFrameMetadata frame;
    frame.preprocess = simaai::gst::tensor_buffer_freeze_preprocess_facts(facts, &error);
    require(bool(frame.preprocess), "compact facts freeze failed");
    require(
        simaai::gst::tensor_buffer_attach_frame_metadata(buffer.get(), std::move(frame), &error),
        "compact facts attach failed");
    return buffer;
  };
  auto top_left = compact({0, 0, 1024, 576});
  const std::array<GstBuffer*, 1> admitted{top_left.get()};
  require(take(resolve(executor.preparer(), admitted)) == base,
          "compact facts of the admitted preprocessing must keep the prepared plan");
  auto centered = compact({0, 224, 1024, 576});
  require(rejects(centered.get(), kPreprocessing), "a compact centered letterbox must be rejected");
  auto composed = compact({0, 0, 1024, 576}, 160.);
  require(rejects(composed.get(), kAffine), "a compact composed crop must be rejected");
}
// #1507: an ordinary detection profile decodes each frame with the affine its preprocess
// published (an ROI, a stretch, a whole frame); legacy ingress that publishes dimensions only keeps
// the recipe's fallback. A profile whose kernel accepts no transform refuses ROI and crop frames.
void set_published_frame(GstStructure* meta, const char* mode, double scale_x, double scale_y,
                         double offset_x, double offset_y, bool affine = true) {
  gst_structure_remove_all_fields(meta);
  gst_structure_set(meta, "preproc_original_width", G_TYPE_INT, 1280, "preproc_original_height",
                    G_TYPE_INT, 720, "preproc_resized_width", G_TYPE_INT, 640,
                    "preproc_resized_height", G_TYPE_INT, 640, "preproc_scaled_width", G_TYPE_INT,
                    640, "preproc_scaled_height", G_TYPE_INT, 640, "preproc_pad_left", G_TYPE_INT,
                    0, "preproc_pad_right", G_TYPE_INT, 0, "preproc_pad_top", G_TYPE_INT, 0,
                    "preproc_pad_bottom", G_TYPE_INT, 0, "preproc_resize_mode", G_TYPE_STRING, mode,
                    nullptr);
  if (affine)
    gst_structure_set(meta, "preproc_affine_m00", G_TYPE_DOUBLE, scale_x, "preproc_affine_m01",
                      G_TYPE_DOUBLE, 0., "preproc_affine_m02", G_TYPE_DOUBLE, offset_x,
                      "preproc_affine_m10", G_TYPE_DOUBLE, 0., "preproc_affine_m11", G_TYPE_DOUBLE,
                      scale_y, "preproc_affine_m12", G_TYPE_DOUBLE, offset_y, nullptr);
}
void published_frame_geometry() {
  probe::OperationProbe executor;
  const auto checked = executable(true);
  require(!executor.prepare(checked), "real A65 preparation failed");
  const auto base = executor.operation();
  core::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = checked;
  // The cold executable carries a seed transform; no frame ever decodes with it.
  core::BoxDecodeGeometryOptions options{640, 640, 640, 640, simaai::neat::ResizeMode::Stretch};
  options.complete_geometry = options.frame_affine_optional = true;
  options.model_to_original = std::array<float, 4>{1.f, 1.f, 0.f, 0.f};
  core::attach_boxdecode_geometry_resolver(recipe, 0, options, true, true);
  auto resolve = recipe.resolver_factories.front().instantiate();
  const gchar* tags[] = {nullptr};
  if (!gst_meta_get_info("GstSimaMeta"))
    require(gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr),
            "metadata registration failed");
  Buffer frame(gst_buffer_new());
  auto* meta =
      gst_custom_meta_get_structure(gst_buffer_add_custom_meta(frame.get(), "GstSimaMeta"));
  const std::array<GstBuffer*, 1> one{frame.get()};
  // The issue's ROI stretched into the canvas, and the same size at another offset.
  set_published_frame(meta, "stretch", 1., .5625, 400., 200.);
  const auto roi = take(resolve(executor.preparer(), one));
  require(roi != base && take(resolve(executor.preparer(), one)) == roi,
          "an ROI frame did not select (and cache) its own affine");
  set_published_frame(meta, "stretch", 1., .5625, 440., 200.);
  const auto moved = take(resolve(executor.preparer(), one));
  require(moved != roi && moved != base, "a second ROI reused the first ROI's affine");
  // The whole frame stretched: its published affine is the recipe's fallback.
  set_published_frame(meta, "stretch", 2., 1.125, 0., 0.);
  const auto whole = take(resolve(executor.preparer(), one));
  require(whole != roi, "a whole frame reused an ROI's affine");
  // Legacy ingress: dimensions and mode, no affine. It decodes with the fallback, not an error
  // and not the seed.
  set_published_frame(meta, "stretch", 0., 0., 0., 0., false);
  const auto legacy = take(resolve(executor.preparer(), one));
  require(legacy != roi && legacy != base,
          "dimensions-only ingress was given an affine, or was rejected");
  // ROI lists: one slot is one frame; several slots carry one affine each.
  set_published_frame(meta, "stretch", 1., .5625, 400., 200.);
  gst_structure_set(meta, "preproc_roi_list_enable", G_TYPE_BOOLEAN, TRUE, nullptr);
  require(std::holds_alternative<rt::Error>(resolve(executor.preparer(), one)),
          "an ROI list of unknown capacity was decoded with one affine");
  gst_structure_set(meta, "preproc_roi_capacity", G_TYPE_INT, 4, nullptr);
  require(std::holds_alternative<rt::Error>(resolve(executor.preparer(), one)),
          "an ROI list of four slots was decoded with one affine");
  gst_structure_set(meta, "preproc_roi_capacity", G_TYPE_INT, 1, nullptr);
  require(take(resolve(executor.preparer(), one)) == roi, "a one-slot ROI list lost its affine");
  // A published affine that is not axis-aligned is not a legacy frame.
  set_published_frame(meta, "stretch", 1., .5625, 400., 200.);
  gst_structure_set(meta, "preproc_affine_m01", G_TYPE_DOUBLE, .1, nullptr);
  require(std::holds_alternative<rt::Error>(resolve(executor.preparer(), one)),
          "a sheared affine was accepted");
}
void full_frame_only_geometry() {
  probe::OperationProbe executor;
  const auto checked = executable(true);
  require(!executor.prepare(checked), "real A65 preparation failed");
  const auto base = executor.operation();
  core::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = checked;
  core::BoxDecodeGeometryOptions options{640, 640, 640, 640, simaai::neat::ResizeMode::Letterbox};
  options.full_frame_only = true;
  options.profile = "yolov8-seg";
  core::attach_boxdecode_geometry_resolver(recipe, 0, options, false, false);
  require(recipe.resolver_factories.size() == 1,
          "a profile that accepts no transform needs a per-frame guard");
  auto resolve = recipe.resolver_factories.front().instantiate();
  const gchar* tags[] = {nullptr};
  if (!gst_meta_get_info("GstSimaMeta"))
    require(gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr),
            "metadata registration failed");
  Buffer frame(gst_buffer_new());
  auto* meta =
      gst_custom_meta_get_structure(gst_buffer_add_custom_meta(frame.get(), "GstSimaMeta"));
  const std::array<GstBuffer*, 1> one{frame.get()};
  const auto refused = [&](const char* message) {
    auto result = resolve(executor.preparer(), one);
    const auto* error = std::get_if<rt::Error>(&result);
    return error && error->message.find("yolov8-seg") != std::string::npos &&
           error->message.find(message) != std::string::npos;
  };
  // Whole frames keep decoding: a centered letterbox (pad 140, scale 2), a placed extent rounded by
  // a pixel, and legacy ingress.
  set_published_frame(meta, "letterbox", 2., 1.125, 0., 0.);
  require(take(resolve(executor.preparer(), one)) == base, "a whole frame was refused");
  set_published_frame(meta, "letterbox", 2., 2., 0., -280.);
  gst_structure_set(meta, "preproc_scaled_height", G_TYPE_INT, 360, "preproc_pad_top", G_TYPE_INT,
                    140, "preproc_pad_bottom", G_TYPE_INT, 140, nullptr);
  require(take(resolve(executor.preparer(), one)) == base, "a whole letterboxed frame was refused");
  // Preprocess rounds the placed extent; a pixel of drift is still the whole frame.
  gst_structure_set(meta, "preproc_affine_m12", G_TYPE_DOUBLE, -282., nullptr);
  require(take(resolve(executor.preparer(), one)) == base, "rounding of the placed extent refused");
  set_published_frame(meta, "letterbox", 0., 0., 0., 0., false);
  require(take(resolve(executor.preparer(), one)) == base, "dimensions-only ingress was refused");
  // An ROI, a crop and an ROI list decoded in full-frame coordinates would be wrong.
  set_published_frame(meta, "letterbox", 1., .5625, 400., 200.);
  require(refused("ROI, a crop or a window"), "an ROI frame decoded in full-frame coordinates");
  set_published_frame(meta, "letterbox", 2., 2., 0., -280.);
  gst_structure_set(meta, "preproc_scaled_height", G_TYPE_INT, 360, "preproc_pad_top", G_TYPE_INT,
                    140, "preproc_pad_bottom", G_TYPE_INT, 140, "preproc_affine_m12", G_TYPE_DOUBLE,
                    -284., nullptr);
  require(refused("ROI, a crop or a window"), "a window of the source decoded as a whole frame");
  set_published_frame(meta, "crop", 2., 1.125, 0., 0.);
  require(refused("ROI, a crop or a window"), "a crop frame decoded in full-frame coordinates");
  set_published_frame(meta, "letterbox", 2., 1.125, 0., 0.);
  gst_structure_set(meta, "preproc_roi_list_enable", G_TYPE_BOOLEAN, TRUE, "preproc_roi_capacity",
                    G_TYPE_INT, 4, nullptr);
  require(refused("ROI, a crop or a window"), "an ROI list decoded in full-frame coordinates");
}
} // namespace
int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  try {
    letterbox_placement();
    std::cout << "BoxDecode letterbox placement passed\n";
    sam_geometry();
    std::cout << "BoxDecode Segment Anything geometry passed\n";
    sam_fixed_image();
    std::cout << "BoxDecode Segment Anything fixed image guard passed\n";
    geometry();
    geometry(true);
    std::cout << "BoxDecode complete geometry and retained variant cache passed\n";
    published_frame_geometry();
    std::cout << "BoxDecode published frame affine passed\n";
    full_frame_only_geometry();
    std::cout << "BoxDecode full-frame-only profiles passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

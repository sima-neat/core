#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/sima/stagesemantics/SsdDecodeContract.h"

#include "pipeline/internal/TensorBufferEnvelope.h"
#include "nodes/sima/internal/LetterboxGeometry.h"

#include <gst/gst.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace simaai::neat::pipeline_internal::sima {
namespace {
// Cold Segment Anything geometry from the nearest Preproc policy, for graphs
// that build before a frame arrives (an EfficientSAM3 concept image model owns its Preproc).
std::optional<PreprocessRuntimeMeta> sam_policy_facts(const ProcessCvuStagePayload* upstream) {
  if (!upstream)
    return std::nullopt;
  namespace rt = ::simaai::scheduler::client;
  using Adapter = kernel_lowering::PreprocFrameAdapter;
  auto policy = *upstream;
  if (policy.resolved_exec_backend.empty())
    policy.resolved_exec_backend = "EVXX";
  auto checked = Adapter::check_policy(policy);
  if (const auto* error = std::get_if<rt::Error>(&checked))
    throw std::invalid_argument("SimaBoxDecode: upstream preprocessing: " + error->message);
  auto frozen = Adapter::checked_initial_preprocess_facts(
      std::get<std::shared_ptr<const Adapter::CheckedPolicy>>(std::move(checked)));
  if (const auto* error = std::get_if<rt::Error>(&frozen))
    throw std::invalid_argument("SimaBoxDecode: upstream preprocessing facts: " + error->message);
  const auto owner = std::get<::simaai::gst::TensorBufferPreprocessFactsOwner>(std::move(frozen));
  const auto* facts = ::simaai::gst::tensor_buffer_preprocess_facts(owner);
  if (!facts || facts->roi_list || facts->input_batch != 1 || facts->capacity != 1)
    throw std::invalid_argument("SimaBoxDecode: Segment Anything requires one upstream image");
  PreprocessRuntimeMeta meta;
  meta.original_width = static_cast<int>(facts->original_width);
  meta.original_height = static_cast<int>(facts->original_height);
  meta.resized_width = static_cast<int>(facts->resized_width);
  meta.resized_height = static_cast<int>(facts->resized_height);
  meta.resize_mode = facts->resize_mode;
  // Placement {x, y, w, h} of the scaled image on the canvas.
  meta.scaled_width = static_cast<int>(facts->placed[2]);
  meta.scaled_height = static_cast<int>(facts->placed[3]);
  meta.pad_left = static_cast<int>(facts->placed[0]);
  meta.pad_top = static_cast<int>(facts->placed[1]);
  meta.pad_right = static_cast<int>(facts->resized_width - facts->placed[0] - facts->placed[2]);
  meta.pad_bottom = static_cast<int>(facts->resized_height - facts->placed[1] - facts->placed[3]);
  meta.affine_m00 = facts->affine[0];
  meta.affine_m01 = facts->affine[1];
  meta.affine_m02 = facts->affine[2];
  meta.affine_m10 = facts->affine[3];
  meta.affine_m11 = facts->affine[4];
  meta.affine_m12 = facts->affine[5];
  return meta;
}

// A SAM mask has the original image extent, so geometry is fixed per prepared
// plan. It comes from the decoder inputs' published preprocess facts (the
// encoder's top-left letterbox or stretch), or the nearest Preproc policy,
// never from tensor shapes or Model options. The model size is the canvas.
CompiledKernelContract compile_sam_kernel(const CompiledBoxDecodeContract& admitted,
                                          std::optional<PreprocessRuntimeMeta> observed,
                                          const ProcessCvuStagePayload* upstream,
                                          BoxDecodeGeometryOptions geometry) {
  const std::string family = box_decode_type_sam_model_name(admitted.payload.decode_type);
  if (!observed)
    observed = sam_policy_facts(upstream);
  // With neither input facts nor an upstream Preproc (Model::build() or
  // execution_plan() without inputs), nothing describes the image the masks are
  // for, so name the calls that take inputs.
  if (!observed)
    throw std::invalid_argument(
        "SimaBoxDecode: " + family +
        " requires semantic.preprocess geometry on its input tensors, so its Model builds only "
        "from inputs: pass the encoder output tensors (for example through .cvu()) to run(), "
        "build(inputs) or execution_plan(inputs) so the image's preprocess facts reach the "
        "decoder");
  geometry = sam_boxdecode_geometry(admitted.payload.decode_type, *observed, geometry);
  auto native = prepare_boxdecode_kernel(admitted, geometry);
  auto recipe = std::make_shared<KernelStageRecipe>(*native.compiled.recipe);
  // No image-size variant exists: a different frame size or preprocessing (resize
  // mode, canvas, placement) needs a new runner, so every frame is checked against
  // the facts admitted here.
  const int scaled_width =
      observed->scaled_width > 0 ? observed->scaled_width : observed->resized_width;
  const int scaled_height =
      observed->scaled_height > 0 ? observed->scaled_height : observed->resized_height;
  attach_boxdecode_fixed_image_guard(
      *recipe, native.decode_member,
      {geometry.original_width,
       geometry.original_height,
       observed->resize_mode,
       geometry.model_width,
       geometry.model_height,
       {observed->pad_left, observed->pad_top, scaled_width, scaled_height}});
  native.compiled.recipe = std::move(recipe);
  return std::move(native.compiled);
}
} // namespace

BoxDecodeGeometryOptions sam_boxdecode_geometry(BoxDecodeType type,
                                                const PreprocessRuntimeMeta& facts,
                                                BoxDecodeGeometryOptions geometry) {
  const std::string family = box_decode_type_sam_model_name(type);
  // Each model was trained on one resize: MobileSAM a top-left letterbox, the
  // others a stretch. The encoder's Model (an EfficientSAM3 concept model's own)
  // preprocess must reproduce it.
  const ResizeMode mode = sam_required_resize(type);
  const bool matches =
      !facts.has_roi_list() &&
      (mode == ResizeMode::Stretch
           ? facts.resize_mode == "stretch"
           : facts.resize_mode == "letterbox" && facts.pad_left == 0 && facts.pad_top == 0);
  if (!matches)
    throw std::invalid_argument(
        "SimaBoxDecode: " + family + " needs the image preprocessed with " +
        (mode == ResizeMode::Letterbox ? "resize.mode=Letterbox and resize.placement=TopLeft"
                                       : "resize.mode=Stretch") +
        " on the Model that preprocesses it (the encoder" +
        (type == BoxDecodeType::EfficientSam3 ? ", or the EfficientSAM3 concept image model" : "") +
        "); its preprocess facts report resize_mode '" + facts.resize_mode + "', pad_left " +
        std::to_string(facts.pad_left) + ", pad_top " + std::to_string(facts.pad_top));
  // The published facts are authoritative. A decoder Model's nominal resize
  // option (Letterbox by default, even without an image input) is not.
  if (geometry.model_to_original)
    throw std::invalid_argument("SimaBoxDecode: " + family +
                                " derives its geometry from the published preprocess facts; an "
                                "explicit model-to-original transform is not admitted");
  const auto width = facts.scaled_width > 0 ? facts.scaled_width : facts.resized_width;
  const auto height = facts.scaled_height > 0 ? facts.scaled_height : facts.resized_height;
  const std::int64_t canvas_width = std::int64_t(width) + facts.pad_left + facts.pad_right;
  const std::int64_t canvas_height = std::int64_t(height) + facts.pad_top + facts.pad_bottom;
  if (canvas_width < 1 || canvas_width > 1048576 || canvas_height < 1 || canvas_height > 1048576)
    throw std::invalid_argument("SimaBoxDecode: " + family + " canvas geometry is invalid");
  if ((geometry.model_width > 0 && geometry.model_width != canvas_width) ||
      (geometry.model_height > 0 && geometry.model_height != canvas_height))
    throw std::invalid_argument("SimaBoxDecode: " + family + " model size differs from the " +
                                std::to_string(canvas_width) + "x" + std::to_string(canvas_height) +
                                " canvas its inputs describe");
  if (facts.original_width < 1 || facts.original_width > 1048576 || facts.original_height < 1 ||
      facts.original_height > 1048576)
    throw std::invalid_argument("SimaBoxDecode: " + family +
                                " original image dimensions must be 1..1048576");
  if ((geometry.original_width > 0 || geometry.original_height > 0) &&
      (geometry.original_width != facts.original_width ||
       geometry.original_height != facts.original_height))
    throw std::invalid_argument("SimaBoxDecode: " + family +
                                " original image size conflicts with the input geometry");
  const auto size = [](std::int64_t w, std::int64_t h) {
    return std::to_string(w) + "x" + std::to_string(h);
  };
  // The kernel derives the scaled extent itself: a stretch fills the canvas, and a
  // top-left letterbox fills a square canvas along the longest side and rounds the
  // other side to the nearest pixel, as SAM does (a standalone Preproc rounds upward).
  std::int64_t expected_width = canvas_width, expected_height = canvas_height;
  if (mode == ResizeMode::Letterbox) {
    if (canvas_width != canvas_height)
      throw std::invalid_argument("SimaBoxDecode: " + family +
                                  " letterboxes onto a square canvas; its inputs describe " +
                                  size(canvas_width, canvas_height));
    using ::simaai::neat::internal::nearest_letterbox_extent;
    const std::int64_t longest = std::max(facts.original_width, facts.original_height);
    expected_width = nearest_letterbox_extent(facts.original_width, canvas_width, longest);
    expected_height = nearest_letterbox_extent(facts.original_height, canvas_width, longest);
  }
  if (width != expected_width || height != expected_height)
    throw std::invalid_argument(
        "SimaBoxDecode: " + family + " masks need the " +
        size(facts.original_width, facts.original_height) + " image scaled to " +
        size(expected_width, expected_height) + " on its " + size(canvas_width, canvas_height) +
        " canvas; its preprocess facts report " + size(width, height) +
        (mode == ResizeMode::Letterbox
             ? ". Letterbox it with resize.placement=TopLeft on the Model that preprocesses it, "
               "which rounds as Segment Anything does"
             : ""));
  if (!boxdecode_affine_covers_image({facts.affine_m00, facts.affine_m01, facts.affine_m02,
                                      facts.affine_m10, facts.affine_m11, facts.affine_m12},
                                     facts.original_width, facts.original_height,
                                     {facts.pad_left, facts.pad_top, width, height}))
    throw std::invalid_argument(
        "SimaBoxDecode: " + family +
        " masks cover the whole image resized once, so the preprocess affine must map the " +
        size(width, height) + " scaled image onto the whole " +
        size(facts.original_width, facts.original_height) + " image (affine_m00 = " +
        std::to_string(facts.original_width) + "/" + std::to_string(width) +
        ", affine_m11 = " + std::to_string(facts.original_height) + "/" + std::to_string(height) +
        ", no offset or shear), as a Model's preprocessing publishes; a crop or ROI before the "
        "resize, or facts without that affine, would place the masks wrongly");
  geometry.model_width = static_cast<int>(canvas_width);
  geometry.model_height = static_cast<int>(canvas_height);
  geometry.original_width = facts.original_width;
  geometry.original_height = facts.original_height;
  geometry.resize = mode;
  geometry.complete_geometry = geometry.uniform_transform_required = false;
  geometry.require_frame_transform = false;
  return geometry;
}

bool boxdecode_letterbox_is_off_center(const PreprocessRuntimeMeta* observed,
                                       const ProcessCvuStagePayload* upstream_preprocess) {
  if (observed && !observed->resize_mode.empty()) {
    // The adapter centers with floor((canvas - scaled) / 2); producers may round
    // the odd pixel either way, so only a larger imbalance is a placement.
    const auto imbalance = [](int a, int b) { return std::abs(std::int64_t(a) - b) > 1; };
    return observed->resize_mode == "letterbox" &&
           (imbalance(observed->pad_left, observed->pad_right) ||
            imbalance(observed->pad_top, observed->pad_bottom));
  }
  if (!upstream_preprocess || upstream_preprocess->aspect_ratio != 1)
    return false;
  std::string padding = upstream_preprocess->padding_type;
  std::transform(padding.begin(), padding.end(), padding.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return padding == "TOPLEFT" || padding == "TOPRIGHT" || padding == "BOTTOMLEFT" ||
         padding == "BOTTOMRIGHT";
}

namespace {
// `kernel_transform` tries the preprocess transform on the profiles that only newer kernels
// accept (pose, segmentation, 91-class multiclass SSD).
CompiledKernelContract compile_boxdecode_kernel_with(const CompiledBoxDecodeContract& compiled,
                                                     const ContractCompileInput& input,
                                                     BoxDecodeGeometryOptions geometry,
                                                     std::optional<ResizeMode> resize_mode,
                                                     bool kernel_transform) {
  namespace si = ::simaai::neat::pipeline_internal::sima;
  // The old plugin resolved source carriers dynamically from metadata.
  // Runner requires their allocation extents once, before admitting work.
  // Import actual producer/storage contracts; tensor shapes cannot establish
  // the capacity of a shared carrier or identify its GstMemory slot.
  auto admitted = compiled;
  auto& runtime = admitted.runtime_contract;
  const auto* upstream_stage = last_effective_child_stage(input.immediate_upstream);
  const auto* upstream = compiled_runtime_contract_from_stage(upstream_stage);
  const auto upstream_id = upstream_stage && !upstream_stage->logical_stage_id.empty()
                               ? upstream_stage->logical_stage_id
                           : upstream_stage ? upstream_stage->element_name
                                            : std::string();
  if (upstream && upstream_id.empty())
    throw std::invalid_argument("SimaBoxDecode: upstream carrier has no stage identity");
  TensorBufferView ingress;
  std::shared_ptr<void> ingress_holder;
  if (!upstream && input.ingress.ingress_sample) {
    // Use the same envelope as graph ingress. Owned outputs may retain producer
    // route offsets after their payloads have been copied into independent
    // storage. Those historical offsets are not offsets in the new carrier.
    // This is cold admission only; no new per-frame conversion is introduced.
    std::string error;
    ingress_holder = sample_to_gst_envelope_holder(*input.ingress.ingress_sample, &error);
    if (!ingress_holder || !tensor_buffer_descriptor_from_sample(
                               static_cast<GstSample*>(ingress_holder.get()), &ingress, &error))
      throw std::invalid_argument("SimaBoxDecode: cannot resolve ingress carriers: " + error);
  }
  runtime.physical_inputs.clear();
  for (std::size_t input_index = 0; input_index < runtime.logical_inputs.size(); ++input_index) {
    auto& logical = runtime.logical_inputs[input_index];
    auto binding = std::find_if(runtime.input_bindings.begin(), runtime.input_bindings.end(),
                                [&](const auto& value) {
                                  return value.local_logical_input_index == logical.logical_index;
                                });
    if (binding == runtime.input_bindings.end())
      throw std::invalid_argument("SimaBoxDecode: missing authored head input binding");
    si::PhysicalBufferStaticSpec physical;
    if (upstream) {
      if (!binding->src_stage_id.empty() &&
          binding->src_stage_id != upstream_stage->logical_stage_id &&
          binding->src_stage_id != upstream_stage->element_name)
        throw std::invalid_argument("SimaBoxDecode: conflicting upstream carrier identity");
      const auto source =
          std::find_if(upstream->physical_outputs.begin(), upstream->physical_outputs.end(),
                       [&](const auto& value) {
                         return value.physical_index == binding->src_physical_output_index;
                       });
      if (source == upstream->physical_outputs.end())
        throw std::invalid_argument(
            "SimaBoxDecode: head has no authoritative upstream physical carrier");
      if (std::count_if(upstream->physical_outputs.begin(), upstream->physical_outputs.end(),
                        [&](const auto& value) {
                          return value.physical_index == binding->src_physical_output_index;
                        }) != 1)
        throw std::invalid_argument("SimaBoxDecode: ambiguous upstream physical carrier");
      physical = *source;
      physical.physical_index = logical.physical_index;
      // The head can retain an authored Slice alias that the producer does not
      // publish. Preserve its numerical view, but bind the imported carrier to
      // the actual compiled producer. Placement can then project this physical
      // port into its final allocation and offset instead of treating the port
      // ordinal as an external GstMemory index. Compile-stage indices are not
      // final manifest indices, so retain the stable producer identity only.
      binding->src_stage_id = upstream_id;
      binding->src_stage_index = -1;
      binding->src_physical_size_bytes = source->size_bytes;
      // Bind the complete carrier; the native port applies the logical
      // tensor offset exactly once, including packed multi-head carriers.
      binding->src_physical_byte_offset = 0;
    } else if (ingress_holder) {
      const auto source =
          std::find_if(ingress.tensors.begin(), ingress.tensors.end(), [&](const auto& tensor) {
            return tensor.logical_index == binding->src_logical_output_index;
          });
      if (source == ingress.tensors.end() || source->memory_index < 0 ||
          static_cast<guint>(source->memory_index) >= gst_buffer_n_memory(ingress.buffer))
        throw std::invalid_argument("SimaBoxDecode: ingress head has no physical carrier");
      const int memory = source->memory_index;
      const auto carrier_bytes =
          gst_memory_get_sizes(gst_buffer_peek_memory(ingress.buffer, memory), nullptr, nullptr);
      logical.physical_index = memory;
      logical.byte_offset = source->byte_offset;
      if (compiled.payload.tensor_storage_kind.at(input_index) ==
          static_cast<int>(BoxDecodeSourceStorageKind::DenseHwcPhysical)) {
        // Envelope materialization may compact a strided CPU view. Bind the
        // actual carrier's descriptor, not the pre-materialization strides.
        const auto* layout = source->layout == SIMA_TENSOR_SET_LAYOUT_CHW_V1  ? "CHW"
                             : source->layout == SIMA_TENSOR_SET_LAYOUT_HW_V1 ? "HW"
                                                                              : "HWC";
        const auto linear =
            boxdecode_linear_storage(source->shape, source->stride_bytes, layout,
                                     si::specbuilders::dtype_size_bytes_from_token(logical.dtype));
        if (!linear || source->byte_offset < 0 ||
            static_cast<std::uint64_t>(source->byte_offset) > carrier_bytes ||
            linear->span_bytes > carrier_bytes - static_cast<std::uint64_t>(source->byte_offset))
          throw std::invalid_argument("SimaBoxDecode: ingress linear storage geometry exceeds "
                                      "its physical carrier");
        logical.stride_bytes = linear->stride_bytes;
      }
      physical = si::specbuilders::build_physical_buffer_static_spec(
          memory, memory, carrier_bytes, si::DeviceKind::Cpu, logical.segment_name);
      binding->src_stage_index = -1;
      binding->src_stage_id.clear();
      binding->src_logical_output_index = -1;
      binding->src_output_slot = -1;
      binding->src_physical_output_index = -1;
      binding->src_physical_size_bytes = 0;
      binding->src_physical_byte_offset = 0;
    } else {
      const auto source =
          std::find_if(compiled.runtime_contract.physical_inputs.begin(),
                       compiled.runtime_contract.physical_inputs.end(), [&](const auto& value) {
                         return value.physical_index == logical.physical_index;
                       });
      if (source == compiled.runtime_contract.physical_inputs.end())
        throw std::invalid_argument("SimaBoxDecode: typed head contract requires authoritative "
                                    "physical input carriers");
      physical = *source;
    }
    if (!physical.size_bytes || logical.byte_offset < 0 ||
        static_cast<std::uint64_t>(logical.byte_offset) >= physical.size_bytes)
      throw std::invalid_argument("SimaBoxDecode: head offset exceeds its physical carrier");
    logical.size_bytes = physical.size_bytes - static_cast<std::uint64_t>(logical.byte_offset);
    const auto existing = std::find_if(
        runtime.physical_inputs.begin(), runtime.physical_inputs.end(),
        [&](const auto& value) { return value.physical_index == physical.physical_index; });
    if (existing == runtime.physical_inputs.end())
      runtime.physical_inputs.push_back(std::move(physical));
    else if (existing->size_bytes != physical.size_bytes ||
             existing->source_byte_offset != physical.source_byte_offset ||
             existing->address_source != physical.address_source)
      throw std::invalid_argument("SimaBoxDecode: shared head carrier contracts disagree");
  }
  const bool upstream_crop = input.upstream_preprocess &&
                             input.upstream_preprocess->payload.resize_mode == ResizeMode::Crop;
  // Image frontends compose spatial provenance even when their own local mode
  // is Stretch (for example Cover -> ColorConvert -> Normalization). That
  // producer contract, not the last resize label, requires affine consumption.
  // Do not infer the same authority from legacy dimension-only ingress metadata.
  const bool upstream_image_transform =
      input.upstream_preprocess && input.upstream_preprocess->payload.image_transform;
  const bool crop_requested = resize_mode == ResizeMode::Crop || upstream_crop;
  const bool detection_transform =
      boxdecode_payload_supports_detection_transform(compiled.payload) ||
      (kernel_transform && boxdecode_payload_may_take_transform(compiled.payload));
  geometry.uniform_transform_required = box_decode_type_is_oriented(compiled.payload.decode_type);
  geometry.complete_geometry =
      compiled.payload.decode_type == BoxDecodeType::YoloXSegPose ||
      geometry.uniform_transform_required ||
      (detection_transform &&
       (crop_requested || upstream_image_transform || geometry.model_to_original.has_value()));
  const auto affine = [](const PreprocessRuntimeMeta& meta) {
    return std::array<double, 6>{meta.affine_m00, meta.affine_m01, meta.affine_m02,
                                 meta.affine_m10, meta.affine_m11, meta.affine_m12};
  };
  std::optional<PreprocessRuntimeMeta> observed;
  if (input.ingress.ingress_sample) {
    for (const auto& tensor : tensors_from_sample(*input.ingress.ingress_sample, true)) {
      if (!tensor.semantic.preprocess)
        continue;
      const auto& facts = *tensor.semantic.preprocess;
      if (observed && (observed->original_width != facts.original_width ||
                       observed->original_height != facts.original_height))
        throw std::invalid_argument("SimaBoxDecode: ingress tensors disagree on image geometry");
      if (observed &&
          (geometry.complete_geometry || observed->resize_mode == "crop" ||
           facts.resize_mode == "crop") &&
          (affine(*observed) != affine(facts) || observed->resize_mode != facts.resize_mode ||
           observed->roi_list_enabled != facts.roi_list_enabled))
        throw std::invalid_argument("SimaBoxDecode: ingress tensors disagree on affine geometry");
      observed = facts;
    }
  }
  if (box_decode_type_is_sam(compiled.payload.decode_type))
    return compile_sam_kernel(
        admitted, observed,
        input.upstream_preprocess ? &input.upstream_preprocess->payload : nullptr, geometry);
  if (observed && observed->resize_mode == "crop" && detection_transform)
    geometry.complete_geometry = true;
  // Keep fixed-size decoding static when there is no source of per-frame geometry.
  const bool frame_geometry_source = input.upstream_preprocess || observed ||
                                     geometry.original_width <= 0 || geometry.original_height <= 0;
  geometry.full_frame_only = frame_geometry_source && !detection_transform &&
                             !geometry.complete_geometry &&
                             compiled.payload.decode_type != BoxDecodeType::SuperPoint;
  geometry.profile = box_decode_type_token(compiled.payload.decode_type);
  if (geometry.full_frame_only && (crop_requested || (observed && observed->resize_mode == "crop")))
    throw std::invalid_argument("SimaBoxDecode: " + geometry.profile +
                                " maps boxes over the whole frame; its kernel accepts no "
                                "model-to-original transform, so a Crop resize cannot be mapped "
                                "back to the source image");
  // Profiles map letterboxed boxes with a centered rule. Any other placement
  // (for example ResizeSpec TopLeft) needs the actual model-to-original affine.
  // Placement comes from published pads or the nearest Preproc policy only.
  const bool off_center_letterbox = boxdecode_letterbox_is_off_center(
      observed ? &*observed : nullptr,
      input.upstream_preprocess ? &input.upstream_preprocess->payload : nullptr);
  if (off_center_letterbox && !geometry.complete_geometry &&
      compiled.payload.decode_type != BoxDecodeType::SuperPoint) {
    if (!detection_transform)
      throw std::invalid_argument(
          std::string("SimaBoxDecode: ") + box_decode_type_token(compiled.payload.decode_type) +
          " decodes only a centered letterbox; a non-centered placement requires the explicit "
          "model-to-original transform, which only detection profiles admit");
    geometry.complete_geometry = true;
  }
  if (input.upstream_preprocess &&
      (upstream_crop || (detection_transform && upstream_image_transform) ||
       (!observed &&
        (crop_requested || ((geometry.uniform_transform_required || off_center_letterbox) &&
                            !geometry.model_to_original))))) {
    // Check the nearest publisher's model canvas to prepare a cold seed. This
    // policy does not describe the whole preceding frontend chain: in particular
    // ColorConvert/Normalization can publish a composed affine while locally
    // reporting Stretch. Only current per-frame facts authorize execution; the
    // local seed or older graph-ingress provenance must never replace them.
    // The Crop seed retains its existing ingress composition, still guarded by
    // the same mandatory current-frame resolver below. A non-centered letterbox
    // seeds from the same checked Preproc policy that publishes frame affines.
    namespace rt = ::simaai::scheduler::client;
    using Adapter = kernel_lowering::PreprocFrameAdapter;
    auto policy = input.upstream_preprocess->payload;
    // Match the canonical Preproc lowering default without mutating its typed snapshot.
    if (policy.resolved_exec_backend.empty())
      policy.resolved_exec_backend = "EVXX";
    auto checked = Adapter::check_policy(policy);
    if (const auto* error = std::get_if<rt::Error>(&checked))
      throw std::invalid_argument("SimaBoxDecode: upstream preprocessing: " + error->message);
    std::optional<::simaai::gst::TensorBufferPreprocessFacts> ingress_provenance;
    if (upstream_crop && observed) {
      ingress_provenance.emplace();
      ingress_provenance->original_width = observed->original_width;
      ingress_provenance->original_height = observed->original_height;
      ingress_provenance->affine = affine(*observed);
      ingress_provenance->color_in = observed->color_in;
      ingress_provenance->roi_list = observed->has_roi_list();
    }
    auto frozen = Adapter::checked_initial_preprocess_facts(
        std::get<std::shared_ptr<const Adapter::CheckedPolicy>>(std::move(checked)),
        ingress_provenance ? &*ingress_provenance : nullptr);
    if (const auto* error = std::get_if<rt::Error>(&frozen))
      throw std::invalid_argument("SimaBoxDecode: upstream preprocessing facts: " + error->message);
    const auto owner = std::get<::simaai::gst::TensorBufferPreprocessFactsOwner>(std::move(frozen));
    const auto* facts = ::simaai::gst::tensor_buffer_preprocess_facts(owner);
    if (!facts || facts->roi_list || facts->input_batch != 1 || facts->capacity != 1)
      throw std::invalid_argument("SimaBoxDecode: complete geometry requires one upstream image");
    if ((geometry.model_width > 0 && geometry.model_width != facts->resized_width) ||
        (geometry.model_height > 0 && geometry.model_height != facts->resized_height))
      throw std::invalid_argument(
          "SimaBoxDecode: upstream preprocessing canvas differs from model");
    PreprocessRuntimeMeta meta;
    meta.original_width = facts->original_width;
    meta.original_height = facts->original_height;
    meta.resized_width = facts->resized_width;
    meta.resized_height = facts->resized_height;
    meta.resize_mode = facts->resize_mode;
    meta.affine_m00 = facts->affine[0];
    meta.affine_m01 = facts->affine[1];
    meta.affine_m02 = facts->affine[2];
    meta.affine_m10 = facts->affine[3];
    meta.affine_m11 = facts->affine[4];
    meta.affine_m12 = facts->affine[5];
    observed = std::move(meta);
    geometry.require_frame_transform = true;
  }
  if (frame_geometry_source && detection_transform && !geometry.complete_geometry) {
    geometry.complete_geometry = true;
    geometry.frame_affine_optional = true;
  }
  if (observed && geometry.model_width <= 0 && geometry.model_height <= 0) {
    const auto width =
        observed->scaled_width > 0 ? observed->scaled_width : observed->resized_width;
    const auto height =
        observed->scaled_height > 0 ? observed->scaled_height : observed->resized_height;
    const std::int64_t padded_width =
        observed->resize_mode == "crop"
            ? observed->resized_width
            : std::int64_t(width) + observed->pad_left + observed->pad_right;
    const std::int64_t padded_height =
        observed->resize_mode == "crop"
            ? observed->resized_height
            : std::int64_t(height) + observed->pad_top + observed->pad_bottom;
    if (padded_width < 1 || padded_width > 1048576 || padded_height < 1 || padded_height > 1048576)
      throw std::invalid_argument("SimaBoxDecode: ingress model geometry is invalid");
    geometry.model_width = static_cast<int>(padded_width);
    geometry.model_height = static_cast<int>(padded_height);
  }
  if ((geometry.model_width <= 0 || geometry.model_height <= 0) &&
      box_decode_type_is_ssd_family(compiled.payload.decode_type)) {
    if (const auto* recipe =
            si::stagesemantics::find_ssd_recipe_descriptor(compiled.payload.ssd_recipe_id)) {
      geometry.model_width = recipe->model_width;
      geometry.model_height = recipe->model_height;
    }
  }
  if (geometry.model_width <= 0 || geometry.model_height <= 0)
    throw std::invalid_argument("SimaBoxDecode: model dimensions require explicit options, "
                                "model contract or ingress preprocess metadata");
  const bool dynamic_image = geometry.original_width <= 0 || geometry.original_height <= 0;
  geometry.original_width = dynamic_image
                                ? (observed ? observed->original_width : geometry.model_width)
                                : geometry.original_width;
  geometry.original_height = dynamic_image
                                 ? (observed ? observed->original_height : geometry.model_height)
                                 : geometry.original_height;
  geometry.resize = default_boxdecode_resize_mode(compiled.payload.decode_type);
  if (resize_mode)
    geometry.resize = *resize_mode;
  else if (observed && !observed->resize_mode.empty()) {
    if (observed->resize_mode == "stretch")
      geometry.resize = ResizeMode::Stretch;
    else if (observed->resize_mode == "letterbox")
      geometry.resize = ResizeMode::Letterbox;
    else if ((compiled.payload.decode_type == BoxDecodeType::SuperPoint ||
              geometry.complete_geometry) &&
             observed->resize_mode == "crop")
      geometry.resize = ResizeMode::Crop;
    else
      throw std::invalid_argument("SimaBoxDecode: unsupported preprocess resize geometry");
  }
  if (observed && observed->resize_mode == "crop" && geometry.resize != ResizeMode::Crop)
    throw std::invalid_argument("SimaBoxDecode: explicit resize mode conflicts with producer crop");
  if (geometry.resize == ResizeMode::Crop && observed && observed->resize_mode != "crop")
    throw std::invalid_argument("SimaBoxDecode: crop requires matching producer resize metadata");
  if (geometry.complete_geometry && observed) {
    // Both compact preprocess facts and the public metadata projection carry
    // model-to-original coefficients. Do not invert them a second time.
    const auto values = affine(*observed);
    // An ROI-list buffer with several slots carries one affine per slot; one slot is one frame.
    if ((observed->roi_list_enabled || observed->has_roi_list()) && observed->roi_capacity != 1)
      throw std::invalid_argument("SimaBoxDecode: complete geometry requires one image "
                                  "transform, not an ROI list with several slots; preprocess one "
                                  "ROI per frame (ROI-list capacity 1)");
    const bool unpublished =
        geometry.frame_affine_optional &&
        std::all_of(values.begin(), values.end(), [](double v) { return v == 0.; });
    if (!unpublished && (std::any_of(values.begin(), values.end(),
                                     [](double v) {
                                       return !std::isfinite(v) ||
                                              std::abs(v) > std::numeric_limits<float>::max();
                                     }) ||
                         values[1] != 0 || values[3] != 0 || values[0] <= 0 || values[4] <= 0))
      throw std::invalid_argument("SimaBoxDecode: model-to-original affine must be finite, "
                                  "axis-aligned and have positive scales");
    if (!unpublished) {
      if (geometry.uniform_transform_required && values[0] != values[4])
        throw std::invalid_argument(
            "SimaBoxDecode: oriented boxes require equal authoritative affine scales");
      geometry.model_to_original =
          std::array<float, 4>{static_cast<float>(values[0]), static_cast<float>(values[4]),
                               static_cast<float>(values[2]), static_cast<float>(values[5])};
    }
  }
  // Identity is only for kernel preparation. Frames without an affine use full-frame mapping.
  if (geometry.frame_affine_optional && !geometry.model_to_original)
    geometry.model_to_original = std::array<float, 4>{1.f, 1.f, 0.f, 0.f};
  if (geometry.complete_geometry && geometry.resize == ResizeMode::Crop &&
      !geometry.model_to_original)
    throw std::invalid_argument("SimaBoxDecode: crop geometry requires authoritative ingress "
                                "preprocess model-to-original affine metadata");
  if (geometry.uniform_transform_required && !geometry.model_to_original)
    throw std::invalid_argument("SimaBoxDecode: oriented boxes require authoritative ingress "
                                "preprocess affine metadata (explicit identity for model pixels)");
  auto native = si::prepare_boxdecode_kernel(admitted, geometry);
  if (compiled.payload.decode_type != BoxDecodeType::SuperPoint) {
    auto recipe = std::make_shared<si::KernelStageRecipe>(*native.compiled.recipe);
    const bool require_mode =
        geometry.require_frame_transform ||
        std::find(compiled.runtime_contract.required_preprocess_meta_fields.begin(),
                  compiled.runtime_contract.required_preprocess_meta_fields.end(),
                  "preproc_resize_mode") !=
            compiled.runtime_contract.required_preprocess_meta_fields.end();
    si::attach_boxdecode_geometry_resolver(*recipe, native.decode_member, geometry, dynamic_image,
                                           require_mode);
    native.compiled.recipe = std::move(recipe);
  }
  return std::move(native.compiled);
}
} // namespace

CompiledKernelContract compile_boxdecode_kernel(const CompiledBoxDecodeContract& compiled,
                                                const ContractCompileInput& input,
                                                BoxDecodeGeometryOptions geometry,
                                                std::optional<ResizeMode> resize_mode) {
  try {
    return compile_boxdecode_kernel_with(compiled, input, geometry, resize_mode, true);
  } catch (const BoxDecodeTransformUnavailable&) {
    // The installed kernel accepts no transform for this profile: whole frames only.
    return compile_boxdecode_kernel_with(compiled, input, geometry, resize_mode, false);
  }
}
} // namespace simaai::neat::pipeline_internal::sima

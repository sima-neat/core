#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <simaai/scheduler/testing/probe.h>

#include <gstsimaaitensorbuffer_meta.h>

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
namespace k = simaai::scheduler::client;
namespace probe = simaai::scheduler::testing;
namespace rt = simaai::scheduler::client;
namespace core = simaai::neat::pipeline_internal::sima;
struct BufferFree { void operator()(GstBuffer* p) const { if (p) gst_buffer_unref(p); } };
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
k::KernelExecutable executable() {
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
  return take(
      k::check_kernel({"boxdecode", k::Target::A65, inputs, outputs, parameters}));
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
void geometry() {
  probe::OperationProbe executor;
  const auto checked = executable();
  require(!executor.prepare(checked), "real A65 preparation failed");
  const auto base = executor.operation();
  core::KernelStageRecipe recipe;
  recipe.members.resize(1);
  recipe.members.front().executable = checked;
  core::BoxDecodeGeometryOptions options{640, 640, 640, 640, simaai::neat::ResizeMode::Letterbox};
  options.complete_geometry = true;
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
  set_geometry(a);
  set_geometry(b);
  const auto first_variant = take(resolve(executor.preparer(), buffers));
  require(first_variant != base && take(resolve(executor.preparer(), buffers)) == first_variant,
          "complete geometry was not cached");
  require(k::same_kernel_contract_resources(first_variant.executable(), checked),
          "complete geometry rebuilt scratch or tensor contract");
  require(take(independent(executor.preparer(), buffers)) != first_variant,
          "independent preparation shared mutable geometry cache");
  set_geometry(a, 12.);
  set_geometry(b, 12.);
  const auto changed = take(resolve(executor.preparer(), buffers));
  require(changed != first_variant && changed != base,
          "same dimensions with changed affine reused stale geometry");
  set_geometry(a);
  set_geometry(b);
  require(take(resolve(executor.preparer(), buffers)) == first_variant,
          "immutable geometry snapshot was mutated");
  for (int failure = 0; failure < 8; ++failure) {
    set_geometry(a);
    set_geometry(b);
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
  set_geometry(a);
  set_geometry(b);
  require(take(resolve(executor.preparer(), buffers)) == first_variant,
          "invalid frame evicted a valid warmed snapshot");
  // Bound cache may evict, but old snapshots remain immutable and retain resources.
  for (int i = 0; i < 6; ++i) {
    set_geometry(a, 30. + i);
    set_geometry(b, 30. + i);
    require(
        k::same_kernel_contract_resources(take(resolve(executor.preparer(), buffers)).executable(), checked),
        "geometry cache eviction changed resource identity");
  }
  set_geometry(a);
  set_geometry(b);
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
  facts.resize_mode = "letterbox";
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
} // namespace
int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  try {
    geometry();
    std::cout << "BoxDecode complete geometry and retained variant cache passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

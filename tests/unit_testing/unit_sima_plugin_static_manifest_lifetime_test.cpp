#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <memory>
#include <string>

namespace {
namespace si = simaai::neat::pipeline_internal::sima;
using Context = std::unique_ptr<GstContext, decltype(&gst_context_unref)>;
using Element = std::unique_ptr<GstElement, decltype(&gst_object_unref)>;

const SimaPluginStaticManifestAccessor* accessor(const Context& context) {
  const auto* handle = sima_plugin_manifest_context_handle(context.get());
  require(handle && handle->accessor, "retained metadata context must expose an accessor");
  return handle->accessor(handle->user_data);
}

void inspect(const Context& context) {
  const auto* api = accessor(context);
  require(api->stage_count(api->user_data) == 5, "all canonical and legacy stages must survive");
  require(g_str_equal(api->session_id(api->user_data),
                      "session-long-enough-to-exercise-owned-string-storage"),
          "session identity must outlive the source manifest");
  require(g_str_equal(api->model_id(api->user_data), "metadata-owner-model"),
          "model identity must outlive the source manifest");
  for (guint i = 0; i < 5; ++i) {
    const auto* stage = api->stage_by_index(api->user_data, i);
    require(stage && api->stage_by_element_name(api->user_data, stage->element_name) == stage &&
                api->stage_by_logical_stage_id(api->user_data, stage->logical_stage_id) == stage,
            "all accessor lookup paths must preserve the same stable stage");
    require(stage->logical_inputs_len == 1 && stage->logical_inputs[0].shape_len == 3 &&
                stage->logical_inputs[0].shape[1] == 5 &&
                stage->logical_inputs[0].stride_bytes[0] == 60 &&
                g_str_equal(stage->logical_inputs[0].dtype, "FP32"),
            "input backing values must remain readable");
    require(stage->logical_inputs[0].quant && stage->logical_inputs[0].quant->scales[0] == 0.5 &&
                stage->logical_inputs[0].quant->zero_points[0] == 3,
            "nested quantization arrays must remain readable");
    require(stage->required_preprocess_meta_fields_len == 1 &&
                g_str_equal(stage->required_preprocess_meta_fields[0], "preproc_original_width"),
            "required metadata names must remain readable");
    require(stage->output_order_len == 1 &&
                g_str_equal(stage->output_order[0].segment_name,
                            "segment-long-enough-to-exercise-owned-string-storage"),
            "output-route backing strings must remain readable");
    require(stage->frame_arena_size_bytes == 4096 &&
                stage->frame_arena_role == SIMA_PLUGIN_FRAME_ARENA_ALLOCATE,
            "placement policy must survive metadata projection");
  }
  const auto* canonical = api->stage_by_index(api->user_data, 0);
  const auto* canonical_superpoint = api->stage_by_index(api->user_data, 1);
  const auto* superpoint = api->stage_by_index(api->user_data, 2);
  const auto* mla = api->stage_by_index(api->user_data, 3);
  const auto* legacy_cvu = api->stage_by_index(api->user_data, 4);
  require(canonical->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_NONE &&
              canonical_superpoint->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_NONE,
          "canonical transport tags must not depend on retained execution owners");
  require(superpoint->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE &&
              superpoint->payload.boxdecode.superpoint &&
              superpoint->payload.boxdecode.superpoint->descriptor_dim == 256 &&
              superpoint->payload.boxdecode.superpoint->tensor_roles_len == 2 &&
              g_str_equal(superpoint->payload.boxdecode.superpoint->profile_fingerprint,
                          "profile-fingerprint-long-enough-for-owned-string-storage"),
          "legacy SuperPoint embedded pointer, role table and strings must remain valid");
  require(mla->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSMLA &&
              g_str_equal(mla->payload.processmla.model_path,
                          "model-path-long-enough-for-owned-string-storage.elf"),
          "legacy MLA payload and borrowed model path must remain valid");
  require(legacy_cvu->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSCVU &&
              g_str_equal(legacy_cvu->payload.processcvu.graph_name, "legacy-metadata-preproc") &&
              legacy_cvu->payload.processcvu.channel_mean_len == 3 &&
              legacy_cvu->payload.processcvu.channel_mean[1] == 0.5,
          "legacy CVU payload strings and converted arrays must remain valid");
}

} // namespace

RUN_TEST("unit_sima_plugin_static_manifest_lifetime_test", ([] {
           gst_init(nullptr, nullptr);
           Element pipeline(gst_pipeline_new("metadata-lifetime"), &gst_object_unref);
           Element second_pipeline(gst_pipeline_new("metadata-copy-lifetime"), &gst_object_unref);
           Context retained(nullptr, &gst_context_unref);
           Context second(nullptr, &gst_context_unref);
           std::weak_ptr<const si::KernelStageRecipe> recipe_observer;
           {
             si::SimaPluginStaticManifest source;
             source.session_id = "session-long-enough-to-exercise-owned-string-storage";
             source.model_id = "metadata-owner-model";
             auto recipe = std::make_shared<si::KernelStageRecipe>();
             recipe_observer = recipe;
             for (int i = 0; i < 5; ++i) {
               auto& stage = source.stages.emplace_back();
               stage.element_name = "stage" + std::to_string(i);
               stage.logical_stage_id = "id" + std::to_string(i);
               stage.payload_kind = i == 0 || i == 4 ? si::StagePayloadKind::ProcessCvu
                                    : i == 3         ? si::StagePayloadKind::ProcessMla
                                                     : si::StagePayloadKind::BoxDecode;
               if (i < 2)
                 stage.native_kernel = recipe;
               stage.boxdecode.decode_type = simaai::neat::BoxDecodeType::SuperPoint;
               stage.boxdecode.superpoint.descriptor_dim = 256;
               stage.boxdecode.superpoint.profile_fingerprint =
                   "profile-fingerprint-long-enough-for-owned-string-storage";
               stage.boxdecode.tensor_roles = {si::BoxDecodeTensorRole::DetectorLogits,
                                               si::BoxDecodeTensorRole::DescriptorGrid};
               stage.processmla.model_path = "model-path-long-enough-for-owned-string-storage.elf";
               stage.processcvu.graph_name = "legacy-metadata-preproc";
               stage.processcvu.channel_mean = {0.25, 0.5, 0.75};
               auto& input = stage.logical_inputs.emplace_back();
               input.shape = {1, 5, 3};
               input.stride_bytes = {60, 12, 4};
               input.dtype = "FP32";
               input.quant = si::QuantStaticSpec{};
               input.quant->scales = {0.5};
               input.quant->zero_points = {3};
               stage.required_preprocess_meta_fields = {"preproc_original_width"};
               stage.output_order.emplace_back().segment_name =
                   "segment-long-enough-to-exercise-owned-string-storage";
               stage.frame_arena_size_bytes = 4096;
               stage.frame_arena_role = si::FrameArenaRole::Allocate;
             }
             std::string error;
             require(si::attach_manifest_context(pipeline.get(), source, &error), error);
             auto copied_source = source;
             require(si::attach_manifest_context(second_pipeline.get(), copied_source, &error),
                     error);
             require(source.stages[0].native_kernel == recipe &&
                         source.stages[0].payload_kind == si::StagePayloadKind::ProcessCvu &&
                         source.stages[1].payload_kind == si::StagePayloadKind::BoxDecode,
                     "metadata attachment must not mutate semantic execution inputs");
             retained.reset(
                 gst_element_get_context(pipeline.get(), SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE));
             second.reset(gst_element_get_context(second_pipeline.get(),
                                                  SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE));
             require(retained && second, "both contexts must be retained independently");
             inspect(retained);
             inspect(second);
           }
           require(recipe_observer.expired(),
                   "static contexts must not retain native recipes even while pipelines live");
           pipeline.reset();
           second_pipeline.reset();
           inspect(retained);
           inspect(second);
           // Copy the actual GstContext/boxed handle, not a hand-authored projection.
           Context copied(gst_context_copy(retained.get()), &gst_context_unref);
           retained.reset();
           second.reset();
           inspect(copied);
           require(recipe_observer.expired(),
                   "copied metadata handles cannot regain execution ownership");
         }));

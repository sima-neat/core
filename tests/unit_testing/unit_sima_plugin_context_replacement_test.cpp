#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

namespace {

using namespace simaai::neat::pipeline_internal::sima;

SimaPluginStaticManifest make_manifest(const char* element_name, const char* stage_id, int topk) {
  SimaPluginStaticManifest manifest;
  manifest.session_id = "ctx-replace";
  manifest.model_id = stage_id;

  StageStaticSpec stage;
  stage.element_name = element_name;
  stage.logical_stage_id = stage_id;
  stage.plugin_kind = "neatboxdecode";
  stage.kernel_kind = "boxdecode";
  stage.payload_kind = StagePayloadKind::BoxDecode;
  stage.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  stage.boxdecode.topk = topk;
  manifest.stages.push_back(stage);
  return manifest;
}

} // namespace

RUN_TEST(
    "unit_sima_plugin_context_replacement_test", ([] {
      gst_init(nullptr, nullptr);

      GstElement* pipeline = GST_ELEMENT(gst_pipeline_new("ctx_replace_pipeline"));
      require(pipeline != nullptr, "pipeline must be created");
      GstElement* existing = GST_ELEMENT(gst_bin_new("stage_old"));
      require(existing != nullptr, "existing child should be created");
      require(gst_bin_add(GST_BIN(pipeline), existing),
              "existing child should be addable to the pipeline");

      std::string attach_error;
      require(attach_manifest_context(pipeline, make_manifest("stage_old", "stage_old_id", 10),
                                      &attach_error),
              "first attach_manifest_context should succeed: " + attach_error);
      require(attach_manifest_context(pipeline, make_manifest("stage_new", "stage_new_id", 20),
                                      &attach_error),
              "second attach_manifest_context should succeed: " + attach_error);

      GstContext* pipeline_context =
          gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
      require(pipeline_context != nullptr, "pipeline should expose replacement context");

      SimaPluginManifestLookupStatus status = SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT;
      require(sima_plugin_manifest_context_stage_lookup_typed_checked(
                  pipeline_context, "stage_old_id", "stage_old",
                  SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE, &status) == nullptr,
              "replaced manifest should not resolve the old stage");
      require(status == SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_STAGE_NOT_FOUND,
              "old stage lookup should report stage_not_found");

      const auto* new_stage = sima_plugin_manifest_context_stage_lookup_typed_checked(
          pipeline_context, "stage_new_id", "stage_new", SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE,
          &status);
      require(new_stage != nullptr, "replacement manifest should resolve the new stage");
      require(new_stage->payload.boxdecode.topk == 20, "replacement manifest payload mismatch");

      GstElement* late = GST_ELEMENT(gst_bin_new("stage_new"));
      require(late != nullptr, "late child should be created");
      require(gst_bin_add(GST_BIN(pipeline), late), "late child should be addable to pipeline");
      GstContext* late_context =
          gst_element_get_context(late, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
      require(late_context != nullptr, "late child should inherit replacement context");

      const auto* late_stage = sima_plugin_manifest_context_stage_lookup_typed_checked(
          late_context, "stage_new_id", "stage_new", SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE, &status);
      require(late_stage != nullptr, "late child should resolve replacement stage");
      require(late_stage->payload.boxdecode.topk == 20, "late child payload mismatch");

      gst_context_unref(late_context);
      gst_context_unref(pipeline_context);
      gst_object_unref(pipeline);
    }));

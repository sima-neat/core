#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

namespace {

using namespace simaai::neat::pipeline_internal::sima;

GstContext* make_context_snapshot() {
  GstElement* pipeline = GST_ELEMENT(gst_pipeline_new("ctx_attach_lifetime_pipeline"));
  require(pipeline != nullptr, "pipeline must be created");

  SimaPluginStaticManifest manifest;
  manifest.session_id = "ctx-attach-lifetime";
  manifest.model_id = "model-lifetime";
  StageStaticSpec stage;
  stage.element_name = "stage_attach";
  stage.logical_stage_id = "stage_attach_id";
  stage.plugin_kind = "neatboxdecode";
  stage.kernel_kind = "boxdecode";
  stage.payload_kind = StagePayloadKind::BoxDecode;
  stage.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  stage.boxdecode.topk = 33;
  manifest.stages.push_back(stage);

  std::string attach_error;
  require(attach_manifest_context(pipeline, manifest, &attach_error),
          "attach_manifest_context should succeed: " + attach_error);

  GstContext* context = gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
  require(context != nullptr, "pipeline should expose manifest context");
  GstContext* copied = gst_context_copy(context);
  require(copied != nullptr, "gst_context_copy should succeed");

  gst_context_unref(context);
  gst_object_unref(pipeline);
  return copied;
}

} // namespace

RUN_TEST("unit_sima_plugin_context_attach_lifetime_test", ([] {
           gst_init(nullptr, nullptr);

           GstContext* context = make_context_snapshot();
           require(context != nullptr, "context snapshot should be created");

           const auto* handle = sima_plugin_manifest_context_handle(context);
           require(handle != nullptr, "snapshot should retain manifest handle");
           require(handle->abi_version == SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION,
                   "manifest handle ABI mismatch");

           SimaPluginManifestLookupStatus status = SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT;
           const auto* stage = sima_plugin_manifest_context_stage_lookup_typed_checked(
               context, "stage_attach_id", "stage_attach", SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE,
               &status);
           require(stage != nullptr, "snapshot should resolve manifest stage");
           require(status == SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK,
                   "snapshot lookup status mismatch");
           require(stage->payload.boxdecode.topk == 33, "snapshot payload mismatch");

           gst_context_unref(context);
         }));

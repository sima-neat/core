#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <string>
#include <vector>

namespace {

void ensure_sima_meta_registered() {
  if (!gst_meta_get_info("GstSimaMeta")) {
    static const gchar* tags[] = {nullptr};
    gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
  }
}

std::vector<std::string> required_fields() {
  return {
      "preproc_original_width", "preproc_original_height", "preproc_resized_width",
      "preproc_resized_height", "preproc_scaled_width",    "preproc_scaled_height",
      "preproc_pad_left",       "preproc_pad_right",       "preproc_pad_top",
      "preproc_pad_bottom",     "preproc_resize_mode",     "preproc_color_in",
      "preproc_color_out",      "preproc_axis_perm",       "preproc_normalize",
      "preproc_quantize",       "preproc_tessellate",      "preproc_affine_m00",
      "preproc_affine_m01",     "preproc_affine_m02",      "preproc_affine_m10",
      "preproc_affine_m11",     "preproc_affine_m12",      "preproc_affine_scale_x",
      "preproc_affine_scale_y", "preproc_affine_offset_x", "preproc_affine_offset_y",
  };
}

simaai::neat::PreprocessRuntimeMeta valid_meta() {
  simaai::neat::PreprocessRuntimeMeta meta;
  meta.original_width = 1280;
  meta.original_height = 720;
  meta.resized_width = 640;
  meta.resized_height = 640;
  meta.scaled_width = 640;
  meta.scaled_height = 360;
  meta.pad_left = 0;
  meta.pad_right = 0;
  meta.pad_top = 140;
  meta.pad_bottom = 140;
  meta.resize_mode = "letterbox";
  meta.color_in = "BGR";
  meta.color_out = "RGB";
  meta.axis_perm = {2, 0, 1};
  meta.normalize = true;
  meta.quantize = true;
  meta.tessellate = true;
  meta.affine_m00 = 2.0;
  meta.affine_m01 = 0.0;
  meta.affine_m02 = 0.0;
  meta.affine_m10 = 0.0;
  meta.affine_m11 = 2.0;
  meta.affine_m12 = -280.0;
  meta.affine_scale_x = 2.0;
  meta.affine_scale_y = 2.0;
  meta.affine_offset_x = 0.0;
  meta.affine_offset_y = -280.0;
  return meta;
}

} // namespace

RUN_TEST("unit_preprocess_meta_validation_test", ([] {
#if !SIMA_HAS_SIMAAI_POOL
           skip_test_exception("unit_preprocess_meta_validation_test requires simaai pool/meta");
#else
           gst_init(nullptr, nullptr);
           ensure_sima_meta_registered();

           GstBuffer* buffer = gst_buffer_new();
           require(buffer != nullptr, "meta validation: failed to allocate GstBuffer");

           const auto req = required_fields();
           const auto meta = valid_meta();
           require(simaai::neat::write_simaai_preprocess_meta(buffer, meta),
                   "meta validation: failed to write preprocess metadata");

           simaai::neat::PreprocessRuntimeMeta parsed{};
           const auto ok =
               simaai::neat::validate_simaai_preprocess_meta_required_fields(buffer, req, &parsed);
           require(!ok.has_value(), "meta validation: valid metadata should pass");
           require(parsed.original_width == 1280 && parsed.original_height == 720,
                   "meta validation: parsed geometry mismatch");
           require(parsed.axis_perm == std::vector<int>({2, 0, 1}),
                   "meta validation: parsed axis_perm mismatch");

           // Missing required field should fail with explicit field name.
           {
             GstCustomMeta* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
             require(custom != nullptr, "meta validation: missing GstSimaMeta");
             GstStructure* s = gst_custom_meta_get_structure(custom);
             require(s != nullptr, "meta validation: missing GstSimaMeta structure");
             gst_structure_remove_field(s, "preproc_color_in");
             const auto err = simaai::neat::validate_simaai_preprocess_meta_required_fields(
                 buffer, req, nullptr);
             require(err.has_value(), "meta validation: missing field should fail");
             require_contains(*err, "preproc_color_in",
                              "meta validation: missing field message mismatch");
             // Restore for next case.
             gst_structure_set(s, "preproc_color_in", G_TYPE_STRING, "BGR", nullptr);
           }

           // Invalid field type/value should fail.
           {
             GstCustomMeta* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
             require(custom != nullptr, "meta validation: missing GstSimaMeta");
             GstStructure* s = gst_custom_meta_get_structure(custom);
             require(s != nullptr, "meta validation: missing GstSimaMeta structure");
             gst_structure_set(s, "preproc_original_width", G_TYPE_INT, 0, nullptr);
             const auto err = simaai::neat::validate_simaai_preprocess_meta_required_fields(
                 buffer, req, nullptr);
             require(err.has_value(), "meta validation: invalid width should fail");
             require_contains(*err, "preproc_original_width",
                              "meta validation: invalid field message mismatch");
           }

           gst_buffer_unref(buffer);
#endif
         }));

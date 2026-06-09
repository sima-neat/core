#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <vector>

namespace {

void ensure_sima_meta_registered() {
  if (!gst_meta_get_info("GstSimaMeta")) {
    static const gchar* tags[] = {nullptr};
    gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
  }
}

simaai::neat::PreprocessRuntimeMeta base_meta() {
  simaai::neat::PreprocessRuntimeMeta meta;
  meta.original_width = 1280;
  meta.original_height = 720;
  meta.resized_width = 640;
  meta.resized_height = 640;
  meta.scaled_width = 640;
  meta.scaled_height = 360;
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
  meta.affine_m11 = 2.0;
  meta.affine_m12 = -280.0;
  meta.affine_scale_x = 2.0;
  meta.affine_scale_y = 2.0;
  meta.affine_offset_y = -280.0;
  return meta;
}

GstStructure* sima_meta_structure(GstBuffer* buffer) {
  GstCustomMeta* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  require(custom != nullptr, "ROI meta: missing GstSimaMeta");
  GstStructure* s = gst_custom_meta_get_structure(custom);
  require(s != nullptr, "ROI meta: missing GstSimaMeta structure");
  return s;
}

std::vector<std::string> roi_list_required_fields() {
  return {
      "preproc_roi_list_enable",        "preproc_roi_count",
      "preproc_roi_batch_indices",      "preproc_roi_rects",
      "preproc_roi_input_batch_size",   "preproc_roi_source_width",
      "preproc_roi_source_height",      "preproc_roi_source_stride_bytes",
      "preproc_roi_pad_value",
  };
}

} // namespace

RUN_TEST("unit_preprocess_roi_meta_test", ([] {
#if !SIMA_HAS_SIMAAI_POOL
           skip_test_exception("unit_preprocess_roi_meta_test requires simaai pool/meta");
#else
           gst_init(nullptr, nullptr);
           ensure_sima_meta_registered();

           {
             GstBuffer* buffer = gst_buffer_new();
             require(buffer != nullptr, "ROI meta: failed to allocate GstBuffer");

             auto meta = base_meta();
             meta.roi_list_enabled = true;
             meta.rois = {
                 {0, 10, 11, 20, 21},
                 {2, -3, 4, 30, 31},
             };
             meta.roi_input_batch_size = 3;
             meta.roi_source_width = 1280;
             meta.roi_source_height = 720;
             meta.roi_source_stride_bytes = 1280 * 3;
             meta.roi_pad_value = 114;
             meta.roi_capacity = 4;
             meta.roi_valid_count = 2;
             meta.roi_input_count = 3;
             meta.roi_dropped_invalid = 1;
             meta.roi_dropped_overflow = 0;
             meta.roi_affines = {
                 {0.5, 0.0, 10.0, 0.0, 0.5, 11.0},
                 {0.25, 0.0, -3.0, 0.0, 0.25, 4.0},
             };

             require(simaai::neat::write_simaai_preprocess_meta(buffer, meta),
                     "ROI meta: failed to write multi-ROI metadata");
             const auto parsed = simaai::neat::read_simaai_preprocess_meta(buffer);
             require(parsed.has_value(), "ROI meta: failed to read multi-ROI metadata");
             require(parsed->roi_list_enabled, "ROI meta: parsed list not enabled");
             require(parsed->rois.size() == 2U, "ROI meta: parsed ROI count mismatch");
             require(parsed->rois[0].batch_index == 0 && parsed->rois[0].x == 10 &&
                         parsed->rois[0].y == 11 && parsed->rois[0].width == 20 &&
                         parsed->rois[0].height == 21,
                     "ROI meta: first ROI mismatch");
             require(parsed->rois[1].batch_index == 2 && parsed->rois[1].x == -3 &&
                         parsed->rois[1].y == 4 && parsed->rois[1].width == 30 &&
                         parsed->rois[1].height == 31,
                     "ROI meta: second ROI mismatch");
             require(parsed->roi_input_batch_size == 3 && parsed->roi_source_width == 1280 &&
                         parsed->roi_source_height == 720 &&
                         parsed->roi_source_stride_bytes == 1280 * 3 &&
                         parsed->roi_pad_value == 114,
                     "ROI meta: parsed ROI source fields mismatch");
             require(parsed->roi_capacity == 4 && parsed->roi_valid_count == 2 &&
                         parsed->roi_input_count == 3 && parsed->roi_dropped_invalid == 1 &&
                         parsed->roi_dropped_overflow == 0,
                     "ROI meta: parsed ROI counters mismatch");
             require(parsed->roi_affines.size() == 2U &&
                         parsed->roi_affines[0].m00 == 0.5 &&
                         parsed->roi_affines[0].m02 == 10.0 &&
                         parsed->roi_affines[1].m00 == 0.25 &&
                         parsed->roi_affines[1].m02 == -3.0,
                     "ROI meta: parsed ROI affines mismatch");

             GstStructure* s = sima_meta_structure(buffer);
             require(!gst_structure_has_field(s, "preproc_roi_enable"),
                     "ROI meta: multi-ROI path must not leave scalar ROI enabled");
             const auto err = simaai::neat::validate_simaai_preprocess_meta_required_fields(
                 buffer, roi_list_required_fields(), nullptr);
             require(!err.has_value(), "ROI meta: required ROI-list fields should validate");
             gst_buffer_unref(buffer);
           }

           {
             GstBuffer* buffer = gst_buffer_new();
             require(buffer != nullptr, "ROI meta: failed to allocate single-ROI buffer");
             auto meta = base_meta();
             meta.roi_list_enabled = true;
             meta.rois = {{1, 7, 8, 9, 10}};
             meta.roi_input_batch_size = 2;
             meta.roi_source_width = 320;
             meta.roi_source_height = 240;
             meta.roi_source_stride_bytes = 320 * 3;
             require(simaai::neat::write_simaai_preprocess_meta(buffer, meta),
                     "ROI meta: failed to write single ROI metadata");
             GstStructure* s = sima_meta_structure(buffer);
             gboolean enabled = FALSE;
             int value = 0;
             require(gst_structure_get_boolean(s, "preproc_roi_enable", &enabled) == TRUE &&
                         enabled == TRUE,
                     "ROI meta: scalar compatibility flag missing");
             require(gst_structure_get_int(s, "preproc_roi_batch_index", &value) == TRUE &&
                         value == 1,
                     "ROI meta: scalar compatibility batch index mismatch");
             require(gst_structure_get_int(s, "preproc_roi_width", &value) == TRUE && value == 9,
                     "ROI meta: scalar compatibility width mismatch");
             gst_buffer_unref(buffer);
           }

           {
             GstBuffer* buffer = gst_buffer_new();
             require(buffer != nullptr, "ROI meta: failed to allocate empty-list buffer");
             auto meta = base_meta();
             meta.roi_list_enabled = true;
             meta.roi_input_batch_size = 2;
             meta.roi_source_width = 1280;
             meta.roi_source_height = 720;
             meta.roi_source_stride_bytes = 1280 * 3;
             require(simaai::neat::write_simaai_preprocess_meta(buffer, meta),
                     "ROI meta: failed to write explicit empty ROI list");
             const auto parsed = simaai::neat::read_simaai_preprocess_meta(buffer);
             require(parsed.has_value() && parsed->roi_list_enabled && parsed->rois.empty(),
                     "ROI meta: explicit empty ROI list did not round trip");
             gst_buffer_unref(buffer);
           }

           {
             GstBuffer* buffer = gst_buffer_new();
             require(buffer != nullptr, "ROI meta: failed to allocate stale-clear buffer");
             auto meta = base_meta();
             meta.roi_list_enabled = true;
             meta.rois = {{0, 1, 2, 3, 4}};
             meta.roi_input_batch_size = 1;
             require(simaai::neat::write_simaai_preprocess_meta(buffer, meta),
                     "ROI meta: failed to write ROI before stale-clear case");
             require(simaai::neat::write_simaai_preprocess_meta(buffer, base_meta()),
                     "ROI meta: failed to rewrite metadata without ROI");
             GstStructure* s = sima_meta_structure(buffer);
             require(!gst_structure_has_field(s, "preproc_roi_list_enable") &&
                         !gst_structure_has_field(s, "preproc_roi_rects") &&
                         !gst_structure_has_field(s, "preproc_roi_enable"),
                     "ROI meta: stale ROI fields were not cleared");
             const auto parsed = simaai::neat::read_simaai_preprocess_meta(buffer);
             require(parsed.has_value() && !parsed->has_roi_list(),
                     "ROI meta: stale-clear read should not report ROI list");
             gst_buffer_unref(buffer);
           }

           {
             GstBuffer* buffer = gst_buffer_new();
             require(buffer != nullptr, "ROI meta: failed to allocate invalid buffer");
             auto meta = base_meta();
             meta.roi_list_enabled = true;
             meta.rois = {{0, 1, 2, 0, 4}};
             require(!simaai::neat::write_simaai_preprocess_meta(buffer, meta),
                     "ROI meta: invalid ROI width should be rejected");
             gst_buffer_unref(buffer);
           }

           {
             GstBuffer* buffer = gst_buffer_new();
             require(buffer != nullptr, "ROI meta: failed to allocate legacy buffer");
             require(simaai::neat::write_simaai_preprocess_meta(buffer, base_meta()),
                     "ROI meta: failed to seed legacy scalar buffer");
             GstStructure* s = sima_meta_structure(buffer);
             gst_structure_set(s, "preproc_roi_enable", G_TYPE_BOOLEAN, TRUE, "preproc_roi_x",
                               G_TYPE_INT, 5, "preproc_roi_y", G_TYPE_INT, 6,
                               "preproc_roi_width", G_TYPE_INT, 7, "preproc_roi_height",
                               G_TYPE_INT, 8, "preproc_roi_source_width", G_TYPE_INT, 1280,
                               "preproc_roi_source_height", G_TYPE_INT, 720, nullptr);
             const auto parsed = simaai::neat::read_simaai_preprocess_meta(buffer);
             require(parsed.has_value() && parsed->roi_list_enabled && parsed->rois.size() == 1U,
                     "ROI meta: legacy scalar ROI did not parse as one ROI");
             require(parsed->rois[0].batch_index == 0 && parsed->rois[0].x == 5 &&
                         parsed->rois[0].y == 6 && parsed->rois[0].width == 7 &&
                         parsed->rois[0].height == 8,
                     "ROI meta: legacy scalar ROI values mismatch");
             gst_buffer_unref(buffer);
           }
#endif
         }));

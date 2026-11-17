#include "nodes/io/Input.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

namespace {

void ensure_sima_meta_registered() {
  if (!gst_meta_get_info("GstSimaMeta")) {
    static const gchar* tags[] = {nullptr};
    gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
  }
}

#if SIMA_HAS_SIMAAI_POOL
void check_input_metadata_reuse() {
  using namespace simaai::neat;
  InputOptions options;
  options.buffer_name = "camera";
  InputBufferPoolGuard guard;
  GstBuffer* buffer = gst_buffer_new();
  auto* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  require(meta != nullptr, "input metadata: initial metadata missing");
  auto* structure = gst_custom_meta_get_structure(meta);
  gst_structure_set(structure, "preproc_original_width", G_TYPE_INT, 1920, "stream-id",
                    G_TYPE_STRING, "camera-1", "custom-field", G_TYPE_STRING, "preserved", nullptr);
  auto* original_buffer = buffer;
  buffer = attach_simaai_meta_inplace(buffer, options, guard, "metadata-test", 41);
  require(buffer == original_buffer, "input metadata: exclusive buffer was replaced");
  require(gst_buffer_get_custom_meta(buffer, "GstSimaMeta") == meta,
          "input metadata: exclusive metadata was unnecessarily rebuilt");
  require(gst_custom_meta_get_structure(meta) == structure,
          "input metadata: exclusive structure was unnecessarily copied");
  gint64 frame = 0;
  require(gst_structure_get_int64(structure, "frame-id", &frame) && frame == 41,
          "input metadata: frame id was not updated");
  require(g_strcmp0(gst_structure_get_string(structure, "orig-stream-id"), "camera-1") == 0,
          "input metadata: original stream id was not retained");

  // The shared source must not change when make_writable copies the envelope.
  GstBuffer* retained = gst_buffer_ref(buffer);
  buffer = attach_simaai_meta_inplace(buffer, options, guard, "metadata-test", 42);
  require(buffer && buffer != retained, "input metadata: shared envelope was not copied");
  auto* copied = gst_custom_meta_get_structure(gst_buffer_get_custom_meta(buffer, "GstSimaMeta"));
  require(gst_structure_get_int64(structure, "frame-id", &frame) && frame == 41,
          "input metadata: shared source was mutated");
  require(gst_structure_get_int64(copied, "frame-id", &frame) && frame == 42,
          "input metadata: copied frame id was not updated");
  require(g_strcmp0(gst_structure_get_string(copied, "custom-field"), "preserved") == 0,
          "input metadata: custom fields were lost");
  gint width = 0;
  require(gst_structure_get_int(copied, "preproc_original_width", &width) && width == 1920,
          "input metadata: preprocessing geometry was lost");
  gst_structure_set(copied, "preproc_original_width", G_TYPE_INT, 1280, nullptr);
  buffer = attach_simaai_meta_inplace(buffer, options, guard, "metadata-test", 43);
  require(gst_structure_get_int(copied, "preproc_original_width", &width) && width == 1280,
          "input metadata: current dynamic geometry was overwritten");
  require(gst_structure_get_int(structure, "preproc_original_width", &width) && width == 1920,
          "input metadata: geometry changes leaked into the retained source");
  gst_buffer_unref(retained);
  gst_buffer_unref(buffer);

  GstBuffer* fresh =
      attach_simaai_meta_inplace(gst_buffer_new(), options, guard, "metadata-test", 44);
  require(fresh && gst_buffer_get_custom_meta(fresh, "GstSimaMeta"),
          "input metadata: fresh envelope was not initialized");
  gst_buffer_unref(fresh);
}
#endif

} // namespace

RUN_TEST("unit_preprocess_meta_template_dynamic_test", ([] {
#if !SIMA_HAS_SIMAAI_POOL
           skip_test_exception(
               "unit_preprocess_meta_template_dynamic_test requires simaai pool/meta");
#else
           using namespace simaai::neat;
           gst_init(nullptr, nullptr);
           ensure_sima_meta_registered();
           check_input_metadata_reuse();

           InputOptions opt;
           PreprocessMetaTemplate tmpl;
           tmpl.enabled = true;
           tmpl.target_width = 640;
           tmpl.target_height = 640;
           tmpl.resize_mode = "letterbox";
           tmpl.color_in = "BGR";
           tmpl.color_out = "RGB";
           tmpl.axis_perm = {2, 0, 1};
           tmpl.normalize = true;
           tmpl.quantize = true;
           tmpl.tessellate = true;
           opt.preprocess_meta = tmpl;

           GstBuffer* b0 = gst_buffer_new();
           require(b0 != nullptr, "meta template: failed to allocate buffer0");
           require(apply_simaai_preprocess_meta_template(b0, opt, 1280, 720),
                   "meta template: failed to apply metadata for frame0");
           const auto m0 = read_simaai_preprocess_meta(b0);
           require(m0.has_value(), "meta template: failed to read metadata for frame0");
           require(m0->original_width == 1280 && m0->original_height == 720,
                   "meta template: frame0 original geometry mismatch");
           require(m0->scaled_width == 640 && m0->scaled_height == 360,
                   "meta template: frame0 scaled geometry mismatch");
           require(m0->axis_perm == std::vector<int>({2, 0, 1}),
                   "meta template: frame0 axis_perm mismatch");
           require(m0->pad_top == 140 && m0->pad_bottom == 140,
                   "meta template: frame0 padding mismatch");
           require(m0->affine_scale_x > 1.9 && m0->affine_scale_x < 2.1,
                   "meta template: frame0 affine_scale_x mismatch");
           require(m0->affine_offset_y < -200.0, "meta template: frame0 affine offset mismatch");
           gst_buffer_unref(b0);

           GstBuffer* b1 = gst_buffer_new();
           require(b1 != nullptr, "meta template: failed to allocate buffer1");
           require(apply_simaai_preprocess_meta_template(b1, opt, 640, 480),
                   "meta template: failed to apply metadata for frame1");
           const auto m1 = read_simaai_preprocess_meta(b1);
           require(m1.has_value(), "meta template: failed to read metadata for frame1");
           require(m1->original_width == 640 && m1->original_height == 480,
                   "meta template: frame1 original geometry mismatch");
           require(m1->scaled_width == 640 && m1->scaled_height == 480,
                   "meta template: frame1 scaled geometry mismatch");
           require(m1->axis_perm == std::vector<int>({2, 0, 1}),
                   "meta template: frame1 axis_perm mismatch");
           require(m1->pad_top == 80 && m1->pad_bottom == 80,
                   "meta template: frame1 padding mismatch");
           require(m1->affine_scale_x > 0.9 && m1->affine_scale_x < 1.1,
                   "meta template: frame1 affine_scale_x mismatch");
           require(m1->affine_offset_y < -70.0, "meta template: frame1 affine offset mismatch");
           gst_buffer_unref(b1);
#endif
         }));

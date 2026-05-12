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

} // namespace

RUN_TEST("unit_preprocess_meta_template_dynamic_test", ([] {
#if !SIMA_HAS_SIMAAI_POOL
           skip_test_exception(
               "unit_preprocess_meta_template_dynamic_test requires simaai pool/meta");
#else
           using namespace simaai::neat;
           gst_init(nullptr, nullptr);
           ensure_sima_meta_registered();

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

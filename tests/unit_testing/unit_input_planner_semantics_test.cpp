#include "model/Model.h"
#include "model/internal/InputPlanner.h"
#include "test_main.h"
#include "test_utils.h"

#include <functional>
#include <string>
#include <vector>

namespace {

using simaai::neat::AutoFlag;
using simaai::neat::InputKind;
using simaai::neat::Model;
using simaai::neat::NormalizePreset;
using simaai::neat::PreprocessGraphFamily;
using simaai::neat::ResizeMode;
using simaai::neat::Transform;
using simaai::neat::TransformType;
using simaai::neat::internal::GraphFamilyCapabilities;
using simaai::neat::internal::PreprocessCapabilities;
using simaai::neat::internal::PreprocessPlannerResult;
using simaai::neat::internal::plan_preprocess;

GraphFamilyCapabilities all_ops_family(bool available) {
  GraphFamilyCapabilities fam;
  fam.available = available;
  fam.supports_resize = true;
  fam.supports_color_convert = true;
  fam.supports_layout_convert = true;
  fam.supports_normalize = true;
  fam.supports_quantize = true;
  fam.supports_tessellate = true;
  return fam;
}

PreprocessCapabilities all_caps() {
  PreprocessCapabilities caps;
  caps.preproc = all_ops_family(true);
  caps.quant = all_ops_family(true);
  caps.tess = all_ops_family(true);
  caps.quanttess = all_ops_family(true);
  return caps;
}

bool has_warning(const PreprocessPlannerResult& out, const std::string& needle) {
  for (const auto& warn : out.resolved_plan.warnings) {
    if (warn.find(needle) != std::string::npos)
      return true;
  }
  return false;
}

void require_throws_contains(const std::function<void()>& fn, const std::string& needle,
                             const std::string& msg) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(e.what(), needle, msg);
    return;
  }
  throw std::runtime_error(msg + " (no exception)");
}

} // namespace

RUN_TEST("unit_input_planner_semantics_test", ([] {
           PreprocessCapabilities caps = all_caps();

           // transforms + simple flags => warn + transforms override.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Image;
             opt.preprocess.resize.enable = AutoFlag::On;

             Transform t;
             t.type = TransformType::Resize;
             t.resize.enable = AutoFlag::On;
             t.resize.width = 320;
             t.resize.height = 320;
             t.resize.mode = ResizeMode::Stretch;
             opt.preprocess.transforms.push_back(t);

             const PreprocessPlannerResult out = plan_preprocess(opt, caps);
             require(out.resolved_plan.transforms_override,
                     "planner: transforms override flag should be true");
             require(has_warning(out, "using transforms"),
                     "planner: expected transforms precedence warning");
             require(out.resolved_plan.effective.resize.width == 320,
                     "planner: transform resize width should override simple flags");
             require(out.resolved_plan.effective.resize.height == 320,
                     "planner: transform resize height should override simple flags");
           }

           // preprocess.enable=Off + transforms => warn + apply transforms.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Image;
             opt.preprocess.enable = AutoFlag::Off;

             Transform t;
             t.type = TransformType::Normalize;
             t.normalize.enable = AutoFlag::On;
             t.normalize.has_explicit_stats = true;
             t.normalize.mean = {0.0f, 0.0f, 0.0f};
             t.normalize.stddev = {255.0f, 255.0f, 255.0f};
             opt.preprocess.transforms.push_back(t);

             const PreprocessPlannerResult out = plan_preprocess(opt, caps);
             require(out.resolved_plan.enabled, "planner: transforms should keep preprocess enabled");
             require(has_warning(out, "enable=Off with non-empty transforms"),
                     "planner: expected enable-off transform warning");
             require(out.resolved_plan.effective.normalize.enable == AutoFlag::On,
                     "planner: normalize should stay enabled from transforms");
           }

           // COCO_YOLO preset expansion defaults.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Image;
             opt.preprocess.preset = NormalizePreset::COCO_YOLO;
             const PreprocessPlannerResult out = plan_preprocess(opt, caps);
             require(out.resolved_plan.effective.resize.enable == AutoFlag::On,
                     "planner: COCO_YOLO should enable resize");
             require(out.resolved_plan.effective.resize.mode == ResizeMode::Letterbox,
                     "planner: COCO_YOLO should set letterbox");
             require(out.resolved_plan.effective.resize.pad_value == 114,
                     "planner: COCO_YOLO pad value must be 114");
             require(out.resolved_plan.effective.normalize.enable == AutoFlag::On,
                     "planner: COCO_YOLO should enable normalize");
             require(out.resolved_plan.effective.normalize.stddev[0] == 1.0f &&
                         out.resolved_plan.effective.normalize.stddev[1] == 1.0f &&
                         out.resolved_plan.effective.normalize.stddev[2] == 1.0f,
                     "planner: COCO_YOLO stddev must default to 1");
           }

           // Tensor auto-family should lower tensor ingress to QuantTess when available.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Tensor;

             PreprocessCapabilities auto_caps = caps;
             auto_caps.tensor_auto_family = PreprocessGraphFamily::QuantTess;

             const PreprocessPlannerResult out = plan_preprocess(opt, auto_caps);
             require(out.resolved_plan.enabled,
                     "planner: tensor auto-family should keep preprocess enabled");
             require(out.resolved_plan.graph_family == PreprocessGraphFamily::QuantTess,
                     "planner: tensor auto-family should resolve to QuantTess");
             require(out.pipeline_type == simaai::neat::internal::PipelineType::QuantTess,
                     "planner: tensor auto-family should set QuantTess pipeline type");
             require(out.resolved_plan.effective.quantize.enable == AutoFlag::On,
                     "planner: tensor auto-family should enable quantize");
             require(out.resolved_plan.effective.tessellate.enable == AutoFlag::On,
                     "planner: tensor auto-family should enable tessellate");
             require(out.modelpack_media_type == "application/vnd.simaai.tensor",
                     "planner: tensor auto-family should preserve tensor ingress media type");
           }

           // Capability gating: missing requested graph family => hard error.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Tensor;
             opt.preprocess.quantize.enable = AutoFlag::On;
             opt.preprocess.tessellate.enable = AutoFlag::Off;

             PreprocessCapabilities missing_quant = caps;
             missing_quant.quant = all_ops_family(false);
             require_throws_contains([&]() { (void)plan_preprocess(opt, missing_quant); },
                                     "unavailable",
                                     "planner: quant family unavailability should throw");
           }

           // Capability gating: op unsupported by selected family => hard error.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Tensor;
             opt.preprocess.quantize.enable = AutoFlag::On;
             opt.preprocess.tessellate.enable = AutoFlag::Off;
             opt.preprocess.color_convert.enable = AutoFlag::On;

             PreprocessCapabilities limited = caps;
             limited.quant.supports_color_convert = false;
             require_throws_contains([&]() { (void)plan_preprocess(opt, limited); }, "color_convert",
                                     "planner: unsupported op should throw");
           }

           // Meta contract fields should include conversion + normalize context.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Image;
             const PreprocessPlannerResult out = plan_preprocess(opt, caps);

             auto has_required = [&](const char* field) {
               for (const auto& entry : out.resolved_plan.meta_contract.required_fields) {
                 if (entry == field)
                   return true;
               }
               return false;
             };

             require(has_required("preproc_color_in"),
                     "planner: meta contract should require preproc_color_in");
             require(has_required("preproc_color_out"),
                     "planner: meta contract should require preproc_color_out");
             require(has_required("preproc_axis_perm"),
                     "planner: meta contract should require preproc_axis_perm");
             require(has_required("preproc_normalize"),
                     "planner: meta contract should require preproc_normalize");
           }

           // Quant+Tess route should select QuantTess when requested and available.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Tensor;
             opt.preprocess.quantize.enable = AutoFlag::On;
             opt.preprocess.tessellate.enable = AutoFlag::On;

             const PreprocessPlannerResult out = plan_preprocess(opt, caps);
             require(out.resolved_plan.graph_family == PreprocessGraphFamily::QuantTess,
                     "planner: quant+tess should lower to QuantTess");
             require(out.pipeline_type == simaai::neat::internal::PipelineType::QuantTess,
                     "planner: pipeline type should be QuantTess");
           }

           // Explicit normalize off with ImageNet preset should emit requirement warning.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Image;
             opt.preprocess.preset = NormalizePreset::ImageNet;
             opt.preprocess.normalize.enable = AutoFlag::Off;
             opt.preprocess.resize.enable = AutoFlag::On;
             opt.preprocess.resize.width = 224;
             opt.preprocess.resize.height = 224;

             const PreprocessPlannerResult out = plan_preprocess(opt, caps);
             require(has_warning(out, "code=PREPROC_REQ_WARN_MODEL_DEFAULT"),
                     "planner: expected warning code for normalize off");
             require(has_warning(out, "op=normalize"),
                     "planner: expected normalize op warning when preset expects normalize");
           }

           // Explicit quantize off with tensor auto-family should keep quantize disabled.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Tensor;
             opt.preprocess.quantize.enable = AutoFlag::Off;
             opt.preprocess.tessellate.enable = AutoFlag::Off;
             opt.preprocess.resize.enable = AutoFlag::On;
             opt.preprocess.resize.width = 320;
             opt.preprocess.resize.height = 320;

             PreprocessCapabilities quant_default_caps = caps;
             quant_default_caps.tensor_auto_family = PreprocessGraphFamily::QuantTess;

             const PreprocessPlannerResult out = plan_preprocess(opt, quant_default_caps);
             require(out.resolved_plan.effective.quantize.enable == AutoFlag::Off,
                     "planner: explicit quantize Off should override tensor auto-family");
             require(out.resolved_plan.graph_family == PreprocessGraphFamily::Preproc,
                     "planner: explicit quantize/tess Off should fall back to Preproc");
           }

           // Tensor auto-family should resolve individual Quant and Tess families too.
           {
             Model::Options quant_opt;
             quant_opt.preprocess.kind = InputKind::Tensor;
             PreprocessCapabilities quant_caps = caps;
             quant_caps.tensor_auto_family = PreprocessGraphFamily::Quant;

             const PreprocessPlannerResult quant_out = plan_preprocess(quant_opt, quant_caps);
             require(quant_out.resolved_plan.graph_family == PreprocessGraphFamily::Quant,
                     "planner: tensor auto-family Quant should resolve to Quant");

             Model::Options tess_opt;
             tess_opt.preprocess.kind = InputKind::Tensor;
             PreprocessCapabilities tess_caps = caps;
             tess_caps.tensor_auto_family = PreprocessGraphFamily::Tess;

             const PreprocessPlannerResult tess_out = plan_preprocess(tess_opt, tess_caps);
             require(tess_out.resolved_plan.graph_family == PreprocessGraphFamily::Tess,
                     "planner: tensor auto-family Tess should resolve to Tess");
           }

           // Explicit normalize off with no preset expectation should not emit requirement warning.
           {
             Model::Options opt;
             opt.preprocess.kind = InputKind::Image;
             opt.preprocess.resize.enable = AutoFlag::On;
             opt.preprocess.resize.width = 224;
             opt.preprocess.resize.height = 224;
             opt.preprocess.normalize.enable = AutoFlag::Off;

             const PreprocessPlannerResult out = plan_preprocess(opt, caps);
             require(!has_warning(out, "op=normalize"),
                     "planner: should not warn normalize off without model/preset expectation");
           }
         }));

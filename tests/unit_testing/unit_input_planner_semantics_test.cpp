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
using simaai::neat::internal::plan_preprocess;
using simaai::neat::internal::PreprocessCapabilities;
using simaai::neat::internal::PreprocessPlannerResult;
using simaai::neat::internal::PreprocessRouteContext;
using simaai::neat::internal::select_shared_input_range;

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

// Multi-ingress MPKs: only a range shared by every ingress tensor is used.
void require_shared_input_range_selection() {
  using Ranges = std::vector<std::vector<double>>;
  bool ambiguous = true;
  require(select_shared_input_range(Ranges{{0.0, 1.0}}, &ambiguous) ==
                  std::vector<double>({0.0, 1.0}) &&
              !ambiguous,
          "ranges: single ingress range is used");
  require(select_shared_input_range(Ranges{{0.0, 1.0}, {0.0, 1.0}}, &ambiguous) ==
                  std::vector<double>({0.0, 1.0}) &&
              !ambiguous,
          "ranges: identical ingress ranges are used");
  require(select_shared_input_range(Ranges{{0.0, 1.0}, {0.0, 255.0}}, &ambiguous).empty() &&
              ambiguous,
          "ranges: differing ingress ranges are ambiguous");
  require(select_shared_input_range(Ranges{{0.0, 1.0}, {}}, &ambiguous).empty() && ambiguous,
          "ranges: a missing range after a declared one is ambiguous");
  require(select_shared_input_range(Ranges{{}, {0.0, 1.0}}, &ambiguous).empty() && ambiguous,
          "ranges: a missing range before a declared one is ambiguous");
  require(select_shared_input_range(Ranges{{}, {}}, &ambiguous).empty() && !ambiguous,
          "ranges: no declared range is not ambiguous");
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
             require(out.resolved_plan.enabled,
                     "planner: transforms should keep preprocess enabled");
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
             require_throws_contains([&]() { (void)plan_preprocess(opt, limited); },
                                     "color_convert", "planner: unsupported op should throw");
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

           // normalize=Auto follows the model input_range once route selection picks the
           // image preproc stage, from simple options or from a transform list.
           {
             PreprocessCapabilities model_caps = all_caps();
             model_caps.has_model_input_normalization = true;
             model_caps.model_input_mean = {0.0f, 0.0f, 0.0f};
             model_caps.model_input_stddev = {1.0f, 1.0f, 1.0f};
             PreprocessRouteContext image_route;
             image_route.image_preproc_selected = true;

             const auto require_model_normalized = [](const PreprocessPlannerResult& out,
                                                      const std::string& label) {
               const auto& rp = out.resolved_plan;
               require(rp.enabled && rp.graph_family == PreprocessGraphFamily::Preproc &&
                           rp.graph_kernel == "preproc" && out.include_preprocess_stage,
                       label + ": resolved graph must report the enabled preproc stage");
               require(rp.resolved_kind == InputKind::Image,
                       label + ": resolved kind must be Image");
               require(rp.effective.normalize.enable == AutoFlag::On && out.normalize,
                       label + ": normalize=Auto must resolve On from model input_range");
               require(rp.effective.normalize.mean[0] == 0.0f &&
                           rp.effective.normalize.stddev[0] == 1.0f && out.mean.size() == 3U &&
                           out.mean[0] == 0.0f && out.stddev.size() == 3U &&
                           out.stddev[0] == 1.0f,
                       label + ": normalize stats must come from input_range");
               size_t reports = 0;
               for (const auto& warn : rp.warnings) {
                 reports += warn.find("resolved On from model input_range") != std::string::npos;
               }
               require(reports == 1U, label + ": model-driven normalize must be reported once");
             };
             const auto require_normalize_off = [](const PreprocessPlannerResult& out,
                                                   const std::string& label) {
               require(out.resolved_plan.effective.normalize.enable == AutoFlag::Off &&
                           !out.normalize,
                       label + ": normalize must stay Off");
               require(!has_warning(out, "resolved On from model input_range"),
                       label + ": must not report model-driven normalize");
             };

             // Before route selection nothing is resolved from input_range.
             Model::Options default_opt;
             const PreprocessPlannerResult unrouted = plan_preprocess(default_opt, model_caps);
             require(!unrouted.resolved_plan.enabled &&
                         unrouted.resolved_plan.graph_family == PreprocessGraphFamily::Disabled,
                     "planner: default options without a route must not enable preprocess");
             require_normalize_off(unrouted, "planner: default options without a route");

             Model::Options image_opt;
             image_opt.preprocess.kind = InputKind::Image;
             require_normalize_off(plan_preprocess(image_opt, model_caps),
                                   "planner: explicit Image without a route");

             // Default options on an MPK-selected image preproc route, with the explicit
             // Image kind as positive control.
             require_model_normalized(plan_preprocess(default_opt, model_caps, image_route),
                                      "route: default options");
             require_model_normalized(plan_preprocess(image_opt, model_caps, image_route),
                                      "route: explicit Image");

             Model::Options resize_opt;
             resize_opt.preprocess.resize.width = 640;
             resize_opt.preprocess.resize.height = 640;
             require_model_normalized(plan_preprocess(resize_opt, model_caps, image_route),
                                      "route: simple resize options");

             // Transform lists: the effective Normalize intent decides, not the list itself.
             Transform normalize_auto;
             normalize_auto.type = TransformType::Normalize;
             Model::Options transform_opt;
             transform_opt.preprocess.transforms = {normalize_auto};
             require_model_normalized(plan_preprocess(transform_opt, model_caps, image_route),
                                      "route: Auto kind + Normalize(Auto) transform");
             Model::Options transform_image_opt = transform_opt;
             transform_image_opt.preprocess.kind = InputKind::Image;
             require_model_normalized(plan_preprocess(transform_image_opt, model_caps, image_route),
                                      "route: Image kind + Normalize(Auto) transform");

             Transform resize_only;
             resize_only.type = TransformType::Resize;
             resize_only.resize.width = 640;
             resize_only.resize.height = 640;
             Model::Options resize_transform_opt;
             resize_transform_opt.preprocess.transforms = {resize_only};
             require_model_normalized(
                 plan_preprocess(resize_transform_opt, model_caps, image_route),
                 "route: transform list without Normalize");

             Transform normalize_off = normalize_auto;
             normalize_off.normalize.enable = AutoFlag::Off;
             Model::Options transform_off_opt;
             transform_off_opt.preprocess.transforms = {normalize_off};
             require_normalize_off(plan_preprocess(transform_off_opt, model_caps, image_route),
                                   "route: Normalize(Off) transform");

             Transform normalize_stats = normalize_auto;
             normalize_stats.normalize.mean = {0.5f, 0.5f, 0.5f};
             normalize_stats.normalize.stddev = {0.25f, 0.25f, 0.25f};
             normalize_stats.normalize.has_explicit_stats = true;
             Model::Options transform_stats_opt;
             transform_stats_opt.preprocess.transforms = {normalize_stats};
             const PreprocessPlannerResult transform_stats_out =
                 plan_preprocess(transform_stats_opt, model_caps, image_route);
             require(transform_stats_out.resolved_plan.effective.normalize.mean[0] == 0.5f &&
                         transform_stats_out.resolved_plan.effective.normalize.stddev[0] == 0.25f,
                     "route: explicit transform normalize stats must win over input_range");

             // Explicit simple options win over the model contract.
             Model::Options off_opt;
             off_opt.preprocess.normalize.enable = AutoFlag::Off;
             require_normalize_off(plan_preprocess(off_opt, model_caps, image_route),
                                   "route: explicit normalize Off");

             Model::Options stats_opt;
             stats_opt.preprocess.normalize.mean = {0.5f, 0.5f, 0.5f};
             stats_opt.preprocess.normalize.stddev = {0.25f, 0.25f, 0.25f};
             stats_opt.preprocess.normalize.has_explicit_stats = true;
             const PreprocessPlannerResult stats_out =
                 plan_preprocess(stats_opt, model_caps, image_route);
             require(stats_out.resolved_plan.effective.normalize.mean[0] == 0.5f &&
                         stats_out.resolved_plan.effective.normalize.stddev[0] == 0.25f,
                     "route: explicit normalize stats must win over input_range");

             Model::Options preset_opt;
             preset_opt.preprocess.preset = NormalizePreset::ImageNet;
             const PreprocessPlannerResult preset_out =
                 plan_preprocess(preset_opt, model_caps, image_route);
             require(preset_out.resolved_plan.effective.normalize.mean[0] == 0.485f,
                     "route: preset stats must win over input_range");

             Model::Options disabled_opt;
             disabled_opt.preprocess.enable = AutoFlag::Off;
             const PreprocessPlannerResult disabled_out =
                 plan_preprocess(disabled_opt, model_caps, image_route);
             require(!disabled_out.resolved_plan.enabled,
                     "route: explicit preprocess.enable=Off must stay disabled");
             require_normalize_off(disabled_out, "route: explicit preprocess.enable=Off");
             Model::Options disabled_transform_opt = transform_opt;
             disabled_transform_opt.preprocess.enable = AutoFlag::Off;
             require_normalize_off(plan_preprocess(disabled_transform_opt, model_caps, image_route),
                                   "route: preprocess.enable=Off with transforms");

             Model::Options tensor_opt;
             tensor_opt.preprocess.kind = InputKind::Tensor;
             require_normalize_off(plan_preprocess(tensor_opt, model_caps, image_route),
                                   "route: tensor input");

             // Models without a usable range keep raw pixels; problems are reported.
             PreprocessCapabilities no_range_caps = model_caps;
             no_range_caps.has_model_input_normalization = false;
             const PreprocessPlannerResult no_range_out =
                 plan_preprocess(default_opt, no_range_caps, image_route);
             require_normalize_off(no_range_out, "route: missing or identity input_range");
             require(no_range_out.requirement_issues.empty() &&
                         !has_warning(no_range_out, "input_range"),
                     "route: a missing input_range is not a contract issue");

             PreprocessCapabilities ambiguous_caps = no_range_caps;
             ambiguous_caps.model_input_range_ambiguous = true;
             const PreprocessPlannerResult ambiguous_out =
                 plan_preprocess(default_opt, ambiguous_caps, image_route);
             require_normalize_off(ambiguous_out, "route: ambiguous multi-ingress input_range");
             require(has_warning(ambiguous_out, "image ingress range is ambiguous"),
                     "route: ambiguous multi-ingress input_range must be reported");

             PreprocessCapabilities invalid_caps = no_range_caps;
             invalid_caps.invalid_model_input_range = "[1, 1]";
             const PreprocessPlannerResult invalid_out =
                 plan_preprocess(default_opt, invalid_caps, image_route);
             require_normalize_off(invalid_out, "route: invalid input_range");
             require(invalid_out.requirement_issues.size() == 1U &&
                         invalid_out.requirement_issues.front().code ==
                             "PREPROC_REQ_WARN_MODEL_INPUT_RANGE_INVALID" &&
                         invalid_out.requirement_issues.front().source ==
                             simaai::neat::internal::RequirementSource::MlaContract,
                     "route: invalid input_range must raise a contract diagnostic");
             require(has_warning(invalid_out, "input_range [1, 1]"),
                     "route: invalid input_range diagnostic must name the declared range");
             require(plan_preprocess(off_opt, invalid_caps, image_route).requirement_issues.empty(),
                     "route: explicit normalize does not depend on the invalid input_range");

             require_shared_input_range_selection();
           }
         }));

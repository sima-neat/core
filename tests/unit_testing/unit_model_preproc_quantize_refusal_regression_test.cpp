#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "nodes/sima/Preproc.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

// The strict fixture's MLA input is INT8 behind a pre-MLA quantizer (q_scale 255.07, q_zp -128).
// Raw 0-255 pixels must never reach it typed INT8, so a Model-managed Preproc stage whose
// normalization resolved Off is refused instead of built without quantization.
RUN_TEST("unit_model_preproc_quantize_refusal_regression_test", ([] {
           using namespace simaai::neat;

           const auto fixture = sima_test::make_strict_model_archive_fixture(
               "model_preproc_quantize_refusal", {}, true, "yolo_v9c_seg",
               /*include_topology_artifacts=*/true);

           Model::Options options;
           options.preprocess.resize.enable = AutoFlag::On;
           options.preprocess.resize.width = 640;
           options.preprocess.resize.height = 640;
           options.preprocess.resize.mode = ResizeMode::Letterbox;
           options.preprocess.resize.pad_value = 114;

           const auto require_refused = [&](const Model::Options& opt, const std::string& label) {
             // Planning-only calls keep working; only building the stage is refused.
             Model model(fixture.tar_path, opt);
             require(model.preprocess_requirements().quant_needed,
                     label + ": the pack must still report that quantization is needed");
             require_neat_error([&] { (void)PreprocOptions(model); },
                                "misconfig.option_out_of_range", "preprocess.normalize");
           };

           // Normalization left unconfigured (Auto) resolves Off.
           require_refused(options, "normalize Auto");

           // The Auto advisory explains the quantized route and no longer suggests Off.
           {
             const auto warnings =
                 Model(fixture.tar_path, options).resolved_preprocess_plan().warnings;
             std::size_t advisories = 0;
             for (const auto& warning : warnings) {
               if (warning.find("preprocess.normalize=Auto resolved Off") == std::string::npos)
                 continue;
               ++advisories;
               require_contains(warning, "quantized", "advisory must say the input is quantized");
               require_contains(warning, "will be refused",
                                "advisory must say the stage is refused");
               require_contains(warning, "preprocess.normalize.enable=On",
                                "advisory must point to explicit normalization");
               require_contains(warning, "preprocess.preset", "advisory must mention presets");
               require(warning.find("normalize.enable=Off") == std::string::npos &&
                           warning.find("raw pixels") == std::string::npos,
                       "advisory must not recommend Off or raw pixels");
             }
             require(advisories == 1U, "exactly one Auto advisory is expected");
           }

           // Explicit Off.
           auto off = options;
           off.preprocess.normalize.enable = AutoFlag::Off;
           require_refused(off, "normalize Off");
           for (const auto& warning :
                Model(fixture.tar_path, off).resolved_preprocess_plan().warnings) {
             require(warning.find("preprocess.normalize=Auto resolved Off") == std::string::npos,
                     "explicit Off must not produce the Auto advisory");
           }

           // Explicit Off stays refused even when a preset is also set.
           auto off_preset = off;
           off_preset.preprocess.preset = NormalizePreset::COCO_YOLO;
           require_refused(off_preset, "normalize Off with preset");

           // Guard: explicit On and presets still build and quantize with the pack's quantizer.
           auto on = options;
           on.preprocess.normalize.enable = AutoFlag::On;
           auto preset = options;
           preset.preprocess.preset = NormalizePreset::COCO_YOLO;
           for (const auto* opt : {&on, &preset}) {
             const PreprocOptions pre(Model(fixture.tar_path, *opt));
             require(pre.normalize, "normalize On/preset must keep normalization enabled");
             require(pre.output_dtype == "INT8" && pre.q_scale.has_value() && pre.q_zp.has_value(),
                     "normalize On/preset must keep the pack's quantizer on the Preproc stage");
           }
         }));

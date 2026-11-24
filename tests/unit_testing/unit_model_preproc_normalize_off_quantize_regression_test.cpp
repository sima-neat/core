#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "nodes/sima/Preproc.h"
#include "test_main.h"
#include "test_utils.h"

#ifdef SIMA_NEAT_INTERNAL
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include <gstsimaaitensorbuffer_meta.h>
#include <simaai/scheduler/client/kernel.h>
#endif

#include <memory>
#include <string>
#include <variant>

namespace {

using namespace simaai::neat;

Model::Options letterbox_options() {
  Model::Options options;
  options.preprocess.resize.enable = AutoFlag::On;
  options.preprocess.resize.width = 640;
  options.preprocess.resize.height = 640;
  options.preprocess.resize.mode = ResizeMode::Letterbox;
  options.preprocess.resize.pad_value = 114;
  return options;
}

#ifdef SIMA_NEAT_INTERNAL
// Lowers the Model's Preproc stage through the frame adapter and checks the dtype the kernel
// produces and the metadata the stage publishes.
void require_lowered_quantized(const PreprocOptions& opt, bool normalize,
                               const std::string& label) {
  namespace rt = simaai::scheduler::client;
  using Adapter = pipeline_internal::sima::kernel_lowering::PreprocFrameAdapter;
  require(opt.compiled_contract != nullptr, label + ": missing compiled contract");
  auto policy = opt.compiled_contract->payload;
  if (policy.resolved_exec_backend.empty())
    policy.resolved_exec_backend = "EVXX";
  auto checked = Adapter::check_policy(policy);
  require(std::holds_alternative<std::shared_ptr<const Adapter::CheckedPolicy>>(checked),
          label + ": frame adapter rejected the stage");
  const auto checked_policy = std::get<std::shared_ptr<const Adapter::CheckedPolicy>>(checked);

  const auto inspection = rt::inspect_kernel(Adapter::checked_executable(*checked_policy));
  require(std::holds_alternative<rt::KernelInspection>(inspection),
          label + ": kernel inspection failed");
  const auto& outputs = std::get<rt::KernelInspection>(inspection).outputs();
  require(outputs.size() == 1U && outputs[0].dtype == rt::DType::Int8,
          label + ": the kernel must produce signed INT8");

  auto frozen = Adapter::checked_initial_preprocess_facts(checked_policy);
  require(std::holds_alternative<::simaai::gst::TensorBufferPreprocessFactsOwner>(frozen),
          label + ": no initial preprocess facts");
  const auto* facts = ::simaai::gst::tensor_buffer_preprocess_facts(
      std::get<::simaai::gst::TensorBufferPreprocessFactsOwner>(frozen));
  require(facts != nullptr && facts->quantize && facts->normalize == normalize,
          label + ": published quantize/normalize flags are wrong");
}
#endif

// Builds the Model's Preproc stage for the given options and checks that the pack's quantizer
// is attached, normalization is off or on as asked, and the stage lowers to quantized INT8.
void require_quantized_stage(const sima_test::ModelArchiveFixture& fixture,
                             const Model::Options& options, bool normalize,
                             const std::string& label) {
  Model model(fixture.tar_path, options);
  const auto requirements = model.preprocess_requirements();
  require(requirements.quant_needed && requirements.q_scale && requirements.q_zp,
          label + ": the pack must need quantization");
  const PreprocOptions opt(model);
  require(opt.normalize == normalize, label + ": unexpected normalize flag");
  require(opt.output_dtype == "INT8", label + ": output must be INT8");
  require(opt.q_scale == requirements.q_scale && opt.q_zp == requirements.q_zp,
          label + ": the pack's quantizer must be attached to the stage");
#ifdef SIMA_NEAT_INTERNAL
  require_lowered_quantized(opt, normalize, label);
#endif
}

} // namespace

// A Model-managed Preproc stage on a quantized INT8 pack must still quantize raw pixels when
// normalization is Off (explicitly, with a preset, or because Auto resolved Off), and it must
// keep the combined normalize + quantize lowering when normalization is On.
RUN_TEST("unit_model_preproc_normalize_off_quantize_regression_test", ([] {
           const auto strict = sima_test::make_strict_model_archive_fixture(
               "model_preproc_normalize_off_quantize", {}, true, "yolo_v9c_seg",
               /*include_topology_artifacts=*/true);
           // Pre-MLA quantizer: multiplier 0.5, zero point 0.
           const auto half = sima_test::make_strict_model_archive_fixture_with_quantizer(
               "model_preproc_normalize_off_quantize_half", 0.5, 0);

           const auto auto_options = letterbox_options();
           auto off = auto_options;
           off.preprocess.normalize.enable = AutoFlag::Off;
           auto off_preset = off;
           off_preset.preprocess.preset = NormalizePreset::COCO_YOLO;
           auto on = auto_options;
           on.preprocess.normalize.enable = AutoFlag::On;
           auto preset = auto_options;
           preset.preprocess.preset = NormalizePreset::COCO_YOLO;

           for (const auto* fixture : {&strict, &half}) {
             const std::string name = fixture == &strict ? "strict 255.07/-128: " : "0.5/0: ";
             require_quantized_stage(*fixture, off, false, name + "normalize Off");
             require_quantized_stage(*fixture, off_preset, false, name + "Off with preset");
             require_quantized_stage(*fixture, auto_options, false, name + "normalize Auto");
             // Guards the combined normalize + quantize lowering.
             require_quantized_stage(*fixture, on, true, name + "normalize On");
             require_quantized_stage(*fixture, preset, true, name + "preset");
           }

           // The pack's quantizer, not a default, reaches the stage.
           {
             const PreprocOptions opt(Model(half.tar_path, off));
             require(opt.q_scale == 0.5 && opt.q_zp == 0, "expected multiplier 0.5, zero point 0");
           }

           // The Auto advisory describes the quantized route.
           {
             std::size_t advisories = 0;
             for (const auto& warning :
                  Model(strict.tar_path, auto_options).resolved_preprocess_plan().warnings) {
               if (warning.find("preprocess.normalize=Auto resolved Off") == std::string::npos)
                 continue;
               ++advisories;
               require_contains(warning, "preprocess.normalize.enable=On",
                                "advisory must point to explicit normalization");
               require_contains(warning, "preprocess.preset", "advisory must mention presets");
               require_contains(warning, "quantiz", "advisory must say Off quantizes raw pixels");
               require(warning.find("refused") == std::string::npos,
                       "advisory must not say the stage is refused");
             }
             require(advisories == 1U, "exactly one Auto advisory is expected");
           }

           // Explicit Off never produces the Auto advisory.
           for (const auto& warning :
                Model(strict.tar_path, off).resolved_preprocess_plan().warnings) {
             require(warning.find("preprocess.normalize=Auto resolved Off") == std::string::npos,
                     "explicit Off must not produce the Auto advisory");
           }
         }));

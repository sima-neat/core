#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <array>
#include <cstring>

namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
using namespace ::simaai::scheduler::client;
using ::simaai::scheduler::client::check_kernel;
namespace {
Error invalid(const char* field, const char* message) {
  return {ErrorCode::InvalidRequest, field, message};
}
} // namespace
bool visual_family(std::string_view name) {
  return name == "feature_histogram" || name == "grider_fast" || name == "track_descriptor" ||
         name == "track_klt";
}
Result<std::shared_ptr<KernelStageRecipe>> check_visual_stage(const StageStaticSpec& stage) {
  const auto& cfg = stage.processcvu;
  if (cfg.resolved_exec_backend == "A65")
    return Error{ErrorCode::UnsupportedTarget, "target", "visual kernels are EV74-only"};
  if (cfg.resolved_exec_backend != "EV74" && cfg.resolved_exec_backend != "EVXX")
    return invalid("target", "Core must resolve the visual backend");
  if (cfg.width <= 0 || cfg.height <= 0 || cfg.batch_size <= 0 || cfg.batch_size > 64)
    return invalid("image", "invalid prepared image geometry");
  const auto b = std::int64_t(cfg.batch_size);
  const TensorSpec image{std::vector<std::int64_t>{b, cfg.height, cfg.width, 1}, DType::UInt8};
  auto result = std::make_shared<KernelStageRecipe>();
  result->target = Hardware::EV74;
  struct VisualMember {
    std::vector<std::size_t> inputs, outputs, dependencies;
  };
  std::optional<Error> failure;
  const auto append = [&](Result<KernelExecutable> checked, VisualMember member) {
    if (failure)
      return;
    if (const auto* error = std::get_if<Error>(&checked)) {
      failure = *error;
      return;
    }
    gs::TensorStageMember compiled;
    compiled.executable = std::get<KernelExecutable>(std::move(checked));
    for (const auto input : member.inputs)
      compiled.inputs.push_back({input == SIZE_MAX ? 0 : input, 0,
                                 input == SIZE_MAX ? gs::TensorStagePortSource::Constant
                                                   : gs::TensorStagePortSource::Input});
    for (const auto output : member.outputs)
      compiled.outputs.push_back({output, 0, gs::TensorStagePortSource::Output});
    compiled.dependencies = std::move(member.dependencies);
    result->members.push_back(std::move(compiled));
  };
  if (cfg.graph_name == "feature_histogram") {
    const std::array inputs{image},
        outputs{TensorSpec{std::vector<std::int64_t>{b, 256}, DType::Int32}};
    append(check_kernel({"histogram", Hardware::EV74, inputs, outputs, {}}), {{0}, {0}, {}});
  } else if ((cfg.graph_name == "grider_fast") || (cfg.graph_name == "track_descriptor")) {
    if (cfg.max_features <= 0 || cfg.max_features > 65536)
      return invalid("max_features", "invalid packed feature capacity");
    const TensorSpec features{std::vector<std::int64_t>{b, 1 + 3 * std::int64_t(cfg.max_features)},
                              DType::Int32};
    std::vector<TensorSpec> out{features};
    VisualMember member{{0}, {0}, {}};
    const bool brief = cfg.graph_name == "track_descriptor";
    if (brief) {
      if (cfg.descriptor_words != 8)
        return invalid("descriptor_words", "BRIEF descriptors require exactly eight Int32 words");
      out.push_back({std::vector<std::int64_t>{b, cfg.max_features, 8}, DType::Int32});
      member.outputs.push_back(1);
    }
    const std::array inputs{image};
    const std::array<Parameter, 2> parameters{
        {{"threshold", cfg.threshold}, {"output_encoding", "packed"}}};
    append(check_kernel({brief ? "fast_brief" : "fast", Hardware::EV74, inputs, out, parameters}),
           std::move(member));
  } else if (cfg.graph_name == "track_klt") {
    if (cfg.num_points <= 0 || cfg.num_points > 65536 ||
        (cfg.detect_new_features != 0 && cfg.detect_new_features != 1))
      return invalid("tracker", "invalid point capacity or detection mode");
    // Legacy TrackKLT supplies a fixed configured count, not a count prefix in
    // input_points. Retain this small immutable device input once per stage.
    result->constant_data.resize(std::size_t(cfg.batch_size) * sizeof(std::int32_t));
    const std::int32_t count = cfg.num_points;
    for (std::size_t i = 0; i < std::size_t(cfg.batch_size); ++i)
      std::memcpy(result->constant_data.data() + i * sizeof(count), &count, sizeof(count));
    if (cfg.win_half < 0 || cfg.max_iters < 0 || cfg.max_level < 0)
      return invalid("tracker", "tracker iteration/window/level parameters must be nonnegative");
    const std::array tracker_inputs{
        image, image, TensorSpec{std::vector<std::int64_t>{b, cfg.num_points, 2}, DType::Int32},
        TensorSpec{std::vector<std::int64_t>{b, 1}, DType::Int32}};
    const std::array tracker_outputs{
        TensorSpec{std::vector<std::int64_t>{b, cfg.num_points, 2}, DType::Float32},
        TensorSpec{std::vector<std::int64_t>{b, cfg.num_points}, DType::Int32}};
    const std::array<Parameter, 3> tracker_parameters{
        {{"window_half", std::uint32_t(cfg.win_half)},
         {"max_iterations", std::uint32_t(cfg.max_iters)},
         {"pyramid_levels", std::uint32_t(cfg.max_level)}}};
    append(check_kernel(
               {"tracker", Hardware::EV74, tracker_inputs, tracker_outputs, tracker_parameters}),
           {{0, 1, 2, SIZE_MAX}, {0, 1}, {}});
    if (cfg.detect_new_features) {
      if (cfg.max_features <= 0 || cfg.max_features > 65536)
        return invalid("max_features", "invalid detection capacity");
      const std::array detection_inputs{image},
          detection_outputs{TensorSpec{
              std::vector<std::int64_t>{b, 1 + 3 * std::int64_t(cfg.max_features)}, DType::Int32}};
      const std::array<Parameter, 2> parameters{
          {{"threshold", cfg.fast_threshold}, {"output_encoding", "packed"}}};
      append(
          check_kernel({"fast", Hardware::EV74, detection_inputs, detection_outputs, parameters}),
          {{1}, {2}, {0}});
    }
  } else
    return invalid("graph", "unknown or contradictory visual graph identity");
  if (failure)
    return *failure;
  return result;
}
} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering

// SimaBEVPoolPlan hardware suite (A65; the kernels implement no other target):
// the public Graph/Node/Run API compared byte for byte with the oracle.
#include "../support/bevfusion_kernel_test_utils.h"

namespace bt = bevfusion_test;
namespace neat = bt::neat;
namespace ref = bt::ref;

namespace {
std::vector<std::uint8_t> used(const neat::Tensor& plan, std::size_t bytes,
                               const std::string& what) {
  auto all = bt::values<std::uint8_t>(plan, neat::TensorDType::UInt8, plan.shape, what);
  bt::require(all.size() >= bytes, what + ": plan shorter than its used length");
  all.resize(bytes);
  return all;
}

// Run every calibration on one Run and compare each plan's used bytes.
void run_sequence(const neat::SimaBEVPoolPlanOptions& options,
                  const std::vector<ref::Calibration>& calibrations, std::int64_t capacity,
                  const std::string& what) {
  std::vector<neat::TensorList> frames;
  for (const auto& calibration : calibrations)
    frames.push_back(bt::calibration_tensors(calibration));
  const auto outputs = bt::execute(neat::nodes::SimaBEVPoolPlan(options), frames);
  bt::require(outputs.size() == calibrations.size(), what + ": one plan per push");
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto name = what + " frame " + std::to_string(i);
    bt::require(outputs[i].size() == 1 &&
                    outputs[i][0].shape == std::vector<std::int64_t>{capacity},
                name + ": plan tensor shape");
    const auto expected = ref::bev_pool_plan(options, calibrations[i]);
    bt::expect_bytes(used(outputs[i][0], expected.size(), name), expected, name);
  }
}

constexpr std::int64_t kWorstCase = 64 + 4 * 17 + 4 * 24; // v1, 16 cells, 24 points

void cache(bool enabled) {
  auto c = ref::bev_pool_plan_case();
  c.options.cache_plan = enabled;
  const auto a = c.calibration, b = ref::calibration_moved(c.calibration, 2.0f);
  // Hit, miss, hit on the new key, miss back, hit.
  run_sequence(c.options, {a, a, b, b, a, a}, kWorstCase,
               enabled ? "cache_plan on" : "cache_plan off");
}

void calibration_sequence() {
  auto c = ref::bev_pool_plan_case();
  std::vector<ref::Calibration> sequence;
  for (const float dx : {0.0f, 1.0f, 2.0f, -1.0f, 3.0f})
    sequence.push_back(ref::calibration_moved(c.calibration, dx));
  // Also an image augmentation change on camera 0: a different frustum mapping.
  sequence.push_back(ref::calibration_zoomed(c.calibration));
  run_sequence(c.options, sequence, kWorstCase, "calibration sequence");
}

void bf16_input_dtype() {
  auto c = ref::bev_pool_plan_case();
  c.options.input_dtype = neat::TensorDType::BFloat16;
  c.options.cache_plan = true;
  // A format 1 plan does not depend on the pooled dtype.
  run_sequence(c.options, {c.calibration, c.calibration}, kWorstCase, "BF16 input_dtype");
}

void explicit_capacity() {
  auto c = ref::bev_pool_plan_case();
  c.options.plan_capacity_bytes = 4096;
  run_sequence(c.options, {c.calibration}, 4096, "4096-byte plan");
}

std::uint32_t word(const std::vector<std::uint8_t>& plan, std::size_t index) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= std::uint32_t(plan[4 * index + i]) << (8 * i);
  return v;
}

// Format 2 holds exactly the v1 points per cell, rescheduled; its bytes depend
// on the schedule, so the oracle decodes it rather than re-deriving it.
void format_2(neat::TensorDType dtype) {
  auto c = ref::bev_pool_plan_case();
  c.options.format = neat::SimaBEVPoolPlanFormat::V2;
  c.options.context_channels = 16;
  c.options.input_dtype = dtype;
  const auto plan = bt::plan_from_node(c.options, c.calibration);
  auto v1 = c.options;
  v1.format = neat::SimaBEVPoolPlanFormat::V1;
  const auto expected = ref::plan_cell_points(ref::bev_pool_plan(v1, c.calibration));
  auto actual = ref::plan_cell_points(plan);
  bt::require(word(plan, 1) == 2 && word(plan, 11) == 4 && word(plan, 13) == 8 &&
                  word(plan, 14) == 8,
              "v2 header: version, workers and work tile");
  bt::require(actual.size() == expected.size(), "v2 cell count");
  for (std::size_t cell = 0; cell < actual.size(); ++cell) {
    auto sorted = actual[cell];
    std::sort(sorted.begin(), sorted.end());
    bt::require(sorted == expected[cell], "v2 cell " + std::to_string(cell) + " points differ");
  }
}

void retained_outputs() {
  for (const bool cache : {false, true}) {
    auto c = ref::bev_pool_plan_case();
    c.options.cache_plan = cache;
    std::vector<ref::Calibration> calibrations;
    std::vector<neat::TensorList> frames;
    // Includes repeated keys and return to an old key while its output is held.
    for (const float shift : {0.0f, 0.0f, 1.0f, 2.0f, 2.0f, 0.0f, -1.0f, 0.0f}) {
      calibrations.push_back(ref::calibration_moved(c.calibration, shift));
      frames.push_back(bt::calibration_tensors(calibrations.back()));
    }
    bt::execute_retained(neat::nodes::SimaBEVPoolPlan(c.options), frames,
                         [&](const neat::TensorList& output, std::size_t frame) {
                           bt::require(output.size() == 1, "retained plan output arity");
                           const auto expected = ref::bev_pool_plan(c.options, calibrations[frame]);
                           bt::expect_bytes(used(output[0], expected.size(), "retained plan"),
                                            expected, "retained plan " + std::to_string(frame));
                         });
  }
}

void shared_offset_view() {
  const auto c = ref::bev_pool_plan_case();
  std::vector<bt::SharedCarrier> carriers(3);
  std::vector<neat::TensorList> frames;
  std::vector<ref::Calibration> calibrations;
  for (int generation = 0; generation < 3; ++generation) {
    const auto calibration = ref::calibration_moved(c.calibration, float(generation));
    const auto values = bt::calibration_values(calibration);
    auto& carrier = carriers[generation];
    carrier.add(std::vector<std::uint8_t>(64, 0x5a));
    std::vector<std::size_t> offsets;
    for (const auto& v : values)
      offsets.push_back(carrier.add(v));
    carrier.seal();
    frames.push_back(
        {carrier.view(offsets[0], neat::TensorDType::Float32, {2, 4, 4}, "camera2lidar"),
         carrier.view(offsets[1], neat::TensorDType::Float32, {2, 4, 4}, "camera_intrinsics"),
         carrier.view(offsets[2], neat::TensorDType::Float32, {2, 4, 4}, "img_aug_matrix"),
         carrier.view(offsets[3], neat::TensorDType::Float32, {4, 4}, "lidar_aug_matrix")});
    calibrations.push_back(calibration);
  }
  const auto outputs = bt::execute(neat::nodes::SimaBEVPoolPlan(c.options), frames);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto expected = ref::bev_pool_plan(c.options, calibrations[i]);
    bt::expect_bytes(used(outputs[i][0], expected.size(), "shared view"), expected,
                     "shared view " + std::to_string(i));
    carriers[i].check();
  }
}

void rejections() {
  const auto c = ref::bev_pool_plan_case();
  const auto node = [](neat::SimaBEVPoolPlanOptions o) { return neat::nodes::SimaBEVPoolPlan(o); };
  auto small = c.options;
  small.plan_capacity_bytes = kWorstCase - 1;
  bt::expect_rejected(node(small), bt::calibration_tensors(c.calibration),
                      "bev_pool_plan admission failed: outputs[0].shape: BEVPoolPlan plan must be "
                      "[P] with room for a worst-case plan",
                      "capacity below the v1 worst case");
  auto no_channels = c.options;
  no_channels.format = neat::SimaBEVPoolPlanFormat::V2;
  bt::expect_rejected(node(no_channels), bt::calibration_tensors(c.calibration),
                      "bev_pool_plan admission failed: params.context_channels: BEVPoolPlan format "
                      "2 needs 1..1024 context channels",
                      "format 2 without context_channels");
  auto ev74 = c.options;
  ev74.target = neat::SimaKernelTarget::EV74;
  bt::expect_rejected(node(ev74), bt::calibration_tensors(c.calibration),
                      "bev_pool_plan admission failed: target: requested implementation is not "
                      "present in this Kernels package",
                      "EV74 target: no fallback to A65");
  auto mismatched = bt::calibration_tensors(c.calibration);
  mismatched[1] = bt::tensor(std::vector<float>(48, 0.0f), {3, 4, 4}, "camera_intrinsics");
  bt::expect_rejected(node(c.options), mismatched, "camera_intrinsics must be [N,4,4]",
                      "camera count mismatch");
  auto missing = bt::calibration_tensors(c.calibration);
  missing.pop_back();
  bt::expect_rejected(node(c.options), missing, "expects 4 input tensors", "missing lidar_aug");
  auto bad = c.options;
  bad.format = static_cast<neat::SimaBEVPoolPlanFormat>(3);
  bt::expect_invalid_options([&] { neat::nodes::SimaBEVPoolPlan(bad); }, "format must be V1 or V2",
                             "unknown plan format");
}
} // namespace

int main(int argc, char**) {
  return bt::main_test(argc, [] {
    bt::Suite suite("SimaBEVPoolPlan",
                    {"v1_reference_plan", "cache_plan_hit_and_miss", "cache_plan_disabled",
                     "calibration_sequence", "bf16_input_dtype", "explicit_capacity",
                     "format2_int8_points", "format2_bf16_points", "retained_outputs_in_flight",
                     "shared_offset_view", "rejects_invalid_requests"});
    suite.run("v1_reference_plan", [] {
      const auto c = ref::bev_pool_plan_case();
      run_sequence(c.options, {c.calibration}, kWorstCase, "v1 plan");
    });
    suite.run("cache_plan_hit_and_miss", [] { cache(true); });
    suite.run("cache_plan_disabled", [] { cache(false); });
    suite.run("calibration_sequence", calibration_sequence);
    suite.run("bf16_input_dtype", bf16_input_dtype);
    suite.run("explicit_capacity", explicit_capacity);
    suite.run("format2_int8_points", [] { format_2(neat::TensorDType::Int8); });
    suite.run("format2_bf16_points", [] { format_2(neat::TensorDType::BFloat16); });
    suite.run("retained_outputs_in_flight", retained_outputs);
    suite.run("shared_offset_view", shared_offset_view);
    suite.run("rejects_invalid_requests", rejections);
    suite.finish();
  });
}

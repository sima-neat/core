// SimaVoxelize hardware suite: the public Graph/Node/Run API on the target this
// executable was built for (sima_voxelize_ev_test or sima_voxelize_a65_test),
// compared byte for byte with the independent oracle.
#include "../support/bevfusion_kernel_test_utils.h"

namespace bt = bevfusion_test;
namespace neat = bt::neat;
namespace ref = bt::ref;

namespace {
struct Frame {
  std::vector<float> points;
  std::int32_t count = 0;
};

neat::TensorList inputs(const Frame& frame) {
  const auto n = static_cast<std::int64_t>(frame.points.size() / 4);
  return {bt::tensor(frame.points, {n, 4}, "points"), bt::count(frame.count, "point_count")};
}

void check(const neat::TensorList& output, const neat::SimaVoxelizeOptions& options,
           const Frame& frame, const std::string& what) {
  bt::require(output.size() == 3, what + ": decorated, coordinates and count");
  const auto expected = ref::voxelize(options, frame.points, frame.count);
  const std::int64_t m = options.max_pillars, t = options.max_points_per_pillar;
  if (options.output_dtype == neat::TensorDType::BFloat16)
    bt::expect_bytes(
        bt::values<std::uint16_t>(output[0], neat::TensorDType::BFloat16, {1, m, t, 10}, what),
        expected.decorated_bf16, what + " decorated");
  else
    bt::expect_bytes(
        bt::values<std::int8_t>(output[0], neat::TensorDType::Int8, {1, m, t, 10}, what),
        expected.decorated, what + " decorated");
  bt::expect_bytes(bt::values<std::int32_t>(output[1], neat::TensorDType::Int32, {m, 2}, what),
                   expected.coordinates, what + " coordinates");
  bt::expect_bytes(bt::values<std::int32_t>(output[2], neat::TensorDType::Int32, {1}, what),
                   std::vector<std::int32_t>{expected.count}, what + " count");
}

void run_frames(const neat::SimaVoxelizeOptions& options, const std::vector<Frame>& frames,
                const std::string& what) {
  std::vector<neat::TensorList> tensors;
  for (const auto& frame : frames)
    tensors.push_back(inputs(frame));
  const auto outputs = bt::execute(neat::nodes::SimaVoxelize(options), tensors);
  bt::require(outputs.size() == frames.size(), what + ": one output per push");
  for (std::size_t i = 0; i < frames.size(); ++i)
    check(outputs[i], options, frames[i], what + " frame " + std::to_string(i));
}

Frame fixture(const ref::VoxelizeCase& c) {
  return {c.points, c.point_count};
}

Frame shifted(const ref::VoxelizeCase& c, float dx, float dy, std::int32_t count) {
  return {ref::voxelize_shifted(c, dx, dy), count};
}

void rounding_modes() {
  for (const auto mode :
       {neat::SimaQuantRounding::NearestEven, neat::SimaQuantRounding::TowardZero,
        neat::SimaQuantRounding::TowardPositive, neat::SimaQuantRounding::TowardNegative}) {
    auto c = ref::voxelize_case(bt::suite_target());
    // A multiplier that leaves fractional products, so the modes differ.
    c.options.quantization = {2.5f, 1, mode};
    run_frames(c.options, {fixture(c)}, "rounding mode " + std::to_string(int(mode)));
  }
}

void pillar_capacity() {
  auto c = ref::voxelize_case(bt::suite_target());
  for (const int pillars : {1, 2, 3, 4}) {
    c.options.max_pillars = pillars;
    run_frames(c.options, {fixture(c)}, "max_pillars " + std::to_string(pillars));
  }
}

void point_capacity() {
  // Sixteen points in one cell against capacities of 8 (half dropped) and 16.
  auto c = ref::voxelize_case(bt::suite_target());
  const Frame dense{ref::voxelize_dense_cell(), 16};
  for (const int points : {8, 16}) {
    c.options.max_points_per_pillar = points;
    run_frames(c.options, {dense}, "max_points_per_pillar " + std::to_string(points));
  }
}

void empty_frame() {
  for (const auto dtype : {neat::TensorDType::Int8, neat::TensorDType::BFloat16}) {
    auto c = dtype == neat::TensorDType::Int8 ? ref::voxelize_case(bt::suite_target())
                                              : ref::voxelize_case_bf16(bt::suite_target());
    // Empty, full, empty: every output is fully rewritten each frame.
    run_frames(c.options, {{c.points, 0}, fixture(c), {c.points, 0}},
               dtype == neat::TensorDType::Int8 ? "INT8 empty frame" : "BF16 empty frame");
  }
}

void out_of_range() {
  auto c = ref::voxelize_case(bt::suite_target());
  run_frames(c.options, {shifted(c, 40.0f, 0.0f, c.point_count)}, "all points outside");
}

void reuse() {
  for (const auto dtype : {neat::TensorDType::Int8, neat::TensorDType::BFloat16}) {
    auto c = dtype == neat::TensorDType::Int8 ? ref::voxelize_case(bt::suite_target())
                                              : ref::voxelize_case_bf16(bt::suite_target());
    run_frames(c.options,
               {fixture(c), shifted(c, 1.0f, 0.0f, 5), shifted(c, 0.0f, -1.0f, 0),
                shifted(c, -1.0f, 1.0f, c.point_count), fixture(c)},
               dtype == neat::TensorDType::Int8 ? "INT8 reuse" : "BF16 reuse");
  }
}

void retained_outputs() {
  for (const bool wide : {false, true}) {
    const auto c =
        wide ? ref::voxelize_case_bf16(bt::suite_target()) : ref::voxelize_case(bt::suite_target());
    std::vector<Frame> expected;
    std::vector<neat::TensorList> frames;
    for (int generation = 0; generation < 8; ++generation) {
      expected.push_back(
          shifted(c, float(generation % 3), 0.0f, generation % 3 == 1 ? 0 : c.point_count));
      frames.push_back(inputs(expected.back()));
    }
    bt::execute_retained(neat::nodes::SimaVoxelize(c.options), frames,
                         [&](const neat::TensorList& output, std::size_t frame) {
                           check(output, c.options, expected[frame],
                                 "retained voxelize " + std::to_string(frame));
                         });
  }
}

void shared_offset_view() {
  const auto c = ref::voxelize_case(bt::suite_target());
  const auto n = static_cast<std::int64_t>(c.points.size() / 4);
  std::vector<bt::SharedCarrier> carriers(3);
  std::vector<neat::TensorList> frames;
  std::vector<Frame> expected;
  for (int generation = 0; generation < 3; ++generation) {
    auto frame = shifted(c, float(generation), 0.0f, c.point_count);
    auto& carrier = carriers[generation];
    carrier.add(std::vector<std::uint8_t>(64, 0x5a));
    const auto points = carrier.add(frame.points);
    const auto count = carrier.add(std::vector<std::int32_t>{frame.count});
    carrier.seal();
    frames.push_back({carrier.view(points, neat::TensorDType::Float32, {n, 4}, "points"),
                      carrier.view(count, neat::TensorDType::Int32, {1}, "point_count")});
    expected.push_back(frame);
  }
  const auto outputs = bt::execute(neat::nodes::SimaVoxelize(c.options), frames);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    check(outputs[i], c.options, expected[i], "shared view " + std::to_string(i));
    carriers[i].check();
  }
}

void rejections() {
  const auto c = ref::voxelize_case(bt::suite_target());
  const auto n = static_cast<std::int64_t>(c.points.size() / 4);
  const auto node = [&] { return neat::nodes::SimaVoxelize(c.options); };
  std::vector<float> three(c.points.begin(), c.points.begin() + 3 * n);
  bt::expect_rejected(node(), {bt::tensor(three, {n, 3}, "points"), bt::count(c.point_count)},
                      "points must be [N,4]", "points [N,3]");
  std::vector<std::int32_t> integers(c.points.size(), 1);
  bt::expect_rejected(node(), {bt::tensor(integers, {n, 4}, "points"), bt::count(c.point_count)},
                      "points must be FP32", "INT32 points");
  bt::expect_rejected(node(), {bt::tensor(c.points, {n, 4}, "points")}, "expects 2 input tensors",
                      "missing point_count");
  bt::expect_rejected(node(),
                      {bt::tensor(c.points, {n, 4}, "points"),
                       bt::tensor(std::vector<std::int32_t>{1, 2}, {2}, "point_count")},
                      "point_count must hold one INT32 value", "two-element point_count");
  auto bf16 = ref::voxelize_case_bf16(bt::suite_target()).options;
  bf16.quantization.multiplier = 2.0f;
  bt::expect_rejected(neat::nodes::SimaVoxelize(bf16), inputs(fixture(c)),
                      "voxelize admission failed: params.quantization: Voxelize BFloat16 "
                      "output takes only the identity quantization",
                      "BF16 with a non-identity quantization");
  auto empty = c.options;
  empty.max_pillars = 0;
  bt::expect_invalid_options([&] { neat::nodes::SimaVoxelize(empty); }, "max_pillars",
                             "zero pillar capacity");
}
} // namespace

int main(int argc, char**) {
  return bt::main_test(argc, [] {
    bt::Suite suite("SimaVoxelize",
                    {"int8_reference_sweep", "int8_rounding_modes", "bf16_reference_sweep",
                     "pillar_capacity", "point_capacity", "empty_frame", "points_out_of_range",
                     "reuse_across_pushes", "retained_outputs_in_flight", "shared_offset_view",
                     "rejects_invalid_requests"});
    suite.run("int8_reference_sweep", [] {
      const auto c = ref::voxelize_case(bt::suite_target());
      run_frames(c.options, {fixture(c)}, "INT8 sweep");
    });
    suite.run("int8_rounding_modes", rounding_modes);
    suite.run("bf16_reference_sweep", [] {
      const auto c = ref::voxelize_case_bf16(bt::suite_target());
      run_frames(c.options, {fixture(c)}, "BF16 sweep");
    });
    suite.run("pillar_capacity", pillar_capacity);
    suite.run("point_capacity", point_capacity);
    suite.run("empty_frame", empty_frame);
    suite.run("points_out_of_range", out_of_range);
    suite.run("reuse_across_pushes", reuse);
    suite.run("retained_outputs_in_flight", retained_outputs);
    suite.run("shared_offset_view", shared_offset_view);
    suite.run("rejects_invalid_requests", rejections);
    suite.finish();
  });
}

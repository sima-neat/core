// SimaBEVPool fed by a connected SimaBEVPoolPlan (core#1528, point 1): a camera
// fragment and an A65 plan fragment joined with graphs::Combine into the pool,
// through the public Graph/Node/Run API. With plan_format unset the pool takes
// the joined plan's header format; an explicit format that the header
// contradicts is a kernel-request error naming both formats.
#include "../support/bevfusion_kernel_test_utils.h"
#include "graphs/Fragments.h"

namespace bt = bevfusion_test;
namespace neat = bt::neat;
namespace ref = bt::ref;

namespace {
constexpr std::int64_t kFrames = 2;

struct Joined {
  ref::BEVPoolCase pool;
  ref::BEVPoolPlanCase plan;
};

// INT8 channels-last depth/context, and a format V1 plan from SimaBEVPoolPlan.
Joined joined_case() {
  Joined j{ref::bev_pool_case(bt::suite_target()), ref::bev_pool_plan_case()};
  j.plan.options.format = neat::SimaBEVPoolPlanFormat::V1;
  j.plan.options.context_channels = static_cast<int>(j.pool.channels);
  j.plan.options.input_dtype = neat::TensorDType::Int8;
  // The oracle pools through the same plan bytes the joined graph produces.
  j.pool.plan = bt::plan_from_node(j.plan.options, j.plan.calibration);
  j.pool.options.plan_format.reset();
  return j;
}

neat::Sample frame_sample(neat::TensorList tensors, std::int64_t frame, const std::string& stream) {
  neat::Sample sample;
  sample.kind = neat::SampleKind::TensorSet;
  sample.tensors = std::move(tensors);
  sample.frame_id = frame;
  sample.stream_id = stream;
  return sample;
}

neat::Sample camera_sample(const ref::BEVPoolCase& c, std::int64_t frame) {
  const std::vector<std::int64_t> depth_shape{c.cameras, c.rows, c.columns, c.depth_bins};
  const std::vector<std::int64_t> context_shape{c.cameras, c.rows, c.columns, c.channels};
  return frame_sample(
      {bt::tensor(ref::to_channels_last(c.depth, c.cameras, c.depth_bins, c.rows, c.columns),
                  depth_shape, "depth"),
       bt::tensor(ref::to_channels_last(c.context, c.cameras, c.channels, c.rows, c.columns),
                  context_shape, "context")},
      frame, "camera");
}

// camera ─┐
//         ├─ Combine(ByFrame) ─ pool_in ─ SimaBEVPool ─ bev
// calibration ─ SimaBEVPoolPlan ─ plan ─┘
neat::Graph joined_graph(const Joined& j, const neat::SimaBEVPoolOptions& pool_options) {
  neat::Graph camera("camera");
  camera.add(neat::nodes::Input("camera", bt::input_options()));
  neat::Graph plan("plan");
  plan.add(neat::nodes::Input("calibration", bt::input_options()));
  plan.add(neat::nodes::SimaBEVPoolPlan(j.plan.options));
  plan.add(neat::nodes::Output("plan"));
  auto join = neat::graphs::Combine({"camera", "plan"}, "pool_in", neat::CombinePolicy::ByFrame);
  neat::Graph pool("pool");
  pool.add(neat::nodes::Input("pool_in", bt::input_options()));
  pool.add(neat::nodes::SimaBEVPool(pool_options));
  pool.add(neat::nodes::Output("bev"));
  neat::Graph app;
  app.connect(camera, join);
  app.connect(plan, join);
  app.connect(join, pool);
  return app;
}

void push_frame(neat::Run& run, const Joined& j, std::int64_t frame) {
  bt::require(run.push("camera", camera_sample(j.pool, frame)),
              "camera push failed: " + run.last_error());
  bt::require(run.push("calibration",
                       frame_sample(bt::calibration_tensors(j.plan.calibration), frame, "plan")),
              "calibration push failed: " + run.last_error());
}

void check(const neat::Sample& output, const ref::BEVPoolCase& c, const std::string& what) {
  const auto tensors = neat::tensors_from_sample(output);
  bt::require(tensors.size() == 1, what + ": one bev");
  const auto& o = c.options;
  const std::vector<std::int64_t> shape{1, o.cells[2] * c.channels, o.cells[0], o.cells[1]};
  bt::expect_bytes(bt::values<std::int8_t>(tensors[0], neat::TensorDType::Int8, shape, what),
                   ref::bev_pool(c), what + " INT8 bev");
}

void inherits_v1_from_the_joined_plan() {
  const auto j = joined_case();
  auto app = joined_graph(j, j.pool.options);
  auto run = app.build(bt::run_options());
  try {
    for (std::int64_t frame = 0; frame < kFrames; ++frame) {
      push_frame(run, j, frame);
      neat::Sample output;
      neat::PullError error;
      const auto status = run.pull("bev", bt::kTimeoutMs, output, &error);
      bt::require(status == neat::PullStatus::Ok, "joined plan frame " + std::to_string(frame) +
                                                      " pull failed: [" + error.code + "] " +
                                                      error.message + " " + run.last_error());
      bt::require(output.frame_id == frame, "joined bev lost its frame_id");
      check(output, j.pool, "joined V1 plan frame " + std::to_string(frame));
    }
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
}

// The joined segment builds on its first sample, so the refusal can surface at
// build, push or pull; it must be the kernel-request error either way.
void refuses_an_explicit_conflicting_format() {
  const auto j = joined_case();
  auto options = j.pool.options;
  options.plan_format = neat::SimaBEVPoolPlanFormat::V2;
  std::string failure;
  std::optional<neat::Run> run;
  try {
    run.emplace(joined_graph(j, options).build(bt::run_options()));
    push_frame(*run, j, 0);
    neat::Sample output;
    neat::PullError error;
    const auto status = run->pull("bev", bt::kTimeoutMs, output, &error);
    bt::require(status != neat::PullStatus::Ok, "a V1 plan joined into plan_format V2 pooled");
    failure = "[" + error.code + "] " + error.message + " " + run->last_error();
  } catch (const bt::Failure&) {
    if (run)
      run->close();
    throw;
  } catch (const std::exception& e) {
    failure = e.what();
    if (run)
      failure += std::string(" ") + run->last_error();
  }
  if (run)
    run->close();
  for (const std::string& needle :
       {std::string(neat::error_codes::kKernelRequest), std::string("plan_format is V2"),
        std::string("the plan header records format V1")})
    bt::require(failure.find(needle) != std::string::npos,
                "joined conflict without '" + needle + "': " + failure);
}
} // namespace

int main(int argc, char**) {
  return bt::main_test(argc, [] {
    bt::Suite suite("SimaBEVPool joined plan",
                    {"inherits_v1_from_the_joined_plan", "refuses_an_explicit_conflicting_format"});
    suite.run("inherits_v1_from_the_joined_plan", inherits_v1_from_the_joined_plan);
    suite.run("refuses_an_explicit_conflicting_format", refuses_an_explicit_conflicting_format);
    suite.finish();
  });
}

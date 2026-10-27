// SimaBEVPool hardware suite: the public Graph/Node/Run API on the target this
// executable was built for (sima_bev_pool_ev_test or sima_bev_pool_a65_test),
// compared byte for byte with the oracle. Plans come from the reference plan
// builder or from SimaBEVPoolPlan's own output, v1 and v2.
#include "../support/bevfusion_kernel_test_utils.h"

namespace bt = bevfusion_test;
namespace neat = bt::neat;
namespace ref = bt::ref;

namespace {
using Storage = neat::SimaBEVPoolStorage;

bool wide(const ref::BEVPoolCase& c) {
  return c.bf16;
}

neat::TensorList inputs(const ref::BEVPoolCase& c) {
  const bool cl = c.options.input_storage == Storage::ChannelsLast;
  const auto depth_shape =
      cl ? std::vector<std::int64_t>{c.cameras, c.rows, c.columns, c.depth_bins}
         : std::vector<std::int64_t>{c.cameras, c.depth_bins, c.rows, c.columns};
  const auto context_shape =
      cl ? std::vector<std::int64_t>{c.cameras, c.rows, c.columns, c.channels}
         : std::vector<std::int64_t>{c.cameras, c.channels, c.rows, c.columns};
  const auto plan = bt::tensor(c.plan, {static_cast<std::int64_t>(c.plan.size())}, "plan");
  if (wide(c)) {
    const auto depth =
        cl ? ref::to_channels_last(c.depth_bf16, c.cameras, c.depth_bins, c.rows, c.columns)
           : c.depth_bf16;
    const auto context =
        cl ? ref::to_channels_last(c.context_bf16, c.cameras, c.channels, c.rows, c.columns)
           : c.context_bf16;
    return {bt::bf16(depth, depth_shape, "depth"), bt::bf16(context, context_shape, "context"),
            plan};
  }
  const auto depth =
      cl ? ref::to_channels_last(c.depth, c.cameras, c.depth_bins, c.rows, c.columns) : c.depth;
  const auto context =
      cl ? ref::to_channels_last(c.context, c.cameras, c.channels, c.rows, c.columns) : c.context;
  return {bt::tensor(depth, depth_shape, "depth"), bt::tensor(context, context_shape, "context"),
          plan};
}

void check(const neat::TensorList& output, const ref::BEVPoolCase& c, const std::string& what) {
  bt::require(output.size() == 1, what + ": one bev");
  const auto& o = c.options;
  // The public bev is logical [1,nZ*C,nX,nY], whatever its byte order.
  const std::vector<std::int64_t> shape{1, o.cells[2] * c.channels, o.cells[0], o.cells[1]};
  if (o.output_dtype == neat::TensorDType::BFloat16)
    bt::expect_bytes(bt::values<std::uint16_t>(output[0], neat::TensorDType::BFloat16, shape, what),
                     ref::bev_pool_bf16(c), what + " BF16 bev");
  else
    bt::expect_bytes(bt::values<std::int8_t>(output[0], neat::TensorDType::Int8, shape, what),
                     wide(c) ? ref::bev_pool_bf16_to_int8(c) : ref::bev_pool(c),
                     what + " INT8 bev");
}

void run_cases(const std::vector<ref::BEVPoolCase>& cases, const std::string& what) {
  std::vector<neat::TensorList> frames;
  for (const auto& c : cases)
    frames.push_back(inputs(c));
  const auto outputs = bt::execute(neat::nodes::SimaBEVPool(cases.front().options), frames);
  bt::require(outputs.size() == cases.size(), what + ": one bev per push");
  for (std::size_t i = 0; i < cases.size(); ++i)
    check(outputs[i], cases[i], what + " frame " + std::to_string(i));
}

ref::BEVPoolCase int8_case(Storage in = Storage::ChannelsLast,
                           Storage out = Storage::ChannelsLast) {
  auto c = ref::bev_pool_case(bt::suite_target());
  c.options.input_storage = in;
  c.options.output_storage = out;
  return c;
}
ref::BEVPoolCase bf16_case(neat::TensorDType output, Storage storage = Storage::ChannelsLast) {
  auto c = ref::bev_pool_case_bf16(bt::suite_target(), output);
  c.options.input_storage = c.options.output_storage = storage;
  return c;
}

// The same depth/context with SimaBEVPoolPlan's own plan, format 1 or 2.
ref::BEVPoolCase with_node_plan(ref::BEVPoolCase c, neat::SimaBEVPoolPlanFormat format) {
  auto p = ref::bev_pool_plan_case();
  p.options.format = format;
  p.options.context_channels = static_cast<int>(c.channels);
  p.options.input_dtype = wide(c) ? neat::TensorDType::BFloat16 : neat::TensorDType::Int8;
  c.plan = bt::plan_from_node(p.options, p.calibration);
  c.options.plan_format = format;
  return c;
}

std::vector<ref::BEVPoolCase> sequence(const ref::BEVPoolCase& base) {
  std::vector<ref::BEVPoolCase> result;
  for (int generation = 0; generation < 3; ++generation)
    result.push_back(ref::bev_pool_generation(base, generation));
  return result;
}

void reuse() {
  run_cases(sequence(int8_case()), "INT8 reuse");
  run_cases(sequence(bf16_case(neat::TensorDType::BFloat16)), "BF16 reuse");
  run_cases(sequence(bf16_case(neat::TensorDType::Int8)), "BF16->INT8 reuse");
}

void shared_offset_view() {
  for (const bool bf16 : {false, true}) {
    const auto c = bf16 ? bf16_case(neat::TensorDType::BFloat16) : int8_case();
    const auto cases = sequence(c);
    std::vector<bt::SharedCarrier> carriers(cases.size());
    std::vector<neat::TensorList> frames;
    for (std::size_t g = 0; g < cases.size(); ++g) {
      const auto& k = cases[g];
      const auto dtype = bf16 ? neat::TensorDType::BFloat16 : neat::TensorDType::Int8;
      auto& carrier = carriers[g];
      carrier.add(std::vector<std::uint8_t>(64, 0x5a));
      const auto depth = bf16 ? carrier.add(ref::to_channels_last(k.depth_bf16, k.cameras,
                                                                  k.depth_bins, k.rows, k.columns))
                              : carrier.add(ref::to_channels_last(k.depth, k.cameras, k.depth_bins,
                                                                  k.rows, k.columns));
      const auto context = bf16 ? carrier.add(ref::to_channels_last(k.context_bf16, k.cameras,
                                                                    k.channels, k.rows, k.columns))
                                : carrier.add(ref::to_channels_last(k.context, k.cameras,
                                                                    k.channels, k.rows, k.columns));
      const auto plan = carrier.add(k.plan);
      carrier.seal();
      frames.push_back(
          {carrier.view(depth, dtype, {k.cameras, k.rows, k.columns, k.depth_bins}, "depth"),
           carrier.view(context, dtype, {k.cameras, k.rows, k.columns, k.channels}, "context"),
           carrier.view(plan, neat::TensorDType::UInt8, {static_cast<std::int64_t>(k.plan.size())},
                        "plan")});
    }
    const auto outputs = bt::execute(neat::nodes::SimaBEVPool(c.options), frames);
    for (std::size_t g = 0; g < outputs.size(); ++g) {
      check(outputs[g], cases[g],
            std::string(bf16 ? "BF16" : "INT8") + " shared view " + std::to_string(g));
      carriers[g].check();
    }
  }
}

void rejections() {
  const auto c = int8_case();
  const auto node = [](neat::SimaBEVPoolOptions o) { return neat::nodes::SimaBEVPool(o); };
  auto to_bf16 = c;
  to_bf16.options.output_dtype = neat::TensorDType::BFloat16;
  to_bf16.options.quantization = {};
  bt::expect_rejected(node(to_bf16.options), inputs(c),
                      "bev_pool admission failed: outputs[0]: BEVPool bev is BFloat16 only for "
                      "BFloat16 depth and context",
                      "INT8 inputs with a BF16 bev");
  auto zp = bf16_case(neat::TensorDType::BFloat16);
  zp.options.context_zero_point = -5;
  bt::expect_rejected(node(zp.options), inputs(bf16_case(neat::TensorDType::BFloat16)),
                      "bev_pool admission failed: params.context_zero_point: BEVPool BFloat16 "
                      "inputs take zero points of 0",
                      "BF16 with a nonzero zero point");
  auto short_plan = c;
  short_plan.plan.resize(c.plan.size() - 4);
  bt::expect_rejected(node(c.options), inputs(short_plan),
                      "bev_pool admission failed: inputs[2].shape: BEVPool plan is smaller than a "
                      "worst-case plan",
                      "plan below the v1 worst case");
  auto plan_dtype = inputs(c);
  plan_dtype[2] = bt::tensor(std::vector<std::int8_t>(c.plan.size()),
                             {static_cast<std::int64_t>(c.plan.size())}, "plan");
  bt::expect_rejected(node(c.options), plan_dtype, "plan must be UINT8", "INT8 plan tensor");
  auto missing = inputs(c);
  missing.pop_back();
  bt::expect_rejected(node(c.options), missing, "expects 3 input tensors", "missing plan");
  if (bt::on_ev74()) {
    // EV74 admits only channels-last context and bev; there is no A65 fallback.
    const auto contiguous = int8_case(Storage::Contiguous, Storage::Contiguous);
    bt::expect_rejected(node(contiguous.options), inputs(contiguous),
                        "bev_pool admission failed: inputs[1].layout: BEVPool on EV74 requires "
                        "dense channels-last context",
                        "EV74 contiguous context and bev");
  }
  auto empty = c.options;
  empty.cells[1] = 0;
  bt::expect_invalid_options([&] { neat::nodes::SimaBEVPool(empty); }, "every cell count",
                             "empty grid");
}

std::vector<std::string> scenarios() {
  std::vector<std::string> names{"int8_v1_channels_last"};
  if (!bt::on_ev74()) // EV74 admits channels-last context and bev only
    names.insert(names.end(), {"int8_v1_contiguous", "int8_v1_contiguous_to_channels_last",
                               "int8_v1_channels_last_to_contiguous"});
  names.insert(names.end(), {"int8_two_height_cells", "int8_plan_node_v1", "int8_plan_node_v2",
                             "bf16_to_bf16_v1", "bf16_to_int8_v1", "bf16_accumulation_order",
                             "bf16_plan_node_v2", "bf16_to_int8_plan_node_v2"});
  if (!bt::on_ev74())
    names.push_back("bf16_contiguous");
  names.insert(names.end(),
               {"reuse_across_pushes", "shared_offset_view", "rejects_invalid_requests"});
  return names;
}
} // namespace

int main(int argc, char**) {
  return bt::main_test(argc, [] {
    bt::Suite suite("SimaBEVPool", scenarios());
    suite.run("int8_v1_channels_last", [] { run_cases({int8_case()}, "INT8 channels-last"); });
    if (!bt::on_ev74()) {
      suite.run("int8_v1_contiguous", [] {
        run_cases({int8_case(Storage::Contiguous, Storage::Contiguous)}, "INT8 contiguous");
      });
      suite.run("int8_v1_contiguous_to_channels_last", [] {
        run_cases({int8_case(Storage::Contiguous, Storage::ChannelsLast)}, "INT8 NCHW->NHWC");
      });
      suite.run("int8_v1_channels_last_to_contiguous", [] {
        run_cases({int8_case(Storage::ChannelsLast, Storage::Contiguous)}, "INT8 NHWC->NCHW");
      });
    }
    suite.run("int8_two_height_cells", [] {
      auto c = ref::bev_pool_case_two_heights(bt::suite_target());
      run_cases({c}, "nZ = 2");
    });
    suite.run("int8_plan_node_v1", [] {
      run_cases({with_node_plan(int8_case(), neat::SimaBEVPoolPlanFormat::V1)},
                "INT8 node plan v1");
    });
    suite.run("int8_plan_node_v2", [] {
      run_cases({with_node_plan(int8_case(), neat::SimaBEVPoolPlanFormat::V2)},
                "INT8 node plan v2");
    });
    suite.run("bf16_to_bf16_v1",
              [] { run_cases({bf16_case(neat::TensorDType::BFloat16)}, "BF16->BF16"); });
    suite.run("bf16_to_int8_v1",
              [] { run_cases({bf16_case(neat::TensorDType::Int8)}, "BF16->INT8"); });
    suite.run("bf16_accumulation_order", [] {
      run_cases({ref::bev_pool_case_bf16_accumulation(bt::suite_target())}, "12-term BF16 cell");
    });
    suite.run("bf16_plan_node_v2", [] {
      run_cases(
          {with_node_plan(bf16_case(neat::TensorDType::BFloat16), neat::SimaBEVPoolPlanFormat::V2)},
          "BF16 node plan v2");
    });
    suite.run("bf16_to_int8_plan_node_v2", [] {
      run_cases(
          {with_node_plan(bf16_case(neat::TensorDType::Int8), neat::SimaBEVPoolPlanFormat::V2)},
          "BF16->INT8 node plan v2");
    });
    if (!bt::on_ev74())
      suite.run("bf16_contiguous", [] {
        run_cases({bf16_case(neat::TensorDType::BFloat16, Storage::Contiguous)}, "BF16 contiguous");
      });
    suite.run("reuse_across_pushes", reuse);
    suite.run("shared_offset_view", shared_offset_view);
    suite.run("rejects_invalid_requests", rejections);
    suite.finish();
  });
}

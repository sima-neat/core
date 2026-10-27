// SimaPillarScatter hardware suite: the public Graph/Node/Run API on the target
// this executable was built for (sima_pillar_scatter_ev_test or
// sima_pillar_scatter_a65_test), compared byte for byte with the oracle.
#include "../support/bevfusion_kernel_test_utils.h"

namespace bt = bevfusion_test;
namespace neat = bt::neat;
namespace ref = bt::ref;

namespace {
// One push: features in INT8 codes or BF16 bits (per `wide`), coordinates, count.
struct Frame {
  bool wide = false;
  std::vector<std::int8_t> features;
  std::vector<std::uint16_t> features_bf16;
  std::vector<std::int32_t> coordinates;
  std::int32_t count = 0;
  std::vector<std::int64_t> feature_shape; // empty: [M,C]
};

Frame frame(const ref::PillarScatterCase& c) {
  return {!c.features_bf16.empty(), c.features, c.features_bf16, c.coordinates, c.count, {}};
}

neat::TensorList inputs(const Frame& f, std::int64_t pillars, std::int64_t channels) {
  auto shape =
      f.feature_shape.empty() ? std::vector<std::int64_t>{pillars, channels} : f.feature_shape;
  return {f.wide ? bt::bf16(f.features_bf16, shape, "features")
                 : bt::tensor(f.features, shape, "features"),
          bt::tensor(f.coordinates, {pillars, 2}, "coordinates"), bt::count(f.count)};
}

void check(const neat::TensorList& output, const neat::SimaPillarScatterOptions& options,
           const Frame& f, std::size_t channels, const std::string& what) {
  bt::require(output.size() == 1, what + ": one canvas");
  const std::vector<std::int64_t> shape{1, options.rows, options.columns,
                                        static_cast<std::int64_t>(channels)};
  // The public canvas is the logical [1,H,W,C] tensor stored as one MlaTiled
  // tile. INT8 tile bytes are dense NHWC; a BF16 cell holds its channels' low
  // bytes, then their high bytes, so it is compared as that byte stream.
  if (f.wide) {
    bt::expect_bytes(
        bt::stored_bytes(output[0], neat::TensorDType::BFloat16, shape, what),
        ref::pillar_scatter_bf16_bytes(options, f.features_bf16, f.coordinates, f.count, channels),
        what + " BF16 canvas");
  } else
    bt::expect_bytes(bt::values<std::int8_t>(output[0], neat::TensorDType::Int8, shape, what),
                     ref::pillar_scatter(options, f.features, f.coordinates, f.count, channels),
                     what + " INT8 canvas");
}

void run_frames(const neat::SimaPillarScatterOptions& options, const std::vector<Frame>& frames,
                std::int64_t pillars, std::int64_t channels, const std::string& what) {
  std::vector<neat::TensorList> tensors;
  for (const auto& f : frames)
    tensors.push_back(inputs(f, pillars, channels));
  const auto outputs = bt::execute(neat::nodes::SimaPillarScatter(options), tensors);
  bt::require(outputs.size() == frames.size(), what + ": one canvas per push");
  for (std::size_t i = 0; i < frames.size(); ++i)
    check(outputs[i], options, frames[i], static_cast<std::size_t>(channels),
          what + " frame " + std::to_string(i));
}

ref::PillarScatterCase fixture(bool wide) {
  return wide ? ref::pillar_scatter_case_bf16(bt::suite_target())
              : ref::pillar_scatter_case(bt::suite_target());
}

std::int64_t pillars(const ref::PillarScatterCase& c) {
  return static_cast<std::int64_t>(c.pillars);
}
std::int64_t channels(const ref::PillarScatterCase& c) {
  return static_cast<std::int64_t>(c.channels);
}

void full_canvas() {
  for (const bool wide : {false, true}) {
    const auto c = ref::pillar_scatter_full(fixture(wide));
    run_frames(c.options, {frame(c)}, pillars(c), channels(c),
               wide ? "BF16 full canvas" : "INT8 full canvas");
  }
}

void empty_canvas() {
  for (const bool wide : {false, true}) {
    const auto c = fixture(wide);
    auto empty = frame(c);
    empty.count = 0;
    // Empty after occupied: no stale pillar may survive.
    run_frames(c.options, {frame(c), empty}, pillars(c), channels(c),
               wide ? "BF16 empty canvas" : "INT8 empty canvas");
  }
}

void channels_64() {
  for (const bool wide : {false, true}) {
    const auto c = ref::pillar_scatter_channels(fixture(wide), 64);
    run_frames(c.options, {frame(c)}, pillars(c), 64, wide ? "BF16 C=64" : "INT8 C=64");
  }
}

void feature_shapes() {
  // A pillar-feature model publishes [1,M,1,C]; [M,C] and [1,M,C] are the same bytes.
  for (const bool wide : {false, true}) {
    const auto c = fixture(wide);
    for (const auto& shape : {std::vector<std::int64_t>{1, pillars(c), 1, channels(c)},
                              std::vector<std::int64_t>{1, pillars(c), channels(c)}}) {
      auto f = frame(c);
      f.feature_shape = shape;
      run_frames(c.options, {f}, pillars(c), channels(c),
                 std::string(wide ? "BF16" : "INT8") + " rank " + std::to_string(shape.size()));
    }
  }
}

void reuse() {
  for (const bool wide : {false, true}) {
    const auto c = fixture(wide);
    std::vector<Frame> frames;
    for (int generation = 0; generation < 4; ++generation)
      frames.push_back(frame(ref::pillar_scatter_generation(c, generation)));
    run_frames(c.options, frames, pillars(c), channels(c), wide ? "BF16 reuse" : "INT8 reuse");
  }
}

void shared_offset_view() {
  for (const bool wide : {false, true}) {
    const auto c = fixture(wide);
    std::vector<bt::SharedCarrier> carriers(3);
    std::vector<neat::TensorList> tensors;
    std::vector<Frame> frames;
    for (int generation = 0; generation < 3; ++generation) {
      auto f = frame(c);
      f.count = c.count - generation;
      auto& carrier = carriers[generation];
      carrier.add(std::vector<std::uint8_t>(64, 0x5a));
      const auto features = wide ? carrier.add(f.features_bf16) : carrier.add(f.features);
      const auto coordinates = carrier.add(f.coordinates);
      const auto count = carrier.add(std::vector<std::int32_t>{f.count});
      carrier.seal();
      tensors.push_back(
          {carrier.view(features, wide ? neat::TensorDType::BFloat16 : neat::TensorDType::Int8,
                        {pillars(c), channels(c)}, "features"),
           carrier.view(coordinates, neat::TensorDType::Int32, {pillars(c), 2}, "coordinates"),
           carrier.view(count, neat::TensorDType::Int32, {1}, "count")});
      frames.push_back(f);
    }
    const auto outputs = bt::execute(neat::nodes::SimaPillarScatter(c.options), tensors);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      check(outputs[i], c.options, frames[i], c.channels,
            std::string(wide ? "BF16" : "INT8") + " shared view " + std::to_string(i));
      carriers[i].check();
    }
  }
}

void rejections() {
  const auto c = fixture(false);
  const auto m = pillars(c);
  const auto node = [&](neat::SimaPillarScatterOptions o) {
    return neat::nodes::SimaPillarScatter(o);
  };
  bt::expect_rejected(node(c.options),
                      {bt::tensor(std::vector<std::int8_t>(m * 8), {m, 8}, "features"),
                       bt::tensor(c.coordinates, {m, 2}, "coordinates"), bt::count(c.count)},
                      "pillar_scatter admission failed: inputs[0].shape: PillarScatter channels "
                      "must be a multiple of 16",
                      "C = 8, not a multiple of 16");
  bt::expect_rejected(node(c.options),
                      {bt::tensor(c.features, {m, channels(c)}, "features"),
                       bt::tensor(std::vector<std::int32_t>(8, 0), {4, 2}, "coordinates"),
                       bt::count(c.count)},
                      "features hold 5 pillars but coordinates hold 4", "pillar count mismatch");
  std::vector<std::uint8_t> unsigned_features(c.features.size(), 1);
  bt::expect_rejected(node(c.options),
                      {bt::tensor(unsigned_features, {m, channels(c)}, "features"),
                       bt::tensor(c.coordinates, {m, 2}, "coordinates"), bt::count(c.count)},
                      "features must be INT8 or BF16", "UINT8 features");
  bt::expect_rejected(node(c.options),
                      {bt::tensor(c.features, {m, channels(c)}, "features"),
                       bt::tensor(c.coordinates, {m, 2}, "coordinates")},
                      "expects 3 input tensors", "missing count");
  const auto w = fixture(true);
  auto nonzero = w.options;
  nonzero.zero_point = -7;
  bt::expect_rejected(node(nonzero), inputs(frame(w), m, channels(w)),
                      "pillar_scatter admission failed: params.zero_point: PillarScatter BFloat16 "
                      "canvas requires zero_point 0",
                      "BF16 with a nonzero zero_point");
  auto empty = c.options;
  empty.rows = 0;
  bt::expect_invalid_options([&] { neat::nodes::SimaPillarScatter(empty); }, "rows and columns",
                             "empty canvas geometry");
}
} // namespace

int main(int argc, char**) {
  return bt::main_test(argc, [] {
    bt::Suite suite("SimaPillarScatter",
                    {"int8_reference_canvas", "bf16_reference_canvas", "empty_canvas",
                     "full_canvas", "channels_64", "feature_shapes", "reuse_across_pushes",
                     "shared_offset_view", "rejects_invalid_requests"});
    for (const bool wide : {false, true})
      suite.run(wide ? "bf16_reference_canvas" : "int8_reference_canvas", [wide] {
        const auto c = fixture(wide);
        run_frames(c.options, {frame(c)}, pillars(c), channels(c),
                   wide ? "BF16 canvas" : "INT8 canvas");
      });
    suite.run("empty_canvas", empty_canvas);
    suite.run("full_canvas", full_canvas);
    suite.run("channels_64", channels_64);
    suite.run("feature_shapes", feature_shapes);
    suite.run("reuse_across_pushes", reuse);
    suite.run("shared_offset_view", shared_offset_view);
    suite.run("rejects_invalid_requests", rejections);
    suite.finish();
  });
}

// Public Graph/Run integration: independent numerical results, dynamic grids,
// asynchronous submissions and producer-buffer lifetime, on the A65.
#include "../support/grid_sample_reference.h"

#include <iostream>
#include <limits>
#include <optional>

namespace gt = grid_sample_test;
namespace neat = simaai::neat;
namespace {
constexpr int timeout = 30000;

template <class Fn> void rejects_layout(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& error) {
    gt::require(std::string(error.what()).find("layout") != std::string::npos,
                "GridSample missing layout diagnostic: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("GridSample accepted incompatible logical axes/layout");
}

void layout_contract() {
  using Axis = neat::TensorAxisSemantic;
  gt::Frame frame({"layout", 3, 2, 1, 2, 3, true}, false);
  frame.features = {0, 1, 2, 3, 4, 5};
  frame.grid[0] = 1;
  frame.grid[1] = -1;
  // Reject both explicit axis orders, including an Unknown layout tag that
  // cannot encode [N,W,H,C]. Unequal H/W exposes accidental reinterpretation.
  for (std::size_t port = 0; port < 2; ++port)
    for (const auto layout : {neat::TensorLayout::Unknown, neat::TensorLayout::HWC}) {
      auto inputs = frame.tensors();
      inputs[port].layout = layout;
      inputs[port].axis_semantics = {Axis::N, Axis::W, Axis::H, Axis::C};
      auto pipeline = gt::graph(frame.options);
      rejects_layout([&] { auto rejected = pipeline.build(inputs); });
    }
  auto initial = frame.tensors();
  auto pipeline = gt::graph(frame.options);
  auto run = pipeline.build(initial);
  try {
    frame.check(run.run(initial, timeout));
    for (std::size_t port = 0; port < 2; ++port)
      for (int metadata = 0; metadata < 3; ++metadata) {
        // Fresh owners with identical shape/dtype must still obey the prepared
        // metadata. Check layout-only, explicit NCHW and unknown swapped axes.
        auto inputs = frame.tensors();
        inputs[port].layout = metadata == 2 ? neat::TensorLayout::Unknown : neat::TensorLayout::CHW;
        if (metadata == 1)
          inputs[port].axis_semantics = {Axis::N, Axis::C, Axis::H, Axis::W};
        else if (metadata == 2)
          inputs[port].axis_semantics = {Axis::N, Axis::W, Axis::H, Axis::C};
        rejects_layout([&] { (void)run.run(inputs, timeout); });
        neat::Sample sample;
        sample.kind = neat::SampleKind::TensorSet;
        sample.tensors = std::move(inputs);
        rejects_layout([&] { (void)run.push(sample); });
        frame.check(run.run(frame.tensors(), timeout));
      }
    // Positional NHWC without any layout/axis annotation remains supported.
    auto unannotated = frame.tensors();
    for (auto& tensor : unannotated)
      tensor.layout = neat::TensorLayout::Unknown;
    auto positional_pipeline = gt::graph(frame.options);
    auto positional_run = positional_pipeline.build(unannotated);
    frame.check(positional_run.run(unannotated, timeout));
    positional_run.close();
    for (auto& tensor : unannotated)
      tensor.axis_semantics = {Axis::N, Axis::H, Axis::W, Axis::C};
    auto annotated_pipeline = gt::graph(frame.options);
    auto annotated_run = annotated_pipeline.build(unannotated);
    frame.check(annotated_run.run(unannotated, timeout));
    annotated_run.close();
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  std::cout << "GRID_SAMPLE_LAYOUT build_rejections=4 run_rejections=6 push_rejections=6 "
               "recovery=PASS unannotated=PASS result=PASS\n";
}

void numerical_cases() {
  for (const auto c : gt::comparison_cases)
    for (const bool bf : {false, true}) {
      gt::Frame a(c, bf), b(c, bf, 1);
      auto pipeline = gt::graph(a.options);
      neat::RunOptions options;
      options.output_memory = neat::OutputMemory::Owned;
      auto run = pipeline.build(a.tensors(), options);
      try {
        // Same plan and feature contents, different grids, then the first grid again.
        for (const auto* frame : {&a, &b, &a})
          frame->check(run.run(frame->tensors(), timeout));
      } catch (...) {
        run.close();
        throw;
      }
      run.close();
      std::cout << "GRID_SAMPLE_CASE name=" << c.name << " bf16=" << bf
                << " grids=A/B/A result=PASS\n";
    }
  for (const bool align : {false, true})
    for (const auto padding :
         {neat::SimaGridSample2DPadding::Zeros, neat::SimaGridSample2DPadding::Border}) {
      gt::Frame frame({"narrow_batched", 3, 5, 3, 7, 11, align}, true, 0, 2, padding);
      frame.grid[0] = -1;
      frame.grid[1] = 1;
      frame.grid[2] = std::numeric_limits<float>::quiet_NaN();
      auto pipeline = gt::graph(frame.options);
      auto run = pipeline.build(frame.tensors());
      const auto output = run.run(frame.tensors(), timeout);
      run.close();
      frame.check(output);
    }
}

neat::Sample input_sample(const gt::Frame& frame, std::size_t index) {
  neat::Sample sample;
  sample.kind = neat::SampleKind::TensorSet;
  sample.tensors = frame.tensors();
  sample.frame_id = 100 + index;
  sample.input_seq = 200 + index;
  sample.orig_input_seq = 300 + index;
  sample.stream_id = "grid-sample";
  sample.pts_ns = (index + 1) * 1000000;
  sample.dts_ns = sample.pts_ns - 100;
  sample.duration_ns = 1000000;
  return sample;
}

void correlation(const neat::Sample& sample, std::size_t index) {
  gt::require(sample.frame_id == 100 + index && sample.input_seq == 200 + index &&
                  sample.orig_input_seq == 300 + index && sample.stream_id == "grid-sample" &&
                  sample.pts_ns == (index + 1) * 1000000 && sample.dts_ns == sample.pts_ns - 100 &&
                  sample.duration_ns == 1000000,
              "GridSample asynchronous output lost frame metadata");
}

void asynchronous_retention() {
  std::vector<gt::Frame> frames;
  for (int i = 0; i < 8; ++i)
    frames.emplace_back(gt::comparison_cases[5], true, i);
  auto pipeline = gt::graph(frames.front().options);
  neat::RunOptions options;
  options.queue_depth = 3;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  auto run = pipeline.build(frames.front().tensors(), options);
  std::optional<neat::Sample> held;
  try {
    std::size_t submitted = 0;
    const auto push = [&] {
      // The caller's temporary Sample and tensor owners disappear after push.
      gt::require(run.push(input_sample(frames[submitted], submitted)),
                  "GridSample push: " + run.last_error());
      ++submitted;
    };
    while (submitted < 3)
      push();
    for (std::size_t i = 0; i < frames.size(); ++i) {
      {
        neat::Sample output;
        gt::require(run.pull(timeout, output) == neat::PullStatus::Ok, "GridSample pull");
        correlation(output, i);
        frames[i].check(neat::tensors_from_sample(output));
        gt::require(!output.owned, "GridSample ZeroCopy unexpectedly became owned");
        if (!i)
          held = std::move(output);
      }
      if (submitted < frames.size())
        push();
    }
    run.close_input();
    neat::Sample extra;
    gt::require(run.pull(timeout, extra) == neat::PullStatus::Closed, "GridSample EOS drain");
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  correlation(*held, 0);
  frames.front().check(neat::tensors_from_sample(*held));
  std::cout << "GRID_SAMPLE_LIFETIME frames=8 pending=3 held_after_close=1 result=PASS\n";
}
} // namespace

int main() {
  try {
    layout_contract();
    numerical_cases();
    asynchronous_retention();
    std::cout << "GRID_SAMPLE_SUITE result=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GRID_SAMPLE_SUITE result=FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

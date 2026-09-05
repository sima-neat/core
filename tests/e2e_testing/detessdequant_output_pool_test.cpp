#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "test_main.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
using namespace simaai::neat;

constexpr int kElements = 16 * 16 * 16;

sima_test::ModelArchiveFixture make_fixture() {
  // Run quantize -> DetessDequant without loading the placeholder MLA executable.
  return sima_test::make_model_archive_fixture("detessdequant_pool", {{"pool_mpk.json", R"json({
    "name": "pool_test", "model_path": "share/placeholder.elf",
    "input_nodes": [{"name": "model_input", "type": "buffer", "size": 16384,
                     "shape": [16,16,16], "dtype": "float32"}],
    "plugins": [
      {"name": "quant", "sequence": 0, "processor": "A65",
       "config_params": {"kernel": "quantize", "params": {
         "input_dtype": "float32", "output_dtype": "int8",
         "input_shapes": [[16,16,16]], "output_shapes": [[16,16,16]],
         "q_scale": [1.0], "q_zp": [0], "round_off": "TONEAREST"}},
       "input_nodes": [{"name": "model_input", "type": "buffer", "size": 16384}],
       "output_nodes": [{"name": "quant_output", "type": "buffer", "size": 4096}]},
      {"name": "MLA_0", "sequence": 1, "processor": "MLA",
       "resources": {"executable": "share/placeholder.elf"},
       "config_params": {"desired_batch_size": 1, "actual_batch_size": 1},
       "input_nodes": [{"name": "quant_output", "type": "buffer", "size": 4096,
                        "shape": [1,16,16,16], "dtype": "int8"}],
       "output_nodes": [{"name": "input_tensor", "type": "buffer", "size": 4096,
                         "shape": [1,16,16,16], "dtype": "int8"}]},
      {"name": "detess", "sequence": 2, "processor": "A65",
       "config_params": {"kernel": "detessellation_transform", "params": {
         "slice_shape": [16,16,16], "frame_shape": [1,16,16,16], "frame_type": "int8",
         "align_c16": false, "cblock": false,
         "input_shapes": [[1,4096]], "output_shapes": [[1,16,16,16]]}},
       "input_nodes": [{"name": "input_tensor", "type": "buffer", "size": 4096}],
       "output_nodes": [{"name": "detess_output", "type": "buffer", "size": 4096}]},
      {"name": "dequant", "sequence": 3, "processor": "A65",
       "config_params": {"kernel": "dequantize", "params": {
         "input_dtype": "int8", "output_dtype": "float32",
         "input_shapes": [[1,16,16,16]], "output_shapes": [[1,16,16,16]],
         "dq_scale": [1.0], "dq_zp": [0]}},
       "input_nodes": [{"name": "detess_output", "type": "buffer", "size": 4096}],
       "output_nodes": [{"name": "output_tensor", "type": "buffer", "size": 16384}]}
    ]
  })json"}});
}

Sample make_input(int frame) {
  Sample sample;
  sample.kind = SampleKind::TensorSet;
  sample.frame_id = frame;
  sample.tensors = {
      Tensor::from_vector(std::vector<float>(kElements, frame), {16, 16, 16}, TensorMemory::EV74)};
  return sample;
}

void check_output(const Sample& sample, int frame) {
  const auto& outputs = sample_tensor_list(sample);
  require(outputs.size() == 1, "expected one output tensor");
  require(sample.frame_id == frame, "output frame identity changed");
  require(outputs.front().dtype == TensorDType::Float32, "expected FP32 output");
  const auto map = outputs.front().map(MapMode::Read);
  require(map.data && map.size_bytes == kElements * sizeof(float), "wrong output size");
  for (int i = 0; i < kElements; ++i) {
    float value;
    std::memcpy(&value, static_cast<const char*>(map.data) + i * sizeof(value), sizeof(value));
    require(value == frame, "retained output was corrupted or delivered out of order");
  }
}

void check_pool_pressure(bool stop_while_full) {
  const auto fixture = make_fixture();
  Model::Options model_options;
  model_options.preprocess.kind = InputKind::Tensor;
  model_options.preprocess.enable = AutoFlag::On;
  model_options.processcvu.pre_run_target = "A65";
  model_options.processcvu.post_run_target = "A65";
  Model model(fixture.tar_path, model_options);
  Graph graph;
  const auto input = model.input_appsrc_options(true);
  graph.add(nodes::Input(input));
  graph.add(model.preprocess());
  graph.add(model.postprocess());
  graph.add(nodes::Output(OutputOptions::EveryFrame(4)));
  RunOptions options;
  options.output_memory = OutputMemory::ZeroCopy;
  auto run = graph.build(make_input(1), options);

  std::array<Sample, 4> held;
  for (int frame = 1; frame <= 4; ++frame) {
    require(run.push(make_input(frame)), "input was rejected");
    require(run.pull(3000, held[frame - 1]) == PullStatus::Ok, run.last_error());
    check_output(held[frame - 1], frame);
  }
  require(run.push(make_input(5)), "fifth input was rejected");
  Sample pending;
  require(run.pull(250, pending) == PullStatus::Timeout,
          "a full output pool must wait without failing or allocating another buffer: " +
              run.last_error());

  if (stop_while_full) {
    const auto start = std::chrono::steady_clock::now();
    run.stop();
    require(run.last_error().empty(), "stopping a full pool must not report an allocation error");
    run.close();
    require(std::chrono::steady_clock::now() - start < std::chrono::seconds(2),
            "close must interrupt output-pool acquisition");
  } else {
    held[0] = {};
    require(run.pull(3000, held[0]) == PullStatus::Ok, run.last_error());
    check_output(held[0], 5);
    run.close_input();
    require(run.pull(3000, pending) == PullStatus::Closed, "expected EOS without duplicate output");
    run.close();
  }
  for (int frame = 2; frame <= 4; ++frame)
    check_output(held[frame - 1], frame);
}
} // namespace

RUN_TEST("detessdequant_output_pool_test", ([] {
           check_pool_pressure(false);
           check_pool_pressure(true);
         }));

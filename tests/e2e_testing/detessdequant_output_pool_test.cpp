#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "test_main.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace {
using namespace simaai::neat;

constexpr int kElements = 16 * 16 * 16;

sima_test::ModelArchiveFixture make_fixture() {
  // Exercise retained zero-copy output pressure on the pre-MLA ProcessCVU
  // arena without executing the placeholder MLA program.
  auto fixture = sima_test::make_model_archive_fixture("detessdequant_pool", {{"pool_mpk.json", R"json({
    "name": "pool_test", "model_sdk_version": "2.0.0",
    "model_path": "share/placeholder.elf", "sequence": 1,
    "input_nodes": [{"name": "model_input", "type": "buffer", "size": 16384,
                     "shape": [1,16,16,16], "dtype": "float32"}],
    "plugins": [
      {"name": "quant", "sequence": 1, "processor": "EV74",
       "config_params": {"desired_batch_size": 1, "actual_batch_size": 1,
        "kernel": "quantization_transform", "params": {
         "input_shapes": [[1,16,16,16]], "output_shapes": [[1,16,16,16]],
         "channel_params": [[1.0, 0]], "num_bits": 8,
         "rounding": "TONEAREST", "output_data_type": "int8"}},
       "input_nodes": [{"name": "model_input", "type": "buffer", "size": 16384}],
       "output_nodes": [{"name": "quant_output", "type": "buffer", "size": 4096}],
       "type": "sgpProcess"},
      {"name": "MLA_0", "sequence": 2, "processor": "MLA",
       "resources": {"executable": "share/placeholder.elf"},
       "config_params": {"desired_batch_size": 1, "actual_batch_size": 1,
                         "number_of_quads_to_user": 1,
                         "input_types": [{"scalar": "int8", "shape": [1,16,16,16]}],
                         "output_types": [{"scalar": "int8", "shape": [1,4096]}]},
       "input_nodes": [{"name": "quant_output", "type": "buffer", "size": 4096,
                        "shape": [1,16,16,16], "dtype": "int8"}],
       "output_nodes": [{"name": "input_tensor", "type": "buffer", "size": 4096,
                         "shape": [1,4096], "dtype": "int8"}],
       "type": "sgpProcess"},
      {"name": "detess", "sequence": 3, "processor": "EV74",
       "config_params": {"desired_batch_size": 1, "actual_batch_size": 1,
        "kernel": "detessellation_transform", "params": {
         "slice_shape": [16,16,16], "frame_shape": [1,16,16,16], "frame_type": "int8",
         "align_c16": false, "cblock": false,
         "input_shapes": [[1,4096]], "output_shapes": [[1,16,16,16]]}},
       "input_nodes": [{"name": "input_tensor", "type": "buffer", "size": 4096}],
       "output_nodes": [{"name": "detess_output", "type": "buffer", "size": 4096}],
       "type": "sgpProcess"},
      {"name": "dequant", "sequence": 4, "processor": "EV74",
       "config_params": {"desired_batch_size": 1, "actual_batch_size": 1,
        "kernel": "dequantization_transform", "params": {
         "input_shapes": [[1,16,16,16]], "output_shapes": [[1,16,16,16]],
         "channel_params": [[1.0, 0]], "input_data_type": "int8"}},
       "input_nodes": [{"name": "detess_output", "type": "buffer", "size": 4096}],
       "output_nodes": [{"name": "output_tensor", "type": "buffer", "size": 16384}],
       "type": "sgpProcess"}
    ]
  })json"}}, false);
  const auto share = std::filesystem::path(fixture.root_dir) / "share";
  std::filesystem::create_directories(share);
  sima_test::write_topology_elf(share / "placeholder.elf", "data.ifm.b0", 4096U,
                                "data.ofm.b0", 4096U);
  const std::string archive =
      "tar -czf " + sima_test::model_archive_shell_quote(fixture.tar_path) + " -C " +
      sima_test::model_archive_shell_quote(fixture.root_dir) + " .";
  require(std::system(archive.c_str()) == 0,
          "failed to archive detess/dequant output-pool topology fixture");
  return fixture;
}

Sample make_input(int frame) {
  Sample sample;
  sample.kind = SampleKind::TensorSet;
  sample.frame_id = frame;
  sample.tensors = {Tensor::from_vector(
      std::vector<float>(kElements, static_cast<float>(frame)), {16, 16, 16},
      TensorMemory::EV74)};
  return sample;
}

void check_output(const Sample& sample, int frame) {
  const auto& outputs = sample_tensor_list(sample);
  require(outputs.size() == 1, "expected one output tensor");
  require(sample.frame_id == frame, "output frame identity changed");
  require(outputs.front().dtype == TensorDType::Int8, "expected INT8 output");
  const auto map = outputs.front().map(MapMode::Read);
  require(map.data && map.size_bytes >= kElements, "wrong output size");
  for (int i = 0; i < kElements; ++i) {
    const auto value = static_cast<const std::int8_t*>(map.data)[i];
    require(value == static_cast<std::int8_t>(frame),
            "retained output was corrupted or delivered out of order at element " +
                std::to_string(i) + ": got " + std::to_string(static_cast<int>(value)) +
                ", expected " + std::to_string(frame) +
                ", mapped bytes " + std::to_string(map.size_bytes) +
                ", byte offset " + std::to_string(outputs.front().byte_offset) +
                ", physical offset " +
                std::to_string(outputs.front().route.physical_byte_offset) +
                ", segment " + outputs.front().route.segment_name +
                ", storage kind " +
                std::to_string(static_cast<int>(outputs.front().storage->kind)));
  }
}

void check_retained_outputs(bool stop_with_outputs_held) {
  const auto fixture = make_fixture();
  Model::Options model_options;
  model_options.preprocess.kind = InputKind::Tensor;
  model_options.preprocess.enable = AutoFlag::On;
  model_options.processcvu.pre_run_target = "EV74";
  model_options.processcvu.post_run_target = "EV74";
  Model model(fixture.tar_path, model_options);
  Graph graph;
  const auto input = model.input_appsrc_options(true);
  graph.add(nodes::Input(input));
  graph.add(model.preprocess());
  graph.add(nodes::Output(OutputOptions::EveryFrame(4)));
  RunOptions options;
  options.output_memory = OutputMemory::ZeroCopy;
  auto run = graph.build(make_input(1), options);

  std::array<Sample, 5> held;
  for (int frame = 1; frame <= 5; ++frame) {
    require(run.push(make_input(frame)), "input was rejected");
    require(run.pull(3000, held[frame - 1]) == PullStatus::Ok, run.last_error());
    check_output(held[frame - 1], frame);
  }

  if (stop_with_outputs_held) {
    const auto start = std::chrono::steady_clock::now();
    run.stop();
    require(run.last_error().empty(),
            "stopping with retained outputs must not report an allocation error");
    run.close();
    require(std::chrono::steady_clock::now() - start < std::chrono::seconds(2),
            "close must not wait for retained zero-copy outputs");
  } else {
    run.close_input();
    Sample pending;
    require(run.pull(3000, pending) == PullStatus::Closed, "expected EOS without duplicate output");
    run.close();
  }
  for (int frame = 1; frame <= 5; ++frame)
    check_output(held[frame - 1], frame);
}
} // namespace

RUN_TEST("detessdequant_output_pool_test", ([] {
           check_retained_outputs(false);
           check_retained_outputs(true);
         }));

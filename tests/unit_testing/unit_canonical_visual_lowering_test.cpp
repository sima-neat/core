#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <simaai/kernels/authoring.h>
#include <simaai/kernels/ops/histogram.h>
#include <simaai/kernels/ops/fast.h>
#include <simaai/kernels/ops/fast_brief.h>
#include <simaai/kernels/ops/tracker.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace rt = simaai::runtime;
namespace k = simaai::kernels;
namespace op = k::ops;
namespace core = simaai::neat::pipeline_internal::sima;
namespace lower = core::kernel_lowering;
void require(bool value) {
  if (!value)
    throw std::runtime_error("visual cold typed parity");
}
void compare(const k::Executable& actual, k::Result<k::Executable> reference) {
  require(std::holds_alternative<k::Executable>(reference));
  const auto& expected = std::get<k::Executable>(reference).state();
  const auto& state = actual.state();
  require(state.packet.entry == expected.packet.entry &&
          state.packet.payload == expected.packet.payload);
  const auto ports = [](const auto& a, const auto& b) {
    require(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      require(a[i].shape == b[i].shape && a[i].dtype == b[i].dtype &&
              a[i].layout.index() == b[i].layout.index());
  };
  ports(state.contract->input_specs, expected.contract->input_specs);
  ports(state.contract->output_specs, expected.contract->output_specs);
}
int main() {
  try {
    for (const int batch : {1, 3})
      for (const int graph : {235, 236, 237, 238})
        for (const bool detect : {false, true}) {
          if (detect && graph != 238)
            continue;
          core::StageStaticSpec semantic;
          auto& cfg = semantic.processcvu;
          cfg.graph_id = graph;
          cfg.graph_name = graph == 235   ? "feature_histogram"
                           : graph == 236 ? "grider_fast"
                           : graph == 237 ? "track_descriptor"
                                          : "track_klt";
          cfg.resolved_exec_backend = "EV74";
          cfg.width = cfg.height = 48;
          cfg.batch_size = batch;
          cfg.max_features = 16;
          cfg.threshold = 19;
          cfg.descriptor_words = 8;
          cfg.num_points = 8;
          cfg.win_half = 3;
          cfg.max_iters = 7;
          cfg.max_level = 0;
          cfg.fast_threshold = 21;
          cfg.detect_new_features = detect;
          auto checked = lower::check_visual_stage(semantic);
          if (auto* error = std::get_if<rt::Error>(&checked))
            throw std::runtime_error(error->message);
          const auto& actual = *std::get<std::shared_ptr<core::KernelStageRecipe>>(checked);
          const k::TensorSpec image{{batch, 48, 48, 1}, k::DType::UInt8};
          const std::array inputs{image};
          if (graph == 235) {
            const std::array outputs{k::TensorSpec{{batch, 256}, k::DType::Int32}};
            compare(actual.members[0].executable,
                    op::histogram::make_executable({inputs, outputs, {}}, k::Target::EV74));
          }
          if (graph == 236) {
            const std::array outputs{k::TensorSpec{{batch, 49}, k::DType::Int32}};
            compare(
                actual.members[0].executable,
                op::fast::make_executable({inputs, outputs, {19, op::fast::OutputEncoding::Packed}},
                                          k::Target::EV74));
          }
          if (graph == 237) {
            const std::array outputs{k::TensorSpec{{batch, 49}, k::DType::Int32},
                                     k::TensorSpec{{batch, 16, 8}, k::DType::Int32}};
            compare(actual.members[0].executable,
                    op::fast_brief::make_executable(
                        {inputs, outputs, {19, op::fast_brief::OutputEncoding::Packed}},
                        k::Target::EV74));
          }
          if (graph == 238) {
            const std::array ins{image, image, k::TensorSpec{{batch, 8, 2}, k::DType::Int32},
                                 k::TensorSpec{{batch, 1}, k::DType::Int32}};
            const std::array outs{k::TensorSpec{{batch, 8, 2}, k::DType::Float32},
                                  k::TensorSpec{{batch, 8}, k::DType::Int32}};
            compare(actual.members[0].executable,
                    op::tracker::make_executable({ins, outs, {3, 7, 0}}, k::Target::EV74));
            std::vector<std::int32_t> counts(batch);
            require(actual.constant_data.size() == counts.size() * sizeof(std::int32_t));
            std::memcpy(counts.data(), actual.constant_data.data(), actual.constant_data.size());
            require(counts == std::vector<std::int32_t>(batch, 8));
            require(actual.members[0].inputs.back().source ==
                    rt::gstreamer::TensorStagePortSource::Constant);
            require(actual.members.size() == (detect ? 2U : 1U));
            if (detect) {
              const std::array outputs{k::TensorSpec{{batch, 49}, k::DType::Int32}};
              compare(
                  actual.members[1].executable,
                  op::fast::make_executable(
                      {inputs, outputs, {21, op::fast::OutputEncoding::Packed}}, k::Target::EV74));
              require(actual.members[1].dependencies == std::vector<std::size_t>{0});
            }
          }
        }
    std::cout << "Visual cold generic/typed packet+contract parity: 10 PASS, no device\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

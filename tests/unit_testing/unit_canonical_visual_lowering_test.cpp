#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <simaai/scheduler/testing/reference.h>
#include <array>
namespace test = simaai::scheduler::testing;
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace rt = simaai::scheduler::client;
namespace k = simaai::scheduler::client;
namespace core = simaai::neat::pipeline_internal::sima;
namespace lower = core::kernel_lowering;
void require(bool value) {
  if (!value)
    throw std::runtime_error("visual cold typed parity");
}
void compare(const k::KernelExecutable& actual, k::Result<k::KernelExecutable> reference) {
  require(std::holds_alternative<k::KernelExecutable>(reference));
  const auto& expected = std::get<k::KernelExecutable>(reference);
  const auto a = test::packet_snapshot(actual), b = test::packet_snapshot(expected);
  require(a.entry == b.entry && a.payload == b.payload);
  const auto ports = [](const auto& a, const auto& b) {
    require(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      require(a[i].shape == b[i].shape && a[i].dtype == b[i].dtype &&
              a[i].layout.index() == b[i].layout.index());
  };
  const auto ai = std::get<k::KernelInspection>(k::inspect_kernel(actual));
  const auto bi = std::get<k::KernelInspection>(k::inspect_kernel(expected));
  ports(ai.inputs(), bi.inputs());
  ports(ai.outputs(), bi.outputs());
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
                    test::reference_visual("histogram", k::Target::EV74, inputs, outputs));
          }
          if (graph == 236) {
            const std::array outputs{k::TensorSpec{{batch, 49}, k::DType::Int32}};
            compare(actual.members[0].executable,
                    test::reference_visual("fast", k::Target::EV74, inputs, outputs,
                                           {.threshold = 19}));
          }
          if (graph == 237) {
            const std::array outputs{k::TensorSpec{{batch, 49}, k::DType::Int32},
                                     k::TensorSpec{{batch, 16, 8}, k::DType::Int32}};
            compare(actual.members[0].executable,
                    test::reference_visual("fastbrief", k::Target::EV74, inputs, outputs,
                                           {.threshold = 19}));
          }
          if (graph == 238) {
            const std::array ins{image, image, k::TensorSpec{{batch, 8, 2}, k::DType::Int32},
                                 k::TensorSpec{{batch, 1}, k::DType::Int32}};
            const std::array outs{k::TensorSpec{{batch, 8, 2}, k::DType::Float32},
                                  k::TensorSpec{{batch, 8}, k::DType::Int32}};
            compare(actual.members[0].executable,
                    test::reference_visual("tracker", k::Target::EV74, ins, outs,
                                           {.win_half = 3, .max_iters = 7, .max_level = 0}));
            std::vector<std::int32_t> counts(batch);
            require(actual.constant_data.size() == counts.size() * sizeof(std::int32_t));
            std::memcpy(counts.data(), actual.constant_data.data(), actual.constant_data.size());
            require(counts == std::vector<std::int32_t>(batch, 8));
            require(actual.members[0].inputs.back().source == rt::TensorStagePortSource::Constant);
            require(actual.members.size() == (detect ? 2U : 1U));
            if (detect) {
              const std::array outputs{k::TensorSpec{{batch, 49}, k::DType::Int32}};
              compare(actual.members[1].executable,
                      test::reference_visual("fast", k::Target::EV74, inputs, outputs,
                                             {.threshold = 21}));
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

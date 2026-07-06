#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

class ScopedEnvVar {
public:
  ScopedEnvVar(const char* key, const std::string& value) : key_(key), had_old_(false) {
    if (const char* old = std::getenv(key_.c_str())) {
      had_old_ = true;
      old_ = old;
    }
    setenv(key_.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (had_old_) {
      setenv(key_.c_str(), old_.c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

private:
  std::string key_;
  std::string old_;
  bool had_old_;
};

simaai::neat::Graph make_rgb_graph() {
  using namespace simaai::neat;

  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(64)));
  return graph;
}

simaai::neat::PreprocOptions make_downstream_preproc_options() {
  simaai::neat::PreprocOptions opt;
  opt.set_input_shape({48, 64, 3});
  opt.set_output_shape({48, 64, 3});
  opt.set_slice_shape({16, 64, 3});
  opt.input_img_type = "RGB";
  opt.output_img_type = "RGB";
  opt.output_dtype = "EVXX_INT8";
  opt.scaled_width = 64;
  opt.scaled_height = 48;
  opt.normalize = false;
  opt.aspect_ratio = false;
  opt.tessellate = true;
  opt.single_output_handoff = true;
  opt.next_cpu = "APU";
  opt.upstream_name = "input";
  opt.num_buffers = 4;
  opt.q_scale = 0.25;
  opt.q_zp = 0;
  return opt;
}

simaai::neat::Sample tensor_to_sample(const simaai::neat::Tensor& tensor) {
  using namespace simaai::neat;
  Sample sample = sample_from_tensors(TensorList{tensor});
  sample.payload_tag = "RGB";
  sample.owned = true;
  return sample;
}

void require_tensor_outputs(const simaai::neat::TensorList& outputs, const std::string& where) {
  require(outputs.size() == 1U, where + ": expected one tensor output");
}

void require_tensor_sample_outputs(const simaai::neat::Sample& outputs, const std::string& where) {
  require(outputs.size() == 1U, where + ": expected one sample output");
  const auto& output = outputs.front();
  const bool has_single_tensor =
      (simaai::neat::sample_has_tensor_list(output) && output.tensors.size() == 1U) ||
      (output.kind == simaai::neat::SampleKind::Tensor && output.tensor.has_value());
  require(has_single_tensor, where + ": expected single-tensor sample output");
}

} // namespace

RUN_TEST(
    "unit_graph_build_input_strategy_test", ([] {
      using namespace simaai::neat;

      const ScopedEnvVar preflight("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "1");

      cv::Mat mat_seed(48, 64, CV_8UC3, cv::Scalar(20, 80, 140));
      Tensor tensor_seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x63);
      const Sample sample_seed = tensor_to_sample(tensor_seed);

      RunOptions run_opt;
      run_opt.queue_depth = 4;
      run_opt.overflow_policy = OverflowPolicy::Block;

      // Sync build path parity for Mat/Tensor/Sample.
      {
        Graph graph = make_rgb_graph();

        Run run_mat =
            graph.build_seeded_internal(std::vector<cv::Mat>{mat_seed}, RunMode::Sync, run_opt);
        require_tensor_outputs(run_mat.run(std::vector<cv::Mat>{mat_seed}, 1000), "sync build mat");
        run_mat.stop();

        Run run_tensor =
            graph.build_seeded_internal(TensorList{tensor_seed}, RunMode::Sync, run_opt);
        require_tensor_outputs(run_tensor.run(TensorList{tensor_seed}, 1000), "sync build tensor");
        run_tensor.stop();

        Run run_sample = graph.build_seeded_internal(Sample{sample_seed}, RunMode::Sync, run_opt);
        require_tensor_sample_outputs(run_sample.run(Sample{sample_seed}, 1000),
                                      "sync build sample");
        run_sample.stop();
      }

      // Sync run cache parity for Mat/Tensor overloads.
      {
        Graph graph = make_rgb_graph();
        require_tensor_outputs(graph.run(std::vector<cv::Mat>{mat_seed}, run_opt),
                               "sync run mat first");
        require_tensor_outputs(graph.run(std::vector<cv::Mat>{mat_seed}, run_opt),
                               "sync run mat second");
        require_tensor_outputs(graph.run(TensorList{tensor_seed}, run_opt),
                               "sync run tensor first");
        require_tensor_outputs(graph.run(TensorList{tensor_seed}, run_opt),
                               "sync run tensor second");
      }

      // Async build parity for Mat/Tensor/Sample.
      {
        Graph graph = make_rgb_graph();

        Run run_mat = graph.build(std::vector<cv::Mat>{mat_seed}, run_opt);
        require_tensor_outputs(run_mat.run(std::vector<cv::Mat>{mat_seed}, 1000),
                               "async build mat");
        run_mat.stop();

        Run run_tensor = graph.build(TensorList{tensor_seed}, run_opt);
        require_tensor_outputs(run_tensor.run(TensorList{tensor_seed}, 1000), "async build tensor");
        run_tensor.stop();

        Run run_sample = graph.build(Sample{sample_seed}, run_opt);
        require_tensor_sample_outputs(run_sample.run(Sample{sample_seed}, 1000),
                                      "async build sample");
        run_sample.stop();
      }

      // Legacy pool opt-out must remain SystemMemory even when downstream auto inference would
      // otherwise choose a device-visible memory policy.
      {
        InputOptions legacy_opt;
        legacy_opt.payload_type = PayloadType::Image;
        legacy_opt.format = FormatTag::RGB;
        legacy_opt.use_simaai_pool = false;

        const ScopedEnvVar no_preflight("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0");

        Graph graph;
        graph.add(nodes::Input(legacy_opt));
        graph.add(nodes::Preproc(make_downstream_preproc_options()));
        graph.add(nodes::Output(OutputOptions::EveryFrame(64)));

        Run run = graph.build(std::vector<cv::Mat>{mat_seed}, run_opt);
        const auto core = run_internal::core(run);
        require(core != nullptr, "legacy pool opt-out: missing run core");
        require(!core->pipeline.stream_opt.require_device_visible_input,
                "legacy pool opt-out should not be rewritten to device-visible input");
        run.stop();
      }
    }));

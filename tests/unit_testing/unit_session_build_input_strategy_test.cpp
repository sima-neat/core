#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Session.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <string>

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

simaai::neat::Session make_rgb_session() {
  using namespace simaai::neat;

  Session session;
  InputOptions src_opt;
  src_opt.media_type = "video/x-raw";
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.use_simaai_pool = false;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  session.add(nodes::Input(src_opt));
  session.add(nodes::Output(OutputOptions::EveryFrame(64)));
  return session;
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

void require_tensor_sample_outputs(const simaai::neat::SampleList& outputs,
                                   const std::string& where) {
  require(outputs.size() == 1U, where + ": expected one sample output");
  const auto& output = outputs.front();
  const bool has_single_tensor =
      (simaai::neat::sample_has_tensor_list(output) && output.tensors.size() == 1U) ||
      (output.kind == simaai::neat::SampleKind::Tensor && output.tensor.has_value());
  require(has_single_tensor, where + ": expected single-tensor sample output");
}

} // namespace

RUN_TEST(
    "unit_session_build_input_strategy_test", ([] {
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
        Session session = make_rgb_session();

        Run run_mat = session.build(std::vector<cv::Mat>{mat_seed}, RunMode::Sync, run_opt);
        require_tensor_outputs(run_mat.run(std::vector<cv::Mat>{mat_seed}, 1000), "sync build mat");
        run_mat.stop();

        Run run_tensor = session.build(TensorList{tensor_seed}, RunMode::Sync, run_opt);
        require_tensor_outputs(run_tensor.run(TensorList{tensor_seed}, 1000), "sync build tensor");
        run_tensor.stop();

        Run run_sample = session.build(SampleList{sample_seed}, RunMode::Sync, run_opt);
        require_tensor_sample_outputs(run_sample.run(SampleList{sample_seed}, 1000),
                                      "sync build sample");
        run_sample.stop();
      }

      // Sync run cache parity for Mat/Tensor overloads.
      {
        Session session = make_rgb_session();
        require_tensor_outputs(session.run(std::vector<cv::Mat>{mat_seed}, run_opt),
                               "sync run mat first");
        require_tensor_outputs(session.run(std::vector<cv::Mat>{mat_seed}, run_opt),
                               "sync run mat second");
        require_tensor_outputs(session.run(TensorList{tensor_seed}, run_opt),
                               "sync run tensor first");
        require_tensor_outputs(session.run(TensorList{tensor_seed}, run_opt),
                               "sync run tensor second");
      }

      // Async build parity for Mat/Tensor/Sample.
      {
        Session session = make_rgb_session();

        Run run_mat = session.build(std::vector<cv::Mat>{mat_seed}, RunMode::Async, run_opt);
        require_tensor_outputs(run_mat.run(std::vector<cv::Mat>{mat_seed}, 1000),
                               "async build mat");
        run_mat.stop();

        Run run_tensor = session.build(TensorList{tensor_seed}, RunMode::Async, run_opt);
        require_tensor_outputs(run_tensor.run(TensorList{tensor_seed}, 1000), "async build tensor");
        run_tensor.stop();

        Run run_sample = session.build(SampleList{sample_seed}, RunMode::Async, run_opt);
        require_tensor_sample_outputs(run_sample.run(SampleList{sample_seed}, 1000),
                                      "async build sample");
        run_sample.stop();
      }
    }));

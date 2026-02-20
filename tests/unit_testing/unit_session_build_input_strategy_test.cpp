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
  src_opt.format = "RGB";
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
  Sample sample;
  sample.kind = SampleKind::Tensor;
  sample.tensor = tensor;
  sample.payload_tag = "RGB";
  sample.owned = true;
  return sample;
}

void require_tensor_output(const simaai::neat::Sample& sample, const std::string& where) {
  require(sample.kind == simaai::neat::SampleKind::Tensor, where + ": expected tensor sample kind");
  require(sample.tensor.has_value(), where + ": missing tensor payload");
}

} // namespace

RUN_TEST("unit_session_build_input_strategy_test", ([] {
           using namespace simaai::neat;

           const ScopedEnvVar preflight("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "1");
           const ScopedEnvVar sync_num_buffers("SIMA_SYNC_RUN_NUM_BUFFERS", "4");

           cv::Mat mat_seed(48, 64, CV_8UC3, cv::Scalar(20, 80, 140));
           Tensor tensor_seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x63);
           const Sample sample_seed = tensor_to_sample(tensor_seed);

           RunOptions run_opt;
           run_opt.queue_depth = 4;
           run_opt.overflow_policy = OverflowPolicy::Block;

           // Sync build path parity for Mat/Tensor/Sample.
           {
             Session session = make_rgb_session();

             Run run_mat = session.build(mat_seed, RunMode::Sync, run_opt);
             require_tensor_output(run_mat.push_and_pull(mat_seed, 1000), "sync build mat");
             run_mat.stop();

             Run run_tensor = session.build(tensor_seed, RunMode::Sync, run_opt);
             require_tensor_output(run_tensor.push_and_pull(tensor_seed, 1000),
                                   "sync build tensor");
             run_tensor.stop();

             Run run_sample = session.build(sample_seed, RunMode::Sync, run_opt);
             require_tensor_output(run_sample.run(sample_seed, 1000), "sync build sample");
             run_sample.stop();
           }

           // Sync run cache parity for Mat/Tensor overloads.
           {
             Session session = make_rgb_session();
             require_tensor_output(session.run(mat_seed, run_opt), "sync run mat first");
             require_tensor_output(session.run(mat_seed, run_opt), "sync run mat second");
             require_tensor_output(session.run(tensor_seed, run_opt), "sync run tensor first");
             require_tensor_output(session.run(tensor_seed, run_opt), "sync run tensor second");
           }

           // Async build parity for Mat/Tensor/Sample.
           {
             Session session = make_rgb_session();

             Run run_mat = session.build(mat_seed, RunMode::Async, run_opt);
             require_tensor_output(run_mat.push_and_pull(mat_seed, 1000), "async build mat");
             run_mat.stop();

             Run run_tensor = session.build(tensor_seed, RunMode::Async, run_opt);
             require_tensor_output(run_tensor.push_and_pull(tensor_seed, 1000),
                                   "async build tensor");
             run_tensor.stop();

             Run run_sample = session.build(sample_seed, RunMode::Async, run_opt);
             require_tensor_output(run_sample.run(sample_seed, 1000), "async build sample");
             run_sample.stop();
           }
         }));

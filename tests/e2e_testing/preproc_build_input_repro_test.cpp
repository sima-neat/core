#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "gst/GstHelpers.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool env_bool(const char* key, bool def) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return def;
  }
  return std::string(v) != "0";
}

int env_int(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return def;
  }
  return std::atoi(v);
}

void run_preproc_build_repro(simaai::neat::RunMode mode, bool tessellate) {
  using namespace simaai::neat;

  cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
  if (!img.isContinuous()) {
    img = img.clone();
  }
  const Tensor tensor_rgb =
      Tensor::from_cv_mat(img, ImageSpec::PixelFormat::RGB, TensorMemory::EV74);

  Graph graph;

  InputOptions src_opt;
  src_opt.format = FormatTag::RGB;
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = false;
  src_opt.use_simaai_pool = env_bool("SIMA_PREPROC_USE_POOL", true);
  src_opt.pool_min_buffers = env_int("SIMA_PREPROC_NUM_BUFFERS", 4);
  src_opt.pool_max_buffers = src_opt.pool_min_buffers;
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = 0;

  PreprocOptions pre_opt;
  pre_opt.set_input_shape({img.rows, img.cols, 3});
  pre_opt.set_output_shape({640, 640, 3});
  pre_opt.scaled_width = 640;
  pre_opt.scaled_height = 640;
  pre_opt.input_img_type = "RGB";
  pre_opt.output_img_type = "RGB";
  pre_opt.input_stride = 1;
  pre_opt.output_stride = 1;
  pre_opt.normalize = false;
  pre_opt.aspect_ratio = false;
  pre_opt.tessellate = tessellate;
  pre_opt.output_dtype = "EVXX_INT8";
  pre_opt.scaling_type = "BILINEAR";
  pre_opt.padding_type = "CENTER";
  pre_opt.next_cpu = "APU";
  pre_opt.upstream_name = "decoder";
  pre_opt.num_buffers = src_opt.pool_min_buffers;
  if (tessellate) {
    pre_opt.set_slice_shape({32, 128, 3});
  }
  pre_opt.q_scale = 0.25;
  pre_opt.q_zp = 0;

  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Preproc(pre_opt));

  OutputOptions sink_opt;
  sink_opt.sync = (mode == RunMode::Sync);
  sink_opt.drop = true;
  sink_opt.max_buffers = 1;
  graph.add(nodes::Output(sink_opt));

  RunOptions run_opt;
  run_opt.output_memory = OutputMemory::Owned;
  run_opt.enable_metrics = true;
  run_opt.queue_depth = 1;

  const int timeout_ms = env_int("SIMA_INPUT_TIMEOUT_MS", 20000);
  std::cerr << "[REPRO] mode=" << ((mode == RunMode::Sync) ? "sync" : "async")
            << " tessellate=" << (tessellate ? 1 : 0) << "\n";

  auto run = graph.build(TensorList{tensor_rgb}, mode, run_opt);
  TensorList outs = run.run(TensorList{tensor_rgb}, timeout_ms);
  run.close();

  require(outs.size() == 1, "preproc repro: expected one output tensor");
  const Tensor& out = outs.front();
  require(out.shape.size() >= 2, "preproc repro: missing output shape");
  require(out.shape[0] == 640 && out.shape[1] == 640, "preproc repro: output shape mismatch");
}

} // namespace

int main() {
  try {
    require(simaai::neat::element_exists("neatprocesscvu"),
            "missing NEAT CVU plugin (neatprocesscvu)");

    const bool tessellate = env_bool("SIMA_PREPROC_REPRO_TESSELLATE", true);
    const bool run_sync = env_bool("SIMA_PREPROC_REPRO_RUN_SYNC", true);
    const bool run_async = env_bool("SIMA_PREPROC_REPRO_RUN_ASYNC", true);

    if (!run_sync && !run_async) {
      throw std::runtime_error("preproc repro: both sync and async modes disabled");
    }

    if (run_sync) {
      run_preproc_build_repro(simaai::neat::RunMode::Sync, tessellate);
    }
    if (run_async) {
      run_preproc_build_repro(simaai::neat::RunMode::Async, tessellate);
    }

    std::cout << "[OK] preproc_build_input_repro_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}

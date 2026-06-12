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
  if (!v)
    return def;
  return std::string(v) != "0";
}

int env_int(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def;
  return std::atoi(v);
}

} // namespace

int main() {
  try {
    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing SIMA preproc plugin (neatprocesscvu).");

    cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
    if (!img.isContinuous())
      img = img.clone();
    const simaai::neat::Tensor tensor_rgb = simaai::neat::Tensor::from_cv_mat(
        img, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);

    simaai::neat::Graph p;

    simaai::neat::InputOptions src_opt;
    src_opt.format = simaai::neat::FormatTag::RGB;
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

    simaai::neat::PreprocOptions pre_opt;
    pre_opt.set_input_shape({img.rows, img.cols, 3});
    pre_opt.set_output_shape({640, 640, 3});
    pre_opt.scaled_width = 640;
    pre_opt.scaled_height = 640;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = false;
    pre_opt.aspect_ratio = false;
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.next_cpu = "APU";
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = src_opt.pool_min_buffers;
    pre_opt.set_slice_shape({32, 128, 3});
    pre_opt.q_scale = 0.25;
    pre_opt.q_zp = 0;

    p.add(simaai::neat::nodes::Input(src_opt));
    p.add(simaai::neat::nodes::Preproc(pre_opt));

    simaai::neat::OutputOptions sink_opt;
    sink_opt.sync = false;
    sink_opt.drop = true;
    sink_opt.max_buffers = 1;
    p.add(simaai::neat::nodes::Output(sink_opt));

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    run_opt.queue_depth = 1;
    const int timeout_ms = env_int("SIMA_INPUT_TIMEOUT_MS", 20000);

    auto run = p.build(simaai::neat::TensorList{tensor_rgb}, run_opt);
    simaai::neat::TensorList outs = run.run(simaai::neat::TensorList{tensor_rgb}, timeout_ms);
    run.close();

    require(outs.size() == 1, "Preproc output missing tensor");
    const simaai::neat::Tensor* tensor = &outs.front();
    require(tensor->shape.size() >= 2, "Preproc output missing shape");
    const int64_t h = tensor->shape[0];
    const int64_t w = tensor->shape[1];
    require(w == 640 && h == 640, "Preproc size mismatch");
    if (tensor->semantic.image.has_value()) {
      require(tensor->semantic.image->format == simaai::neat::ImageSpec::PixelFormat::RGB,
              "Preproc format mismatch");
    }
    require(tensor->dtype == simaai::neat::TensorDType::UInt8 ||
                tensor->dtype == simaai::neat::TensorDType::Int8,
            "Preproc dtype mismatch");

    simaai::neat::Tensor cpu = tensor->clone();
    simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
    const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    require(map.data != nullptr && map.size_bytes >= expected, "Preproc bytes missing");
    const uint8_t* plane = static_cast<const uint8_t*>(map.data);
    for (size_t i = 0; i < expected; i += 3) {
      if (plane[i] != 64 || plane[i + 1] != 128 || plane[i + 2] != 192) {
        throw std::runtime_error("Preproc output bytes mismatch");
      }
    }

    std::cout << "[OK] preproc_standalone_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return fail_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}

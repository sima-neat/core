#include "perf/codec_perf_common.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool has_byte_variation(const std::vector<std::uint8_t>& bytes) {
  if (bytes.empty()) {
    return false;
  }
  const std::uint8_t first = bytes.front();
  const std::size_t step = std::max<std::size_t>(1U, bytes.size() / 4096U);
  for (std::size_t i = 0; i < bytes.size(); i += step) {
    if (bytes[i] != first) {
      return true;
    }
  }
  return false;
}

std::vector<std::uint8_t>
require_decoded_accuracy_sample(const sima_codec_perf::CodecPerfConfig& config,
                                const simaai::neat::Sample& sample, bool i420) {
  sima_codec_perf::require_decoded_sample(sample, config.scenario_id);
  const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
  const simaai::neat::Tensor& tensor = tensors.front();
  require(tensor.width() == config.width, config.scenario_id + ": decoded width mismatch");
  require(tensor.height() == config.height, config.scenario_id + ": decoded height mismatch");
  require(i420 ? tensor.is_i420() : tensor.is_nv12(),
          config.scenario_id + ": decoded output format mismatch");
  const std::vector<std::uint8_t> bytes =
      i420 ? tensor.copy_i420_contiguous() : tensor.copy_nv12_contiguous();
  const std::size_t min_luma_bytes =
      static_cast<std::size_t>(config.width) * static_cast<std::size_t>(config.height);
  require(bytes.size() >= min_luma_bytes, config.scenario_id + ": decoded payload is too small");
  require(has_byte_variation(bytes), config.scenario_id + ": decoded payload is constant");
  if (i420) {
    return bytes;
  }
  std::vector<std::uint8_t> planar(bytes.begin(), bytes.begin() + min_luma_bytes);
  for (std::size_t component = 0; component < 2; ++component) {
    for (std::size_t i = min_luma_bytes + component; i < bytes.size(); i += 2) {
      planar.push_back(bytes[i]);
    }
  }
  return planar;
}

std::vector<std::uint8_t>
run_accuracy_case(const sima_codec_perf::CodecPerfConfig& config,
                  const std::vector<sima_codec_perf::EncodedFrame>& frames, bool i420 = false) {
  require(!frames.empty(), config.scenario_id + ": no encoded frames");
  const std::vector<simaai::neat::Sample> samples =
      sima_codec_perf::make_sample_sequence(frames, config, 1);
  const simaai::neat::Sample& seed = samples.front();
  simaai::neat::Graph graph =
      sima_codec_perf::make_decode_graph(config, seed, 8, false, false, i420);
  simaai::neat::Run run =
      graph.build(simaai::neat::Sample{seed}, sima_codec_perf::codec_run_options(8));

  if (!run.push(simaai::neat::Sample{seed})) {
    throw std::runtime_error(config.scenario_id + ": push failed");
  }
  run.close_input();
  const simaai::neat::Sample out =
      sima_codec_perf::pull_or_throw(run, 5000, config.scenario_id + ": pull");
  const auto pixels = require_decoded_accuracy_sample(config, out, i420);
  run.stop();
  std::cout << "[OK] " << config.scenario_id << " width=" << config.width
            << " height=" << config.height << " fps=" << config.fps
            << " format=" << (i420 ? "I420" : "NV12") << "\n";
  return pixels;
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    const sima_codec_perf::CodecPerfConfig mjpeg{.scenario_id = "codec_accuracy_mjpeg_decode",
                                                 .run_mode = "codec_accuracy",
                                                 .decode_type = simaai::neat::SimaDecodeType::MJPEG,
                                                 .width = 1280,
                                                 .height = 720,
                                                 .fps = 30};
    const auto jpeg_frames = sima_codec_perf::make_mjpeg_frames(mjpeg);
    require(run_accuracy_case(mjpeg, jpeg_frames) == run_accuracy_case(mjpeg, jpeg_frames, true),
            "MJPEG NV12 and I420 visible pixels differ");

    const sima_codec_perf::CodecPerfConfig h264{.scenario_id = "codec_accuracy_h264_decode",
                                                .run_mode = "codec_accuracy",
                                                .decode_type = simaai::neat::SimaDecodeType::H264,
                                                .width = 1280,
                                                .height = 720,
                                                .fps = 30};
    const auto h264_frames = sima_codec_perf::extract_h264_access_units(1);
    require(run_accuracy_case(h264, h264_frames) == run_accuracy_case(h264, h264_frames, true),
            "H.264 NV12 and I420 visible pixels differ");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

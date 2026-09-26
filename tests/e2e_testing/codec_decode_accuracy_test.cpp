#include "perf/codec_perf_common.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

struct DecoderOptionsCase {
  const char* name;
  int input_buffers;
  int output_buffers;
  const char* tuning;
};

std::uint64_t pixel_hash(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto byte : bytes) {
    hash = (hash ^ byte) * 1099511628211ULL;
  }
  return hash;
}

using FrameIdentity = std::pair<std::int64_t, std::uint64_t>;

std::vector<FrameIdentity> run_buffer_case(const sima_codec_perf::CodecPerfConfig& config,
                                           const std::vector<simaai::neat::Sample>& samples,
                                           const DecoderOptionsCase& control, bool owned) {
  const std::string context =
      config.scenario_id + "/" + control.name + (owned ? "/owned" : "/retained-native");
  simaai::neat::Graph graph(context);
  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Encoded;
  input.caps_override = samples.front().caps_string;
  input.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  input.block = true;
  input.is_live = false;
  input.do_timestamp = false;
  input.pool_max_buffers = 8;
  graph.add(simaai::neat::nodes::Input(input));
  simaai::neat::SimaDecodeOptions decode;
  decode.type = config.decode_type;
  decode.dec_width = config.width;
  decode.dec_height = config.height;
  decode.dec_fps = config.fps;
  decode.input_buffers = control.input_buffers;
  decode.num_buffers = control.output_buffers;
  decode.decoder_tuning = control.tuning;
  graph.add(simaai::neat::nodes::SimaDecode(decode));
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(8)));
  auto options = sima_codec_perf::codec_run_options(8);
  options.output_memory =
      owned ? simaai::neat::OutputMemory::Owned : simaai::neat::OutputMemory::ZeroCopy;
  options.startup_preflight = false;
  std::cout << "[CASE] " << context << " input_buffers=" << control.input_buffers
            << " output_buffers=" << control.output_buffers << " tuning=" << control.tuning << "\n"
            << std::flush;
  auto run = graph.build(simaai::neat::Sample{samples.front()}, options);

  std::exception_ptr producer_error;
  std::thread producer([&] {
    try {
      for (const auto& sample : samples) {
        require(run.push(simaai::neat::Sample{sample}), context + ": input push failed");
      }
      run.close_input();
    } catch (...) {
      producer_error = std::current_exception();
      try {
        run.close_input();
      } catch (...) {
      }
    }
  });

  // The automatic JPEG TLL pool has two outputs. Appsink retains the most
  // recently pulled buffer, so a consumer must release an older held sample
  // before waiting for further progress. Roomier cases keep the first frame.
  const bool bounded_retention = !owned && control.output_buffers < 0 &&
                                 std::string(control.tuning) == "throughput-low-latency";
  simaai::neat::Sample retained;
  std::vector<FrameIdentity> identities;
  std::exception_ptr consumer_error;
  std::exception_ptr stop_error;
  try {
    for (std::size_t frame = 0; frame < samples.size(); ++frame) {
      auto sample = sima_codec_perf::pull_or_throw(run, 5000, context);
      require(sample.pts_ns == samples[frame].pts_ns,
              context + ": input/output PTS mismatch at frame " + std::to_string(frame));
      require(sample.owned == owned, context + ": unexpected output ownership");
      const auto pixels = require_decoded_accuracy_sample(config, sample, false);
      identities.emplace_back(sample.pts_ns, pixel_hash(pixels));
      if (bounded_retention && frame == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        require(pixel_hash(require_decoded_accuracy_sample(config, sample, false)) ==
                    identities.front().second,
                context + ": output changed while held for 250 ms");
        sample = {};
      } else if ((!bounded_retention && frame == 0) ||
                 (bounded_retention && frame + 1 == samples.size())) {
        retained = std::move(sample);
      }
    }
    simaai::neat::Sample extra;
    simaai::neat::PullError error;
    require(run.pull(5000, extra, &error) == simaai::neat::PullStatus::Closed,
            context + ": expected EOS after exactly " + std::to_string(samples.size()) + " frames");
  } catch (...) {
    consumer_error = std::current_exception();
    try {
      run.stop();
    } catch (...) {
      stop_error = std::current_exception();
    }
  }
  producer.join();
  if (!consumer_error) {
    try {
      run.stop();
    } catch (...) {
      stop_error = std::current_exception();
    }
  }
  if (stop_error && (consumer_error || producer_error)) {
    try {
      std::rethrow_exception(stop_error);
    } catch (const std::exception& error) {
      std::cerr << "[CLEANUP] " << context << ": " << error.what() << "\n";
    } catch (...) {
      std::cerr << "[CLEANUP] " << context << ": unknown shutdown failure\n";
    }
  }
  if (consumer_error) {
    std::rethrow_exception(consumer_error);
  }
  if (producer_error) {
    std::rethrow_exception(producer_error);
  }
  if (stop_error) {
    std::rethrow_exception(stop_error);
  }
  require(pixel_hash(require_decoded_accuracy_sample(config, retained, false)) ==
              (bounded_retention ? identities.back().second : identities.front().second),
          context + ": retained output changed after subsequent frames or close");
  retained = {};
  std::cout << "[OK] " << context << " exact_frames=" << identities.size()
            << " pts=pixels=retained-output=pass retention="
            << (bounded_retention ? "first-250ms,last-through-stop" : "first-through-stop") << "\n";
  return identities;
}

void run_buffer_matrix(const sima_codec_perf::CodecPerfConfig& config,
                       const std::vector<sima_codec_perf::EncodedFrame>& frames) {
  constexpr std::array<DecoderOptionsCase, 4> cases{{
      {"automatic", -1, -1, ""},
      {"tuning-automatic", -1, -1, "throughput-low-latency"},
      {"explicit-counts", 3, 8, ""},
      {"tuning-and-counts", 3, 8, "throughput-low-latency"},
  }};
  require(frames.size() == 30U, config.scenario_id + ": expected 30 no-B fixture frames");
  const auto samples = sima_codec_perf::make_sample_sequence(frames, config, 30);
  const auto baseline = run_buffer_case(config, samples, cases.front(), false);
  for (std::size_t i = 1; i < cases.size(); ++i) {
    require(run_buffer_case(config, samples, cases[i], false) == baseline,
            config.scenario_id + ": overridden decoder changed frame pixels or PTS");
  }
  require(run_buffer_case(config, samples, cases.front(), true) == baseline,
          config.scenario_id + ": owned and native output differ");
}

std::vector<sima_codec_perf::EncodedFrame>
make_distinct_mjpeg_frames(const sima_codec_perf::CodecPerfConfig& config) {
  const auto base = sima_codec_perf::make_mjpeg_frames(config).front();
  const auto image = cv::imdecode(base.bytes, cv::IMREAD_COLOR);
  require(!image.empty(), "cannot read generated MJPEG image");
  std::vector<sima_codec_perf::EncodedFrame> frames;
  for (int frame = 0; frame < 30; ++frame) {
    auto varied = image.clone();
    varied.rowRange(0, 32).setTo(cv::Scalar(frame * 7, frame * 5, frame * 3));
    sima_codec_perf::EncodedFrame encoded;
    encoded.caps = base.caps;
    require(cv::imencode(".jpg", varied, encoded.bytes), "cannot encode distinct MJPEG image");
    frames.push_back(std::move(encoded));
  }
  return frames;
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
    const sima_codec_perf::CodecPerfConfig h265{.scenario_id = "codec_accuracy_h265_decode",
                                                .run_mode = "codec_accuracy",
                                                .decode_type = simaai::neat::SimaDecodeType::H265,
                                                .width = 1280,
                                                .height = 720,
                                                .fps = 30};
    run_buffer_matrix(h264, sima_codec_perf::extract_h264_access_units(30));
    run_buffer_matrix(h265, sima_codec_perf::extract_h265_access_units(30));
    run_buffer_matrix(mjpeg, make_distinct_mjpeg_frames(mjpeg));
    std::cout << "[INFO] buffer matrix uses no-B fixtures; TLL with reordered B frames is not "
                 "qualified\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

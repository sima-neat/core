#include "asset_utils.h"
#include "gst/GstInit.h"
#include "nodes/common/FileInput.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <future>
#include <iostream>
#include <numeric>
#include <set>

namespace {
using namespace simaai::neat;
namespace fs = std::filesystem;

std::vector<std::uint8_t> read_bytes(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  require(file.good(), "missing decoder fixture: " + path.string());
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

Sample pull_frame(Run& run) {
  Sample sample;
  PullError error;
  const auto status = run.pull(5000, sample, &error);
  require(status == PullStatus::Ok, "decode/pull failed: " + error.message);
  return sample;
}

void add_parser(Graph& graph, SimaDecodeType codec) {
  if (codec == SimaDecodeType::H264) {
    H264ParseOptions options;
    options.config_interval = -1;
    options.enforce_caps = true;
    options.alignment = H264ParseOptions::Alignment::AU;
    options.stream_format = H264ParseOptions::StreamFormat::ByteStream;
    graph.add(nodes::H264Parse(options));
  } else if (codec == SimaDecodeType::H265) {
    graph.custom("h265parse disable-passthrough=true config-interval=-1 ! "
                 "video/x-h265,parsed=true,stream-format=byte-stream,alignment=au");
  }
}

std::vector<Sample> load_access_units(const fs::path& path, SimaDecodeType codec, bool reordered) {
  Graph graph("accuracy-access-units");
  graph.add(nodes::FileInput(path.string()));
  if (reordered) {
    graph.add(nodes::VideoTrackSelect());
  }
  add_parser(graph, codec);
  graph.add(nodes::Output(OutputOptions::EveryFrame(16)));
  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  auto run = graph.build(options);
  std::vector<Sample> frames;
  for (int i = 0; i < 12; ++i) {
    auto sample = pull_frame(run);
    require(sample_payload_type(sample) == PayloadType::Encoded, "expected encoded access unit");
    if (reordered) {
      require(sample.pts_ns >= 0 && sample.dts_ns >= 0 && sample.duration_ns > 0,
              "B-frame fixture must preserve packet PTS/DTS/duration");
    } else {
      sample.pts_ns = sample.dts_ns = i * (1000000000LL / 30);
      sample.duration_ns = 1000000000LL / 30;
    }
    frames.push_back(std::move(sample));
  }
  run.stop();
  if (reordered) {
    std::set<int64_t> timestamps;
    bool saw_reordering = false;
    for (std::size_t i = 0; i < frames.size(); ++i) {
      require(timestamps.insert(frames[i].pts_ns).second, "duplicate fixture PTS");
      if (i != 0 && frames[i].pts_ns < frames[i - 1].pts_ns) {
        saw_reordering = true;
      }
    }
    require(saw_reordering, "fixture did not expose B-frame decode order");
  }
  return frames;
}

std::vector<std::uint8_t> visible_pixels(const Sample& sample, bool i420, int width, int height) {
  const auto tensors = tensors_from_sample(sample, true);
  require(tensors.size() == 1, "expected one decoded tensor");
  const auto& tensor = tensors.front();
  require(tensor.width() == width && tensor.height() == height, "decoded dimensions changed");
  require(i420 ? tensor.is_i420() : tensor.is_nv12(), "unexpected decoded layout");
  auto bytes = i420 ? tensor.copy_i420_contiguous() : tensor.copy_nv12_contiguous();
  require(bytes.size() == static_cast<std::size_t>(width * height * 3 / 2),
          "incomplete visible decoded planes");
  return bytes;
}

void run_case(const std::string& name, std::vector<Sample> frames,
              const std::vector<std::uint8_t>& reference, SimaDecodeOptions decode,
              OutputMemory memory, int width, int height, int fps) {
  const bool i420 = decode.out_format == FormatTag::I420;
  const std::size_t frame_bytes = static_cast<std::size_t>(width * height * 3 / 2);
  require(reference.size() >= frames.size() * frame_bytes, "short software reference");
  for (std::size_t i = 0; i < frames.size(); ++i) {
    frames[i].frame_id = static_cast<int64_t>(i);
    frames[i].stream_id = name;
    frames[i].attributes["fixture-frame"] = std::to_string(i);
  }
  std::vector<std::size_t> presentation(frames.size());
  std::iota(presentation.begin(), presentation.end(), 0);
  std::sort(presentation.begin(), presentation.end(),
            [&](auto a, auto b) { return frames[a].pts_ns < frames[b].pts_ns; });

  Graph graph(name);
  InputOptions input;
  input.payload_type = PayloadType::Encoded;
  input.caps_override = frames.front().caps_string;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  input.block = true;
  input.pool_max_buffers = 4;
  graph.add(nodes::Input(input));
  decode.dec_width = width;
  decode.dec_height = height;
  decode.dec_fps = fps;
  graph.add(nodes::SimaDecode(decode));
  graph.add(nodes::Output(OutputOptions::EveryFrame(2)));
  RunOptions options;
  options.output_memory = memory;
  options.overflow_policy = OverflowPolicy::Block;
  options.queue_depth = 4;
  // Reordered video needs later access units before its first output.
  options.startup_preflight = false;
  auto run = graph.build(Sample{frames.front()}, options);
  auto producer = std::async(std::launch::async, [&] {
    for (const auto& frame : frames) {
      require(run.push(Sample{frame}), "encoded input was rejected");
    }
    run.close_input();
  });
  Sample retained;
  std::vector<std::uint8_t> retained_bytes;
  try {
    for (std::size_t i = 0; i < frames.size(); ++i) {
      auto out = pull_frame(run);
      const auto& expected = frames[presentation[i]];
      require(out.owned == (memory == OutputMemory::Owned), "output ownership mode changed");
      require(out.pts_ns == expected.pts_ns, "presentation timestamp/order changed");
      require(out.dts_ns == expected.dts_ns && out.duration_ns == expected.duration_ns,
              "frame timing changed");
      require(out.stream_id == expected.stream_id, "stream identity changed");
      // Ordinary decoder output uses presentation-order IDs; attributes retain source identity.
      require(out.frame_id == static_cast<int64_t>(i), "decoder output sequence changed");
      require(out.attributes == expected.attributes, "frame attributes shifted");
      auto pixels = visible_pixels(out, i420, width, height);
      int max_error = 0;
      for (std::size_t byte = 0; byte < pixels.size(); ++byte) {
        max_error = std::max(max_error,
                             std::abs(int(pixels[byte]) - int(reference[i * frame_bytes + byte])));
      }
      require(max_error <= (i420 ? 3 : 0),
              "software-reference pixel mismatch: " + std::to_string(max_error));
      if (i == 0) {
        retained = out;
        retained_bytes = std::move(pixels);
      }
    }
    Sample extra;
    PullError error;
    require(run.pull(5000, extra, &error) == PullStatus::Closed, "missing EOS or extra output");
    producer.get();
    run.stop();
  } catch (...) {
    run.stop();
    producer.wait();
    throw;
  }
  require(visible_pixels(retained, i420, width, height) == retained_bytes,
          "retained output changed during later decoding or stop");
  std::cout << "[OK] " << name << " frames=" << frames.size() << "\n";
}
} // namespace

int main() {
  try {
    gst_init_once();
    const auto root = sima_test::test_codec_perf_h264_fixture_path().parent_path();
    for (auto codec : {SimaDecodeType::H264, SimaDecodeType::H265}) {
      const std::string name = codec == SimaDecodeType::H264 ? "h264" : "h265";
      SimaDecodeOptions options;
      options.type = codec;
      options.decoder_tuning = "default";
      run_case(name + "-ipb-auto", load_access_units(root / (name + "_ipb.mp4"), codec, true),
               read_bytes(root / (name + "_ipb.nv12")), options,
               codec == SimaDecodeType::H264 ? OutputMemory::ZeroCopy : OutputMemory::Owned, 160,
               96, 10);
      const auto no_b = codec == SimaDecodeType::H264
                            ? sima_test::test_codec_perf_h264_fixture_path()
                            : sima_test::test_codec_perf_h265_fixture_path();
      options.decoder_tuning = "throughput-low-latency";
      options.input_buffers = 2;
      options.num_buffers = 4;
      run_case(name + "-no-b-overrides", load_access_units(no_b, codec, false),
               read_bytes(root / (name + "_no_b.nv12")), options,
               codec == SimaDecodeType::H264 ? OutputMemory::Owned : OutputMemory::ZeroCopy, 1280,
               720, 30);
    }
    std::vector<Sample> jpeg;
    std::vector<std::uint8_t> reference;
    for (int i = 0; i < 4; ++i) {
      const auto stem = "jpeg422_" + std::to_string(i);
      jpeg.push_back(make_encoded_sample(read_bytes(root / (stem + ".jpg")),
                                         "image/jpeg,width=160,height=96,framerate=10/1",
                                         i * 100000000LL, i * 100000000LL, 100000000LL));
      const auto bytes = read_bytes(root / (stem + ".i420"));
      reference.insert(reference.end(), bytes.begin(), bytes.end());
    }
    SimaDecodeOptions options;
    options.type = SimaDecodeType::MJPEG;
    options.out_format = FormatTag::I420;
    run_case("mjpeg-422-i420", std::move(jpeg), reference, options, OutputMemory::Owned, 160, 96,
             10);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}

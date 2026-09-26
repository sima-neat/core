#include "encoder_perf_common.h"
#include "codec_perf_common.h"
#include "nodes/common/Output.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/SimaEncode.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <memory>

namespace ep = sima_encoder_perf;
namespace neat = simaai::neat;
namespace groups = neat::nodes::groups;
using json = nlohmann::json;

namespace {
struct Config {
  std::string path = "standalone", codec = "h264", memory = "dma", capture;
  double median_fps = 0;
  bool sender() const {
    return path == "raw-sender" || path == "encoded-sender";
  }
  bool passthrough() const {
    return path == "encoded-sender";
  }
  bool paced() const {
    return median_fps > 0;
  }
};

Config arguments(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    require(i + 1 < argc, "missing CLI value for " + key);
    const std::string value = argv[++i];
    if (key == "--path")
      c.path = value;
    else if (key == "--codec")
      c.codec = value;
    else if (key == "--input")
      c.memory = value;
    else if (key == "--capture")
      c.capture = value;
    else if (key == "--median-fps") {
      std::size_t used = 0;
      c.median_fps = std::stod(value, &used);
      require(used == value.size() && std::isfinite(c.median_fps) && c.median_fps > 0,
              "median FPS must be finite and positive");
    } else
      throw std::runtime_error("unknown argument " + key);
  }
  require(c.path == "standalone" || c.path == "legacy" || c.sender(), "unknown encoder path");
  require(c.codec == "h264" || c.codec == "h265" || c.codec == "mjpeg", "unknown encoder codec");
  require(c.path != "legacy" || c.codec == "h264", "legacy encoder supports only H.264");
  require(c.memory == "cpu" || c.memory == "dma", "unknown input memory");
  require(c.path != "legacy" || c.memory == "dma", "legacy encoder requires --input dma");
  require(!c.passthrough() || c.memory == "cpu", "encoded sender requires --input cpu");
  require(!c.paced() ||
              std::ceil(.95 * c.median_fps * ep::kPacedSeconds) + ep::kWarmup < ep::kMaximumFrames,
          "paced run exceeds frame-accounting bound");
  return c;
}

neat::SimaEncodeType encode_type(const Config& c) {
  return c.codec == "h264"   ? neat::SimaEncodeType::H264
         : c.codec == "h265" ? neat::SimaEncodeType::H265
                             : neat::SimaEncodeType::MJPEG;
}
groups::RtspCodec sender_codec(const Config& c) {
  return c.codec == "h264"   ? groups::RtspCodec::H264
         : c.codec == "h265" ? groups::RtspCodec::H265
                             : groups::RtspCodec::MJPEG;
}
int payload_type(const Config& c) {
  return c.codec == "h264" ? 96 : c.codec == "h265" ? 98 : 26;
}
std::string payloader(const Config& c) {
  return c.codec == "h264" ? "rtph264pay" : c.codec == "h265" ? "rtph265pay" : "rtpjpegpay";
}

// Same static-blocks-v1 pixels as the native encoder workload. Preparing and
// mapping input memory happen before Graph::build and before warmup.
neat::Tensor patterned_input(const Config& c) {
  constexpr int width = 1280, height = 720;
  auto tensor = make_nv12_tensor(width, height);
  {
    auto map = tensor.storage->map(neat::MapMode::Write);
    require(map.data != nullptr, "cannot prepare NV12 input");
    auto* data = static_cast<std::uint8_t*>(map.data);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        data[y * width + x] = 16 + (x / 8 * 11 + y / 8 * 7) % 220;
    for (int y = 0; y < height / 2; ++y)
      for (int x = 0; x < width / 2; ++x) {
        data[width * height + y * width + 2 * x] = 64 + ((x / 8 + y) % 2) * 112;
        data[width * height + y * width + 2 * x + 1] = 72 + ((x / 8 + y + 1) % 2) * 104;
      }
  }
  if (c.memory == "dma") {
    tensor = tensor.cvu();
    require(neat::pipeline_internal::tensor_has_dmabuf_memory(tensor),
            "DMA fixture is not DMA-BUF backed");
  }
  tensor.read_only = true;
  return tensor;
}

std::vector<neat::Sample> inputs(const Config& c) {
  if (c.passthrough()) {
    sima_codec_perf::CodecPerfConfig config;
    config.scenario_id = "encoder_sender_fixture";
    config.width = 1280;
    config.height = 720;
    config.fps = ep::kInputFps;
    // Keep the whole prepared one-second sequence. It starts at an IDR and
    // cycles only at the fixture boundary; never repeat a random dependent AU.
    const auto frames = c.codec == "mjpeg"  ? sima_codec_perf::make_mjpeg_frames(config)
                        : c.codec == "h264" ? sima_codec_perf::extract_h264_access_units(30)
                                            : sima_codec_perf::extract_h265_access_units(30);
    require(c.codec == "mjpeg" || frames.size() == 30,
            "prepared codec fixture must contain 30 AUs");
    return sima_codec_perf::make_sample_sequence(frames, config, static_cast<int>(frames.size()));
  }
  std::vector<neat::Sample> result;
  for (int i = 0; i < ep::kBuffers; ++i) {
    neat::Sample sample;
    sample.kind = neat::SampleKind::Tensor;
    sample.tensor = patterned_input(c);
    result.push_back(std::move(sample));
  }
  return result;
}

// Fingerprint the actual prepared samples, including DMA readback, before build/timing.
// Length prefixes preserve row/AU boundaries. Padding bytes are excluded; layout is recorded.
json input_fingerprint(const Config& c, const std::vector<neat::Sample>& samples) {
  std::unique_ptr<GChecksum, decltype(&g_checksum_free)> sum(g_checksum_new(G_CHECKSUM_SHA256),
                                                             g_checksum_free);
  require(sum != nullptr, "cannot allocate input checksum");
  const auto update = [&](const void* data, std::size_t size) {
    const auto length = GUINT64_TO_BE(static_cast<guint64>(size));
    g_checksum_update(sum.get(), reinterpret_cast<const guchar*>(&length), sizeof(length));
    g_checksum_update(sum.get(), static_cast<const guchar*>(data), size);
  };
  json layouts = json::array();
  for (const auto& sample : samples) {
    update(sample.caps_string.data(), sample.caps_string.size());
    if (c.passthrough()) {
      const auto tensors = neat::tensors_from_sample(sample, true);
      require(tensors.size() == 1, "encoded fingerprint needs one AU tensor");
      const auto bytes = tensors.front().copy_payload_bytes();
      require(!bytes.empty(), "empty encoded fingerprint input");
      update(bytes.data(), bytes.size());
      layouts.push_back({{"bytes", bytes.size()}, {"caps", sample.caps_string}});
    } else {
      require(sample.tensor.has_value(), "missing raw fingerprint tensor");
      const auto mapped = sample.tensor->map_nv12_read();
      require(mapped.has_value(), "cannot read prepared NV12 input");
      const auto& view = mapped->view;
      require(view.width == 1280 && view.height == 720, "unexpected prepared NV12 dimensions");
      for (int y = 0; y < view.height; ++y)
        update(view.y + y * view.y_stride, view.width);
      for (int y = 0; y < view.height / 2; ++y)
        update(view.uv + y * view.uv_stride, view.width);
      layouts.push_back(
          {{"y_stride", view.y_stride},
           {"uv_stride", view.uv_stride},
           {"y_offset", view.y - static_cast<const uint8_t*>(mapped->mapping.data)},
           {"uv_offset", view.uv - static_cast<const uint8_t*>(mapped->mapping.data)}});
    }
  }
  return {{"format", c.passthrough() ? "sha256-length-prefixed-au-v1"
                                     : "sha256-length-prefixed-nv12-rows-v1"},
          {"sha256", g_checksum_get_string(sum.get())},
          {"sequence_frames", samples.size()},
          {"layouts", layouts}};
}

neat::Graph graph_for(const Config& c, const neat::Sample& seed, int port) {
  neat::InputOptions input;
  input.is_live = true;
  input.do_timestamp = false;
  input.block = true;
  input.pool_min_buffers = ep::kBuffers;
  input.pool_max_buffers = ep::kBuffers;
  input.memory_policy =
      c.memory == "dma" ? neat::InputMemoryPolicy::Ev74 : neat::InputMemoryPolicy::SystemMemory;
  input.fps_n = ep::kInputFps;
  if (c.passthrough()) {
    input.payload_type = neat::PayloadType::Encoded;
    input.caps_override = seed.caps_string;
  } else {
    input.payload_type = neat::PayloadType::Image;
    input.format = neat::FormatTag::NV12;
    input.width = 1280;
    input.height = 720;
  }
  neat::Graph graph("encoder_perf");
  graph.add(neat::nodes::Input(input));
  neat::SimaEncodeOptions encode;
  encode.type = encode_type(c);
  encode.width = 1280;
  encode.height = 720;
  encode.fps = ep::kInputFps;
  // Native output default is checked below. Legacy GOP/IDR zero sentinels
  // resolve to FPS and three times FPS, matching the explicit 30/90 preset.
  encode.num_buffers = -1;
  if (c.codec == "mjpeg")
    encode.quality = 80;
  else {
    encode.bitrate_kbps = 4000;
    encode.rate_control = "vbr";
    encode.gop_length = 30;
    encode.idr_interval = 90;
  }
  if (c.sender()) {
    auto sender = c.passthrough() ? groups::VideoSenderOptions::Passthrough(sender_codec(c))
                                  : groups::VideoSenderOptions::FromRaw(encode);
    sender.host = "127.0.0.1";
    sender.video_port_base = port;
    sender.sync = false;
    sender.async = false;
    sender.rtp.payload_type = payload_type(c);
    graph.add(groups::VideoSender(sender));
  } else {
    if (c.path == "legacy")
      graph.add(neat::nodes::H264EncodeSima(1280, 720, ep::kInputFps, 4000));
    else
      graph.add(neat::nodes::SimaEncode(encode));
    graph.add(neat::nodes::Output(neat::OutputOptions::EveryFrame(ep::kBuffers)));
  }
  return graph;
}

json encoder_settings(GstElement* encoder) {
  json result;
  for (const char* name : {"enc-type", "enc-fmt", "enc-width", "enc-height", "enc-frame-rate",
                           "enc-bitrate", "enc-bitrate-mode", "enc-gop-length", "enc-idr-interval",
                           "enc-profile", "enc-level", "enc-quality", "num-output-buffers"}) {
    auto* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(encoder), name);
    require(spec != nullptr, std::string("missing encoder property ") + name);
    GValue value = G_VALUE_INIT;
    g_value_init(&value, spec->value_type);
    g_object_get_property(G_OBJECT(encoder), name, &value);
    gchar* serialized = gst_value_serialize(&value);
    require(serialized != nullptr, std::string("cannot read encoder property ") + name);
    result[name] = serialized;
    g_free(serialized);
    g_value_unset(&value);
  }
  require(result.at("num-output-buffers") == "4",
          "native default is not four encoder output buffers");
  return result;
}

json execute(const Config& c) {
  const auto prepared = inputs(c);
  const auto fingerprint = input_fingerprint(c, prepared);
  std::unique_ptr<sima_test::UdpReceiver> receiver;
  if (c.sender())
    receiver = std::make_unique<sima_test::UdpReceiver>();
  auto graph = graph_for(c, prepared.front(), receiver ? receiver->port() : 0);
  neat::RunOptions options;
  options.queue_depth = ep::kBuffers;
  options.overflow_policy = neat::OverflowPolicy::Block;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  options.advanced.copy_input = false;
  options.startup_preflight = false;
  const auto build_start = ep::now_ns();
  auto run = graph.build(neat::Sample{prepared.front()}, options);
  const double startup = ep::seconds(build_start, ep::now_ns()) * 1000;
  ep::Accounting state(payload_type(c));
  using Element = std::unique_ptr<GstElement, decltype(&gst_object_unref)>;
  Element encoder(c.passthrough() ? nullptr : ep::find_factory(run, "neatencoder"),
                  gst_object_unref);
  Element pay(c.sender() ? ep::find_factory(run, payloader(c)) : nullptr, gst_object_unref);
  const auto settings = encoder ? encoder_settings(encoder.get()) : json(nullptr);
  ep::PadCounter completed(c.passthrough() ? pay.get() : encoder.get(),
                           c.passthrough() ? "sink" : "src", state, false);
  std::unique_ptr<ep::PadCounter> packets;
  if (pay)
    packets = std::make_unique<ep::PadCounter>(pay.get(), "src", state, true);
  std::atomic<bool> stop{false};
  bool eos = false;
  const double rate = .95 * c.median_fps;
  const std::uint64_t target = c.paced() ? static_cast<std::uint64_t>(std::ceil(rate * 60)) : 0;
  std::uint64_t late = 0, accepted_late = 0;
  double maximum_lateness = 0;
  const auto pause_until = [&](std::int64_t deadline) {
    while (!stop && !state.failed && ep::now_ns() < deadline)
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(std::min<std::int64_t>(deadline - ep::now_ns(), 20000000)));
  };
  std::thread producer;
  try {
    producer = std::thread([&] {
      try {
        const auto push = [&] {
          const auto id = state.submit(); // Publish timestamp before asynchronous callbacks.
          auto sample = prepared[id % prepared.size()];
          sample.frame_id = id;
          sample.pts_ns = ep::pts_for(id);
          sample.dts_ns = sample.pts_ns;
          sample.duration_ns = ep::pts_for(id + 1) - ep::pts_for(id);
          sample.stream_id = "encoder_perf";
          require(run.push(sample), "Core rejected input");
          ++state.accepted;
          if (target && state.start && ep::seconds(state.start, ep::now_ns()) >= 60)
            ++accepted_late;
        };
        for (unsigned i = 0; i < ep::kWarmup && !stop && !state.failed; ++i)
          push();
        const auto warmup_deadline = ep::now_ns() + 30000000000LL;
        while (!stop && !state.failed && state.output < ep::kWarmup &&
               ep::now_ns() < warmup_deadline)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(state.output == ep::kWarmup && state.accepted == ep::kWarmup,
                "same-session warmup failed");
        state.start = ep::now_ns();
        std::uint64_t measured_attempts = 0;
        while (!stop && !state.failed) {
          if (target) {
            if (measured_attempts == target)
              break;
            const auto deadline =
                state.start + static_cast<std::int64_t>(measured_attempts / rate * 1e9);
            pause_until(deadline);
            const auto now = ep::now_ns();
            if (stop || state.failed || ep::seconds(state.start, now) >= 60)
              break;
            const double lateness = std::max(0.0, ep::seconds(deadline, now));
            maximum_lateness = std::max(maximum_lateness, lateness * 1000);
            if (lateness > 1 / rate)
              ++late;
          } else if (state.output >= ep::kWarmup + ep::kMinimumFrames &&
                     ep::seconds(state.start, state.last_output) >= ep::kMinimumSeconds)
            break;
          push();
          ++measured_attempts;
        }
        if (target)
          pause_until(state.start + 60000000000LL);
        run.close_input();
        state.producer_end = ep::now_ns();
        state.producer_done = true;
      } catch (const std::exception& error) {
        state.fail(error.what());
      }
    });
    const auto deadline = ep::now_ns() + 180000000000LL;
    std::int64_t drained = 0, last_error_check = 0;
    std::string packet;
    while (!state.failed && ep::now_ns() < deadline) {
      if (receiver) {
        if (receiver->recv_one(&packet, 50))
          state.received(packet);
      } else {
        neat::Sample sample;
        neat::PullError error;
        const auto status = run.pull(50, sample, &error);
        if (status == neat::PullStatus::Closed)
          eos = true;
        if (status == neat::PullStatus::Error)
          throw std::runtime_error(error.message);
        if (status == neat::PullStatus::Ok) {
          require(neat::sample_payload_type(sample) == neat::PayloadType::Encoded,
                  "non-encoded public output");
          state.delivered(ep::id_from_pts(sample.pts_ns));
        }
      }

      if (state.producer_done && state.output == state.accepted) {
        if (!drained)
          drained = ep::now_ns();
        if ((!receiver && eos) || (receiver && ep::seconds(drained, ep::now_ns()) >= .25))
          break;
      }
      require(!state.producer_done || ep::seconds(state.producer_end, ep::now_ns()) <= 10,
              "accepted frames did not drain within ten seconds");
      if (ep::seconds(last_error_check, ep::now_ns()) >= .1) {
        const auto error = run.last_error();
        require(error.empty(), error);
        last_error_check = ep::now_ns();
      }
    }
    require(ep::now_ns() < deadline, "encoder perf session exceeded 180 seconds");
  } catch (const std::exception& error) {
    state.fail(error.what());
  }
  stop = true;
  // Release blocked pushes before joining. The existing runner's process timeout
  // remains necessary if a broken driver hangs inside Run::stop itself.
  try {
    run.stop();
  } catch (const std::exception& error) {
    state.fail(error.what());
  }
  if (producer.joinable())
    producer.join();
  if (!c.capture.empty()) {
    std::ofstream file(c.capture, std::ios::binary);
    for (const auto& bytes : state.reference)
      file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    require(file.good(), "cannot write warmup reference capture");
  }
  const auto measured = state.output > ep::kWarmup ? state.output - ep::kWarmup : 0;
  const auto attempts = state.attempted > ep::kWarmup ? state.attempted - ep::kWarmup : 0;
  const auto accepted = state.accepted > ep::kWarmup ? state.accepted - ep::kWarmup : 0;
  const double duration =
      state.start ? std::max(target ? 60.0 : 0.0, ep::seconds(state.start, state.last_output)) : 0;
  const double producer_seconds = state.start ? ep::seconds(state.start, state.producer_end) : 0;
  const auto drops = neat::run_internal::stats(run);
  if (!state.failed && (!state.producer_done || state.attempted != state.accepted ||
                        state.accepted != state.completed || state.output != state.accepted ||
                        (receiver && (state.sent_frames != state.accepted ||
                                      state.sent_packets != state.received_packets)) ||
                        drops.inputs_dropped || drops.outputs_dropped))
    state.fail("frame accounting differs");
  if (!state.failed && (target ? attempts != target : measured < 1000 || duration < 10))
    state.fail("measurement/producer target not satisfied");
  std::string scenario = c.path;
  std::replace(scenario.begin(), scenario.end(), '-', '_');
  json result = {{"scenario_id", "runtime_encoder_" + scenario + "_" + c.codec + "_" + c.memory},
                 {"iterations", measured},
                 {"run_mode", target ? "paced" : "unpaced"},
                 {"throughput", duration > 0 ? measured / duration : 0},
                 {"p50", sima_perf::percentile(state.output_latency, 50)},
                 {"p95", sima_perf::percentile(state.output_latency, 95)},
                 {"startup", startup},
                 {"rss_peak_kb", sima_perf::rss_peak_kb()},
                 {"input_drop_count", drops.inputs_dropped},
                 {"output_drop_count", drops.outputs_dropped},
                 {"failure", state.error},
                 {"measured_seconds", duration},
                 {"native_encoder_settings", settings}};
  result["counts"] = {{"attempted", state.attempted.load()},
                      {"accepted", state.accepted.load()},
                      {"completed", state.completed.load()},
                      {"output", state.output.load()},
                      {"sent_frames", state.sent_frames.load()},
                      {"sent_packets", state.sent_packets.load()},
                      {"received_packets", state.received_packets.load()}};
  const double completion_seconds =
      state.start ? std::max(target ? 60.0 : 0.0, ep::seconds(state.start, state.last_encoder)) : 0;
  const auto completions = state.completed > ep::kWarmup ? state.completed - ep::kWarmup : 0;
  result["completion"] = {{"fps", completion_seconds > 0 ? completions / completion_seconds : 0},
                          {"p50_ms", sima_perf::percentile(state.encoder_latency, 50)},
                          {"p95_ms", sima_perf::percentile(state.encoder_latency, 95)},
                          {"seconds", completion_seconds}};
  result["pacing"] = {{"target_fps", rate},
                      {"target_frames", target},
                      {"producer_seconds", producer_seconds},
                      {"producer_fps", producer_seconds > 0 ? accepted / producer_seconds : 0},
                      {"shortfall_frames", target > attempts ? target - attempts : 0},
                      {"late_submissions", late},
                      {"accepted_after_deadline", accepted_late},
                      {"max_lateness_ms", maximum_lateness}};
  result["workload"] = {
      {"width", 1280},
      {"height", 720},
      {"input_fps", 30},
      {"warmup_frames", ep::kWarmup},
      {"input_fingerprint", fingerprint},
      {"input_buffers", 4},
      {"output_buffers", c.passthrough() ? json(nullptr) : json(4)},
      {"bitrate_kbps", 4000},
      {"gop", 30},
      {"idr_interval", 90},
      {"jpeg_quality", 80},
      {"pattern", c.passthrough() ? "prepared-codec-perf-fixture" : "static-blocks-v1"}};
  result["completion_semantics"] =
      c.passthrough() ? "forwarded_access_units" : "native_encoder_access_units";
  result["output_semantics"] =
      receiver ? "complete_rtp_frames_received" : "public_encoded_samples_pulled";
  return result;
}
} // namespace

int main(int argc, char** argv) {
  sima_perf::ScopedStdoutToStderr quiet;
  try {
    const auto config = arguments(argc, argv);
    g_setenv("SIMA_PIPELINE_ABORT_ON_HUNG_STOP_THREADS", "1", TRUE);
    neat::gst_init_once();
    const auto result = execute(config);
    quiet.restore();
    std::cout << result.dump(2) << '\n';
    return result.at("failure").get<std::string>().empty() ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "perf_runtime_encoder_test: " << error.what() << '\n';
    return 1;
  }
}

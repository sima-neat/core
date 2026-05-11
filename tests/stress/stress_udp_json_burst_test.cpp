#include "nodes/groups/OptiViewOutputGroup.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(40, std::min(value, 4000));
}

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

std::vector<uint8_t> make_bbox_payload(uint32_t count, const std::vector<RawBox>& boxes) {
  std::vector<uint8_t> out(sizeof(uint32_t) + boxes.size() * sizeof(RawBox), 0);
  std::memcpy(out.data(), &count, sizeof(uint32_t));
  if (!boxes.empty()) {
    std::memcpy(out.data() + sizeof(uint32_t), boxes.data(), boxes.size() * sizeof(RawBox));
  }
  return out;
}

simaai::neat::Sample make_bbox_tensor_sample(const std::vector<uint8_t>& payload, int frame_id,
                                             int64_t pts_ns, const std::string& stream_id) {
  using namespace simaai::neat;
  auto storage = make_cpu_owned_storage(payload.size());
  auto map = storage->map(MapMode::Write);
  if (map.data && map.size_bytes >= payload.size()) {
    std::memcpy(map.data, payload.data(), payload.size());
  }

  Tensor tensor;
  tensor.storage = storage;
  tensor.dtype = TensorDType::UInt8;
  tensor.layout = TensorLayout::Unknown;
  tensor.shape = {static_cast<int64_t>(payload.size())};
  tensor.device = {DeviceType::CPU, 0};
  tensor.read_only = true;
  tensor.semantic.tess =
      TessSpec{.slice_shape = {}, .format = "BBOX"};

  Sample sample = sample_from_tensors(TensorList{std::move(tensor)});
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  sample.pts_ns = pts_ns;
  return sample;
}

bool is_parseable_optiview_json(const std::string& payload) {
  try {
    const auto parsed = nlohmann::json::parse(payload);
    if (!parsed.contains("type") || !parsed["type"].is_string())
      return false;
    if (!parsed.contains("data") || !parsed["data"].is_object())
      return false;
    if (!parsed["data"].contains("objects") || !parsed["data"]["objects"].is_array())
      return false;
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace

RUN_TEST("stress_udp_json_burst_test", ([] {
           using simaai::neat::nodes::groups::OptiViewJsonInput;
           using simaai::neat::nodes::groups::OptiViewJsonResult;
           using simaai::neat::nodes::groups::OptiViewOutputNodeGroup;
           using simaai::neat::nodes::groups::OptiViewOutputNodeGroupOptions;

           const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 180));
           const int streams = 2;
           const int json_port_base = 9900;

           sima_test::UdpReceiver rx0(json_port_base);
           sima_test::UdpReceiver rx1(json_port_base + 1);

           OptiViewOutputNodeGroup group;
           OptiViewOutputNodeGroupOptions opt;
           opt.send_json = true;
           opt.udp.h264_caps =
               "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au";
           opt.udp.host = "127.0.0.1";
           opt.udp.video_port_base = 9800;
           opt.udp.udp_sync = false;
           opt.udp.udp_async = false;
           opt.json_port_base = json_port_base;
           opt.frame_w = 640;
           opt.frame_h = 480;
           opt.topk = 16;
           opt.labels = {"person", "car", "dog"};

           std::string init_err;
           if (!group.init(opt, streams, &init_err)) {
             if (sima_test::likely_runtime_missing(init_err)) {
               skip_long_test_exception("Skipping UDP JSON burst stress due runtime limitations: " +
                                        init_err);
             }
             throw std::runtime_error("OptiViewOutputNodeGroup init failed: " + init_err);
           }

           struct Guard {
             OptiViewOutputNodeGroup* group_ptr = nullptr;
             ~Guard() {
               if (!group_ptr)
                 return;
               try {
                 group_ptr->stop();
               } catch (...) {
               }
             }
           } guard{&group};

           int emitted = 0;
           int emit_fail = 0;

           for (int i = 0; i < iters; ++i) {
             for (int s = 0; s < streams; ++s) {
               const std::vector<RawBox> boxes = {
                   RawBox{.x = 10 + (i % 30),
                          .y = 20 + (i % 25),
                          .w = 40,
                          .h = 35,
                          .score = 0.90f,
                          .cls = s},
                   RawBox{.x = 120 + (i % 15),
                          .y = 100 + (i % 20),
                          .w = 30,
                          .h = 28,
                          .score = 0.75f,
                          .cls = 2},
               };
               const auto payload = make_bbox_payload(2, boxes);
               const auto yolo = make_bbox_tensor_sample(
                   payload, i, static_cast<int64_t>(i) * 33000000LL, "stream" + std::to_string(s));

               OptiViewJsonInput in;
               in.stream_idx = static_cast<size_t>(s);
               in.yolo_sample = &yolo;
               in.frame_id = i;
               in.output_frame_id = i;
               in.capture_ms = 1000 + i;

               OptiViewJsonResult out;
               if (group.emit_json(in, &out)) {
                 ++emitted;
               } else {
                 ++emit_fail;
               }
             }
           }

           std::vector<std::string> packets0;
           std::vector<std::string> packets1;
           const int expected_per_stream = iters;
           const int got0 = rx0.drain(&packets0, expected_per_stream, 120);
           const int got1 = rx1.drain(&packets1, expected_per_stream, 120);

           int parseable = 0;
           for (const auto& payload : packets0) {
             if (is_parseable_optiview_json(payload))
               ++parseable;
           }
           for (const auto& payload : packets1) {
             if (is_parseable_optiview_json(payload))
               ++parseable;
           }

           const int total_expected = streams * iters;
           const int total_received = got0 + got1;
           const int dropped = total_expected - total_received;

           require(emit_fail == 0, "UDP JSON burst stress should not fail emit_json calls");
           require(emitted == total_expected, "UDP JSON burst stress emitted count mismatch");
           require(total_received > 0,
                   "UDP JSON burst stress should receive at least one datagram");
           require(parseable == total_received,
                   "UDP JSON burst stress should produce parseable JSON datagrams");
           require(dropped <= (total_expected / 2),
                   "UDP JSON burst stress drop count exceeded bounded threshold");

           guard.group_ptr = nullptr;
           group.stop();
         }));

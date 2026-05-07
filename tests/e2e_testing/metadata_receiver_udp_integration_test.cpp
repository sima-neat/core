#include "nodes/groups/MetadataReceiverOutputGroup.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

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

simaai::neat::Sample make_bbox_tensor_sample(const std::vector<uint8_t>& payload, int64_t pts_ns,
                                             int frame_id) {
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
      TessSpec{.tile_width = 0, .tile_height = 0, .tile_channels = 0, .format = "BBOX"};

  Sample sample;
  sample.kind = SampleKind::Tensor;
  sample.tensor = std::move(tensor);
  sample.frame_id = frame_id;
  sample.stream_id = "stream0";
  sample.pts_ns = pts_ns;
  return sample;
}

} // namespace

RUN_TEST("metadata_receiver_udp_integration_test", ([] {
           using nlohmann::json;
           using simaai::neat::nodes::groups::MetadataReceiverObjectDetectionInput;
           using simaai::neat::nodes::groups::MetadataReceiverObjectDetectionResult;
           using simaai::neat::nodes::groups::MetadataReceiverOutputNodeGroup;
           using simaai::neat::nodes::groups::MetadataReceiverOutputNodeGroupOptions;

           sima_test::UdpReceiver rx;

           MetadataReceiverOutputNodeGroup group;
           MetadataReceiverOutputNodeGroupOptions opt;
           opt.send_metadata = true;
           opt.udp.h264_caps =
               "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au";
           opt.udp.host = "127.0.0.1";
           opt.udp.video_port_base = 9700;
           opt.udp.udp_sync = false;
           opt.udp.udp_async = false;
           opt.metadata_port_base = rx.port();
           opt.frame_w = 640;
           opt.frame_h = 480;
           opt.topk = 16;
           opt.labels = {"person", "car", "dog"};

           std::string init_err;
           if (!group.init(opt, 1, &init_err)) {
             if (sima_test::likely_runtime_missing(init_err)) {
               throw std::runtime_error(
                   "Skipping MetadataReceiver metadata UDP integration due runtime limitations: " +
                   init_err);
             }
             throw std::runtime_error("MetadataReceiverOutputNodeGroup init failed: " + init_err);
           }

           struct Guard {
             MetadataReceiverOutputNodeGroup* g = nullptr;
             ~Guard() {
               if (!g)
                 return;
               try {
                 g->stop();
               } catch (...) {
               }
             }
           } guard{&group};

           const std::vector<RawBox> boxes = {
               RawBox{.x = 10, .y = 20, .w = 30, .h = 40, .score = 0.92f, .cls = 1},
               RawBox{.x = 100, .y = 120, .w = 50, .h = 40, .score = 0.75f, .cls = 2},
           };

           const auto payload = make_bbox_payload(2, boxes);
           const auto yolo_sample = make_bbox_tensor_sample(payload, 987000000, 77);

           MetadataReceiverObjectDetectionInput in;
           in.stream_idx = 0;
           in.yolo_sample = &yolo_sample;
           in.frame_id = 77;
           in.output_frame_id = 777;
           in.capture_ms = 4444;

           MetadataReceiverObjectDetectionResult out;
           require(group.emit_object_detection(in, &out),
                   "MetadataReceiver metadata integration emit_object_detection failed");
           require(out.ok, "MetadataReceiver metadata integration result should be ok");
           require(out.nonempty,
                   "MetadataReceiver metadata integration should report non-empty detections");
           require(out.boxes == 2,
                   "MetadataReceiver metadata integration detection count mismatch");

           std::string payload_metadata;
           require(rx.recv_one(&payload_metadata, 2000),
                   "MetadataReceiver metadata integration expected UDP payload not received");

           const json parsed = json::parse(payload_metadata);
           require(parsed["type"].get<std::string>() == "object-detection",
                   "MetadataReceiver metadata integration type mismatch");
           require(parsed["frame_id"].get<std::string>() == "777",
                   "MetadataReceiver metadata integration frame id mismatch");
           require(parsed["timestamp"].get<int64_t>() == 4444,
                   "MetadataReceiver metadata integration timestamp mismatch");
           require(parsed["data"]["objects"].size() == 2,
                   "MetadataReceiver metadata integration object count mismatch");
           require(parsed["data"]["objects"][0]["label"].get<std::string>() == "car",
                   "MetadataReceiver metadata integration label mapping mismatch");

           guard.g = nullptr;
           group.stop();
         }));

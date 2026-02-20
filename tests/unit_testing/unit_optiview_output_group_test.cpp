#include "nodes/groups/OptiViewOutputGroup.h"
#include "pipeline/TensorCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

class UdpReceiver {
public:
  UdpReceiver() {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) {
      throw std::runtime_error("UdpReceiver socket() failed");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (::bind(fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
      ::close(fd_);
      throw std::runtime_error("UdpReceiver bind() failed");
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
      ::close(fd_);
      throw std::runtime_error("UdpReceiver getsockname() failed");
    }

    port_ = static_cast<int>(ntohs(addr.sin_port));
  }

  ~UdpReceiver() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  int port() const {
    return port_;
  }

  bool recv_one(std::string* payload, int timeout_ms = 2000) const {
    if (!payload)
      return false;
    payload->clear();

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd_, &rfds);

    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    const int ready = ::select(fd_ + 1, &rfds, nullptr, nullptr, &tv);
    if (ready <= 0)
      return false;

    std::array<char, 4096> buf{};
    const ssize_t n = ::recvfrom(fd_, buf.data(), buf.size(), 0, nullptr, nullptr);
    if (n <= 0)
      return false;

    payload->assign(buf.data(), static_cast<size_t>(n));
    return true;
  }

private:
  int fd_ = -1;
  int port_ = -1;
};

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

simaai::neat::Sample make_bbox_tensor_sample(const std::vector<uint8_t>& payload,
                                             const std::string& fmt = "BBOX") {
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
      TessSpec{.tile_width = 0, .tile_height = 0, .tile_channels = 0, .format = fmt};

  Sample sample;
  sample.kind = SampleKind::Tensor;
  sample.tensor = std::move(tensor);
  sample.frame_id = 7;
  sample.stream_id = "stream0";
  sample.pts_ns = 123000000;
  return sample;
}

bool likely_runtime_missing(const std::string& err) {
  return is_dispatcher_unavailable(err) || err.find("no element") != std::string::npos ||
         err.find("No such element") != std::string::npos ||
         err.find("not found") != std::string::npos || err.find("failed to") != std::string::npos;
}

} // namespace

RUN_TEST("unit_optiview_output_group_test", ([] {
           using nlohmann::json;
           using simaai::neat::nodes::groups::OptiViewJsonInput;
           using simaai::neat::nodes::groups::OptiViewJsonResult;
           using simaai::neat::nodes::groups::OptiViewOutputNodeGroup;
           using simaai::neat::nodes::groups::OptiViewOutputNodeGroupOptions;

           // Configuration failure paths.
           {
             OptiViewOutputNodeGroup group;
             OptiViewOutputNodeGroupOptions bad;
             bad.udp.h264_caps.clear();
             bad.frame_w = 640;
             bad.frame_h = 480;

             std::string err;
             require(!group.init(bad, 1, &err),
                     "OptiViewOutputNodeGroup should reject empty h264_caps");
             require_contains(err, "missing h264_caps",
                              "OptiViewOutputNodeGroup h264_caps error mismatch");

             err.clear();
             bad.udp.h264_caps = "video/x-h264";
             require(!group.init(bad, 0, &err),
                     "OptiViewOutputNodeGroup should reject zero streams");
             require_contains(err, "streams must be > 0",
                              "OptiViewOutputNodeGroup streams error mismatch");
           }

           UdpReceiver rx;

           OptiViewOutputNodeGroup group;
           OptiViewOutputNodeGroupOptions opt;
           opt.send_json = true;
           opt.udp.h264_caps =
               "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au";
           opt.udp.host = "127.0.0.1";
           opt.udp.video_port_base = 9500;
           opt.udp.udp_sync = false;
           opt.udp.udp_async = false;
           opt.json_port_base = rx.port();
           opt.frame_w = 640;
           opt.frame_h = 480;
           opt.topk = 8;
           opt.labels = {"person", "car"};

           std::string init_err;
           if (!group.init(opt, 1, &init_err)) {
             if (likely_runtime_missing(init_err)) {
               throw std::runtime_error(
                   "Skipping OptiViewOutputNodeGroup runtime-dependent checks: " + init_err);
             }
             throw std::runtime_error("OptiViewOutputNodeGroup init failed: " + init_err);
           }

           struct GroupStopGuard {
             OptiViewOutputNodeGroup* g = nullptr;
             ~GroupStopGuard() {
               if (!g)
                 return;
               try {
                 g->stop();
               } catch (...) {
               }
             }
           } guard{&group};

           // Failure path: invalid stream index.
           {
             OptiViewJsonInput in;
             in.stream_idx = 1;
             OptiViewJsonResult out;
             require(!group.emit_json(in, &out),
                     "OptiViewOutputNodeGroup should reject invalid JSON stream index");
             require_contains(out.error, "invalid stream index",
                              "OptiViewOutputNodeGroup invalid stream error mismatch");
           }

           // Failure path: missing yolo sample.
           {
             OptiViewJsonInput in;
             in.stream_idx = 0;
             OptiViewJsonResult out;
             require(!group.emit_json(in, &out),
                     "OptiViewOutputNodeGroup should reject missing yolo sample");
             require_contains(out.error, "missing yolo sample",
                              "OptiViewOutputNodeGroup missing yolo sample error mismatch");
           }

           // Failure path: malformed/non-bbox payload.
           {
             simaai::neat::Sample wrong;
             wrong.kind = simaai::neat::SampleKind::Tensor;
             wrong.tensor = make_color_tensor(16, 12, simaai::neat::ImageSpec::PixelFormat::RGB);
             wrong.payload_tag = "RGB";

             OptiViewJsonInput in;
             in.stream_idx = 0;
             in.yolo_sample = &wrong;
             in.frame_id = 3;

             OptiViewJsonResult out;
             require(!group.emit_json(in, &out),
                     "OptiViewOutputNodeGroup should reject non-bbox yolo sample");
             require_contains(out.error, "bbox tensor not found",
                              "OptiViewOutputNodeGroup non-bbox error mismatch");
           }

           // Failure path: malformed bbox payload with BBOX tag.
           {
             const auto malformed = make_bbox_tensor_sample(std::vector<uint8_t>{1, 2, 3}, "BBOX");

             OptiViewJsonInput in;
             in.stream_idx = 0;
             in.yolo_sample = &malformed;
             in.frame_id = 4;

             OptiViewJsonResult out;
             require(!group.emit_json(in, &out),
                     "OptiViewOutputNodeGroup should reject malformed bbox payload");
             require_contains(out.error, "bbox parse failed",
                              "OptiViewOutputNodeGroup malformed bbox error mismatch");
           }

           // Happy path: valid bbox payload produces JSON datagram.
           {
             const std::vector<RawBox> boxes = {
                 RawBox{.x = 10, .y = 20, .w = 30, .h = 40, .score = 0.9f, .cls = 1}};
             const auto payload = make_bbox_payload(1, boxes);
             const auto yolo = make_bbox_tensor_sample(payload, "BBOX");

             OptiViewJsonInput in;
             in.stream_idx = 0;
             in.yolo_sample = &yolo;
             in.frame_id = 7;
             in.output_frame_id = 77;
             in.capture_ms = 555;

             OptiViewJsonResult out;
             require(group.emit_json(in, &out),
                     "OptiViewOutputNodeGroup should emit JSON for valid bbox payload");
             require(out.ok, "OptiViewOutputNodeGroup result should mark success");
             require(out.nonempty, "OptiViewOutputNodeGroup should report non-empty detection set");
             require(out.boxes == 1, "OptiViewOutputNodeGroup box count mismatch");

             std::string payload_json;
             require(rx.recv_one(&payload_json, 2000),
                     "OptiViewOutputNodeGroup valid emit_json payload not received");

             const json parsed = json::parse(payload_json);
             require(parsed["frame_id"].get<std::string>() == "77",
                     "OptiViewOutputNodeGroup emitted frame_id mismatch");
             require(parsed["timestamp"].get<int64_t>() == 555,
                     "OptiViewOutputNodeGroup emitted timestamp mismatch");
             require(parsed["data"]["objects"].size() == 1,
                     "OptiViewOutputNodeGroup emitted object count mismatch");
           }

           guard.g = nullptr;
           group.stop();
         }));

#include "nodes/groups/OptiViewOutputGroup.h"
#include "optiview_test_utils.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <string>

RUN_TEST(
    "unit_optiview_output_group_test", ([] {
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
        require(!group.init(bad, 1, &err), "OptiViewOutputNodeGroup should reject empty h264_caps");
        require_contains(err, "missing h264_caps",
                         "OptiViewOutputNodeGroup h264_caps error mismatch");

        err.clear();
        bad.udp.h264_caps = "video/x-h264";
        require(!group.init(bad, 0, &err), "OptiViewOutputNodeGroup should reject zero streams");
        require_contains(err, "streams must be > 0",
                         "OptiViewOutputNodeGroup streams error mismatch");
      }

      sima_test::UdpReceiver rx;

      OptiViewOutputNodeGroup group;
      OptiViewOutputNodeGroupOptions opt;
      opt.send_json = true;
      opt.udp.h264_caps = "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au";
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
        if (sima_test::likely_runtime_missing(init_err)) {
          throw std::runtime_error("Skipping OptiViewOutputNodeGroup runtime-dependent checks: " +
                                   init_err);
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
        const auto malformed =
            sima_test::optiview::make_bbox_tensor_sample(std::vector<uint8_t>{1, 2, 3}, "BBOX");

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
        const std::vector<sima_test::optiview::RawBox> boxes = {sima_test::optiview::RawBox{
            .x = 10, .y = 20, .w = 30, .h = 40, .score = 0.9f, .cls = 1}};
        const auto payload = sima_test::optiview::make_bbox_payload(1, boxes);
        const auto yolo = sima_test::optiview::make_bbox_tensor_sample(payload, "BBOX");

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

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/OptiViewOutputGroup.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Session.h"

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "rtsp_port_utils.h"
#include "test_utils.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kStreams = 16;
constexpr int kTopK = 100;
constexpr float kMinScore = 0.52f;
constexpr float kMinIou = 0.30f;

std::string sample_note(const simaai::neat::Sample& out) {
  std::ostringstream oss;
  oss << "frame_id=" << out.frame_id;
  if (!out.stream_id.empty()) {
    oss << " stream_id=" << out.stream_id;
  }
  return oss.str();
}

int run_case(const fs::path& root) {
  const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
  const cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

  using simaai::neat::nodes::groups::OptiViewJsonInput;
  using simaai::neat::nodes::groups::OptiViewJsonResult;
  using simaai::neat::nodes::groups::OptiViewOutputNodeGroup;
  using simaai::neat::nodes::groups::OptiViewOutputNodeGroupOptions;

  const int json_base = rtsp_find_free_port_range(/*base_port=*/18000,
                                                  /*ports_needed=*/kStreams,
                                                  /*stride=*/1,
                                                  /*max_tries=*/5000);
  require(json_base > 0, "failed to reserve contiguous UDP port range for OptiView JSON");
  std::vector<sima_test::UdpReceiver> receivers;
  receivers.reserve(kStreams);
  for (int i = 0; i < kStreams; ++i) {
    receivers.emplace_back(json_base + i, "127.0.0.1");
  }

  OptiViewOutputNodeGroup optiview;
  OptiViewOutputNodeGroupOptions opt;
  opt.send_json = true;
  opt.udp.h264_caps = "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au";
  opt.udp.host = "127.0.0.1";
  opt.udp.video_port_base = 9900;
  opt.udp.udp_sync = false;
  opt.udp.udp_async = false;
  opt.json_port_base = json_base;
  opt.frame_w = img_bgr.cols;
  opt.frame_h = img_bgr.rows;
  opt.topk = kTopK;
  opt.labels = simaai::neat::OptiViewDefaultLabels();

  std::string init_err;
  if (!optiview.init(opt, kStreams, &init_err)) {
    if (sima_test::likely_runtime_missing(init_err)) {
      skip_long_test_exception("OptiView runtime unavailable: " + init_err);
    }
    throw std::runtime_error("OptiViewOutputNodeGroup init failed: " + init_err);
  }

  struct Guard {
    simaai::neat::nodes::groups::OptiViewOutputNodeGroup* group = nullptr;
    simaai::neat::Run* run = nullptr;
    ~Guard() {
      if (run) {
        try {
          run->close();
        } catch (...) {
        }
      }
      if (group) {
        try {
          group->stop();
        } catch (...) {
        }
      }
    }
  } guard{&optiview, nullptr};

  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "BGR";
  model_opt.input_max_width = img_bgr.cols;
  model_opt.input_max_height = img_bgr.rows;
  model_opt.input_max_depth = 3;
  simaai::neat::Model model(tar_gz, model_opt);

  simaai::neat::Session pipeline;
  pipeline.add(simaai::neat::nodes::Input());
  pipeline.add(simaai::neat::nodes::groups::Preprocess(model));
  pipeline.add(simaai::neat::nodes::groups::Infer(model));
  pipeline.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", img_bgr.cols, img_bgr.rows,
                                                  kMinScore, 0.5f, kTopK));
  pipeline.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.preset = simaai::neat::RunPreset::Reliable;
  run_opt.queue_depth = 1;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;

  simaai::neat::Run run = pipeline.build(img_bgr, simaai::neat::RunMode::Async, run_opt);
  guard.run = &run;

  const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();
  std::unordered_set<std::string> expected_frame_ids;

  for (int i = 0; i < kStreams; ++i) {
    simaai::neat::Sample out = run.push_and_pull(img_bgr, 15000);

    std::vector<uint8_t> payload;
    std::string extract_err;
    require(objdet::extract_bbox_payload(out, payload, extract_err),
            "bbox payload extraction failed: " + extract_err + " " + sample_note(out));

    const std::vector<objdet::Box> boxes =
        objdet::parse_boxes_strict(payload, img_bgr.cols, img_bgr.rows, kTopK, false);
    const objdet::MatchResult match =
        objdet::match_expected_boxes(boxes, expected, kMinScore, kMinIou);
    require(match.ok, "bbox accuracy mismatch before OptiView UDP send: " + match.note + " " +
                          sample_note(out));

    const int64_t frame_id = static_cast<int64_t>(i + 1);
    const std::string stream_id = "stream" + std::to_string(i);
    expected_frame_ids.insert(std::to_string(frame_id));

    OptiViewJsonInput json_in;
    json_in.stream_idx = static_cast<std::size_t>(i);
    json_in.stream_id = stream_id;
    json_in.frame_id = frame_id;
    json_in.output_frame_id = static_cast<int>(frame_id);
    json_in.capture_ms = frame_id;
    json_in.yolo_sample = &out;

    OptiViewJsonResult json_out;
    require(optiview.emit_json(json_in, &json_out), "emit_json failed for stream " + stream_id);
    require(json_out.ok,
            "emit_json result not ok for stream " + stream_id + " err=" + json_out.error);
    require(json_out.nonempty, "emit_json expected non-empty detections for stream " + stream_id);
  }

  std::unordered_set<std::string> received_frame_ids;
  for (int i = 0; i < kStreams; ++i) {
    std::string payload;
    require(receivers[static_cast<size_t>(i)].recv_one(&payload, 5000),
            "missing OptiView UDP JSON packet for stream " + std::to_string(i));

    const nlohmann::json parsed = nlohmann::json::parse(payload);
    require(parsed["type"].get<std::string>() == "object-detection",
            "OptiView UDP JSON type mismatch");
    require(parsed["data"]["objects"].is_array(), "OptiView UDP JSON objects field is not array");
    require(!parsed["data"]["objects"].empty(), "OptiView UDP JSON objects unexpectedly empty");

    received_frame_ids.insert(parsed["frame_id"].get<std::string>());
  }

  require(received_frame_ids == expected_frame_ids, "OptiView UDP JSON frame-id set mismatch");

  guard.run = nullptr;
  run.close();
  guard.group = nullptr;
  optiview.stop();

  std::cout << "[OK] graphpipes_yolo_optiview_16stream_test passed\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    return run_case(root);
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

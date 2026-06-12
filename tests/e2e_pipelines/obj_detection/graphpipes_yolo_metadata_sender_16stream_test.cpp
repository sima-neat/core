#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/io/MetadataSender.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Graph.h"

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
constexpr float kMinScore = 0.49f;
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

  const int json_base = rtsp_find_free_port_range(/*base_port=*/18000,
                                                  /*ports_needed=*/kStreams,
                                                  /*stride=*/1,
                                                  /*max_tries=*/5000);
  require(json_base > 0, "failed to reserve contiguous UDP port range for metadata JSON");
  std::vector<sima_test::UdpReceiver> receivers;
  receivers.reserve(kStreams);
  for (int i = 0; i < kStreams; ++i) {
    receivers.emplace_back(json_base + i, "127.0.0.1");
  }

  std::vector<simaai::neat::MetadataSender> senders;
  senders.reserve(kStreams);
  for (int i = 0; i < kStreams; ++i) {
    simaai::neat::MetadataSenderOptions opt;
    opt.host = "127.0.0.1";
    opt.channel = i;
    opt.metadata_port_base = json_base;
    std::string init_err;
    senders.emplace_back(opt, &init_err);
    require(senders.back().ok(), "MetadataSender init failed: " + init_err);
  }

  struct Guard {
    simaai::neat::Run* run = nullptr;
    ~Guard() {
      if (run) {
        try {
          run->close();
        } catch (...) {
        }
      }
    }
  } guard{nullptr};

  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  model_opt.score_threshold = kMinScore;
  model_opt.nms_iou_threshold = 0.5f;
  model_opt.top_k = kTopK;
  simaai::neat::Model model(tar_gz, model_opt);

  simaai::neat::Graph pipeline;
  pipeline.add(simaai::neat::nodes::Input());
  pipeline.add(simaai::neat::nodes::groups::Preprocess(model));
  pipeline.add(simaai::neat::nodes::groups::Infer(model));
  pipeline.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::YoloV8,
                                                  kMinScore, 0.5f, kTopK));
  pipeline.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.preset = simaai::neat::RunPreset::Reliable;
  run_opt.queue_depth = 1;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;

  simaai::neat::Run run = pipeline.build(
      simaai::neat::Sample{simaai::neat::Sample::from_image(
          img_bgr, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
      run_opt);
  guard.run = &run;

  const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();
  std::unordered_set<std::string> expected_frame_ids;

  for (int i = 0; i < kStreams; ++i) {
    require(
        run.push(simaai::neat::Sample{simaai::neat::Sample::from_image(
            img_bgr, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)}),
        "graphpipes push failed");
    simaai::neat::Sample outs = run.pull_samples(15000);
    require(!outs.empty(), "graphpipes expected at least one sample");
    simaai::neat::Sample out = std::move(outs.front());

    std::vector<uint8_t> payload;
    std::string extract_err;
    require(objdet::extract_bbox_payload(out, payload, extract_err),
            "bbox payload extraction failed: " + extract_err + " " + sample_note(out));

    const std::vector<objdet::Box> boxes =
        objdet::parse_boxes_strict(payload, img_bgr.cols, img_bgr.rows, kTopK, false);
    const objdet::MatchResult match =
        objdet::match_expected_boxes(boxes, expected, kMinScore, kMinIou);
    require(match.ok, "bbox accuracy mismatch before MetadataSender UDP send: " + match.note + " " +
                          sample_note(out));

    const int64_t frame_id = static_cast<int64_t>(i + 1);
    const std::string stream_id = "stream" + std::to_string(i);
    expected_frame_ids.insert(std::to_string(frame_id));

    nlohmann::json data;
    data["stream_id"] = stream_id;
    data["objects"] = nlohmann::json::array();
    for (const objdet::Box& box : boxes) {
      if (box.score < kMinScore) {
        continue;
      }
      data["objects"].push_back({
          {"class_id", box.class_id},
          {"confidence", box.score},
          {"bbox", {box.x1, box.y1, box.x2, box.y2}},
      });
    }
    require(!data["objects"].empty(), "metadata sender expected non-empty detections");

    std::string send_err;
    require(senders[static_cast<std::size_t>(i)].send_metadata(
                "object-detection", data.dump(), frame_id, std::to_string(frame_id), &send_err),
            "MetadataSender send_metadata failed for " + stream_id + ": " + send_err);
  }

  std::unordered_set<std::string> received_frame_ids;
  for (int i = 0; i < kStreams; ++i) {
    std::string payload;
    require(receivers[static_cast<size_t>(i)].recv_one(&payload, 5000),
            "missing MetadataSender UDP JSON packet for stream " + std::to_string(i));

    const nlohmann::json parsed = nlohmann::json::parse(payload);
    require(parsed["type"].get<std::string>() == "object-detection",
            "MetadataSender UDP JSON type mismatch");
    require(parsed["data"]["objects"].is_array(),
            "MetadataSender UDP JSON objects field is not array");
    require(!parsed["data"]["objects"].empty(),
            "MetadataSender UDP JSON objects unexpectedly empty");

    received_frame_ids.insert(parsed["frame_id"].get<std::string>());
  }

  require(received_frame_ids == expected_frame_ids,
          "MetadataSender UDP JSON frame-id set mismatch");

  guard.run = nullptr;
  run.close();

  std::cout << "[OK] graphpipes_yolo_metadata_sender_16stream_test passed\n";
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

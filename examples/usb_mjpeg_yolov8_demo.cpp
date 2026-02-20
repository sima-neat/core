#include "example_utils.h"
#include "support/obj_detection_utils.h"
#include "model/Model.h"
#include "nodes/common/Caps.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Session.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Config {
  std::string model_path;
  std::string device = "/dev/video0";
  int width = 1280;
  int height = 720;
  int fps = 30;
  int frames = -1;
  int topk = 100;
  float min_score = 0.60f;
  float nms_iou = 0.50f;
  std::string out_path = "usb_mjpeg_yolov8_preview.jpg";
};

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  --model <path.tar.gz>  YOLOv8 MPK tarball path (optional)\n"
            << "  --device </dev/videoX> USB camera device (default: /dev/video0)\n"
            << "  --width <int>          Capture width (default: 1280)\n"
            << "  --height <int>         Capture height (default: 720)\n"
            << "  --fps <int>            Capture FPS (default: 30)\n"
            << "  --frames <int>         Frames to process (-1 for continuous, default: -1)\n"
            << "  --min-score <float>    Box score threshold (default: 0.60)\n"
            << "  --topk <int>           Max decoded boxes (default: 100)\n"
            << "  --nms-iou <float>      NMS IoU threshold (default: 0.50)\n"
            << "  --out <file.jpg>       Preview output image path\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" || arg == "--device" || arg == "--width" || arg == "--height" ||
        arg == "--fps" || arg == "--frames" || arg == "--min-score" || arg == "--topk" ||
        arg == "--nms-iou" || arg == "--out") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
        return false;
      }
      ++i;
      const std::string val = argv[i];
      if (arg == "--model")
        cfg.model_path = val;
      else if (arg == "--device")
        cfg.device = val;
      else if (arg == "--width")
        cfg.width = std::stoi(val);
      else if (arg == "--height")
        cfg.height = std::stoi(val);
      else if (arg == "--fps")
        cfg.fps = std::stoi(val);
      else if (arg == "--frames")
        cfg.frames = std::stoi(val);
      else if (arg == "--min-score")
        cfg.min_score = std::stof(val);
      else if (arg == "--topk")
        cfg.topk = std::stoi(val);
      else if (arg == "--nms-iou")
        cfg.nms_iou = std::stof(val);
      else if (arg == "--out")
        cfg.out_path = val;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    return false;
  }

  if (cfg.width <= 0 || cfg.height <= 0 || cfg.fps <= 0 || cfg.topk <= 0) {
    std::cerr << "Invalid numeric argument(s)\n";
    return false;
  }
  if (cfg.frames == 0) {
    std::cerr << "--frames cannot be 0 (use -1 for continuous)\n";
    return false;
  }
  if ((cfg.width % 2) != 0 || (cfg.height % 2) != 0) {
    std::cerr << "NV12 path requires even --width and --height\n";
    return false;
  }
  return true;
}

std::string build_camera_caps(const Config& cfg) {
  std::ostringstream ss;
  ss << "image/jpeg,width=" << cfg.width << ",height=" << cfg.height << ",framerate=" << cfg.fps
     << "/1";
  return ss.str();
}

const std::vector<std::string>& coco_labels() {
  static const std::vector<std::string> kLabels = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
  return kLabels;
}

std::string class_name_for_id(int class_id) {
  const auto& labels = coco_labels();
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return "class_" + std::to_string(class_id);
}

void draw_boxes(cv::Mat& bgr, const std::vector<objdet::Box>& boxes, float min_score) {
  for (const auto& b : boxes) {
    if (b.score < min_score)
      continue;
    const int x1 = std::max(0, static_cast<int>(b.x1));
    const int y1 = std::max(0, static_cast<int>(b.y1));
    const int x2 = std::min(bgr.cols - 1, static_cast<int>(b.x2));
    const int y2 = std::min(bgr.rows - 1, static_cast<int>(b.y2));
    if (x2 <= x1 || y2 <= y1)
      continue;

    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    std::ostringstream label;
    label << class_name_for_id(b.class_id) << " " << b.score;
    cv::putText(bgr, label.str(), cv::Point(x1, std::max(0, y1 - 8)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
      usage(argv[0]);
      return 2;
    }

    if (cfg.model_path.empty()) {
      cfg.model_path = sima_examples::resolve_yolov8s_tar(fs::current_path());
      if (cfg.model_path.empty()) {
        std::cerr << "Missing YOLOv8 MPK tarball.\n";
        return 3;
      }
    }

    std::cout << "model=" << cfg.model_path << " device=" << cfg.device << " capture=" << cfg.width
              << "x" << cfg.height << "@" << cfg.fps << " frames=" << cfg.frames
              << " min_score=" << cfg.min_score << " out=" << cfg.out_path << "\n";

    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "NV12";
    model_opt.preproc.input_width = cfg.width;
    model_opt.preproc.input_height = cfg.height;
    model_opt.input_max_width = cfg.width;
    model_opt.input_max_height = cfg.height;
    model_opt.input_max_depth = 1;
    simaai::neat::Model model(cfg.model_path, model_opt);

    simaai::neat::Session cam;
    cam.custom("v4l2src device=" + cfg.device, simaai::neat::InputRole::Source);
    cam.custom(build_camera_caps(cfg));
    cam.custom("jpegdec");
    cam.add(simaai::neat::nodes::VideoConvert());
    cam.add(simaai::neat::nodes::CapsNV12SysMem(cfg.width, cfg.height, cfg.fps));
    cam.add(simaai::neat::nodes::Output());

    simaai::neat::InputOptions ysrc = model.input_appsrc_options(false);
    ysrc.media_type = "video/x-raw";
    ysrc.format = "NV12";
    ysrc.width = cfg.width;
    ysrc.height = cfg.height;
    ysrc.depth = 1;

    simaai::neat::Session yolo;
    yolo.add(simaai::neat::nodes::Input(ysrc));
    yolo.add(simaai::neat::nodes::groups::Preprocess(model));
    yolo.add(simaai::neat::nodes::groups::Infer(model));
    yolo.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", cfg.width, cfg.height,
                                                cfg.min_score, cfg.nms_iou, cfg.topk));
    yolo.add(simaai::neat::nodes::Output());

    simaai::neat::RunOptions cam_opt;
    cam_opt.queue_depth = 4;
    cam_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;

    simaai::neat::RunOptions det_opt;
    det_opt.queue_depth = 4;
    det_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;

    std::string nv12_err;
    simaai::neat::Tensor dummy_nv12;
    if (!sima_examples::make_blank_nv12_tensor(cfg.width, cfg.height, dummy_nv12, nv12_err)) {
      throw std::runtime_error("failed to create dummy NV12 tensor: " + nv12_err);
    }

    auto cam_run = cam.build(cam_opt);
    auto det_run = yolo.build(dummy_nv12, simaai::neat::RunMode::Async, det_opt);
    std::cout << "camera pipeline:\n" << cam.last_pipeline() << "\n";
    std::cout << "yolo pipeline:\n" << yolo.last_pipeline() << "\n";

    cv::Mat preview_bgr;
    bool wrote_preview = false;
    int i = 0;
    while (!g_stop.load() && (cfg.frames < 0 || i < cfg.frames)) {
      auto frame_opt = cam_run.pull_tensor(5000);
      if (!frame_opt.has_value()) {
        std::cerr << "camera pull timeout/closed at frame " << i << "\n";
        break;
      }
      std::string cvt_err;
      if (!sima_examples::nv12_to_bgr(*frame_opt, preview_bgr, cvt_err)) {
        std::cerr << "failed converting camera frame to BGR at frame " << i << " err=" << cvt_err
                  << "\n";
        ++i;
        continue;
      }

      simaai::neat::Sample out;
      try {
        out = det_run.push_and_pull(*frame_opt, 5000);
      } catch (const std::exception& ex) {
        std::cerr << "yolo push_and_pull failed at frame " << i << " err=" << ex.what() << "\n";
        ++i;
        continue;
      }

      std::vector<uint8_t> payload;
      std::string err;
      if (!sima_examples::extract_bbox_payload(out, payload, err)) {
        std::cerr << "bbox payload extraction failed at frame " << i << " err=" << err << "\n";
        ++i;
        continue;
      }

      std::vector<objdet::Box> boxes;
      try {
        boxes = objdet::parse_boxes_strict(payload, cfg.width, cfg.height, cfg.topk, false);
      } catch (const std::exception& ex) {
        std::cerr << "bbox parse failed at frame " << i << " err=" << ex.what() << "\n";
        ++i;
        continue;
      }

      draw_boxes(preview_bgr, boxes, cfg.min_score);
      if (!cv::imwrite(cfg.out_path, preview_bgr)) {
        throw std::runtime_error("failed to write preview image: " + cfg.out_path);
      }
      if ((i % 30) == 0) {
        std::cout << "updated preview image: " << cfg.out_path << " frame=" << i
                  << " detections=" << boxes.size() << "\n";
      }
      wrote_preview = true;
      ++i;
    }

    det_run.close();
    cam_run.close();
    if (!wrote_preview) {
      std::cerr << "no preview image written (no valid processed frames)\n";
      return 4;
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[ERR] " << ex.what() << "\n";
    return 1;
  }
}

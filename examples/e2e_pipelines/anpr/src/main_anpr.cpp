/**
 * ANPR Pipeline - C++ Port (Refactored)
 *
 * Architecture: Decoupled threaded pipelines following 4PipesYOLOMetadataReceiver.cpp pattern.
 * Key principle: Keep pipelines HOT - push/pull immediately after build() to avoid
 * the 15-second lifecycle timeout.
 */

#include "example_utils.h"
#include "anpr_utils.h"
#include "support/obj_detection_utils.h"

#include "pipeline/Session.h"
#include "pipeline/TensorCore.h"
#include "model/Model.h"
#include "nodes/io/Input.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "builder/ConfigJsonOverride.h"

#include <iostream>
#include <fstream>
#include <thread>
#include <deque>
#include <map>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>

namespace fs = std::filesystem;

using namespace simaai::neat;

// --- Configuration ---
static const int MODEL_WIDTH = 640;
static const int MODEL_HEIGHT = 384;
static const int STREAM_WIDTH = 1280;
static const int STREAM_HEIGHT = 720;
static const int OCR_SIZE = 256;
static const int VIDEO_PORT = 5000;

// --- Global State ---
static std::vector<std::string> VEH_CLASSES;
static std::vector<std::string> OCR_CHARS;

// ------------------------ BoundedQueue (drop-oldest) ------------------------
// From 4PipesYOLOMetadataReceiver.cpp - thread-safe bounded queue with drop policy

template <class T> class BoundedQueue {
public:
  explicit BoundedQueue(size_t cap) : cap_(cap) {}

  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  bool closed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return closed_;
  }

  // Push; if full, drop oldest (unless keep_latest is true, then drop incoming).
  bool push_drop_oldest(T item, bool keep_latest = false) {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_)
      return false;
    if (cap_ == 0)
      return false;
    if (q_.size() >= cap_) {
      if (keep_latest) {
        return true; // silently drop incoming
      }
      q_.pop_front();
    }
    q_.push_back(std::move(item));
    cv_.notify_one();
    return true;
  }

  // Pop with timeout (ms). Returns false on timeout or closed+empty.
  bool pop_wait(T& out, int timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (timeout_ms < 0) {
      cv_.wait(lk, [&]() { return closed_ || !q_.empty(); });
    } else {
      cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                   [&]() { return closed_ || !q_.empty(); });
    }
    if (q_.empty())
      return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
  }

private:
  size_t cap_ = 0;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> q_;
  bool closed_ = false;
};

// --- Helper Functions ---
static void load_labels(const std::string& path, std::vector<std::string>& out) {
  std::ifstream in(path);
  if (!in.is_open()) {
    std::cerr << "[WARN] Could not open labels file: " << path << std::endl;
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (!line.empty()) {
      out.push_back(line);
    }
  }
}

static bool get_arg_value(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

static bool init_nv12_meta(Tensor& t, int w, int h) {
  if (w <= 0 || h <= 0 || (w % 2) || (h % 2))
    return false;
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::HW;
  t.shape = {h, w};
  t.strides_bytes = {w, 1};
  t.byte_offset = 0;
  t.device = {DeviceType::CPU, 0};
  t.read_only = true;
  ImageSpec img;
  img.format = ImageSpec::PixelFormat::NV12;
  t.semantic.image = img;
  Plane y;
  y.role = PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;
  Plane uv;
  uv.role = PlaneRole::UV;
  uv.shape = {h / 2, w};
  uv.strides_bytes = {w, 1};
  uv.byte_offset = static_cast<int64_t>(w) * h;
  t.planes = {y, uv};
  return true;
}

static bool init_rgb_meta(Tensor& t, int w, int h) {
  if (w <= 0 || h <= 0)
    return false;
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::HWC;
  t.shape = {h, w, 3};
  t.strides_bytes = {w * 3, 3, 1};
  t.byte_offset = 0;
  t.device = {DeviceType::CPU, 0};
  t.read_only = true;
  ImageSpec img;
  img.format = ImageSpec::PixelFormat::RGB;
  t.semantic.image = img;
  t.planes.clear();
  return true;
}

static Sample make_blank_nv12_sample(int w, int h) {
  Tensor t;
  t.storage = make_cpu_owned_storage(static_cast<size_t>(w) * h * 3 / 2);
  if (t.storage && t.storage->data) {
    std::memset(t.storage->data, 0, t.storage->size_bytes);
  }
  init_nv12_meta(t, w, h);
  t.read_only = false;
  return make_tensor_sample("mysrc", std::move(t));
}

// Frame with detection results for output pipeline
struct FrameWithBoxes {
  Sample frame;
  std::vector<objdet::Box> boxes;
  std::map<int, std::string> ocr_texts;
};

int main(int argc, char** argv) {
  std::cout << "[INIT] ANPR Pipeline - Threaded Architecture\n";

  // 1. Parse Args and Config
  std::string rtsp_url;
  std::string rtsp_url_arg;
  std::string rtsp_host_arg;
  std::string rtsp_port_arg;
  if (get_arg_value(argc, argv, "--rtsp-url", rtsp_url_arg)) {
    rtsp_url = rtsp_url_arg;
    std::cout << "[INIT] Using RTSP URL from --rtsp-url: " << rtsp_url << std::endl;
  } else if (argc > 2 && std::string(argv[1]) == "--rtsp-list") {
    std::ifstream in(argv[2]);
    std::getline(in, rtsp_url);
    std::cout << "[INIT] Using first RTSP URL: " << rtsp_url << std::endl;
  } else {
    std::string host;
    std::string port;
    if (!get_arg_value(argc, argv, "--rtsp-host", rtsp_host_arg)) {
      const char* rtsp_host_env = std::getenv("RTSP_HOST");
      const char* host_ip = std::getenv("HOST_IP");
      host = rtsp_host_env ? rtsp_host_env : (host_ip ? host_ip : "10.42.0.1");
    } else {
      host = rtsp_host_arg;
    }
    if (!get_arg_value(argc, argv, "--rtsp-port", rtsp_port_arg)) {
      const char* rtsp_port_env = std::getenv("RTSP_PORT");
      port = rtsp_port_env ? rtsp_port_env : "8554";
    } else {
      port = rtsp_port_arg;
    }
    rtsp_url = "rtsp://" + host + ":" + port + "/cam1";
    std::cout << "[INIT] Using default RTSP URL: " << rtsp_url << std::endl;
  }

  std::string output_host_arg;
  std::string output_port_arg;
  std::string rtsp_fps_arg;
  std::string veh_tar_arg;
  std::string ocr_tar_arg;
  std::string output_host;
  int output_port = VIDEO_PORT;
  int rtsp_fallback_fps = 30;
  if (get_arg_value(argc, argv, "--udp-host", output_host_arg)) {
    output_host = output_host_arg;
  } else {
    const char* udp_host_env = std::getenv("UDP_HOST");
    const char* output_host_env = std::getenv("OUTPUT_HOST");
    const char* host_ip_env = std::getenv("HOST_IP");
    output_host = udp_host_env ? udp_host_env
                               : (output_host_env ? output_host_env
                                                  : (host_ip_env ? host_ip_env : "10.42.0.1"));
  }
  if (get_arg_value(argc, argv, "--udp-port", output_port_arg)) {
    output_port = std::stoi(output_port_arg);
  } else {
    const char* udp_port_env = std::getenv("UDP_PORT");
    const char* video_port_env = std::getenv("VIDEO_PORT");
    output_port = udp_port_env ? std::stoi(udp_port_env)
                               : (video_port_env ? std::stoi(video_port_env) : VIDEO_PORT);
  }
  if (get_arg_value(argc, argv, "--rtsp-fps", rtsp_fps_arg)) {
    int parsed = std::atoi(rtsp_fps_arg.c_str());
    if (parsed > 0)
      rtsp_fallback_fps = parsed;
  } else {
    const char* rtsp_fps_env = std::getenv("RTSP_FPS");
    if (rtsp_fps_env && *rtsp_fps_env) {
      int parsed = std::atoi(rtsp_fps_env);
      if (parsed > 0)
        rtsp_fallback_fps = parsed;
    }
  }

  std::cout << "[INIT] Output UDP: " << output_host << ":" << output_port << std::endl;
  std::cout << "[INIT] RTSP fallback FPS: " << rtsp_fallback_fps << std::endl;

  // Throttle Config
  const char* throttle_env = std::getenv("OCR_THROTTLE");
  bool enable_ocr_throttle =
      (!throttle_env || std::string(throttle_env) == "1" || std::string(throttle_env) == "true");
  if (enable_ocr_throttle) {
    std::cout << "[INIT] OCR Throttling ENABLED (Max 5 FPS)\n";
  } else {
    std::cout << "[INIT] OCR Throttling DISABLED (Running on every frame)\n";
  }

  std::string repo_root = "examples/e2e_pipelines/anpr";
  if (const char* env_root = std::getenv("REPO_ROOT_SIMA")) {
    repo_root = env_root;
  }

  std::string veh_tar = repo_root + "/models/YOLO_TRAFFIC_384H_640W_accurate_mod_mpk.tar.gz";
  std::string ocr_tar = repo_root + "/models/YOLO_OCR_256_mod_2_0_INT8_accurate_mpk.tar.gz";
  if (get_arg_value(argc, argv, "--vehicle-model", veh_tar_arg)) {
    veh_tar = veh_tar_arg;
    std::cout << "[INIT] Using vehicle model from --vehicle-model: " << veh_tar << std::endl;
  }
  if (get_arg_value(argc, argv, "--ocr-model", ocr_tar_arg)) {
    ocr_tar = ocr_tar_arg;
    std::cout << "[INIT] Using OCR model from --ocr-model: " << ocr_tar << std::endl;
  }
  std::string labels_veh = repo_root + "/models/labels/labels_vehicle.txt";
  std::string labels_ocr = repo_root + "/models/labels/labels_ocr.txt";

  load_labels(labels_veh, VEH_CLASSES);
  load_labels(labels_ocr, OCR_CHARS);

  if (VEH_CLASSES.empty()) {
    VEH_CLASSES = {"lp", "2w", "3w", "helmet", "no_helmet", "lmv", "hmv"};
  }
  if (OCR_CHARS.empty()) {
    OCR_CHARS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
                 "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                 "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
  }
  std::cout << "[INIT] Loaded " << VEH_CLASSES.size() << " vehicle classes, " << OCR_CHARS.size()
            << " OCR chars\n";

  // --- Global control ---
  std::atomic<bool> stop{false};
  std::atomic<bool> det_ready{false};
  std::atomic<int64_t> frame_counter{0};

  // --- Bounded queues for inter-thread communication ---
  BoundedQueue<Sample> decoded_q(16);        // RTSP -> Detection
  BoundedQueue<FrameWithBoxes> output_q(16); // Detection -> Output

  // ========================================================================
  // PIPELINE 1: RTSP + Decode (using RtspInputGroup)
  // ========================================================================
  std::cout << "[BUILD] Building RTSP+Decode pipeline...\n";

  nodes::groups::RtspDecodedInputOptions rtsp_opts;
  rtsp_opts.url = rtsp_url;
  rtsp_opts.latency_ms = 200;
  rtsp_opts.tcp = true;
  rtsp_opts.decoder_name = "decoder";
  rtsp_opts.out_format = "NV12";
  rtsp_opts.decoder_raw_output = true;
  rtsp_opts.insert_queue = true;
  rtsp_opts.auto_caps_from_stream = true;
  rtsp_opts.fallback_h264_fps = rtsp_fallback_fps;
  rtsp_opts.fallback_h264_width = STREAM_WIDTH;
  rtsp_opts.fallback_h264_height = STREAM_HEIGHT;

  Session rtsp_sess;
  rtsp_sess.add(nodes::groups::RtspDecodedInput(rtsp_opts));
  rtsp_sess.add(nodes::Output());

  RunOptions rtsp_run_opts;
  rtsp_run_opts.queue_depth = 4;
  rtsp_run_opts.overflow_policy = OverflowPolicy::KeepLatest;

  auto rtsp_run = rtsp_sess.build(rtsp_run_opts);
  std::cout << "[BUILD] RTSP+Decode pipeline built\n";

  // IMMEDIATELY start RTSP worker thread to keep pipeline hot
  std::thread rtsp_thread([&]() {
    std::cout << "[RTSP] Worker started\n";
    int pull_count = 0;
    int timeout_count = 0;
    auto last_time = std::chrono::steady_clock::now();
    while (!stop.load()) {
      Sample dec_sample;
      PullError perr;
      auto st = rtsp_run.pull(200, dec_sample, &perr);

      if (st == PullStatus::Timeout) {
        timeout_count++;
        if (timeout_count % 50 == 0) {
          std::cout << "[RTSP] No frames pulled for ~" << (timeout_count * 200 / 1000)
                    << "s (still waiting)\n";
        }
        continue;
      }
      if (st == PullStatus::Closed) {
        std::cerr << "[RTSP] Pipeline closed\n";
        break;
      }
      if (st == PullStatus::Error) {
        std::cerr << "[RTSP] Error: " << perr.message << "\n";
        continue;
      }

      timeout_count = 0;
      pull_count++;
      dec_sample.frame_id = frame_counter.fetch_add(1);

      // Copy to CPU contiguous buffer for processing
      if (dec_sample.tensor) {
        // NV12 size = W * H * 1.5
        size_t req_size = static_cast<size_t>(STREAM_WIDTH) * STREAM_HEIGHT * 3 / 2;
        auto storage = make_cpu_owned_storage(req_size);

        if (dec_sample.tensor->copy_nv12_contiguous_to(static_cast<uint8_t*>(storage->data),
                                                       req_size)) {
          Tensor cpu_tensor;
          cpu_tensor.storage = std::move(storage);
          init_nv12_meta(cpu_tensor, STREAM_WIDTH, STREAM_HEIGHT);
          cpu_tensor.read_only = false;
          dec_sample.tensor = std::move(cpu_tensor);
        } else {
          std::cerr << "[WARN] RTSP copy_nv12_contiguous_to failed\n";
          continue;
        }
      }

      // Push to detection queue (drop oldest if full)
      decoded_q.push_drop_oldest(std::move(dec_sample));

      if (pull_count % 100 == 0) {
        auto now = std::chrono::steady_clock::now();
        auto dur_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - last_time).count();
        double fps = (dur_seconds > 0) ? (100.0 / dur_seconds) : 0.0;
        last_time = now;
        std::cout << "[RTSP] Pulled " << pull_count << " frames (FPS: " << fps << ")\n";
      }
    }
    decoded_q.close();
    std::cout << "[RTSP] Worker stopped\n";
  });

  // ========================================================================
  // PIPELINE 2: Vehicle Detection
  // ========================================================================
  std::cout << "[BUILD] Building Detection pipeline...\n";

  Model::Options veh_model_opt;
  veh_model_opt.media_type = "video/x-raw";
  veh_model_opt.format = "NV12";
  veh_model_opt.preproc.input_width = STREAM_WIDTH;
  veh_model_opt.preproc.input_height = STREAM_HEIGHT;
  veh_model_opt.input_max_width = STREAM_WIDTH;
  veh_model_opt.input_max_height = STREAM_HEIGHT;
  veh_model_opt.input_max_depth = 1;
  Model veh_model(veh_tar, veh_model_opt);

  Session det_sess;
  InputOptions det_src_opts = veh_model.input_appsrc_options(false);
  det_src_opts.format = "NV12";
  det_src_opts.width = STREAM_WIDTH;
  det_src_opts.height = STREAM_HEIGHT;
  det_src_opts.is_live = true;
  det_src_opts.do_timestamp = true;
  det_src_opts.buffer_name = "decoder"; // Match decoder_name
  det_src_opts.caps_override = "video/x-raw,format=NV12,width=1280,height=720,framerate=0/1";
  det_sess.add(nodes::Input(det_src_opts));

  det_sess.add(nodes::groups::Preprocess(veh_model));
  det_sess.add(nodes::groups::MLA(veh_model));

  // Create BoxDecode node and override num_classes to 7 (patching default 80)
  auto box_dec = nodes::SimaBoxDecode(veh_model, "yolo", MODEL_WIDTH, MODEL_HEIGHT, 0.4f, 0.5f, 20);
  std::shared_ptr<simaai::neat::SimaBoxDecode> box_node =
      std::dynamic_pointer_cast<simaai::neat::SimaBoxDecode>(box_dec);

  if (box_node) {
    box_node->override_config_json(
        [](nlohmann::json& j) {
          j["num_classes"] = 7;
          std::cout << "[Config] Patched num_classes to 7\n";
        },
        "boxdecode");
    det_sess.add(box_node);
  } else {
    std::cerr << "[WARN] Failed to look up SimaBoxDecode for config override. Adding as is.\n";
    det_sess.add(box_dec);
  }

  det_sess.add(nodes::Output());

  Tensor dummy_nv12;
  dummy_nv12.storage =
      make_cpu_owned_storage(static_cast<size_t>(STREAM_WIDTH) * STREAM_HEIGHT * 3 / 2);
  init_nv12_meta(dummy_nv12, STREAM_WIDTH, STREAM_HEIGHT);

  RunOptions det_run_opts;
  det_run_opts.queue_depth = 4;
  det_run_opts.overflow_policy = OverflowPolicy::KeepLatest;

  auto det_run = det_sess.build(dummy_nv12, RunMode::Async, det_run_opts);
  // ========================================================================
  // PIPELINE 3: OCR (Crop -> Recognition)
  // ========================================================================
  std::cout << "[BUILD] Building OCR pipeline...\n";

  // Setup OCR model (256x256 RGB input)
  Model::Options ocr_model_opt;
  ocr_model_opt.media_type = "video/x-raw";
  ocr_model_opt.format = "RGB";
  ocr_model_opt.preproc.input_width = OCR_SIZE;
  ocr_model_opt.preproc.input_height = OCR_SIZE;
  ocr_model_opt.input_max_width = OCR_SIZE;
  ocr_model_opt.input_max_height = OCR_SIZE;
  ocr_model_opt.input_max_depth = 1;
  Model ocr_model(ocr_tar, ocr_model_opt);
  Session ocr_sess;

  InputOptions ocr_src_opts;
  ocr_src_opts.format = "RGB";
  ocr_src_opts.width = OCR_SIZE;
  ocr_src_opts.height = OCR_SIZE;
  ocr_src_opts.is_live = true;
  ocr_src_opts.do_timestamp = true;

  ocr_sess.add(nodes::Input(ocr_src_opts));
  ocr_sess.add(nodes::groups::Preprocess(ocr_model));
  ocr_sess.add(nodes::groups::MLA(ocr_model));

  // OCR uses SimaBoxDecode with num_classes=36 (0-9, A-Z)
  auto ocr_dec_node = nodes::SimaBoxDecode(ocr_model, "yolo", OCR_SIZE, OCR_SIZE, 0.3f, 0.3f, 50);
  std::shared_ptr<simaai::neat::SimaBoxDecode> ocr_box_ptr =
      std::dynamic_pointer_cast<simaai::neat::SimaBoxDecode>(ocr_dec_node);

  if (ocr_box_ptr) {
    ocr_box_ptr->override_config_json([](nlohmann::json& j) { j["num_classes"] = 36; },
                                      "boxdecode");
    ocr_sess.add(ocr_box_ptr);
  } else {
    std::cerr << "[WARN] Failed to cast OCR BoxDecode node\n";
    ocr_sess.add(ocr_dec_node);
  }

  ocr_sess.add(nodes::Output());

  RunOptions ocr_run_opts;
  ocr_run_opts.queue_depth = 4;
  ocr_run_opts.overflow_policy = OverflowPolicy::Block;

  Tensor dummy_ocr;
  init_rgb_meta(dummy_ocr, OCR_SIZE, OCR_SIZE);

  // Allocate dummy storage for build() to succeed
  auto dummy_store = make_cpu_owned_storage(OCR_SIZE * OCR_SIZE * 3);
  std::memset(dummy_store->data, 0, dummy_store->size_bytes);
  dummy_ocr.storage = std::move(dummy_store);

  auto ocr_run = ocr_sess.build(dummy_ocr, RunMode::Async, ocr_run_opts);
  std::cout << "[BUILD] OCR pipeline built\n";

  // ========================================================================
  // PIPELINE 4: Output (Encode + UDP)
  // ========================================================================
  std::cout << "[BUILD] Detection pipeline built\n";
  det_ready.store(true);

  // IMMEDIATELY start Detection worker thread
  std::thread det_thread([&]() {
    std::cout << "[DET] Worker started\n";
    int det_count = 0;
    int push_count = 0;
    auto last_time = std::chrono::steady_clock::now();

    // Throttling for OCR (max 5 FPS)
    auto ocr_last_time = std::chrono::steady_clock::now();
    // Force first run
    ocr_last_time -= std::chrono::seconds(1);

    int window_boxes = 0;
    int window_ocr_reads = 0;

    // FIFO queue: frames are processed in order, so we match by order not by frame_id
    std::deque<Sample> pending_frames;

    while (!stop.load()) {
      // Pull decoded frames and push to detector
      Sample input;
      if (decoded_q.pop_wait(input, 50)) {
        // Store frame for later (we'll draw on it when detection result arrives)
        pending_frames.push_back(std::move(input));

        // Push the last frame's tensor to detector
        if (pending_frames.back().tensor) {
          if (det_run.try_push(*pending_frames.back().tensor)) {
            push_count++;
          }
        }

        // Limit pending queue size
        while (pending_frames.size() > 32) {
          pending_frames.pop_front();
        }
      }

      // Pull detection results
      Sample det_out;
      PullError perr;
      auto st = det_run.pull(10, det_out, &perr);

      if (st == PullStatus::Ok) {
        det_count++;

        // Match with oldest pending frame (FIFO order)
        if (!pending_frames.empty()) {
          FrameWithBoxes result;
          result.frame = std::move(pending_frames.front());
          pending_frames.pop_front();

          // Parse boxes
          std::string err;
          std::vector<uint8_t> payload;
          if (objdet::extract_bbox_payload(det_out, payload, err)) {
            result.boxes = objdet::parse_boxes_lenient(payload, MODEL_WIDTH, MODEL_HEIGHT, 20);

            window_boxes += result.boxes.size();

            // Scale boxes from MODEL to STREAM dimensions
            float scale_x = static_cast<float>(STREAM_WIDTH) / MODEL_WIDTH;
            float scale_y = static_cast<float>(STREAM_HEIGHT) / MODEL_HEIGHT;

            // Access NV12 planes for cropping
            uint8_t* nv12_data = nullptr;
            if (result.frame.tensor && result.frame.tensor->storage) {
              nv12_data = static_cast<uint8_t*>(result.frame.tensor->storage->data);
            }

            int idx = 0;
            for (auto& box : result.boxes) {
              box.x1 *= scale_x;
              box.y1 *= scale_y;
              box.x2 *= scale_x;
              box.y2 *= scale_y;

              // OCR Logic: Only run on 'lp' class (0)
              // VEH_CLASSES = {"lp", "2w", ...} -> 0 is lp
              // Throttle: Max 5 OCRs per second
              if (nv12_data && box.class_id == 0) {
                auto now = std::chrono::steady_clock::now();
                bool should_run_ocr = true;

                if (enable_ocr_throttle) {
                  auto elapsed =
                      std::chrono::duration_cast<std::chrono::duration<double>>(now - ocr_last_time)
                          .count();
                  if (elapsed < 0.2) {
                    should_run_ocr = false;
                  }
                }

                if (should_run_ocr) {
                  ocr_last_time = now;

                  int x1 = std::max(0, (int)box.x1);
                  int y1 = std::max(0, (int)box.y1);
                  int x2 = std::min(STREAM_WIDTH, (int)box.x2);
                  int y2 = std::min(STREAM_HEIGHT, (int)box.y2);

                  uint8_t* y_plane = nv12_data;
                  uint8_t* uv_plane = nv12_data + (STREAM_WIDTH * STREAM_HEIGHT);

                  cv::Mat crop_bgr = anpr::crop_nv12_to_bgr(y_plane, uv_plane, STREAM_WIDTH,
                                                            STREAM_HEIGHT, x1, y1, x2, y2);

                  if (!crop_bgr.empty()) {
                    cv::Mat ocr_input_img = anpr::preprocess_crop_for_ocr(crop_bgr, OCR_SIZE);

                    Tensor ocr_tensor;
                    init_rgb_meta(ocr_tensor, OCR_SIZE, OCR_SIZE);
                    ocr_tensor.read_only = false;
                    auto storage = make_cpu_owned_storage(OCR_SIZE * OCR_SIZE * 3);
                    std::memcpy(storage->data, ocr_input_img.data, storage->size_bytes);
                    ocr_tensor.storage = std::move(storage);

                    // Run OCR
                    if (ocr_run.try_push(ocr_tensor)) {
                      Sample ocr_out;
                      if (ocr_run.pull(200, ocr_out, nullptr) == PullStatus::Ok) {
                        std::vector<uint8_t> ocr_payload;
                        std::string ocr_err;
                        if (objdet::extract_bbox_payload(ocr_out, ocr_payload, ocr_err)) {
                          auto char_boxes =
                              objdet::parse_boxes_lenient(ocr_payload, OCR_SIZE, OCR_SIZE, 50);

                          // Sort by X coordinate
                          std::sort(char_boxes.begin(), char_boxes.end(),
                                    [](const objdet::Box& a, const objdet::Box& b) {
                                      return a.x1 < b.x1;
                                    });

                          std::string text;
                          for (const auto& cb : char_boxes) {
                            if (cb.class_id >= 0 && cb.class_id < (int)OCR_CHARS.size()) {
                              text += OCR_CHARS[cb.class_id];
                            }
                          }

                          if (!text.empty()) {
                            result.ocr_texts[idx] = text;
                            window_ocr_reads++;
                            if (window_ocr_reads < 10) {
                              std::cout << "[OCR] Frame " << det_count << " Box " << idx
                                        << " (LP): " << text << "\n";
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
              idx++;
            } // end for boxes
          } else if (!err.empty()) {
            std::cerr << "[DET] Extract payload error: " << err << "\n";
          }

          output_q.push_drop_oldest(std::move(result));
        }

        if (det_count % 100 == 0) {
          auto now = std::chrono::steady_clock::now();
          auto dur_seconds =
              std::chrono::duration_cast<std::chrono::duration<double>>(now - last_time).count();
          double fps = (dur_seconds > 0) ? (100.0 / dur_seconds) : 0.0;
          last_time = now;

          std::cout << "[DET] Processed " << det_count << " frames (FPS: " << fps
                    << "). Boxes: " << window_boxes << ", OCR reads: " << window_ocr_reads << "\n";
          window_boxes = 0;
          window_ocr_reads = 0;
        }
      }
    }
    output_q.close();
    std::cout << "[DET] Worker stopped\n";
  });

  // ========================================================================
  // PIPELINE 4: Output (Encode + UDP)
  // ========================================================================
  std::cout << "[BUILD] Building Output pipeline...\n";

  Session out_sess;
  InputOptions out_src_opts;
  out_src_opts.format = "NV12";
  out_src_opts.width = STREAM_WIDTH;
  out_src_opts.height = STREAM_HEIGHT;
  out_src_opts.caps_override = "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1";
  out_src_opts.is_live = true;
  out_src_opts.do_timestamp = true;
  out_sess.add(nodes::Input(out_src_opts));
  out_sess.add(nodes::H264EncodeSima(STREAM_WIDTH, STREAM_HEIGHT, 30, 4000));
  H264ParseOptions h264_parse_opt;
  h264_parse_opt.config_interval = 1;
  h264_parse_opt.enforce_caps = true;
  h264_parse_opt.alignment = H264ParseOptions::Alignment::AU;
  h264_parse_opt.stream_format = H264ParseOptions::StreamFormat::ByteStream;
  out_sess.add(nodes::H264Parse(h264_parse_opt));
  out_sess.add(nodes::H264Packetize()); // Defaults config_interval=1
  UdpOutputOptions udp_opts;
  udp_opts.host = output_host;
  udp_opts.port = output_port;
  out_sess.add(nodes::UdpOutput(udp_opts));

  Tensor dummy_out;
  dummy_out.storage =
      make_cpu_owned_storage(static_cast<size_t>(STREAM_WIDTH) * STREAM_HEIGHT * 3 / 2);
  init_nv12_meta(dummy_out, STREAM_WIDTH, STREAM_HEIGHT);

  RunOptions out_run_opts;
  out_run_opts.queue_depth = 4;
  out_run_opts.overflow_policy = OverflowPolicy::KeepLatest;

  auto out_run = out_sess.build(dummy_out, RunMode::Async, out_run_opts);
  std::cout << "[BUILD] Output pipeline built\n";
  std::cout << "[DEBUG] Output Pipeline: " << out_sess.describe_backend() << "\n";

  if (output_q.size() == 0) {
    const int warmup_frames = 3;
    for (int i = 0; i < warmup_frames; ++i) {
      FrameWithBoxes warm;
      warm.frame = make_blank_nv12_sample(STREAM_WIDTH, STREAM_HEIGHT);
      output_q.push_drop_oldest(std::move(warm));
    }
    std::cout << "[OUT] Seeded " << warmup_frames << " warmup frames\n";
  }

  // IMMEDIATELY start Output worker thread
  std::thread out_thread([&]() {
    std::cout << "[OUT] Worker started\n";
    int out_pushed = 0;
    int out_dropped = 0;
    int idle_polls = 0;
    auto last_time = std::chrono::steady_clock::now();
    auto last_stats_time = std::chrono::steady_clock::now();

    while (!stop.load()) {
      FrameWithBoxes result;
      if (!output_q.pop_wait(result, 200)) {
        idle_polls++;
        if (idle_polls % 25 == 0) {
          std::cout << "[OUT] Waiting for frames (output_q empty, size=" << output_q.size()
                    << ")\n";
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 5) {
          auto stats = out_run.stats();
          std::cout << "[OUT] Stats inputs_enqueued=" << stats.inputs_enqueued
                    << " inputs_dropped=" << stats.inputs_dropped
                    << " outputs_ready=" << stats.outputs_ready
                    << " outputs_pulled=" << stats.outputs_pulled
                    << " avg_latency_ms=" << stats.avg_latency_ms << "\n";
          last_stats_time = now;
        }
        continue;
      }
      idle_polls = 0;

      // Draw overlays on the frame
      if (result.frame.tensor && result.frame.tensor->storage) {
        uint8_t* base = static_cast<uint8_t*>(result.frame.tensor->storage->data);
        uint8_t* y = base;
        uint8_t* uv = base + STREAM_WIDTH * STREAM_HEIGHT;

        for (size_t i = 0; i < result.boxes.size(); ++i) {
          auto& box = result.boxes[i];
          std::string label = (box.class_id < static_cast<int>(VEH_CLASSES.size()))
                                  ? VEH_CLASSES[box.class_id]
                                  : "unk";
          auto color = anpr::get_color_for_label(label);
          anpr::draw_nv12_box(y, uv, STREAM_WIDTH, STREAM_HEIGHT, static_cast<int>(box.x1),
                              static_cast<int>(box.y1), static_cast<int>(box.x2),
                              static_cast<int>(box.y2), color);

          std::string display_text = label;
          auto text_it = result.ocr_texts.find(static_cast<int>(i));
          if (text_it != result.ocr_texts.end() && !text_it->second.empty()) {
            display_text += ": " + text_it->second;
            if (out_pushed % 100 == 0) { // Throttle draw debugs too
              std::cout << "[DRAW] Drawing OCR for box " << i << ": " << text_it->second << "\n";
            }
          }
          anpr::draw_nv12_text(y, STREAM_WIDTH, STREAM_HEIGHT, display_text,
                               static_cast<int>(box.x1), static_cast<int>(box.y1) - 5);
        }
      }

      // Push to output pipeline
      if (!result.frame.tensor || !out_run.try_push(*result.frame.tensor)) {
        out_dropped++;
        if (out_dropped % 100 == 0) {
          std::cerr << "[OUT] Dropped " << out_dropped << " frames (output backpressure)"
                    << " running=" << out_run.running() << " can_push=" << out_run.can_push()
                    << "\n";
          const std::string last_err = out_run.last_error();
          if (!last_err.empty()) {
            std::cerr << "[OUT] Last error: " << last_err << "\n";
          }
        }
        continue;
      }
      out_pushed++;

      if (out_pushed % 100 == 0) {
        auto now = std::chrono::steady_clock::now();
        auto dur_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - last_time).count();
        double fps = (dur_seconds > 0) ? (100.0 / dur_seconds) : 0.0;
        last_time = now;
        std::cout << "[OUT] Pushed " << out_pushed << " frames (FPS: " << fps
                  << ", dropped: " << out_dropped << ")\n";
      }

      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 5) {
        auto stats = out_run.stats();
        std::cout << "[OUT] Stats inputs_enqueued=" << stats.inputs_enqueued
                  << " inputs_dropped=" << stats.inputs_dropped
                  << " outputs_ready=" << stats.outputs_ready
                  << " outputs_pulled=" << stats.outputs_pulled
                  << " avg_latency_ms=" << stats.avg_latency_ms << "\n";
        last_stats_time = now;
      }
    }
    std::cout << "[OUT] Worker stopped\n";
  });

  // ========================================================================
  // Main thread: Just wait for shutdown
  // ========================================================================
  std::cout << "[RUN] All pipelines running. Press Ctrl+C to stop.\n";

  // Simple signal handling (in real code, use proper signal handlers)
  while (!stop.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  // Cleanup
  stop.store(true);
  decoded_q.close();
  output_q.close();

  rtsp_thread.join();
  det_thread.join();
  out_thread.join();

  std::cout << "[DONE] Shutdown complete\n";
  return 0;
}

#include "pipeline/Session.h"
#include "nodes/common/Output.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "model/Model.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using sima_yolov8_test::append_note;
using sima_yolov8_test::sanitize_note;
using sima_yolov8_test::step_log;

struct AsyncTestConfig {
  int iters = 200;
  int warm = 100;
  double min_fps = 330.0;
  float min_score = 0.52f;
  float min_iou = 0.30f;
};

struct RunSummary {
  bool ok = false;
  int outputs = 0;
  double avg_fps = 0.0;
  std::string note;
};

std::string hex_prefix(const std::vector<uint8_t>& payload, size_t max_bytes) {
  std::ostringstream ss;
  ss << std::hex << std::setfill('0');
  const size_t count = std::min(payload.size(), max_bytes);
  for (size_t i = 0; i < count; ++i) {
    ss << std::setw(2) << static_cast<int>(payload[i]);
    if (i + 1 < count)
      ss << " ";
  }
  return ss.str();
}

std::string bbox_payload_summary(const std::vector<uint8_t>& payload, int expected_topk) {
  std::ostringstream ss;
  ss << "bytes=" << payload.size();
  if (payload.size() >= sizeof(uint32_t)) {
    uint32_t header = 0;
    std::memcpy(&header, payload.data(), sizeof(header));
    const size_t data_bytes = payload.size() - sizeof(uint32_t);
    const size_t max_boxes = data_bytes / 24;
    ss << " header=" << header << " data_bytes=" << data_bytes << " data_rem=" << (data_bytes % 24)
       << " max_boxes=" << max_boxes;
    if (expected_topk > 0) {
      const size_t expected_bytes = sizeof(uint32_t) + static_cast<size_t>(expected_topk) * 24;
      ss << " expected_bytes=" << expected_bytes;
    }
  }
  return ss.str();
}

bool pull_with_timeout(simaai::neat::Run& async, int pull_timeout_ms, int max_timeouts,
                       int& timeout_count, std::optional<simaai::neat::Sample>& out,
                       std::string& err) {
  while (true) {
    simaai::neat::Sample temp;
    simaai::neat::PullError perr;
    const simaai::neat::PullStatus status = async.pull(pull_timeout_ms, temp, &perr);
    if (status == simaai::neat::PullStatus::Ok) {
      out = temp;
      timeout_count = 0;
      return true;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      timeout_count += 1;
      if (max_timeouts > 0 && timeout_count >= max_timeouts) {
        err = "Run::pull: timeout";
        return false;
      }
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      err = "Run::pull: closed";
      return false;
    }
    err = perr.message.empty() ? "Run::pull: error" : perr.message;
    return false;
  }
}

RunSummary run_yolov8_async_tput(const std::string& tar_gz, const cv::Mat& img,
                                 const AsyncTestConfig& cfg) {
  RunSummary res;

  require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

  simaai::neat::Model::Options model_opt;
  model_opt.format = "BGR";
  model_opt.input_max_width = img.cols;
  model_opt.input_max_height = img.rows;
  model_opt.input_max_depth = 3;
  auto model = simaai::neat::Model(tar_gz, model_opt);
  const int topk = 100;

  simaai::neat::Session p;

  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Preprocess(model));
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", img.cols, img.rows, cfg.min_score, 0.5f,
                                           topk));

  p.add(simaai::neat::nodes::Output());

  const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();

  step_log("async: before build");
  auto async = p.build(img);
  step_log("async: after build");

  const int warm_timeout_ms = 60000;
  const int pull_timeout_ms = 1000;
  const int max_timeouts = 3;

  try {
    step_log("async: before warmup");
    async.warmup(img, cfg.warm, warm_timeout_ms);
    step_log("async: after warmup");
  } catch (const std::exception& e) {
    res.ok = false;
    append_note(res.note, "warmup_error=" + sanitize_note(e.what()));
    return res;
  }

  std::mutex error_mu;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;

  std::atomic<int> measured_out{0};
  std::string consumer_error;
  std::atomic<bool> stop_requested{false};
  std::vector<std::vector<uint8_t>> verify_payloads;
  verify_payloads.reserve(static_cast<size_t>(cfg.iters));

  auto set_error = [&](const std::string& msg) {
    {
      std::lock_guard<std::mutex> lock(error_mu);
      if (consumer_error.empty()) {
        consumer_error = msg;
      }
      stop_requested.store(true);
    }
  };

  std::thread consumer([&]() {
    int timeout_count = 0;
    try {
      while (true) {
        std::optional<simaai::neat::Sample> out;
        std::string err;
        if (!pull_with_timeout(async, pull_timeout_ms, max_timeouts, timeout_count, out, err)) {
          if (!err.empty()) {
            set_error(err);
          }
          break;
        }
        if (!out.has_value())
          break;
        const int m = measured_out.fetch_add(1) + 1;
        std::vector<uint8_t> payload;
        if (!objdet::extract_bbox_payload(*out, m - 1, payload, err)) {
          set_error(err);
          break;
        }
        verify_payloads.push_back(std::move(payload));
        if (m == cfg.iters)
          break;
      }
    } catch (const std::exception& e) {
      set_error(e.what());
    }
  });

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.iters; ++i) {
    if (stop_requested.load())
      break;
    (void)async.push(img);
  }

  async.close_input();
  consumer.join();
  end = std::chrono::steady_clock::now();

  const double elapsed_s = std::chrono::duration<double>(end - start).count();

  res.outputs = measured_out.load();
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;
  res.ok = (res.outputs == cfg.iters);

  if (!consumer_error.empty()) {
    append_note(res.note, "async_pull_error=" + sanitize_note(consumer_error));
    res.ok = false;
  }
  if (elapsed_s <= 0.0) {
    append_note(res.note, "async_timing_incomplete");
    res.ok = false;
  }

  const size_t payloads = verify_payloads.size();
  if (payloads != static_cast<size_t>(cfg.iters)) {
    append_note(res.note, "verify_payloads=" + std::to_string(payloads));
    res.ok = false;
  }
  const size_t verify_count = std::min(payloads, static_cast<size_t>(cfg.iters));
  for (size_t i = 0; i < verify_count; ++i) {
    std::vector<objdet::Box> boxes;
    try {
      boxes = objdet::parse_boxes_strict(verify_payloads[i], img.cols, img.rows, topk, false);
    } catch (const std::exception& ex) {
      std::cerr << "[WARN] bbox parse failed iter=" << i << " err=" << ex.what() << " "
                << bbox_payload_summary(verify_payloads[i], topk)
                << " prefix=" << hex_prefix(verify_payloads[i], 64) << "\n";
      append_note(res.note, "bbox_parse_failed iter=" + std::to_string(i));
      res.ok = false;
      break;
    }
    const objdet::MatchResult match =
        objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
    if (!match.ok) {
      append_note(res.note, "verify_mismatch iter=" + std::to_string(i) + " " + match.note);
      res.ok = false;
      break;
    }
  }

  if (!res.ok) {
    const std::string last_err = async.last_error();
    if (!last_err.empty()) {
      append_note(res.note, "async_last_error=" + sanitize_note(last_err));
    }
  }

  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    AsyncTestConfig cfg;

    RunSummary res = run_yolov8_async_tput(tar_gz, img_bgr, cfg);
    if (res.avg_fps <= cfg.min_fps) {
      append_note(res.note, "avg_fps=" + std::to_string(res.avg_fps));
      append_note(res.note, "min_fps=" + std::to_string(cfg.min_fps));
      res.ok = false;
    }

    std::cout << "ASYNC_TPUT outputs=" << res.outputs << " avg_fps=" << res.avg_fps
              << " ok=" << (res.ok ? "1" : "0") << " note=" << res.note << "\n";

    return res.ok ? 0 : 2;
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

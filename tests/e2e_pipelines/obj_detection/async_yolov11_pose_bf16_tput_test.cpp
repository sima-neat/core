#include "gst/GstHelpers.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Session.h"

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {
using json = nlohmann::json;

using sima_yolov8_test::append_note;
using sima_yolov8_test::sanitize_note;
using sima_yolov8_test::step_log;

struct AsyncTestConfig {
  int iters = 40;
  int warm = 8;
  double min_fps = 1.0;
  int pull_timeout_ms = 1000;
  int warm_timeout_ms = 20000;
  int max_timeouts = 4;
};

struct RunSummary {
  std::string name;
  bool ok = false;
  int outputs = 0;
  int payloads = 0;
  double avg_fps = 0.0;
  std::string note;
  std::string pipeline;
};

struct PipelineConfigs {
  std::string tess_cfg;
  std::string pre_cfg;
  std::string mla_cfg;
  std::string box_cfg;
  std::string post_cfg;
};

struct PluginFactories {
  std::string processcvu;
  std::string processmla;
  std::string boxdecode;
};

std::string resolve_yolov11_pose_bf16_tar_or_skip(const fs::path& root) {
  const char* env = std::getenv("SIMA_YOLO11_POSE_BF16_TAR");
  if (env && *env && file_exists(env))
    return env;
  const char* generic = std::getenv("SIMA_MODEL_TAR");
  if (generic && *generic && file_exists(generic))
    return generic;

  const fs::path local = root / "tmp" / "yolov11_mpk_bf16.tar.gz";
  if (fs::exists(local))
    return local.string();

  static const fs::path hardcoded =
      "/home/sima/stable_pipeline_session/PipelineSession/tmp/yolov11_mpk_bf16.tar.gz";
  if (fs::exists(hardcoded))
    return hardcoded.string();

  skip_long_test_exception(
      "Missing yolov11 BF16 tarball. Set SIMA_MODEL_TAR (or SIMA_YOLO11_POSE_BF16_TAR) or place "
      "tmp/yolov11_mpk_bf16.tar.gz.");
  return {};
}

PipelineConfigs extract_and_resolve_configs_or_throw(const std::string& tar_gz,
                                                     const fs::path& root) {
  const fs::path extract_root = root / "tmp" / "yolov11_pose_bf16_extract";
  std::error_code ec;
  fs::remove_all(extract_root, ec);
  fs::create_directories(extract_root, ec);
  const std::string cmd = "tar -xzf '" + tar_gz + "' -C '" + extract_root.string() + "'";
  const int rc = std::system(cmd.c_str());
  require(rc == 0, "Failed to extract model tarball: " + tar_gz);

  PipelineConfigs cfg;
  cfg.tess_cfg = (extract_root / "0_tessellate.json").string();
  cfg.pre_cfg = (extract_root / "0_preproc.json").string();
  cfg.mla_cfg = (extract_root / "0_process_mla.runtime.json").string();
  cfg.box_cfg = (extract_root / "0_boxdecoder.json").string();
  cfg.post_cfg = (extract_root / "0_postproc.json").string();

  {
    const fs::path mla_src = extract_root / "0_process_mla.json";
    std::ifstream in(mla_src);
    require(in.is_open(), "Failed to open MLA config: " + mla_src.string());
    json mla_json;
    in >> mla_json;

    if (mla_json.contains("simaai__params") && mla_json["simaai__params"].is_object()) {
      auto& params = mla_json["simaai__params"];
      if (params.contains("model_path") && params["model_path"].is_string()) {
        fs::path model_path = params["model_path"].get<std::string>();
        if (model_path.is_relative()) {
          model_path = extract_root / model_path;
        }
        require(fs::exists(model_path), "Missing MLA model binary: " + model_path.string());
        params["model_path"] = model_path.string();
      }
    }

    std::ofstream out(cfg.mla_cfg);
    require(out.is_open(), "Failed to write runtime MLA config: " + cfg.mla_cfg);
    out << mla_json.dump(2);
  }

  require(!cfg.tess_cfg.empty() && file_exists(cfg.tess_cfg),
          "Failed to resolve tessellate config");
  require(!cfg.pre_cfg.empty() && file_exists(cfg.pre_cfg), "Failed to resolve preproc config");
  require(!cfg.mla_cfg.empty() && file_exists(cfg.mla_cfg), "Failed to resolve MLA config");
  require(!cfg.box_cfg.empty() && file_exists(cfg.box_cfg), "Failed to resolve boxdecode config");
  require(!cfg.post_cfg.empty() && file_exists(cfg.post_cfg), "Failed to resolve postproc config");

  return cfg;
}

std::string require_factory_or_throw(const char* factory, const char* role_label) {
  if (factory && *factory && simaai::neat::element_exists(factory))
    return factory;
  throw std::runtime_error(std::string("Missing required NEAT plugin factory for ") + role_label +
                           " (expected '" + (factory ? factory : "") + "')");
}

uint16_t fp32_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

simaai::neat::Tensor make_bf16_rgb_tensor(const cv::Mat& img_bgr, int width, int height) {
  require(!img_bgr.empty(), "Input image is empty");

  cv::Mat resized;
  cv::resize(img_bgr, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  const size_t elems = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
  auto storage = simaai::neat::make_cpu_owned_storage(elems * sizeof(uint16_t));
  auto map = storage->map(simaai::neat::MapMode::Write);
  require(map.data != nullptr, "Failed to map BF16 tensor storage");
  require(map.size_bytes >= elems * sizeof(uint16_t), "BF16 tensor storage too small");

  auto* out = static_cast<uint16_t*>(map.data);
  const uint8_t* src = rgb.data;
  for (size_t i = 0; i < elems; ++i) {
    out[i] = fp32_to_bf16(static_cast<float>(src[i]));
  }

  simaai::neat::Tensor tensor;
  tensor.storage = storage;
  tensor.dtype = simaai::neat::TensorDType::BFloat16;
  tensor.layout = simaai::neat::TensorLayout::HWC;
  tensor.shape = {height, width, 3};
  tensor.strides_bytes = {static_cast<int64_t>(width * 3 * static_cast<int>(sizeof(uint16_t))),
                          static_cast<int64_t>(3 * static_cast<int>(sizeof(uint16_t))),
                          static_cast<int64_t>(sizeof(uint16_t))};
  tensor.device = {simaai::neat::DeviceType::CPU, 0};
  tensor.read_only = true;
  return tensor;
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
        err = "Run::pull timeout";
        return false;
      }
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      err = "Run::pull closed";
      return false;
    }
    err = perr.message.empty() ? "Run::pull error" : perr.message;
    return false;
  }
}

void do_warmup(simaai::neat::Run& async, const cv::Mat& input, int warm, int timeout_ms) {
  (void)async.warmup(std::vector<cv::Mat>{input}, warm, timeout_ms);
}

void do_warmup(simaai::neat::Run& async, const simaai::neat::Tensor& input, int warm,
               int timeout_ms) {
  for (int i = 0; i < warm; ++i) {
    if (!async.push(simaai::neat::TensorList{input})) {
      throw std::runtime_error("Run::push returned false during tensor warmup");
    }
    simaai::neat::Sample out;
    simaai::neat::PullError perr;
    const simaai::neat::PullStatus status = async.pull(timeout_ms, out, &perr);
    if (status != simaai::neat::PullStatus::Ok) {
      throw std::runtime_error("tensor warmup pull failed: " +
                               (perr.message.empty() ? std::string("unknown") : perr.message));
    }
  }
}

const std::vector<cv::Mat>& as_build_inputs(const std::vector<cv::Mat>& inputs) {
  return inputs;
}
std::vector<cv::Mat> as_build_inputs(const cv::Mat& input) {
  return {input};
}

const simaai::neat::TensorList& as_build_inputs(const simaai::neat::TensorList& inputs) {
  return inputs;
}
simaai::neat::TensorList as_build_inputs(const simaai::neat::Tensor& input) {
  return {input};
}

const simaai::neat::SampleList& as_build_inputs(const simaai::neat::SampleList& inputs) {
  return inputs;
}
simaai::neat::SampleList as_build_inputs(const simaai::neat::Sample& input) {
  return {input};
}

template <typename BuildInputT, typename PushInputT>
RunSummary run_async_pipeline(const std::string& name, simaai::neat::Session& session,
                              const BuildInputT& build_input, const PushInputT& push_input,
                              const AsyncTestConfig& cfg, bool expect_bbox) {
  RunSummary res;
  res.name = name;

  simaai::neat::RunOptions run_opt;
  run_opt.preset = simaai::neat::RunPreset::Reliable;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;

  step_log((name + ": before build").c_str());
  auto async = session.build(as_build_inputs(build_input), simaai::neat::RunMode::Async, run_opt);
  step_log((name + ": after build").c_str());

  try {
    step_log((name + ": before warmup").c_str());
    do_warmup(async, push_input, cfg.warm, cfg.warm_timeout_ms);
    step_log((name + ": after warmup").c_str());
  } catch (const std::exception& e) {
    append_note(res.note, "warmup_error=" + sanitize_note(e.what()));
    return res;
  }

  std::mutex error_mu;
  std::string consumer_error;
  std::atomic<int> measured_out{0};
  std::atomic<int> payload_count{0};
  std::atomic<bool> stop_requested{false};

  auto set_error = [&](const std::string& msg) {
    std::lock_guard<std::mutex> lock(error_mu);
    if (consumer_error.empty())
      consumer_error = msg;
    stop_requested.store(true);
  };

  std::thread consumer([&]() {
    int timeout_count = 0;
    while (true) {
      std::optional<simaai::neat::Sample> out;
      std::string err;
      if (!pull_with_timeout(async, cfg.pull_timeout_ms, cfg.max_timeouts, timeout_count, out,
                             err)) {
        if (!err.empty())
          set_error(err);
        break;
      }
      if (!out.has_value())
        break;

      const int m = measured_out.fetch_add(1) + 1;

      std::vector<uint8_t> payload;
      if (expect_bbox) {
        if (!objdet::extract_bbox_payload(*out, m - 1, payload, err)) {
          set_error(err);
          break;
        }
      } else {
        if (!out->tensor.has_value()) {
          set_error("missing tensor output");
          break;
        }
        try {
          payload = out->tensor->copy_payload_bytes();
        } catch (const std::exception& ex) {
          set_error(std::string("payload_copy_failed: ") + ex.what());
          break;
        }
      }
      if (expect_bbox) {
        if (payload.size() < sizeof(uint32_t)) {
          set_error("bbox payload too small");
          break;
        }
      } else if (payload.empty()) {
        set_error("detess payload empty");
        break;
      }

      payload_count.fetch_add(1);
      if (m >= cfg.iters)
        break;
    }
  });

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.iters; ++i) {
    if (stop_requested.load())
      break;
    if (!async.push(as_build_inputs(push_input))) {
      set_error("Run::push returned false");
      break;
    }
  }
  async.close_input();
  consumer.join();
  auto end = std::chrono::steady_clock::now();

  const double elapsed_s = std::chrono::duration<double>(end - start).count();
  res.outputs = measured_out.load();
  res.payloads = payload_count.load();
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;

  if (!consumer_error.empty()) {
    append_note(res.note, "async_error=" + sanitize_note(consumer_error));
  }
  if (res.outputs != cfg.iters) {
    append_note(res.note, "outputs=" + std::to_string(res.outputs));
  }
  if (res.payloads != cfg.iters) {
    append_note(res.note, "payloads=" + std::to_string(res.payloads));
  }
  if (res.avg_fps < cfg.min_fps) {
    append_note(res.note, "avg_fps=" + std::to_string(res.avg_fps));
  }

  if (res.note.empty()) {
    res.ok = true;
  }

  if (!res.ok) {
    const std::string last_err = async.last_error();
    if (!last_err.empty()) {
      append_note(res.note, "last_error=" + sanitize_note(last_err));
    }
  }

  res.pipeline = session.last_pipeline();
  return res;
}

RunSummary run_tess_bf16_path(const PipelineConfigs& cfgs, const PluginFactories& factories,
                              const cv::Mat& img_bgr, const AsyncTestConfig& cfg) {
  const int model_w = 640;
  const int model_h = 640;

  simaai::neat::InputOptions in;
  in.media_type = "application/vnd.simaai.tensor";
  in.format = simaai::neat::FormatTag::BF16;
  in.width = model_w;
  in.height = model_h;
  in.depth = 3;
  in.max_width = model_w;
  in.max_height = model_h;
  in.max_depth = 3;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::Input(in));
  session.custom(factories.processcvu + " name=tessellate_1 stage-id=tessellate_1 config=\"" +
                 cfgs.tess_cfg + "\" num-buffers=4");
  session.custom(factories.processmla +
                 " name=neatprocessmla_1 stage-id=neatprocessmla_1 config=\"" + cfgs.mla_cfg +
                 "\" multi-pipeline=true num-buffers=4");
  session.custom(factories.boxdecode + " name=n3_boxdecode_1 stage-id=n3_boxdecode_1 config=\"" +
                 cfgs.box_cfg +
                 "\" silent=true emit-signals=false sima-allocator-type=2 decode-type=yolo "
                 "detection-threshold=0.5 nms-iou-threshold=0.5 topk=24 transmit=false "
                 "num-buffers=4");
  session.add(simaai::neat::nodes::Output());

  const simaai::neat::Tensor input_bf16 = make_bf16_rgb_tensor(img_bgr, model_w, model_h);
  return run_async_pipeline("yolov11_tess_bf16", session, input_bf16, input_bf16, cfg, true);
}

RunSummary run_preproc_bf16_boxdecode_path(const PipelineConfigs& cfgs,
                                           const PluginFactories& factories, const cv::Mat& img_bgr,
                                           const AsyncTestConfig& cfg) {
  simaai::neat::InputOptions in;
  in.media_type = "video/x-raw";
  in.format = simaai::neat::FormatTag::BGR;
  in.width = img_bgr.cols;
  in.height = img_bgr.rows;
  in.depth = 3;
  in.max_width = std::max(img_bgr.cols, 1280);
  in.max_height = std::max(img_bgr.rows, 720);
  in.max_depth = 3;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::Input(in));
  session.custom(factories.processcvu + " name=neatpreproc_1 stage-id=neatpreproc_1 config=\"" +
                 cfgs.pre_cfg + "\" num-buffers=4");
  session.custom(factories.processmla +
                 " name=neatprocessmla_1 stage-id=neatprocessmla_1 config=\"" + cfgs.mla_cfg +
                 "\" multi-pipeline=true num-buffers=4");
  session.custom(factories.boxdecode + " name=n3_boxdecode_1 stage-id=n3_boxdecode_1 config=\"" +
                 cfgs.box_cfg +
                 "\" silent=true emit-signals=false sima-allocator-type=2 decode-type=yolo "
                 "detection-threshold=0.5 nms-iou-threshold=0.5 topk=24 transmit=false "
                 "num-buffers=4");
  session.add(simaai::neat::nodes::Output());

  return run_async_pipeline("yolov11_preproc_bf16_boxdecode", session, img_bgr, img_bgr, cfg, true);
}

RunSummary run_preproc_bf16_detess_path(const PipelineConfigs& cfgs,
                                        const PluginFactories& factories, const cv::Mat& img_bgr,
                                        const AsyncTestConfig& cfg) {
  simaai::neat::InputOptions in;
  in.media_type = "video/x-raw";
  in.format = simaai::neat::FormatTag::BGR;
  in.width = img_bgr.cols;
  in.height = img_bgr.rows;
  in.depth = 3;
  in.max_width = std::max(img_bgr.cols, 1280);
  in.max_height = std::max(img_bgr.rows, 720);
  in.max_depth = 3;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::Input(in));
  session.custom(factories.processcvu + " name=neatpreproc_1 stage-id=neatpreproc_1 config=\"" +
                 cfgs.pre_cfg + "\" num-buffers=4");
  session.custom(factories.processmla +
                 " name=neatprocessmla_1 stage-id=neatprocessmla_1 config=\"" + cfgs.mla_cfg +
                 "\" multi-pipeline=true num-buffers=4");
  session.custom(factories.processcvu + " name=detessdequant_1 stage-id=detessdequant_1 config=\"" +
                 cfgs.post_cfg + "\" silent=true num-buffers=4");
  session.add(simaai::neat::nodes::Output());

  return run_async_pipeline("yolov11_preproc_bf16_detess", session, img_bgr, img_bgr, cfg, false);
}

} // namespace

int main(int argc, char** argv) {
  if (!env_flag("SIMA_ENABLE_YOLOV11_BF16_TEST", false)) {
    return skip_long_test("YOLOv11 BF16 test disabled for now "
                          "(set SIMA_ENABLE_YOLOV11_BF16_TEST=1 to run)");
  }
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    PluginFactories factories;
    factories.processcvu = require_factory_or_throw("neatprocesscvu", "processcvu");
    factories.processmla = require_factory_or_throw("neatprocessmla", "processmla");
    factories.boxdecode = require_factory_or_throw("neatboxdecode", "boxdecode");

    const std::string tar_gz = resolve_yolov11_pose_bf16_tar_or_skip(root);
    cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);
    const PipelineConfigs cfgs = extract_and_resolve_configs_or_throw(tar_gz, root);
    AsyncTestConfig cfg;

    RunSummary tess = run_tess_bf16_path(cfgs, factories, img_bgr, cfg);
    RunSummary pre_box = run_preproc_bf16_boxdecode_path(cfgs, factories, img_bgr, cfg);
    RunSummary pre_detess;
    pre_detess = run_preproc_bf16_detess_path(cfgs, factories, img_bgr, cfg);

    const bool require_pre = env_flag("SIMA_Y11_REQUIRE_PREPROC_BF16", false);
    const bool require_box = env_flag("SIMA_Y11_REQUIRE_BOXDECODE", false);
    bool overall_ok =
        require_box ? (tess.ok || pre_box.ok) : (tess.ok || pre_box.ok || pre_detess.ok);
    if (require_pre) {
      overall_ok = overall_ok && (pre_box.ok || pre_detess.ok);
    }

    std::cout << "YOLOV11_BF16_TESS ok=" << (tess.ok ? "1" : "0") << " outputs=" << tess.outputs
              << " payloads=" << tess.payloads << " avg_fps=" << tess.avg_fps
              << " note=" << tess.note << "\n";
    std::cout << "YOLOV11_BF16_TESS pipeline\n" << tess.pipeline << "\n";

    std::cout << "YOLOV11_PREPROC_BF16_BOX ok=" << (pre_box.ok ? "1" : "0")
              << " outputs=" << pre_box.outputs << " payloads=" << pre_box.payloads
              << " avg_fps=" << pre_box.avg_fps << " note=" << pre_box.note << "\n";
    std::cout << "YOLOV11_PREPROC_BF16_BOX pipeline\n" << pre_box.pipeline << "\n";

    std::cout << "YOLOV11_PREPROC_BF16_DETESS ok=" << (pre_detess.ok ? "1" : "0")
              << " outputs=" << pre_detess.outputs << " payloads=" << pre_detess.payloads
              << " avg_fps=" << pre_detess.avg_fps << " note=" << pre_detess.note << "\n";
    std::cout << "YOLOV11_PREPROC_BF16_DETESS pipeline\n" << pre_detess.pipeline << "\n";

    std::cout << "YOLOV11_CFG tess=" << cfgs.tess_cfg << " pre=" << cfgs.pre_cfg
              << " mla=" << cfgs.mla_cfg << " box=" << cfgs.box_cfg << " post=" << cfgs.post_cfg
              << "\n";
    std::cout << "YOLOV11_FACTORIES processcvu=" << factories.processcvu
              << " processmla=" << factories.processmla << " boxdecode=" << factories.boxdecode
              << "\n";

    const bool require_success = env_flag("SIMA_Y11_REQUIRE_SUCCESS", false);
    if (!overall_ok && !require_success) {
      skip_long_test_exception("YOLOv11 BF16 probe paths unavailable on this runtime "
                               "(set SIMA_Y11_REQUIRE_SUCCESS=1 to enforce pass/fail)");
    }

    return overall_ok ? 0 : 2;
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

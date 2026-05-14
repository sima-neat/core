#include "asset_utils.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "pipeline/Session.h"
#include "nodes/common/Output.h"
#include "nodes/groups/ImageInputGroup.h"
#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int kInferWidth = 224;
constexpr int kInferHeight = 224;
constexpr const char* kFallbackImageUrl =
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg";

namespace fs = std::filesystem;

fs::path find_repo_root() {
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::string pick_model_pack() {
  if (const char* env = std::getenv("SIMA_WATCHDOG_MODEL_PACK")) {
    if (*env && fs::exists(env))
      return env;
  }
  if (const char* env = std::getenv("SIMA_MODEL_TAR")) {
    if (*env && fs::exists(env))
      return env;
  }
  if (file_exists("tmp/resnet_50_mpk.tar.gz"))
    return "tmp/resnet_50_mpk.tar.gz";
  if (file_exists("build/tmp/resnet_50_mpk.tar.gz"))
    return "build/tmp/resnet_50_mpk.tar.gz";
  const fs::path root = find_repo_root();
  const fs::path root_tmp = root / "tmp" / "resnet_50_mpk.tar.gz";
  if (fs::exists(root_tmp))
    return root_tmp.string();
  const fs::path root_build_tmp = root / "build" / "tmp" / "resnet_50_mpk.tar.gz";
  if (fs::exists(root_build_tmp))
    return root_build_tmp.string();

  const std::string resolved = sima_test::resolve_resnet50_tar(root);
  if (!resolved.empty())
    return resolved;
  return {};
}

std::string pick_image_path() {
  if (const char* env = std::getenv("SIMA_WATCHDOG_IMAGE")) {
    if (*env)
      return env;
  }
  if (file_exists("test.jpg"))
    return "test.jpg";
  const fs::path root = find_repo_root();
  const fs::path root_img = root / "test.jpg";
  if (fs::exists(root_img))
    return root_img.string();

  const fs::path fallback = root / "tmp" / "watchdog_input.jpg";
  if (sima_test::download_file(kFallbackImageUrl, fallback) && fs::exists(fallback)) {
    return fallback.string();
  }
  return {};
}

cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }
  if (w > 0 && h > 0 && (bgr.cols != w || bgr.rows != h)) {
    cv::resize(bgr, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

[[nodiscard]] simaai::neat::Run run_resnet_pipeline() {
  const std::string model_pack = pick_model_pack();
  const std::string image_path = pick_image_path();
  if (model_pack.empty()) {
    throw std::runtime_error("resnet model pack not found and sima-cli fetch failed");
  }
  if (image_path.empty()) {
    throw std::runtime_error("input image not found and fallback download failed");
  }

  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  simaai::neat::Model model(model_pack, model_opt);

  simaai::neat::nodes::groups::ImageInputGroupOptions src_opt;
  src_opt.path = image_path;
  src_opt.imagefreeze_num_buffers = 8;
  src_opt.fps = 30;
  src_opt.use_videorate = true;
  src_opt.use_videoscale = true;
  src_opt.output_caps.enable = true;
  src_opt.output_caps.format = simaai::neat::FormatTag::NV12;
  src_opt.output_caps.width = kInferWidth;
  src_opt.output_caps.height = kInferHeight;
  src_opt.output_caps.fps = 30;
  src_opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  src_opt.sima_decoder.enable = true;
  src_opt.sima_decoder.decoder_name = "decoder";
  src_opt.sima_decoder.raw_output = true;

  simaai::neat::Session p;
  p.add(simaai::neat::nodes::groups::ImageInputGroup(src_opt));
  simaai::neat::Model::SessionOptions session_opt;
  session_opt.include_appsrc = false;
  session_opt.include_appsink = false;
  p.add(model.session(session_opt));
  p.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run run = p.build(run_opt);

  simaai::neat::Sample out;
  (void)run.pull(500, out, nullptr);
  return run;
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    simaai::neat::Run run = run_resnet_pipeline();
    const char* path = std::getenv("SIMA_DISPATCHER_WATCHDOG_PATH");
    if (!path || !*path) {
      path = "dispatcher_watchdog";
    }

    if (!file_exists(path)) {
      return fail_test(std::string("dispatcher_watchdog not found at ") + path);
    }

    const char* handle_env = std::getenv("SIMA_WATCHDOG_TEST_HANDLE");
    const char* model_env = std::getenv("SIMA_WATCHDOG_TEST_MODEL");
    const std::string handle = (handle_env && *handle_env) ? handle_env : "1";
    const std::string model = (model_env && *model_env) ? model_env : "1";

    int fds[2];
    if (pipe(fds) != 0) {
      throw std::runtime_error("pipe() failed");
    }

    pid_t pid = fork();
    if (pid < 0) {
      close(fds[0]);
      close(fds[1]);
      throw std::runtime_error("fork() failed");
    }

    if (pid == 0) {
      close(fds[1]);
      char fd_str[32];
      std::snprintf(fd_str, sizeof(fd_str), "%d", fds[0]);
      execl(path, path, fd_str, handle.c_str(), model.c_str(), (char*)nullptr);
      _exit(127);
    }

    close(fds[0]);
    close(fds[1]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
      run.close();
      throw std::runtime_error("waitpid() failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      run.close();
      throw std::runtime_error("dispatcher_watchdog exited with failure");
    }

    run.close();
    std::cout << "[OK] unit_dispatcher_watchdog_test passed\n";
    return 0;
  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return fail_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

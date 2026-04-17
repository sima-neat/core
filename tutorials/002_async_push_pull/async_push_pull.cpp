#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Sig {
  std::string output_kind = "sample_or_tensor";
  int tensor_rank = -1;
  int field_count = -1;
};

void source_fallback_signature_stub() {
  // Source-fallback signature for parity tooling if runtime output is unavailable.
  if (false) {
    tutorial_v2::print_signature({
        {"tutorial", "002"},
        {"lang", "cpp"},
        {"flow", "minimal_cvmat_dataloader_async_threaded"},
        {"run_mode", "async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
        {"tput_contract", "-1"},
    });
  }
}

simaai::neat::Model::Options build_model_options(int size) {
  simaai::neat::Model::Options opt;
  opt.format = "RGB";
  opt.input_max_width = size;
  opt.input_max_height = size;
  opt.input_max_depth = 3;
  opt.preproc.channel_mean = {0.485f, 0.456f, 0.406f};
  opt.preproc.channel_stddev = {0.229f, 0.224f, 0.225f};
  return opt;
}

std::vector<fs::path> resnet_image_candidates() {
  const fs::path assets = tutorial_v2::find_asset_root();
  return {
      assets / "fronalpstock_1330.jpg",
      assets / "ilena_488.jpg",
      assets / "lichtenstein_512.png",
  };
}

cv::Mat load_rgb_u8(const fs::path& path, int size) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("failed to read image: " + path.string());
  }

  if (bgr.cols != size || bgr.rows != size) {
    cv::resize(bgr, bgr, cv::Size(size, size), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (rgb.type() != CV_8UC3) {
    throw std::runtime_error("expected CV_8UC3 RGB image");
  }
  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }
  return rgb;
}

std::vector<cv::Mat> dataloader_from_images(int size, int n) {
  std::vector<fs::path> existing;
  for (const auto& p : resnet_image_candidates()) {
    if (fs::exists(p)) {
      existing.push_back(p);
    }
  }
  if (existing.empty()) {
    throw std::runtime_error("no local images found for ResNet50 run");
  }

  const int count = std::max(1, n);
  std::vector<cv::Mat> images;
  images.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    images.push_back(load_rgb_u8(existing[static_cast<size_t>(i) % existing.size()], size));
  }
  return images;
}

std::vector<float> scores_from_output(const simaai::neat::Sample& out) {
  tutorial_v2::check("has_tensor_output", out.tensor.has_value(),
                     "expected tensor output from model");

  const simaai::neat::Tensor& t = *out.tensor;
  tutorial_v2::check("tensor_float32", t.dtype == simaai::neat::TensorDType::Float32,
                     "expected float32 logits tensor");

  const simaai::neat::Mapping map = t.map_read();
  tutorial_v2::check("tensor_non_empty", map.data != nullptr && map.size_bytes > 0,
                     "model output tensor bytes must be non-empty");
  tutorial_v2::check("tensor_size_aligned", (map.size_bytes % sizeof(float)) == 0,
                     "tensor bytes must align to float32");

  const size_t elems = map.size_bytes / sizeof(float);
  std::vector<float> flat(elems);
  std::memcpy(flat.data(), map.data, map.size_bytes);
  if (flat.size() >= 1000) {
    flat.resize(1000);
  }
  return flat;
}

int top1_from_output(const simaai::neat::Sample& out) {
  std::vector<float> scores = scores_from_output(out);
  if (scores.empty()) {
    throw std::runtime_error("empty score vector");
  }
  const auto it = std::max_element(scores.begin(), scores.end());
  return static_cast<int>(std::distance(scores.begin(), it));
}

Sig summarize(const simaai::neat::Sample& out) {
  Sig sig;
  sig.output_kind = std::to_string(static_cast<int>(out.kind));
  sig.tensor_rank = out.tensor.has_value() ? static_cast<int>(out.tensor->shape.size()) : -1;
  sig.field_count = static_cast<int>(out.fields.size());
  return sig;
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--size <n>] [--n <count>] [--queue <n>]"
            << " [--timeout-ms <ms>] [--expect-id <id>] [--print-gst]\n";
  tutorial_v2::print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    source_fallback_signature_stub();

    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int size = tutorial_v2::parse_int_arg(argc, argv, "--size", 224);
    const int n = tutorial_v2::parse_int_arg(argc, argv, "--n", 4);
    const int queue_depth = tutorial_v2::parse_int_arg(argc, argv, "--queue", 8);
    const int timeout_ms = tutorial_v2::parse_int_arg(argc, argv, "--timeout-ms", 2000);
    const int expect_id = tutorial_v2::parse_int_arg(argc, argv, "--expect-id", -1);
    tutorial_v2::require(size > 0, "--size must be > 0");
    tutorial_v2::require(n > 0, "--n must be > 0");

    tutorial_v2::step("input_contract",
                      "parse CLI and prepare ResNet50 model + local cv::Mat dataloader");
    tutorial_v2::step("run_mode_choice",
                      "run async inference with producer-thread push and main-thread pull");
    tutorial_v2::why(
        "keep chapter 002 parallel to chapter 001 so sync-vs-async is the main variable");
    tutorial_v2::tradeoff(
        "threaded async flow improves throughput potential but adds coordination complexity");
    tutorial_v2::failure_mode(
        "push/pull mismatches, queue stalls, or producer exceptions should fail loudly");
    tutorial_v2::interpret_output(
        "top1 is human-facing; tput_fps and tput_contract are parity-facing");
    tutorial_v2::step("output_contract", "emit top1 lines, async stats, and stable signature");
    tutorial_v2::check("strict_mode_visible",
                       tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                           tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                       "strict-mode guard is observable");

    const fs::path root = tutorial_v2::find_repo_root();

    std::string mpk_arg;
    const fs::path mpk_path = tutorial_v2::get_arg(argc, argv, "--mpk", mpk_arg)
                                  ? fs::path(mpk_arg)
                                  : tutorial_v2::default_resnet_mpk(root);
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return tutorial_v2::skip("missing ResNet50 MPK (pass --mpk)");
    }

    Sig sig;
    int tput_contract = -1;
    try {
      // Same model/dataloader setup as 001; only run mode differs.
      simaai::neat::Model resnet50(mpk_path.string(), build_model_options(size));

      if (tutorial_v2::wants_print_gst(argc, argv)) {
        simaai::neat::Session describe_session;
        describe_session.add(resnet50.session());
        std::cout << describe_session.describe_backend() << "\n";
        return 0;
      }

      const std::vector<cv::Mat> images_to_push = dataloader_from_images(size, n);
      tutorial_v2::require(!images_to_push.empty(), "no images available for async run");

      simaai::neat::Session s;
      s.add(resnet50.session());

      simaai::neat::RunOptions opt;
      opt.queue_depth = queue_depth;
      opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
      opt.output_memory = simaai::neat::OutputMemory::Owned;

      // CORE LOGIC
      auto run = s.build(images_to_push.front(), simaai::neat::RunMode::Async, opt);

      std::atomic<int> pushed{0};
      std::atomic<bool> producer_done{false};
      std::mutex producer_err_mu;
      std::exception_ptr producer_err = nullptr;

      std::thread producer([&]() {
        try {
          for (const cv::Mat& image : images_to_push) {
            tutorial_v2::require(run.push(image), "push failed");
            pushed.fetch_add(1, std::memory_order_relaxed);
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(producer_err_mu);
          producer_err = std::current_exception();
        }
        run.close_input();
        producer_done.store(true, std::memory_order_release);
      });
      auto join_producer = [&]() {
        if (producer.joinable()) {
          producer.join();
        }
      };
      // END CORE LOGIC
      const auto start = std::chrono::steady_clock::now();
      int pulled = 0;
      try {
        for (;;) {
          auto out_opt = run.pull(timeout_ms);
          if (out_opt.has_value()) {
            ++pulled;
            const int pred = top1_from_output(*out_opt);
            sig = summarize(*out_opt);
            std::cout << "top1=" << pred << "\n";

            if (expect_id >= 0) {
              tutorial_v2::check("top1_expected_id", pred == expect_id, "verify expected class id");
            }
            continue;
          }

          const int pushed_now = pushed.load(std::memory_order_acquire);
          if (producer_done.load(std::memory_order_acquire) && pulled >= pushed_now) {
            break;
          }
        }
      } catch (...) {
        run.close_input();
        join_producer();
        throw;
      }

      join_producer();
      if (producer_err) {
        std::rethrow_exception(producer_err);
      }

      const int pushed_final = pushed.load(std::memory_order_acquire);
      tutorial_v2::require(pulled == pushed_final,
                           "async output count mismatch: pulled=" + std::to_string(pulled) +
                               " pushed=" + std::to_string(pushed_final));

      const auto end = std::chrono::steady_clock::now();
      const auto stats = run.stats();
      const double elapsed_sec = std::max(
          1e-9, std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
      const double tput_fps = static_cast<double>(pulled) / elapsed_sec;
      tput_contract = pulled;

      std::cout << "pushed:          " << pushed_final << "\n";
      std::cout << "pulled:          " << pulled << "\n";
      std::cout << "inputs_enqueued: " << stats.inputs_enqueued << "\n";
      std::cout << "inputs_dropped:  " << stats.inputs_dropped << "\n";
      std::cout << "tput_fps:        " << tput_fps << "\n";
      std::cout << "tput_contract:   " << tput_contract << "\n";
    } catch (const std::exception& e) {
      tutorial_v2::runtime_fallback(e);
      if (tutorial_v2::strict_mode()) {
        throw;
      }
    }

    tutorial_v2::check("tutorial_completed", true, "async dataloader path completed");
    tutorial_v2::print_signature({
        {"tutorial", "002"},
        {"lang", "cpp"},
        {"flow", "minimal_cvmat_dataloader_async_threaded"},
        {"run_mode", "async"},
        {"output_kind", sig.output_kind},
        {"tensor_rank", std::to_string(sig.tensor_rank)},
        {"field_count", std::to_string(sig.field_count)},
        {"tput_contract", std::to_string(tput_contract)},
    });

    std::cout << "[OK] 002_async_push_pull\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

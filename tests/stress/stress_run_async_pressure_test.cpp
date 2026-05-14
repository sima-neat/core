#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Session.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(20, std::min(value, 4000));
}

} // namespace

RUN_TEST("stress_run_async_pressure_test", [] {
  using namespace simaai::neat;

  const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 200));
  const int total_frames = std::max(80, std::min(iters * 3, 3000));

  Session session;
  InputOptions src_opt;
  src_opt.media_type = "video/x-raw";
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.use_simaai_pool = false;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  session.add(nodes::Input(src_opt));
  session.add(nodes::Output(OutputOptions::EveryFrame(128)));

  RunOptions run_opt;
  run_opt.queue_depth = 128;
  run_opt.overflow_policy = OverflowPolicy::Block;

  Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x22);
  Run run = session.build(TensorList{seed}, RunMode::Async, run_opt);

  std::atomic<int> pushed{0};
  std::atomic<int> pulled{0};
  std::atomic<bool> producer_done{false};
  std::mutex err_mu;
  std::string producer_err;
  std::string consumer_err;

  std::thread producer([&] {
    try {
      for (int i = 0; i < total_frames; ++i) {
        Sample sample;
        sample.kind = SampleKind::Tensor;
        sample.tensor =
            make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, static_cast<uint8_t>(i & 0xFF));
        sample.frame_id = i;
        sample.stream_id = "stress";

        if (!run.push(SampleList{sample})) {
          const std::string err = run.last_error();
          throw std::runtime_error("push failed: " + (err.empty() ? std::string("unknown") : err));
        }
        ++pushed;
      }
      run.close_input();
      producer_done = true;
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(err_mu);
      producer_err = e.what();
    }
  });

  std::thread consumer([&] {
    try {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while (std::chrono::steady_clock::now() < deadline) {
        auto out = run.pull(100);
        if (out.has_value()) {
          ++pulled;
        }
        if (producer_done.load() && pulled.load() >= pushed.load()) {
          break;
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(err_mu);
      consumer_err = e.what();
    }
  });

  producer.join();
  consumer.join();

  {
    std::lock_guard<std::mutex> lock(err_mu);
    if (!producer_err.empty()) {
      throw std::runtime_error("producer thread failed: " + producer_err);
    }
    if (!consumer_err.empty()) {
      throw std::runtime_error("consumer thread failed: " + consumer_err);
    }
  }

  require(pushed.load() == total_frames,
          "async stress did not push the expected number of samples");
  require(pulled.load() == pushed.load(), "async stress pull count mismatch");

  run.stop();
});

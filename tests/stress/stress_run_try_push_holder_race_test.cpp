#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
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
  return std::max(20, std::min(value, 2000));
}

} // namespace

RUN_TEST("stress_run_try_push_holder_race_test", ([] {
           using namespace simaai::neat;

           const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 200));

           Graph graph;
           InputOptions src_opt;
           src_opt.payload_type = simaai::neat::PayloadType::Image;
           src_opt.format = simaai::neat::FormatTag::RGB;
           src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
           src_opt.max_width = 96;
           src_opt.max_height = 96;
           src_opt.max_depth = 3;
           graph.add(nodes::Input(src_opt));
           graph.add(nodes::Output(OutputOptions::EveryFrame(256)));

           RunOptions run_opt;
           run_opt.queue_depth = 256;
           run_opt.overflow_policy = OverflowPolicy::Block;

           Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x66);
           Run run = graph.build(TensorList{seed}, run_opt);

           // Prime one holder from a regular push/pull path.
           TensorList prime = run.run(TensorList{seed}, 1000);
           require(prime.size() == 1, "stress holder prime expected one tensor");
           require(prime.front().storage != nullptr, "stress holder prime storage missing");
           require(prime.front().storage->holder != nullptr, "stress holder prime holder missing");
           const std::shared_ptr<void> holder = prime.front().storage->holder;

           std::atomic<int> pushed{0};
           std::atomic<int> pulled{0};
           std::atomic<bool> producer_done{false};
           std::mutex err_mu;
           std::string producer_err;
           std::string consumer_err;

           std::thread producer([&] {
             try {
               while (pushed.load() < iters) {
                 if (run.try_push_holder(holder)) {
                   ++pushed;
                   continue;
                 }
                 std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
               const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
               while (std::chrono::steady_clock::now() < deadline) {
                 auto outs = run.pull_tensors(100);
                 pulled += static_cast<int>(outs.size());
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
               throw std::runtime_error("holder stress producer failed: " + producer_err);
             }
             if (!consumer_err.empty()) {
               throw std::runtime_error("holder stress consumer failed: " + consumer_err);
             }
           }

           require(pushed.load() == iters,
                   "holder stress did not push the expected number of samples");
           require(pulled.load() == pushed.load(), "holder stress pull count mismatch");

           run.stop();
         }));

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Session.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace {

class EnvVarGuard {
public:
  EnvVarGuard(const char* key, const char* value) : key_(key), had_(false) {
    const char* cur = std::getenv(key_);
    if (cur && *cur) {
      had_ = true;
      old_ = cur;
    }
    ::setenv(key_, value, 1);
  }

  ~EnvVarGuard() {
    if (had_) {
      ::setenv(key_, old_.c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_;
  std::string old_;
};

} // namespace

RUN_TEST("unit_run_lifecycle_teardown_test", ([] {
           using namespace simaai::neat;

           EnvVarGuard input_stop_1("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "200");
           EnvVarGuard stream_stop_1("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", "200");

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x3A);
           Run run = sima_test::make_async_rgb_run(seed, 128, 128);

           std::atomic<bool> keep_running{true};
           std::atomic<int> pushes{0};
           std::atomic<int> pulls{0};
           std::mutex err_mu;
           std::string producer_err;
           std::string consumer_err;

           std::thread producer([&] {
             try {
               while (keep_running.load(std::memory_order_relaxed)) {
                 if (run.try_push(TensorList{seed})) {
                   pushes.fetch_add(1, std::memory_order_relaxed);
                 } else {
                   std::this_thread::sleep_for(std::chrono::milliseconds(1));
                 }
               }
             } catch (const std::exception& e) {
               std::lock_guard<std::mutex> lock(err_mu);
               producer_err = e.what();
             }
           });

           std::thread consumer([&] {
             try {
               while (keep_running.load(std::memory_order_relaxed)) {
                 auto out = run.pull(20);
                 if (out.has_value()) {
                   pulls.fetch_add(1, std::memory_order_relaxed);
                 }
               }
             } catch (const std::exception& e) {
               std::lock_guard<std::mutex> lock(err_mu);
               consumer_err = e.what();
             }
           });

           std::this_thread::sleep_for(std::chrono::milliseconds(200));

           const auto t0 = std::chrono::steady_clock::now();
           run.close_input();
           run.stop();
           run.stop(); // idempotent stop under prior teardown
           const auto t1 = std::chrono::steady_clock::now();

           keep_running.store(false, std::memory_order_relaxed);
           producer.join();
           consumer.join();

           {
             std::lock_guard<std::mutex> lock(err_mu);
             if (!producer_err.empty()) {
               throw std::runtime_error("run lifecycle producer failed: " + producer_err);
             }
             if (!consumer_err.empty()) {
               throw std::runtime_error("run lifecycle consumer failed: " + consumer_err);
             }
           }

           const int stop_ms = sima_test::elapsed_ms(t0, t1);
           require(stop_ms < 5000, "Run::stop teardown exceeded expected bound");

           require(!run.try_push(TensorList{seed}),
                   "Run::try_push should fail after stop()");
           (void)pushes.load(std::memory_order_relaxed);
           (void)pulls.load(std::memory_order_relaxed);
         }));

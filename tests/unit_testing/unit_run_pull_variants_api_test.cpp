#include "pipeline/Run.h"
#include "pipeline/ErrorCodes.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>

RUN_TEST("unit_run_pull_variants_api_test", ([] {
           using namespace simaai::neat;

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x71);
           Run run = sima_test::make_async_rgb_run(seed, 32, 32);

           // pull_tensor_matching timeout/miss behavior.
           {
             const int timeout_ms = 120;
             const auto t0 = std::chrono::steady_clock::now();
             auto miss = run.pull_tensor_matching("missing-tag", timeout_ms);
             const auto t1 = std::chrono::steady_clock::now();
             require(!miss.has_value(),
                     "Run::pull_tensor_matching should timeout-miss when queue is empty");
             sima_test::require_timeout_window("pull_tensor_matching",
                                               sima_test::elapsed_ms(t0, t1), timeout_ms);
           }

           // payload-tag matching via canonical RGB output tag.
           {
             require(run.push(seed), "Run::push seed failed for matching check");

             auto hit = run.pull_tensor_matching("RGB", 1000);
             require(hit.has_value(),
                     "Run::pull_tensor_matching should return matching payload_tag tensor");
           }

           // payload-tag miss should drop non-matching outputs and return timeout miss.
           {
             require(run.push(seed), "Run::push seed failed for miss check");

             auto miss = run.pull_tensor_matching("NON_MATCHING_TAG", 180);
             require(!miss.has_value(),
                     "Run::pull_tensor_matching should miss on non-matching payload_tag");
           }

           // pull_tensor_or_throw timeout should throw actionable message.
           {
             Sample tmp;
             PullError err;
             const PullStatus status = run.pull(120, tmp, &err);
             require(status == PullStatus::Timeout, "Run::pull timeout status mismatch");
             require(err.code == error_codes::kRuntimePull,
                     "Run::pull timeout should set runtime.pull error code");
             require_contains(err.message, "runtime.pull",
                              "Run::pull timeout should include runtime.pull code in message");
           }

           require(
               sima_test::throws_with([&]() { (void)run.pull_tensor_or_throw(120); }, "timeout"),
               "Run::pull_tensor_or_throw timeout error message mismatch");

           run.stop();

           {
             Run closed;
             Sample tmp;
             PullError err;
             const PullStatus status = closed.pull(10, tmp, &err);
             require(status == PullStatus::Closed, "Run::pull closed-state status mismatch");
             require(err.code == error_codes::kRuntimePull,
                     "Run::pull closed-state should set runtime.pull error code");
             require_contains(err.message, "closed",
                              "Run::pull closed-state should include closure context");
           }

           require(sima_test::throws_with(
                       []() {
                         Run closed;
                         (void)closed.pull_tensor_or_throw(10);
                       },
                       "closed"),
                   "Run::pull_tensor_or_throw closed-state error mismatch");
         }));

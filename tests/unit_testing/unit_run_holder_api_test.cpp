#include "pipeline/Run.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("unit_run_holder_api_test", ([] {
           using namespace simaai::neat;

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x58);

           require(sima_test::throws_with(
                       []() {
                         Run closed;
                         (void)closed.try_push_holder(std::shared_ptr<void>{});
                       },
                       "stream is closed"),
                   "Run::try_push_holder default behavior mismatch");

           Run run = sima_test::make_async_rgb_run(seed, 32, 32);

           // Null holder must fail with deterministic argument error.
           require(
               sima_test::throws_with([&]() { (void)run.try_push_holder(std::shared_ptr<void>{}); },
                                      "missing holder"),
               "Run::try_push_holder should reject null holder");

           // Prime a valid holder from tensor output.
           const Sample first = run.push_and_pull(seed, 1000);
           require(first.tensor.has_value(), "holder test: missing tensor output");
           require(first.tensor->storage != nullptr, "holder test: missing storage");
           require(first.tensor->storage->holder != nullptr, "holder test: missing holder");
           const std::shared_ptr<void> holder = first.tensor->storage->holder;

           require(run.try_push_holder(holder), "Run::try_push_holder valid holder should succeed");
           auto out = run.pull(1000);
           require(out.has_value(), "Run::try_push_holder should produce output");
           require(out->tensor.has_value(), "Run::try_push_holder output missing tensor");

           const Sample out2 = run.push_and_pull_holder(holder, 1000);
           require(out2.tensor.has_value(), "Run::push_and_pull_holder output missing tensor");

           run.stop();
           require(!run.try_push_holder(holder),
                   "Run::try_push_holder should fail with stale holder after stop()");
         }));

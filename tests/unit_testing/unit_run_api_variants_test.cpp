#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Run.h"
#include "pipeline/Session.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

simaai::neat::Run make_async_rgb_run(const simaai::neat::Tensor& seed,
                                     int producer_queue_depth = 16, int consumer_queue_depth = 16) {
  using namespace simaai::neat;

  Session session;
  InputOptions src_opt;
  src_opt.media_type = "video/x-raw";
  src_opt.format = "RGB";
  src_opt.use_simaai_pool = false;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;

  session.add(nodes::Input(src_opt));
  session.add(nodes::Output(OutputOptions::EveryFrame(128)));

  RunOptions opt;
  opt.queue_depth =
      (producer_queue_depth > consumer_queue_depth) ? producer_queue_depth : consumer_queue_depth;
  opt.overflow_policy = OverflowPolicy::Block;

  return session.build(seed, RunMode::Async, opt);
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_run_api_variants_test", ([] {
      using namespace simaai::neat;

      const Tensor seed = make_color_tensor(48, 32, ImageSpec::PixelFormat::RGB, 0x41);

      // Closed/default Run negative behavior.
      Run closed;
      require(throws_with([&]() { (void)closed.try_push(seed); }, "stream is closed"),
              "default Run::try_push should throw stream is closed");
      require(throws_with([&]() { (void)closed.try_push_holder(std::shared_ptr<void>{}); },
                          "stream is closed"),
              "default Run::try_push_holder should throw stream is closed");
      require(throws_with([&]() { (void)closed.push_and_pull_holder(std::shared_ptr<void>{}, 10); },
                          "stream is closed"),
              "default Run::push_and_pull_holder should throw stream is closed");
      require(throws_with([&]() { (void)closed.pull_tensor_or_throw(10); }, "closed"),
              "default Run::pull_tensor_or_throw should report closed");

      // Timeout / payload-tag pull behavior.
      Run run = make_async_rgb_run(seed, 32, 32);
      auto none = run.pull_tensor_matching("RGB", 120);
      require(!none.has_value(), "pull_tensor_matching should timeout when no output is pending");
      require(throws_with([&]() { (void)run.pull_tensor_or_throw(120); }, "timeout"),
              "pull_tensor_or_throw should timeout when no output is pending");

      require(run.push(seed), "push seed failed");
      auto miss = run.pull_tensor_matching("UNMATCHED_TAG", 120);
      require(!miss.has_value(),
              "pull_tensor_matching should return nullopt for unmatched payload_tag");

      require(run.push(seed), "push seed for matching tag failed");
      auto hit = run.pull_tensor_matching("RGB", 1000);
      require(hit.has_value(), "pull_tensor_matching should return tensor for RGB payload_tag");

      // Try-push should eventually report backpressure (false) under queue pressure.
      bool saw_try_push_false = false;
      for (int i = 0; i < 1000; ++i) {
        if (!run.try_push(seed)) {
          saw_try_push_false = true;
          break;
        }
      }
      require(saw_try_push_false, "Run::try_push never returned false under queue pressure");

      run.stop();

      // Holder-based push/pull positive path.
      Run holder_run = make_async_rgb_run(seed, 32, 32);
      Sample first = holder_run.push_and_pull(seed, 1000);
      require(first.tensor.has_value(), "holder test: first output missing tensor");
      require(first.tensor->storage != nullptr, "holder test: missing tensor storage");
      require(first.tensor->storage->holder != nullptr, "holder test: missing tensor holder");

      const std::shared_ptr<void> holder = first.tensor->storage->holder;
      require(holder_run.try_push_holder(holder),
              "try_push_holder should accept valid tensor holder");
      auto out = holder_run.pull(1000);
      require(out.has_value(), "try_push_holder path should produce output");
      require(out->tensor.has_value(), "try_push_holder output missing tensor");

      Sample out2 = holder_run.push_and_pull_holder(holder, 1000);
      require(out2.tensor.has_value(), "push_and_pull_holder output missing tensor");

      holder_run.stop();
    }));

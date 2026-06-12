#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"
#include "pipeline/runtime/RunInternal.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

namespace {

simaai::neat::Graph input_output_graph() {
  simaai::neat::Graph input;
  input.add(simaai::neat::nodes::Input());
  simaai::neat::Graph output;
  output.add(simaai::neat::nodes::Output());
  simaai::neat::Graph app;
  app.connect(input, output);
  return app;
}

simaai::neat::Graph slow_input_output_graph() {
  simaai::neat::Graph input;
  input.add(simaai::neat::nodes::Input());

  simaai::neat::Graph slow_sink;
  slow_sink.custom("identity sleep-time=250000000 ! queue max-size-buffers=1 leaky=no");
  slow_sink.add(simaai::neat::nodes::Output());

  simaai::neat::Graph app;
  app.connect(input, slow_sink);
  return app;
}

} // namespace

RUN_TEST("graph_migration_phase3_lifecycle_backpressure_test", [] {
  {
    auto app = input_output_graph();
    simaai::neat::Run run = app.build();
    simaai::neat::Sample out;
    simaai::neat::PullError err;
    const simaai::neat::PullStatus st = run.pull(10, out, &err);
    require(st == simaai::neat::PullStatus::Timeout,
            "empty connected Graph pull should time out before any push");
    run.stop();
  }

  {
    auto app = input_output_graph();
    simaai::neat::Run run = app.build();
    run.stop();
    bool threw = false;
    bool ok = false;
    try {
      ok = run.push(simaai::neat::Sample{graph_phase3_test::make_tensor_sample(0, "after-stop")});
    } catch (const std::exception& e) {
      threw = true;
      require_contains(std::string(e.what()), "stopped", "push-after-stop diagnostic");
    }
    require(threw || !ok, "push after stop must not succeed");
    run.close();
    run.close();
  }

  {
    auto app = input_output_graph();
    simaai::neat::Run run = app.build();
    auto fut = std::async(std::launch::async, [&run] {
      simaai::neat::Sample out;
      simaai::neat::PullError err;
      return run.pull(5000, out, &err);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    run.stop();
    const auto ready = fut.wait_for(std::chrono::seconds(1));
    require(ready == std::future_status::ready, "pending pull should unblock after stop");
  }

  {
    simaai::neat::RunOptions opt;
    opt.queue_depth = 1;
    opt.overflow_policy = simaai::neat::OverflowPolicy::DropIncoming;
    auto app = slow_input_output_graph();
    simaai::neat::Run run = app.build(opt);
    bool saw_reject = false;
    bool saw_fail_closed_backpressure = false;
    try {
      for (int i = 0; i < 512; ++i) {
        const bool ok = run.try_push(
            simaai::neat::Sample{graph_phase3_test::make_tensor_sample(i, "backpressure")});
        if (!ok) {
          saw_reject = true;
          break;
        }
      }
      const auto stats_after_burst = simaai::neat::run_internal::input_stats(run);
      saw_fail_closed_backpressure =
          saw_reject || stats_after_burst.dropped_frames > 0 || stats_after_burst.push_failures > 0;
      require(saw_fail_closed_backpressure,
              "rapid non-blocking push burst at queue_depth=1 should reject, fail, record a "
              "drop, or raise an explained backpressure timeout");
      auto out = run.pull(5000);
      require(out.has_value(), "backpressure run should still produce at least one sample");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      require_contains(msg, "backpressure", "backpressure timeout diagnostic");
      require_contains(msg, "not draining outputs as fast as inputs are pushed",
                       "backpressure timeout descriptor");
      saw_fail_closed_backpressure = true;
    }
    require(saw_fail_closed_backpressure,
            "rapid non-blocking push burst at queue_depth=1 should fail closed on backpressure");
    run.stop();
  }
});

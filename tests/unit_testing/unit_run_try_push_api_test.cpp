#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Session.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <string>
#include <thread>

RUN_TEST(
    "unit_run_try_push_api_test", ([] {
      using namespace simaai::neat;

      const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x39);

      // Default run should fail with deterministic "stream is closed" error.
      require(sima_test::throws_with(
                  [&]() {
                    Run closed;
                    (void)closed.try_push(seed);
                  },
                  "stream is closed"),
              "Run::try_push default-constructed behavior mismatch");

      // Queue pressure: try_push must report backpressure without blocking.
      {
        Run pressure = sima_test::make_async_rgb_run(seed, 1, 1);
        const auto fill = sima_test::fill_try_push_queue_non_blocking(pressure, seed, 4096);
        require(fill.saw_backpressure, "Run::try_push should eventually report queue backpressure");
        require(fill.max_call_ms < 50, "Run::try_push should be non-blocking under queue pressure");
        pressure.stop();
      }

      // After close_input and stop, try_push should fail cleanly.
      {
        Run run = sima_test::make_async_rgb_run(seed, 4, 4);
        run.close_input();
        require(!run.try_push(seed), "Run::try_push should fail after close_input()");
        run.stop();
        require(!run.try_push(seed), "Run::try_push should fail after stop()");
      }

      // Format/depth mismatch should surface deterministic error text.
      {
        Session session;
        InputOptions src_opt;
        src_opt.media_type = "video/x-raw";
        src_opt.format = "RGB";
        src_opt.use_simaai_pool = false;
        src_opt.max_width = 96;
        src_opt.max_height = 96;
        src_opt.max_depth = 3;
        session.add(nodes::Input(src_opt));
        session.add(nodes::Output(OutputOptions::EveryFrame(32)));

        RunOptions run_opt;
        run_opt.queue_depth = 8;

        cv::Mat rgb_seed(48, 64, CV_8UC3, cv::Scalar(20, 40, 60));
        Run run = session.build(rgb_seed, RunMode::Async, run_opt);

        bool threw = false;
        std::string msg;
        try {
          cv::Mat gray(48, 64, CV_8UC1, cv::Scalar(70));
          (void)run.push_and_pull(gray, 300);
        } catch (const std::exception& e) {
          threw = true;
          msg = e.what();
        }
        run.stop();

        require(threw, "Run::try_push mismatch path should throw on incompatible frame depth");
        require(!msg.empty(), "Run::try_push mismatch path should report non-empty error text");
      }

      // Invalid input argument path should surface actionable error text.
      {
        Session session;
        InputOptions src_opt;
        src_opt.media_type = "video/x-raw";
        src_opt.format = "RGB";
        src_opt.use_simaai_pool = false;
        src_opt.max_width = 96;
        src_opt.max_height = 96;
        src_opt.max_depth = 3;
        session.add(nodes::Input(src_opt));
        session.add(nodes::Output(OutputOptions::EveryFrame(32)));

        RunOptions run_opt;
        run_opt.queue_depth = 8;

        require(sima_test::throws_with(
                    [&]() {
                      cv::Mat empty;
                      (void)session.build(empty, RunMode::Async, run_opt);
                    },
                    "input frame is empty"),
                "Run::try_push invalid input path should report empty-input error");
      }
    }));

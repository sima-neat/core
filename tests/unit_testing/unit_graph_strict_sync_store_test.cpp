#include "graph/StrictSync.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "graph_test_utils.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

std::shared_ptr<simaai::neat::Run> build_integration_run() {
  using namespace simaai::neat;

  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.use_simaai_pool = false;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;

  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(128)));

  RunOptions run_opt;
  run_opt.queue_depth = 128;
  run_opt.overflow_policy = OverflowPolicy::Block;

  const Tensor seed = make_color_tensor(12, 8, ImageSpec::PixelFormat::RGB, 0x22);
  return std::make_shared<Run>(graph.build(TensorList{seed}, RunMode::Async, run_opt));
}

} // namespace

RUN_TEST(
    "unit_graph_strict_sync_store_test", ([] {
      using simaai::neat::graph::strict_sync::PendingVideoStore;
      using simaai::neat::graph::strict_sync::ReleasePacer;
      using simaai::neat::graph::strict_sync::YoloTokenStore;

      // PendingVideoStore: enqueue/take correctness, replacement, and stats.
      {
        PendingVideoStore store(2);

        require(!store.enqueue(3, 1, sima_test::make_tensor_sample(1, "a"), 10, 64),
                "PendingVideoStore should reject invalid stream index");
        require(!store.enqueue(0, -1, sima_test::make_tensor_sample(-1, "a"), 10, 64),
                "PendingVideoStore should reject invalid frame_id");

        require(store.enqueue(0, 5, sima_test::make_tensor_sample(5, "s0"), 20, 100),
                "PendingVideoStore enqueue should succeed");
        require(store.enqueue(0, 5, sima_test::make_tensor_sample(5, "s0-new"), 22, 80),
                "PendingVideoStore duplicate enqueue replacement should succeed");

        auto stats = store.stats(0);
        require(stats.enqueued == 2, "PendingVideoStore enqueued stats mismatch");
        require(stats.pending_depth == 1, "PendingVideoStore pending depth mismatch");
        require(stats.pending_bytes == 80, "PendingVideoStore pending bytes mismatch");

        auto got = store.take(0, 5);
        require(got.has_value(), "PendingVideoStore should return queued frame");
        require(got->sample.stream_id == "s0-new",
                "PendingVideoStore should return latest replacement sample");
        require(got->cap_ms == 22, "PendingVideoStore cap_ms replacement mismatch");

        auto miss = store.take(0, 5);
        require(!miss.has_value(), "PendingVideoStore should miss consumed frame");

        stats = store.stats(0);
        require(stats.matched == 1, "PendingVideoStore matched stats mismatch");
        require(stats.miss >= 1, "PendingVideoStore miss stats should increment");
      }

      // YoloTokenStore: ordered dequeue consistency and per-stream stats.
      {
        YoloTokenStore store(2);

        store.enqueue(1, 100);
        store.enqueue(0, 200);
        store.enqueue(1, 101);

        auto o0 = store.take_ordered();
        auto o1 = store.take_ordered();
        auto o2 = store.take_ordered();
        require(o0.has_value() && o0->stream_idx == 1 && o0->frame_id == 100,
                "YoloTokenStore ordered token[0] mismatch");
        require(o1.has_value() && o1->stream_idx == 0 && o1->frame_id == 200,
                "YoloTokenStore ordered token[1] mismatch");
        require(o2.has_value() && o2->stream_idx == 1 && o2->frame_id == 101,
                "YoloTokenStore ordered token[2] mismatch");

        auto t0 = store.take(1);
        auto t1 = store.take(1);
        auto t2 = store.take(1);
        require(t0.has_value() && t0->frame_id == 100,
                "YoloTokenStore stream dequeue order mismatch");
        require(t1.has_value() && t1->frame_id == 101,
                "YoloTokenStore stream dequeue order mismatch");
        require(!t2.has_value(), "YoloTokenStore stream queue should drain");

        const auto stats = store.stats(1);
        require(stats.enqueued == 2, "YoloTokenStore enqueued stats mismatch");
        require(stats.dequeued == 2, "YoloTokenStore dequeued stats mismatch");
        require(stats.miss >= 1, "YoloTokenStore miss stats should increment");
      }

      // ReleasePacer integration-backed send path + pressure accounting.
      {
        std::shared_ptr<simaai::neat::Run> run;
        try {
          run = build_integration_run();
        } catch (const std::exception& e) {
          if (sima_test::likely_runtime_missing(e.what())) {
            throw std::runtime_error("Skipping strict-sync integration path: " +
                                     std::string(e.what()));
          }
          throw;
        }

        std::atomic<int64_t> cb_ok{0};
        std::atomic<int64_t> cb_fail{0};
        std::atomic<int64_t> cb_drop{0};

        ReleasePacer pacer(
            std::vector<std::shared_ptr<simaai::neat::Run>>{run},
            /*target_fps=*/120,
            /*max_queue=*/64,
            [&](size_t, bool ok) {
              if (ok) {
                cb_ok.fetch_add(1);
              } else {
                cb_fail.fetch_add(1);
              }
            },
            [&](size_t, int64_t dropped) { cb_drop.fetch_add(dropped); });

        const int send_count = 24;
        for (int i = 0; i < send_count; ++i) {
          require(pacer.enqueue(
                      0, sima_test::make_tensor_sample(i, "pacer", -1, static_cast<uint8_t>(i))),
                  "ReleasePacer enqueue should succeed");
        }

        int pulled = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
        while (std::chrono::steady_clock::now() < deadline && pulled < send_count) {
          auto out = run->pull(120);
          if (out.has_value()) {
            ++pulled;
          }
        }

        pacer.stop();
        const auto stats = pacer.stats(0);

        require(stats.enqueued == send_count, "ReleasePacer enqueued stats mismatch");
        require(stats.sent > 0, "ReleasePacer should deliver samples to real Run backend");
        require(stats.send_fail == 0, "ReleasePacer should not fail sends on healthy Run backend");
        require(cb_ok.load() == stats.sent, "ReleasePacer callback sent count mismatch");
        require(cb_fail.load() == stats.send_fail, "ReleasePacer callback fail count mismatch");
        require(cb_drop.load() == stats.dropped, "ReleasePacer callback drop count mismatch");

        run->stop();

        // Pressure-only path: bounded queue drops should be accounted.
        ReleasePacer pressure(
            std::vector<std::shared_ptr<simaai::neat::Run>>{std::shared_ptr<simaai::neat::Run>{}},
            /*target_fps=*/30,
            /*max_queue=*/2, {}, {});

        for (int i = 0; i < 18; ++i) {
          require(pressure.enqueue(0, sima_test::make_tensor_sample(i, "pressure")),
                  "ReleasePacer pressure enqueue should succeed");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        pressure.stop();
        const auto pstats = pressure.stats(0);
        require(pstats.dropped > 0, "ReleasePacer should drop when bounded queue is exceeded");
      }
    }));

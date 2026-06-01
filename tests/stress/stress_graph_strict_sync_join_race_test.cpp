#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StrictSync.h"
#include "graph/nodes/JoinBundle.h"
#include "graph_test_utils.h"
#include "test_main.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(40, std::min(value, 4000));
}

} // namespace

RUN_TEST(
    "stress_graph_strict_sync_join_race_test", ([] {
      using simaai::neat::graph::Graph;
      using simaai::neat::graph::GraphRun;
      using simaai::neat::graph::GraphRunOptions;
      using simaai::neat::graph::build;
      using simaai::neat::graph::PortId;
      using simaai::neat::graph::strict_sync::PendingVideoStore;
      using simaai::neat::graph::strict_sync::YoloTokenStore;

      const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 240));
      const int cycles = std::max(1, iters / 80);
      const int frames_per_cycle = std::max(60, std::min(600, iters * 2));

      // StrictSync stores under concurrent producer/consumer pressure.
      for (int cycle = 0; cycle < cycles; ++cycle) {
        PendingVideoStore pending(2);
        YoloTokenStore tokens(2);

        const int total = frames_per_cycle * 2;
        std::atomic<int> matched{0};
        std::atomic<int> misses{0};

        auto producer = [&](size_t stream_idx) {
          const std::string stream_id = "strict-" + std::to_string(stream_idx);
          for (int i = 0; i < frames_per_cycle; ++i) {
            simaai::neat::Sample sample = sima_test::make_tensor_sample(
                i, stream_id, -1, static_cast<uint8_t>((i + stream_idx) & 0xFF));
            require(pending.enqueue(stream_idx, i, std::move(sample), 10, 128),
                    "PendingVideoStore enqueue failed in strict-sync stress");
            tokens.enqueue(stream_idx, i);
          }
        };

        std::thread p0(producer, 0);
        std::thread p1(producer, 1);

        std::thread consumer([&] {
          const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
          while (matched.load() < total && std::chrono::steady_clock::now() < deadline) {
            auto ordered = tokens.take_ordered();
            if (!ordered.has_value()) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
              continue;
            }
            auto frame = pending.take(ordered->stream_idx, ordered->frame_id);
            if (frame.has_value()) {
              matched.fetch_add(1);
            } else {
              misses.fetch_add(1);
            }
          }
        });

        p0.join();
        p1.join();
        consumer.join();

        require(matched.load() == total, "strict-sync store race stress lost matched frames");
        require(misses.load() == 0, "strict-sync store race stress observed misses");
      }

      // Concurrent JoinBundle push paths with deterministic cardinality.
      for (int cycle = 0; cycle < cycles; ++cycle) {
        Graph g;

        simaai::neat::graph::nodes::JoinBundleOptions join_opt;
        join_opt.inputs = {"encoded", "meta"};
        join_opt.emit_partial = false;
        join_opt.max_pending_keys = static_cast<size_t>(frames_per_cycle * 2);

        const auto join = g.add(simaai::neat::graph::nodes::JoinBundleNode(join_opt.inputs, "join",
                                                                           "bundle", join_opt));
        const auto sink = g.add(sima_test::make_pass_stage_node("sink"));

        g.connect(join, sink, "bundle", "in");

        const PortId encoded_port = g.intern_port("encoded");
        const PortId meta_port = g.intern_port("meta");
        GraphRunOptions run_opt;
        run_opt.edge_queue = static_cast<size_t>(std::max(512, frames_per_cycle * 4));
        run_opt.pull_timeout_ms = 150;

        GraphRun run = simaai::neat::graph::build(std::move(g), run_opt);

        std::mutex err_mu;
        std::string producer_err;
        std::string consumer_err;
        std::atomic<int> produced_encoded{0};
        std::atomic<int> produced_meta{0};
        std::atomic<int> consumed{0};

        std::thread producer_encoded([&] {
          try {
            for (int i = 0; i < frames_per_cycle; ++i) {
              simaai::neat::Sample sample =
                  sima_test::make_tensor_sample(i, "join-race", -1, static_cast<uint8_t>(i & 0xFF));
              if (!run.push(join, encoded_port, simaai::neat::Sample{sample})) {
                throw std::runtime_error("encoded push failed");
              }
              produced_encoded.fetch_add(1);
              if ((i % 16) == 0) {
                std::this_thread::yield();
              }
            }
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(err_mu);
            producer_err = e.what();
          }
        });

        std::thread producer_meta([&] {
          try {
            for (int i = 0; i < frames_per_cycle; ++i) {
              simaai::neat::Sample sample = sima_test::make_tensor_sample(
                  i, "join-race", -1, static_cast<uint8_t>((i + 33) & 0xFF));
              if (!run.push(join, meta_port, simaai::neat::Sample{sample})) {
                throw std::runtime_error("meta push failed");
              }
              produced_meta.fetch_add(1);
              if ((i % 16) == 0) {
                std::this_thread::yield();
              }
            }
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(err_mu);
            producer_err = e.what();
          }
        });

        std::thread consumer([&] {
          try {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(25);
            while (consumed.load() < frames_per_cycle &&
                   std::chrono::steady_clock::now() < deadline) {
              auto out = run.pull(sink, 200);
              if (!out.has_value()) {
                continue;
              }
              if (out->kind != simaai::neat::SampleKind::Bundle) {
                throw std::runtime_error("join output kind mismatch");
              }
              if (out->fields.size() != 2) {
                throw std::runtime_error("join output field cardinality mismatch");
              }
              consumed.fetch_add(1);
            }
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(err_mu);
            consumer_err = e.what();
          }
        });

        producer_encoded.join();
        producer_meta.join();
        consumer.join();

        {
          std::lock_guard<std::mutex> lock(err_mu);
          if (!producer_err.empty()) {
            throw std::runtime_error("strict-sync join stress producer failed: " + producer_err);
          }
          if (!consumer_err.empty()) {
            throw std::runtime_error("strict-sync join stress consumer failed: " + consumer_err);
          }
        }

        require(produced_encoded.load() == frames_per_cycle,
                "strict-sync join stress encoded producer count mismatch");
        require(produced_meta.load() == frames_per_cycle,
                "strict-sync join stress meta producer count mismatch");
        require(consumed.load() == frames_per_cycle,
                "strict-sync join stress consumed cardinality mismatch");

        run.stop();
      }
    }));

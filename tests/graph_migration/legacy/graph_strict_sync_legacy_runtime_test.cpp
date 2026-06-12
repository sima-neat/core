#include "graph/StrictSync.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace {

simaai::neat::Sample make_sample(int64_t frame_id, const std::string& stream_id) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(8, 6, simaai::neat::ImageSpec::PixelFormat::RGB,
                                    static_cast<uint8_t>(frame_id & 0xFF));
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  return sample;
}

} // namespace

RUN_TEST("graph_migration_legacy_graph_strict_sync_test", ([] {
           using simaai::neat::graph::strict_sync::PendingVideoStore;
           using simaai::neat::graph::strict_sync::ReleasePacer;
           using simaai::neat::graph::strict_sync::YoloTokenStore;

           // PendingVideoStore: enqueue/take semantics and stats.
           {
             PendingVideoStore store(2);

             require(!store.enqueue(9, 1, make_sample(1, "a"), 11, 32),
                     "PendingVideoStore should reject out-of-range stream index");
             require(!store.enqueue(0, -1, make_sample(-1, "a"), 11, 32),
                     "PendingVideoStore should reject negative frame ids");

             require(store.enqueue(0, 5, make_sample(5, "s0"), 20, 100),
                     "PendingVideoStore enqueue failed");
             require(store.enqueue(0, 5, make_sample(5, "s0-override"), 21, 80),
                     "PendingVideoStore duplicate enqueue replacement failed");

             auto stats = store.stats(0);
             require(stats.enqueued == 2, "PendingVideoStore enqueue stats mismatch");
             require(stats.pending_depth == 1,
                     "PendingVideoStore pending depth should dedupe by frame id");
             require(stats.pending_bytes == 80,
                     "PendingVideoStore pending bytes should reflect replacement");

             auto hit = store.take(0, 5);
             require(hit.has_value(), "PendingVideoStore take should find enqueued frame");
             require(hit->sample.stream_id == "s0-override",
                     "PendingVideoStore should return latest sample for duplicate frame id");
             require(hit->cap_ms == 21, "PendingVideoStore should return latest cap_ms");

             auto miss = store.take(0, 5);
             require(!miss.has_value(), "PendingVideoStore take should miss consumed frame");

             stats = store.stats(0);
             require(stats.matched == 1, "PendingVideoStore matched stats mismatch");
             require(stats.miss >= 1, "PendingVideoStore miss stats should increment");
           }

           // YoloTokenStore: ordered and per-stream token behavior.
           {
             YoloTokenStore tokens(2);

             tokens.enqueue(1, 100);
             tokens.enqueue(0, 200);
             tokens.enqueue(1, 101);

             auto o0 = tokens.take_ordered();
             auto o1 = tokens.take_ordered();
             auto o2 = tokens.take_ordered();
             auto o3 = tokens.take_ordered();

             require(o0.has_value() && o0->stream_idx == 1 && o0->frame_id == 100,
                     "YoloTokenStore ordered token #0 mismatch");
             require(o1.has_value() && o1->stream_idx == 0 && o1->frame_id == 200,
                     "YoloTokenStore ordered token #1 mismatch");
             require(o2.has_value() && o2->stream_idx == 1 && o2->frame_id == 101,
                     "YoloTokenStore ordered token #2 mismatch");
             require(!o3.has_value(), "YoloTokenStore ordered queue should drain");

             auto s1t0 = tokens.take(1);
             auto s1t1 = tokens.take(1);
             auto s1t2 = tokens.take(1);
             require(s1t0.has_value() && s1t0->frame_id == 100,
                     "YoloTokenStore per-stream dequeue order mismatch (first)");
             require(s1t1.has_value() && s1t1->frame_id == 101,
                     "YoloTokenStore per-stream dequeue order mismatch (second)");
             require(!s1t2.has_value(), "YoloTokenStore per-stream queue should drain");

             const auto stats = tokens.stats(1);
             require(stats.enqueued == 2, "YoloTokenStore stream stats enqueue mismatch");
             require(stats.dequeued == 2, "YoloTokenStore stream stats dequeue mismatch");
             require(stats.miss >= 1, "YoloTokenStore stream miss stats should increment");
           }

           // ReleasePacer: bounded queue drop/send-fail accounting and callback paths.
           {
             std::vector<std::shared_ptr<simaai::neat::Run>> runs(1);

             std::atomic<int64_t> callback_send_ok{0};
             std::atomic<int64_t> callback_send_fail{0};
             std::atomic<int64_t> callback_drops{0};

             ReleasePacer pacer(
                 runs,
                 /*target_fps=*/20,
                 /*max_queue=*/2,
                 [&](size_t idx, bool ok) {
                   require(idx == 0, "ReleasePacer callback stream index mismatch");
                   if (ok) {
                     callback_send_ok.fetch_add(1);
                   } else {
                     callback_send_fail.fetch_add(1);
                   }
                 },
                 [&](size_t idx, int64_t dropped) {
                   require(idx == 0, "ReleasePacer drop callback stream index mismatch");
                   callback_drops.fetch_add(dropped);
                 });

             for (int i = 0; i < 20; ++i) {
               require(pacer.enqueue(0, make_sample(i, "pacer")), "ReleasePacer enqueue failed");
             }

             std::this_thread::sleep_for(std::chrono::milliseconds(500));
             pacer.stop();
             pacer.stop(); // idempotent

             const auto stats = pacer.stats(0);
             require(stats.enqueued == 20, "ReleasePacer enqueued stats mismatch");
             require(stats.send_fail > 0,
                     "ReleasePacer should report send failures when run handle is null");
             require(stats.dropped > 0,
                     "ReleasePacer should drop when enqueue pressure exceeds max_queue");
             require(callback_send_fail.load() == stats.send_fail,
                     "ReleasePacer send-fail callback count mismatch");
             require(callback_drops.load() == stats.dropped,
                     "ReleasePacer drop callback count mismatch");
           }
         }));

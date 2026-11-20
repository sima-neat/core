// DeferredPoolBufferReleaser returns a pooled parent only after every other buffer and memory
// reference is gone. Re-checks follow enqueue() and wake(); a release that reports nothing is
// still found by the backed-off timed re-check, and stopping the releaser drops what it holds.
#include "pipeline/gst/DeferredPoolBufferReleaser.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace {

using simaai::neat::DeferredPoolBufferReleaser;
using Clock = std::chrono::steady_clock;
// Far above the 4 ms re-check bound, so only a missing re-check fails on a loaded host.
constexpr std::chrono::milliseconds kPrompt{250};

// Records when a root is finalized, which is when the releaser dropped the last reference.
struct FinalizeProbe {
  std::atomic<bool> finalized{false};
  std::atomic<Clock::rep> at{0};
};

void on_finalize(gpointer data, GstMiniObject*) {
  auto* probe = static_cast<FinalizeProbe*>(data);
  probe->at.store(Clock::now().time_since_epoch().count());
  probe->finalized.store(true);
}

GstBuffer* make_root(FinalizeProbe& probe) {
  GstBuffer* root = gst_buffer_new_allocate(nullptr, 64, nullptr);
  require(root != nullptr, "failed to allocate root buffer");
  gst_mini_object_weak_ref(GST_MINI_OBJECT_CAST(root), on_finalize, &probe);
  return root;
}

bool wait_finalized(const FinalizeProbe& probe, std::chrono::milliseconds limit) {
  const auto deadline = Clock::now() + limit;
  while (!probe.finalized.load() && Clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return probe.finalized.load();
}

std::chrono::milliseconds since(Clock::time_point start, const FinalizeProbe& probe) {
  const Clock::time_point at{Clock::duration(probe.at.load())};
  return std::chrono::duration_cast<std::chrono::milliseconds>(at - start);
}

void root_held_elsewhere_is_rechecked_with_backoff() {
  DeferredPoolBufferReleaser releaser;
  FinalizeProbe probe;
  GstBuffer* root = make_root(probe);
  GstBuffer* other_holder = gst_buffer_ref(root);

  const auto shared_from = Clock::now();
  releaser.enqueue({root});
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  require(!probe.finalized.load(), "root released while another holder still referenced it");
  // A 1 ms poll checks about once per elapsed millisecond; the backed-off re-check about once
  // per kMaxRecheckInterval. Bound by the elapsed time so a slow host cannot fail the test.
  const auto checks = releaser.checks();
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - shared_from).count();
  require(checks >= 2 && checks <= static_cast<std::uint64_t>(elapsed_ms / 2 + 4),
          "unexpected number of re-checks while the root stayed shared: " + std::to_string(checks) +
              " in " + std::to_string(elapsed_ms) + " ms");

  const auto released_from = Clock::now();
  gst_buffer_unref(other_holder);
  releaser.wake();
  require(wait_finalized(probe, std::chrono::seconds(2)), "root not released after wake()");
  require(since(released_from, probe) < kPrompt, "wake() did not trigger a prompt re-check");
}

void silent_release_is_found_by_timed_recheck() {
  DeferredPoolBufferReleaser releaser;
  FinalizeProbe probe;
  GstBuffer* root = make_root(probe);
  GstBuffer* other_holder = gst_buffer_ref(root);

  releaser.enqueue({root});
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const auto released_from = Clock::now();
  gst_buffer_unref(other_holder); // no wake(): nothing reports this release
  require(wait_finalized(probe, std::chrono::seconds(2)),
          "root not released after its last other holder dropped it");
  require(since(released_from, probe) < DeferredPoolBufferReleaser::kMaxRecheckInterval + kPrompt,
          "timed re-check exceeded its bound");
}

void shared_memory_keeps_root_until_child_is_gone() {
  DeferredPoolBufferReleaser releaser;
  FinalizeProbe probe;
  GstBuffer* root = make_root(probe);
  // A child that shares the root's memory, as a downstream transfer buffer does.
  GstBuffer* child = gst_buffer_new();
  gst_buffer_append_memory(child, gst_buffer_get_memory(root, 0));

  releaser.enqueue({root});
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  require(!probe.finalized.load(), "root released while its memory was still shared");

  gst_buffer_unref(child); // the memory release itself reports nothing
  require(wait_finalized(probe, std::chrono::seconds(2)),
          "root not released after the shared memory was dropped");
}

void redundant_enqueue_is_consumed_and_rechecks() {
  DeferredPoolBufferReleaser releaser;
  FinalizeProbe probe;
  GstBuffer* root = make_root(probe);
  GstBuffer* second_proxy_ref = gst_buffer_ref(root);

  releaser.enqueue({root});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  require(!probe.finalized.load(), "root released while a second proxy still held it");

  const auto released_from = Clock::now();
  releaser.enqueue({second_proxy_ref}); // the second proxy hands over its reference
  require(wait_finalized(probe, std::chrono::seconds(2)),
          "root not released after the last proxy reference arrived");
  require(since(released_from, probe) < kPrompt,
          "enqueue of the last reference did not trigger a prompt re-check");
}

void stop_releases_pending_roots() {
  FinalizeProbe probe;
  GstBuffer* root = make_root(probe);
  GstBuffer* other_holder = gst_buffer_ref(root);
  {
    DeferredPoolBufferReleaser releaser;
    releaser.wake(); // nothing pending: no effect
    releaser.enqueue({root});
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(GST_MINI_OBJECT_REFCOUNT_VALUE(other_holder) == 1,
          "stopping the releaser did not drop its retained reference");
  require(!probe.finalized.load(), "root finalized while another holder still referenced it");
  gst_buffer_unref(other_holder);
  require(probe.finalized.load(), "root not finalized after its last reference");
}

} // namespace

RUN_TEST("unit_deferred_pool_buffer_releaser_test", ([] {
           gst_init(nullptr, nullptr);
           root_held_elsewhere_is_rechecked_with_backoff();
           silent_release_is_found_by_timed_recheck();
           shared_memory_keeps_root_until_child_is_gone();
           redundant_enqueue_is_consumed_and_rechecks();
           stop_releases_pending_roots();
         }));

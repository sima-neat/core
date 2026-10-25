#include "gst/GstInit.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <time.h>
#include <type_traits>
#include <vector>

namespace {
// Executable-local fault: std::thread allocates its state after start() has
// installed all four callbacks. No production hook or device operation.
thread_local bool fail_next_allocation = false;

// Executable-local syscall observation: pause a real registry release after it
// removed its entry but before a retained lane signals the link's condition.
// No production hook, private-member access or freed-condition signal is needed.
using ConditionFn = int (*)(pthread_cond_t*);
ConditionFn real_condition_signal() {
  static const auto fn = reinterpret_cast<ConditionFn>(dlsym(RTLD_NEXT, "pthread_cond_signal"));
  return fn;
}
ConditionFn real_condition_destroy() {
  static const auto fn = reinterpret_cast<ConditionFn>(dlsym(RTLD_NEXT, "pthread_cond_destroy"));
  return fn;
}
struct LateWakeProbe {
  unsigned pause_on = 0;
  std::atomic<pthread_cond_t*> condition{nullptr};
  std::atomic<bool> entered{false}, resume{false}, timed_out{false}, destroyed{false};
  std::atomic<unsigned> signals{0}, actual_signals{0}, destroys{0};
};
static_assert(std::is_trivially_destructible_v<LateWakeProbe>);
std::atomic<LateWakeProbe*> active_wake_probe{nullptr};
thread_local LateWakeProbe* release_wake_probe = nullptr;
} // namespace

void* operator new(std::size_t bytes) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(bytes == 0 ? 1 : bytes)) {
    return memory;
  }
  throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
  std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}

extern "C" int pthread_cond_signal(pthread_cond_t* condition) noexcept {
  auto* probe = release_wake_probe;
  if (probe) {
    const auto ordinal = probe->signals.fetch_add(1) + 1;
    if (ordinal == probe->pause_on) {
      probe->condition.store(condition);
      probe->entered.store(true);
      for (unsigned attempt = 0; !probe->resume.load() && attempt < 2000; ++attempt) {
        const timespec delay{0, 1000000};
        (void)nanosleep(&delay, nullptr);
      }
      if (!probe->resume.load()) {
        probe->timed_out.store(true);
      }
    }
    // The negative control records the lifetime violation without invoking UB.
    if (probe->condition.load() == condition && probe->destroyed.load()) {
      return 0;
    }
    probe->actual_signals.fetch_add(1);
  }
  const auto signal = real_condition_signal();
  if (!signal)
    std::abort();
  return signal(condition);
}

extern "C" int pthread_cond_destroy(pthread_cond_t* condition) noexcept {
  if (auto* probe = active_wake_probe.load(); probe && probe->condition.load() == condition) {
    probe->destroys.fetch_add(1);
    probe->destroyed.store(true);
  }
  const auto destroy = real_condition_destroy();
  if (!destroy)
    std::abort();
  return destroy(condition);
}

namespace {

simaai::neat::Sample make_raw_holder_sample(std::string stream_id, std::int64_t frame_id) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  require(buffer != nullptr, "failed to allocate GstBuffer");

  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
                                      G_TYPE_INT, 4, "height", G_TYPE_INT, 4, nullptr);
  require(caps != nullptr, "failed to allocate GstCaps");
  GstSample* gst_sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(gst_sample != nullptr, "failed to allocate GstSample");

  simaai::neat::Tensor tensor;
  tensor.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(gst_sample);
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.shape = {16};
  gst_sample_unref(gst_sample);

  simaai::neat::Sample sample = simaai::neat::sample_from_tensors({tensor});
  sample.stream_id = std::move(stream_id);
  sample.frame_id = frame_id;
  sample.media_type = "video/x-raw";
  sample.payload_tag = "NV12";
  sample.format = "NV12";
  return sample;
}

simaai::neat::runtime::RealtimeLatestLink::Stats
run_one_raw_sample(const simaai::neat::GraphLinkOptions& options, int stream_count = 1,
                   bool use_explicit_stream_ids = true) {
  simaai::neat::runtime::DownstreamTarget target{
      simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
      0U,
      simaai::neat::graph::kInvalidPort,
      0U,
  };
  simaai::neat::runtime::RealtimeLatestLink link(target, options,
                                                 use_explicit_stream_ids ? "stream0" : "");
  for (int i = 1; i < stream_count; ++i) {
    link.add_edge_stream_id(static_cast<std::size_t>(i),
                            use_explicit_stream_ids ? "stream" + std::to_string(i) : "");
  }

  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  link.start(
      [&](const simaai::neat::runtime::DownstreamTarget&, simaai::neat::Sample&& sample,
          std::size_t) {
        simaai::neat::pipeline_internal::release_realtime_frame_credits(
            simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(sample),
            "unit-realtime-link-options");
        {
          std::lock_guard<std::mutex> lock(mu);
          done = true;
        }
        cv.notify_all();
        return true;
      },
      [] { return false; },
      [&](std::string msg) {
        std::lock_guard<std::mutex> lock(mu);
        error = std::move(msg);
        done = true;
        cv.notify_all();
      });

  require(
      link.offer(make_raw_holder_sample("stream0", 1), simaai::neat::runtime::invalid_edge_index()),
      "RealtimeLatestLink should accept the sample");
  {
    std::unique_lock<std::mutex> lock(mu);
    require(cv.wait_for(lock, std::chrono::seconds(2), [&] { return done; }),
            "timed out waiting for realtime link dispatch");
  }
  require(error.empty(), "RealtimeLatestLink dispatch failed: " + error);
  const auto stats = link.stats();
  link.close();
  link.join();
  return stats;
}

void verify_raw_frame_cap_blocks_after(
    const simaai::neat::GraphLinkOptions& options, int stream_count, int admitted_frames,
    const std::string& label, bool use_explicit_link_stream_ids = true,
    const std::vector<simaai::neat::GraphLinkOptions>& additional_edge_options = {}) {
  simaai::neat::runtime::DownstreamTarget target{
      simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
      0U,
      simaai::neat::graph::kInvalidPort,
      0U,
  };

  simaai::neat::runtime::RealtimeLatestLink link(target, options,
                                                 use_explicit_link_stream_ids ? "stream0" : "");
  for (int i = 1; i < stream_count; ++i) {
    const std::string stream_id = use_explicit_link_stream_ids ? "stream" + std::to_string(i) : "";
    const std::size_t options_index = static_cast<std::size_t>(i - 1);
    if (options_index < additional_edge_options.size()) {
      link.add_edge_stream_id(static_cast<std::size_t>(i), stream_id,
                              additional_edge_options[options_index]);
    } else {
      link.add_edge_stream_id(static_cast<std::size_t>(i), stream_id);
    }
  }

  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::int64_t> dispatched_frames;
  std::vector<std::vector<simaai::neat::pipeline_internal::RealtimeFrameCredit>> dispatched_credits;
  std::string error;

  link.start(
      [&](const simaai::neat::runtime::DownstreamTarget&, simaai::neat::Sample&& sample,
          std::size_t) {
        const auto credits =
            simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(sample);
        {
          std::lock_guard<std::mutex> lock(mu);
          dispatched_frames.push_back(sample.frame_id);
          dispatched_credits.push_back(credits);
        }
        cv.notify_all();
        return true;
      },
      [] { return false; },
      [&](std::string msg) {
        std::lock_guard<std::mutex> lock(mu);
        error = std::move(msg);
        cv.notify_all();
      });

  const auto wait_for_dispatch_count = [&](std::size_t count) {
    std::unique_lock<std::mutex> lock(mu);
    require(cv.wait_for(lock, std::chrono::seconds(2),
                        [&] { return dispatched_frames.size() >= count || !error.empty(); }),
            "timed out waiting for realtime dispatch count " + std::to_string(count));
    require(error.empty(), "RealtimeLatestLink dispatch failed: " + error);
  };

  for (int frame = 0; frame < admitted_frames; ++frame) {
    const int stream = frame % stream_count;
    require(link.offer(make_raw_holder_sample("stream" + std::to_string(stream), frame),
                       simaai::neat::runtime::invalid_edge_index()),
            "RealtimeLatestLink should accept raw frame " + std::to_string(frame));
    wait_for_dispatch_count(static_cast<std::size_t>(frame + 1));
  }

  require(link.offer(make_raw_holder_sample("stream0", admitted_frames),
                     simaai::neat::runtime::invalid_edge_index()),
          "RealtimeLatestLink should accept a pending frame after " + label + " is full");
  {
    std::unique_lock<std::mutex> lock(mu);
    (void)cv.wait_for(lock, std::chrono::milliseconds(200), [&] {
      return dispatched_frames.size() > static_cast<std::size_t>(admitted_frames) || !error.empty();
    });
    require(error.empty(), "RealtimeLatestLink dispatch failed: " + error);
    require(dispatched_frames.size() == static_cast<std::size_t>(admitted_frames),
            label + " should hold the next frame pending");
  }
  require(link.stats().no_credit_skips > 0U,
          label + " should reject admission while raw frames are held");

  require(!dispatched_credits.empty(), "expected held realtime frame credits");
  simaai::neat::pipeline_internal::release_realtime_frame_credits(
      dispatched_credits.front(), "unit-realtime-link-options-release-one");
  wait_for_dispatch_count(static_cast<std::size_t>(admitted_frames + 1));

  link.close();
  link.join();
}

void verify_per_stream_cap_blocks_after_two_raw_frames() {
  simaai::neat::GraphLinkOptions options;
  options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  options.max_inflight_per_stream = 2;
  options.max_inflight_total = 10;
  verify_raw_frame_cap_blocks_after(options, 1, 2, "explicit per-stream raw-frame cap");
}

void verify_multiplexed_stream_ids_scale_default_total_cap() {
  simaai::neat::GraphLinkOptions options;
  options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  options.max_inflight_per_stream = 4;
  verify_raw_frame_cap_blocks_after(options, 4, 8,
                                    "multiplexed stream-id default global raw-frame cap",
                                    /*use_explicit_link_stream_ids=*/false);
}

void verify_asymmetric_fan_in_uses_strictest_caps() {
  simaai::neat::GraphLinkOptions first_options;
  first_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  first_options.max_inflight_per_stream = 8;
  first_options.max_inflight_total = 8;

  simaai::neat::GraphLinkOptions second_options = first_options;
  second_options.max_inflight_per_stream = 1;
  second_options.max_inflight_total = 2;

  simaai::neat::runtime::DownstreamTarget target{
      simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
      0U,
      simaai::neat::graph::kInvalidPort,
      0U,
  };
  simaai::neat::runtime::RealtimeLatestLink link(target, first_options, "stream0");
  link.add_edge_stream_id(1U, "stream1", second_options);
  require(link.options().max_inflight_per_stream == 1,
          "non-fused fan-in must enforce the strictest producer per-stream cap");
  require(link.options().max_inflight_total == 2,
          "non-fused fan-in must enforce the strictest producer total cap");
  verify_raw_frame_cap_blocks_after(first_options, 2, 2,
                                    "asymmetric non-fused fan-in admission caps",
                                    /*use_explicit_link_stream_ids=*/true, {second_options});
}

void verify_callbacks_retained_until_join() {
  using namespace simaai::neat;
  runtime::DownstreamTarget target{runtime::DownstreamTarget::Kind::PipelineInput, 0U,
                                   graph::kInvalidPort, 0U};
  GraphLinkOptions options;
  options.policy = GraphLinkPolicy::RealtimeLatestByStream;
  runtime::RealtimeLatestLink link(target, options, "stream0");
  std::array<std::shared_ptr<int>, 4> owners;
  std::array<std::weak_ptr<int>, 4> observed;
  for (std::size_t i = 0; i < owners.size(); ++i) {
    owners[i] = std::make_shared<int>(static_cast<int>(i));
    observed[i] = owners[i];
  }
  std::mutex mu;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;
  bool timed_out = false;
  bool joining = false;
  bool joined = false;
  bool self_join_rejected = false;
  link.start(
      [owner = owners[0], &link, &mu, &cv, &entered, &released, &timed_out,
       &self_join_rejected](const runtime::DownstreamTarget&, Sample&&, std::size_t) {
        try {
          link.join();
        } catch (const std::system_error&) {
          self_join_rejected = true;
        }
        std::unique_lock lock(mu);
        entered = true;
        cv.notify_all();
        timed_out = !cv.wait_for(lock, std::chrono::seconds(2), [&] { return released; });
        return true;
      },
      [owner = owners[1]] { return false; }, [owner = owners[2]](const std::string&) {},
      [owner = owners[3]] {});
  owners = {};
  require(link.offer(make_raw_holder_sample("stream0", 1), runtime::invalid_edge_index()),
          "callback retirement: sample offer failed");
  bool dispatched = false;
  {
    std::unique_lock lock(mu);
    dispatched = cv.wait_for(lock, std::chrono::seconds(2), [&] { return entered; });
  }
  link.close();
  std::thread joiner([&] {
    {
      std::lock_guard lock(mu);
      joining = true;
    }
    cv.notify_all();
    link.join();
    {
      std::lock_guard lock(mu);
      joined = true;
    }
    cv.notify_all();
  });
  bool retained = false;
  {
    std::unique_lock lock(mu);
    const bool join_started = cv.wait_for(lock, std::chrono::seconds(2), [&] { return joining; });
    retained = join_started && !joined;
    for (const auto& owner : observed) {
      retained = retained && !owner.expired();
    }
    released = true;
  }
  cv.notify_all();
  joiner.join();
  require(dispatched && self_join_rejected && retained && !timed_out,
          "callbacks must retain every owner after failed join and while dispatch is active");
  for (const auto& owner : observed) {
    require(owner.expired(), "joined realtime link retained a callback owner");
  }
  link.join();
  link.close();
  link.join();
}

void verify_late_credit_wake_outlives_link() {
  using namespace simaai::neat;
  namespace credits = pipeline_internal;
  // A real frame owns one stream credit and one global companion credit.
  // Exercise both retained-lane wake positions.
  // Check the last wake first: the raw-this negative control then fails before
  // another callback could dereference the destroyed link.
  // Interposed destroy calls on other threads may have observed the probe
  // before it was disarmed. Keep each once-used probe alive for the process.
  static std::array<LateWakeProbe, 2> probes;
  for (unsigned wake_ordinal : {2U, 1U}) {
    runtime::DownstreamTarget target{runtime::DownstreamTarget::Kind::PipelineInput, 0U,
                                     graph::kInvalidPort, 0U};
    GraphLinkOptions options;
    options.policy = GraphLinkPolicy::RealtimeLatestByStream;
    options.max_inflight_per_stream = 1;
    options.max_inflight_total = 1;
    auto link = std::make_unique<runtime::RealtimeLatestLink>(target, options, "stream0");
    std::mutex mu;
    std::condition_variable cv;
    std::vector<credits::RealtimeFrameCredit> held;
    link->start(
        [&](const runtime::DownstreamTarget&, Sample&& sample, std::size_t) {
          {
            std::lock_guard lock(mu);
            held = credits::realtime_frame_credits_for_sample(sample);
          }
          cv.notify_all();
          return true;
        },
        [] { return false; }, [](const std::string&) {});
    require(link->offer(make_raw_holder_sample("stream0", 1), runtime::invalid_edge_index()),
            "late wake: sample offer failed");
    {
      std::unique_lock lock(mu);
      require(cv.wait_for(lock, std::chrono::seconds(2), [&] { return !held.empty(); }),
              "late wake: dispatch did not retain a frame credit");
    }
    require(held.size() == 1, "late wake: expected one registered frame identity");
    const auto before = link->stats();
    require(before.credit_registered == 2 && before.credit_inflight == 2,
            "late wake: both stream and global lane must hold the frame");

    auto& probe = probes[wake_ordinal - 1];
    probe.pause_on = wake_ordinal;
    bool released = false;
    std::thread releaser([&] {
      active_wake_probe.store(&probe);
      release_wake_probe = &probe;
      released = credits::release_registered_realtime_frame_credit(held.front(),
                                                                   "unit-late-link-wake", true);
      release_wake_probe = nullptr;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!probe.entered.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool entered = probe.entered.load();
    bool removed = false;
    bool accounting = false;
    bool destroyed_early = false;
    if (entered) {
      const auto pending = link->stats();
      accounting = pending.credit_released_by_output == wake_ordinal &&
                   pending.credit_inflight == 2 - wake_ordinal &&
                   pending.credit_released_without_output == 0;
      removed = !credits::release_registered_realtime_frame_credit(
          held.front(), "unit-late-link-duplicate", true);
      // close()/join() cannot find the removed registry entry, but its release
      // thread still owns both lanes and has not called the selected real wake.
      link.reset();
      destroyed_early = probe.destroyed.load();
    }
    // Resume/join before any assertion, including timeout failures. If no wake
    // was observed, keep the link alive until the release thread has returned.
    probe.resume.store(true);
    releaser.join();
    link.reset();
    active_wake_probe.store(nullptr);
    require(entered && !probe.timed_out.load(), "late wake: interposition was not observed");
    require(removed && accounting && released,
            "late wake: registry removal or exact-once lane accounting failed");
    require(!destroyed_early && probe.signals.load() == 2 && probe.actual_signals.load() == 2,
            "retained credit lane signalled a condition destroyed with its link");
    require(probe.destroys.load() == 1,
            "late wake: retained condition was not destroyed exactly once after release");
    require(!credits::release_registered_realtime_frame_credit(
                held.front(), "unit-late-link-after-release", true),
            "late wake: completed credit remained registered");
  }
}

void verify_failed_thread_start_releases_callbacks() {
  using namespace simaai::neat;
  runtime::DownstreamTarget target{runtime::DownstreamTarget::Kind::PipelineInput, 0U,
                                   graph::kInvalidPort, 0U};
  runtime::RealtimeLatestLink link(target, GraphLinkOptions{}, "stream0");
  auto owner = std::make_shared<int>(0);
  const std::weak_ptr<int> observed = owner;
  runtime::RealtimeLatestLink::DispatchFn dispatch =
      [owner](const runtime::DownstreamTarget&, Sample&&, std::size_t) { return true; };
  runtime::RealtimeLatestLink::StopFn stop = [owner] { return false; };
  runtime::RealtimeLatestLink::ErrorFn error = [owner](const std::string&) {};
  runtime::RealtimeLatestLink::CompleteFn complete = [owner] {};
  owner.reset();
  bool failed = false;
  // All potentially allocating callback conversions are complete. Moving them
  // into start() is noexcept; the next allocation belongs to std::thread.
  fail_next_allocation = true;
  try {
    link.start(std::move(dispatch), std::move(stop), std::move(error), std::move(complete));
  } catch (const std::bad_alloc&) {
    failed = true;
  }
  const bool injected = !fail_next_allocation;
  fail_next_allocation = false;
  // Do not depend on the unspecified state of moved-from std::function objects.
  dispatch = {};
  stop = {};
  error = {};
  complete = {};
  const bool retained_before_rollback = !observed.expired();
  link.close();
  link.join();
  require(failed && injected && retained_before_rollback,
          "thread-start failure must leave installed callbacks for terminal rollback");
  require(observed.expired(), "no-thread rollback retained installed callback owners");
  link.join();
}

} // namespace

int main() {
  try {
    require(real_condition_signal() && real_condition_destroy(),
            "failed to resolve condition-variable syscall observation");
    simaai::neat::gst_init_once();
    ::unsetenv("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL");
    verify_callbacks_retained_until_join();
    verify_failed_thread_start_releases_callbacks();
    verify_late_credit_wake_outlives_link();

    simaai::neat::GraphLinkOptions default_options;
    default_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    default_options.queue_depth = 1;
    verify_raw_frame_cap_blocks_after(default_options, 4, 8, "default global raw-frame cap",
                                      /*use_explicit_link_stream_ids=*/false);

    simaai::neat::GraphLinkOptions explicit_total_options;
    explicit_total_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    explicit_total_options.max_inflight_per_stream = 4;
    explicit_total_options.max_inflight_total = 10;
    verify_raw_frame_cap_blocks_after(explicit_total_options, 4, 10,
                                      "explicit global raw-frame cap above default");
    verify_per_stream_cap_blocks_after_two_raw_frames();
    verify_multiplexed_stream_ids_scale_default_total_cap();
    verify_asymmetric_fan_in_uses_strictest_caps();

    simaai::neat::GraphLinkOptions zero_options;
    zero_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    zero_options.max_inflight_per_stream = 0;
    bool rejected_zero = false;
    try {
      (void)run_one_raw_sample(zero_options);
    } catch (const std::runtime_error&) {
      rejected_zero = true;
    }
    require(rejected_zero, "max_inflight_per_stream=0 should be rejected");

    simaai::neat::GraphLinkOptions zero_total_options;
    zero_total_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    zero_total_options.max_inflight_total = 0;
    bool rejected_zero_total = false;
    try {
      (void)run_one_raw_sample(zero_total_options);
    } catch (const std::runtime_error&) {
      rejected_zero_total = true;
    }
    require(rejected_zero_total, "max_inflight_total=0 should be rejected");

    simaai::neat::GraphLinkOptions negative_options;
    negative_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    negative_options.max_inflight_per_stream = -2;
    bool rejected_negative = false;
    try {
      (void)run_one_raw_sample(negative_options);
    } catch (const std::runtime_error&) {
      rejected_negative = true;
    }
    require(rejected_negative, "max_inflight_per_stream<-1 should be rejected");

    simaai::neat::GraphLinkOptions negative_total_options;
    negative_total_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    negative_total_options.max_inflight_total = -2;
    bool rejected_negative_total = false;
    try {
      (void)run_one_raw_sample(negative_total_options);
    } catch (const std::runtime_error&) {
      rejected_negative_total = true;
    }
    require(rejected_negative_total, "max_inflight_total<-1 should be rejected");
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return 0;
}

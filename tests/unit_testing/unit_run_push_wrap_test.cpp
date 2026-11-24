// Zero-copy push on Core's public API (W2: agent-work/h0b/plan.md 7.3, D14-D16), on the process's
// backend (--backend=gstreamer|dataflow, set_backend). The same executable is registered once per
// backend.
//
// Both backends (what every push keeps):
//   - the bytes pulled are the bytes pushed: push and try_push, TensorList and Sample, the new and
//     the old pull;
//   - the pushed storage is released once every pulled tensor is gone: its TensorBuffer's
//     use_count back to the application's own reference, its holder's deleter run exactly once;
//   - a pushed tensor the application drops before the pull, and a pulled tensor that outlives
//     its Run, keep their bytes;
//   - a try_push the full link refuses keeps nothing of its tensor.
// Dataflow, W2's decisions on top (GStreamer copies its CPU input on Core's input thread; whether
// its pull shares the pushed memory is printed, not asserted):
//   - wrapped: the pulled tensor's storage is the pushed TensorBuffer and its first byte the
//     pushed one, the slot returned at the pull: a dense tensor at a fixed port (every verb), at a
//     nonzero aligned byte_offset, at a bounded port's bound (the runtime wraps exactly a slot),
//     NV12 at the port's pitches and size, every output of a fan-out, each input of a Run of two
//     (whose orig_input_seq, the Run's push count, the message itself carries), an owning cv::Mat;
//     held until
//     the last pulled tensor or queued message is gone, a KeepLatest eviction included; the push
//     reads none of its bytes;
//   - copied: an address or a byte_offset off the slot alignment (64 bytes), a strided view, a
//     frame below a bounded port's bound (in its own memory or in one a slot long), NV12 at other
//     pitches or below a bounded port's bound, a record, a
//     tensor pulled from a Dataflow Run whose message was copied (a lease on a slot), a Run::alloc
//     tensor through try_push (its blocking push still publishes its slot), copy_input, a storage
//     without a holder, a cv::Mat over memory it does not own, and OutputMemory::Owned at the
//     pull.

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/Result.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

using sima_test::on_dataflow;

RunOptions options(int queue_depth = 4, OverflowPolicy policy = OverflowPolicy::Block) {
  RunOptions opt;
  opt.queue_depth = queue_depth;
  opt.overflow_policy = policy;
  opt.startup_preflight = false;
  return opt;
}

InputOptions image_input(FormatTag format) {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = format;
  in.memory_policy = InputMemoryPolicy::SystemMemory;
  return in;
}

OutputOptions output_options(int max_buffers = 4) {
  OutputOptions out;
  out.max_buffers = max_buffers;
  out.drop = false;
  out.sync = false;
  return out;
}

Graph input_output(const InputOptions& in, const OutputOptions& out = output_options()) {
  Graph g;
  g.add(nodes::Input(in));
  g.add(nodes::Output(out));
  return g;
}

// Where a tensor's bytes start.
const std::byte* first_byte(const Tensor& t) {
  return static_cast<const std::byte*>(t.storage->data) + t.byte_offset;
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

// The releases of a buffer's owner (its deleter's runs).
using Releases = std::shared_ptr<std::atomic<int>>;

Releases releases() {
  return std::make_shared<std::atomic<int>>(0);
}

// `bytes` of zeroed memory aligned to 64, owned by `owner`, whose deleter counts in `count`.
struct Buffer {
  std::shared_ptr<void> owner;
  std::byte* base = nullptr;
};

Buffer aligned_buffer(std::size_t bytes, const Releases& count) {
  const std::size_t total = ((bytes + 63U) / 64U) * 64U;
  void* raw = std::aligned_alloc(64, total);
  require(raw != nullptr, "aligned_alloc");
  std::memset(raw, 0, total);
  Buffer b;
  b.owner = std::shared_ptr<void>(raw, [count](void* p) {
    std::free(p);
    count->fetch_add(1);
  });
  b.base = static_cast<std::byte*>(raw);
  return b;
}

std::vector<std::uint8_t> rgb_pattern(int w, int h, std::uint8_t tag) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3U);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>(tag + i * 7U);
  }
  return out;
}

// An RGB {h, w, 3} tensor over `storage` from `byte_offset`, its rows `row_bytes` apart (w * 3: a
// dense one; more: a strided view), holding rgb_pattern(w, h, tag).
Tensor rgb_over(std::shared_ptr<TensorBuffer> storage, int w, int h, std::int64_t byte_offset,
                std::int64_t row_bytes, std::uint8_t tag) {
  Tensor t;
  t.storage = std::move(storage);
  t.byte_offset = byte_offset;
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::HWC;
  t.shape = {h, w, 3};
  if (row_bytes != static_cast<std::int64_t>(w) * 3) {
    t.strides_bytes = {row_bytes, 3, 1};
  }
  t.device = {DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = ImageSpec{ImageSpec::PixelFormat::RGB, ""};
  const std::vector<std::uint8_t> pattern = rgb_pattern(w, h, tag);
  auto* base = static_cast<std::uint8_t*>(t.storage->data) + byte_offset;
  const std::size_t row = static_cast<std::size_t>(w) * 3U;
  for (int r = 0; r < h; ++r) {
    std::memcpy(base + static_cast<std::size_t>(r) * static_cast<std::size_t>(row_bytes),
                pattern.data() + static_cast<std::size_t>(r) * row, row);
  }
  return t;
}

// A dense RGB tensor whose storage is `pad` + its bytes of 64-aligned memory, its bytes at
// byte_offset `pad`; the storage's holder counts its release in `count`.
Tensor counted_rgb(int w, int h, std::uint8_t tag, const Releases& count, std::size_t pad = 0) {
  const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3U + pad;
  Buffer b = aligned_buffer(bytes, count);
  return rgb_over(make_cpu_external_storage(b.base, bytes, b.owner, false), w, h,
                  static_cast<std::int64_t>(pad), static_cast<std::int64_t>(w) * 3, tag);
}

bool eventually(const std::function<bool()>& done, std::chrono::milliseconds budget = 2000ms) {
  const auto end = Clock::now() + budget;
  while (!done()) {
    if (Clock::now() > end) {
      return false;
    }
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

// The one tensor of the next item of `output` ("" = the only one), through the new pull.
Tensor pull_new(Run& run, const std::string& output, const std::string& what) {
  const Result<TensorList> r = output.empty() ? run.pull(2000ms) : run.pull(output, 2000ms);
  require(r.ok(), what + ": pull status " + std::to_string(static_cast<int>(r.status())) + " " +
                      r.error().message);
  require(r->size() == 1U, what + ": one tensor expected");
  return r->front();
}

// The same through the old pull (a Sample).
Tensor pull_old(Run& run, const std::string& what) {
  Sample s;
  PullError err;
  const PullStatus st = run.pull(2000, s, &err);
  require(st == PullStatus::Ok,
          what + ": pull status " + std::to_string(static_cast<int>(st)) + " " + err.message);
  const TensorList tensors = tensors_from_sample(s);
  require(tensors.size() == 1U, what + ": one tensor expected");
  return tensors.front();
}

// Whether `pulled` is the pushed storage itself (Dataflow's wrap) or a copy: asserted on Dataflow,
// printed on GStreamer.
void expect_shared(const std::string& what, const Tensor& pushed, const Tensor& pulled,
                   bool shared) {
  const bool same = pulled.storage == pushed.storage && first_byte(pulled) == first_byte(pushed);
  const bool distinct =
      pulled.storage != pushed.storage && first_byte(pulled) != first_byte(pushed);
  if (on_dataflow()) {
    require(shared ? same : distinct,
            what + (shared ? ": the pulled tensor should be the pushed storage (wrapped)"
                           : ": the pulled tensor should be a copy"));
  } else {
    std::fprintf(stderr, "[INFO] %s (gstreamer): the pulled tensor %s the pushed storage\n",
                 what.c_str(), same ? "is" : "is not");
  }
}

// Every verb: push and try_push, TensorList and Sample, new and old pull. A wrapped push reads
// none of its tensor's bytes (a counting map_fn), keeps the storage (one reference) until the pull,
// and the pull hands it over and returns the slot.
void wrap_round_trips() {
  const Tensor seed = counted_rgb(64, 48, 0, releases());
  Run run = input_output(image_input(FormatTag::RGB)).build(TensorList{seed}, options());
  // A first push of the geometry runs Core's full push-time checks; the rows below repeat it.
  require(run.push(TensorList{counted_rgb(64, 48, 1, releases())}), "warm-up push");
  (void)pull_new(run, "", "warm-up");
  for (int verb = 0; verb < 8; ++verb) {
    const bool blocking = (verb & 1) == 0;
    const bool sample = (verb & 2) != 0;
    const bool old_pull = (verb & 4) != 0;
    const std::string what = std::string("wrap ") + (blocking ? "push" : "try_push") +
                             (sample ? "(Sample)" : "(TensorList)") +
                             (old_pull ? ", old pull" : ", new pull");
    const Releases count = releases();
    Tensor t = counted_rgb(64, 48, static_cast<std::uint8_t>(10 + verb), count);
    const auto maps = std::make_shared<std::atomic<int>>(0);
    t.storage->map_fn = [maps, data = t.storage->data, size = t.storage->size_bytes](MapMode) {
      maps->fetch_add(1);
      Mapping m;
      m.data = data;
      m.size_bytes = size;
      return m;
    };
    bool accepted = false;
    if (sample) {
      Sample s;
      s.kind = SampleKind::Tensor;
      s.tensor = t;
      accepted = blocking ? run.push(s) : run.try_push(s);
    } else {
      accepted = blocking ? run.push(TensorList{t}) : run.try_push(TensorList{t});
    }
    require(accepted, what + ": accepted");
    if (on_dataflow()) {
      require(maps->load() == 0, what + ": a wrapped push reads none of the tensor's bytes");
      require(t.storage.use_count() == 2, what + ": the queued message keeps one reference, got " +
                                              std::to_string(t.storage.use_count()));
    }
    {
      const Tensor out = old_pull ? pull_old(run, what) : pull_new(run, "", what);
      require(bytes_of(out) == rgb_pattern(64, 48, static_cast<std::uint8_t>(10 + verb)),
              what + ": bytes");
      expect_shared(what, t, out, true);
      if (on_dataflow()) {
        require(out.read_only, what + ": a pulled tensor is read-only");
        require(t.storage.use_count() == 2,
                what + ": the pulled tensor holds the storage and the slot is back, got " +
                    std::to_string(t.storage.use_count()));
      }
    }
    require(eventually([&] { return t.storage.use_count() == 1; }),
            what + ": released once the pulled tensor is gone");
    t = Tensor{};
    require(count->load() == 1, what + ": the holder released exactly once");
  }
  run.close();
}

// The application drops its tensor before the pull; then the pulled tensor outlives its Run.
void lifetimes() {
  for (const bool old_pull : {false, true}) {
    const std::string what = std::string("lifetime, ") + (old_pull ? "old pull" : "new pull");
    const Releases count = releases();
    Tensor kept;
    {
      Run run = input_output(image_input(FormatTag::RGB))
                    .build(TensorList{counted_rgb(32, 24, 0, releases())}, options());
      {
        Tensor t = counted_rgb(32, 24, 0x41, count);
        require(run.push(TensorList{t}), what + ": push");
      } // the application's tensor is gone; only the Run holds the storage
      if (on_dataflow()) {
        require(count->load() == 0, what + ": the queued message keeps the storage");
      }
      kept = old_pull ? pull_old(run, what) : pull_new(run, "", what);
      run.close();
    } // the Run is destroyed
    require(bytes_of(kept) == rgb_pattern(32, 24, 0x41),
            what + ": the pulled tensor outlives its Run");
    if (on_dataflow()) {
      require(count->load() == 0, what + ": the pulled tensor keeps the storage past its Run");
    }
    kept = Tensor{};
    require(eventually([&] { return count->load() == 1; }),
            what + ": released exactly once after the last pulled tensor");
    std::this_thread::sleep_for(5ms);
    require(count->load() == 1, what + ": released exactly once");
  }
}

// One input given to two outputs: every output has the pushed storage; released after both.
void fan_out() {
  Graph g;
  g.add(nodes::Input("image", image_input(FormatTag::RGB)));
  g.add(nodes::Output("a", output_options()));
  g.add(nodes::Output("b", output_options()));
  g.connect("image", "a");
  g.connect("image", "b");
  Run run = g.build(options());
  for (int round = 0; round < 2; ++round) {
    const std::string what = "fan-out round " + std::to_string(round);
    const Releases count = releases();
    Tensor t = counted_rgb(32, 24, static_cast<std::uint8_t>(0x50 + round), count);
    Tensor pushed = t;
    require(run.push("image", TensorList{t}), what + ": push");
    t = Tensor{};
    // Round 0 pulls through the new verbs, round 1 through the Sample pull (as
    // graph_migration_phaseA3_branch_helper_test does), each output in turn first.
    const auto pull = [&](const std::string& output, const std::string& label) {
      if (round == 0) {
        return pull_new(run, output, label);
      }
      Sample s;
      PullError err;
      require(run.pull(output, 2000, s, &err) == PullStatus::Ok, label + ": " + err.message);
      const TensorList tensors = tensors_from_sample(s);
      require(tensors.size() == 1U, label + ": one tensor");
      return tensors.front();
    };
    Tensor a = pull(round == 0 ? "a" : "b", what + " first output");
    Tensor b = pull(round == 0 ? "b" : "a", what + " second output");
    const auto expected = rgb_pattern(32, 24, static_cast<std::uint8_t>(0x50 + round));
    require(bytes_of(a) == expected && bytes_of(b) == expected, what + ": bytes on both outputs");
    expect_shared(what + " first output", pushed, a, true);
    expect_shared(what + " second output", pushed, b, true);
    if (on_dataflow()) {
      require(a.storage == b.storage, what + ": both outputs lease the same wrap");
    }
    const std::weak_ptr<TensorBuffer> watch = pushed.storage;
    pushed = Tensor{};
    a = Tensor{};
    if (on_dataflow()) {
      require(count->load() == 0 && !watch.expired(), what + ": the second output still holds it");
    }
    b = Tensor{};
    require(eventually([&] { return count->load() == 1; }),
            what + ": released exactly once after both outputs");
  }
  run.close();
}

// KeepLatest on Dataflow: the evicted message's wrap is released by the push that evicted it, on
// the pushing thread, before the push returns.
void keep_latest_eviction() {
  if (!on_dataflow()) {
    std::fprintf(stderr,
                 "[INFO] keep_latest_eviction: Dataflow's link rule, not run on GStreamer\n");
    return;
  }
  Run run = input_output(image_input(FormatTag::RGB), output_options(2))
                .build(TensorList{counted_rgb(32, 24, 0, releases())},
                       options(1, OverflowPolicy::KeepLatest));
  std::vector<Releases> counts;
  for (int i = 0; i < 3; ++i) { // the link holds max(queue_depth 1, max_buffers 2) = 2
    counts.push_back(releases());
    require(run.try_push(TensorList{
                counted_rgb(32, 24, static_cast<std::uint8_t>(0x60 + i), counts.back())}),
            "KeepLatest never refuses");
  }
  require(counts[0]->load() == 1, "the evicted message's storage is released by the evicting push");
  require(counts[1]->load() == 0 && counts[2]->load() == 0, "the queued ones are kept");
  for (int i = 1; i < 3; ++i) {
    require(bytes_of(pull_new(run, "", "KeepLatest")) ==
                rgb_pattern(32, 24, static_cast<std::uint8_t>(0x60 + i)),
            "the kept frames, in order");
  }
  require(counts[1]->load() == 1 && counts[2]->load() == 1,
          "released at their pulled tensors' drop");
  run.close();
}

// A try_push the full link refuses keeps nothing: the caller's storage is its own when it returns.
void refused_try_push_keeps_nothing() {
  for (const OverflowPolicy policy : {OverflowPolicy::DropIncoming, OverflowPolicy::Block}) {
    const std::string what = std::string("refused try_push, ") +
                             (policy == OverflowPolicy::DropIncoming ? "DropIncoming" : "Block");
    Run run = input_output(image_input(FormatTag::RGB), output_options(1))
                  .build(TensorList{counted_rgb(32, 24, 0, releases())}, options(1, policy));
    std::vector<Releases> accepted;
    bool refused = false;
    for (int i = 0; i < 4096 && !refused; ++i) {
      const Releases count = releases();
      Tensor t = counted_rgb(32, 24, static_cast<std::uint8_t>(i), count);
      if (run.try_push(TensorList{t})) {
        accepted.push_back(count);
        continue;
      }
      refused = true;
      require(t.storage.use_count() == 1,
              what + ": nothing kept, use_count " + std::to_string(t.storage.use_count()));
      t = Tensor{};
      require(count->load() == 1, what + ": the caller's drop releases it at once");
    }
    require(refused, what + ": the link fills");
    if (on_dataflow()) {
      for (std::size_t i = 0; i < accepted.size(); ++i) {
        (void)pull_new(run, "", what + " drain");
      }
    } else {
      // GStreamer's queues: drain whatever arrives.
      for (;;) {
        const Result<TensorList> r = run.pull(300ms);
        if (!r.ok()) {
          break;
        }
      }
    }
    require(eventually([&] {
              for (const Releases& c : accepted) {
                if (c->load() != 1) {
                  return false;
                }
              }
              return true;
            }),
            what + ": every accepted storage released after the drain");
    run.close();
  }
}

// One push and pull of `t` into `run`: the bytes, and whether the pull shares the pushed storage.
void round_trip(Run& run, const Tensor& t, bool shared, const std::string& what,
                const std::string& input = "", const std::string& output = "") {
  const std::vector<std::uint8_t> expected = bytes_of(t);
  require(input.empty() ? run.push(TensorList{t}) : run.push(input, TensorList{t}),
          what + ": push");
  const Tensor out = pull_new(run, output, what);
  require(bytes_of(out) == expected, what + ": bytes");
  expect_shared(what, t, out, shared);
}

// Dense tensors, fixed and bounded ports.
void dense_cases() {
  {
    // A plain tensor input (an image's byte_offset must be 0, Core's own rule for packed video):
    // the byte_offset cases.
    const auto plain = [](Tensor t) {
      t.semantic.image.reset();
      t.layout = TensorLayout::Unknown;
      return t;
    };
    Run run = input_output(InputOptions{})
                  .build(TensorList{plain(counted_rgb(64, 48, 0, releases()))}, options());
    round_trip(run, plain(counted_rgb(64, 48, 1, releases(), 64)), true,
               "byte_offset 64 (aligned): wrapped");
    round_trip(run, plain(counted_rgb(64, 48, 2, releases(), 8)), false, "byte_offset 8: copied");
    {
      // The storage's own first byte aligned, the tensor's 32 bytes in: copied.
      const std::size_t bytes = 64U * 48U * 3U;
      Buffer b = aligned_buffer(bytes + 64U, releases());
      round_trip(run,
                 plain(rgb_over(make_cpu_external_storage(b.base, bytes + 64U, b.owner, false), 64,
                                48, 32, 64 * 3, 4)),
                 false, "byte_offset 32: copied");
    }
    run.close();
  }
  {
    Run run = input_output(image_input(FormatTag::RGB))
                  .build(TensorList{counted_rgb(64, 48, 0, releases())}, options());
    {
      const std::size_t bytes = 64U * 48U * 3U;
      Buffer b = aligned_buffer(bytes + 64U, releases());
      round_trip(run,
                 rgb_over(make_cpu_external_storage(b.base + 8, bytes, b.owner, false), 64, 48, 0,
                          64 * 3, 3),
                 false, "an address 8 bytes off: copied");
    }
    {
      const std::size_t row = 64U * 3U + 64U;
      Buffer b = aligned_buffer(row * 48U, releases());
      round_trip(run,
                 rgb_over(make_cpu_external_storage(b.base, row * 48U, b.owner, false), 64, 48, 0,
                          static_cast<std::int64_t>(row), 5),
                 false, "a strided view (padded rows): copied");
    }
    {
      // No holder: memory nobody owns cannot be kept alive. The test keeps it until the pull.
      const std::size_t bytes = 64U * 48U * 3U;
      Buffer b = aligned_buffer(bytes, releases());
      round_trip(
          run, rgb_over(make_cpu_external_storage(b.base, bytes, {}, false), 64, 48, 0, 64 * 3, 6),
          false, "a storage without a holder: copied");
    }
    run.close();
  }
  {
    // A bounded port (max 96 x 96): a slot holds 96 x 96 x 3 bytes, and the runtime takes a wrap
    // only of exactly that size, so only a frame at the bound is wrapped. Realtime: GStreamer's
    // other presets hold a frame of a new shape until two frames agree (stability_frames).
    InputOptions in = image_input(FormatTag::RGB);
    in.max_width = 96;
    in.max_height = 96;
    RunOptions opt = options();
    opt.preset = RunPreset::Realtime;
    Run run = input_output(in).build(TensorList{counted_rgb(64, 48, 0, releases())}, opt);
    round_trip(run, counted_rgb(64, 48, 7, releases()), false,
               "bounded port, a frame below the bound: copied");
    round_trip(run, counted_rgb(96, 96, 8, releases()), true,
               "bounded port, a frame at the bound: wrapped");
    {
      const std::size_t slot = 96U * 96U * 3U;
      Buffer b = aligned_buffer(slot, releases());
      round_trip(
          run,
          rgb_over(make_cpu_external_storage(b.base, slot, b.owner, false), 64, 48, 0, 64 * 3, 9),
          false, "bounded port, a frame below the bound in a storage a slot long: copied");
    }
    round_trip(run, counted_rgb(96, 96, 10, releases()), true,
               "bounded port, a frame at the bound again: wrapped");
    run.close();
  }
}

// An NV12 {h, w} frame in `storage_rows` rows of `pitch` bytes (the UV plane after them), its
// bytes a pattern of `tag`.
Tensor nv12_frame(int w, int h, std::int64_t pitch, std::int64_t storage_rows, std::uint8_t tag) {
  const std::size_t bytes =
      static_cast<std::size_t>(pitch) * static_cast<std::size_t>(storage_rows) * 3U / 2U;
  Buffer b = aligned_buffer(bytes, releases());
  auto* p = reinterpret_cast<std::uint8_t*>(b.base);
  for (std::size_t i = 0; i < bytes; ++i) {
    p[i] = static_cast<std::uint8_t>(i * 13U + tag);
  }
  Tensor t;
  t.storage = make_cpu_external_storage(b.base, bytes, b.owner, false);
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::HW;
  t.shape = {h, w};
  t.device = {DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = ImageSpec{ImageSpec::PixelFormat::NV12, ""};
  Plane y;
  y.role = PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {pitch, 1};
  Plane uv;
  uv.role = PlaneRole::UV;
  uv.shape = {h / 2, w};
  uv.strides_bytes = {pitch, 1};
  uv.byte_offset = pitch * storage_rows;
  t.planes = {y, uv};
  return t;
}

void nv12_round_trip(Run& run, const Tensor& t, bool shared, const std::string& what) {
  require(run.push(TensorList{t}), what + ": push");
  const Tensor out = pull_new(run, "", what);
  require(out.is_nv12() && out.copy_nv12_contiguous() == t.copy_nv12_contiguous(),
          what + ": the planes' bytes");
  expect_shared(what, t, out, shared);
}

// NV12: a frame at the port's pitches and size is wrapped; at other pitches, or below a bounded
// port's bound (even at the bound's pitches, in a storage a slot long), it is copied.
void nv12_cases() {
  {
    InputOptions in = image_input(FormatTag::NV12);
    in.width = 64;
    in.height = 48;
    RunOptions opt = options();
    opt.advanced.max_input_bytes = 1U << 16; // GStreamer caps a declared input at its frame size
    Run run = input_output(in).build(TensorList{make_nv12_tensor(64, 48, 0x22)}, opt);
    nv12_round_trip(run, nv12_frame(64, 48, 64, 48, 0), true,
                    "NV12 at the port's pitches: wrapped");
    nv12_round_trip(run, nv12_frame(64, 48, 128, 48, 1), false, "NV12 at a 128-byte pitch: copied");
    run.close();
  }
  {
    InputOptions in = image_input(FormatTag::NV12);
    in.max_width = 64;
    in.max_height = 48;
    RunOptions opt = options();
    opt.preset = RunPreset::Realtime; // GStreamer renegotiates a new shape at once
    opt.advanced.max_input_bytes = 1U << 16;
    Run run = input_output(in).build(TensorList{make_nv12_tensor(64, 48, 0x22)}, opt);
    nv12_round_trip(run, nv12_frame(64, 48, 64, 48, 2), true,
                    "NV12 at a bounded port's bound: wrapped");
    nv12_round_trip(run, nv12_frame(32, 24, 32, 24, 3), false,
                    "NV12 below a bounded port's bound: copied");
    nv12_round_trip(run, nv12_frame(32, 24, 64, 48, 4), false,
                    "NV12 below the bound at the bound's pitches, a slot long: copied");
    nv12_round_trip(run, nv12_frame(64, 48, 64, 48, 5), true,
                    "NV12 at a bounded port's bound again: wrapped");
    run.close();
  }
}

// A record: copied.
void record() {
  {
    // Two FP32 tensors of one shape, the record GStreamer's tensor media carries (as
    // unit_run_tensor_surface_test's positional records).
    InputOptions in;
    in.payload_type = PayloadType::Tensor;
    in.format = FormatTag::FP32;
    in.memory_policy = InputMemoryPolicy::SystemMemory;
    const auto floats = [](float start) {
      std::vector<float> v(16);
      for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = start + static_cast<float>(i);
      }
      return Tensor::from_vector(v, {16}, TensorMemory::CPU);
    };
    Graph g;
    g.add(nodes::Input(in));
    g.add(nodes::Output());
    Run run = g.build(TensorList{floats(0.0F), floats(100.0F)}, options());
    const Tensor a = floats(1.0F);
    const Tensor b = floats(50.0F);
    require(run.push(TensorList{a, b}), "record push");
    const Result<TensorList> out = run.pull(2000ms);
    require(out.ok() && out->size() == 2U, "record: two fields");
    require(bytes_of((*out)[0]) == bytes_of(a) && bytes_of((*out)[1]) == bytes_of(b),
            "record: bytes");
    expect_shared("record field 0: copied", a, (*out)[0], false);
    expect_shared("record field 1: copied", b, (*out)[1], false);
    run.close();
  }
}

// A Run of two inputs wraps as a Run of one does: on each input the pulled tensor is the pushed
// storage itself, the push reads none of its bytes and keeps one reference until the pull, and
// the storage is released exactly once. Sample::orig_input_seq is the Run's push count across
// both inputs, which the message carries.
void two_inputs() {
  Graph g;
  g.add(nodes::Input("left", image_input(FormatTag::RGB)));
  g.add(nodes::Output("left_out", output_options()));
  g.add(nodes::Input("right", image_input(FormatTag::RGB)));
  g.add(nodes::Output("right_out", output_options()));
  g.connect("left", "left_out");
  g.connect("right", "right_out");
  Run run = g.build(options());
  const std::string inputs[2] = {"left", "right"};
  const std::string outputs[2] = {"left_out", "right_out"};
  for (int i = 0; i < 6; ++i) {
    const std::string& input = inputs[i % 2];
    const std::string what = "two inputs, push " + std::to_string(i) + " into " + input;
    const Releases count = releases();
    Tensor t = counted_rgb(32, 24, static_cast<std::uint8_t>(0x70 + i), count);
    const auto maps = std::make_shared<std::atomic<int>>(0);
    t.storage->map_fn = [maps, data = t.storage->data, size = t.storage->size_bytes](MapMode) {
      maps->fetch_add(1);
      Mapping m;
      m.data = data;
      m.size_bytes = size;
      return m;
    };
    require(run.push(input, TensorList{t}), what + ": push");
    // Each input's first push of the geometry runs Core's full push-time checks.
    if (on_dataflow() && i >= 2) {
      require(maps->load() == 0, what + ": a wrapped push reads none of the tensor's bytes");
      require(t.storage.use_count() == 2, what + ": the queued message keeps one reference, got " +
                                              std::to_string(t.storage.use_count()));
    }
    Sample s;
    PullError err;
    require(run.pull(outputs[i % 2], 2000, s, &err) == PullStatus::Ok, what + ": " + err.message);
    if (on_dataflow()) {
      require(s.orig_input_seq == i, what + ": orig_input_seq is the Run's push count, got " +
                                         std::to_string(s.orig_input_seq));
    } else {
      std::fprintf(stderr, "[INFO] %s (gstreamer): orig_input_seq %lld\n", what.c_str(),
                   static_cast<long long>(s.orig_input_seq));
    }
    TensorList tensors = tensors_from_sample(s);
    require(tensors.size() == 1U, what + ": one tensor");
    require(bytes_of(tensors.front()) == rgb_pattern(32, 24, static_cast<std::uint8_t>(0x70 + i)),
            what + ": bytes");
    expect_shared(what, t, tensors.front(), true);
    t = Tensor{};
    s = Sample{};
    if (on_dataflow()) {
      require(count->load() == 0, what + ": the pulled tensor holds the pushed storage");
    }
    tensors.clear();
    require(eventually([&] { return count->load() == 1; }), what + ": released exactly once");
  }
  run.close();
}

// A tensor pulled from another Run, Run::alloc tensors, copy_input, Owned output.
void origin_and_options() {
  const Tensor seed = counted_rgb(32, 24, 0, releases());
  // Pulled from a Run whose message was copied (copy_input): a lease on that Run's slot. Dataflow's
  // D15 rule (a GStreamer-pulled tensor has GStreamer's own cross-Run rules, RunPush.cpp:39-50).
  if (!on_dataflow()) {
    std::fprintf(stderr, "[INFO] cross-Run pushes: Dataflow's rule, not run on GStreamer\n");
  } else {
    RunOptions copying = options();
    copying.advanced.copy_input = true;
    Run first = input_output(image_input(FormatTag::RGB)).build(TensorList{seed}, copying);
    Run second = input_output(image_input(FormatTag::RGB)).build(TensorList{seed}, options());
    const Tensor t = counted_rgb(32, 24, 0x81, releases());
    require(first.push(TensorList{t}), "copy_input push");
    const Tensor leased = pull_new(first, "", "copy_input");
    require(bytes_of(leased) == bytes_of(t), "copy_input: bytes");
    expect_shared("copy_input: copied", t, leased, false);
    if (on_dataflow()) {
      require(t.storage.use_count() == 1, "copy_input: the push kept nothing");
    }
    round_trip(second, leased, false, "a tensor pulled from another Run (a slot lease): copied");
    // Pulled from a Run that wrapped it: the pushed storage itself (D14), which another Run wraps
    // again; no slot of the first Run is held.
    Run third = input_output(image_input(FormatTag::RGB)).build(TensorList{seed}, options());
    const Tensor u = counted_rgb(32, 24, 0x82, releases());
    require(third.push(TensorList{u}), "push into the third Run");
    const Tensor wrapped = pull_new(third, "", "third Run");
    expect_shared("the third Run's pull: wrapped", u, wrapped, true);
    round_trip(second, wrapped, true,
               "a pulled wrapped tensor (the application's storage) pushed again: wrapped");
    first.close();
    second.close();
    third.close();
  }
  // Run::alloc: try_push reads it like any CPU tensor (a copy); the blocking push alone in a list
  // publishes its slot, as before W2.
  {
    Graph g;
    const auto x = g.input("x", {4, 8}, TensorDType::Float32);
    g.output("y", x);
    Run run = g.build(options());
    Tensor t = run.alloc("x");
    {
      Mapping map = t.map_write();
      auto* f = static_cast<float*>(map.data);
      for (int i = 0; i < 32; ++i) {
        f[i] = static_cast<float>(i) + 0.5F;
      }
    }
    const std::vector<std::uint8_t> expected = bytes_of(t);
    const std::byte* slot = first_byte(t);
    require(run.try_push("x", TensorList{t}), "alloc'd try_push");
    const Tensor copied = pull_new(run, "y", "alloc'd try_push");
    require(bytes_of(copied) == expected, "alloc'd try_push: bytes");
    if (on_dataflow()) {
      require(first_byte(copied) != slot, "alloc'd try_push: copied");
    }
    require(run.push("x", TensorList{t}), "alloc'd push: the commit");
    const Tensor committed = pull_new(run, "y", "alloc'd push");
    require(bytes_of(committed) == expected, "alloc'd push: bytes");
    if (on_dataflow()) {
      require(first_byte(committed) == slot, "alloc'd push: the slot itself, published");
    }
    run.close();
  }
  // OutputMemory::Owned: the pull copies the wrapped bytes; the storage is released at the pull.
  {
    RunOptions owned = options();
    owned.output_memory = OutputMemory::Owned;
    Run run = input_output(image_input(FormatTag::RGB)).build(TensorList{seed}, owned);
    const Tensor t = counted_rgb(32, 24, 0x83, releases());
    require(run.push(TensorList{t}), "Owned: push");
    const Tensor out = pull_new(run, "", "Owned");
    require(bytes_of(out) == bytes_of(t), "Owned: bytes");
    expect_shared("Owned output: copied at the pull", t, out, false);
    if (on_dataflow()) {
      require(out.storage->kind == StorageKind::CpuOwned, "Owned: a CPU copy");
      require(t.storage.use_count() == 1, "Owned: the slot and its wrap released at the pull");
    }
    run.close();
  }
}

// cv::Mat pushes (the deprecated forms): on GStreamer an owning Mat and a Mat over memory it does
// not own round trip; on Dataflow the deprecated form fails at the call, before anything is kept.
void mat_cases() {
  const cv::Mat seed(24, 32, CV_8UC3, cv::Scalar(1, 2, 3));
  Run run = input_output(image_input(FormatTag::BGR))
                .build(TensorList{Tensor::from_cv_mat(seed, ImageSpec::PixelFormat::BGR,
                                                      TensorMemory::CPU)},
                       options());
  cv::Mat owning(24, 32, CV_8UC3, cv::Scalar(9, 8, 7));
  Buffer b = aligned_buffer(24U * 32U * 3U, releases());
  cv::Mat borrowed(24, 32, CV_8UC3, b.base);
  borrowed.setTo(cv::Scalar(4, 5, 6));
  for (const cv::Mat* mat : {&owning, &borrowed}) {
    const bool owns = mat == &owning;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(mat->data) % 64U) == 0U;
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    if (on_dataflow()) {
      require_neat_error([&] { (void)run.push(std::vector<cv::Mat>{*mat}); },
                         error_codes::kPipelineShape,
                         "Run::push(const std::vector<cv::Mat>&) is deprecated and is not carried "
                         "by the Dataflow backend");
      continue;
    }
    require(run.push(std::vector<cv::Mat>{*mat}), "cv::Mat push");
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    const Tensor out = pull_new(run, "", "cv::Mat");
    const std::vector<std::uint8_t> expected(mat->data, mat->data + 24U * 32U * 3U);
    require(bytes_of(out) == expected, "cv::Mat: bytes");
    const bool same = first_byte(out) == reinterpret_cast<const std::byte*>(mat->data);
    std::fprintf(stderr, "[INFO] cv::Mat %s (aligned %d): the pulled tensor %s the Mat's memory\n",
                 owns ? "owning" : "borrowed", aligned ? 1 : 0, same ? "is" : "is not");
  }
  run.close();
}

} // namespace

RUN_BACKEND_TEST("unit_run_push_wrap_test", ([] {
                   std::fprintf(stderr, "[INFO] backend %s\n",
                                on_dataflow() ? "dataflow" : "gstreamer");
                   wrap_round_trips();
                   lifetimes();
                   fan_out();
                   keep_latest_eviction();
                   refused_try_push_keeps_nothing();
                   dense_cases();
                   nv12_cases();
                   record();
                   two_inputs();
                   origin_and_options();
                   mat_cases();
                 }));

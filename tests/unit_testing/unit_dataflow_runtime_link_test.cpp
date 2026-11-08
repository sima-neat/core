// The Neat Dataflow runtime as Core embeds it (SIMANEAT_DATAFLOW_SOURCE_DIR): the smallest Net
// through the runtime objects Core's libraries carry. One Feed, one Fetch, one link; one alloc and
// push, one pull; then close_input, which drains the link, and stop.
//
// libsima_neat.so keeps the runtime's symbols hidden, so this test takes them from
// libsima_neat.a, which holds the same objects. tests/CMakeLists.txt links no other copy.

#include "dataflow/dataflow.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace df = simaai::dataflow;

namespace {

constexpr std::uint32_t kElements = 16;
constexpr std::int64_t kFrameId = 42;
constexpr df::StreamId kStream{7};

// Every call has a deadline, so a lost wake fails the test instead of hanging it.
df::Deadline watchdog() {
  return std::chrono::steady_clock::now() + std::chrono::seconds(10);
}

template <class T> T must(df::Result<T>&& result, const char* what) {
  if (const df::Error* error = df::failure(result)) {
    throw std::runtime_error(std::string(what) + ": " + error->message);
  }
  return std::get<T>(std::move(result));
}

void must_ok(const df::Status& status, const char* what) {
  if (status) {
    throw std::runtime_error(std::string(what) + ": " + status->message);
  }
}

float pattern(std::uint32_t i) {
  return 0.5f + static_cast<float>(i);
}

} // namespace

RUN_TEST("unit_dataflow_runtime_link_test", ([] {
           // f32 Dense [16]: a fixed tensor port, 64 bytes a slot.
           const df::PortSpec spec{.payload = df::TensorSpec{.dtype = df::DType::Float32,
                                                             .shape = df::Shape::of({kElements}),
                                                             .layout = df::Dense{}}};

           df::Net net;
           const df::OperatorRef<df::Feed> feed =
               net.add<df::Feed>("in", spec, df::Feed::Options{.stream = kStream});
           const df::OperatorRef<df::Fetch> fetch =
               net.add<df::Fetch>("out", spec, df::Fetch::Options{});
           must_ok(net.connect(feed.out(), fetch.in(), df::LinkPolicy{.depth = 1}), "connect");

           df::Plan plan = must(std::move(net).compile(), "compile");
           // Core finds its named endpoints this way.
           require(must(plan.feed("in"), "plan.feed").op == feed.index, "plan.feed(\"in\")");
           require(must(plan.fetch("out"), "plan.fetch").op == fetch.index, "plan.fetch(\"out\")");

           df::Session session = must(std::move(plan).instantiate(), "instantiate");
           must_ok(session.start(), "start");

           df::WriteLoan loan = must(session.alloc(feed, df::Acquire::Block, watchdog()), "alloc");
           require(loan.bytes().size() == kElements * sizeof(float), "slot size");
           for (std::uint32_t i = 0; i < kElements; ++i) {
             const float v = pattern(i);
             std::memcpy(loan.bytes().data() + i * sizeof(float), &v, sizeof(float));
           }
           const df::Stamp stamp = session.stamp(feed, {.source_time = {}, .frame_id = kFrameId});
           const df::PublishResult pushed = must(
               session.push(feed, std::move(loan), stamp, df::Extent::of({kElements}), watchdog()),
               "push");
           require(pushed.published() && pushed.size() == 1 && pushed.accepted() == 1,
                   "push: published on the one link");

           df::Message message = must(session.pull(fetch, watchdog()), "pull");
           const std::span<const std::byte> bytes = message.bytes();
           require(bytes.size() == kElements * sizeof(float), "pull: byte count");
           for (std::uint32_t i = 0; i < kElements; ++i) {
             float v = 0.0f;
             std::memcpy(&v, bytes.data() + i * sizeof(float), sizeof(float));
             require(v == pattern(i), "pull: bytes differ at element " + std::to_string(i));
           }
           require(message.stamp().stream == kStream, "pull: stream");
           require(message.stamp().frame_id == kFrameId, "pull: frame_id");
           require(message.stamp().seq == 1, "pull: seq");
           message = df::Message{};

           must_ok(session.close_input(feed, watchdog()), "close_input");
           df::Result<df::Message> drained = session.pull(fetch, watchdog());
           const df::Error* end = df::failure(drained);
           require(end != nullptr && end->code == df::ErrorCode::Closed,
                   "pull after close_input: closed and drained");

           must_ok(session.stop(watchdog()), "stop");
           const df::SessionReport report = session.report();
           require(!report.first_error && !report.retained_owner, "report: clean stop");
           require(report.feeds.size() == 1 && report.feeds[0].pushes == 1, "report: one push");
           require(report.dropped_frames == 0, "report: no drops");
         }));

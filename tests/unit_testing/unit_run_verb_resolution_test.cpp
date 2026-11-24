// Run's overload set with the TensorList verbs (H0 C2b: Core's Tensor surface is TensorList-only;
// decision D5 for the pull timeouts): which overload each call form takes, pinned at compile time
// against the real header, then observed at run time.
//
// Compile time: ResolutionProbe derives from Run, brings every Run::push and Run::pull overload in
// with using-declarations, and redeclares the overloads under test with the same parameter lists
// and tagged return types. A member declared in the derived class hides the base member with the
// same parameter-type-list, so overload resolution over the probe is resolution over Run's real
// set, with only the resolved overload's identity made visible in the result type. Each redeclared
// signature is first checked to exist in Run exactly (static_cast of the member pointer). No
// single-Tensor verb exists: push(t), try_push(t) and run(t) with a bare Tensor do not compile and
// on_output takes no Tensor callback (requires-expressions, false because the call is ill-formed,
// each beside a control that is true), and push({t}) is the TensorList overload.
//
// Run time (on the backend --backend=gstreamer|dataflow selects, set_backend): push({t}),
// push(TensorList{t}) and push("in", {t}) are the same list verb and each carries the first
// tensor's stamp; push({t, u}) pushes one item of two tensors; the int pulls return
// std::optional<Sample>, the chrono pulls Result<TensorList>.

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;

template <int N> using Tag = std::integral_constant<int, N>;

// The signatures under test, as Run declares them.
[[maybe_unused]] constexpr auto kPushList =
    static_cast<bool (Run::*)(const TensorList&)>(&Run::push);
[[maybe_unused]] constexpr auto kPushNamedList =
    static_cast<bool (Run::*)(std::string_view, const TensorList&)>(&Run::push);
[[maybe_unused]] constexpr auto kPullInt =
    static_cast<std::optional<Sample> (Run::*)(int)>(&Run::pull);
[[maybe_unused]] constexpr auto kPullNamedInt =
    static_cast<std::optional<Sample> (Run::*)(std::string_view, int)>(&Run::pull);
[[maybe_unused]] constexpr auto kPullMs =
    static_cast<Result<TensorList> (Run::*)(std::chrono::milliseconds)>(&Run::pull);
[[maybe_unused]] constexpr auto kPullNamedMs =
    static_cast<Result<TensorList> (Run::*)(std::string_view, std::chrono::milliseconds)>(
        &Run::pull);
[[maybe_unused]] constexpr auto kCall =
    static_cast<TensorList (Run::*)(const TensorList&)>(&Run::operator());
[[maybe_unused]] constexpr auto kOnOutput =
    static_cast<void (Run::*)(std::function<void(TensorList)>)>(&Run::on_output);
[[maybe_unused]] constexpr auto kOnOutputNamed =
    static_cast<void (Run::*)(std::string_view, std::function<void(TensorList)>)>(&Run::on_output);
[[maybe_unused]] constexpr auto kAlloc =
    static_cast<Tensor (Run::*)(std::string_view)>(&Run::alloc);

struct ResolutionProbe : Run {
  using Run::pull;
  using Run::push;
  // Named only in unevaluated operands; the bodies exist so nothing is declared without one.
  Tag<3> push(const TensorList&) {
    return {};
  }
  Tag<4> push(std::string_view, const TensorList&) {
    return {};
  }
  Tag<5> pull(int = -1) {
    return {};
  }
  Tag<6> pull(std::string_view, int = -1) {
    return {};
  }
  Tag<7> pull(std::chrono::milliseconds) {
    return {};
  }
  Tag<8> pull(std::string_view, std::chrono::milliseconds) {
    return {};
  }
};

template <class Expr, int N> constexpr bool resolves_to = std::is_same_v<Expr, Tag<N>>;

static_assert(
    resolves_to<decltype(std::declval<ResolutionProbe&>().push({std::declval<const Tensor&>()})),
                3>,
    "push({t}) -> push(const TensorList&)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().push(
                              {std::declval<const Tensor&>(), std::declval<const Tensor&>()})),
                          3>,
              "push({t, u}) -> push(const TensorList&)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().push(TensorList{
                              std::declval<const Tensor&>()})),
                          3>,
              "push(TensorList{t}) -> push(const TensorList&)");
static_assert(
    resolves_to<
        decltype(std::declval<ResolutionProbe&>().push("x", {std::declval<const Tensor&>()})), 4>,
    "push(\"x\", {t}) -> push(string_view, const TensorList&)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().push(
                              "x", {std::declval<const Tensor&>(), std::declval<const Tensor&>()})),
                          4>,
              "push(\"x\", {t, u}) -> push(string_view, const TensorList&)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().pull()), 5>,
              "pull() -> pull(int)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().pull(100)), 5>,
              "pull(100) -> pull(int)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().pull("x")), 6>,
              "pull(\"x\") -> pull(string_view, int)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().pull("x", 100)), 6>,
              "pull(\"x\", 100) -> pull(string_view, int)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().pull(100ms)), 7>,
              "pull(100ms) -> pull(milliseconds)");
static_assert(
    resolves_to<decltype(std::declval<ResolutionProbe&>().pull(std::chrono::seconds(1))), 7>,
    "pull(1s) -> pull(milliseconds)");
static_assert(resolves_to<decltype(std::declval<ResolutionProbe&>().pull("x", 100ms)), 8>,
              "pull(\"x\", 100ms) -> pull(string_view, milliseconds)");

// And on Run itself: every form is unambiguous, with these result types.
static_assert(
    std::is_same_v<decltype(std::declval<Run&>().push({std::declval<const Tensor&>()})), bool>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().push({std::declval<const Tensor&>(),
                                                                 std::declval<const Tensor&>()})),
                             bool>);
static_assert(
    std::is_same_v<decltype(std::declval<Run&>().push(TensorList{std::declval<const Tensor&>()})),
                   bool>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull()), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull(100)), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull("x")), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull("x", 100)), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull(100ms)), Result<TensorList>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull("x", 100ms)), Result<TensorList>>);
static_assert(
    std::is_same_v<decltype(std::declval<Run&>()({std::declval<const Tensor&>()})), TensorList>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().alloc("x")), Tensor>);

// No single-Tensor verb. Each detector is one requires-expression over an argument type; it is
// false for a bare Tensor because that call is ill-formed (no overload takes it), and the same
// detector is true for a TensorList (the control), so a false result is not a broken detector.
template <class R, class Arg>
concept Pushes = requires(R& run, const Arg& arg) { run.push(arg); };
template <class R, class Arg>
concept PushesNamed = requires(R& run, const Arg& arg) { run.push("x", arg); };
template <class R, class Arg>
concept TryPushes = requires(R& run, const Arg& arg) { run.try_push(arg); };
template <class R, class Arg>
concept Calls = requires(R& run, const Arg& arg) { run(arg); };
template <class Arg> struct Callback {
  void operator()(Arg) const {}
};
template <class R, class Arg>
concept Delivers = requires(R& run, Callback<Arg> callback) { run.on_output(callback); };

static_assert(Pushes<Run, TensorList> && !Pushes<Run, Tensor>,
              "run.push(t) with a bare Tensor does not compile; use TensorList");
static_assert(PushesNamed<Run, TensorList> && !PushesNamed<Run, Tensor>,
              "run.push(\"x\", t) with a bare Tensor does not compile; use TensorList");
static_assert(TryPushes<Run, TensorList> && !TryPushes<Run, Tensor>,
              "run.try_push(t) with a bare Tensor does not compile; use TensorList");
static_assert(Calls<Run, TensorList> && !Calls<Run, Tensor>,
              "run(t) with a bare Tensor does not compile; use TensorList");
static_assert(Delivers<Run, TensorList> && !Delivers<Run, Tensor>,
              "on_output takes a TensorList callback, not a Tensor one");

// The process's backend is the one under test (a Run Dataflow does not carry fails at build).
RunOptions options() {
  return RunOptions{};
}

Tensor rgb(int w, int h, std::uint8_t tag) {
  Tensor t = make_color_tensor(w, h, ImageSpec::PixelFormat::RGB, tag);
  return t;
}

Tensor floats(float a, float b) {
  return Tensor::from_vector(std::vector<float>{a, b}, {2}, TensorMemory::CPU);
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

// The int pull, std::optional<Sample>: on GStreamer the deprecated form itself; on Dataflow, where
// the deprecated form fails at the call (checked here), the structured Sample pull.
std::optional<Sample> int_pull(Run& run, int timeout_ms, const std::string& output = {}) {
  if (sima_test::on_dataflow()) {
    require_neat_error(
        [&] { (void)(output.empty() ? run.pull(timeout_ms) : run.pull(output, timeout_ms)); },
        error_codes::kPipelineShape,
        " is deprecated and is not carried by the Dataflow backend: use Run::pull(");
    Sample s;
    PullError e;
    const PullStatus st =
        output.empty() ? run.pull(timeout_ms, s, &e) : run.pull(output, timeout_ms, s, &e);
    return st == PullStatus::Ok ? std::optional<Sample>(std::move(s)) : std::nullopt;
  }
  return output.empty() ? run.pull(timeout_ms) : run.pull(output, timeout_ms);
}

Stamp proposal(std::int64_t frame_id) {
  Stamp s;
  s.frame_id = frame_id;
  return s;
}

// push({t}), push(TensorList{t}) and push("in", {t}) are one verb, the list verb: each carries the
// first tensor's stamp; an unstamped tensor proposes nothing.
void list_pushes_carry_the_stamp() {
  Graph g;
  const auto in = g.input("in");
  g.output("out", in);
  Run run = g.build(options());
  const Tensor stamped = rgb(16, 8, 1).with_stamp(proposal(42));
  require(run.push({stamped}), "push({t})");
  require(run.push(TensorList{rgb(16, 8, 2).with_stamp(proposal(43))}), "push(TensorList{t})");
  require(run.push("in", {rgb(16, 8, 3).with_stamp(proposal(44))}), "push(\"in\", {t})");
  require(run.push({rgb(16, 8, 4)}), "push({t}) without a stamp");
  const Result<TensorList> first = run.pull(2000ms);
  const Result<TensorList> second = run.pull("out", 2000ms);
  const std::optional<Sample> third = int_pull(run, 2000, "out");
  const Result<TensorList> fourth = run.pull("out", 2000ms);
  require(first.ok() && second.ok() && third.has_value() && fourth.ok(), "four outputs");
  require(first->size() == 1U && second->size() == 1U && fourth->size() == 1U,
          "a leaf is pulled as a list of one");
  require(first->front().stamp().frame_id == 42, "push({t}) carried the frame_id");
  require(second->front().stamp().frame_id == 43, "push(TensorList{t}) carried the frame_id");
  require(third->frame_id == 44, "push(\"in\", {t}) carried the frame_id (old pull)");
  require(fourth->front().stamp().frame_id == -1, "an unstamped push carried no frame_id");
  require(bytes_of(first->front()) == bytes_of(stamped) &&
              bytes_of(second->front()) == bytes_of(rgb(16, 8, 2)),
          "the payloads round trip");
  run.close();
}

// push({t, u}) is one item of two tensors (a record); the new pull returns both, in order.
void list_of_two_is_one_item() {
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::FP32;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  Graph g;
  g.add(nodes::Input(input));
  g.add(nodes::Output());
  Run run = g.build(TensorList{floats(1, 2), floats(3, 4)}, options());
  require(run.push({floats(5, 6), floats(7, 8)}), "push({t, u})");
  const std::optional<Sample> out = int_pull(run, 2000);
  require(out && out->tensors.size() == 2U, "push({t, u}) pushed one item of two tensors");
  require(bytes_of(out->tensors[1]) == bytes_of(floats(7, 8)), "the second tensor's bytes");
  require(run.push({floats(9, 10), floats(11, 12)}), "push({t, u}) again");
  const Result<TensorList> list = run.pull(2000ms);
  require(list.ok() && list->size() == 2U && bytes_of((*list)[0]) == bytes_of(floats(9, 10)) &&
              bytes_of((*list)[1]) == bytes_of(floats(11, 12)),
          "pull(ms): the item's two tensors, in order");
  run.close();
}

} // namespace

RUN_BACKEND_TEST("unit_run_verb_resolution_test", ([] {
           std::cout << "[INFO] backend under test: " << sima_test::backend_name() << "\n";
           list_pushes_carry_the_stamp();
           list_of_two_is_one_item();
         }));

// Run's overload set with the Tensor verbs (H0 C2, decision D5): which overload each call form
// takes, pinned at compile time against the real header, then observed at run time.
//
// Compile time: ResolutionProbe derives from Run, brings every Run::push and Run::pull overload in
// with using-declarations, and redeclares the overloads under test with the same parameter lists
// and tagged return types. A member declared in the derived class hides the base member with the
// same parameter-type-list, so overload resolution over the probe is resolution over Run's real
// set, with only the resolved overload's identity made visible in the result type. Each redeclared
// signature is first checked to exist in Run exactly (static_cast of the member pointer).
//
// Run time (on the backend SIMA_NEAT_BACKEND names, requested explicitly): push({t}) carries the
// Tensor's stamp (the Tensor overload proposes it), push(TensorList{t}) does not (the list overload
// keeps its develop meaning: no identity); push({t, u}) and push(TensorList{t}) push one item; the
// int pulls return std::optional<Sample>, the chrono pulls Result<Tensor>.

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
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
[[maybe_unused]] constexpr auto kPushTensor = static_cast<bool (Run::*)(const Tensor&)>(&Run::push);
[[maybe_unused]] constexpr auto kPushNamedTensor =
    static_cast<bool (Run::*)(std::string_view, const Tensor&)>(&Run::push);
[[maybe_unused]] constexpr auto kPushList =
    static_cast<bool (Run::*)(const TensorList&)>(&Run::push);
[[maybe_unused]] constexpr auto kPushNamedList =
    static_cast<bool (Run::*)(std::string_view, const TensorList&)>(&Run::push);
[[maybe_unused]] constexpr auto kPullInt =
    static_cast<std::optional<Sample> (Run::*)(int)>(&Run::pull);
[[maybe_unused]] constexpr auto kPullNamedInt =
    static_cast<std::optional<Sample> (Run::*)(std::string_view, int)>(&Run::pull);
[[maybe_unused]] constexpr auto kPullMs =
    static_cast<Result<Tensor> (Run::*)(std::chrono::milliseconds)>(&Run::pull);
[[maybe_unused]] constexpr auto kPullNamedMs =
    static_cast<Result<Tensor> (Run::*)(std::string_view, std::chrono::milliseconds)>(&Run::pull);

struct ResolutionProbe : Run {
  using Run::pull;
  using Run::push;
  // Named only in unevaluated operands; the bodies exist so nothing is declared without one.
  Tag<1> push(const Tensor&) {
    return {};
  }
  Tag<2> push(std::string_view, const Tensor&) {
    return {};
  }
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
                1>,
    "push({t}) -> push(const Tensor&)");
static_assert(
    resolves_to<decltype(std::declval<ResolutionProbe&>().push(std::declval<const Tensor&>())), 1>,
    "push(t) -> push(const Tensor&)");
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
        decltype(std::declval<ResolutionProbe&>().push("x", {std::declval<const Tensor&>()})), 2>,
    "push(\"x\", {t}) -> push(string_view, const Tensor&)");
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
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull()), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull(100)), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull("x")), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull("x", 100)), std::optional<Sample>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull(100ms)), Result<Tensor>>);
static_assert(std::is_same_v<decltype(std::declval<Run&>().pull("x", 100ms)), Result<Tensor>>);
static_assert(
    std::is_same_v<decltype(std::declval<Run&>()(std::declval<const Tensor&>())), Tensor>);

// The backend under test: SIMA_NEAT_BACKEND=dataflow requests Dataflow explicitly (a Run it cannot
// take fails at build), anything else GStreamer.
RunBackend backend_under_test() {
  const char* env = std::getenv("SIMA_NEAT_BACKEND");
  return env != nullptr && std::string(env) == "dataflow" ? RunBackend::Dataflow
                                                          : RunBackend::GStreamer;
}

RunOptions options() {
  RunOptions opt;
  opt.backend = backend_under_test();
  return opt;
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

Stamp proposal(std::int64_t frame_id) {
  Stamp s;
  s.frame_id = frame_id;
  return s;
}

// push({t}) is the Tensor verb: the stamp's frame_id reaches the pulled message;
// push(TensorList{t}) is the list verb, which proposes nothing; push("in", {t}) the named Tensor
// verb.
void tensor_and_list_pushes_differ() {
  Graph g;
  const auto in = g.input("in");
  g.output("out", in);
  Run run = g.build(options());
  const Tensor stamped = rgb(16, 8, 1).with_stamp(proposal(42));
  require(run.push({stamped}), "push({t})");
  require(run.push(TensorList{stamped}), "push(TensorList{t})");
  require(run.push("in", {stamped}), "push(\"in\", {t})");
  const Result<Tensor> first = run.pull(2000ms);
  const Result<Tensor> second = run.pull("out", 2000ms);
  const std::optional<Sample> third = run.pull("out", 2000);
  require(first.ok() && second.ok() && third.has_value(), "three outputs");
  require(first->stamp().frame_id == 42, "push({t}) carried the Tensor's frame_id (Tensor verb)");
  require(second->stamp().frame_id == -1,
          "push(TensorList{t}) carried no frame_id (the list verb, unchanged)");
  require(third->frame_id == 42, "push(\"in\", {t}) carried the frame_id (named Tensor verb)");
  require(bytes_of(*first) == bytes_of(stamped) && bytes_of(*second) == bytes_of(stamped),
          "the payloads round trip");
  run.close();
}

// push({t, u}) is the list verb: one item of two tensors (a record).
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
  const std::optional<Sample> out = run.pull(2000);
  require(out && out->tensors.size() == 2U, "push({t, u}) pushed one item of two tensors");
  require(bytes_of(out->tensors[1]) == bytes_of(floats(7, 8)), "the second tensor's bytes");
  run.close();
}

} // namespace

RUN_TEST("unit_run_verb_resolution_test", ([] {
           std::cout << "[INFO] backend under test: "
                     << (backend_under_test() == RunBackend::Dataflow ? "dataflow" : "gstreamer")
                     << "\n";
           tensor_and_list_pushes_differ();
           list_of_two_is_one_item();
         }));

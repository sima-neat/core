#include "neat/runtime.h"
#include "graph/GraphRun.h"
#include "model/Model.h"
#include "pipeline/Graph.h"

#include <future>
#include <iostream>
#include <type_traits>
#include <utility>

template <typename T, typename = void>
struct has_graph_build_input_runoptions_only : std::false_type {};

template <typename T>
struct has_graph_build_input_runoptions_only<T,
                                             std::void_t<decltype(std::declval<T&>().build(
                                                 std::declval<const simaai::neat::TensorList&>(),
                                                 std::declval<const simaai::neat::RunOptions&>()))>>
    : std::true_type {};

template <typename T, typename = void> struct has_graph_build_input_runmode : std::false_type {};

template <typename T>
struct has_graph_build_input_runmode<
    T, std::void_t<decltype(std::declval<T&>().build(
           std::declval<const simaai::neat::TensorList&>(), std::declval<simaai::neat::RunMode>(),
           std::declval<const simaai::neat::RunOptions&>()))>> : std::true_type {};

template <typename T, typename Input, typename = void>
struct has_build_single_input : std::false_type {};

template <typename T, typename Input>
struct has_build_single_input<
    T, Input, std::void_t<decltype(std::declval<T&>().build(std::declval<const Input&>()))>>
    : std::true_type {};

template <typename T, typename Input, typename = void>
struct has_run_single_input : std::false_type {};

template <typename T, typename Input>
struct has_run_single_input<
    T, Input, std::void_t<decltype(std::declval<T&>().run(std::declval<const Input&>()))>>
    : std::true_type {};

template <typename T, typename Input, typename = void>
struct has_push_single_input : std::false_type {};

template <typename T, typename Input>
struct has_push_single_input<
    T, Input, std::void_t<decltype(std::declval<T&>().push(std::declval<const Input&>()))>>
    : std::true_type {};

template <typename T, typename Input, typename = void>
struct has_graphrun_push_single_input : std::false_type {};

template <typename T, typename Input>
struct has_graphrun_push_single_input<
    T, Input,
    std::void_t<decltype(std::declval<T&>().push(std::declval<simaai::neat::graph::NodeId>(),
                                                 std::declval<const Input&>()))>> : std::true_type {
};

template <typename T, typename Input, typename = void>
struct has_graphrun_input_push_single_input : std::false_type {};

template <typename T, typename Input>
struct has_graphrun_input_push_single_input<
    T, Input, std::void_t<decltype(std::declval<T&>().push(std::declval<const Input&>()))>>
    : std::true_type {};

template <typename T, typename = void> struct has_enable_metrics_field : std::false_type {};

template <typename T>
struct has_enable_metrics_field<T, std::void_t<decltype(std::declval<T&>().enable_metrics)>>
    : std::true_type {};

template <typename T, typename = void> struct has_stats_method : std::false_type {};

template <typename T>
struct has_stats_method<T, std::void_t<decltype(std::declval<const T&>().stats())>>
    : std::true_type {};

template <typename T, typename = void> struct has_input_stats_method : std::false_type {};

template <typename T>
struct has_input_stats_method<T, std::void_t<decltype(std::declval<const T&>().input_stats())>>
    : std::true_type {};

template <typename T, typename = void> struct has_measurement_summary_method : std::false_type {};

template <typename T>
struct has_measurement_summary_method<
    T, std::void_t<decltype(std::declval<const T&>().measurement_summary())>> : std::true_type {};

template <typename T, typename = void> struct has_metrics_report_method : std::false_type {};

template <typename T>
struct has_metrics_report_method<T,
                                 std::void_t<decltype(std::declval<const T&>().metrics_report())>>
    : std::true_type {};

template <typename T, typename = void> struct has_metrics_method : std::false_type {};

template <typename T>
struct has_metrics_method<T, std::void_t<decltype(std::declval<const T&>().metrics())>>
    : std::true_type {};

struct NoopMeasureFn {
  void operator()() const {}
};

template <typename T, typename = void> struct has_measure_method : std::false_type {};

template <typename T>
struct has_measure_method<T, std::void_t<decltype(std::declval<T&>().measure(
                                 std::declval<simaai::neat::MeasureOptions>(), NoopMeasureFn{}))>>
    : std::true_type {};

template <typename T, typename = void> struct has_report_method : std::false_type {};

template <typename T>
struct has_report_method<T, std::void_t<decltype(std::declval<const T&>().report())>>
    : std::true_type {};

template <typename T, typename = void> struct has_diag_snapshot_method : std::false_type {};

template <typename T>
struct has_diag_snapshot_method<T, std::void_t<decltype(std::declval<const T&>().diag_snapshot())>>
    : std::true_type {};

template <typename T, typename = void> struct has_power_summary_method : std::false_type {};

template <typename T>
struct has_power_summary_method<T, std::void_t<decltype(std::declval<const T&>().power_summary())>>
    : std::true_type {};

template <typename T, typename = void> struct has_diagnostics_summary_method : std::false_type {};

template <typename T>
struct has_diagnostics_summary_method<
    T, std::void_t<decltype(std::declval<const T&>().diagnostics_summary())>> : std::true_type {};

template <typename T, typename = void> struct has_set_frame_callback : std::false_type {};

template <typename T>
struct has_set_frame_callback<T, std::void_t<decltype(&T::set_frame_callback)>> : std::true_type {};

template <typename T, typename = void> struct has_add_group_method : std::false_type {};

template <typename T>
struct has_add_group_method<T, std::void_t<decltype(&T::add_group)>> : std::true_type {};

template <typename T, typename = void> struct has_preprocess_graph : std::false_type {};

template <typename T>
struct has_preprocess_graph<T, std::void_t<decltype(std::declval<const T&>().preprocess_graph())>>
    : std::true_type {};

template <typename T, typename = void> struct has_inference_graph : std::false_type {};

template <typename T>
struct has_inference_graph<T, std::void_t<decltype(std::declval<const T&>().inference_graph())>>
    : std::true_type {};

template <typename T, typename = void> struct has_postprocess_graph : std::false_type {};

template <typename T>
struct has_postprocess_graph<T, std::void_t<decltype(std::declval<const T&>().postprocess_graph())>>
    : std::true_type {};

template <typename T>
concept accepts_brace_build = requires(T& graph) { graph.build({}); };

template <typename T>
concept accepts_run_options_brace_build =
    requires(T& graph) { graph.build({simaai::neat::RunOptions{}}); };

template <typename T>
concept accepts_brace_connect =
    requires(T& graph, const T& from, const T& to) { graph.connect(from, to, {}); };

template <typename T>
concept has_connect_realtime =
    requires(T& graph, const T& from, const T& to, const simaai::neat::GraphLinkOptions& options) {
      graph.connect_realtime(from, to, options);
    };

template <typename T>
concept has_fused_realtime_build = requires(T& graph) { graph.build_fused_realtime_sources(); };

int main() {
  using simaai::neat::Graph;
  using simaai::neat::Run;
  using simaai::neat::Sample;
  using simaai::neat::Tensor;
  using simaai::neat::TensorList;

  static_assert(std::is_same_v<simaai::neat::Memory, simaai::neat::TensorMemory>,
                "Memory should be the public alias for TensorMemory");
  static_assert(std::is_same_v<simaai::neat::ImageType, simaai::neat::ImageSpec::PixelFormat>,
                "ImageType should be the public alias for ImageSpec::PixelFormat");

  static_assert(
      std::is_same_v<decltype(std::declval<Graph&>().build(std::declval<const TensorList&>())),
                     Run>,
      "Graph::build(TensorList) should resolve to Run");
  static_assert(
      std::is_same_v<
          decltype(std::declval<Graph&>().build(std::declval<const std::vector<cv::Mat>&>())), Run>,
      "Graph::build(vector<cv::Mat>) should resolve to Run");
  static_assert(
      std::is_same_v<decltype(std::declval<Graph&>().build(std::declval<const Sample&>())), Run>,
      "Graph::build(Sample) should resolve to Run");
  static_assert(
      std::is_same_v<decltype(std::declval<Graph&>().run(std::declval<const TensorList&>())),
                     TensorList>,
      "Graph::run(TensorList) should resolve to TensorList");
  static_assert(std::is_same_v<decltype(std::declval<Graph&>().run(
                                   std::declval<const std::vector<cv::Mat>&>())),
                               TensorList>,
                "Graph::run(vector<cv::Mat>) should resolve to TensorList");
  static_assert(
      std::is_same_v<decltype(std::declval<Graph&>().run(std::declval<const Sample&>())), Sample>,
      "Graph::run(Sample) should resolve to Sample");
  static_assert(!has_build_single_input<Graph, Tensor>::value,
                "Graph::build(Tensor) should not exist; use TensorList");
  static_assert(!has_run_single_input<Graph, Tensor>::value,
                "Graph::run(Tensor) should not exist; use TensorList");
  static_assert(!has_push_single_input<Run, Tensor>::value,
                "Run::push(Tensor) should not exist; use TensorList");
  static_assert(!has_run_single_input<Run, Tensor>::value,
                "Run::run(Tensor) should not exist; use TensorList");
  static_assert(!has_run_single_input<simaai::neat::Model, Tensor>::value,
                "Model::run(Tensor) should not exist; use TensorList");
  static_assert(has_graph_build_input_runoptions_only<Graph>::value,
                "Graph::build(TensorList, RunOptions) is the reusable runner shorthand");
  static_assert(!has_graph_build_input_runmode<Graph>::value,
                "Graph::build(TensorList, RunMode, RunOptions) should not be public");
  static_assert(accepts_brace_build<Graph>, "Graph::build({}) must remain unambiguous");
  static_assert(accepts_run_options_brace_build<Graph>,
                "Graph::build({RunOptions{}}) must retain its released meaning");
  static_assert(accepts_brace_connect<Graph>, "Graph::connect(from, to, {}) must remain valid");
  static_assert(!has_fused_realtime_build<Graph>,
                "fusion is an internal lowering selected by ordinary Graph::build()");
  static_assert(!has_connect_realtime<Graph>,
                "bounded realtime links must use ordinary Graph::connect()");
  static_assert(
      std::is_same_v<decltype(simaai::neat::GraphLinkOptions::max_inflight_per_stream), int> &&
          std::is_same_v<decltype(simaai::neat::GraphLinkOptions::max_inflight_total), int>,
      "GraphLinkOptions must directly expose realtime admission limits");
  static_assert(std::is_same_v<decltype(std::declval<Graph&>().connect(
                                   std::declval<const Graph&>(), std::declval<const Graph&>(),
                                   std::declval<simaai::neat::GraphLinkOptions&>())),
                               Graph&>,
                "ordinary Graph::connect() must accept GraphLinkOptions");

  static_assert(!has_enable_metrics_field<simaai::neat::RunOptions>::value,
                "RunOptions::enable_metrics should not be public");

  static_assert(!has_stats_method<Run>::value, "Run::stats() should not be public");
  static_assert(!has_input_stats_method<Run>::value, "Run::input_stats() should not be public");
  static_assert(!has_measurement_summary_method<Run>::value,
                "Run::measurement_summary() should not be public");
  static_assert(!has_metrics_report_method<Run>::value,
                "Run::metrics_report() should not be public");
  static_assert(!has_metrics_method<Run>::value, "Run::metrics() should not be public");
  static_assert(!has_measure_method<Run>::value, "Run::measure() should not be public");
  static_assert(!has_report_method<Run>::value, "Run::report() should not be public");
  static_assert(!has_diag_snapshot_method<Run>::value, "Run::diag_snapshot() should not be public");
  static_assert(!has_power_summary_method<Run>::value, "Run::power_summary() should not be public");
  static_assert(!has_diagnostics_summary_method<Run>::value,
                "Run::diagnostics_summary() should not be public");
  static_assert(!has_stats_method<simaai::neat::Model::Runner>::value,
                "Model::Runner::stats() should not be public");
  static_assert(!has_measurement_summary_method<simaai::neat::Model::Runner>::value,
                "Model::Runner::measurement_summary() should not be public");
  static_assert(!has_metrics_report_method<simaai::neat::Model::Runner>::value,
                "Model::Runner::metrics_report() should not be public");
  static_assert(!has_metrics_method<simaai::neat::Model::Runner>::value,
                "Model::Runner::metrics() should not be public");
  static_assert(!has_measure_method<simaai::neat::Model::Runner>::value,
                "Model::Runner::measure() should not be public");
  static_assert(!has_report_method<simaai::neat::Model::Runner>::value,
                "Model::Runner::report() should not be public");
  static_assert(!has_diag_snapshot_method<simaai::neat::Model::Runner>::value,
                "Model::Runner::diag_snapshot() should not be public");

  static_assert(!has_set_frame_callback<Graph>::value,
                "Graph::set_frame_callback should not exist");
  static_assert(!has_add_group_method<Graph>::value,
                "Graph::add_group should not exist; reusable fragments are Graph objects");
  static_assert(
      std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().preprocess()), Graph>,
      "Model::preprocess() should return Graph");
  static_assert(
      std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().inference()), Graph>,
      "Model::inference() should return Graph");
  static_assert(
      std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().postprocess()), Graph>,
      "Model::postprocess() should return Graph");
  static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().graph()), Graph>,
                "Model::graph() should return Graph during Phase 7A");
  static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().fragment(
                                   simaai::neat::Model::Stage::Full)),
                               Graph>,
                "Model::fragment() should return Graph");
  static_assert(!has_preprocess_graph<simaai::neat::Model>::value,
                "Model::preprocess_graph() should not exist");
  static_assert(!has_inference_graph<simaai::neat::Model>::value,
                "Model::inference_graph() should not exist");
  static_assert(!has_postprocess_graph<simaai::neat::Model>::value,
                "Model::postprocess_graph() should not exist");
  static_assert(std::is_same_v<decltype(simaai::neat::prewarm_runtime()), void>,
                "prewarm_runtime should return void");
  static_assert(std::is_same_v<decltype(simaai::neat::prewarm_runtime_async()), std::future<void>>,
                "prewarm_runtime_async should return std::future<void>");

  std::cout << "[OK] unit_graph_api_surface_test passed\n";
  return 0;
}

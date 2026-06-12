#include "asset_utils.h"
#include "test_main.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path);
  require(in.is_open(), "failed to open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string strip_cpp_comments(const std::string& text) {
  std::string out;
  out.reserve(text.size());

  bool in_line_comment = false;
  bool in_block_comment = false;
  bool in_string = false;
  bool in_char = false;
  bool escaped = false;

  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    const char next = (i + 1U < text.size()) ? text[i + 1U] : '\0';

    if (in_line_comment) {
      if (c == '\n') {
        in_line_comment = false;
        out.push_back(c);
      }
      continue;
    }

    if (in_block_comment) {
      if (c == '\n') {
        out.push_back(c);
      }
      if (c == '*' && next == '/') {
        in_block_comment = false;
        ++i;
      }
      continue;
    }

    if (!in_string && !in_char && c == '/' && next == '/') {
      in_line_comment = true;
      ++i;
      continue;
    }

    if (!in_string && !in_char && c == '/' && next == '*') {
      in_block_comment = true;
      ++i;
      continue;
    }

    out.push_back(c);

    if (escaped) {
      escaped = false;
      continue;
    }
    if ((in_string || in_char) && c == '\\') {
      escaped = true;
      continue;
    }
    if (!in_char && c == '"') {
      in_string = !in_string;
      continue;
    }
    if (!in_string && c == '\'') {
      in_char = !in_char;
      continue;
    }
  }

  return out;
}

void require_absent(const std::string& text, const std::string& token,
                    const std::filesystem::path& path) {
  require(text.find(token) == std::string::npos,
          path.string() + " must not contain active token: " + token);
}

void require_present(const std::string& text, const std::string& token,
                     const std::filesystem::path& path) {
  require(text.find(token) != std::string::npos,
          path.string() + " must contain active token: " + token);
}

} // namespace

RUN_TEST(
    "graph_migration_phase3_architecture_guard_test", ([] {
      const std::filesystem::path root = sima_test::test_source_root();
      const std::filesystem::path graph_run_state = root / "src/graph/internal/GraphRunState.h";
      const std::filesystem::path graph_build = root / "src/graph/GraphBuild.cpp";
      const std::filesystem::path graph_run_threads = root / "src/graph/GraphRunThreads.cpp";
      const std::filesystem::path edge_router = root / "src/pipeline/runtime/EdgeRouter.h";
      const std::filesystem::path run_core = root / "src/pipeline/runtime/RunCore.h";
      const std::filesystem::path run_core_graph_start =
          root / "src/pipeline/runtime/RunCoreGraphStart.cpp";
      const std::filesystem::path run_core_graph_stop =
          root / "src/pipeline/runtime/RunCoreGraphStop.cpp";
      const std::filesystem::path public_graph = root / "include/pipeline/Graph.h";
      const std::filesystem::path public_run = root / "include/pipeline/Run.h";
      const std::filesystem::path runtime_graph_run = root / "include/graph/GraphRun.h";

      if (!std::filesystem::exists(graph_run_state) || !std::filesystem::exists(graph_build) ||
          !std::filesystem::exists(graph_run_threads) || !std::filesystem::exists(edge_router) ||
          !std::filesystem::exists(run_core) || !std::filesystem::exists(run_core_graph_start) ||
          !std::filesystem::exists(run_core_graph_stop) || !std::filesystem::exists(public_graph) ||
          !std::filesystem::exists(public_run) || !std::filesystem::exists(runtime_graph_run)) {
        std::cout << "[INFO] complete source tree unavailable at " << root
                  << "; source architecture guard is build-tree only\n";
        return;
      }

      const std::string graph_run_state_text = strip_cpp_comments(read_text(graph_run_state));
      require_absent(graph_run_state_text, "Graph graph;", graph_run_state);
      require_absent(graph_run_state_text, "Run run;", graph_run_state);
      require_absent(graph_run_state_text, "CompiledGraph compiled;", graph_run_state);
      require_absent(graph_run_state_text, "ExecutionGraphRuntime execution;", graph_run_state);
      require_absent(graph_run_state_text, "GraphRunOptions opt", graph_run_state);
      require_absent(graph_run_state_text, "std::atomic<bool> stop", graph_run_state);
      require_absent(graph_run_state_text, "std::mutex error_mu", graph_run_state);
      require_absent(graph_run_state_text, "std::string error", graph_run_state);
      require_absent(graph_run_state_text, "PowerMonitor", graph_run_state);
      require_absent(graph_run_state_text, "verbose_guard", graph_run_state);
      require_absent(graph_run_state_text, "output_rate_reported", graph_run_state);
      require_absent(graph_run_state_text, "sched_reported", graph_run_state);
      require_absent(graph_run_state_text, "GraphRunStats", graph_run_state);
      require_absent(graph_run_state_text, "ensure_pipeline_built", graph_run_state);
      require_absent(graph_run_state_text, "void signal_stop", graph_run_state);
      require_absent(graph_run_state_text, "void request_stop", graph_run_state);
      require_absent(graph_run_state_text, "dispatch_to_stage_group", graph_run_state);
      require_present(graph_run_state_text, "std::shared_ptr<simaai::neat::runtime::RunCore> core",
                      graph_run_state);
      require_present(graph_run_state_text, "bool stop_requested() const", graph_run_state);
      require_absent(graph_run_state_text, "struct PipelineRuntime", graph_run_state);
      require_absent(graph_run_state_text, "struct StageRuntime", graph_run_state);
      require_absent(graph_run_state_text, "struct StageGroup", graph_run_state);
      require_present(graph_run_state_text, "runtime::ExecutionGraphRuntime", graph_run_state);
      require_absent(graph_run_state_text, "runtime::PipelineSegmentRuntime", graph_run_state);
      require_absent(graph_run_state_text, "runtime::StageRuntime", graph_run_state);
      require_absent(graph_run_state_text, "runtime::StageGroup", graph_run_state);

      const std::string edge_router_text = strip_cpp_comments(read_text(edge_router));
      require_present(edge_router_text, "class EdgeRouter", edge_router);
      require_present(edge_router_text, "dispatch_to_targets", edge_router);

      const std::string run_core_text = strip_cpp_comments(read_text(run_core));
      require_present(run_core_text, "struct GraphRuntimeOptions", run_core);
      require_present(run_core_text, "GraphRuntimeOptions graph_options", run_core);
      require_present(run_core_text, "GraphRunStats> graph_stats", run_core);
      require_present(run_core_text, "graph_signal_stop", run_core);
      require_present(run_core_text, "graph_request_stop", run_core);
      require_present(run_core_text, "stop_graph", run_core);
      require_present(run_core_text, "ensure_graph_pipeline_built", run_core);
      require_present(run_core_text, "graph_dispatch_to_stage_group", run_core);
      require_present(run_core_text, "graph_push", run_core);
      require_present(run_core_text, "graph_sanitize_pipeline_input", run_core);
      require_present(run_core_text, "graph_restore_stream_id_if_needed", run_core);
      require_present(run_core_text, "graph_pull", run_core);

      const std::string graph_build_text = strip_cpp_comments(read_text(graph_build));
      require_present(graph_build_text, "RunCore::start", graph_build);
      require_absent(graph_build_text, "runtime::EdgeRouter", graph_build);
      require_absent(graph_build_text, "std::thread", graph_build);
      require_absent(graph_build_text, "start_pipeline_segment", graph_build);
      require_absent(graph_build_text, "transport.input_queue", graph_build);
      require_absent(graph_build_text, "auto dispatch_target", graph_build);

      const std::string run_core_graph_start_text =
          strip_cpp_comments(read_text(run_core_graph_start));
      require_present(run_core_graph_start_text, "RunCore::start(ExecutionGraphPlan plan",
                      run_core_graph_start);
      require_present(run_core_graph_start_text, "start_graph_plan", run_core_graph_start);
      require_present(run_core_graph_start_text, "EdgeRouter", run_core_graph_start);

      const std::string run_core_graph_stop_text =
          strip_cpp_comments(read_text(run_core_graph_stop));
      require_present(run_core_graph_stop_text, "RunCore::stop_graph", run_core_graph_stop);
      require_present(run_core_graph_stop_text, "graph_signal_stop", run_core_graph_stop);

      const std::string graph_run_threads_text = strip_cpp_comments(read_text(graph_run_threads));
      require_present(graph_run_threads_text, "stop_graph", graph_run_threads);
      require_absent(graph_run_threads_text, "transport.pull_thread", graph_run_threads);
      require_absent(graph_run_threads_text, "transport.push_thread", graph_run_threads);
      require_absent(graph_run_threads_text, "graph_signal_stop", graph_run_threads);

      const std::string public_graph_text = strip_cpp_comments(read_text(public_graph));
      require_absent(public_graph_text, "build_graph", public_graph);

      const std::string public_run_text = strip_cpp_comments(read_text(public_run));
      require_present(public_run_text, "std::shared_ptr<runtime::RunCore> core_", public_run);
      require_absent(public_run_text, "struct RunStats", public_run);
      require_absent(public_run_text, "struct RunDiagSnapshot", public_run);
      require_absent(public_run_text, "struct RunElementTimingStats", public_run);
      require_absent(public_run_text, "struct InputStreamStats", public_run);
      require_absent(public_run_text, "pipeline_state_", public_run);
      require_absent(public_run_text, "graph_state_", public_run);

      const std::string runtime_graph_run_text = strip_cpp_comments(read_text(runtime_graph_run));
      require_absent(runtime_graph_run_text, "GraphRun* run_", runtime_graph_run);
      require_absent(runtime_graph_run_text, "const std::vector<Output>* outputs_",
                     runtime_graph_run);
      require_present(runtime_graph_run_text, "std::weak_ptr<State> state_", runtime_graph_run);
    }));

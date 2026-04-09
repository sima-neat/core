#include "internal/GraphRunState.h"

#include <unistd.h>

namespace simaai::neat::graph {
GraphSession::GraphSession(Graph graph) : graph_(std::move(graph)) {}

GraphRun GraphSession::build(const GraphRunOptions& opt) {
  auto state = std::make_shared<GraphRun::State>();
  state->opt = opt;
  if (state->opt.push_timeout_ms < 0) {
    throw std::invalid_argument(
        "GraphSession::build: GraphRunOptions.push_timeout_ms must be >= 0");
  }
  if (state->opt.pull_timeout_ms < 0) {
    throw std::invalid_argument(
        "GraphSession::build: GraphRunOptions.pull_timeout_ms must be >= 0");
  }

  Compiler compiler;
  state->compiled = compiler.compile(graph_);
  state->node_labels.resize(graph_.node_count());
  for (NodeId id = 0; id < graph_.node_count(); ++id) {
    std::string label;
    const auto& node = graph_.node(id);
    if (node) {
      label = node->user_label();
      if (label.empty())
        label = node->kind();
    }
    if (label.empty()) {
      label = "node" + std::to_string(id);
    }
    state->node_labels[id] = std::move(label);
  }

  // Build pipeline runtimes.
  state->pipelines.reserve(state->compiled.pipelines.size());
  for (const auto& seg : state->compiled.pipelines) {
    SessionOptions sess_opt;
    // Use a shared suffix across graph segments so buffer-name metadata
    // stays consistent between producer/consumer pipelines.
    sess_opt.element_name_suffix = "_graph";

    Session session(sess_opt);
    std::vector<std::shared_ptr<simaai::neat::Node>> nodes = seg.group.nodes();

    if (!seg.source_like && seg.input_edges.empty() && has_internal_source(seg.group) &&
        !has_input_appsrc(seg.group)) {
      throw std::runtime_error(
          "GraphSession: pipeline segment has internal source but is not source_like. "
          "Refusing to inject Input into a source pipeline.");
    }

    if (!seg.source_like && !has_input_appsrc(seg.group)) {
      InputOptions opt_src = input_opts_from_spec(seg.input_spec, seg.input_complete);
      nodes.insert(nodes.begin(), simaai::neat::nodes::Input(opt_src));
    }

    const bool is_graph_sink = graph_.out_degree(seg.node_ids.back()) == 0;
    const bool need_output = !seg.output_edges.empty() || is_graph_sink;
    if (need_output && !has_output_appsink(seg.group)) {
      nodes.push_back(simaai::neat::nodes::Output());
    }

    const simaai::neat::Input* src_node = nullptr;
    if (!seg.source_like) {
      for (const auto& node : nodes) {
        if (!node)
          continue;
        if (auto* input = dynamic_cast<const simaai::neat::Input*>(node.get())) {
          src_node = input;
          break;
        }
      }
    }

    pipeline_internal::PipelineBuildContext build_ctx(sess_opt);
    build_ctx.apply_name_transform_to_configs(nodes);
    build_ctx.check_config_wiring(nodes);

    for (const auto& n : nodes) {
      session.add(n);
    }

    auto runtime = std::make_unique<GraphRun::State::PipelineRuntime>();
    runtime->seg = seg;
    runtime->session = std::move(session);
    runtime->has_input = !seg.source_like;
    runtime->has_output = need_output;
    if (runtime->has_input && src_node) {
      runtime->expected_buffer_names =
          build_ctx.resolve_expected_buffer_names(src_node->options().buffer_name);
    }

    if (runtime->has_input) {
      runtime->input_queue = std::make_shared<BlockingQueueSample>(state->opt.edge_queue);
    }

    const std::size_t index = state->pipelines.size();
    for (NodeId nid : seg.node_ids) {
      state->node_to_pipeline[nid] = index;
    }

    state->pipelines.push_back(std::move(runtime));
  }

  // Build stage runtimes (with optional pooling).
  state->stage_groups.reserve(state->compiled.stages.size());
  for (const auto& st : state->compiled.stages) {
    if (!st.node)
      continue;

    StageNodeOptions opt_node = st.node->options();
    if (opt_node.instances < 1)
      opt_node.instances = 1;

    const std::size_t group_index = state->stage_groups.size();
    state->stage_groups.emplace_back();
    auto& group = state->stage_groups.back();
    group.node_id = st.node_id;
    group.options = opt_node;
    group.instances.reserve(static_cast<std::size_t>(opt_node.instances));

    const std::size_t capacity =
        (opt_node.max_inflight > 0) ? opt_node.max_inflight : state->opt.edge_queue;

    for (int i = 0; i < opt_node.instances; ++i) {
      auto rt = std::make_unique<GraphRun::State::StageRuntime>(capacity);
      rt->node_id = st.node_id;
      rt->exec = st.node->factory()();
      for (const auto& port : st.node->input_ports()) {
        const PortId pid = graph_.intern_port(port.name);
        rt->input_ports.push_back(pid);
        rt->ports.in.emplace(port.name, pid);
      }
      for (const auto& port : st.node->output_ports()) {
        const PortId pid = graph_.intern_port(port.name);
        rt->output_ports.push_back(pid);
        rt->ports.out.emplace(port.name, pid);
      }
      if (rt->exec) {
        rt->exec->set_ports(rt->ports);
      }

      group.instances.push_back(state->stages.size());
      state->stages.push_back(std::move(rt));
    }

    state->node_to_stage_group[st.node_id] = group_index;
  }

  // Build adjacency.
  for (std::size_t eidx = 0; eidx < state->compiled.edges.size(); ++eidx) {
    const Edge& e = state->compiled.edges[eidx];
    std::vector<DownstreamTarget>& outs = state->adjacency[port_key(e.from, e.from_port)];

    auto it_stage = state->node_to_stage_group.find(e.to);
    if (it_stage != state->node_to_stage_group.end()) {
      outs.push_back(
          DownstreamTarget{DownstreamTarget::Kind::StageGroup, it_stage->second, e.to_port});
      continue;
    }
    auto it_pipe = state->node_to_pipeline.find(e.to);
    if (it_pipe != state->node_to_pipeline.end()) {
      outs.push_back(
          DownstreamTarget{DownstreamTarget::Kind::PipelineInput, it_pipe->second, e.to_port});
      continue;
    }
  }

  // Sink queues for terminal nodes.
  for (NodeId id = 0; id < graph_.node_count(); ++id) {
    if (graph_.out_degree(id) == 0) {
      state->sinks[id] = std::make_shared<BlockingQueueSample>(state->opt.edge_queue);
    }
  }

  // Prebuild pipelines with fully known input specs (best effort).
  for (std::size_t i = 0; i < state->pipelines.size(); ++i) {
    auto& rt = *state->pipelines[i];
    if (!rt.has_input)
      continue;
    if (!rt.seg.input_complete)
      continue;
    std::string spec_err;
    auto sample_opt = sample_from_input_spec(rt.seg.input_spec, &spec_err);
    if (!sample_opt.has_value()) {
      if (graph_debug_enabled() && !spec_err.empty()) {
        std::fprintf(stderr, "[GRAPH] prebuild_skip seg=%zu reason=%s\n",
                     static_cast<std::size_t>(rt.seg.id), spec_err.c_str());
      }
      continue;
    }
    std::string build_err;
    if (!state->ensure_pipeline_built(i, *sample_opt, &build_err)) {
      if (graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] prebuild_failed seg=%zu err=%s\n",
                     static_cast<std::size_t>(rt.seg.id),
                     build_err.empty() ? "<unknown>" : build_err.c_str());
      }
    }
  }

  // Start stage workers.
  for (std::size_t i = 0; i < state->stages.size(); ++i) {
    auto& rt = *state->stages[i];
    rt.worker = std::thread([state, i]() {
      auto& st = *state->stages[i];
      struct DoneGuard {
        std::atomic<bool>& flag;
        ~DoneGuard() {
          flag.store(true);
        }
      } done_guard{st.worker_done};
      while (!state->stop.load()) {
        StageMsg msg;
        if (!st.mailbox.inbox.pop(msg, state->opt.pull_timeout_ms)) {
          continue;
        }
        if (msg.in_port == kInvalidPort && st.input_ports.size() == 1) {
          msg.in_port = st.input_ports[0];
        }
        if (graph_debug_enabled()) {
          graph_debug_sample("stage_on_input", msg.sample);
        }
        std::vector<StageOutMsg> outputs;
        try {
          st.exec->on_input(std::move(msg), outputs);
        } catch (const std::exception& e) {
          state->request_stop(e.what());
          break;
        }

        for (auto& out_msg : outputs) {
          PortId out_port = out_msg.out_port;
          if (out_port == kInvalidPort && st.output_ports.size() == 1) {
            out_port = st.output_ports[0];
          }
          const std::uint64_t key = port_key(st.node_id, out_port);
          const auto it = state->adjacency.find(key);
          if (it == state->adjacency.end() || it->second.empty()) {
            auto sink_it = state->sinks.find(st.node_id);
            if (sink_it != state->sinks.end()) {
              Sample sample_move = std::move(out_msg.sample);
              const std::size_t qsize = sink_it->second->size();
              maybe_force_copy_for_backpressure(sample_move, qsize, "sink_queue", st.node_id);
              if (!sink_it->second->push(std::move(sample_move), state->opt.push_timeout_ms)) {
                if (!state->stop.load(std::memory_order_relaxed)) {
                  std::ostringstream msg;
                  msg << "GraphRun: sink backpressure timeout (node="
                      << static_cast<std::size_t>(st.node_id)
                      << ", edge_queue=" << state->opt.edge_queue
                      << ", push_timeout_ms=" << state->opt.push_timeout_ms
                      << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
                  state->request_stop(msg.str());
                }
              }
            }
            continue;
          }
          auto dispatch_target = [&](const DownstreamTarget& target, Sample&& sample) {
            if (target.kind == DownstreamTarget::Kind::StageGroup) {
              (void)state->dispatch_to_stage_group(target.index, target.port, std::move(sample));
              return;
            }
            if (target.kind == DownstreamTarget::Kind::PipelineInput) {
              std::string build_err;
              if (!state->ensure_pipeline_built(target.index, sample, &build_err)) {
                state->request_stop(build_err.empty() ? "GraphRun: pipeline build failed"
                                                      : build_err);
                return;
              }
              state->sanitize_sample_for_pipeline_input(*state->pipelines[target.index], sample);
              auto& input_queue = state->pipelines[target.index]->input_queue;
              if (input_queue) {
                const std::size_t qsize = input_queue->size();
                maybe_force_copy_for_backpressure(sample, qsize, "pipeline_input_queue",
                                                  state->pipelines[target.index]->seg.id);
                if (!input_queue->push(std::move(sample), state->opt.push_timeout_ms)) {
                  if (!state->stop.load(std::memory_order_relaxed)) {
                    std::ostringstream msg;
                    msg << "GraphRun: pipeline input backpressure timeout (seg="
                        << static_cast<std::size_t>(state->pipelines[target.index]->seg.id)
                        << ", edge_queue=" << state->opt.edge_queue
                        << ", push_timeout_ms=" << state->opt.push_timeout_ms
                        << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
                    state->request_stop(msg.str());
                  }
                }
              }
              return;
            }
            auto sink_it = state->sinks.find(st.node_id);
            if (sink_it != state->sinks.end()) {
              const std::size_t qsize = sink_it->second->size();
              maybe_force_copy_for_backpressure(sample, qsize, "sink_queue", st.node_id);
              if (!sink_it->second->push(std::move(sample), state->opt.push_timeout_ms)) {
                if (!state->stop.load(std::memory_order_relaxed)) {
                  std::ostringstream msg;
                  msg << "GraphRun: sink backpressure timeout (node="
                      << static_cast<std::size_t>(st.node_id)
                      << ", edge_queue=" << state->opt.edge_queue
                      << ", push_timeout_ms=" << state->opt.push_timeout_ms
                      << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
                  state->request_stop(msg.str());
                }
              }
            }
          };

          const auto& targets = it->second;
          if (targets.size() == 1) {
            dispatch_target(targets.front(), std::move(out_msg.sample));
          } else {
            for (const auto& target : targets) {
              Sample sample_copy = out_msg.sample;
              dispatch_target(target, std::move(sample_copy));
            }
          }
        }
      }
    });
  }

  // Start pipeline threads.
  for (std::size_t i = 0; i < state->pipelines.size(); ++i) {
    auto& rt = *state->pipelines[i];

    if (!rt.has_input && rt.seg.source_like) {
      // Source pipeline: build immediately.
      {
        std::lock_guard<std::mutex> lock(rt.mu);
        rt.run = rt.session.build(state->opt.pipeline);
        rt.built.store(true, std::memory_order_release);
      }
      rt.cv.notify_all();
    }

    if (rt.has_output) {
      rt.pull_thread = std::thread([state, i]() {
        auto& pipe = *state->pipelines[i];
        struct DoneGuard {
          std::atomic<bool>& flag;
          ~DoneGuard() {
            flag.store(true);
          }
        } done_guard{pipe.pull_done};
        if (graph_debug_enabled()) {
          std::fprintf(stderr, "[GRAPH] pipeline_pull_thread_start seg=%zu\n",
                       static_cast<std::size_t>(pipe.seg.id));
        }

        // Wait for build if needed.
        {
          std::unique_lock<std::mutex> lock(pipe.mu);
          pipe.cv.wait(lock, [&] {
            return pipe.built.load(std::memory_order_acquire) || state->stop.load();
          });
        }

        if (state->stop.load()) {
          if (graph_debug_enabled()) {
            std::fprintf(stderr, "[GRAPH] pipeline_pull_thread_stop_before_ready seg=%zu\n",
                         static_cast<std::size_t>(pipe.seg.id));
          }
          return;
        }
        if (graph_debug_enabled()) {
          std::fprintf(stderr, "[GRAPH] pipeline_pull_thread_ready seg=%zu\n",
                       static_cast<std::size_t>(pipe.seg.id));
        }

        try {
          const int diag_ms = env_int("SIMA_GRAPH_PIPELINE_DIAG_MS", 0);
          const bool diag_summary = env_bool("SIMA_GRAPH_PIPELINE_DIAG_SUMMARY", false);
          auto last_output = std::chrono::steady_clock::now();
          auto last_diag = last_output;
          auto emit_diag = [&](const char* reason) {
            if (diag_ms <= 0)
              return;
            const auto now = std::chrono::steady_clock::now();
            if (now - last_diag < std::chrono::milliseconds(diag_ms))
              return;
            last_diag = now;
            const RunStats stats = pipe.run.stats();
            const InputStreamStats in_stats = pipe.run.input_stats();
            const std::size_t in_q = pipe.input_queue ? pipe.input_queue->size() : 0;
            const auto idle_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_output).count();
            std::fprintf(
                stderr,
                "[GRAPH] pipeline_diag seg=%zu reason=%s idle_ms=%lld in_q=%zu "
                "in_enq=%llu in_push=%llu out_ready=%llu out_pulled=%llu out_drop=%llu "
                "is_push=%llu is_fail=%llu is_pull=%llu "
                "identity_rewrite_count=%lld identity_map_miss_count=%lld\n",
                static_cast<std::size_t>(pipe.seg.id), reason ? reason : "tick",
                static_cast<long long>(idle_ms), in_q,
                static_cast<unsigned long long>(stats.inputs_enqueued),
                static_cast<unsigned long long>(stats.inputs_pushed),
                static_cast<unsigned long long>(stats.outputs_ready),
                static_cast<unsigned long long>(stats.outputs_pulled),
                static_cast<unsigned long long>(stats.outputs_dropped),
                static_cast<unsigned long long>(in_stats.push_count),
                static_cast<unsigned long long>(in_stats.push_failures),
                static_cast<unsigned long long>(in_stats.pull_count),
                static_cast<long long>(pipe.identity_rewrite_count.load(std::memory_order_relaxed)),
                static_cast<long long>(
                    pipe.identity_map_miss_count.load(std::memory_order_relaxed)));
            if (diag_summary) {
              const std::string summary = pipe.run.diagnostics_summary();
              if (!summary.empty()) {
                std::fprintf(stderr, "[GRAPH] pipeline_diag_summary seg=%zu %s\n",
                             static_cast<std::size_t>(pipe.seg.id), summary.c_str());
              }
            }
          };
          int pull_logs = 0;
          while (!state->stop.load() && pipe.built.load(std::memory_order_acquire)) {
            if (graph_debug_enabled() && pull_logs < 5) {
              std::fprintf(stderr, "[GRAPH] pipeline_pull_wait seg=%zu\n",
                           static_cast<std::size_t>(pipe.seg.id));
              pull_logs++;
            }
            auto sample_opt = pipe.run.pull(state->opt.pull_timeout_ms);
            if (!sample_opt.has_value()) {
              emit_diag("empty");
              if (graph_debug_enabled()) {
                const std::string err = pipe.run.last_error();
                if (!err.empty()) {
                  std::fprintf(stderr, "[GRAPH] pipeline_pull_error seg=%zu err=%s\n",
                               static_cast<std::size_t>(pipe.seg.id), err.c_str());
                } else {
                  static thread_local int empty_logs = 0;
                  if (empty_logs < 5) {
                    std::fprintf(stderr, "[GRAPH] pipeline_pull_empty seg=%zu\n",
                                 static_cast<std::size_t>(pipe.seg.id));
                    empty_logs++;
                  }
                }
              }
              continue;
            }
            Sample sample = *sample_opt;
            last_output = std::chrono::steady_clock::now();
            emit_diag("sample");
            state->restore_stream_id_if_needed(pipe, sample);
            log_first_decoded_once(sample, pipe.seg);
            if (graph_debug_enabled()) {
              graph_debug_sample("pipeline_pull", sample);
            }

            PortId out_port = kInvalidPort;
            if (!pipe.seg.output_edges.empty()) {
              const Edge& e = state->compiled.edges[pipe.seg.output_edges[0]];
              out_port = e.from_port;
            }

            const std::uint64_t key = port_key(pipe.seg.node_ids.back(), out_port);
            const auto it = state->adjacency.find(key);
            if (it == state->adjacency.end() || it->second.empty()) {
              auto sink_it = state->sinks.find(pipe.seg.node_ids.back());
              if (sink_it != state->sinks.end()) {
                if (graph_debug_enabled()) {
                  std::fprintf(stderr, "[GRAPH] model_output_received seg=%zu\n",
                               static_cast<std::size_t>(pipe.seg.id));
                  graph_debug_sample("pipeline_to_sink", sample);
                }
                Sample sample_move = std::move(sample);
                const std::size_t qsize = sink_it->second->size();
                maybe_force_copy_for_backpressure(sample_move, qsize, "sink_queue", pipe.seg.id);
                if (!sink_it->second->push(std::move(sample_move), state->opt.push_timeout_ms)) {
                  if (!state->stop.load(std::memory_order_relaxed)) {
                    std::ostringstream msg;
                    msg << "GraphRun: sink backpressure timeout (node="
                        << static_cast<std::size_t>(pipe.seg.node_ids.back())
                        << ", edge_queue=" << state->opt.edge_queue
                        << ", push_timeout_ms=" << state->opt.push_timeout_ms
                        << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
                    state->request_stop(msg.str());
                  }
                }
              }
              continue;
            }

            auto dispatch_target = [&](const DownstreamTarget& target, Sample&& out_sample) {
              if (target.kind == DownstreamTarget::Kind::StageGroup) {
                (void)state->dispatch_to_stage_group(target.index, target.port,
                                                     std::move(out_sample));
                return;
              }
              if (target.kind == DownstreamTarget::Kind::PipelineInput) {
                std::string build_err;
                if (!state->ensure_pipeline_built(target.index, out_sample, &build_err)) {
                  state->request_stop(build_err.empty() ? "GraphRun: pipeline build failed"
                                                        : build_err);
                  return;
                }
                auto& input_queue = state->pipelines[target.index]->input_queue;
                if (input_queue) {
                  const std::size_t qsize = input_queue->size();
                  maybe_force_copy_for_backpressure(out_sample, qsize, "pipeline_input_queue",
                                                    state->pipelines[target.index]->seg.id);
                  if (!input_queue->push(std::move(out_sample), state->opt.push_timeout_ms)) {
                    if (!state->stop.load(std::memory_order_relaxed)) {
                      std::ostringstream msg;
                      msg << "GraphRun: pipeline input backpressure timeout (seg="
                          << static_cast<std::size_t>(state->pipelines[target.index]->seg.id)
                          << ", edge_queue=" << state->opt.edge_queue
                          << ", push_timeout_ms=" << state->opt.push_timeout_ms
                          << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
                      state->request_stop(msg.str());
                    }
                  }
                }
                return;
              }
              auto sink_it = state->sinks.find(pipe.seg.node_ids.back());
              if (sink_it != state->sinks.end()) {
                if (graph_debug_enabled()) {
                  std::fprintf(stderr, "[GRAPH] model_output_received seg=%zu\n",
                               static_cast<std::size_t>(pipe.seg.id));
                  graph_debug_sample("pipeline_to_sink", out_sample);
                }
                const std::size_t qsize = sink_it->second->size();
                maybe_force_copy_for_backpressure(out_sample, qsize, "sink_queue", pipe.seg.id);
                if (!sink_it->second->push(std::move(out_sample), state->opt.push_timeout_ms)) {
                  if (!state->stop.load(std::memory_order_relaxed)) {
                    std::ostringstream msg;
                    msg << "GraphRun: sink backpressure timeout (node="
                        << static_cast<std::size_t>(pipe.seg.node_ids.back())
                        << ", edge_queue=" << state->opt.edge_queue
                        << ", push_timeout_ms=" << state->opt.push_timeout_ms
                        << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
                    state->request_stop(msg.str());
                  }
                }
              }
            };

            const auto& targets = it->second;
            if (targets.size() == 1) {
              dispatch_target(targets.front(), std::move(sample));
            } else {
              for (const auto& target : targets) {
                Sample sample_copy = sample;
                dispatch_target(target, std::move(sample_copy));
              }
            }
          }
        } catch (const std::exception& e) {
          state->request_stop(e.what());
        }
      });
    }

    if (rt.has_input) {
      rt.push_thread = std::thread([state, i]() {
        auto& pipe = *state->pipelines[i];
        struct DoneGuard {
          std::atomic<bool>& flag;
          ~DoneGuard() {
            flag.store(true);
          }
        } done_guard{pipe.push_done};
        while (!state->stop.load()) {
          Sample sample;
          if (!pipe.input_queue->pop(sample, state->opt.pull_timeout_ms))
            continue;
          if (graph_debug_enabled()) {
            graph_debug_sample("pipeline_push_pop", sample);
          }

          state->sanitize_sample_for_pipeline_input(pipe, sample);

          if (is_encoded_sample(sample) && sample.caps_string.empty()) {
            state->request_stop("GraphRun: encoded Sample missing caps_string");
            return;
          }

          if (!pipe.built.load(std::memory_order_acquire)) {
            std::string build_err;
            if (!state->ensure_pipeline_built(i, sample, &build_err)) {
              state->request_stop(build_err.empty() ? "GraphRun: pipeline build failed"
                                                    : build_err);
              return;
            }
          }

          if (!pipe.run.push(sample)) {
            if (graph_debug_enabled() || graph_push_fail_debug_enabled()) {
              std::fprintf(stderr, "[GRAPH] pipeline_push_failed seg=%zu stop=%d\n",
                           static_cast<std::size_t>(pipe.seg.id),
                           static_cast<int>(state->stop.load()));
              std::fprintf(stderr, "[GRAPH] pipeline_push_failed pid=%d thread=%zu\n",
                           static_cast<int>(::getpid()),
                           static_cast<std::size_t>(
                               std::hash<std::thread::id>{}(std::this_thread::get_id())));
              graph_debug_sample("pipeline_push_failed", sample);
              const std::string err = pipe.run.last_error();
              if (!err.empty()) {
                std::fprintf(stderr, "[GRAPH] pipeline_push_failed err=%s\n", err.c_str());
              }
              const RunStats stats = pipe.run.stats();
              const InputStreamStats in_stats = pipe.run.input_stats();
              std::fprintf(
                  stderr,
                  "[GRAPH] pipeline_push_failed stats in_enq=%llu in_push=%llu out_ready=%llu "
                  "out_pulled=%llu is_push=%llu is_fail=%llu is_pull=%llu\n",
                  static_cast<unsigned long long>(stats.inputs_enqueued),
                  static_cast<unsigned long long>(stats.inputs_pushed),
                  static_cast<unsigned long long>(stats.outputs_ready),
                  static_cast<unsigned long long>(stats.outputs_pulled),
                  static_cast<unsigned long long>(in_stats.push_count),
                  static_cast<unsigned long long>(in_stats.push_failures),
                  static_cast<unsigned long long>(in_stats.pull_count));
              const std::string diag = pipe.run.diagnostics_summary();
              if (!diag.empty()) {
                std::fprintf(stderr, "[GRAPH] pipeline_push_failed diag=%s\n", diag.c_str());
              }
            }
            if (pipeline_internal::env_bool("SIMA_GRAPH_GDB_ON_PUSH_FAIL", false)) {
              std::fprintf(stderr, "[GRAPH] pipeline_push_failed: SIGSTOP for gdb attach pid=%d\n",
                           static_cast<int>(::getpid()));
              std::raise(SIGSTOP);
            }
            state->request_stop("GraphRun: pipeline push failed");
            return;
          }
        }
      });
    }
  }

  return GraphRun(std::move(state));
}
} // namespace simaai::neat::graph

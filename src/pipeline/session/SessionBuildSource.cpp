/**
 * @file SessionBuildSource.cpp
 * @brief Session source-mode build/run methods.
 */

#include "SessionDetail.h"
#include "internal/SessionBuildInternal.h"

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/SessionError.h"
#include "pipeline/SessionReport.h"
#include "pipeline/internal/UxLogging.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <utility>

namespace simaai::neat {

namespace {

void maybe_compile_source_contracts(BuildResult* build_result,
                                    const std::vector<std::shared_ptr<Node>>& nodes,
                                    const SessionOptions& sess_opt, const char* where) {
  ContractCompileInput compile_input;
  compile_input.pipeline_label = where ? where : "Session::build(source)";
  compile_input.processcvu_requested_run_target = sess_opt.processcvu_requested_run_target;
  compile_input.processcvu = sess_opt.processcvu;
  session_build_compile_contracts(build_result, nodes, compile_input, where, nullptr);
}

} // namespace

void Session::run() {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  gst_init_once();

  enforce_caps_behavior(nodes_, "Session::run");
  enforce_source_run_mode(nodes_, "Session::run");

  enforce_sink_last(nodes_);

  require_element("appsink", "Session::run");
  require_element("identity", "Session::run");

  if (!tensor_cb_) {
    session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                             "Session::run: tensor callback not set");
  }

  RunOptions run_opt;
  auto runner = build(run_opt);
  const int timeout_ms = std::max(-1, opt_.callback_timeout_ms);

  while (true) {
    Sample out;
    PullError err;
    const PullStatus status = runner.pull(timeout_ms, out, &err);
    if (status == PullStatus::Ok) {
      if (!sample_has_tensor_list(out) || out.tensors.size() != 1U) {
        session_build_throw_session_error_simple(error_codes::kRuntimePull,
                                                 "Session::run: expected tensor output");
      }
      if (!tensor_cb_(out.tensors.front()))
        break;
      continue;
    }
    if (status == PullStatus::Timeout) {
      continue;
    }
    if (status == PullStatus::Closed) {
      break;
    }
    if (err.report.has_value()) {
      SessionReport rep = std::move(*err.report);
      if (rep.error_code.empty()) {
        rep.error_code = err.code.empty() ? error_codes::kRuntimePull : err.code;
      }
      if (rep.repro_note.empty()) {
        rep.repro_note = "Session::run: runtime pull failed (status=Error)\n"
                         "Hint: inspect Run::report()/SessionReport bus diagnostics.";
      }
      const std::string msg =
          err.message.empty()
              ? session_build_decorate_with_error_code(rep.error_code, rep.repro_note)
              : err.message;
      throw SessionError(msg, std::move(rep));
    }
    const std::string detail = err.message.empty() ? "Session::run: pull failed" : err.message;
    SessionReport rep;
    rep.error_code = err.code.empty() ? error_codes::kRuntimePull : err.code;
    rep.repro_note = detail + "\nHint: inspect Run::report()/SessionReport bus diagnostics.";
    const std::string msg = session_build_decorate_with_error_code(rep.error_code, rep.repro_note);
    throw SessionError(msg, std::move(rep));
  }
}

Run Session::build(const RunOptions& opt) {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  pipeline_internal::ux::ProgressReporter progress(opt_.verbose, 4);
  progress.step("Initializing runtime...");
  gst_init_once();

  enforce_caps_behavior(nodes_, "Session::build");
  enforce_source_run_mode(nodes_, "Session::build");
  enforce_sink_last_if_present(nodes_, "Session::build");

  std::vector<std::shared_ptr<Node>> build_nodes =
      session_build_materialize_model_bound_nodes(nodes_, false);
  session_build_apply_derived_input_contracts(&build_nodes);
  const bool has_sink = session_build_has_output_appsink(build_nodes);
  if (has_sink) {
    require_element("appsink", "Session::build");
    require_element("identity", "Session::build");
  }

  const RunOptions requested_opt = session_build_resolve_build_opt(RunMode::Async, opt);
  RunOptions merged_opt = session_build_apply_run_defaults(requested_opt, opt_);
  InputStreamOptions stream_opt = session_build_make_stream_options(merged_opt, RunMode::Async);
  session_build_maybe_enable_rtsp_appsink_drop(stream_opt, build_nodes);
  session_build_maybe_apply_detess_output_override(build_nodes, stream_opt);
  const bool insert_queue2 = session_build_should_insert_async_queue2(RunMode::Async, merged_opt);

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_RUN_INSERT_BOUNDARIES", false);

  progress.step("Building session...");
  const NameTransform name_transform = make_name_transform(opt_);
  BuildResult br = build_pipeline_full(build_nodes, insert_boundaries, "mysink", insert_queue2,
                                       name_transform, &opt_);
  maybe_compile_source_contracts(&br, build_nodes, opt_, "Session::build");
  last_pipeline_ = br.pipeline_string;
  session_build_enforce_mla_num_buffers(last_pipeline_, "Session::build");
  progress.detail("mode=async nodes=" + std::to_string(build_nodes.size()));

  GstElement* pipeline = session_build_parse_pipeline_or_throw(br, "Session::build");
  session_build_dump_pipeline_element_properties(pipeline);

  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline, br);
  }

  attach_boundary_probes(pipeline, br.diag);
  attach_stage_timing_probes(pipeline, br.diag);
  attach_element_timing_probes(pipeline, br.diag);
  attach_element_flow_probes(pipeline, br.diag);
  session_build_attach_debug_detess_input_probes(pipeline);
  session_build_attach_debug_appsink_probes(pipeline);
  session_build_attach_debug_all_buffer_probes(pipeline);
  session_build_attach_debug_element_buffer_probes(pipeline);
  session_build_attach_boxdecode_debug_probes(pipeline);
  session_build_attach_h264_caps_fixups(pipeline, build_nodes, name_transform);
  session_build_attach_rtsp_debug(pipeline, build_nodes, name_transform);

  GstElement* sink = nullptr;
  if (has_sink) {
    const std::string appsink_name = apply_name_transform(name_transform, "mysink");
    sink = gst_bin_get_by_name(GST_BIN(pipeline), appsink_name.c_str());
    if (!sink) {
      maybe_dump_dot(pipeline, "build_missing_mysink");
      stop_and_unref(pipeline);
      session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                               "Session::build: appsink '" + appsink_name +
                                                   "' not found.\nPipeline:\n" + last_pipeline_,
                                               "Add Output() as the last node.", last_pipeline_);
    }
    session_build_configure_appsink_for_input_stream(sink, stream_opt);
    session_build_configure_appsink_allocation_preference(sink, build_nodes);
  }

  progress.step("Starting pipeline...");
  try {
    set_state_or_throw(pipeline, GST_STATE_PLAYING, "Session::build", br.diag);
  } catch (...) {
    if (sink)
      gst_object_unref(sink);
    stop_and_unref(pipeline);
    throw;
  }

  SampleSpec spec = session_build_make_placeholder_spec();
  InputOptions src_opt = session_build_resolve_appsrc_options(InputOptions{}, name_transform);
  InputStream stream =
      InputStream::create(pipeline, nullptr, sink, spec, src_opt, stream_opt, br.diag, guard_);
  progress.done("Session ready");
  return Run::create(std::move(stream), merged_opt, stream_opt, RunMode::Async);
}

void Session::build_cached_source() {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  gst_init_once();

  enforce_caps_behavior(nodes_, "Session::build");
  enforce_source_run_mode(nodes_, "Session::build");
  enforce_sink_last(nodes_);

  require_element("appsink", "Session::build");
  require_element("identity", "Session::build");

  const uint64_t version = nodes_version_.load(std::memory_order_relaxed);
  if (built_ && built_->pipeline && built_->sink && built_version_ == version) {
    return;
  }

  if (built_ && built_->sink) {
    gst_object_unref(built_->sink);
  }
  if (built_ && built_->pipeline) {
    stop_and_unref(built_->pipeline);
  }
  built_.reset();

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_RUN_INSERT_BOUNDARIES", false);

  std::vector<std::shared_ptr<Node>> build_nodes =
      session_build_materialize_model_bound_nodes(nodes_, false);
  session_build_apply_derived_input_contracts(&build_nodes);
  const NameTransform name_transform = make_name_transform(opt_);
  BuildResult br =
      build_pipeline_full(build_nodes, insert_boundaries, "mysink", false, name_transform, &opt_);
  maybe_compile_source_contracts(&br, build_nodes, opt_, "Session::build");
  last_pipeline_ = br.pipeline_string;
  session_build_enforce_mla_num_buffers(last_pipeline_, "Session::build");
  session_build_maybe_dump_pipeline_string(last_pipeline_, "build_cached_source");

  GstElement* pipeline = session_build_parse_pipeline_or_throw(br, "Session::build");
  session_build_dump_pipeline_element_properties(pipeline);

  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline, br);
  }

  attach_boundary_probes(pipeline, br.diag);
  attach_stage_timing_probes(pipeline, br.diag);
  attach_element_timing_probes(pipeline, br.diag);
  attach_element_flow_probes(pipeline, br.diag);
  session_build_attach_debug_detess_input_probes(pipeline);
  session_build_attach_debug_appsink_probes(pipeline);
  session_build_attach_debug_all_buffer_probes(pipeline);
  session_build_attach_debug_element_buffer_probes(pipeline);
  session_build_attach_boxdecode_debug_probes(pipeline);

  const std::string appsink_name = apply_name_transform(name_transform, "mysink");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), appsink_name.c_str());
  if (!sink) {
    maybe_dump_dot(pipeline, "build_missing_mysink");
    stop_and_unref(pipeline);
    session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                             "Session::build: appsink '" + appsink_name +
                                                 "' not found.\nPipeline:\n" + last_pipeline_,
                                             "Add Output() as the last node.", last_pipeline_);
  }

  try {
    set_state_or_throw(pipeline, GST_STATE_PAUSED, "Session::build", br.diag);
  } catch (...) {
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    throw;
  }

  built_ = std::make_unique<BuiltState>();
  built_->pipeline = pipeline;
  built_->sink = sink;
  built_->diag = br.diag;
  built_version_ = version;
}

std::string Session::describe_backend(bool insert_boundaries) const {
  const NameTransform name_transform = make_name_transform(opt_);
  std::vector<std::shared_ptr<Node>> build_nodes =
      session_build_materialize_model_bound_nodes(nodes_, false);
  session_build_apply_derived_input_contracts(&build_nodes);
  BuildResult br =
      build_pipeline_full(build_nodes, insert_boundaries, "mysink", false, name_transform, &opt_);
  return br.pipeline_string;
}

} // namespace simaai::neat

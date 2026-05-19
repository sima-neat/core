/**
 * @file SessionValidate.cpp
 * @brief Session validation helpers.
 *
 * This is a mechanical split from the original monolithic Session.cpp.
 * No behavior is intended to change.
 */
#include "pipeline/Session.h"
#include "SessionDetail.h"
#include "internal/SessionBuildInternal.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/SessionError.h"
#include "pipeline/SessionReport.h"
#include "pipeline/ErrorCodes.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "builder/OutputSpec.h"
#include "builder/GraphPrinter.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

namespace {

std::string pipeline_snippet(const std::string& pipeline, std::size_t max_chars = 220) {
  if (pipeline.empty())
    return "<empty>";
  std::string out = pipeline.substr(0, std::min(max_chars, pipeline.size()));
  for (char& c : out) {
    if (c == '\n' || c == '\r' || c == '\t')
      c = ' ';
  }
  if (out.size() < pipeline.size())
    out += "...";
  return out;
}

std::string format_validate_note(const std::string& where, const std::string& code,
                                 const std::string& summary, const std::string& details,
                                 const std::string& hint = {}) {
  std::ostringstream oss;
  oss << "where=" << where << " code=" << code << " summary=" << summary;
  if (!details.empty()) {
    oss << " details=" << details;
  }
  if (!hint.empty()) {
    oss << "\nHint: " << hint;
  }
  return oss.str();
}

int resolve_validate_timeout_ms(const ValidateOptions& opt, int default_timeout_ms) {
  (void)opt;
  return std::max(10, env_int("SIMA_GST_VALIDATE_TIMEOUT_MS", default_timeout_ms));
}

} // namespace

SessionReport Session::validate(const ValidateOptions& opt) const {
  gst_init_once();

  enforce_caps_behavior(nodes_, "Session::validate");

  SessionReport rep;

  try {
    throw_if_input_appsrc_present(nodes_, "Session::validate");
  } catch (const SessionError& e) {
    rep.pipeline_string = "<input required>";
    rep.error_code = e.report().error_code;
    rep.repro_note = e.what();
    return rep;
  } catch (const std::exception& e) {
    rep.pipeline_string = "<input required>";
    rep.error_code = error_codes::kPipelineShape;
    rep.repro_note = e.what();
    return rep;
  }

  {
    const std::vector<std::shared_ptr<Node>> validate_nodes =
        session_build_materialize_model_bound_nodes(nodes_, false);
    NodeGroup group(validate_nodes);
    ValidationContext ctx;
    ctx.mode = ValidationContext::Mode::Validate;
    ctx.strict = true;

    ContractRegistry reg;
    reg.add(validators::NonEmptyPipeline());
    reg.add(validators::NoNullNodes());

    ValidationReport vrep = reg.validate(group, ctx);
    if (vrep.has_errors()) {
      rep.pipeline_string = "<builder-validation failed>";
      rep.error_code = error_codes::kPipelineShape;
      rep.repro_note = "validate: contract checks failed.\n" + vrep.to_string();
      return rep;
    }
  }

  if (!opt.parse_launch) {
    rep.pipeline_string = "<parse_launch disabled>";
    rep.repro_note = "validate(parse_launch=false): skipped.";
    return rep;
  }

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_VALIDATE_INSERT_BOUNDARIES", true);

  const std::vector<std::shared_ptr<Node>> validate_nodes =
      session_build_materialize_model_bound_nodes(nodes_, false);
  const NameTransform name_transform = make_name_transform(opt_);
  BuildResult br = build_pipeline_full(validate_nodes, insert_boundaries, "mysink", false,
                                       name_transform, &opt_);
  {
    ContractCompileInput compile_input;
    compile_input.pipeline_label = "Session::validate";
    compile_input.processcvu_requested_run_target = opt_.processcvu_requested_run_target;
    compile_input.processcvu = opt_.processcvu;
    session_build_compile_contracts(&br, validate_nodes, compile_input, "Session::validate",
                                    nullptr);
  }
  rep.pipeline_string = br.pipeline_string;
  enforce_mla_pipeline_guard("Session::validate", rep.pipeline_string, this);

  GstElement* pipeline = nullptr;
  try {
    pipeline = session_build_parse_pipeline_or_throw(br, "Session::validate");
  } catch (const SessionError& e) {
    rep.error_code = error_codes::kParseLaunch;
    rep.repro_note =
        format_validate_note("Session::validate", rep.error_code, "gst_parse_launch failed",
                             "gst_error='" + std::string(e.what()) + "' pipeline_snippet='" +
                                 pipeline_snippet(rep.pipeline_string) + "'",
                             "Validate fragment syntax and plugin availability (gst-inspect-1.0).");
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + rep.pipeline_string + "'";
    return rep;
  } catch (const std::exception& e) {
    rep.error_code = error_codes::kParseLaunch;
    rep.repro_note =
        format_validate_note("Session::validate", rep.error_code, "gst_parse_launch failed",
                             "gst_error='" + std::string(e.what()) + "' pipeline_snippet='" +
                                 pipeline_snippet(rep.pipeline_string) + "'",
                             "Validate fragment syntax and plugin availability (gst-inspect-1.0).");
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + rep.pipeline_string + "'";
    return rep;
  }

  attach_boundary_probes(pipeline, br.diag);
  attach_stage_timing_probes(pipeline, br.diag);
  attach_element_timing_probes(pipeline, br.diag);
  attach_element_flow_probes(pipeline, br.diag);

  if (opt.enforce_names) {
    enforce_names_contract(pipeline, br);
  }

  const std::string appsink_name = apply_name_transform(name_transform, "mysink");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), appsink_name.c_str());
  if (!sink) {
    maybe_dump_dot(pipeline, "validate_missing_mysink");
    rep.error_code = error_codes::kPipelineShape;
    rep.repro_note = format_validate_note("Session::validate", rep.error_code, "appsink not found",
                                          "element='" + appsink_name + "' pipeline_snippet='" +
                                              pipeline_snippet(br.pipeline_string) + "'",
                                          "Ensure Output() is the final node in this session.");
    stop_and_unref(pipeline);
    return rep;
  }

  try {
    set_state_or_throw(pipeline, GST_STATE_PAUSED, "Session::validate", br.diag);
  } catch (const SessionError& e) {
    rep = e.report();
    if (rep.error_code.empty()) {
      rep.error_code = error_codes::kCaps;
    }
    rep.repro_note = format_validate_note(
        "Session::validate", rep.error_code, "failed to PAUSE/preroll", rep.repro_note,
        "Inspect bus diagnostics and offending caps negotiation near the first GST ERROR.");
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }

  const int timeout_ms = resolve_validate_timeout_ms(opt, 2000);

  GstSample* sample = nullptr;
#if GST_CHECK_VERSION(1, 10, 0)
  sample = gst_app_sink_try_pull_preroll(GST_APP_SINK(sink), (guint64)timeout_ms * GST_MSECOND);
#else
  sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), (guint64)timeout_ms * GST_MSECOND);
#endif
  if (sample)
    gst_sample_unref(sample);

  // snapshot_basic already snapshots boundaries atomics -> BoundaryFlowStats.
  rep = br.diag->snapshot_basic();
  rep.pipeline_string = br.pipeline_string;

  rep.repro_note =
      sample ? "validate: preroll OK (PAUSED)."
             : "validate: preroll timed out in PAUSED (live source or negotiation stall).";
  rep.error_code = sample ? "" : std::string(error_codes::kRuntimePull);
  rep.repro_note += "\n" + boundary_summary_local(br.diag);

  gst_object_unref(sink);
  stop_and_unref(pipeline);
  return rep;
}

SessionReport Session::validate(const ValidateOptions& opt, const cv::Mat& input) const {
  gst_init_once();

  enforce_caps_behavior(nodes_, "Session::validate(input)");

  SessionReport rep;

  const Input* src_node = nullptr;
  try {
    require_input_appsrc(nodes_, "Session::validate(input)", &src_node);
  } catch (const SessionError& e) {
    rep.pipeline_string = "<input required>";
    rep.error_code = e.report().error_code;
    rep.repro_note = e.what();
    return rep;
  } catch (const std::exception& e) {
    rep.pipeline_string = "<input required>";
    rep.error_code = error_codes::kPipelineShape;
    rep.repro_note = e.what();
    return rep;
  }

  {
    const std::vector<std::shared_ptr<Node>> validate_nodes =
        session_build_materialize_model_bound_nodes(nodes_, true);
    NodeGroup group(validate_nodes);
    ValidationContext ctx;
    ctx.mode = ValidationContext::Mode::Validate;
    ctx.strict = true;

    ContractRegistry reg;
    reg.add(validators::NonEmptyPipeline());
    reg.add(validators::NoNullNodes());

    ValidationReport vrep = reg.validate(group, ctx);
    if (vrep.has_errors()) {
      rep.pipeline_string = "<builder-validation failed>";
      rep.error_code = error_codes::kPipelineShape;
      rep.repro_note = "validate(input): contract checks failed.\n" + vrep.to_string();
      return rep;
    }
  }

  if (!opt.parse_launch) {
    rep.pipeline_string = "<parse_launch disabled>";
    rep.repro_note = "validate(input, parse_launch=false): skipped.";
    return rep;
  }

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_VALIDATE_INSERT_BOUNDARIES", true);

  const std::vector<std::shared_ptr<Node>> validate_nodes =
      session_build_materialize_model_bound_nodes(nodes_, true);
  const NameTransform name_transform = make_name_transform(opt_);
  BuildResult br = build_pipeline_full(validate_nodes, insert_boundaries, "mysink", false,
                                       name_transform, &opt_);
  {
    ContractCompileInput compile_input;
    compile_input.pipeline_label = "Session::validate(input)";
    compile_input.processcvu_requested_run_target = opt_.processcvu_requested_run_target;
    compile_input.processcvu = opt_.processcvu;
    session_build_compile_contracts(&br, validate_nodes, compile_input, "Session::validate(input)",
                                    nullptr);
  }
  rep.pipeline_string = br.pipeline_string;
  enforce_mla_pipeline_guard("Session::validate(input)", rep.pipeline_string, this);

  GstElement* pipeline = nullptr;
  try {
    pipeline = session_build_parse_pipeline_or_throw(br, "Session::validate(input)");
  } catch (const SessionError& e) {
    rep.error_code = error_codes::kParseLaunch;
    rep.repro_note =
        format_validate_note("Session::validate(input)", rep.error_code, "gst_parse_launch failed",
                             "gst_error='" + std::string(e.what()) + "' pipeline_snippet='" +
                                 pipeline_snippet(rep.pipeline_string) + "'",
                             "Validate fragment syntax and plugin availability (gst-inspect-1.0).");
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + rep.pipeline_string + "'";
    return rep;
  } catch (const std::exception& e) {
    rep.error_code = error_codes::kParseLaunch;
    rep.repro_note =
        format_validate_note("Session::validate(input)", rep.error_code, "gst_parse_launch failed",
                             "gst_error='" + std::string(e.what()) + "' pipeline_snippet='" +
                                 pipeline_snippet(rep.pipeline_string) + "'",
                             "Validate fragment syntax and plugin availability (gst-inspect-1.0).");
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + rep.pipeline_string + "'";
    return rep;
  }

  attach_boundary_probes(pipeline, br.diag);
  attach_stage_timing_probes(pipeline, br.diag);
  attach_element_timing_probes(pipeline, br.diag);
  attach_element_flow_probes(pipeline, br.diag);

  if (opt.enforce_names) {
    enforce_names_contract(pipeline, br);
  }

  const std::string appsink_name = apply_name_transform(name_transform, "mysink");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), appsink_name.c_str());
  if (!sink) {
    maybe_dump_dot(pipeline, "validate_input_missing_mysink");
    rep.error_code = error_codes::kPipelineShape;
    rep.repro_note =
        format_validate_note("Session::validate(input)", rep.error_code, "appsink not found",
                             "element='" + appsink_name + "' pipeline_snippet='" +
                                 pipeline_snippet(br.pipeline_string) + "'",
                             "Ensure Output() is the final node in this session.");
    stop_and_unref(pipeline);
    return rep;
  }

  configure_appsink_for_input(sink);

  const std::string appsrc_name = apply_name_transform(name_transform, "mysrc");
  GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), appsrc_name.c_str());
  if (!appsrc) {
    rep.error_code = error_codes::kPipelineShape;
    rep.repro_note =
        format_validate_note("Session::validate(input)", rep.error_code, "appsrc not found",
                             "element='" + appsrc_name + "' pipeline_snippet='" +
                                 pipeline_snippet(br.pipeline_string) + "'",
                             "Ensure Input() is the first node in this session.");
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }

  const SampleSpec spec = infer_input_spec(src_node->options(), input, "Session::validate(input)");
  InputOptions src_opt = src_node->options();
  src_opt.max_bytes = resolve_appsrc_max_bytes(src_opt, spec);

  GstCaps* caps = caps_from_spec(spec);
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
  gst_caps_unref(caps);

  configure_appsrc(appsrc, src_opt);

  try {
    set_state_or_throw(pipeline, GST_STATE_PAUSED, "Session::validate(input)", br.diag);
  } catch (const SessionError& e) {
    rep = e.report();
    if (rep.error_code.empty()) {
      rep.error_code = error_codes::kCaps;
    }
    rep.repro_note = format_validate_note(
        "Session::validate(input)", rep.error_code, "failed to PAUSE/preroll", rep.repro_note,
        "Inspect bus diagnostics and offending caps negotiation near the first GST ERROR.");
    gst_object_unref(appsrc);
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }

  cv::Mat contiguous = input;
  if (!contiguous.isContinuous()) {
    contiguous = input.clone();
  }

  InputBufferPoolGuard pool_guard;
  GstBuffer* buf = allocate_input_buffer(spec.required_bytes_actual, src_opt, pool_guard);
  if (!buf) {
    rep.error_code = error_codes::kInputShape;
    rep.repro_note = format_validate_note(
        "Session::validate(input)", rep.error_code, "failed to allocate GstBuffer",
        "required_bytes=" + std::to_string(spec.required_bytes_actual),
        "Increase max_input_bytes/pool sizing or reduce input resolution.");
    gst_object_unref(appsrc);
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }

  GstMapInfo mi;
  if (!gst_buffer_map(buf, &mi, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    rep.error_code = error_codes::kInputShape;
    rep.repro_note =
        format_validate_note("Session::validate(input)", rep.error_code, "failed to map GstBuffer",
                             "required_bytes=" + std::to_string(spec.required_bytes_actual),
                             "Check allocator compatibility and GstBuffer memory type.");
    gst_object_unref(appsrc);
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }
  const size_t contiguous_bytes =
      static_cast<size_t>(contiguous.step[0]) * static_cast<size_t>(contiguous.rows);
  if (spec.required_bytes_actual > contiguous_bytes) {
    gst_buffer_unmap(buf, &mi);
    gst_buffer_unref(buf);
    rep.error_code = error_codes::kInputShape;
    std::ostringstream oss;
    oss << "required_bytes=" << spec.required_bytes_actual << " actual_bytes=" << contiguous_bytes
        << " width=" << contiguous.cols << " height=" << contiguous.rows
        << " channels=" << contiguous.channels();
    rep.repro_note = format_validate_note(
        "Session::validate(input)", rep.error_code, "input frame smaller than required bytes",
        oss.str(), "Ensure InputOptions format/size match the provided cv::Mat.");
    gst_object_unref(appsrc);
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }
  std::memcpy(mi.data, contiguous.data, spec.required_bytes_actual);
  gst_buffer_unmap(buf, &mi);
  if (spec.kind == SampleMediaKind::RawVideo) {
    std::string meta_err;
    GstBuffer* meta_buf = buf;
    if (!pipeline_internal::attach_video_meta(&meta_buf, spec, &meta_err)) {
      gst_buffer_unref(buf);
      rep.error_code = error_codes::kCaps;
      rep.repro_note = format_validate_note(
          "Session::validate(input)", rep.error_code, "failed to attach video meta",
          "detail='" + meta_err + "'", "Check video caps/format and plane layout compatibility.");
      gst_object_unref(appsrc);
      gst_object_unref(sink);
      stop_and_unref(pipeline);
      return rep;
    }
    buf = meta_buf;
  }
  if (src_opt.use_simaai_pool && !maybe_add_simaai_meta(buf, next_input_frame_id(), src_opt)) {
    gst_buffer_unref(buf);
    rep.error_code = error_codes::kCaps;
    rep.repro_note = format_validate_note(
        "Session::validate(input)", rep.error_code, "failed to attach GstSimaMeta",
        "use_simaai_pool=true", "Verify SimaAI allocator/buffer-pool runtime availability.");
    gst_object_unref(appsrc);
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }

  const GstFlowReturn push_ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buf);
  if (push_ret != GST_FLOW_OK) {
    gst_buffer_unref(buf);
    const char* flow_name = gst_flow_get_name(push_ret);
    std::ostringstream details;
    details << "flow=" << static_cast<int>(push_ret) << ":"
            << (flow_name ? flow_name : "<unknown>");
    std::string hint = "Inspect SessionReport bus diagnostics for the first terminal GST error.";
    if (push_ret == GST_FLOW_FLUSHING || push_ret == GST_FLOW_EOS) {
      hint = "Pipeline may already be stopping or at EOS; verify lifecycle ordering.";
    }
    rep.error_code = error_codes::kCaps;
    rep.repro_note = format_validate_note("Session::validate(input)", rep.error_code,
                                          "appsrc push failed", details.str(), hint);
    gst_object_unref(appsrc);
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }
  gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

  const int timeout_ms = resolve_validate_timeout_ms(opt, 10000);

  GstSample* sample = nullptr;
#if GST_CHECK_VERSION(1, 10, 0)
  sample = gst_app_sink_try_pull_preroll(GST_APP_SINK(sink), (guint64)timeout_ms * GST_MSECOND);
#else
  sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), (guint64)timeout_ms * GST_MSECOND);
#endif
  if (sample)
    gst_sample_unref(sample);

  rep = br.diag->snapshot_basic();
  rep.pipeline_string = br.pipeline_string;

  rep.repro_note = sample ? "validate(input): preroll OK (PAUSED)."
                          : "validate(input): preroll timed out in PAUSED.";
  rep.error_code = sample ? "" : std::string(error_codes::kRuntimePull);

  gst_object_unref(appsrc);
  gst_object_unref(sink);
  stop_and_unref(pipeline);
  return rep;
}

} // namespace simaai::neat

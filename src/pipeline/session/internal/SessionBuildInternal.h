#pragma once

#include "../SessionDetail.h"

#include "internal/InputStream.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

std::string session_build_decorate_with_error_code(const std::string& code,
                                                   const std::string& message);

[[noreturn]] void session_build_throw_session_error_simple(const std::string& code,
                                                           const std::string& message,
                                                           const std::string& hint = {},
                                                           const std::string& pipeline = {});

void session_build_drain_bus_into_diag(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag);
void session_build_throw_if_bus_error_local(GstElement* pipeline,
                                            const std::shared_ptr<DiagCtx>& diag,
                                            const char* where);
void session_build_dump_flow_snapshot(const std::shared_ptr<DiagCtx>& diag, const char* label);
void session_build_dump_pipeline_string_force(const std::shared_ptr<DiagCtx>& diag,
                                              const char* label);

RunOptions session_build_apply_run_defaults(const RunOptions& opt, const SessionOptions& sess_opt);
RunOptions session_build_resolve_build_opt(RunMode mode, const RunOptions& opt);
bool session_build_should_insert_async_queue2(RunMode mode, const RunOptions& opt);
InputStreamOptions session_build_make_stream_options(const RunOptions& opt, RunMode mode);
void session_build_maybe_enable_rtsp_appsink_drop(InputStreamOptions& stream_opt,
                                                  const std::vector<std::shared_ptr<Node>>& nodes);
void session_build_maybe_apply_detess_output_override(
    const std::vector<std::shared_ptr<Node>>& nodes, InputStreamOptions& stream_opt);
void session_build_maybe_dump_pipeline_string(const std::string& pipeline, const char* label);
std::string session_build_manifest_json_for_pipeline(const std::string& pipeline_string);
std::string session_build_maybe_force_model_num_buffers(std::string pipeline);
std::string session_build_clamp_sync_pipeline(std::string pipeline, int num_buffers_override);
std::string session_build_clamp_detess_num_buffers(std::string pipeline, int num_buffers_override);
std::uint64_t session_build_estimate_frame_bytes_limit(const InputOptions& opt,
                                                       const SampleSpec& spec);

bool session_build_has_output_appsink(const std::vector<std::shared_ptr<Node>>& nodes);

GstElement* session_build_parse_pipeline_or_throw(const std::string& pipeline_string,
                                                  const char* where);
void session_build_dump_pipeline_element_properties(GstElement* pipeline);
void session_build_attach_debug_detess_input_probes(GstElement* pipeline);
void session_build_attach_debug_appsink_probes(GstElement* pipeline);
void session_build_attach_debug_all_buffer_probes(GstElement* pipeline);
void session_build_attach_debug_element_buffer_probes(GstElement* pipeline);
void session_build_attach_boxdecode_debug_probes(GstElement* pipeline);
void session_build_attach_rtsp_debug(GstElement* pipeline,
                                     const std::vector<std::shared_ptr<Node>>& nodes,
                                     const NameTransform& name_transform);
void session_build_attach_h264_caps_fixups(GstElement* pipeline,
                                           const std::vector<std::shared_ptr<Node>>& nodes,
                                           const NameTransform& name_transform);

void session_build_enforce_mla_num_buffers(const std::string& pipeline, const char* context,
                                           bool allow_one = false);

void session_build_configure_appsink_allocation_preference(
    GstElement* appsink, const std::vector<std::shared_ptr<Node>>& nodes);
void session_build_configure_appsink_for_input_stream(GstElement* appsink,
                                                      const InputStreamOptions& opt);

InputOptions session_build_resolve_appsrc_options(const InputOptions& opt,
                                                  const NameTransform& name_transform);
SampleSpec session_build_make_placeholder_spec();

struct BuildInputContext {
  RunMode mode = RunMode::Async;
  RunOptions merged_opt{};
  const Input* src_node = nullptr;
  InputStreamOptions stream_opt{};
  bool insert_queue2 = false;
  int sync_num_buffers_override = -1;
  NameTransform name_transform{};
};

BuildInputContext
session_build_prepare_build_input_context(const std::vector<std::shared_ptr<Node>>& nodes,
                                          const SessionOptions& sess_opt, RunMode mode,
                                          const RunOptions& opt);

InputStream session_build_run_input_stream_internal(
    const std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<void>& guard,
    const void* owner, std::string& last_pipeline, const cv::Mat& sample,
    const InputStreamOptions& opt, const NameTransform& name_transform, bool insert_queue2,
    int sync_num_buffers_override, bool sync_mode);

InputStream session_build_run_input_stream_internal(
    const std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<void>& guard,
    const void* owner, std::string& last_pipeline, const simaai::neat::Tensor& sample,
    const InputStreamOptions& opt, const NameTransform& name_transform, bool insert_queue2,
    int sync_num_buffers_override, bool sync_mode);

InputStream session_build_run_input_stream_internal(
    const std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<void>& guard,
    const void* owner, std::string& last_pipeline, const Sample& sample,
    const InputStreamOptions& opt, const NameTransform& name_transform, bool insert_queue2,
    int sync_num_buffers_override, bool sync_mode);

} // namespace simaai::neat

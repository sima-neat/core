/**
 * @file SessionBuildBridge.cpp
 * @brief Bridge wrappers exposing SessionBuild internals to split translation units.
 */

#include "SessionDetail.h"
#include "internal/SessionBuildInternal.h"

#include "pipeline/SessionError.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/sima/SimaPluginStaticManifestResolver.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

// Implemented in SessionBuild.cpp.
void drain_bus_into_diag(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag);
void throw_if_bus_error_local(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
                              const char* where);
void dump_flow_snapshot(const std::shared_ptr<DiagCtx>& diag, const char* label);
void dump_pipeline_string_force(const std::shared_ptr<DiagCtx>& diag, const char* label);
void maybe_dump_pipeline_string(const std::string& pipeline, const char* label);
std::string maybe_force_model_num_buffers(std::string pipeline);
std::string clamp_sync_pipeline(std::string pipeline, int num_buffers_override);
std::string clamp_detess_num_buffers(std::string pipeline, int num_buffers_override);
std::uint64_t estimate_frame_bytes_limit(const InputOptions& opt, const SampleSpec& spec);
bool has_output_appsink(const std::vector<std::shared_ptr<Node>>& nodes);
void dump_pipeline_element_properties(GstElement* pipeline);
void attach_debug_detess_input_probes(GstElement* pipeline);
void attach_debug_appsink_probes(GstElement* pipeline);
void attach_debug_all_buffer_probes(GstElement* pipeline);
void attach_debug_element_buffer_probes(GstElement* pipeline);
void attach_boxdecode_debug_probes(GstElement* pipeline);
void enforce_mla_num_buffers(const std::string& pipeline, const char* context, bool allow_one);
void configure_appsink_for_input_stream(GstElement* appsink, const InputStreamOptions& opt);
void session_build_configure_appsink_allocation_preference_internal(
    GstElement* appsink, const std::vector<std::shared_ptr<Node>>& nodes);
InputOptions resolve_appsrc_options(const InputOptions& opt, const NameTransform& name_transform);
SampleSpec make_placeholder_spec();

std::string session_build_decorate_with_error_code(const std::string& code,
                                                   const std::string& message) {
  return pipeline_internal::error_util::decorate_error(code, message);
}

[[noreturn]] void session_build_throw_session_error_simple(const std::string& code,
                                                           const std::string& message,
                                                           const std::string& hint,
                                                           const std::string& pipeline) {
  SessionReport rep = pipeline_internal::error_util::make_report(code, message, pipeline, hint);
  throw SessionError(session_build_decorate_with_error_code(rep.error_code, rep.repro_note),
                     std::move(rep));
}

void session_build_drain_bus_into_diag(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag) {
  drain_bus_into_diag(pipeline, diag);
}

void session_build_throw_if_bus_error_local(GstElement* pipeline,
                                            const std::shared_ptr<DiagCtx>& diag,
                                            const char* where) {
  throw_if_bus_error_local(pipeline, diag, where);
}

void session_build_dump_flow_snapshot(const std::shared_ptr<DiagCtx>& diag, const char* label) {
  dump_flow_snapshot(diag, label);
}

void session_build_dump_pipeline_string_force(const std::shared_ptr<DiagCtx>& diag,
                                              const char* label) {
  dump_pipeline_string_force(diag, label);
}

void session_build_maybe_dump_pipeline_string(const std::string& pipeline, const char* label) {
  maybe_dump_pipeline_string(pipeline, label);
}

std::string session_build_manifest_json_for_pipeline(const std::string& pipeline_string) {
  using namespace simaai::neat::pipeline_internal::sima;
  ManifestBuildDiagnostics diag;
  const SimaPluginStaticManifest manifest =
      resolve_manifest_from_pipeline(pipeline_string, /*session_id=*/"", &diag);
  return serialize_manifest_json(manifest);
}

std::string session_build_maybe_force_model_num_buffers(std::string pipeline) {
  return maybe_force_model_num_buffers(std::move(pipeline));
}

std::string session_build_clamp_sync_pipeline(std::string pipeline, int num_buffers_override) {
  return clamp_sync_pipeline(std::move(pipeline), num_buffers_override);
}

std::string session_build_clamp_detess_num_buffers(std::string pipeline, int num_buffers_override) {
  return clamp_detess_num_buffers(std::move(pipeline), num_buffers_override);
}

std::uint64_t session_build_estimate_frame_bytes_limit(const InputOptions& opt,
                                                       const SampleSpec& spec) {
  return estimate_frame_bytes_limit(opt, spec);
}

bool session_build_has_output_appsink(const std::vector<std::shared_ptr<Node>>& nodes) {
  return has_output_appsink(nodes);
}

void session_build_dump_pipeline_element_properties(GstElement* pipeline) {
  dump_pipeline_element_properties(pipeline);
}

void session_build_attach_debug_detess_input_probes(GstElement* pipeline) {
  attach_debug_detess_input_probes(pipeline);
}

void session_build_attach_debug_appsink_probes(GstElement* pipeline) {
  attach_debug_appsink_probes(pipeline);
}

void session_build_attach_debug_all_buffer_probes(GstElement* pipeline) {
  attach_debug_all_buffer_probes(pipeline);
}

void session_build_attach_debug_element_buffer_probes(GstElement* pipeline) {
  attach_debug_element_buffer_probes(pipeline);
}

void session_build_attach_boxdecode_debug_probes(GstElement* pipeline) {
  attach_boxdecode_debug_probes(pipeline);
}

void session_build_enforce_mla_num_buffers(const std::string& pipeline, const char* context,
                                           bool allow_one) {
  enforce_mla_num_buffers(pipeline, context, allow_one);
}

void session_build_configure_appsink_allocation_preference(
    GstElement* appsink, const std::vector<std::shared_ptr<Node>>& nodes) {
  session_build_configure_appsink_allocation_preference_internal(appsink, nodes);
}

void session_build_configure_appsink_for_input_stream(GstElement* appsink,
                                                      const InputStreamOptions& opt) {
  configure_appsink_for_input_stream(appsink, opt);
}

InputOptions session_build_resolve_appsrc_options(const InputOptions& opt,
                                                  const NameTransform& name_transform) {
  return resolve_appsrc_options(opt, name_transform);
}

SampleSpec session_build_make_placeholder_spec() {
  return make_placeholder_spec();
}

} // namespace simaai::neat

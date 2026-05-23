/**
 * @file GraphDetail.h
 * @brief Internal helpers used by the split Graph compilation units.
 *
 * This header is **private to src/pipeline** and exists only to share a small set of
 * implementation details between:
 * - SessionNaming.cpp
 * - SessionBuild.cpp
 * - SessionValidate.cpp
 * - SessionRtsp.cpp
 * - SessionIo.cpp
 *
 * It is a mechanical refactor from the original monolithic Graph.cpp:
 * no behavior is intended to change.
 *
 * @internal
 */

#pragma once

#include "pipeline/Graph.h"

#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"

#include <gst/gst.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

// =====================================================================================
// RAII handles for owning GStreamer resources held inside BuiltState.
//
// Single source of truth for teardown. The two deleters are stateless functors
// declared `noexcept`; they swallow any (extremely rare) exception from the
// underlying C++ wrappers so that destruction is always safe. Using them as
// the deleter type of `std::unique_ptr` means every reset/move/destroy of a
// BuiltState — including the implicit ones inside unique_ptr<BuiltState>'s
// own move-assignment and destructor — does the right thing automatically.
// =====================================================================================

// Forward decl from this file's own implementation TU (SessionBuild.cpp).
void stop_and_unref(GstElement*& e);

namespace pipeline_internal {

struct GstPipelineStopUnref {
  void operator()(GstElement* p) const noexcept {
    if (!p)
      return;
    // stop_and_unref takes its argument by reference and nulls it on return;
    // we pass a local copy so the deleter contract is independent of that
    // implementation detail.
    GstElement* local = p;
    try {
      ::simaai::neat::stop_and_unref(local);
    } catch (...) {
      // Deleters must never propagate exceptions — log and continue.
      std::fprintf(stderr, "[STOP] stop_and_unref threw during teardown; suppressed\n");
    }
  }
};

struct GstObjectUnrefDeleter {
  void operator()(GstElement* o) const noexcept {
    if (o)
      gst_object_unref(o);
  }
};

using GstPipelinePtr = std::unique_ptr<GstElement, GstPipelineStopUnref>;
using GstSinkPtr = std::unique_ptr<GstElement, GstObjectUnrefDeleter>;

} // namespace pipeline_internal

// =====================================================================================
// BuiltState — self-owning record of a materialised (not yet running) pipeline.
//
// Declaration order encodes the teardown contract (members are destroyed in
// REVERSE declaration order):
//   1. `sink`     destroyed first  — release the child element ref before…
//   2. `pipeline` destroyed second — …stopping and unref'ing the parent bin, while…
//   3. `diag`     destroyed last   — …diagnostics payload outlives both, since
//                                    bus/probe callbacks attached to the pipeline
//                                    may hold pointers into it.
// =====================================================================================
struct Graph::BuiltState {
  std::shared_ptr<void> diag;                 // outlives pipeline (probes index it)
  pipeline_internal::GstPipelinePtr pipeline; // stop-and-unref on destruction
  pipeline_internal::GstSinkPtr sink;         // gst_object_unref on destruction
};

// =====================================================================================
// CompositionGraph — private source of truth for top-level Graph composition.
//
// Phase 2 keeps the public surface linear (`Graph::add(...)`) while storing composition as
// vertices + edges. The existing GStreamer build path still consumes a linear snapshot, so
// this intentionally does not expose ports, public connect(), or runtime graph lowering yet.
// =====================================================================================
struct Graph::CompositionGraph {
  using VertexId = std::size_t;
  using NodePtr = std::shared_ptr<Node>;

  static constexpr VertexId kInvalid = static_cast<VertexId>(-1);

  std::vector<NodePtr> vertices;
  std::vector<CompositionEdge> edges;
  std::vector<runtime::FragmentPlan> fragments;
  std::vector<NamedFragment> named_fragments;
  VertexId tail = kInvalid;
  bool endpoint_mode = false;

  struct ImportedFragment {
    std::uint64_t source_version = 0;
    VertexId start = kInvalid;
    VertexId end = kInvalid;
  };
  std::unordered_map<std::uint64_t, ImportedFragment> imported_fragments;
  std::unordered_map<const Node*, ImportedFragment> imported_nodes;
  std::unordered_map<std::string, ImportedFragment> imported_models;

  bool empty() const noexcept {
    return vertices.empty();
  }

  std::size_t size() const noexcept {
    return vertices.size();
  }

  VertexId append_vertex(NodePtr node);
  std::pair<VertexId, VertexId> append_linear_fragment_copy(const std::vector<NodePtr>& nodes);
  void recompute_unique_tail() noexcept;
  void connect_runtime_port(VertexId from, VertexId to, std::string from_port, std::string to_port);
  void connect_endpoint(VertexId from, VertexId to, std::string from_endpoint,
                        std::string to_endpoint);
  std::pair<VertexId, VertexId> append_node(NodePtr node);
  bool is_linear() const noexcept;
  std::vector<NodePtr> linear_nodes_or_throw(const char* where) const;
};

// =====================================================================================
// BuildResult: from Node list → pipeline string + diag + sink name
// =====================================================================================

/** @brief Input kind used for Graph::RunCache fast-path caching. */
enum class RunInputKind {
  Mat = 0,
  Tensor,
  Sample,
};

/**
 * @brief Cache entry for fast-path Graph::run(input).
 *
 * Holds a prepared Run and the cap key used to configure appsrc, so
 * repeated calls with identical input specs can reuse the same runner.
 */
struct Graph::RunCache {
  Run runner;
  CapKey caps_key;
  RunOptions opt;
  uint64_t nodes_version = 0;
  RunInputKind input_kind = RunInputKind::Mat;
  bool sync_prefill_warmed = false;
};

// -------------------------------------------------------------------------------------
// Small helpers (shared internal utilities)
// -------------------------------------------------------------------------------------
using pipeline_internal::env_bool;
using pipeline_internal::env_int;
using pipeline_internal::env_str;
using pipeline_internal::lower_copy;
using pipeline_internal::sanitize_name;
using pipeline_internal::trim_copy;
using pipeline_internal::upper_copy;

/** Lightweight debug tracing hook (enabled by SIMA_PIPELINE_TRACE_STEP). */
void trace_step(const char* label);

// -------------------------------------------------------------------------------------
// Naming helpers (element name rewrite)
// -------------------------------------------------------------------------------------

/**
 * A node's pipeline fragment along with the element names contained within that fragment.
 *
 * Element name tracking is used for diagnostics/probes and "names contract" enforcement.
 */
struct NodeFragment {
  std::string fragment;
  std::vector<std::string> element_names;
};

/**
 * Rewrite element names inside a pipeline fragment.
 *
 * This is a targeted string rewrite used to avoid naming collisions when the fragment
 * contains explicit element names (e.g. "appsink name=mysink").
 */
std::string rewrite_fragment_names(const std::string& fragment,
                                   const std::unordered_map<std::string, std::string>& mapping);

/**
 * Build the pipeline fragment for a single Node, optionally applying NameTransform.
 *
 * @param node Node to convert to a pipeline fragment.
 * @param index Index used for stable generated names.
 * @param transform Optional name transform applied to all element names.
 */
NodeFragment make_node_fragment(const std::shared_ptr<Node>& node, int index,
                                const NameTransform& transform);

// -------------------------------------------------------------------------------------
// Build/diagnostics shared helpers
// -------------------------------------------------------------------------------------

/**
 * Optionally dumps a DOT graph to SIMA_PIPELINE_DOT_DIR when enabled via env.
 *
 * This is used on error paths to capture the pipeline state.
 */
void maybe_dump_dot(GstElement* pipeline, const std::string& tag);

/** Convenience wrapper: set element state to NULL and unref; sets pointer to null. */
void stop_and_unref(GstElement*& e);

/** Human-readable boundary summary used in validation errors. */
std::string boundary_summary_local(const std::shared_ptr<DiagCtx>& diag);

/**
 * Result of build_pipeline_full (pipeline string + diagnostics metadata).
 *
 * This remains an implementation detail (not part of the public Graph API).
 */
struct BuildResult {
  std::string pipeline_string;
  std::shared_ptr<DiagCtx> diag;
  std::string appsink_name; // mysink or tap_* (after session name transform)
  int tap_node_index = -1;
  NameTransform name_transform{};
  std::shared_ptr<CompiledPipelineContracts> compiled_contracts;
  std::optional<pipeline_internal::sima::SimaPluginStaticManifest> rendered_manifest;
  std::vector<std::string> model_source_paths;
  // Compile + render diagnostics from session_build_compile_contracts. Carried
  // forward so the wrapper throws in parse_pipeline_or_throw can include the
  // specific failure messages (which `render_manifest_from_compiled_contracts`
  // and `compile_node_contracts` push here on error) rather than a generic
  // "manifest is missing" message.
  pipeline_internal::sima::ManifestBuildDiagnostics manifest_diagnostics;
};

/** Select boundary insertion behavior based on run/build mode env. */
bool should_insert_boundaries_for_mode(const char* mode_key, bool def_val);

/** Attach boundary probes (buffer events crossing boundaries). */
void attach_boundary_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag);

/** Attach "stage timing" probes (internal perf metrics). */
void attach_stage_timing_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag);

/** Attach per-element timing probes (internal perf metrics). */
void attach_element_timing_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag);

/** Attach per-element flow probes (buffers/bytes, drops, etc.). */
void attach_element_flow_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag);

/**
 * Enforce a "names contract" after pipeline construction.
 *
 * Ensures every name that was referenced during build exists in the parsed pipeline,
 * and conversely, that we didn't accidentally rewrite away an expected name.
 */
void enforce_names_contract(GstElement* pipeline, const BuildResult& br);

/** Set a pipeline state and throw NeatError on failure. */
void set_state_or_throw(GstElement* pipeline, GstState target, const char* where,
                        const std::shared_ptr<DiagCtx>& diag);

// -------------------------------------------------------------------------------------
// Shared build/run validation helpers
// -------------------------------------------------------------------------------------

/**
 * Infer a SampleSpec for a cv::Mat input.
 *
 */
SampleSpec infer_input_spec(const InputOptions& opt, const cv::Mat& input, const char* where);

/**
 * Infer a SampleSpec for a simaai::neat::Tensor input.
 *
 */
SampleSpec infer_input_spec(const InputOptions& opt, const simaai::neat::Tensor& input,
                            const char* where);

/**
 * Infer a SampleSpec for a Sample input.
 *
 */
SampleSpec infer_input_spec(const InputOptions& opt, const Sample& sample, const char* where);

/**
 * Prevent a pipeline string from being reused by a different Graph instance.
 *
 * This is a guardrail for debugging/misuse; it is enforced when enabled by options.
 */
void enforce_mla_pipeline_guard(const char* where, const std::string& pipeline, const void* owner);

/** Require that the terminal sink is last in the node list. */
void enforce_sink_last(const std::vector<std::shared_ptr<Node>>& nodes);

/** Enforce per-node caps behavior constraints. */
void enforce_caps_behavior(const std::vector<std::shared_ptr<Node>>& nodes,
                           const std::string& where);

/** Enforce that a sink, if present, is last. */
void enforce_sink_last_if_present(const std::vector<std::shared_ptr<Node>>& nodes,
                                  const std::string& where);

/** Enforce that the node graph is valid for SOURCE run mode. */
void enforce_source_run_mode(const std::vector<std::shared_ptr<Node>>& nodes,
                             const std::string& where);

/** Enforce that the node graph is valid for PUSH run mode. */
void enforce_push_run_mode(const std::vector<std::shared_ptr<Node>>& nodes,
                           const std::string& where);

/** Throw if an Input node is present (used for SOURCE pipelines). */
void throw_if_input_appsrc_present(const std::vector<std::shared_ptr<Node>>& nodes,
                                   const std::string& where);

/** Locate the Input node or throw; returns a pointer to the node's options. */
void require_input_appsrc(const std::vector<std::shared_ptr<Node>>& nodes, const std::string& where,
                          const Input** out_src);

/** Configure appsrc element properties from InputOptions. */
void configure_appsrc(GstElement* appsrc, const InputOptions& opt);

/** Resolve an appropriate max-bytes setting for appsrc based on input spec. */
std::uint64_t resolve_appsrc_max_bytes(const InputOptions& opt, const SampleSpec& spec);

/** Configure appsink element properties for validation (sync/preroll). */
void configure_appsink_for_input(GstElement* appsink);

// -------------------------------------------------------------------------------------
// Core builder (pipeline string + diag)
// -------------------------------------------------------------------------------------

/**
 * Build a full pipeline string from nodes, optionally inserting boundaries and queue2.
 *
 * Returns a BuildResult that also includes diagnostics metadata (element name tracking).
 */
BuildResult build_pipeline_full(const std::vector<std::shared_ptr<Node>>& nodes,
                                bool insert_boundaries, const std::string& appsink_name,
                                bool insert_queue2, const NameTransform& name_transform,
                                const GraphOptions* sess_opt = nullptr);

} // namespace simaai::neat

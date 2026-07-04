#include "RunCore.h"

#include "ExecutionGraphPlan.h"
#include "builder/OutputSpec.h"
#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphReport.h"
#include "pipeline/NeatError.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/internal/BuildTiming.h"
#include "pipeline/internal/DecoderAdmissionClient.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace simaai::neat::graph {

bool graph_debug_enabled();
bool graph_push_fail_debug_enabled();
bool graph_sched_debug_enabled();
void graph_debug_sample(const char* tag, const Sample& sample);
void log_first_decoded_once(const Sample& sample, std::size_t segment_id);

bool has_input_appsrc(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
bool has_output_appsink(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
bool has_internal_source(std::span<const std::shared_ptr<simaai::neat::Node>> nodes);
void maybe_force_copy_for_backpressure(Sample& sample, std::size_t qsize, const char* where,
                                       std::size_t seg_id);
InputOptions input_opts_from_spec(const OutputSpec& spec, bool complete);
bool is_encoded_sample(const Sample& sample);
std::optional<Sample> sample_from_input_spec(const OutputSpec& spec, std::string* err);

} // namespace simaai::neat::graph

namespace simaai::neat::runtime {
namespace {

using simaai::neat::graph::kInvalidPort;
using simaai::neat::graph::NodeId;
using simaai::neat::graph::PortId;
using simaai::neat::graph::StageMsg;
using simaai::neat::graph::StageOutMsg;
using simaai::neat::pipeline_internal::env_bool;
using simaai::neat::pipeline_internal::env_int;

std::string gst_double_quote(std::string value) {
  std::string out;
  out.reserve(value.size() + 2U);
  out.push_back('"');
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

std::uint64_t elapsed_ns_since(std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
          .count());
}

void atomic_add_max(std::atomic<std::uint64_t>& total, std::atomic<std::uint64_t>& max_value,
                    std::uint64_t ns) {
  total.fetch_add(ns, std::memory_order_relaxed);
  std::uint64_t cur = max_value.load(std::memory_order_relaxed);
  while (cur < ns && !max_value.compare_exchange_weak(cur, ns, std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
  }
}

std::uint64_t edge_port_key(NodeId id, PortId port) {
  return (static_cast<std::uint64_t>(id) << 32) | static_cast<std::uint64_t>(port);
}

void configure_graph_public_output_loan_gate(const std::shared_ptr<RunCore>& core,
                                             const GraphRuntimeOptions& graph_options,
                                             RunMode mode) {
  if (!core) {
    return;
  }

  InputStreamOptions stream_opt =
      simaai::neat::session_build_make_stream_options(graph_options.pipeline, mode);
  stream_opt.public_output_contract = true;
  simaai::neat::session_build_finalize_public_zero_copy_holder_loan_credits(stream_opt);

  core->pipeline.stream_opt = stream_opt;
  core->pipeline.copy_output_latched.store(stream_opt.copy_output, std::memory_order_relaxed);
  core->holder_loan_gate =
      stream_opt.holder_loan_credits > 0
          ? std::make_shared<pipeline_internal::HolderLoanGate>(stream_opt.holder_loan_credits)
          : nullptr;
}

bool is_simple_linear_plan(const ExecutionGraphPlan& plan) {
  return plan.linear_compat && plan.pipeline_segments.size() == 1U && plan.stage_nodes.empty() &&
         plan.edges.empty() && plan.named_inputs.empty() && plan.named_outputs.empty();
}

std::size_t runtime_node_count(const ExecutionGraphPlan& plan) {
  std::size_t count = plan.node_labels.size();
  const auto bump = [&count](NodeId id) {
    if (id != simaai::neat::graph::kInvalidNode) {
      count = std::max(count, static_cast<std::size_t>(id) + 1U);
    }
  };
  for (const auto& seg : plan.pipeline_segments) {
    for (NodeId id : seg.node_ids) {
      bump(id);
    }
  }
  for (const auto& stage : plan.stage_nodes) {
    bump(stage.node_id);
  }
  for (const auto& edge : plan.edges) {
    bump(edge.from);
    bump(edge.to);
  }
  return count;
}

bool decoder_plan_debug_enabled() {
  return env_bool("SIMA_DECODER_PLAN_DEBUG", false) ||
         env_bool("SIMA_DECODER_ADMISSION_DEBUG", false);
}

const char* sima_decode_type_debug_name(simaai::neat::SimaDecodeType type) {
  switch (type) {
  case simaai::neat::SimaDecodeType::H264:
    return "h264";
  case simaai::neat::SimaDecodeType::JPEG:
    return "jpeg";
  case simaai::neat::SimaDecodeType::MJPEG:
    return "mjpeg";
  }
  return "unknown";
}

void dump_decoder_plan_debug(const ExecutionGraphRuntime& execution) {
  if (!decoder_plan_debug_enabled()) {
    return;
  }

  std::size_t count = 0;
  for (const auto& runtime : execution.pipelines) {
    if (!runtime) {
      continue;
    }
    const auto& seg = runtime->seg;
    for (std::size_t i = 0; i < runtime->nodes.size(); ++i) {
      const auto& node = runtime->nodes[i];
      const auto* dec = dynamic_cast<const simaai::neat::SimaDecode*>(node.get());
      if (!dec) {
        continue;
      }
      ++count;
      simaai::neat::graph::NodeId runtime_node = simaai::neat::graph::kInvalidNode;
      if (i < seg.materialized_node_attribution.size()) {
        runtime_node = seg.materialized_node_attribution[i].runtime_node;
      } else if (i < seg.node_ids.size()) {
        runtime_node = seg.node_ids[i];
      }
      std::string label = "<unknown>";
      if (runtime_node != simaai::neat::graph::kInvalidNode &&
          static_cast<std::size_t>(runtime_node) < execution.node_labels.size() &&
          !execution.node_labels[static_cast<std::size_t>(runtime_node)].empty()) {
        label = execution.node_labels[static_cast<std::size_t>(runtime_node)];
      }
      const auto& opt = dec->options();
      std::fprintf(
          stderr,
          "[DECPLAN] seg=%zu mat_index=%zu runtime_node=%lld label=%s "
          "source_like=%d needs_input=%d needs_output=%d graph_internal_output=%d "
          "type=%s decoder=%s raw_output=%d next=%s width=%d height=%d fps=%d "
          "num_buffers=%d\n",
          static_cast<std::size_t>(seg.id), i, static_cast<long long>(runtime_node), label.c_str(),
          seg.boundary.source_like ? 1 : 0, seg.boundary.needs_input ? 1 : 0,
          seg.boundary.needs_output ? 1 : 0, seg.boundary.graph_internal_output ? 1 : 0,
          sima_decode_type_debug_name(opt.type),
          opt.decoder_name.empty() ? "<auto>" : opt.decoder_name.c_str(), opt.raw_output ? 1 : 0,
          opt.next_element.empty() ? "<default>" : opt.next_element.c_str(), opt.dec_width,
          opt.dec_height, opt.dec_fps, opt.num_buffers);
    }
  }

  std::fprintf(stderr, "[DECPLAN] total_decoders=%zu segments=%zu stages=%zu edges=%zu\n", count,
               execution.pipelines.size(), execution.stage_groups.size(),
               execution.plan.edges.size());
}

struct DecoderAdmissionCandidate {
  std::size_t pipeline_index = 0;
  std::size_t node_index = 0;
  simaai::neat::graph::NodeId runtime_node = simaai::neat::graph::kInvalidNode;
  simaai::neat::SimaDecodeOptions options;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 1;
  bool fused_branch = false;
  std::size_t fused_branch_index = static_cast<std::size_t>(-1);
};

simaai::neat::graph::NodeId
runtime_node_for_materialized_decoder(const PipelineSegmentRuntime& runtime, std::size_t index) {
  if (index < runtime.seg.materialized_node_attribution.size()) {
    return runtime.seg.materialized_node_attribution[index].runtime_node;
  }
  if (index < runtime.seg.node_ids.size()) {
    return runtime.seg.node_ids[index];
  }
  return simaai::neat::graph::kInvalidNode;
}

std::string decoder_runtime_label(const ExecutionGraphRuntime& execution,
                                  simaai::neat::graph::NodeId runtime_node) {
  if (runtime_node != simaai::neat::graph::kInvalidNode &&
      static_cast<std::size_t>(runtime_node) < execution.node_labels.size() &&
      !execution.node_labels[static_cast<std::size_t>(runtime_node)].empty()) {
    return execution.node_labels[static_cast<std::size_t>(runtime_node)];
  }
  return "<unknown>";
}

std::uint32_t positive_u32_or_zero(int value) {
  return value > 0 ? static_cast<std::uint32_t>(value) : 0U;
}

bool decoder_candidate_uses_zero_copy_output(const DecoderAdmissionCandidate& candidate) {
  const auto& opt = candidate.options;
  if (opt.type != simaai::neat::SimaDecodeType::H264 || !opt.raw_output) {
    return false;
  }
  if (!opt.out_format.empty() && opt.out_format.tag != simaai::neat::FormatTag::NV12) {
    return false;
  }
  const std::string next = simaai::neat::upper_copy_ascii(opt.next_element);
  return next.empty() || next == "CVU";
}

// Admission reserves decoder resources, so its shape must describe the
// decoder boundary itself.  Do not fall back to the segment tail: downstream
// Caps/Preproc/Model nodes may intentionally scale 1080p decode down to 640p.
OutputSpec decoder_local_output_spec(std::span<const std::shared_ptr<simaai::neat::Node>> nodes,
                                     std::size_t decoder_index,
                                     const OutputSpec& segment_input_spec) {
  if (decoder_index >= nodes.size()) {
    return {};
  }
  try {
    return derive_output_spec(
        std::span<const std::shared_ptr<simaai::neat::Node>>(nodes.data(), decoder_index + 1U),
        segment_input_spec);
  } catch (const std::exception& e) {
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr, "[DECPLAN] admission_local_spec_skip node_index=%zu reason=%s\n",
                   decoder_index, e.what());
    }
  } catch (...) {
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr, "[DECPLAN] admission_local_spec_skip node_index=%zu reason=unknown\n",
                   decoder_index);
    }
  }
  return {};
}

void collect_decoder_candidate_from_node(
    ExecutionGraphRuntime& execution, std::size_t pipeline_index, std::size_t node_index,
    const std::shared_ptr<simaai::neat::Node>& node, simaai::neat::graph::NodeId runtime_node,
    const OutputSpec& decoder_output_spec, bool fused_branch, std::size_t fused_branch_index,
    std::vector<DecoderAdmissionCandidate>& candidates, std::size_t* auto_h264_decoders,
    std::size_t* missing_shape_decoders) {
  (void)execution;
  const auto* dec = dynamic_cast<const simaai::neat::SimaDecode*>(node.get());
  if (!dec) {
    return;
  }
  const auto& opt = dec->options();
  if (opt.type != simaai::neat::SimaDecodeType::H264) {
    return;
  }
  if (auto_h264_decoders) {
    ++(*auto_h264_decoders);
  }

  const std::uint32_t explicit_width = positive_u32_or_zero(opt.dec_width);
  const std::uint32_t explicit_height = positive_u32_or_zero(opt.dec_height);
  const std::uint32_t explicit_fps = positive_u32_or_zero(opt.dec_fps);
  const std::uint32_t width =
      explicit_width != 0U ? explicit_width : positive_u32_or_zero(decoder_output_spec.width);
  const std::uint32_t height =
      explicit_height != 0U ? explicit_height : positive_u32_or_zero(decoder_output_spec.height);
  std::uint32_t fps_num =
      explicit_fps != 0U ? explicit_fps : positive_u32_or_zero(decoder_output_spec.fps_num);
  std::uint32_t fps_den = decoder_output_spec.fps_den > 0
                              ? static_cast<std::uint32_t>(decoder_output_spec.fps_den)
                              : 1U;
  if (explicit_fps != 0U) {
    fps_den = 1;
  }
  if (fps_num == 0U) {
    fps_num = 1;
    fps_den = 1;
  }

  if (width == 0U || height == 0U) {
    if (missing_shape_decoders) {
      ++(*missing_shape_decoders);
    }
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr,
                   "[DECPLAN] admission_candidate_skip reason=missing_decoder_local_shape "
                   "pipeline=%zu node_index=%zu runtime_node=%lld local=%dx%d explicit=%ux%u\n",
                   pipeline_index, node_index, static_cast<long long>(runtime_node),
                   decoder_output_spec.width, decoder_output_spec.height, explicit_width,
                   explicit_height);
    }
    return;
  }

  DecoderAdmissionCandidate candidate;
  candidate.pipeline_index = pipeline_index;
  candidate.node_index = node_index;
  candidate.runtime_node = runtime_node;
  candidate.options = opt;
  candidate.width = width;
  candidate.height = height;
  candidate.fps_num = fps_num;
  candidate.fps_den = fps_den == 0U ? 1U : fps_den;
  candidate.fused_branch = fused_branch;
  candidate.fused_branch_index = fused_branch_index;
  candidates.push_back(std::move(candidate));
}

bool collect_decoder_admission_candidates(ExecutionGraphRuntime& execution,
                                          std::vector<DecoderAdmissionCandidate>& candidates,
                                          std::size_t* auto_h264_decoders,
                                          std::size_t* missing_shape_decoders) {
  if (auto_h264_decoders) {
    *auto_h264_decoders = 0;
  }
  if (missing_shape_decoders) {
    *missing_shape_decoders = 0;
  }

  for (std::size_t pipeline_index = 0; pipeline_index < execution.pipelines.size();
       ++pipeline_index) {
    auto& runtime = execution.pipelines[pipeline_index];
    if (!runtime) {
      continue;
    }
    for (std::size_t node_index = 0; node_index < runtime->nodes.size(); ++node_index) {
      if (!dynamic_cast<const simaai::neat::SimaDecode*>(runtime->nodes[node_index].get())) {
        continue;
      }
      const OutputSpec decoder_spec =
          decoder_local_output_spec(std::span<const std::shared_ptr<simaai::neat::Node>>(
                                        runtime->nodes.data(), runtime->nodes.size()),
                                    node_index, runtime->seg.input_spec);
      collect_decoder_candidate_from_node(
          execution, pipeline_index, node_index, runtime->nodes[node_index],
          runtime_node_for_materialized_decoder(*runtime, node_index), decoder_spec,
          /*fused_branch=*/false, static_cast<std::size_t>(-1), candidates, auto_h264_decoders,
          missing_shape_decoders);
    }

    if (runtime->seg.fused_realtime_ingress.has_value()) {
      const auto& ingress = *runtime->seg.fused_realtime_ingress;
      for (std::size_t branch_index = 0; branch_index < ingress.branches.size(); ++branch_index) {
        const auto& branch = ingress.branches[branch_index];
        for (std::size_t node_index = 0; node_index < branch.nodes.size(); ++node_index) {
          if (!dynamic_cast<const simaai::neat::SimaDecode*>(branch.nodes[node_index].get())) {
            continue;
          }
          const OutputSpec decoder_spec =
              decoder_local_output_spec(std::span<const std::shared_ptr<simaai::neat::Node>>(
                                            branch.nodes.data(), branch.nodes.size()),
                                        node_index, {});
          collect_decoder_candidate_from_node(
              execution, pipeline_index, node_index, branch.nodes[node_index], branch.source_node,
              decoder_spec, /*fused_branch=*/true, branch_index, candidates, auto_h264_decoders,
              missing_shape_decoders);
        }
      }
    }
  }
  return !candidates.empty();
}

std::string decoder_admission_candidate_description(const ExecutionGraphRuntime& execution,
                                                    const DecoderAdmissionCandidate& candidate) {
  std::ostringstream oss;
  oss << "seg=" << execution.pipelines[candidate.pipeline_index]->seg.id;
  if (candidate.fused_branch) {
    oss << " branch=" << candidate.fused_branch_index;
  }
  oss << " mat_index=" << candidate.node_index
      << " runtime_node=" << static_cast<long long>(candidate.runtime_node)
      << " label=" << decoder_runtime_label(execution, candidate.runtime_node) << " "
      << candidate.width << "x" << candidate.height << "@" << candidate.fps_num << "/"
      << candidate.fps_den;
  return oss.str();
}

struct DecoderAdmissionProperties {
  std::string group_id;
  int stream_index = -1;
  std::uint64_t lease_token_hi = 0;
  std::uint64_t lease_token_lo = 0;
  int input_buffers = -1;
  std::string tuning;
  bool memory_opt = false;
  bool zero_copy_output = false;
};

class RuntimeAdmittedSimaDecode final : public simaai::neat::Node,
                                        public simaai::neat::OutputSpecProvider {
public:
  RuntimeAdmittedSimaDecode(simaai::neat::SimaDecodeOptions opt,
                            DecoderAdmissionProperties admission)
      : opt_(std::move(opt)), admission_(std::move(admission)), inner_(opt_) {}

  std::string kind() const override {
    return "SimaDecode";
  }

  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return inner_.caps_behavior();
  }

  simaai::neat::MemoryContract memory_contract() const override {
    return admission_.zero_copy_output ? simaai::neat::MemoryContract::PreferDeviceZeroCopy
                                       : simaai::neat::MemoryContract::AllowEitherButReport;
  }

  std::string buffer_name_hint(int node_index) const override {
    return opt_.decoder_name.empty() ? ("n" + std::to_string(node_index) + "_decoder")
                                     : opt_.decoder_name;
  }

  std::string backend_fragment(int node_index) const override {
    std::string fragment = inner_.backend_fragment(node_index);
    const std::string props = admission_properties_fragment();
    if (props.empty()) {
      return fragment;
    }
    const std::size_t next = fragment.find(" ! ");
    if (next == std::string::npos) {
      fragment += props;
      return fragment;
    }
    fragment.insert(next, props);
    return fragment;
  }

  std::vector<std::string> element_names(int node_index) const override {
    return inner_.element_names(node_index);
  }

  simaai::neat::OutputSpec output_spec(const simaai::neat::OutputSpec& input) const override {
    return inner_.output_spec(input);
  }

private:
  std::string admission_properties_fragment() const {
    std::ostringstream ss;
    if (admission_.zero_copy_output) {
      ss << " zero-copy-output=true";
    }
    ss << " decoder-admission-required=true";
    if (!admission_.group_id.empty()) {
      ss << " admission-group-id=" << gst_double_quote(admission_.group_id);
    }
    if (admission_.stream_index >= 0) {
      ss << " admission-stream-index=" << admission_.stream_index;
    }
    if (admission_.lease_token_hi != 0 || admission_.lease_token_lo != 0) {
      ss << " admission-lease-token-hi=" << admission_.lease_token_hi;
      ss << " admission-lease-token-lo=" << admission_.lease_token_lo;
    }
    if (admission_.input_buffers > 0) {
      ss << " dec-ip-cnt=" << admission_.input_buffers;
    }
    if (!admission_.tuning.empty() && admission_.tuning != "default") {
      ss << " decoder-tuning=" << admission_.tuning;
    }
    if (admission_.memory_opt) {
      ss << " memory-opt=true";
    }
    return ss.str();
  }

  simaai::neat::SimaDecodeOptions opt_;
  DecoderAdmissionProperties admission_;
  simaai::neat::SimaDecode inner_;
};

void apply_decoder_admission_lease(ExecutionGraphRuntime& execution,
                                   const DecoderAdmissionCandidate& candidate,
                                   const pipeline_internal::DecoderAdmissionResult& admission,
                                   const pipeline_internal::DecoderAdmissionLease& lease) {
  if (candidate.pipeline_index >= execution.pipelines.size() ||
      !execution.pipelines[candidate.pipeline_index]) {
    throw std::runtime_error(
        "RunCore::start(graph): decoder admission internal pipeline index invalid");
  }
  auto& runtime = *execution.pipelines[candidate.pipeline_index];
  if (!candidate.fused_branch && candidate.node_index >= runtime.nodes.size()) {
    throw std::runtime_error(
        "RunCore::start(graph): decoder admission internal node index invalid");
  }

  auto opt = candidate.options;
  if (lease.resolved_output_buffers > 0) {
    opt.num_buffers = static_cast<int>(lease.resolved_output_buffers);
  }

  DecoderAdmissionProperties admission_props;
  admission_props.group_id =
      pipeline_internal::decoder_admission_uuid_to_string(admission.group_uuid);
  admission_props.stream_index = static_cast<int>(lease.stream_index);
  admission_props.lease_token_hi = lease.lease_token_hi;
  admission_props.lease_token_lo = lease.lease_token_lo;
  if (lease.resolved_input_buffers > 0) {
    admission_props.input_buffers = static_cast<int>(lease.resolved_input_buffers);
  }
  admission_props.tuning = pipeline_internal::decoder_admission_tuning_name(lease.resolved_tuning);
  admission_props.memory_opt = lease.resolved_tuning == 1U || lease.resolved_tuning == 2U;
  admission_props.zero_copy_output = decoder_candidate_uses_zero_copy_output(candidate);

  auto replacement =
      std::make_shared<RuntimeAdmittedSimaDecode>(std::move(opt), std::move(admission_props));
  if (candidate.fused_branch) {
    if (!runtime.seg.fused_realtime_ingress.has_value() ||
        candidate.fused_branch_index >= runtime.seg.fused_realtime_ingress->branches.size()) {
      throw std::runtime_error(
          "RunCore::start(graph): decoder admission fused branch index invalid");
    }
    auto& branch = runtime.seg.fused_realtime_ingress->branches[candidate.fused_branch_index];
    if (candidate.node_index >= branch.nodes.size()) {
      throw std::runtime_error(
          "RunCore::start(graph): decoder admission fused branch node index invalid");
    }
    branch.nodes[candidate.node_index] = replacement;
  } else {
    runtime.nodes[candidate.node_index] = replacement;
    if (candidate.node_index < runtime.seg.nodes.size()) {
      runtime.seg.nodes[candidate.node_index] = replacement;
    }
  }

  if (decoder_plan_debug_enabled()) {
    std::fprintf(stderr,
                 "[DECPLAN] admission_bind %s stream=%u out=%u in=%u tuning=%s token=%llu:%llu "
                 "bytes=%llu\n",
                 decoder_admission_candidate_description(execution, candidate).c_str(),
                 lease.stream_index, lease.resolved_output_buffers, lease.resolved_input_buffers,
                 pipeline_internal::decoder_admission_tuning_name(lease.resolved_tuning),
                 static_cast<unsigned long long>(lease.lease_token_hi),
                 static_cast<unsigned long long>(lease.lease_token_lo),
                 static_cast<unsigned long long>(lease.estimated_reserved_bytes));
  }
}

void apply_decoder_admission_if_needed(ExecutionGraphRuntime& execution) {
  if (env_bool("SIMA_DECODER_ADMISSION_DISABLE", false)) {
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr, "[DECPLAN] admission_skip reason=disabled_by_env\n");
    }
    return;
  }

  std::vector<DecoderAdmissionCandidate> candidates;
  std::size_t auto_h264_decoders = 0;
  std::size_t missing_shape_decoders = 0;
  collect_decoder_admission_candidates(execution, candidates, &auto_h264_decoders,
                                       &missing_shape_decoders);
  if (auto_h264_decoders <= 1) {
    return;
  }
  if (missing_shape_decoders > 0) {
    const std::string msg =
        "RunCore::start(graph): automatic decoder admission requires decoded width/height for "
        "each H.264 decoder in a multi-decoder graph; missing shape for " +
        std::to_string(missing_shape_decoders) + " of " + std::to_string(auto_h264_decoders) +
        " decoder(s).";
    if (env_bool("SIMA_DECODER_ADMISSION_REQUIRE", false)) {
      throw std::runtime_error(msg);
    }
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr,
                   "[DECPLAN] admission_skip reason=missing_shape auto_h264=%zu missing=%zu\n",
                   auto_h264_decoders, missing_shape_decoders);
    }
    return;
  }
  if (candidates.size() <= 1) {
    return;
  }

  std::vector<pipeline_internal::DecoderAdmissionStreamRequest> streams;
  streams.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    pipeline_internal::DecoderAdmissionStreamRequest stream;
    stream.stream_index = static_cast<std::uint32_t>(i);
    stream.codec = 101;       // Decoder daemon admission protocol AVC/H.264 id.
    stream.stream_mode = 202; // Align-split input mode used by parser/depay paths.
    stream.width = candidate.width;
    stream.height = candidate.height;
    stream.fps_num = candidate.fps_num;
    stream.fps_den = candidate.fps_den;
    if (decoder_candidate_uses_zero_copy_output(candidate)) {
      stream.requested_policy = pipeline_internal::kDecoderAdmissionPolicyZeroCopyOutput |
                                pipeline_internal::kDecoderAdmissionPolicyNoOutputCopy;
    } else {
      stream.requested_policy = 0; // daemon AUTO policy.
    }
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr,
                   "[DECPLAN] admission_request %s stream=%u policy=0x%x zero_copy_output=%d\n",
                   decoder_admission_candidate_description(execution, candidate).c_str(),
                   stream.stream_index, stream.requested_policy,
                   decoder_candidate_uses_zero_copy_output(candidate) ? 1 : 0);
    }
    streams.push_back(stream);
  }

  auto admission = pipeline_internal::admit_decoder_graph(streams, false);
  if (!admission.admitted) {
    const bool require = env_bool("SIMA_DECODER_ADMISSION_REQUIRE", false);
    if (admission.endpoint_missing && !require) {
      if (decoder_plan_debug_enabled()) {
        std::fprintf(stderr, "[DECPLAN] admission_skip reason=endpoint_unavailable err=%s\n",
                     admission.error.c_str());
      }
      return;
    }
    throw std::runtime_error(
        "RunCore::start(graph): decoder admission rejected this multi-decoder graph before "
        "starting hardware decode. " +
        admission.error +
        ". Reduce the number of streams/fps/resolution, stop another decoder workload, or check "
        "the decoder daemon/admission socket.");
  }
  if (admission.leases.size() != candidates.size()) {
    throw std::runtime_error(
        "RunCore::start(graph): decoder admission returned a lease count that does not match the "
        "planned decoder count");
  }

  std::unordered_map<std::uint32_t, const pipeline_internal::DecoderAdmissionLease*>
      lease_by_stream;
  for (const auto& lease : admission.leases) {
    lease_by_stream[lease.stream_index] = &lease;
  }
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto stream_index = static_cast<std::uint32_t>(i);
    auto it = lease_by_stream.find(stream_index);
    if (it == lease_by_stream.end()) {
      throw std::runtime_error(
          "RunCore::start(graph): decoder admission response missing a stream lease");
    }
    apply_decoder_admission_lease(execution, candidates[i], admission, *it->second);
  }
  execution.decoder_admission_active = true;
  execution.decoder_admission_group_uuid = admission.group_uuid;

  if (decoder_plan_debug_enabled()) {
    std::fprintf(stderr, "[DECPLAN] admission_accepted streams=%zu group=%s reserved_bytes=%llu\n",
                 candidates.size(),
                 pipeline_internal::decoder_admission_uuid_to_string(admission.group_uuid).c_str(),
                 static_cast<unsigned long long>(admission.estimated_reserved_bytes));
  }
}

GraphReport make_graph_start_report(const ExecutionGraphPlan& plan, const std::string& detail) {
  GraphReport rep;
  rep.error_code = error_codes::kPipelineShape;
  rep.nodes.reserve(plan.node_labels.size());
  for (std::size_t i = 0; i < plan.node_labels.size(); ++i) {
    if (plan.node_labels[i].empty()) {
      continue;
    }
    NodeReport node;
    node.index = static_cast<int>(i);
    node.user_label = plan.node_labels[i];
    rep.nodes.push_back(std::move(node));
  }

  const bool already_scoped = detail.rfind("RunCore::start(graph):", 0) == 0;
  std::ostringstream note;
  note << (already_scoped ? detail : "RunCore::start(graph): " + detail);
  note << "\nHint: this is a graph build/configuration error. ";
  if (detail.find("unambiguous default input") != std::string::npos) {
    note << "The graph has no single default input for seeded build(input). Add exactly one "
            "public Input endpoint, or use a named push path for multi-input graphs.";
  } else if (detail.find("default input") != std::string::npos) {
    note << "Check the graph's public Input endpoints and connected-fragment boundaries.";
  } else {
    note << "Check Graph::describe()/describe_backend(), public Input/Output endpoints, and "
            "fragment wiring.";
  }
  rep.repro_note = note.str();
  return rep;
}

[[noreturn]] void throw_graph_start_error(const ExecutionGraphPlan& plan,
                                          const std::string& detail) {
  GraphReport rep = make_graph_start_report(plan, detail.empty() ? "unknown failure" : detail);
  throw NeatError("[" + rep.error_code + "] " + rep.repro_note, std::move(rep));
}

PortId intern_runtime_port(ExecutionGraphRuntime& execution, const std::string& name) {
  if (name.empty()) {
    throw std::invalid_argument("RunCore::start(graph): empty stage port name");
  }
  for (std::size_t i = 0; i < execution.plan.port_names.size(); ++i) {
    if (execution.plan.port_names[i] == name) {
      return static_cast<PortId>(i);
    }
  }
  const PortId id = static_cast<PortId>(execution.plan.port_names.size());
  execution.plan.port_names.push_back(name);
  return id;
}

void populate_node_labels(ExecutionGraphRuntime& execution) {
  const std::size_t count = runtime_node_count(execution.plan);
  execution.node_labels = execution.plan.node_labels;
  execution.node_labels.resize(count);
  for (NodeId id = 0; id < count; ++id) {
    std::string& label = execution.node_labels[id];
    if (label.empty()) {
      label = "node" + std::to_string(id);
    }
  }
}

EdgeRouterCallbacks make_edge_router_callbacks(const std::shared_ptr<RunCore>& core) {
  EdgeRouterCallbacks callbacks;
  callbacks.dispatch_to_stage_group = [core](std::size_t group_index, PortId port, Sample&& sample,
                                             std::size_t edge_index) {
    return core->graph_dispatch_to_stage_group(group_index, port, std::move(sample), edge_index,
                                               core->graph_options.router_options());
  };
  callbacks.ensure_pipeline_built = [core](std::size_t index, const Sample& sample,
                                           std::string* err) {
    return core->ensure_graph_pipeline_built(index, sample, err);
  };
  callbacks.sanitize_pipeline_input = [core](std::size_t index, Sample& sample) {
    core->graph_sanitize_pipeline_input(index, sample);
  };
  callbacks.prepare_sink_sample = [](NodeId, Sample& sample, std::size_t qsize,
                                     std::size_t diag_id) {
    simaai::neat::graph::maybe_force_copy_for_backpressure(sample, qsize, "sink_queue", diag_id);
  };
  callbacks.request_stop = [core](const std::string& err) { core->graph_request_stop(err); };
  callbacks.stop_requested = [core] { return core->graph_stop_requested(); };
  return callbacks;
}

bool route_stage_output(const std::shared_ptr<RunCore>& core, StageRuntime& st,
                        StageOutMsg&& out_msg) {
  if (!core || core->graph_stop_requested()) {
    return false;
  }

  PortId out_port = out_msg.out_port;
  if (out_port == kInvalidPort && st.output_ports.size() == 1) {
    out_port = st.output_ports[0];
  }

  auto& execution = core->graph_execution();
  EdgeRouter router(execution);
  const auto router_options = core->graph_options.router_options();
  const auto router_callbacks = make_edge_router_callbacks(core);
  EdgeRouterDispatchOptions dispatch_options;
  dispatch_options.sanitize_pipeline_input_before_enqueue = true;

  const auto* targets = router.targets(st.node_id, out_port);
  if (!targets) {
    return router.push_to_sink(st.node_id, std::move(out_msg.sample), invalid_edge_index(),
                               router_options, router_callbacks, st.node_id);
  }

  return router.dispatch_to_targets(*targets, std::move(out_msg.sample), router_options,
                                    router_callbacks, dispatch_options);
}

void materialize_pipeline_runtimes(const std::shared_ptr<RunCore>& core) {
  ExecutionGraphRuntime& execution = core->graph_execution();
  execution.pipelines.reserve(execution.plan.pipeline_segments.size());
  for (const auto& seg : execution.plan.pipeline_segments) {
    if (seg.consumed_by_fused_realtime_ingress) {
      continue;
    }
    GraphOptions sess_opt = seg.route_options;
    if (sess_opt.element_name_suffix.empty()) {
      sess_opt.element_name_suffix = "_graph";
    }
    sess_opt.verbose = core->graph_options.verbose;
    sess_opt.verbose.progress = false;

    std::vector<std::shared_ptr<simaai::neat::Node>> nodes = seg.nodes;
    bool injected_input = false;
    bool injected_output = false;

    if (!seg.boundary.source_like && seg.input_edges.empty() &&
        simaai::neat::graph::has_internal_source(seg.nodes) &&
        !simaai::neat::graph::has_input_appsrc(seg.nodes)) {
      std::string node_names;
      for (NodeId nid : seg.node_ids) {
        if (!node_names.empty()) {
          node_names += ", ";
        }
        node_names += std::to_string(nid);
        if (nid < execution.node_labels.size()) {
          node_names += "('" + execution.node_labels[nid] + "')";
        }
      }
      throw std::runtime_error("RunCore::start(graph): pipeline segment " + std::to_string(seg.id) +
                               " has internal source but is not source_like."
                               " Refusing to inject Input into a source pipeline."
                               " Segment nodes: [" +
                               node_names + "].");
    }

    if (seg.boundary.needs_input && !seg.boundary.source_like &&
        !simaai::neat::graph::has_input_appsrc(seg.nodes)) {
      InputOptions opt_src;
      if (seg.boundary_hints.has_value() && !seg.boundary_hints->ingress_inputs.empty()) {
        opt_src = seg.boundary_hints->ingress_inputs.front();
      } else {
        opt_src = simaai::neat::graph::input_opts_from_spec(seg.input_spec, seg.input_complete);
      }
      nodes.insert(nodes.begin(), simaai::neat::nodes::Input(opt_src));
      injected_input = true;
    }

    if (seg.boundary.needs_output && !simaai::neat::graph::has_output_appsink(seg.nodes)) {
      nodes.push_back(simaai::neat::nodes::Output());
      injected_output = true;
    }

    const simaai::neat::Input* src_node = nullptr;
    if (!seg.boundary.source_like) {
      for (const auto& node : nodes) {
        if (!node) {
          continue;
        }
        if (auto* input = dynamic_cast<const simaai::neat::Input*>(node.get())) {
          src_node = input;
          break;
        }
      }
    }

    auto runtime = std::make_unique<PipelineSegmentRuntime>();
    runtime->seg = seg;
    runtime->nodes = std::move(nodes);
    runtime->route_options = sess_opt;
    runtime->run_options = core->graph_options.pipeline;
    if (seg.boundary.graph_internal_output) {
      runtime->run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
      // Graph-internal appsinks are transport edges to downstream stages or
      // pipeline inputs, not user CPU-read boundaries. Preserve zero-copy
      // forwarding and leave CPU visibility to the true consumer boundary.
      runtime->run_options.advanced.prepare_output_cpu_visible = false;
    }
    runtime->seg.nodes = runtime->nodes;
    runtime->seg.route_options = runtime->route_options;
    runtime->seg.run_options = runtime->run_options;
    runtime->seg.materialized_node_attribution =
        make_materialized_node_attribution(seg, injected_input, injected_output);
    runtime->transport.has_input = seg.boundary.needs_input;
    runtime->transport.has_output = seg.boundary.needs_output;
    if (seg.boundary.direct_graph_source) {
      // A public Input-only fragment is a graph ingress endpoint, not executable
      // GStreamer work.  Keep the runtime node/endpoint mapping so Run::push()
      // can route through EdgeRouter, but mark it ready so seeded builds and
      // startup prebuild never materialize a one-node appsrc pipeline.
      runtime->transport.built.store(true, std::memory_order_release);
    }
    if (seg.boundary.direct_graph_sink && !seg.node_ids.empty()) {
      execution.direct_sink_nodes.insert(seg.node_ids.back());
    }
    if (runtime->transport.has_input && src_node) {
      const pipeline_internal::PipelineBuildContext build_ctx(sess_opt);
      runtime->transport.expected_buffer_names =
          build_ctx.resolve_expected_buffer_names(src_node->options().buffer_name);
    }
    if (runtime->transport.has_input) {
      std::size_t input_capacity = core->graph_options.edge_queue;
      bool realtime_input = false;
      for (const std::size_t edge_index : seg.input_edges) {
        if (edge_index >= execution.plan.edges.size()) {
          continue;
        }
        const auto& link = execution.plan.edges[edge_index].link_options;
        if (link.policy == GraphLinkPolicy::RealtimeLatestByStream) {
          realtime_input = true;
        }
      }
      if (realtime_input) {
        input_capacity = 1;
      }
      runtime->transport.input_queue =
          std::make_shared<simaai::neat::graph::runtime::BlockingQueue<RuntimePipelineQueueMsg>>(
              input_capacity);
    }

    const std::size_t index = execution.pipelines.size();
    for (NodeId nid : seg.node_ids) {
      execution.node_to_pipeline[nid] = index;
    }
    execution.pipelines.push_back(std::move(runtime));
  }
}

void materialize_stage_runtimes(const std::shared_ptr<RunCore>& core) {
  ExecutionGraphRuntime& execution = core->graph_execution();
  execution.stage_groups.reserve(execution.plan.stage_nodes.size());
  for (const auto& st : execution.plan.stage_nodes) {
    if (!st.node) {
      continue;
    }

    StageNodeOptions opt_node = st.node->options();
    if (opt_node.instances < 1) {
      opt_node.instances = 1;
    }

    const std::size_t group_index = execution.stage_groups.size();
    execution.stage_groups.emplace_back();
    auto& group = execution.stage_groups.back();
    group.node_id = st.node_id;
    group.options = opt_node;
    group.instances.reserve(static_cast<std::size_t>(opt_node.instances));

    const std::size_t capacity =
        (opt_node.max_inflight > 0) ? opt_node.max_inflight : core->graph_options.edge_queue;

    for (int i = 0; i < opt_node.instances; ++i) {
      auto rt = std::make_unique<StageRuntime>(capacity);
      rt->node_id = st.node_id;
      rt->exec = st.node->factory()();
      for (const auto& port : st.node->input_ports()) {
        const PortId pid = intern_runtime_port(execution, port.name);
        rt->input_ports.push_back(pid);
        rt->ports.in.emplace(port.name, pid);
      }
      for (const auto& port : st.node->output_ports()) {
        const PortId pid = intern_runtime_port(execution, port.name);
        rt->output_ports.push_back(pid);
        rt->ports.out.emplace(port.name, pid);
      }
      if (rt->exec) {
        auto weak_core = std::weak_ptr<RunCore>(core);
        StageRuntime* stage_runtime = rt.get();
        rt->emitter.stop_requested_fn = [weak_core] {
          auto locked = weak_core.lock();
          return !locked || locked->graph_stop_requested();
        };
        rt->emitter.emit_fn = [weak_core, stage_runtime](StageOutMsg msg) {
          auto locked = weak_core.lock();
          if (!locked || stage_runtime == nullptr) {
            return false;
          }
          return route_stage_output(locked, *stage_runtime, std::move(msg));
        };
        rt->exec->set_ports(rt->ports);
        rt->exec->set_emitter(&rt->emitter);
      }

      group.instances.push_back(execution.stages.size());
      execution.stages.push_back(std::move(rt));
    }

    execution.node_to_stage_group[st.node_id] = group_index;
  }
}

void build_adjacency_and_sinks(const std::shared_ptr<RunCore>& core) {
  ExecutionGraphRuntime& execution = core->graph_execution();
  std::vector<bool> has_out(runtime_node_count(execution.plan), false);
  std::unordered_map<std::string, std::size_t> realtime_link_by_target;

  const auto downstream_target_for = [&](const EdgePlan& e,
                                         std::size_t eidx) -> std::optional<DownstreamTarget> {
    auto it_stage = execution.node_to_stage_group.find(e.to);
    if (it_stage != execution.node_to_stage_group.end()) {
      return DownstreamTarget{DownstreamTarget::Kind::StageGroup, it_stage->second, e.to_port,
                              eidx};
    }
    auto it_pipe = execution.node_to_pipeline.find(e.to);
    if (it_pipe == execution.node_to_pipeline.end()) {
      return std::nullopt;
    }
    if (execution.direct_sink_nodes.find(e.to) != execution.direct_sink_nodes.end()) {
      return DownstreamTarget{DownstreamTarget::Kind::GraphSink, static_cast<std::size_t>(e.to),
                              e.to_port, eidx};
    }
    return DownstreamTarget{DownstreamTarget::Kind::PipelineInput, it_pipe->second, e.to_port,
                            eidx};
  };

  const auto target_key = [](const DownstreamTarget& target) {
    return std::to_string(static_cast<int>(target.kind)) + ":" + std::to_string(target.index) +
           ":" + std::to_string(static_cast<std::size_t>(target.port));
  };

  for (std::size_t eidx = 0; eidx < execution.plan.edges.size(); ++eidx) {
    const auto& e = execution.plan.edges[eidx];
    if (e.consumed_by_fused_realtime_ingress) {
      continue;
    }
    if (e.from < has_out.size()) {
      has_out[e.from] = true;
    }
    std::vector<DownstreamTarget>& outs = execution.adjacency[edge_port_key(e.from, e.from_port)];

    auto downstream = downstream_target_for(e, eidx);
    if (!downstream.has_value()) {
      continue;
    }

    if (e.link_options.policy == GraphLinkPolicy::RealtimeLatestByStream) {
      const std::string key = target_key(*downstream);
      auto it = realtime_link_by_target.find(key);
      if (it == realtime_link_by_target.end()) {
        const std::size_t link_index = execution.realtime_links.size();
        realtime_link_by_target.emplace(key, link_index);
        execution.realtime_links.push_back(
            std::make_unique<RealtimeLatestLink>(*downstream, e.link_options, e.stream_id));
        it = realtime_link_by_target.find(key);
      } else if (it->second < execution.realtime_links.size() &&
                 execution.realtime_links[it->second]) {
        execution.realtime_links[it->second]->add_edge_stream_id(eidx, e.stream_id);
      }
      outs.push_back(DownstreamTarget{DownstreamTarget::Kind::RealtimeLatestLink, it->second,
                                      e.to_port, eidx});
      continue;
    }

    outs.push_back(*downstream);
  }

  for (NodeId id = 0; id < has_out.size(); ++id) {
    if (!has_out[id]) {
      execution.sinks[id] = std::make_shared<GraphSinkQueue>(core->graph_options.edge_queue);
    }
  }
}

void prebuild_complete_pipeline_segments(const std::shared_ptr<RunCore>& core, bool has_seed) {
  ExecutionGraphRuntime& execution = core->graph_execution();
  const auto& default_input = execution.plan.default_input;
  for (std::size_t i = 0; i < execution.pipelines.size(); ++i) {
    auto& rt = *execution.pipelines[i];
    if (!rt.transport.has_input || !rt.seg.input_complete) {
      continue;
    }
    // For a seedless build, defer the default-input segment's pipeline build
    // to the first push. Otherwise we synthesize a CPU/SystemMemory sample
    // from the fragment-hint-derived input_spec and commit a buffer pool
    // against it — which then mismatches the user's real first push when
    // that push is, e.g., a device-backed Tensor (TensorMemory::EV74).
    // Seeded builds keep their prior behavior: prebuild runs eagerly, and
    // prebuild_seeded_default_input_segment is a no-op against it.
    if (!has_seed && default_input.has_value() && default_input->segment == i) {
      continue;
    }
    std::string spec_err;
    auto sample_opt = simaai::neat::graph::sample_from_input_spec(rt.seg.input_spec, &spec_err);
    if (!sample_opt.has_value()) {
      if (simaai::neat::graph::graph_debug_enabled() && !spec_err.empty()) {
        std::fprintf(stderr, "[GRAPH] prebuild_skip seg=%zu reason=%s\n",
                     static_cast<std::size_t>(rt.seg.id), spec_err.c_str());
      }
      continue;
    }
    std::string build_err;
    if (!core->ensure_graph_pipeline_built(i, *sample_opt, &build_err) &&
        simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr, "[GRAPH] prebuild_failed seg=%zu err=%s\n",
                   static_cast<std::size_t>(rt.seg.id),
                   build_err.empty() ? "<unknown>" : build_err.c_str());
    }
  }
}

void prebuild_seeded_default_input_segment(const std::shared_ptr<RunCore>& core,
                                           const std::optional<Sample>& seed,
                                           bool allow_startup_preflight) {
  if (!seed.has_value()) {
    return;
  }
  ExecutionGraphRuntime& execution = core->graph_execution();
  const auto& endpoint = execution.plan.default_input;
  if (!endpoint.has_value()) {
    throw std::runtime_error(
        "RunCore::start(graph): seeded build requires an unambiguous default input");
  }
  if (endpoint->segment == static_cast<std::size_t>(-1) ||
      endpoint->segment >= execution.pipelines.size()) {
    throw std::runtime_error(
        "RunCore::start(graph): seeded default input does not resolve to a pipeline segment");
  }
  std::string build_err;
  if (!core->ensure_graph_pipeline_built(endpoint->segment, *seed, &build_err,
                                         allow_startup_preflight)) {
    throw std::runtime_error(build_err.empty()
                                 ? "RunCore::start(graph): seeded default input build failed"
                                 : "RunCore::start(graph): " + build_err);
  }
}

void start_stage_workers(const std::shared_ptr<RunCore>& core) {
  ExecutionGraphRuntime& execution = core->graph_execution();
  for (std::size_t i = 0; i < execution.stages.size(); ++i) {
    auto& rt = *execution.stages[i];
    rt.worker = std::thread([core, i]() {
      auto& execution = core->graph_execution();
      auto& st = *execution.stages[i];
      struct DoneGuard {
        std::atomic<bool>& flag;
        ~DoneGuard() {
          flag.store(true);
        }
      } done_guard{st.worker_done};
      while (!core->graph_stop_requested()) {
        RuntimeStageQueueMsg queued;
        const auto pop_start = std::chrono::steady_clock::now();
        if (!st.inbox.pop(queued, core->graph_options.pull_timeout_ms)) {
          st.telemetry.mailbox_pop_miss.fetch_add(1, std::memory_order_relaxed);
          atomic_add_max(st.telemetry.mailbox_pop_wait_ns, st.telemetry.mailbox_pop_wait_max_ns,
                         elapsed_ns_since(pop_start));
          continue;
        }
        st.telemetry.mailbox_pop_calls.fetch_add(1, std::memory_order_relaxed);
        atomic_add_max(st.telemetry.mailbox_pop_wait_ns, st.telemetry.mailbox_pop_wait_max_ns,
                       elapsed_ns_since(pop_start));
        if (graph_message_trace_enabled(&execution) && queued.edge_index != invalid_edge_index()) {
          const TraceGraphMessageArgs args =
              make_trace_graph_message_args(&execution, queued.edge_index, queued.sample);
          trace_graph_message_event(TraceGraphMessageEventType::QueueOut, args);
          trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv, args);
        }
        StageMsg msg{.in_port = queued.in_port, .sample = std::move(queued.sample)};
        if (msg.in_port == kInvalidPort && st.input_ports.size() == 1) {
          msg.in_port = st.input_ports[0];
        }
        if (simaai::neat::graph::graph_debug_enabled()) {
          simaai::neat::graph::graph_debug_sample("stage_on_input", msg.sample);
        }
        std::vector<StageOutMsg> outputs;
        try {
          const auto exec_start = std::chrono::steady_clock::now();
          st.exec->on_input(std::move(msg), outputs);
          st.telemetry.on_input_calls.fetch_add(1, std::memory_order_relaxed);
          atomic_add_max(st.telemetry.on_input_ns, st.telemetry.on_input_max_ns,
                         elapsed_ns_since(exec_start));
        } catch (const std::exception& e) {
          core->graph_request_stop(e.what());
          break;
        }

        for (auto& out_msg : outputs) {
          const auto route_start = std::chrono::steady_clock::now();
          const bool routed = route_stage_output(core, st, std::move(out_msg));
          st.telemetry.route_output_calls.fetch_add(1, std::memory_order_relaxed);
          atomic_add_max(st.telemetry.route_output_ns, st.telemetry.route_output_max_ns,
                         elapsed_ns_since(route_start));
          if (!routed) {
            break;
          }
        }
      }
    });
  }
}

void start_realtime_links(const std::shared_ptr<RunCore>& core) {
  ExecutionGraphRuntime& execution = core->graph_execution();
  for (auto& link : execution.realtime_links) {
    if (!link) {
      continue;
    }
    link->start(
        [core](const DownstreamTarget& target, Sample&& sample, std::size_t edge_index) {
          EdgeRouter router(core->graph_execution());
          const auto router_options = core->graph_options.router_options();
          const auto router_callbacks = make_edge_router_callbacks(core);
          EdgeRouterDispatchOptions dispatch_options;
          dispatch_options.sanitize_pipeline_input_before_enqueue = true;
          /*
           * RealtimeLatestLink already gives live sources the non-blocking
           * "latest frame per stream" behavior: offers overwrite each stream's
           * pending slot instead of backpressuring decoders.  The scheduler
           * side must not drop again when the downstream Model/input queue is
           * temporarily full; doing so biases fan-in toward whichever streams
           * happened to arrive last.  Block here until the downstream queue can
           * accept the selected stream, preserving the C++ fair-mux semantics
           * while keeping source producers non-blocking and zero-copy.
           */
          dispatch_options.drop_pipeline_input_when_full = false;
          dispatch_options.sink_backpressure_context = target.index;
          DownstreamTarget routed = target;
          routed.edge_index = edge_index;
          return router.dispatch_to_target(routed, std::move(sample), router_options,
                                           router_callbacks, dispatch_options);
        },
        [core] { return core->graph_stop_requested(); },
        [core](const std::string& err) { core->graph_request_stop(err); });
  }
}

void build_source_pipeline_if_needed(const std::shared_ptr<RunCore>& core,
                                     PipelineSegmentRuntime& rt) {
  if (rt.seg.boundary.direct_graph_source) {
    rt.transport.built.store(true, std::memory_order_release);
    rt.transport.cv.notify_all();
    return;
  }

  if (rt.transport.has_input || !rt.seg.boundary.source_like) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(rt.transport.mu);
    RunCoreStartOptions start_opt;
    start_opt.run_options = rt.run_options;
    start_opt.mode = RunMode::Async;
    start_opt.last_pipeline = &rt.last_pipeline;
    start_opt.push_sample_policy = PushSamplePolicy::PreserveSample;
    rt.run_core = RunCore::start_pipeline_segment(rt.seg, std::move(start_opt));
    rt.transport.built.store(true, std::memory_order_release);
  }
  rt.transport.cv.notify_all();
}

void start_pipeline_pull_thread(const std::shared_ptr<RunCore>& core, std::size_t i) {
  auto& pipe = *core->graph_execution().pipelines[i];
  pipe.transport.pull_thread = std::thread([core, i]() {
    auto& execution = core->graph_execution();
    auto& pipe = *execution.pipelines[i];
    struct DoneGuard {
      std::atomic<bool>& flag;
      ~DoneGuard() {
        flag.store(true);
      }
    } done_guard{pipe.transport.pull_done};
    if (simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr, "[GRAPH] pipeline_pull_thread_start seg=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id));
    }

    {
      std::unique_lock<std::mutex> lock(pipe.transport.mu);
      pipe.transport.cv.wait(lock, [&] {
        return pipe.transport.built.load(std::memory_order_acquire) || core->graph_stop_requested();
      });
    }

    if (core->graph_stop_requested()) {
      if (simaai::neat::graph::graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] pipeline_pull_thread_stop_before_ready seg=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id));
      }
      return;
    }
    if (simaai::neat::graph::graph_debug_enabled()) {
      std::fprintf(stderr, "[GRAPH] pipeline_pull_thread_ready seg=%zu\n",
                   static_cast<std::size_t>(pipe.seg.id));
    }

    try {
      EdgeRouter router(execution);
      const auto router_options = core->graph_options.router_options();
      const auto router_callbacks = make_edge_router_callbacks(core);
      EdgeRouterDispatchOptions dispatch_options;
      dispatch_options.sink_backpressure_context = static_cast<std::size_t>(pipe.seg.id);

      const int diag_ms = env_int("SIMA_GRAPH_PIPELINE_DIAG_MS", 0);
      const bool diag_summary = env_bool("SIMA_GRAPH_PIPELINE_DIAG_SUMMARY", false);
      auto last_output = std::chrono::steady_clock::now();
      auto last_diag = last_output;
      auto emit_diag = [&](const char* reason) {
        if (diag_ms <= 0) {
          return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_diag < std::chrono::milliseconds(diag_ms)) {
          return;
        }
        last_diag = now;
        const RunStats stats = pipe.run_core ? pipe.run_core->stats() : RunStats{};
        const InputStreamStats in_stats =
            pipe.run_core ? pipe.run_core->input_stats() : InputStreamStats{};
        const std::size_t in_q =
            pipe.transport.input_queue ? pipe.transport.input_queue->size() : 0;
        const auto idle_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_output).count();
        std::fprintf(stderr,
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
                     static_cast<long long>(
                         pipe.transport.identity_rewrite_count.load(std::memory_order_relaxed)),
                     static_cast<long long>(
                         pipe.transport.identity_map_miss_count.load(std::memory_order_relaxed)));
        if (diag_summary) {
          const std::string summary =
              pipe.run_core ? pipe.run_core->diagnostics_summary() : std::string{};
          if (!summary.empty()) {
            std::fprintf(stderr, "[GRAPH] pipeline_diag_summary seg=%zu %s\n",
                         static_cast<std::size_t>(pipe.seg.id), summary.c_str());
          }
        }
      };
      int pull_logs = 0;
      while (!core->graph_stop_requested() &&
             pipe.transport.built.load(std::memory_order_acquire)) {
        if (simaai::neat::graph::graph_debug_enabled() && pull_logs < 5) {
          std::fprintf(stderr, "[GRAPH] pipeline_pull_wait seg=%zu\n",
                       static_cast<std::size_t>(pipe.seg.id));
          pull_logs++;
        }
        const auto pull_start = std::chrono::steady_clock::now();
        auto sample_opt = pipe.run_core
                              ? pipe.run_core->pull_optional(core->graph_options.pull_timeout_ms)
                              : std::optional<Sample>{};
        pipe.transport.telemetry.pull_thread_pull_calls.fetch_add(1, std::memory_order_relaxed);
        atomic_add_max(pipe.transport.telemetry.pull_thread_pull_ns,
                       pipe.transport.telemetry.pull_thread_pull_max_ns,
                       elapsed_ns_since(pull_start));
        if (!sample_opt.has_value()) {
          pipe.transport.telemetry.pull_thread_pull_miss.fetch_add(1, std::memory_order_relaxed);
          emit_diag("empty");
          if (simaai::neat::graph::graph_debug_enabled()) {
            const std::string err = pipe.run_core ? pipe.run_core->last_error() : std::string{};
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
        core->graph_restore_stream_id_if_needed(i, sample);
        simaai::neat::graph::log_first_decoded_once(sample, pipe.seg.id);
        if (simaai::neat::graph::graph_debug_enabled()) {
          simaai::neat::graph::graph_debug_sample("pipeline_pull", sample);
        }

        PortId out_port = kInvalidPort;
        if (!pipe.seg.output_edges.empty()) {
          const auto& e = execution.plan.edges[pipe.seg.output_edges[0]];
          out_port = e.from_port;
        }

        const auto* targets = router.targets(pipe.seg.node_ids.back(), out_port);
        if (!targets) {
          const NodeId sink_node = pipe.seg.node_ids.back();
          if (simaai::neat::graph::graph_debug_enabled() &&
              execution.sinks.find(sink_node) != execution.sinks.end()) {
            std::fprintf(stderr, "[GRAPH] model_output_received seg=%zu\n",
                         static_cast<std::size_t>(pipe.seg.id));
            simaai::neat::graph::graph_debug_sample("pipeline_to_sink", sample);
          }
          const auto route_start = std::chrono::steady_clock::now();
          (void)router.push_to_sink(sink_node, std::move(sample), invalid_edge_index(),
                                    router_options, router_callbacks,
                                    static_cast<std::size_t>(pipe.seg.id));
          pipe.transport.telemetry.pull_thread_route_calls.fetch_add(1, std::memory_order_relaxed);
          atomic_add_max(pipe.transport.telemetry.pull_thread_route_ns,
                         pipe.transport.telemetry.pull_thread_route_max_ns,
                         elapsed_ns_since(route_start));
          continue;
        }

        const auto route_start = std::chrono::steady_clock::now();
        (void)router.dispatch_to_targets(*targets, std::move(sample), router_options,
                                         router_callbacks, dispatch_options);
        pipe.transport.telemetry.pull_thread_route_calls.fetch_add(1, std::memory_order_relaxed);
        atomic_add_max(pipe.transport.telemetry.pull_thread_route_ns,
                       pipe.transport.telemetry.pull_thread_route_max_ns,
                       elapsed_ns_since(route_start));
      }
    } catch (const std::exception& e) {
      core->graph_request_stop(e.what());
    }
  });
}

void start_pipeline_push_thread(const std::shared_ptr<RunCore>& core, std::size_t i) {
  auto& pipe = *core->graph_execution().pipelines[i];
  pipe.transport.push_thread = std::thread([core, i]() {
    auto& pipe = *core->graph_execution().pipelines[i];
    struct DoneGuard {
      std::atomic<bool>& flag;
      ~DoneGuard() {
        flag.store(true);
      }
    } done_guard{pipe.transport.push_done};
    while (!core->graph_stop_requested()) {
      RuntimePipelineQueueMsg queued;
      const auto pop_start = std::chrono::steady_clock::now();
      if (!pipe.transport.input_queue->pop(queued, core->graph_options.pull_timeout_ms)) {
        pipe.transport.telemetry.push_thread_pop_miss.fetch_add(1, std::memory_order_relaxed);
        atomic_add_max(pipe.transport.telemetry.push_thread_pop_wait_ns,
                       pipe.transport.telemetry.push_thread_pop_wait_max_ns,
                       elapsed_ns_since(pop_start));
        continue;
      }
      pipe.transport.telemetry.push_thread_pop_calls.fetch_add(1, std::memory_order_relaxed);
      atomic_add_max(pipe.transport.telemetry.push_thread_pop_wait_ns,
                     pipe.transport.telemetry.push_thread_pop_wait_max_ns,
                     elapsed_ns_since(pop_start));
      auto& execution = core->graph_execution();
      if (graph_message_trace_enabled(&execution) && queued.edge_index != invalid_edge_index()) {
        const TraceGraphMessageArgs args =
            make_trace_graph_message_args(&execution, queued.edge_index, queued.sample);
        trace_graph_message_event(TraceGraphMessageEventType::QueueOut, args);
        trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv, args);
      }
      Sample sample = std::move(queued.sample);
      if (simaai::neat::graph::graph_debug_enabled()) {
        simaai::neat::graph::graph_debug_sample("pipeline_push_pop", sample);
      }

      const auto sanitize_start = std::chrono::steady_clock::now();
      core->graph_sanitize_pipeline_input(i, sample);
      pipe.transport.telemetry.push_thread_sanitize_calls.fetch_add(1, std::memory_order_relaxed);
      atomic_add_max(pipe.transport.telemetry.push_thread_sanitize_ns,
                     pipe.transport.telemetry.push_thread_sanitize_max_ns,
                     elapsed_ns_since(sanitize_start));

      if (simaai::neat::graph::is_encoded_sample(sample) && sample.caps_string.empty()) {
        core->graph_request_stop("GraphRun: encoded Sample missing caps_string");
        return;
      }

      if (!pipe.transport.built.load(std::memory_order_acquire)) {
        std::string build_err;
        const auto ensure_start = std::chrono::steady_clock::now();
        pipe.transport.telemetry.push_thread_ensure_build_calls.fetch_add(
            1, std::memory_order_relaxed);
        if (!core->ensure_graph_pipeline_built(i, sample, &build_err)) {
          atomic_add_max(pipe.transport.telemetry.push_thread_ensure_build_ns,
                         pipe.transport.telemetry.push_thread_ensure_build_max_ns,
                         elapsed_ns_since(ensure_start));
          core->graph_request_stop(build_err.empty() ? "GraphRun: pipeline build failed"
                                                     : build_err);
          return;
        }
        atomic_add_max(pipe.transport.telemetry.push_thread_ensure_build_ns,
                       pipe.transport.telemetry.push_thread_ensure_build_max_ns,
                       elapsed_ns_since(ensure_start));
      }

      const auto push_start = std::chrono::steady_clock::now();
      const bool realtime_edge =
          queued.edge_index != invalid_edge_index() &&
          queued.edge_index < core->graph_execution().plan.edges.size() &&
          core->graph_execution().plan.edges[queued.edge_index].link_options.policy ==
              GraphLinkPolicy::RealtimeLatestByStream;
      const bool pushed =
          pipe.run_core && pipe.run_core->push_samples(Sample{sample}, !realtime_edge);
      pipe.transport.telemetry.push_thread_push_samples_calls.fetch_add(1,
                                                                        std::memory_order_relaxed);
      atomic_add_max(pipe.transport.telemetry.push_thread_push_samples_ns,
                     pipe.transport.telemetry.push_thread_push_samples_max_ns,
                     elapsed_ns_since(push_start));
      if (!pushed) {
        const std::string err = pipe.run_core ? pipe.run_core->last_error() : std::string{};
        if (realtime_edge && !core->graph_stop_requested() && err.empty()) {
          continue;
        }
        if (simaai::neat::graph::graph_debug_enabled() ||
            simaai::neat::graph::graph_push_fail_debug_enabled()) {
          std::fprintf(stderr, "[GRAPH] pipeline_push_failed seg=%zu stop=%d\n",
                       static_cast<std::size_t>(pipe.seg.id),
                       static_cast<int>(core->graph_stop_requested()));
          std::fprintf(
              stderr, "[GRAPH] pipeline_push_failed pid=%d thread=%zu\n",
              static_cast<int>(::getpid()),
              static_cast<std::size_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
          simaai::neat::graph::graph_debug_sample("pipeline_push_failed", sample);
          if (!err.empty()) {
            std::fprintf(stderr, "[GRAPH] pipeline_push_failed err=%s\n", err.c_str());
          }
          const RunStats stats = pipe.run_core ? pipe.run_core->stats() : RunStats{};
          const InputStreamStats in_stats =
              pipe.run_core ? pipe.run_core->input_stats() : InputStreamStats{};
          std::fprintf(stderr,
                       "[GRAPH] pipeline_push_failed stats in_enq=%llu in_push=%llu "
                       "out_ready=%llu out_pulled=%llu is_push=%llu is_fail=%llu "
                       "is_pull=%llu\n",
                       static_cast<unsigned long long>(stats.inputs_enqueued),
                       static_cast<unsigned long long>(stats.inputs_pushed),
                       static_cast<unsigned long long>(stats.outputs_ready),
                       static_cast<unsigned long long>(stats.outputs_pulled),
                       static_cast<unsigned long long>(in_stats.push_count),
                       static_cast<unsigned long long>(in_stats.push_failures),
                       static_cast<unsigned long long>(in_stats.pull_count));
          const std::string diag =
              pipe.run_core ? pipe.run_core->diagnostics_summary() : std::string{};
          if (!diag.empty()) {
            std::fprintf(stderr, "[GRAPH] pipeline_push_failed diag=%s\n", diag.c_str());
          }
        }
        if (pipeline_internal::env_bool("SIMA_GRAPH_GDB_ON_PUSH_FAIL", false)) {
          std::fprintf(stderr, "[GRAPH] pipeline_push_failed: SIGSTOP for gdb attach pid=%d\n",
                       static_cast<int>(::getpid()));
          std::raise(SIGSTOP);
        }
        core->graph_request_stop("GraphRun: pipeline push failed");
        return;
      }
    }
  });
}

void start_pipeline_threads(const std::shared_ptr<RunCore>& core) {
  ExecutionGraphRuntime& execution = core->graph_execution();
  for (std::size_t i = 0; i < execution.pipelines.size(); ++i) {
    auto& rt = *execution.pipelines[i];
    build_source_pipeline_if_needed(core, rt);
    if (rt.transport.has_output) {
      start_pipeline_pull_thread(core, i);
    }
    if (rt.transport.has_input) {
      start_pipeline_push_thread(core, i);
    }
  }
}

std::shared_ptr<RunCore> start_graph_plan(ExecutionGraphPlan plan, RunCoreStartOptions opt) {
  const auto total_start = pipeline_internal::build_timing_now();
  if (opt.graph_options.push_timeout_ms < 0) {
    throw_graph_start_error(plan, "RunCore::start(graph): push_timeout_ms must be >= 0");
  }
  if (opt.graph_options.pull_timeout_ms < 0) {
    throw_graph_start_error(plan, "RunCore::start(graph): pull_timeout_ms must be >= 0");
  }

  auto core = RunCore::create_graph_compat();
  core->opt = opt.run_options;
  core->mode = opt.mode;
  core->graph_options = std::move(opt.graph_options);
  configure_graph_public_output_loan_gate(core, core->graph_options, opt.mode);
  core->graph_verbose_guard = std::move(opt.graph_verbose_guard);
  core->graph_execution().plan = std::move(plan);

  try {
    auto step_start = pipeline_internal::build_timing_now();
    populate_node_labels(core->graph_execution());
    const auto labels_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    materialize_pipeline_runtimes(core);
    const auto materialize_pipelines_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    materialize_stage_runtimes(core);
    const auto materialize_stages_us = pipeline_internal::build_timing_us(step_start);
    dump_decoder_plan_debug(core->graph_execution());
    step_start = pipeline_internal::build_timing_now();
    apply_decoder_admission_if_needed(core->graph_execution());
    const auto decoder_admission_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    build_adjacency_and_sinks(core);
    const auto adjacency_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    prebuild_complete_pipeline_segments(core, opt.seed.has_value());
    const auto prebuild_complete_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    prebuild_seeded_default_input_segment(core, opt.seed, opt.allow_startup_preflight);
    const auto prebuild_seed_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    start_stage_workers(core);
    const auto start_stages_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    start_realtime_links(core);
    const auto start_links_us = pipeline_internal::build_timing_us(step_start);
    step_start = pipeline_internal::build_timing_now();
    start_pipeline_threads(core);
    const auto start_pipelines_us = pipeline_internal::build_timing_us(step_start);

    if (core->graph_options.power_monitor.enabled) {
      core->power_monitor =
          std::make_unique<simaai::neat::PowerMonitor>(core->graph_options.power_monitor);
      core->power_monitor->start();
    }

    const auto& execution = core->graph_execution();
    pipeline_internal::emit_build_timing(
        "RunCore::start(graph)",
        {{"labels", labels_us},
         {"materialize_pipelines", materialize_pipelines_us},
         {"materialize_stages", materialize_stages_us},
         {"decoder_admission", decoder_admission_us},
         {"adjacency", adjacency_us},
         {"prebuild_complete", prebuild_complete_us},
         {"prebuild_seed", prebuild_seed_us},
         {"start_stage_workers", start_stages_us},
         {"start_realtime_links", start_links_us},
         {"start_pipeline_threads", start_pipelines_us},
         {"total", pipeline_internal::build_timing_us(total_start)}},
        "segments=" + std::to_string(execution.pipelines.size()) +
            " stages=" + std::to_string(execution.stages.size()) +
            " edges=" + std::to_string(execution.plan.edges.size()));

    return core;
  } catch (const NeatError&) {
    throw;
  } catch (const std::exception& e) {
    throw_graph_start_error(core->graph_execution().plan, e.what());
  } catch (...) {
    throw_graph_start_error(core->graph_execution().plan, "unknown failure");
  }
}

} // namespace

std::shared_ptr<RunCore> RunCore::start(ExecutionGraphPlan plan, RunCoreStartOptions opt) {
  gst_init_once();

  if (is_simple_linear_plan(plan)) {
    auto export_plan = std::make_unique<ExecutionGraphPlan>(plan);
    auto core = RunCore::start_pipeline_segment(plan.pipeline_segments.front(), std::move(opt));
    core->graph_export_plan_ = std::move(export_plan);
    return core;
  }

  return start_graph_plan(std::move(plan), std::move(opt));
}

} // namespace simaai::neat::runtime

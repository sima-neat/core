#include "DecoderAdmission.h"

#include "builder/OutputSpec.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace simaai::neat::runtime {
namespace {

using pipeline_internal::DecoderAdmissionLease;
using pipeline_internal::DecoderAdmissionResult;
using pipeline_internal::DecoderAdmissionStreamRequest;

struct DecoderAdmissionCandidate {
  std::size_t segment_index = 0;
  std::size_t node_index = 0;
  graph::NodeId runtime_node = graph::kInvalidNode;
  SimaDecodeOptions options;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 1;
  bool zero_copy_output = false;
  bool fused_branch = false;
  std::size_t fused_branch_index = static_cast<std::size_t>(-1);
};

struct DecoderAdmissionProperties {
  std::string group_id;
  int stream_index = -1;
  std::uint64_t lease_token_hi = 0;
  std::uint64_t lease_token_lo = 0;
  int input_buffers = -1;
  std::string tuning;
  bool memory_opt = false;
  bool zero_copy_output = false;
  bool require_external_lease = true;
};

bool decoder_plan_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_DECODER_PLAN_DEBUG", false) ||
         pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_DEBUG", false);
}

bool decoder_cma_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_DECODER_CMA_DEBUG", false) ||
         pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_DEBUG", false);
}

struct CmaSnapshot {
  long mem_free_kb = -1;
  long mem_available_kb = -1;
  long cma_total_kb = -1;
  long cma_free_kb = -1;
};

bool read_cma_snapshot(CmaSnapshot& out) {
  std::ifstream in("/proc/meminfo");
  if (!in.is_open()) {
    return false;
  }
  std::string key;
  long value = 0;
  std::string unit;
  while (in >> key >> value >> unit) {
    if (key == "MemFree:") {
      out.mem_free_kb = value;
    } else if (key == "MemAvailable:") {
      out.mem_available_kb = value;
    } else if (key == "CmaTotal:") {
      out.cma_total_kb = value;
    } else if (key == "CmaFree:") {
      out.cma_free_kb = value;
    }
  }
  return true;
}

void log_decoder_cma_snapshot(const char* event, std::size_t streams = 0,
                              std::uint64_t reserved_bytes = 0) {
  if (!decoder_cma_debug_enabled()) {
    return;
  }
  CmaSnapshot snap;
  if (!read_cma_snapshot(snap)) {
    std::fprintf(stderr, "[DECCMA] event=%s streams=%zu reserved_bytes=%llu read_failed=1\n",
                 event ? event : "snapshot", streams,
                 static_cast<unsigned long long>(reserved_bytes));
    return;
  }
  std::fprintf(stderr,
               "[DECCMA] event=%s streams=%zu reserved_bytes=%llu mem_free_kb=%ld "
               "mem_available_kb=%ld cma_total_kb=%ld cma_free_kb=%ld cma_used_kb=%ld\n",
               event ? event : "snapshot", streams, static_cast<unsigned long long>(reserved_bytes),
               snap.mem_free_kb, snap.mem_available_kb, snap.cma_total_kb, snap.cma_free_kb,
               (snap.cma_total_kb >= 0 && snap.cma_free_kb >= 0)
                   ? (snap.cma_total_kb - snap.cma_free_kb)
                   : -1);
}

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

std::uint32_t positive_u32_or_zero(int value) {
  return value > 0 ? static_cast<std::uint32_t>(value) : 0U;
}

constexpr bool decode_type_uses_video_admission(SimaDecodeType type) {
  return type == SimaDecodeType::H264 || type == SimaDecodeType::H265;
}

constexpr std::uint32_t decoder_admission_codec(SimaDecodeType type) {
  return type == SimaDecodeType::H265 ? pipeline_internal::kDecoderAdmissionCodecH265
                                      : pipeline_internal::kDecoderAdmissionCodecH264;
}

static_assert(decoder_admission_codec(SimaDecodeType::H264) ==
              pipeline_internal::kDecoderAdmissionCodecH264);
static_assert(decoder_admission_codec(SimaDecodeType::H265) ==
              pipeline_internal::kDecoderAdmissionCodecH265);

bool decoder_options_allow_zero_copy_output(const SimaDecodeOptions& opt) {
  if (!opt.raw_output) {
    return false;
  }
  if (!opt.out_format.empty() && opt.out_format.tag != FormatTag::NV12) {
    return false;
  }
  const std::string next = upper_copy_ascii(opt.next_element);
  return next.empty() || next == "CVU";
}

bool nodes_require_packed_decoder_output(std::span<const std::shared_ptr<Node>> nodes,
                                         std::size_t begin = 0) {
  bool layout_aware_encoder_ingress = false;
  for (std::size_t i = begin; i < nodes.size(); ++i) {
    if (!nodes[i]) {
      continue;
    }
    const std::string kind = nodes[i]->kind();
    if (kind == "VideoSenderRawIngress[direct_nv12]") {
      layout_aware_encoder_ingress = true;
    } else if (kind == "VideoSenderRawIngress[convert_to_nv12]") {
      return true;
    } else if (kind == "H264EncodeSima") {
      if (!layout_aware_encoder_ingress) {
        return true;
      }
      layout_aware_encoder_ingress = false;
    }
  }
  return false;
}

bool downstream_edge_requires_packed_decoder_output(const ExecutionGraphPlan& plan,
                                                    std::size_t edge_index,
                                                    std::unordered_set<std::size_t>& visited) {
  if (edge_index >= plan.edges.size() || !visited.insert(edge_index).second ||
      plan.edges[edge_index].consumed_by_fused_realtime_ingress) {
    return false;
  }

  for (const auto& segment : plan.pipeline_segments) {
    if (segment.consumed_by_fused_realtime_ingress ||
        std::find(segment.input_edges.begin(), segment.input_edges.end(), edge_index) ==
            segment.input_edges.end()) {
      continue;
    }
    if (nodes_require_packed_decoder_output(segment.nodes)) {
      return true;
    }
    for (const std::size_t output_edge : segment.output_edges) {
      if (downstream_edge_requires_packed_decoder_output(plan, output_edge, visited)) {
        return true;
      }
    }
  }

  const graph::NodeId target = plan.edges[edge_index].to;
  for (std::size_t next = 0; next < plan.edges.size(); ++next) {
    if (plan.edges[next].from == target &&
        downstream_edge_requires_packed_decoder_output(plan, next, visited)) {
      return true;
    }
  }
  return false;
}

bool downstream_requires_packed_decoder_output(const ExecutionGraphPlan& plan,
                                               const DecoderAdmissionCandidate& candidate) {
  const auto& source = plan.pipeline_segments[candidate.segment_index];
  if (candidate.fused_branch) {
    const auto& branch = source.fused_realtime_ingress->branches[candidate.fused_branch_index];
    if (nodes_require_packed_decoder_output(branch.nodes, candidate.node_index + 1U) ||
        nodes_require_packed_decoder_output(source.nodes)) {
      return true;
    }
  } else if (nodes_require_packed_decoder_output(source.nodes, candidate.node_index + 1U)) {
    return true;
  }

  std::unordered_set<std::size_t> visited_edges;
  for (const std::size_t edge_index : source.output_edges) {
    if (downstream_edge_requires_packed_decoder_output(plan, edge_index, visited_edges)) {
      return true;
    }
  }
  return false;
}

OutputSpec decoder_local_output_spec(std::span<const std::shared_ptr<Node>> nodes,
                                     std::size_t decoder_index,
                                     const OutputSpec& segment_input_spec) {
  if (decoder_index >= nodes.size()) {
    return {};
  }
  try {
    return derive_output_spec(
        std::span<const std::shared_ptr<Node>>(nodes.data(), decoder_index + 1U),
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

std::string decoder_runtime_label(const ExecutionGraphPlan& plan, graph::NodeId runtime_node) {
  if (runtime_node != graph::kInvalidNode &&
      static_cast<std::size_t>(runtime_node) < plan.node_labels.size() &&
      !plan.node_labels[static_cast<std::size_t>(runtime_node)].empty()) {
    return plan.node_labels[static_cast<std::size_t>(runtime_node)];
  }
  return "<unknown>";
}

std::string candidate_description(const ExecutionGraphPlan& plan,
                                  const DecoderAdmissionCandidate& candidate) {
  std::ostringstream oss;
  oss << "seg=" << plan.pipeline_segments[candidate.segment_index].id;
  if (candidate.fused_branch) {
    oss << " branch=" << candidate.fused_branch_index;
  }
  oss << " node_index=" << candidate.node_index
      << " runtime_node=" << static_cast<long long>(candidate.runtime_node)
      << " label=" << decoder_runtime_label(plan, candidate.runtime_node) << " " << candidate.width
      << "x" << candidate.height << "@" << candidate.fps_num << "/" << candidate.fps_den;
  return oss.str();
}

void collect_candidate(std::size_t segment_index, std::size_t node_index,
                       const std::shared_ptr<Node>& node, graph::NodeId runtime_node,
                       const OutputSpec& decoder_output_spec, bool fused_branch,
                       std::size_t fused_branch_index,
                       std::vector<DecoderAdmissionCandidate>& candidates) {
  const auto* decoder = dynamic_cast<const SimaDecode*>(node.get());
  if (!decoder || !decode_type_uses_video_admission(decoder->options().type)) {
    return;
  }

  const auto& opt = decoder->options();
  DecoderAdmissionCandidate candidate;
  candidate.segment_index = segment_index;
  candidate.node_index = node_index;
  candidate.runtime_node = runtime_node;
  candidate.options = opt;
  candidate.width = positive_u32_or_zero(opt.dec_width);
  candidate.height = positive_u32_or_zero(opt.dec_height);
  candidate.fps_num = positive_u32_or_zero(opt.dec_fps);
  if (candidate.width == 0U) {
    candidate.width = positive_u32_or_zero(decoder_output_spec.width);
  }
  if (candidate.height == 0U) {
    candidate.height = positive_u32_or_zero(decoder_output_spec.height);
  }
  if (candidate.fps_num == 0U) {
    candidate.fps_num = positive_u32_or_zero(decoder_output_spec.fps_num);
    candidate.fps_den = decoder_output_spec.fps_den > 0
                            ? static_cast<std::uint32_t>(decoder_output_spec.fps_den)
                            : 1U;
  }
  candidate.fused_branch = fused_branch;
  candidate.fused_branch_index = fused_branch_index;
  candidates.push_back(std::move(candidate));
}

std::vector<DecoderAdmissionCandidate> collect_candidates(const ExecutionGraphPlan& plan) {
  std::vector<DecoderAdmissionCandidate> candidates;
  for (std::size_t segment_index = 0; segment_index < plan.pipeline_segments.size();
       ++segment_index) {
    const auto& segment = plan.pipeline_segments[segment_index];
    if (!segment.consumed_by_fused_realtime_ingress) {
      for (std::size_t node_index = 0; node_index < segment.nodes.size(); ++node_index) {
        const auto* decoder = dynamic_cast<const SimaDecode*>(segment.nodes[node_index].get());
        if (!decoder || !decode_type_uses_video_admission(decoder->options().type)) {
          continue;
        }
        const OutputSpec decoder_spec = decoder_local_output_spec(
            std::span<const std::shared_ptr<Node>>(segment.nodes.data(), segment.nodes.size()),
            node_index, segment.input_spec);
        collect_candidate(segment_index, node_index, segment.nodes[node_index],
                          attributed_runtime_node_for_segment_node(segment, node_index),
                          decoder_spec, false, static_cast<std::size_t>(-1), candidates);
      }
    }

    if (!segment.fused_realtime_ingress.has_value()) {
      continue;
    }
    const auto& ingress = *segment.fused_realtime_ingress;
    for (std::size_t branch_index = 0; branch_index < ingress.branches.size(); ++branch_index) {
      const auto& branch = ingress.branches[branch_index];
      for (std::size_t node_index = 0; node_index < branch.nodes.size(); ++node_index) {
        const auto* decoder = dynamic_cast<const SimaDecode*>(branch.nodes[node_index].get());
        if (!decoder || !decode_type_uses_video_admission(decoder->options().type)) {
          continue;
        }
        const OutputSpec decoder_spec = decoder_local_output_spec(
            std::span<const std::shared_ptr<Node>>(branch.nodes.data(), branch.nodes.size()),
            node_index, {});
        collect_candidate(segment_index, node_index, branch.nodes[node_index], branch.source_node,
                          decoder_spec, true, branch_index, candidates);
      }
    }
  }
  for (auto& candidate : candidates) {
    candidate.zero_copy_output = decoder_options_allow_zero_copy_output(candidate.options) &&
                                 !downstream_requires_packed_decoder_output(plan, candidate);
  }
  return candidates;
}

std::string missing_contract_message(const ExecutionGraphPlan& plan,
                                     const std::vector<DecoderAdmissionCandidate>& candidates) {
  std::ostringstream missing;
  std::size_t count = 0;
  for (const auto& candidate : candidates) {
    if (candidate.width > 0U && candidate.height > 0U && candidate.fps_num > 0U) {
      continue;
    }
    if (count++ > 0U) {
      missing << "; ";
    }
    missing << candidate_description(plan, candidate) << " missing=";
    bool separator = false;
    const auto append = [&](const char* field) {
      if (separator) {
        missing << ",";
      }
      missing << field;
      separator = true;
    };
    if (candidate.width == 0U) {
      append("width");
    }
    if (candidate.height == 0U) {
      append("height");
    }
    if (candidate.fps_num == 0U) {
      append("fps");
    }
  }
  if (count == 0U) {
    return {};
  }
  return "automatic decoder admission requires known width, height, and FPS for every "
         "H.264/H.265 decoder; " +
         missing.str();
}

bool explicit_decoder_tuning(const std::string& tuning) {
  return !tuning.empty() && tuning != "auto" && tuning != "default";
}

bool decoder_tuning_uses_memory_opt(const std::string& tuning) {
  return tuning == "low-memory" || tuning == "throughput-low-latency";
}

class RuntimeAdmittedSimaDecode final : public Node, public OutputSpecProvider {
public:
  RuntimeAdmittedSimaDecode(SimaDecodeOptions opt, DecoderAdmissionProperties admission)
      : opt_(std::move(opt)), admission_(std::move(admission)), inner_(opt_) {}

  std::string kind() const override {
    return "SimaDecode";
  }

  NodeCapsBehavior caps_behavior() const override {
    return inner_.caps_behavior();
  }

  MemoryContract memory_contract() const override {
    return admission_.zero_copy_output ? MemoryContract::PreferDeviceZeroCopy
                                       : MemoryContract::AllowEitherButReport;
  }

  std::string buffer_name_hint(int node_index) const override {
    return opt_.decoder_name.empty() ? ("n" + std::to_string(node_index) + "_decoder")
                                     : opt_.decoder_name;
  }

  std::string backend_fragment(int node_index) const override {
    std::string fragment = inner_.backend_fragment(node_index);
    const std::string props = admission_properties_fragment();
    const std::size_t next = fragment.find(" ! ");
    if (next == std::string::npos) {
      fragment += props;
    } else {
      fragment.insert(next, props);
    }
    return fragment;
  }

  std::vector<std::string> element_names(int node_index) const override {
    return inner_.element_names(node_index);
  }

  OutputSpec output_spec(const OutputSpec& input) const override {
    return inner_.output_spec(input);
  }

private:
  std::string admission_properties_fragment() const {
    std::ostringstream ss;
    if (admission_.zero_copy_output) {
      ss << " zero-copy-output=true";
    }
    if (admission_.require_external_lease) {
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

  SimaDecodeOptions opt_;
  DecoderAdmissionProperties admission_;
  SimaDecode inner_;
};

std::shared_ptr<Node> make_admitted_decoder(const DecoderAdmissionCandidate& candidate,
                                            const DecoderAdmissionResult& admission,
                                            const DecoderAdmissionLease& lease) {
  auto opt = candidate.options;
  const int resolved_output_buffers =
      lease.resolved_output_buffers > 0 ? static_cast<int>(lease.resolved_output_buffers) : 0;
  if (resolved_output_buffers > 0 && opt.num_buffers > 0 &&
      opt.num_buffers < resolved_output_buffers) {
    opt.num_buffers = resolved_output_buffers;
  }

  DecoderAdmissionProperties props;
  props.group_id = pipeline_internal::decoder_admission_uuid_to_string(admission.group_uuid);
  props.stream_index = static_cast<int>(lease.stream_index);
  props.lease_token_hi = lease.lease_token_hi;
  props.lease_token_lo = lease.lease_token_lo;
  props.input_buffers =
      opt.input_buffers > 0 ? opt.input_buffers : static_cast<int>(lease.resolved_input_buffers);
  props.tuning = explicit_decoder_tuning(opt.decoder_tuning)
                     ? opt.decoder_tuning
                     : pipeline_internal::decoder_admission_tuning_name(lease.resolved_tuning);
  props.memory_opt = opt.memory_opt || decoder_tuning_uses_memory_opt(props.tuning) ||
                     lease.resolved_tuning == 1U || lease.resolved_tuning == 2U;
  props.zero_copy_output = candidate.zero_copy_output;

  opt.input_buffers = -1;
  opt.decoder_tuning.clear();
  opt.memory_opt = false;
  return std::make_shared<RuntimeAdmittedSimaDecode>(std::move(opt), std::move(props));
}

std::shared_ptr<Node> make_kernel_reserved_decoder(
    const DecoderAdmissionCandidate& candidate) {
  /*
   * AL5_RESERVE is a command-channel contract.  The direct decoder owns that
   * channel and reserves it immediately before codec allocation/start.  Core
   * must not create an unrelated daemon socket lease which cannot authorize
   * the eventual command fd.  It still owns graph-derived zero-copy policy.
   */
  DecoderAdmissionProperties props;
  props.zero_copy_output = candidate.zero_copy_output;
  props.require_external_lease = false;
  return std::make_shared<RuntimeAdmittedSimaDecode>(candidate.options,
                                                      std::move(props));
}

void bind_replacements(ExecutionGraphPlan& plan,
                       const std::vector<DecoderAdmissionCandidate>& candidates,
                       const std::vector<std::shared_ptr<Node>>& replacements) {
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    auto& segment = plan.pipeline_segments[candidate.segment_index];
    if (candidate.fused_branch) {
      segment.fused_realtime_ingress->branches[candidate.fused_branch_index]
          .nodes[candidate.node_index] = replacements[i];
    } else {
      segment.nodes[candidate.node_index] = replacements[i];
    }
  }
}

void emit_optional_warning(const std::string& warning) {
  std::fprintf(stderr, "[WARN] RunCore::start: decoder admission skipped: %s\n", warning.c_str());
}

} // namespace

DecoderAdmissionReservation::DecoderAdmissionReservation(
    std::shared_ptr<DecoderAdmissionBackend> backend, std::array<std::uint8_t, 16> group_uuid,
    std::size_t stream_count, std::uint64_t reserved_bytes)
    : backend_(std::move(backend)), group_uuid_(group_uuid), stream_count_(stream_count),
      reserved_bytes_(reserved_bytes), active_(backend_ != nullptr) {}

DecoderAdmissionReservation::~DecoderAdmissionReservation() {
  release();
}

DecoderAdmissionReservation::DecoderAdmissionReservation(
    DecoderAdmissionReservation&& other) noexcept
    : backend_(std::move(other.backend_)), group_uuid_(other.group_uuid_),
      stream_count_(other.stream_count_), reserved_bytes_(other.reserved_bytes_),
      active_(std::exchange(other.active_, false)) {}

DecoderAdmissionReservation&
DecoderAdmissionReservation::operator=(DecoderAdmissionReservation&& other) noexcept {
  if (this != &other) {
    release();
    backend_ = std::move(other.backend_);
    group_uuid_ = other.group_uuid_;
    stream_count_ = other.stream_count_;
    reserved_bytes_ = other.reserved_bytes_;
    active_ = std::exchange(other.active_, false);
  }
  return *this;
}

bool DecoderAdmissionReservation::active() const noexcept {
  return active_;
}

void DecoderAdmissionReservation::release() noexcept {
  if (!std::exchange(active_, false) || !backend_) {
    return;
  }
  log_decoder_cma_snapshot("before_admission_release", stream_count_, reserved_bytes_);
  std::string error;
  bool released = false;
  try {
    released = backend_->release(group_uuid_, &error);
  } catch (const std::exception& e) {
    error = e.what();
  } catch (...) {
    error = "unknown exception";
  }
  if (!released && decoder_plan_debug_enabled()) {
    std::fprintf(stderr, "[DECPLAN] admission_release_failed group=%s err=%s\n",
                 pipeline_internal::decoder_admission_uuid_to_string(group_uuid_).c_str(),
                 error.empty() ? "<unknown>" : error.c_str());
  } else if (released && decoder_plan_debug_enabled()) {
    std::fprintf(stderr, "[DECPLAN] admission_released group=%s\n",
                 pipeline_internal::decoder_admission_uuid_to_string(group_uuid_).c_str());
  }
  log_decoder_cma_snapshot(released ? "after_admission_release" : "after_admission_release_failed",
                           stream_count_, reserved_bytes_);
}

DecoderAdmissionPreparation
prepare_decoder_admission(ExecutionGraphPlan& plan,
                          std::shared_ptr<DecoderAdmissionBackend> backend) {
  DecoderAdmissionPreparation preparation;
  if (pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_DISABLE", false)) {
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr, "[DECPLAN] admission_skip reason=disabled_by_env\n");
    }
    return preparation;
  }

  std::vector<DecoderAdmissionCandidate> candidates = collect_candidates(plan);
  preparation.eligible_decoders = candidates.size();
  if (candidates.empty()) {
    return preparation;
  }

  const std::string missing = missing_contract_message(plan, candidates);
  if (!missing.empty()) {
    if (pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_REQUIRE", false)) {
      throw std::runtime_error("RunCore::start: " + missing);
    }
    preparation.warning = missing;
    emit_optional_warning(preparation.warning);
    return preparation;
  }

  if (!backend) {
    std::vector<std::shared_ptr<Node>> replacements;
    replacements.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      replacements.push_back(make_kernel_reserved_decoder(candidate));
    }
    bind_replacements(plan, candidates, replacements);
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr,
                   "[DECPLAN] kernel_direct_admission streams=%zu owner=decoder-command-fd\n",
                   candidates.size());
    }
    return preparation;
  }
  std::vector<DecoderAdmissionStreamRequest> streams;
  streams.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    DecoderAdmissionStreamRequest stream;
    stream.stream_index = static_cast<std::uint32_t>(i);
    stream.codec = decoder_admission_codec(candidate.options.type);
    stream.stream_mode = 202;
    stream.width = candidate.width;
    stream.height = candidate.height;
    stream.fps_num = candidate.fps_num;
    stream.fps_den = candidate.fps_den;
    if (candidate.zero_copy_output) {
      stream.requested_policy = pipeline_internal::kDecoderAdmissionPolicyZeroCopyOutput |
                                pipeline_internal::kDecoderAdmissionPolicyNoOutputCopy;
    }
    if (decoder_plan_debug_enabled()) {
      std::fprintf(stderr, "[DECPLAN] admission_request %s stream=%u policy=0x%x\n",
                   candidate_description(plan, candidate).c_str(), stream.stream_index,
                   stream.requested_policy);
    }
    streams.push_back(stream);
  }

  log_decoder_cma_snapshot("before_admission_request", candidates.size());
  DecoderAdmissionResult admission = backend->admit(streams, false);
  if (!admission.admitted) {
    if (admission.may_have_committed) {
      DecoderAdmissionReservation uncertain_admission(
          backend, admission.group_uuid, candidates.size(), admission.estimated_reserved_bytes);
      uncertain_admission.release();
    }
    log_decoder_cma_snapshot("after_admission_rejected", candidates.size());
    if (admission.endpoint_missing &&
        !pipeline_internal::env_bool("SIMA_DECODER_ADMISSION_REQUIRE", false)) {
      preparation.warning = "admission endpoint unavailable";
      if (!admission.error.empty()) {
        preparation.warning += ": " + admission.error;
      }
      emit_optional_warning(preparation.warning);
      return preparation;
    }
    throw std::runtime_error(
        "RunCore::start: decoder admission rejected this decoder run before starting hardware "
        "decode: " +
        admission.error +
        ". Reduce streams/fps/resolution, stop another decoder workload, or check the decoder "
        "daemon/admission socket.");
  }

  auto reservation = std::make_unique<DecoderAdmissionReservation>(
      backend, admission.group_uuid, candidates.size(), admission.estimated_reserved_bytes);
  log_decoder_cma_snapshot("after_admission_accepted", candidates.size(),
                           admission.estimated_reserved_bytes);

  if (admission.leases.size() != candidates.size()) {
    throw std::runtime_error(
        "RunCore::start: decoder admission returned a lease count that does not match the "
        "planned decoder count");
  }
  std::vector<const DecoderAdmissionLease*> leases(candidates.size(), nullptr);
  for (const auto& lease : admission.leases) {
    if (lease.stream_index >= leases.size() || leases[lease.stream_index] != nullptr) {
      throw std::runtime_error(
          "RunCore::start: decoder admission returned an invalid or duplicate stream lease");
    }
    leases[lease.stream_index] = &lease;
  }

  std::vector<std::shared_ptr<Node>> replacements;
  replacements.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (!leases[i]) {
      throw std::runtime_error(
          "RunCore::start: decoder admission response is missing a stream lease");
    }
    replacements.push_back(make_admitted_decoder(candidates[i], admission, *leases[i]));
  }
  bind_replacements(plan, candidates, replacements);

  if (decoder_plan_debug_enabled()) {
    std::fprintf(stderr, "[DECPLAN] admission_accepted streams=%zu group=%s reserved_bytes=%llu\n",
                 candidates.size(),
                 pipeline_internal::decoder_admission_uuid_to_string(admission.group_uuid).c_str(),
                 static_cast<unsigned long long>(admission.estimated_reserved_bytes));
  }
  log_decoder_cma_snapshot("after_admission_bound", candidates.size(),
                           admission.estimated_reserved_bytes);
  preparation.reservation = std::move(reservation);
  return preparation;
}

} // namespace simaai::neat::runtime

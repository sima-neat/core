#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "nodes/sima/SimaDecode.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "pipeline/runtime/DecoderAdmission.h"
#include "pipeline/runtime/RunCore.h"
#include "test_utils.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using simaai::neat::SimaDecodeOptions;
using simaai::neat::SimaDecodeType;
using simaai::neat::pipeline_internal::DecoderAdmissionLease;
using simaai::neat::pipeline_internal::DecoderAdmissionResult;
using simaai::neat::pipeline_internal::DecoderAdmissionStreamRequest;
using simaai::neat::runtime::DecoderAdmissionBackend;
using simaai::neat::runtime::ExecutionGraphPlan;
using simaai::neat::runtime::PipelineSegmentPlan;

class MarkerNode final : public simaai::neat::Node {
public:
  explicit MarkerNode(std::string kind) : kind_(std::move(kind)) {}

  std::string kind() const override {
    return kind_;
  }

  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override {
    return "identity name=n" + std::to_string(node_index) + "_marker";
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_marker"};
  }

private:
  std::string kind_;
};

std::shared_ptr<simaai::neat::Node> marker(std::string kind) {
  return std::make_shared<MarkerNode>(std::move(kind));
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char* key, const char* value) : key_(key) {
    if (const char* old = std::getenv(key)) {
      old_ = old;
      had_old_ = true;
    }
    setenv(key, value, 1);
  }

  ~ScopedEnvVar() {
    if (had_old_) {
      setenv(key_.c_str(), old_.c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

private:
  std::string key_;
  std::string old_;
  bool had_old_ = false;
};

class FakeBackend final : public DecoderAdmissionBackend {
public:
  DecoderAdmissionResult admit(const std::vector<DecoderAdmissionStreamRequest>& streams,
                               bool dry_run) override {
    ++admit_count;
    requests = streams;
    last_dry_run = dry_run;
    DecoderAdmissionResult out = response;
    if (out.admitted && auto_leases) {
      out.group_uuid[0] = 42;
      out.leases.clear();
      for (std::size_t i = 0; i < streams.size(); ++i) {
        DecoderAdmissionLease lease;
        lease.stream_index = static_cast<std::uint32_t>(i);
        lease.resolved_output_buffers = 6;
        lease.resolved_input_buffers = 3;
        lease.resolved_tuning = 1;
        lease.lease_token_hi = 100U + i;
        lease.lease_token_lo = 200U + i;
        out.leases.push_back(lease);
      }
    }
    return out;
  }

  bool release(const std::array<std::uint8_t, 16>& group_uuid, std::string* error) override {
    ++release_count;
    released_uuid = group_uuid;
    if (error) {
      error->clear();
    }
    return true;
  }

  DecoderAdmissionResult response{.admitted = true};
  bool auto_leases = true;
  int admit_count = 0;
  int release_count = 0;
  bool last_dry_run = true;
  std::vector<DecoderAdmissionStreamRequest> requests;
  std::array<std::uint8_t, 16> released_uuid{};
};

SimaDecodeOptions decoder_options(SimaDecodeType type = SimaDecodeType::H264) {
  SimaDecodeOptions options;
  options.type = type;
  options.dec_width = 1280;
  options.dec_height = 720;
  options.dec_fps = 60;
  options.raw_output = true;
  return options;
}

ExecutionGraphPlan ordinary_plan(const std::vector<SimaDecodeOptions>& options) {
  ExecutionGraphPlan plan;
  PipelineSegmentPlan segment;
  segment.id = 0;
  for (std::size_t i = 0; i < options.size(); ++i) {
    segment.nodes.push_back(simaai::neat::nodes::SimaDecode(options[i]));
    segment.node_ids.push_back(static_cast<simaai::neat::graph::NodeId>(i));
    plan.node_labels.push_back("decoder_" + std::to_string(i));
  }
  plan.pipeline_segments.push_back(std::move(segment));
  plan.linear_compat = true;
  return plan;
}

ExecutionGraphPlan fused_plan(const SimaDecodeOptions& options) {
  ExecutionGraphPlan plan;
  const auto decoder = simaai::neat::nodes::SimaDecode(options);

  PipelineSegmentPlan consumed;
  consumed.id = 3;
  consumed.nodes.push_back(decoder);
  consumed.node_ids.push_back(3);
  consumed.consumed_by_fused_realtime_ingress = true;
  plan.pipeline_segments.push_back(std::move(consumed));

  PipelineSegmentPlan target;
  target.id = 7;
  simaai::neat::runtime::FusedRealtimeIngress ingress;
  simaai::neat::runtime::FusedRealtimeIngressBranch branch;
  branch.source_node = 3;
  branch.nodes.push_back(std::move(decoder));
  ingress.branches.push_back(std::move(branch));
  target.fused_realtime_ingress = std::move(ingress);
  plan.pipeline_segments.push_back(std::move(target));
  plan.node_labels.resize(4);
  plan.node_labels[3] = "fused_decoder";
  return plan;
}

ExecutionGraphPlan encoder_plan(bool layout_aware) {
  ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
  auto& nodes = plan.pipeline_segments.front().nodes;
  nodes.push_back(marker(layout_aware ? "VideoSenderRawIngress[direct_nv12]"
                                      : "VideoSenderRawIngress[convert_to_nv12]"));
  nodes.push_back(marker("H264EncodeSima"));
  return plan;
}

ExecutionGraphPlan branched_encoder_plan() {
  ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
  auto& source = plan.pipeline_segments.front();
  source.output_edges = {0};

  simaai::neat::runtime::EdgePlan to_fanout;
  to_fanout.from = 0;
  to_fanout.to = 10;
  plan.edges.push_back(std::move(to_fanout));

  simaai::neat::runtime::StageNodePlan fanout;
  fanout.node_id = 10;
  plan.stage_nodes.push_back(std::move(fanout));

  simaai::neat::runtime::EdgePlan to_preproc;
  to_preproc.from = 10;
  to_preproc.to = 20;
  plan.edges.push_back(std::move(to_preproc));

  simaai::neat::runtime::EdgePlan to_encoder;
  to_encoder.from = 10;
  to_encoder.to = 30;
  plan.edges.push_back(std::move(to_encoder));

  PipelineSegmentPlan preproc;
  preproc.id = 20;
  preproc.input_edges = {1};
  preproc.nodes.push_back(marker("Preproc"));
  plan.pipeline_segments.push_back(std::move(preproc));

  PipelineSegmentPlan encoder;
  encoder.id = 30;
  encoder.input_edges = {2};
  encoder.nodes.push_back(marker("VideoSenderRawIngress[convert_to_nv12]"));
  encoder.nodes.push_back(marker("H264EncodeSima"));
  plan.pipeline_segments.push_back(std::move(encoder));
  return plan;
}

void require_throws_with(const std::function<void()>& fn, const std::string& needle,
                         const std::string& where) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(e.what(), needle, where);
    return;
  }
  throw std::runtime_error(where + ": expected exception");
}

void check_h264_h265_and_release() {
  auto backend = std::make_shared<FakeBackend>();
  ExecutionGraphPlan plan =
      ordinary_plan({decoder_options(SimaDecodeType::H264), decoder_options(SimaDecodeType::H265)});

  auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
  require(prepared.eligible_decoders == 2U, "H264/H265 decoder count mismatch");
  require(prepared.reservation && prepared.reservation->active(),
          "accepted admission should return an active reservation");
  require(backend->admit_count == 1 && backend->requests.size() == 2U,
          "all decoder requests should be admitted atomically");
  require(!backend->last_dry_run, "runtime admission should not be a dry run");
  require(
      backend->requests[0].codec == simaai::neat::pipeline_internal::kDecoderAdmissionCodecH264 &&
          backend->requests[1].codec == simaai::neat::pipeline_internal::kDecoderAdmissionCodecH265,
      "decoder codec mapping mismatch");
  for (const auto& request : backend->requests) {
    require(request.width == 1280U && request.height == 720U && request.fps_num == 60U &&
                request.fps_den == 1U,
            "decoder request contract mismatch");
  }
  for (const auto& node : plan.pipeline_segments.front().nodes) {
    const std::string fragment = node->backend_fragment(0);
    require_contains(fragment, "decoder-admission-required=true",
                     "admitted decoder should require its lease");
    require_contains(fragment,
                     "admission-stream-index=", "admitted decoder should bind its stream index");
  }

  prepared.reservation->release();
  prepared.reservation->release();
  require(backend->release_count == 1, "reservation release must be idempotent");
}

void check_shared_reservation_releases_after_last_owner() {
  auto backend = std::make_shared<FakeBackend>();
  ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
  auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
  std::shared_ptr<simaai::neat::runtime::DecoderAdmissionReservation> graph_owner =
      std::move(prepared.reservation);
  auto detached_worker = graph_owner;

  graph_owner.reset();
  require(backend->release_count == 0,
          "a detached graph worker must keep decoder admission reserved");
  detached_worker.reset();
  require(backend->release_count == 1,
          "decoder admission must release after the last detached graph worker exits");
}

void check_non_video_codecs_are_ignored() {
  auto backend = std::make_shared<FakeBackend>();
  ExecutionGraphPlan plan = ordinary_plan(
      {decoder_options(SimaDecodeType::JPEG), decoder_options(SimaDecodeType::MJPEG)});
  auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
  require(prepared.eligible_decoders == 0U && !prepared.reservation,
          "JPEG/MJPEG should not use video decoder admission");
  require(backend->admit_count == 0, "ignored codecs should not call the backend");
}

void check_fused_branch_is_admitted() {
  auto backend = std::make_shared<FakeBackend>();
  ExecutionGraphPlan plan = fused_plan(decoder_options());
  auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
  require(prepared.eligible_decoders == 1U && prepared.reservation &&
              backend->requests.size() == 1U,
          "a decoder copied into a fused branch must be admitted exactly once");
  const auto& node =
      plan.pipeline_segments.back().fused_realtime_ingress->branches.front().nodes.front();
  require_contains(node->backend_fragment(0), "decoder-admission-required=true",
                   "fused decoder should bind the lease into its node");
}

void check_zero_copy_policy_follows_downstream() {
  constexpr std::uint32_t zero_copy_policy =
      simaai::neat::pipeline_internal::kDecoderAdmissionPolicyZeroCopyOutput |
      simaai::neat::pipeline_internal::kDecoderAdmissionPolicyNoOutputCopy;

  {
    auto backend = std::make_shared<FakeBackend>();
    ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
    auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
    require(prepared.reservation && backend->requests.front().requested_policy == zero_copy_policy,
            "terminal raw decoder output should retain zero-copy admission");
    require_contains(plan.pipeline_segments.front().nodes.front()->backend_fragment(0),
                     "zero-copy-output=true",
                     "terminal raw decoder should receive the zero-copy property");
  }

  {
    auto backend = std::make_shared<FakeBackend>();
    ExecutionGraphPlan plan = encoder_plan(false);
    auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
    require(prepared.reservation && backend->requests.front().requested_policy == 0U,
            "legacy encoder path should request packed decoder output");
    require(plan.pipeline_segments.front().nodes.front()->backend_fragment(0).find(
                "zero-copy-output=true") == std::string::npos,
            "legacy encoder path must not enable decoder zero-copy output");
  }

  {
    auto backend = std::make_shared<FakeBackend>();
    ExecutionGraphPlan plan = encoder_plan(true);
    auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
    require(prepared.reservation && backend->requests.front().requested_policy == zero_copy_policy,
            "layout-aware encoder path should retain zero-copy admission");
  }

  {
    auto backend = std::make_shared<FakeBackend>();
    ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
    plan.pipeline_segments.front().nodes.push_back(marker("Preproc"));
    plan.pipeline_segments.front().nodes.push_back(marker("MLA"));
    auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
    require(prepared.reservation && backend->requests.front().requested_policy == zero_copy_policy,
            "Preproc and MLA path should retain zero-copy admission");
  }

  {
    auto backend = std::make_shared<FakeBackend>();
    ExecutionGraphPlan plan = branched_encoder_plan();
    auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
    require(prepared.reservation && backend->requests.front().requested_policy == 0U,
            "a reachable legacy encoder branch should request packed decoder output");
  }
}

void check_output_buffer_floor_preserves_default() {
  const auto admitted_fragment = [](int configured_buffers) {
    auto backend = std::make_shared<FakeBackend>();
    SimaDecodeOptions options = decoder_options();
    options.num_buffers = configured_buffers;
    ExecutionGraphPlan plan = ordinary_plan({options});
    auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
    require(prepared.reservation != nullptr, "buffer-floor case should be admitted");
    return plan.pipeline_segments.front().nodes.front()->backend_fragment(0);
  };

  const std::string unspecified = admitted_fragment(-1);
  require(unspecified.find("num-buffers=") == std::string::npos,
          "an unspecified output pool must preserve the decoder element default");
  require_contains(admitted_fragment(4), "num-buffers=6",
                   "an explicit output pool below the lease floor should be raised");
  require_contains(admitted_fragment(9), "num-buffers=9",
                   "an explicit output pool above the lease floor should be preserved");
}

void check_admission_errors_are_structured() {
  ScopedEnvVar require_admission("SIMA_DECODER_ADMISSION_REQUIRE", "1");
  SimaDecodeOptions options = decoder_options();
  options.dec_fps = 0;
  ExecutionGraphPlan plan = ordinary_plan({options});

  bool threw = false;
  try {
    (void)simaai::neat::runtime::RunCore::start(std::move(plan),
                                                simaai::neat::runtime::RunCoreStartOptions{});
  } catch (const simaai::neat::NeatError& error) {
    threw = true;
    require(error.report().error_code == simaai::neat::error_codes::kPipelineShape,
            "admission failure should retain the graph-start error code");
    require_contains(error.what(), "missing=fps",
                     "structured admission failure should retain the root cause");
  }
  require(threw, "RunCore::start should wrap admission failures as NeatError");
}

void check_missing_contracts() {
  const std::vector<std::pair<const char*, std::function<void(SimaDecodeOptions&)>>> cases = {
      {"width", [](SimaDecodeOptions& options) { options.dec_width = 0; }},
      {"height", [](SimaDecodeOptions& options) { options.dec_height = 0; }},
      {"fps", [](SimaDecodeOptions& options) { options.dec_fps = 0; }},
  };
  for (const auto& [field, clear_field] : cases) {
    auto backend = std::make_shared<FakeBackend>();
    SimaDecodeOptions options = decoder_options();
    clear_field(options);
    ExecutionGraphPlan plan = ordinary_plan({options});
    auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
    require(!prepared.reservation && backend->admit_count == 0,
            std::string("missing ") + field + " should skip optional admission");
    require_contains(prepared.warning, std::string("missing=") + field,
                     std::string("missing ") + field + " warning");
  }

  ScopedEnvVar require_admission("SIMA_DECODER_ADMISSION_REQUIRE", "1");
  auto backend = std::make_shared<FakeBackend>();
  SimaDecodeOptions options = decoder_options();
  options.dec_fps = 0;
  ExecutionGraphPlan plan = ordinary_plan({options});
  require_throws_with(
      [&]() { (void)simaai::neat::runtime::prepare_decoder_admission(plan, backend); },
      "missing=fps", "required missing FPS");
  require(backend->admit_count == 0, "missing FPS must fail before calling the backend");
}

void check_endpoint_policy() {
  auto backend = std::make_shared<FakeBackend>();
  backend->response.admitted = false;
  backend->response.endpoint_missing = true;
  backend->response.error = "not running";
  ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
  auto prepared = simaai::neat::runtime::prepare_decoder_admission(plan, backend);
  require(!prepared.reservation, "optional missing endpoint should not reserve resources");
  require_contains(prepared.warning, "endpoint unavailable", "missing endpoint warning");

  ScopedEnvVar require_admission("SIMA_DECODER_ADMISSION_REQUIRE", "1");
  ExecutionGraphPlan required_plan = ordinary_plan({decoder_options()});
  require_throws_with(
      [&]() { (void)simaai::neat::runtime::prepare_decoder_admission(required_plan, backend); },
      "decoder admission rejected", "required missing endpoint");
}

void check_rejections_and_malformed_leases_release() {
  {
    auto backend = std::make_shared<FakeBackend>();
    backend->response.admitted = false;
    backend->response.error = "capacity exceeded";
    ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
    require_throws_with(
        [&]() { (void)simaai::neat::runtime::prepare_decoder_admission(plan, backend); },
        "capacity exceeded", "capacity rejection");
    require(backend->release_count == 0, "rejected admission has nothing to release");
  }

  {
    auto backend = std::make_shared<FakeBackend>();
    backend->auto_leases = false;
    backend->response.admitted = true;
    ExecutionGraphPlan plan = ordinary_plan({decoder_options()});
    require_throws_with(
        [&]() { (void)simaai::neat::runtime::prepare_decoder_admission(plan, backend); },
        "lease count", "missing lease response");
    require(backend->release_count == 1,
            "accepted malformed response must release its reservation");
  }

  {
    auto backend = std::make_shared<FakeBackend>();
    backend->auto_leases = false;
    backend->response.admitted = true;
    backend->response.leases.resize(2);
    backend->response.leases[0].stream_index = 0;
    backend->response.leases[1].stream_index = 0;
    ExecutionGraphPlan plan = ordinary_plan({decoder_options(), decoder_options()});
    require_throws_with(
        [&]() { (void)simaai::neat::runtime::prepare_decoder_admission(plan, backend); },
        "invalid or duplicate", "duplicate lease response");
    require(backend->release_count == 1,
            "duplicate lease response must release its reservation exactly once");
    for (const auto& node : plan.pipeline_segments.front().nodes) {
      require(node->backend_fragment(0).find("decoder-admission-required=true") ==
                  std::string::npos,
              "failed admission must not partially mutate the plan");
    }
  }
}

void check_sync_cache_rebuild_order() {
  const std::vector<std::string> success =
      simaai::neat::session_test::sync_cache_rebuild_events_for_test(false);
  require(success == std::vector<std::string>({"old-release", "build", "new-release"}),
          "cache replacement must release old runner before building its replacement");

  const std::vector<std::string> failure =
      simaai::neat::session_test::sync_cache_rebuild_events_for_test(true);
  require(failure == std::vector<std::string>({"old-release", "build", "throw"}),
          "failed cache replacement must leave the old runner released and cache empty");
}

} // namespace

int main() {
  try {
    unsetenv("SIMA_DECODER_ADMISSION_DISABLE");
    unsetenv("SIMA_DECODER_ADMISSION_REQUIRE");
    check_h264_h265_and_release();
    check_shared_reservation_releases_after_last_owner();
    check_non_video_codecs_are_ignored();
    check_fused_branch_is_admitted();
    check_zero_copy_policy_follows_downstream();
    check_output_buffer_floor_preserves_default();
    check_missing_contracts();
    check_endpoint_policy();
    check_rejections_and_malformed_leases_release();
    check_admission_errors_are_structured();
    check_sync_cache_rebuild_order();
    std::cout << "[OK] unit_decoder_admission_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "builder/internal/InputSpecSpecialization.h"
#include "gst/internal/ElementCapability.h"
#include "nodes/common/Output.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/groups/internal/VideoSenderRawIngress.h"
#include "nodes/io/Input.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "test_main.h"
#include "test_utils.h"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat {
SampleSpec device_visible_nv12_materialization_spec_or_throw(
    const SampleSpec& source, const char* where);
}

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 30;
constexpr std::string_view kDirectKind = "VideoSenderRawIngress[direct_nv12]";
constexpr std::string_view kMaterializeKind = "VideoSenderRawIngress[materialize_nv12]";
constexpr std::string_view kConvertKind = "VideoSenderRawIngress[convert_to_nv12]";

simaai::neat::OutputSpec raw_spec(std::string format, std::string memory,
                                  simaai::neat::SpecCertainty certainty) {
  simaai::neat::OutputSpec spec;
  spec.payload_type = simaai::neat::PayloadType::Image;
  spec.media_type = "video/x-raw";
  spec.format = std::move(format);
  spec.width = kWidth;
  spec.height = kHeight;
  spec.fps_num = kFps;
  spec.fps_den = 1;
  spec.memory = std::move(memory);
  spec.certainty = certainty;
  return spec;
}

std::size_t count_substrings(const std::string& text, std::string_view needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

std::size_t count_plan_kinds(const simaai::neat::runtime::ExecutionGraphPlan& plan,
                             std::string_view kind) {
  std::size_t count = 0;
  for (const auto& segment : plan.pipeline_segments) {
    for (const auto& node : segment.nodes) {
      if (node && node->kind() == kind) {
        ++count;
      }
    }
  }
  for (const auto& stage : plan.stage_nodes) {
    if (stage.node && stage.node->kind() == kind) {
      ++count;
    }
  }
  return count;
}

bool plan_has_kind(const simaai::neat::runtime::ExecutionGraphPlan& plan, std::string_view kind) {
  return count_plan_kinds(plan, kind) != 0U;
}

std::string plan_backend_for_kind(const simaai::neat::runtime::ExecutionGraphPlan& plan,
                                  std::string_view kind) {
  for (const auto& segment : plan.pipeline_segments) {
    for (const auto& node : segment.nodes) {
      if (node && node->kind() == kind) {
        return node->backend_fragment(99);
      }
    }
  }
  return {};
}

void require_selected_ingress(const simaai::neat::OutputSpec& input, bool layout_aware,
                              std::string_view expected_kind, std::size_t expected_converters) {
  using simaai::neat::internal::InputSpecSpecializationContext;
  using simaai::neat::internal::specialize_nodes_for_input;
  using simaai::neat::nodes::groups::internal::kNeatEncoderInputLayoutAwareCapability;
  using simaai::neat::nodes::groups::internal::VideoSenderRawIngress;

  std::vector<std::shared_ptr<simaai::neat::Node>> nodes = {
      VideoSenderRawIngress(kWidth, kHeight, kFps)};
  InputSpecSpecializationContext context;
  context.set_capability(kNeatEncoderInputLayoutAwareCapability, layout_aware);
  const auto selected = specialize_nodes_for_input(nodes, input, context);

  require(selected.nodes.size() == 1U,
          "adaptive sender ingress must remain one semantic Node after specialization");
  require(selected.nodes.front()->kind() == expected_kind,
          "adaptive sender ingress selected the wrong variant");

  const std::string backend = selected.nodes.front()->backend_fragment(4);
  require(count_substrings(backend, "videoconvert name=") == expected_converters,
          "adaptive sender ingress emitted the wrong converter count");
  require_contains(backend, "format=NV12,width=1280,height=720,framerate=30/1",
                   "adaptive sender ingress must preserve encoder caps");
  require(selected.output_spec.format == "NV12" && selected.output_spec.width == kWidth &&
              selected.output_spec.height == kHeight && selected.output_spec.fps_num == kFps,
          "adaptive sender ingress output contract must remain fixed NV12");
}

simaai::neat::Graph make_raw_sender(int channel = 0) {
  using namespace simaai::neat;

  Graph graph("adaptive_raw_sender");
  graph.add(nodes::groups::internal::VideoSenderRawIngress(kWidth, kHeight, kFps));
  graph.add(nodes::H264EncodeSima(kWidth, kHeight, kFps));
  graph.add(nodes::H264Parse());
  graph.add(nodes::H264Packetize());

  UdpOutputOptions udp;
  udp.host = "127.0.0.1";
  udp.port = 9000 + channel;
  graph.add(nodes::UdpOutput(udp));
  return graph;
}

simaai::neat::Graph make_explicit_push_source(simaai::neat::FormatSpec format,
                                              simaai::neat::InputMemoryPolicy memory,
                                              std::string name = "raw_source") {
  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Image;
  input.format = std::move(format);
  input.width = kWidth;
  input.height = kHeight;
  input.fps_n = kFps;
  input.fps_d = 1;
  input.memory_policy = memory;

  simaai::neat::Graph graph(std::move(name));
  graph.add(simaai::neat::nodes::Input("frames", std::move(input)));
  return graph;
}

simaai::neat::runtime::ExecutionGraphPlan
compile_with_context(const simaai::neat::Graph& graph,
                     const simaai::neat::internal::InputSpecSpecializationContext& context,
                     std::optional<simaai::neat::Sample> seed = std::nullopt) {
  auto plan = simaai::neat::runtime::compile_public_graph(graph, simaai::neat::RunOptions{},
                                                          std::move(seed));
  simaai::neat::runtime::session_test::specialize_input_specs_for_test(&plan, context);
  return plan;
}

} // namespace

RUN_TEST(
    "unit_video_sender_adaptive_ingress_test", ([] {
      using simaai::neat::FormatTag;
      using simaai::neat::InputMemoryPolicy;
      using simaai::neat::internal::InputSpecSpecializationContext;
      using simaai::neat::OutputSpec;
      using simaai::neat::SpecCertainty;
      using simaai::neat::nodes::groups::RtspCodec;
      using simaai::neat::nodes::groups::RtspDecodedInput;
      using simaai::neat::nodes::groups::RtspDecodedInputOptions;
      using simaai::neat::nodes::groups::internal::can_encode_nv12_direct;
      using simaai::neat::nodes::groups::internal::kNeatEncoderInputLayoutAwareCapability;

      InputSpecSpecializationContext layout_aware_context;
      layout_aware_context.set_capability(kNeatEncoderInputLayoutAwareCapability, true);

      // Direct mode is intentionally narrow: a stable raw NV12 contract, a
      // known supported memory domain, and an encoder that explicitly promises
      // to honor GstVideoMeta plane layout.
      const auto derived_system = raw_spec("NV12", "SystemMemory", SpecCertainty::Derived);
      const auto authoritative_system =
          raw_spec("NV12", "SystemMemory", SpecCertainty::Authoritative);
      const auto derived_simaai = raw_spec("NV12", "SimaAI", SpecCertainty::Derived);
      require(!can_encode_nv12_direct(derived_system, true),
              "system-memory NV12 requires the explicit materializing operation");
      require(!can_encode_nv12_direct(authoritative_system, true),
              "authoritative system-memory NV12 still requires materialization");
      require(can_encode_nv12_direct(derived_simaai, true),
              "layout-aware encoder should accept proven SiMaAI NV12");
      require(!can_encode_nv12_direct(derived_system, false),
              "system memory must not bypass the layout capability gate");
      require(!can_encode_nv12_direct(derived_simaai, false),
              "SiMaAI memory must not bypass the layout capability gate");

      require(!can_encode_nv12_direct(raw_spec("NV12", "", SpecCertainty::Authoritative), true),
              "unknown memory must retain the conservative converter");
      require(!can_encode_nv12_direct(raw_spec("NV12", "Any", SpecCertainty::Authoritative), true),
              "noncommittal memory must retain the conservative converter");
      require(!can_encode_nv12_direct(raw_spec("NV12", "SystemMemory", SpecCertainty::Hint), true),
              "hint-only NV12 must retain the conservative converter");
      require(!can_encode_nv12_direct(OutputSpec{}, true),
              "unknown input must retain the conservative converter");

      for (const std::string_view format :
           std::array<std::string_view, 4>{"RGB", "BGR", "GRAY8", "I420"}) {
        require(
            !can_encode_nv12_direct(
                raw_spec(std::string(format), "SystemMemory", SpecCertainty::Authoritative), true),
            std::string(format) + " must retain exactly one conversion to NV12");
      }
      auto encoded = derived_system;
      encoded.media_type = "video/x-h264";
      require(!can_encode_nv12_direct(encoded, true),
              "encoded input must never select the raw direct ingress");

      {
        simaai::neat::SampleSpec source;
        source.kind = simaai::neat::SampleMediaKind::RawVideo;
        source.media_type = "video/x-raw";
        source.format = "NV12";
        source.width = 680;
        source.height = 382;
        source.required_bytes_actual = 680U * 382U * 3U / 2U;
        const auto transport =
            simaai::neat::device_visible_nv12_materialization_spec_or_throw(
                source, "unit VideoSender NV12 materialization");
        require(transport.planes.size() == 2U,
                "NV12 materialization must publish exactly two planes");
        require(transport.planes[0].stride_bytes == 704 &&
                    transport.planes[1].stride_bytes == 704,
                "NV12 materialization must use the Allegro minimum raster pitch");
        require(transport.planes[1].offset_bytes == 704 * 384,
                "NV12 materialization must align the luma storage height");
        require(transport.required_bytes_actual == 704U * 384U * 3U / 2U,
                "NV12 materialization must allocate both aligned surfaces");
      }

      require_selected_ingress(derived_system, true, kMaterializeKind, 0U);
      require_selected_ingress(derived_simaai, true, kDirectKind, 0U);
      require_selected_ingress(derived_system, false, kConvertKind, 1U);
      require_selected_ingress(raw_spec("NV12", "SystemMemory", SpecCertainty::Hint), true,
                               kConvertKind, 1U);
      for (const std::string_view format :
           std::array<std::string_view, 4>{"RGB", "BGR", "GRAY8", "I420"}) {
        require_selected_ingress(
            raw_spec(std::string(format), "SystemMemory", SpecCertainty::Authoritative), true,
            kConvertKind, 1U);
      }

      // Missing factories/properties are represented explicitly, then collapsed
      // to "unsupported" by the sender policy. This is the compatibility path
      // for an older Internals package.
      require(!simaai::neat::internal::element_boolean_capability(
                   "__neat_missing_encoder_for_test__", "input-layout-aware")
                   .has_value(),
              "missing encoder factory must not synthesize a capability");
      require(!simaai::neat::internal::element_boolean_capability(
                   "capsfilter", "__neat_missing_property_for_test__")
                   .has_value(),
              "missing boolean property must not synthesize a capability");

      {
        // A seed is one observation, not a run-wide format/layout promise. It
        // may populate diagnostics but must not freeze an optimized topology.
        const auto public_options =
            simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(kWidth, kHeight,
                                                                               kFps);
        auto public_sender = simaai::neat::nodes::groups::VideoSender(public_options);
        simaai::neat::Sample seed;
        seed.kind = simaai::neat::SampleKind::Tensor;
        seed.tensor = make_nv12_tensor(kWidth, kHeight);
        const auto plan =
            compile_with_context(public_sender, layout_aware_context, std::move(seed));
        require(count_plan_kinds(plan, kConvertKind) == 1U,
                "seed-only NV12 must retain the conservative sender ingress");
      }

      {
        // Explicit InputOptions are a stable linear contract. The compiler may
        // specialize from them without making the application repeat the format
        // in VideoSenderOptions.
        simaai::neat::Graph linear;
        linear.add(make_explicit_push_source(FormatTag::NV12, InputMemoryPolicy::SystemMemory));
        linear.add(make_raw_sender());
        const auto plan = compile_with_context(linear, layout_aware_context);
        require(count_plan_kinds(plan, kMaterializeKind) == 1U,
                "linear system NV12 must select the explicit CMA materializer");
        require(count_plan_kinds(plan, kDirectKind) == 0U,
                "system memory must not bypass the explicit materializer");
      }

      {
        // Version-1 save/load must retain both the semantic adaptive Node and
        // names already transformed into the serialized backend.
        const auto path =
            std::filesystem::temp_directory_path() / "neat_video_sender_adaptive_ingress.json";
        std::error_code ec;
        std::filesystem::remove(path, ec);

        simaai::neat::GraphOptions naming;
        naming.element_name_prefix = "saved_";
        naming.element_name_suffix = "_instance";
        const auto public_options =
            simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(kWidth, kHeight,
                                                                               kFps);
        simaai::neat::Graph original("named_sender", naming);
        original.add(simaai::neat::nodes::groups::VideoSender(public_options));
        original.save(path.string());
        auto loaded = simaai::neat::Graph::load(path.string());
        std::filesystem::remove(path, ec);

        require_contains(loaded.describe(), std::string(kConvertKind),
                         "save/load should rehydrate the semantic adaptive ingress");
        const std::string backend = loaded.describe_backend(false);
        require_contains(backend, "name=saved_n0_raw_input_caps_instance",
                         "v1 load must preserve the transformed raw caps name");
        require_contains(backend, "name=saved_n0_videoconvert_instance",
                         "v1 load must preserve the transformed converter name");
        require_contains(backend, "name=saved_n0_nv12_caps_instance",
                         "v1 load must preserve the transformed NV12 caps name");
        require_contains(backend, "format=NV12,width=1280,height=720,framerate=30/1",
                         "save/load should preserve adaptive ingress geometry");
        require(count_substrings(backend, "videoconvert name=") == 1U,
                "unknown loaded graph should retain one safe converter");
      }

      {
        // Version-2 connected Graphs retain the same names while keeping the
        // SystemMemory materializer explicit.
        const auto path =
            std::filesystem::temp_directory_path() / "neat_video_sender_connected_ingress.json";
        std::error_code ec;
        std::filesystem::remove(path, ec);

        auto source = make_explicit_push_source(FormatTag::NV12, InputMemoryPolicy::SystemMemory);
        auto sender = make_raw_sender();
        simaai::neat::GraphOptions naming;
        naming.element_name_prefix = "saved_";
        naming.element_name_suffix = "_instance";
        simaai::neat::Graph original("named_connected_sender", naming);
        original.connect(source, sender);
        original.save(path.string());
        auto loaded = simaai::neat::Graph::load(path.string());
        std::filesystem::remove(path, ec);

        const auto plan = compile_with_context(loaded, layout_aware_context);
        const std::string materialized_backend = plan_backend_for_kind(plan, kMaterializeKind);
        require(!materialized_backend.empty(),
                "loaded connected SystemMemory sender should materialize explicitly");
        require_contains(materialized_backend, "name=saved_n1_nv12_caps_instance",
                         "materialization must preserve the serialized NV12 caps name");
        require(materialized_backend.find("videoconvert name=") == std::string::npos,
                "same-format NV12 materialization must not add a redundant color conversion");
      }

      {
        // Auto plus the default SiMa pool is the same stable SimaAI contract
        // reported by Input::output_spec(), including through boundary hints.
        simaai::neat::Graph linear;
        linear.add(make_explicit_push_source(FormatTag::NV12, InputMemoryPolicy::Auto));
        linear.add(make_raw_sender());
        const auto plan = compile_with_context(linear, layout_aware_context);
        require(count_plan_kinds(plan, kDirectKind) == 1U,
                "default Auto NV12 InputOptions should select direct SimaAI ingress");
      }

      {
        // A connected edge carries the same static boundary contract into the
        // sender segment.
        auto source = make_explicit_push_source(FormatTag::NV12, InputMemoryPolicy::SystemMemory);
        auto sender = make_raw_sender();
        simaai::neat::Graph connected("connected_nv12_sender");
        connected.connect(source, sender);
        const auto plan = compile_with_context(connected, layout_aware_context);
        require(count_plan_kinds(plan, kMaterializeKind) == 1U,
                "connected SystemMemory NV12 must select explicit materialization");

        require_contains(connected.describe_backend(false), std::string(kMaterializeKind),
                         "connected SystemMemory diagnostic must expose materialization");
      }

      {
        // FanOut is spec-preserving. Adding a preview branch must not discard
        // the format evidence on the sender branch.
        auto source = make_explicit_push_source(FormatTag::NV12, InputMemoryPolicy::SystemMemory);
        auto sender = make_raw_sender();
        simaai::neat::Graph fanout("fanout_nv12_sender");
        fanout.connect(source, sender);
        fanout.connect(source, simaai::neat::nodes::Output("preview"));
        const auto plan = compile_with_context(fanout, layout_aware_context);
        require(count_plan_kinds(plan, kMaterializeKind) == 1U,
                "FanOut should preserve the SystemMemory materialization contract");
        require(plan_has_kind(plan, "FanOut"), "test topology should materialize a FanOut");
      }

      {
        // Decoder-native output is only a hint unless the graph requests
        // explicit tail caps. A hint must never remove the safe converter.
        RtspDecodedInputOptions hint;
        hint.url = "rtsp://example.test/hint";
        hint.dec_width = kWidth;
        hint.dec_height = kHeight;
        hint.dec_fps = kFps;
        auto source = RtspDecodedInput(hint);
        auto sender = make_raw_sender();
        simaai::neat::Graph connected("hint_nv12_sender");
        connected.connect(source, sender);
        const auto plan = compile_with_context(connected, layout_aware_context);
        require(count_plan_kinds(plan, kConvertKind) == 1U,
                "hint-only decoder NV12 should retain the fallback converter");
      }

      {
        // H.265 is an input codec here. After decode, raw NV12 is still encoded
        // by the existing H.264 sender; raw H.265 encoding is not implied.
        RtspDecodedInputOptions h265;
        h265.url = "rtsp://example.test/h265";
        h265.codec = RtspCodec::H265;
        h265.dec_width = kWidth;
        h265.dec_height = kHeight;
        h265.dec_fps = kFps;
        h265.output_caps.enable = true;
        h265.output_caps.format = FormatTag::NV12;
        h265.output_caps.width = kWidth;
        h265.output_caps.height = kHeight;
        h265.output_caps.fps = kFps;
        // Apps leave this as Any: the caps fix the pixel format while retaining
        // SimaDecode's SimaAI memory contract.
        h265.output_caps.memory = simaai::neat::CapsMemory::Any;

        simaai::neat::Graph app("h265_decode_h264_send");
        app.connect(RtspDecodedInput(h265), make_raw_sender());
        const auto plan = compile_with_context(app, layout_aware_context);
        require(count_plan_kinds(plan, kDirectKind) == 1U,
                "explicit H265 decoder NV12 output should select direct raw ingress");
        require(plan_has_kind(plan, "H264Packetize"),
                "decoded H265 raw sender must still packetize H264");
        require(!plan_has_kind(plan, "H265Packetize"),
                "decoded H265 raw sender must not imply raw H265 encoding");
        for (const auto& segment : plan.pipeline_segments) {
          require(!segment.fused_realtime_ingress.has_value(),
                  "raw sender must not be mistaken for encoded passthrough fusion");
        }
      }
    }));

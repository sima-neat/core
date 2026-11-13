#include "../support/CompactFrameFixture.h"
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include "pipeline/internal/sima/PreprocGeometryContract.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "nodes/io/Input.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <cmath>
#include <bit>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <span>
#include <tuple>
#include <vector>
#include <thread>
#include <simaai/scheduler/testing/probe.h>
namespace rt = simaai::scheduler::client;
namespace probe = simaai::scheduler::testing;
namespace core = simaai::neat::pipeline_internal::sima;
namespace lower = core::kernel_lowering;
namespace tr = simaai::gst;
namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class T> T take(rt::Result<T> result) {
  if (auto* e = std::get_if<rt::Error>(&result))
    throw std::runtime_error(e->field + ": " + e->message);
  return std::get<T>(std::move(result));
}
void okay(rt::Status result) {
  if (result)
    throw std::runtime_error(result->field + ": " + result->message);
}
// Scheduler test support uses a real admitted executor and records the real
// provider control packet. No EV pixel implementation is emulated here.
using Capture = probe::Capture;
using Owner = probe::Owner;
using Storage = probe::Storage;
// Frozen compatibility profiles for the two supported installed cohorts, not a
// general rule deriving firmware versions from descriptor arity. Expectations
// are selected before capture, independently of the actual packet under test.
std::optional<std::uint32_t> expected_preproc_abi(std::span<const std::string_view> fields,
                                                  rt::KernelVariantPolicy policy) {
  constexpr std::array<std::string_view, 4> legacy{"source_roi", "source_rois", "destination",
                                                   "padding"};
  constexpr std::array<std::string_view, 5> window{"source_roi", "source_rois", "destination",
                                                   "padding", "resize_window"};
  if (policy != rt::KernelVariantPolicy::FullRecheck)
    return std::nullopt;
  if (std::equal(fields.begin(), fields.end(), legacy.begin(), legacy.end()))
    return 7;
  if (std::equal(fields.begin(), fields.end(), window.begin(), window.end()))
    return 8;
  return std::nullopt;
}
std::uint32_t preproc_wire_abi() {
  static const auto abi = [] {
    const rt::TensorSpec input{{1, 8, 8, 3}, rt::DType::UInt8};
    const std::array parameters{rt::Parameter{"input_format", "rgb"},
                                rt::Parameter{"output_format", "rgb"},
                                rt::Parameter{"destination", rt::ParameterArray{0, 0, 8, 8}}};
    const auto checked =
        take(rt::check_kernel({"preproc", rt::Target::EV74, {&input, 1}, {&input, 1}, parameters}));
    const auto capability = take(rt::bind_kernel_variant(checked, "geometry"));
    const auto profile = expected_preproc_abi(capability.fields(), capability.contract_policy());
    require(profile.has_value(), "unsupported Preproc compatibility profile");
    return *profile;
  }();
  return abi;
}
// Recording only: captures real prepared packets, not EV pixels or firmware.
std::shared_ptr<Capture> preproc_capture() {
  auto capture = std::make_shared<Capture>();
  capture->maximum_abi = preproc_wire_abi();
  return capture;
}
void check_packet_envelope(const probe::Packet& packet, std::size_t expected_bytes) {
  require(expected_bytes >= 312 && packet.abi == preproc_wire_abi() &&
              packet.payload.size() == expected_bytes,
          "Preproc packet disagrees with the supported cohort ABI/length");
  const auto u16 = [&](std::size_t offset) {
    return std::to_integer<unsigned>(packet.payload[offset]) |
           (std::to_integer<unsigned>(packet.payload[offset + 1]) << 8);
  };
  require(u16(4) == preproc_wire_abi() && u16(6) == expected_bytes,
          "encoded Preproc header disagrees with the supported cohort");
  // ABI8 uses formerly reserved bytes without growing the 336-byte base.
  // These legacy-operation checks author no windows; crop checks are separate.
  require(std::all_of(packet.payload.begin() + 296, packet.payload.begin() + 312,
                      [](std::byte value) { return value == std::byte{}; }),
          "legacy geometry retained an unexpected resize window");
}
void geometry_descriptor_compatibility() {
  constexpr auto full = rt::KernelVariantPolicy::FullRecheck;
  const std::vector<std::string_view> legacy{"source_roi", "source_rois", "destination", "padding"};
  auto window = legacy;
  window.push_back("resize_window");
  require(lower::supports_preproc_geometry_fields(legacy) &&
              lower::supports_preproc_geometry_fields(window) &&
              expected_preproc_abi(legacy, full) == 7 && expected_preproc_abi(window, full) == 8,
          "known Preproc descriptor profile rejected");
  require(!lower::supports_preproc_geometry_fields(legacy, true) &&
              lower::supports_preproc_geometry_fields(window, true),
          "crop accepted a descriptor without the resize-window slot");
  for (const auto policy :
       {rt::KernelVariantPolicy::PreserveContract, rt::KernelVariantPolicy::Unavailable})
    require(!expected_preproc_abi(legacy, policy) && !expected_preproc_abi(window, policy),
            "unsupported geometry policy accepted by a compatibility profile");
  const auto rejects = [&](const std::vector<std::string_view>& fields) {
    require(!lower::supports_preproc_geometry_fields(fields) &&
                !lower::supports_preproc_geometry_fields(fields, true) &&
                !expected_preproc_abi(fields, full),
            "unknown geometry descriptor was treated as a compatible extension");
  };
  rejects({});
  for (std::size_t n = 1; n < legacy.size(); ++n)
    rejects({legacy.begin(), legacy.begin() + n});
  for (std::size_t i = 0; i < window.size(); ++i) {
    auto unknown = window;
    unknown[i] = "unknown";
    rejects(unknown);
    if (i > 0) {
      auto reordered = window;
      std::swap(reordered[0], reordered[i]);
      rejects(reordered);
    }
  }
  auto duplicate = window;
  duplicate.back() = "padding";
  rejects(duplicate);
  auto extra = window;
  extra.push_back("future_extension");
  rejects(extra);
}
class PreprocProbe : public probe::AdapterProbe {
  std::shared_ptr<lower::PreprocFrameAdapter> adapter_;

public:
  using CheckedPolicy = lower::PreprocFrameAdapter::CheckedPolicy;
  using PlaneResolver = probe::PlaneResolver;
  explicit PreprocProbe(std::shared_ptr<lower::PreprocFrameAdapter> adapter)
      : probe::AdapterProbe(adapter, adapter->initial_executable()), adapter_(std::move(adapter)) {}
  static auto check_policy(std::shared_ptr<const core::ProcessCvuStagePayload> p) {
    return lower::PreprocFrameAdapter::check_policy(std::move(p));
  }
  static const auto& checked_executable(const CheckedPolicy& p) {
    return lower::PreprocFrameAdapter::checked_executable(p);
  }
  static rt::Result<std::shared_ptr<PreprocProbe>>
  instantiate(std::shared_ptr<const CheckedPolicy> p) {
    auto result = lower::PreprocFrameAdapter::instantiate(std::move(p));
    if (auto* e = std::get_if<rt::Error>(&result))
      return *e;
    return std::make_shared<PreprocProbe>(
        std::get<std::shared_ptr<lower::PreprocFrameAdapter>>(std::move(result)));
  }
  static rt::Result<std::shared_ptr<PreprocProbe>>
  check(std::shared_ptr<const core::ProcessCvuStagePayload> p) {
    auto result = check_policy(std::move(p));
    if (auto* e = std::get_if<rt::Error>(&result))
      return *e;
    return instantiate(std::get<std::shared_ptr<const CheckedPolicy>>(std::move(result)));
  }
  static rt::Result<std::shared_ptr<PreprocProbe>>
  create(std::shared_ptr<const core::ProcessCvuStagePayload> p, probe::Options options) {
    auto result = check(std::move(p));
    if (auto* e = std::get_if<rt::Error>(&result))
      return *e;
    auto value = std::get<std::shared_ptr<PreprocProbe>>(std::move(result));
    if (auto e = value->prepare(std::move(options)))
      return *e;
    return value;
  }
  const auto& initial_executable() const {
    return adapter_->initial_executable();
  }
  auto check_caps(const GstCaps* caps) const {
    return adapter_->check_caps(caps);
  }
  auto accepts_caps(const GstCaps* caps) const {
    return adapter_->accepts_caps(caps);
  }
  static bool owns_publication_field(GQuark q) noexcept {
    return lower::PreprocFrameAdapter::owns_publication_field(q);
  }
};
sima_ev_tensor_desc descriptor(bool input) {
  sima_ev_tensor_desc d{};
  d.shape.rank = 3;
  d.dtype = input ? SIMA_EV_DTYPE_INT8 : SIMA_EV_DTYPE_BF16;
  const int bytes = input ? 1 : 2;
  for (unsigned i = 0; i < 3; ++i) {
    d.shape.sizes[i] = i == 2 ? 3 : 8;
    d.shape.axis_semantics[i] = SIMA_EV_AXIS_H + i;
  }
  d.layout_kind = SIMA_EV_LAYOUT_STRIDED;
  d.layout.strided.strides_bytes[0] = 24 * bytes;
  d.layout.strided.strides_bytes[1] = 3 * bytes;
  d.layout.strided.strides_bytes[2] = bytes;
  return d;
}
std::shared_ptr<core::ProcessCvuStagePayload> config(int capacity = 1, bool quantized = false) {
  auto p = std::make_shared<core::ProcessCvuStagePayload>();
  auto& c = *p;
  c.graph_id = 200;
  c.graph_name = "preproc";
  c.resolved_exec_backend = "EV74";
  c.batch_size = capacity;
  c.input_img_type = "RGB";
  c.output_img_type = "RGB";
  c.scaling_type = "BILINEAR";
  c.padding_type = "CENTER";
  c.aspect_ratio = 1;
  c.normalize = 1;
  c.channel_mean = {0.5F, 0.25F, 0.125F};
  c.channel_stddev = {0.5F, 0.25F, 1.0F};
  c.output_dtype = quantized ? "INT8" : "BF16";
  c.pad_value = 0;
  c.has_q_scale = c.has_q_zp = true;
  c.q_scale = 31.75F;
  c.q_zp = -7;
  c.input_tensors = {{descriptor(true), "UINT8"}};
  c.output_tensors = {{descriptor(!quantized ? false : true), c.output_dtype}};
  return p;
}
float word(const probe::Packet& p, std::size_t offset) {
  require(p.payload.size() >= offset + 4, "short native packet");
  float value;
  std::memcpy(&value, p.payload.data() + offset, 4);
  return value;
}
void set_list(GstStructure* s, const char* name, std::span<const int> entries) {
  GValue list = G_VALUE_INIT;
  g_value_init(&list, GST_TYPE_LIST);
  for (int v : entries) {
    GValue e = G_VALUE_INIT;
    g_value_init(&e, G_TYPE_INT);
    g_value_set_int(&e, v);
    gst_value_list_append_value(&list, &e);
    g_value_unset(&e);
  }
  gst_structure_set_value(s, name, &list);
  g_value_unset(&list);
}
void set_list(GstStructure* s, const char* name, std::initializer_list<int> entries) {
  set_list(s, name, std::span<const int>(entries.begin(), entries.size()));
}
int integer(GstStructure* s, const char* name) {
  int v = 0;
  require(gst_structure_get_int(s, name, &v), name);
  return v;
}
void cold_generic_preparation() {
  // Checking and fact readback require neither a transport nor a live device.
  auto checked = take(PreprocProbe::check(config(3)));
  const auto inspection = take(rt::inspect_kernel(checked->initial_executable()));
  require(inspection.inputs().size() == 1 && inspection.outputs().size() == 1 &&
              inspection.outputs()[0].shape[0] == 3,
          "cold check lost the materialized ROI output capacity");
  const auto capability = take(rt::bind_kernel_variant(checked->initial_executable(), "geometry"));
  require(expected_preproc_abi(capability.fields(), capability.contract_policy()) ==
              preproc_wire_abi(),
          "preproc geometry capability does not describe the canonical update slots");
  const auto facts = inspection.preparation_facts();
  const auto accepted = std::find_if(facts.begin(), facts.end(),
                                     [](const auto& fact) { return fact.name == "accepted_rois"; });
  require(accepted != facts.end() && std::get<rt::ParameterRecords>(accepted->value.value).empty(),
          "empty initial ROI admission omitted its checked empty facts");
  // Both kinds of optional ROI override are genuinely cleared, not inherited
  // from the initial capacity-only admission. The new ports are fully rechecked.
  std::array<std::optional<rt::ParameterValue>, 4> values{
      std::nullopt, std::nullopt, rt::ParameterArray{0, 0, 8, 8}, rt::ParameterArray{0, 0, 0}};
  auto output = inspection.outputs()[0];
  output.shape = std::vector<std::int64_t>{1, 8, 8, 3};
  const auto ordinary =
      take(rt::apply_kernel_variant(capability, {inspection.inputs(), {&output, 1}, values}));
  require(take(rt::inspect_kernel(ordinary)).preparation_facts().empty(),
          "cleared ROI overrides retained stale admission facts");
  require(!rt::same_kernel_contract_resources(ordinary, inspection.executable()),
          "full geometry recheck incorrectly claimed same admitted resources");
  // Retained checked state and readback outlive the caller's preparation holder.
  checked.reset();
  require(inspection.outputs()[0].shape[0] == 3 &&
              std::get<rt::ParameterRecords>(accepted->value.value).empty(),
          "checked facts borrowed request or preparation-holder lifetime");
}
void immutable_policy_instances() {
  auto authored = config(3);
  const auto policy = take(PreprocProbe::check_policy(authored));
  const auto executable = PreprocProbe::checked_executable(*policy);
  // An application alias is not authority to mutate already checked policy.
  authored->input_img_type = "invalid-after-check";
  authored->batch_size = -1;
  authored->channel_mean.clear();
  std::array<std::shared_ptr<PreprocProbe>, 8> instances;
  std::array<std::thread, 8> workers;
  std::array<rt::Status, 8> errors;
  for (std::size_t i = 0; i < workers.size(); ++i)
    workers[i] = std::thread([&, i] {
      auto result = PreprocProbe::instantiate(policy);
      if (const auto* error = std::get_if<rt::Error>(&result))
        errors[i] = *error;
      else
        instances[i] = std::get<std::shared_ptr<PreprocProbe>>(std::move(result));
    });
  for (auto& worker : workers)
    worker.join();
  for (std::size_t i = 0; i < instances.size(); ++i) {
    okay(errors[i]);
    require(instances[i] && instances[i]->initial_executable() == executable &&
                instances[i]->session().size() == 0,
            "policy instantiation rechecked numerical state or admitted a session");
    for (std::size_t j = 0; j < i; ++j)
      require(instances[i] != instances[j], "concurrent policy instantiation shared mutable state");
  }
  auto capture = preproc_capture();
  probe::Options options;
  options.domain = std::make_shared<probe::Domain>(capture);
  std::weak_ptr<probe::Domain> domain = options.domain;
  okay(instances[0]->prepare(options));
  okay(instances[1]->prepare(options));
  require(instances[0]->session().context() == instances[1]->session().context() &&
              instances[0]->session().operation(0).executable() == executable,
          "fresh instances lost shared Runtime domain or exact checked executable identity");
  require(bool(instances[0]->prepare(options)), "admitted instance was prepared twice");
  options.domain.reset();
  instances[0].reset();
  require(!domain.expired(), "closing A stopped or released B's shared domain");
  instances[1].reset();
  require(domain.expired(), "immutable policy retained the last execution session");
  std::weak_ptr<PreprocProbe> expired = instances[2];
  instances[2].reset();
  require(expired.expired(), "checked policy retained an instantiated adapter");

  // A failed direct admission also consumes its instance. Retrying from policy
  // is cheap and must retain the same numerical checked identity.
  auto unsupported = std::make_shared<Capture>();
  unsupported->maximum_abi = 0;
  options.domain = std::make_shared<probe::Domain>(unsupported);
  auto failed = take(PreprocProbe::instantiate(policy));
  require(bool(failed->prepare(options)), "unsupported transport unexpectedly admitted policy");
  options.domain = std::make_shared<probe::Domain>(capture);
  const auto repeated = failed->prepare(options);
  require(repeated && repeated->field == "preproc.preparation",
          "failed instance was reset for a later admission");
  auto retried = take(PreprocProbe::instantiate(policy));
  okay(retried->prepare(options));
  require(retried->initial_executable() == executable,
          "retry rechecked or changed immutable numerical policy");
  std::cout << "Preproc immutable policy: concurrent instances, shared domain and retry PASS\n";
}
void fixed_arithmetic_contract() {
  using namespace simaai::neat;
  constexpr float factor = 0x1.010102p-8f;
  // This source step cannot be recovered by reciprocal-round-tripping its
  // FP32 multiplier. Inspect the original payload, not merely its inverse.
  constexpr auto step_bits = std::uint32_t{0x3c46a60c};
  constexpr float source_step = std::bit_cast<float>(step_bits);
  struct Mode {
    bool quantized, nearest_even, normalized, half;
  };
  for (const auto mode : {Mode{false, false, true, false}, Mode{false, true, true, false},
                          Mode{true, false, true, false}, Mode{true, false, false, false},
                          Mode{false, false, true, true}}) {
    const auto [quantized, nearest_even, normalized, half] = mode;
    auto cfg = config(1, quantized);
    if (half) {
      cfg->output_dtype = "FP16";
      auto output = cfg->output_tensors[0].descriptor();
      output.dtype = SIMA_EV_DTYPE_FP16;
      cfg->output_tensors[0] = {output, "FP16"};
    }
    cfg->scaling_type = "BILINEAR_ANTIALIASED";
    cfg->normalize = normalized ? 1 : 0;
    if (normalized)
      cfg->rescale_factor = factor;
    cfg->channel_stddev[1] = -0.25f;
    cfg->floating_conversion =
        nearest_even ? PreprocFloatingConversion::NearestEven : PreprocFloatingConversion::Legacy;
    if (quantized) {
      cfg->quantization_step = source_step;
      cfg->q_scale = float(1.0f / source_step);
    }
    auto capture = preproc_capture();
    probe::Options options;
    options.capture = capture;
    auto stage = take(PreprocProbe::create(cfg, options));
    auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
    auto* output_meta = gst_structure_new_empty("test");
    probe::Binding binding;
    okay(stage->bind(input, Storage{}, binding, output_meta));
    const auto bytes = binding.operation->output_requirements(0).storage_bytes;
    binding.outputs.push_back({std::make_shared<Owner>(bytes, 0x200000), 0, bytes});
    auto done =
        stage->session().executor(0)->submit(*binding.operation, binding.inputs, binding.outputs);
    okay(done.wait());
    check_packet_envelope(capture->packet, 336);
    const auto bits = [&](std::size_t offset) {
      require(capture->packet.payload.size() >= offset + 4, "short fixed arithmetic packet");
      std::uint32_t value;
      std::memcpy(&value, capture->packet.payload.data() + offset, 4);
      return value;
    };
    require(bits(256) ==
                (1U | (normalized ? 6U : 0U) | (quantized ? 8U : 0U) | (nearest_even ? 16U : 0U)),
            "fixed arithmetic flags or BF16 legacy cast policy changed");
    require(bits(260) == std::bit_cast<std::uint32_t>(normalized ? factor : 1.f),
            "source rescale was folded");
    for (unsigned c = 0; c < 3; ++c) {
      require(word(capture->packet, 264 + 4 * c) ==
                      (normalized ? float(cfg->channel_mean[c]) : 0.f) &&
                  word(capture->packet, 276 + 4 * c) ==
                      (normalized ? float(cfg->channel_stddev[c]) : 1.f),
              "normalization was folded into legacy affine coefficients");
      require(word(capture->packet, 140 + 4 * c) == 0.0f &&
                  word(capture->packet, 152 + 4 * c) == 1.0f,
              "fixed route retained a second legacy normalization");
    }
    if (quantized) {
      require(bits(288) == step_bits, "source quantizer step was reconstructed from reciprocal");
      require(bits(20) == 1U, "quantize-only INT8 was reinterpreted as raw UInt8");
    }
    if (half)
      require(bits(20) == 3U, "source FP16 must keep the binary16 nearest-even sink");
    require(integer(output_meta, "preproc_original_width") == 8,
            "fixed arithmetic lost original-frame metadata");
    gst_structure_free(output_meta);
    gst_buffer_unref(input);
  }
  const auto rejects = [](std::shared_ptr<core::ProcessCvuStagePayload> cfg,
                          const char* expected_field) {
    auto checked = PreprocProbe::check(cfg);
    const auto* error = std::get_if<rt::Error>(&checked);
    require(error && error->field == expected_field, "fixed arithmetic did not reject precisely");
  };
  auto missing_factor = config(1);
  missing_factor->floating_conversion = PreprocFloatingConversion::NearestEven;
  rejects(missing_factor, "rescale_factor");
  auto missing_step = config(1, true);
  missing_step->rescale_factor = factor;
  rejects(missing_step, "quantization_step");
  auto conflicting_step = config(1, true);
  conflicting_step->rescale_factor = factor;
  conflicting_step->quantization_step = 0.5f;
  rejects(conflicting_step, "quantization_step");
  auto invalid_enum = config(1);
  invalid_enum->floating_conversion = static_cast<PreprocFloatingConversion>(999);
  rejects(invalid_enum, "floating_conversion");
  std::cout << "Fixed Preproc source coefficients, casts, step and rejection contracts PASS\n";
}

void contract() {
  for (bool quantized : {false, true}) {
    auto capture = preproc_capture();
    probe::Options options;
    options.capture = capture;
    auto cfg = config(1, quantized);
    auto p = take(PreprocProbe::create(cfg, options));
    auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
    auto* out = gst_structure_new_empty("test");
    probe::Binding binding;
    okay(p->bind(input, Storage{}, binding, out));
    const auto bytes = binding.operation->output_requirements(0).storage_bytes;
    binding.outputs.push_back({std::make_shared<Owner>(bytes, 0x200000), 0, bytes});
    auto completion =
        p->session().executor(0)->submit(*binding.operation, binding.inputs, binding.outputs);
    okay(completion.wait());
    check_packet_envelope(capture->packet, 336);
    // Reference coefficients from legacy preproc.cc: mean*255.0 and
    // (q_scale or 1)/(stddev*255.0), each rounded once to float.
    for (unsigned i = 0; i < 3; ++i) {
      require(word(capture->packet, 140 + 4 * i) == float(cfg->channel_mean[i] * 255.0),
              "legacy offset coefficient");
      require(word(capture->packet, 152 + 4 * i) ==
                  float((quantized ? cfg->q_scale : 1.0F) / (cfg->channel_stddev[i] * 255.0)),
              "legacy combined multiplier coefficient");
    }
    require(word(capture->packet, 136) == 1.0F, "quantization must not scale twice");
    require(integer(out, "preproc_scaled_width") == 8, "ordinary publication");
    gst_structure_free(out);
    gst_buffer_unref(input);
  }
  auto capture = preproc_capture();
  probe::Options options;
  options.capture = capture;
  auto p = take(PreprocProbe::create(config(3), options));
  auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* custom = gst_buffer_add_custom_meta(input, "GstSimaMeta");
  require(custom, "custom meta");
  auto* meta = gst_custom_meta_get_structure(custom);
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 3, nullptr);
  set_list(meta, "preproc_roi_rects", {0, 0, 8, 4, 0, 0, 0, 4, -1, 1, 4, 8});
  set_list(meta, "preproc_roi_batch_indices", {0, 0, 0});
  auto* out = gst_structure_new_empty("test");
  probe::Binding binding;
  okay(p->bind(input, Storage{}, binding, out));
  require(integer(out, "preproc_roi_input_batch_size") == 1,
          "ROI output capacity is not source batch");
  require(integer(out, "preproc_roi_count") == 2 &&
              integer(out, "preproc_roi_dropped_invalid") == 1,
          "kernel-owned filtering");
  require(integer(out, "preproc_scaled_width") == 8 && integer(out, "preproc_scaled_height") == 4 &&
              integer(out, "preproc_pad_top") == 2,
          "first ROI letterbox");
  const auto* rects = gst_structure_get_value(out, "preproc_roi_rects");
  require(rects && gst_value_list_get_size(rects) == 8, "accepted ROI rectangles");
  const auto* affines = gst_structure_get_value(out, "preproc_roi_affines");
  require(affines && gst_value_list_get_size(affines) == 12, "accepted ROI affine count");
  require(g_value_get_double(gst_value_list_get_value(affines, 8)) == -3.0,
          "second ROI destination differs");
  const auto roi_bytes = binding.operation->output_requirements(0).storage_bytes;
  binding.outputs.push_back({std::make_shared<Owner>(roi_bytes, 0x200000), 0, roi_bytes});
  okay(
      p->session().executor(0)->submit(*binding.operation, binding.inputs, binding.outputs).wait());
  check_packet_envelope(capture->packet, 2136);
  set_list(meta, "preproc_roi_batch_indices", {INT_MAX, 0, 0});
  probe::Binding bad;
  require(bool(p->bind(input, Storage{}, bad, out)), "unbounded inferred batch admitted");
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 0, nullptr);
  set_list(meta, "preproc_roi_rects", {});
  set_list(meta, "preproc_roi_batch_indices", {});
  probe::Binding empty;
  okay(p->bind(input, Storage{}, empty, out));
  require(integer(out, "preproc_roi_count") == 0 &&
              gst_value_list_get_size(gst_structure_get_value(out, "preproc_roi_affines")) == 0,
          "empty ROI list retained previous output geometry records");
  gst_structure_free(out);
  gst_buffer_unref(input);
  // Raw legacy INT8 is an unsigned byte stream, not saturating signed math.
  auto raw = config(1, true);
  raw->normalize = 0;
  auto raw_capture = preproc_capture();
  probe::Options raw_options;
  raw_options.capture = raw_capture;
  auto raw_op = take(PreprocProbe::create(raw, raw_options));
  auto* raw_input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* raw_meta = gst_structure_new("test", "preproc_roi_count", G_TYPE_INT, 99, nullptr);
  probe::Binding raw_binding;
  okay(raw_op->bind(raw_input, Storage{}, raw_binding, raw_meta));
  require(!gst_structure_has_field(raw_meta, "preproc_roi_count"),
          "stale ROI-list metadata retained");
  const auto raw_bytes = raw_binding.operation->output_requirements(0).storage_bytes;
  raw_binding.outputs.push_back({std::make_shared<Owner>(raw_bytes, 0x200000), 0, raw_bytes});
  okay(raw_op->session()
           .executor(0)
           ->submit(*raw_binding.operation, raw_binding.inputs, raw_binding.outputs)
           .wait());
  check_packet_envelope(raw_capture->packet, 336);
  std::uint32_t dtype = 99;
  std::memcpy(&dtype, raw_capture->packet.payload.data() + 20, 4);
  require(dtype == 0 && word(raw_capture->packet, 152) == 1.0F,
          "raw byte path must use native UInt8 identity affine");
  gst_structure_free(raw_meta);
  gst_buffer_unref(raw_input);
  auto tile = raw->output_tensors[0].descriptor();
  tile.layout_kind = SIMA_EV_LAYOUT_TILED;
  tile.layout.tiled = {};
  tile.layout.tiled.tile_align_bytes = 16;
  tile.layout.tiled.flags = SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
  tile.layout.tiled.tile_sizes[0] = 8;
  tile.layout.tiled.tile_sizes[1] = 8;
  tile.layout.tiled.tile_sizes[2] = 3;
  raw->output_tensors[0] = raw->output_tensors[0].with_geometry(tile);
  for (std::uint32_t maximum_abi = 4; maximum_abi < preproc_wire_abi(); ++maximum_abi) {
    auto old_firmware = std::make_shared<Capture>();
    old_firmware->maximum_abi = maximum_abi;
    auto old_options = raw_options;
    old_options.capture = old_firmware;
    const auto rejected = PreprocProbe::create(raw, old_options);
    const auto* error = std::get_if<rt::Error>(&rejected);
    require(error && error->field == "prepared.ev74" && old_firmware->admissions > 0 &&
                old_firmware->submissions == 0 && old_firmware->packet.empty(),
            "older transports must reject the selected Preproc ABI before submission");
  }
  auto tiled = take(PreprocProbe::create(raw, raw_options));
  auto* tiled_input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* tiled_meta = gst_structure_new_empty("test");
  probe::Binding tiled_binding;
  okay(tiled->bind(tiled_input, Storage{}, tiled_binding, tiled_meta));
  const auto tiled_bytes = tiled_binding.operation->output_requirements(0).storage_bytes;
  tiled_binding.outputs.push_back({std::make_shared<Owner>(tiled_bytes, 0x200000), 0, tiled_bytes});
  okay(tiled->session()
           .executor(0)
           ->submit(*tiled_binding.operation, tiled_binding.inputs, tiled_binding.outputs)
           .wait());
  std::uint32_t encoding = 0;
  std::memcpy(&dtype, raw_capture->packet.payload.data() + 20, 4);
  std::memcpy(&encoding, raw_capture->packet.payload.data() + 116, 4);
  check_packet_envelope(raw_capture->packet, 336);
  require(dtype == 0 && encoding == 1, "raw tiled bytes must advertise native UInt8 encoding");
  gst_structure_free(tiled_meta);
  gst_buffer_unref(tiled_input);

  auto crop = config();
  crop->scaling_type = "NO_SCALING";
  auto cropped = crop->output_tensors[0].descriptor();
  cropped.shape.sizes[0] = 4;
  cropped.shape.sizes[1] = 6;
  cropped.layout.strided.strides_bytes[0] = 36;
  crop->output_tensors[0] = crop->output_tensors[0].with_geometry(cropped);
  auto crop_op = take(PreprocProbe::create(crop, raw_options));
  auto* crop_input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* crop_meta = gst_structure_new_empty("test");
  probe::Binding crop_binding;
  okay(crop_op->bind(crop_input, Storage{}, crop_binding, crop_meta));
  require(integer(crop_meta, "preproc_scaled_width") == 6 &&
              integer(crop_meta, "preproc_scaled_height") == 4,
          "NO_SCALING must crop source rather than resize");
  double inverse = 0;
  require(gst_structure_get_double(crop_meta, "preproc_affine_scale_x", &inverse) && inverse == 1.0,
          "NO_SCALING inverse map must stay1:1");
  // Virtual ROI copies the top-left subset with its original negative origin;
  // the kernel still owns outside-frame padding and every numerical write.
  auto* crop_custom = gst_buffer_add_custom_meta(crop_input, "GstSimaMeta");
  auto* crop_source = gst_custom_meta_get_structure(crop_custom);
  gst_structure_set(crop_source, "preproc_roi_enable", G_TYPE_BOOLEAN, TRUE, "preproc_roi_x",
                    G_TYPE_INT, -1, "preproc_roi_y", G_TYPE_INT, -2, "preproc_roi_width",
                    G_TYPE_INT, 8, "preproc_roi_height", G_TYPE_INT, 8, nullptr);
  probe::Binding virtual_crop;
  okay(crop_op->bind(crop_input, Storage{}, virtual_crop, crop_meta));
  require(integer(crop_meta, "preproc_roi_x") == -1 &&
              integer(crop_meta, "preproc_roi_width") == 6 &&
              integer(crop_meta, "preproc_roi_height") == 4,
          "virtual crop must retain origin and clip consumed extent");
  require(gst_structure_get_double(crop_meta, "preproc_affine_offset_x", &inverse) &&
              inverse == -1.0,
          "virtual crop inverse map origin");
  gst_structure_free(crop_meta);
  gst_buffer_unref(crop_input);
  std::cout << "Preproc legacy coefficient, ROI geometry/admission/publication "
               "contract PASS (no device pixels)\n";
}
// The legacy translator uses generic preparation. Assert its canonical mapping
// against the actual kernel-produced wire packet, not a second decoder table.
void typed_format_mapping() {
  struct Format {
    const char* name;
    GstVideoFormat video;
    std::uint32_t wire;
  };
  const std::array formats{
      Format{"GRAY", GST_VIDEO_FORMAT_GRAY8, 0}, Format{"RGB", GST_VIDEO_FORMAT_RGB, 1},
      Format{"BGR", GST_VIDEO_FORMAT_BGR, 2},    Format{"NV12", GST_VIDEO_FORMAT_NV12, 3},
      Format{"I420", GST_VIDEO_FORMAT_I420, 4},  Format{"NV16", GST_VIDEO_FORMAT_NV16, 5},
      Format{"IYUV", GST_VIDEO_FORMAT_I420, 4}};
  const std::array interpolation{"NO_SCALING", "NEAREST",   "BILINEAR",
                                 "BICUBIC",    "INTERAREA", "NEAREST_NEIGHBOUR"};
  for (const auto& format : formats) {
    for (std::size_t mode = 0; mode < interpolation.size(); ++mode) {
      auto cfg = config();
      auto& c = *cfg;
      c.input_img_type = format.name;
      c.output_img_type = format.video == GST_VIDEO_FORMAT_GRAY8 ? "GRAY8" : "BGR";
      c.scaling_type = interpolation[mode];
      if (format.video == GST_VIDEO_FORMAT_GRAY8) {
        auto output = c.output_tensors[0].descriptor();
        output.shape.sizes[2] = 1;
        output.layout.strided.strides_bytes[0] = 16;
        output.layout.strided.strides_bytes[1] = 2;
        c.output_tensors[0] = c.output_tensors[0].with_geometry(output);
      }
      auto capture = preproc_capture();
      probe::Options options;
      options.capture = capture;
      auto stage = take(PreprocProbe::create(cfg, options));
      GstVideoInfo info;
      gst_video_info_init(&info);
      require(gst_video_info_set_format(&info, format.video, 8, 8), "format geometry");
      auto* input = gst_buffer_new_allocate(nullptr, info.size, nullptr);
      auto* meta = gst_structure_new_empty("test");
      probe::Binding binding;
      okay(stage->bind(input, Storage{}, binding, meta));
      const auto bytes = binding.operation->output_requirements(0).storage_bytes;
      binding.outputs.push_back({std::make_shared<Owner>(bytes, 0x200000), 0, bytes});
      okay(stage->session()
               .executor(0)
               ->submit(*binding.operation, binding.inputs, binding.outputs)
               .wait());
      check_packet_envelope(capture->packet, 336);
      const auto field = [&](std::size_t offset) {
        std::uint32_t value = 0;
        require(capture->packet.payload.size() >= offset + sizeof(value), "short packet");
        std::memcpy(&value, capture->packet.payload.data() + offset, sizeof(value));
        return value;
      };
      require(field(12) == format.wire &&
                  field(16) == (format.video == GST_VIDEO_FORMAT_GRAY8 ? 0U : 2U) &&
                  field(108) == (mode == 5 ? 1U : mode),
              "typed format/interpolation translation changed wire semantics");
      gst_structure_free(meta);
      gst_buffer_unref(input);
    }
  }
  std::cout << "Preproc typed format/interpolation mappings PASS\n";
}

// Decoder display geometry and allocated rows are independent. Exercise the
// historical pitched NV12 cases through the actual installed preparation path;
// this test observes binding/admission, not EV pixel execution.
void decoder_storage_geometry() {
  for (const auto& [visible_height, storage_height] :
       {std::pair{512, 512}, std::pair{576, 576}, std::pair{480, 512}}) {
    constexpr std::size_t width = 640, pitch = 768;
    const std::size_t uv_offset = pitch * storage_height;
    const std::size_t y_span = pitch * (visible_height - 1) + width;
    const std::size_t uv_span = pitch * (visible_height / 2 - 1) + width;
    const std::size_t bytes = uv_offset + pitch * (storage_height / 2);
    auto cfg = config();
    cfg->input_img_type = "NV12";
    auto input_descriptor = cfg->input_tensors[0].descriptor();
    input_descriptor.shape.sizes[0] = 576;
    input_descriptor.shape.sizes[1] = width;
    input_descriptor.layout.strided.strides_bytes[0] = width * 3;
    cfg->input_tensors[0] = cfg->input_tensors[0].with_geometry(input_descriptor);
    probe::Options options;
    options.capture = preproc_capture();
    auto stage = take(PreprocProbe::create(cfg, options));
    std::unique_ptr<GstStructure, decltype(&gst_structure_free)> published(
        gst_structure_new_empty("test"), gst_structure_free);
    const auto make_input = [&](std::size_t length) {
      std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> result(
          gst_buffer_new_allocate(nullptr, length, nullptr), gst_buffer_unref);
      gsize offsets[GST_VIDEO_MAX_PLANES]{0, uv_offset};
      gint strides[GST_VIDEO_MAX_PLANES]{pitch, pitch};
      require(gst_buffer_add_video_meta_full(result.get(), GST_VIDEO_FRAME_FLAG_NONE,
                                             GST_VIDEO_FORMAT_NV12, width, visible_height, 2,
                                             offsets, strides),
              "decoder NV12 metadata allocation failed");
      return result;
    };
    auto input = make_input(bytes);
    struct ExactStorage final : Storage {
      GstBuffer* carrier = nullptr;
      std::shared_ptr<Owner> owner;
      rt::Result<probe::Buffer> resolve(GstBuffer* buffer, GstMemory* memory, std::size_t offset,
                                        std::size_t length, probe::Access access) const override {
        require(buffer == carrier && memory == gst_buffer_peek_memory(carrier, 0) &&
                    access == probe::Access::Read,
                "decoder binding introduced a copy or lost original memory identity");
        return probe::Buffer{owner, offset, length};
      }
    } storage;
    storage.carrier = input.get();
    storage.owner = std::make_shared<Owner>(bytes, 0x100000);
    probe::Binding binding;
    okay(stage->bind(input.get(), storage, binding, published.get()));
    require(binding.inputs.size() == 2 && binding.inputs[0].offset_bytes == 0 &&
                binding.inputs[1].offset_bytes == uv_offset &&
                binding.inputs[0].length_bytes == y_span &&
                binding.inputs[1].length_bytes == uv_span &&
                binding.operation->input_requirements(0).storage_bytes == y_span &&
                binding.operation->input_requirements(1).storage_bytes == uv_span,
            "decoder pitch/plane offset/visible extent lost at preparation boundary");
    require(integer(published.get(), "preproc_original_width") == int(width) &&
                integer(published.get(), "preproc_original_height") == visible_height,
            "decoder allocated rows incorrectly became visible image geometry");
    auto short_input = make_input(uv_offset + uv_span - 1);
    probe::Binding rejected;
    require(stage->bind(short_input.get(), Storage{}, rejected, published.get()).has_value(),
            "cached NV12 geometry admitted a one-byte-short current allocation");
    probe::Binding replay;
    okay(stage->bind(input.get(), storage, replay, published.get()));
    require(replay.operation == binding.operation,
            "rejected NV12 backing displaced the valid prepared geometry");
  }
  std::cout << "Decoder explicit pitch, storage rows and visible NV12 geometry PASS\n";
}

void checked_plane_resolver() {
  using Preproc = PreprocProbe;
  struct Format {
    const char* name;
    GstVideoFormat video;
    guint planes;
  };
  for (const auto& format :
       {Format{"NV12", GST_VIDEO_FORMAT_NV12, 2}, Format{"NV16", GST_VIDEO_FORMAT_NV16, 2},
        Format{"I420", GST_VIDEO_FORMAT_I420, 3}}) {
    auto cfg = config();
    cfg->input_img_type = format.name;
    probe::Options options;
    options.capture = preproc_capture();
    auto stage = take(Preproc::create(cfg, options));
    auto* input = gst_buffer_new();
    gsize offsets[GST_VIDEO_MAX_PLANES]{};
    gint strides[GST_VIDEO_MAX_PLANES]{};
    std::array<std::size_t, 3> lengths{};
    std::size_t total = 0;
    for (guint i = 0; i < format.planes; ++i) {
      offsets[i] = total;
      strides[i] = i && format.video == GST_VIDEO_FORMAT_I420 ? 6 : 12;
      const auto rows = i && format.video != GST_VIDEO_FORMAT_NV16 ? 4 : 8;
      lengths[i] = rows * strides[i];
      total += lengths[i];
      auto* memory = gst_allocator_alloc(nullptr, lengths[i], nullptr);
      require(memory != nullptr, "plane allocation failed");
      gst_buffer_append_memory(input, memory);
    }
    auto* video = gst_buffer_add_video_meta_full(input, GST_VIDEO_FRAME_FLAG_NONE, format.video, 8,
                                                 8, format.planes, offsets, strides);
    require(video != nullptr, "plane VideoMeta missing");
    auto* published = gst_structure_new_empty("test");

    struct Imports {
      GstBuffer* carrier;
      std::array<std::size_t, 3> lengths;
      mutable std::size_t calls = 0;
      std::size_t reject = SIZE_MAX;
      std::array<std::shared_ptr<Owner>, 3> owners{};
      bool collect_separately = false;
      mutable std::vector<probe::Buffer> collected;
      static rt::Status append(const void* context, std::size_t plane, GstMemory* memory,
                               std::size_t offset, std::size_t bytes, probe::Binding& binding) {
        const auto& self = *static_cast<const Imports*>(context);
        require(plane == self.calls++, "plane callback lost logical port order");
        require(plane < self.owners.size() && self.owners[plane], "unexpected plane import");
        require(memory == gst_buffer_peek_memory(self.carrier, plane) && offset == 0 && bytes > 0 &&
                    bytes <= self.lengths[plane],
                "checked plane did not preserve current memory/range");
        if (plane == self.reject)
          return rt::Error{rt::ErrorCode::InvalidRequest, "input.loan",
                           "rejected current plane loan"};
        auto& ports = self.collect_separately ? self.collected : binding.inputs;
        ports.push_back({self.owners[plane], offset, bytes});
        return {};
      }
    } imports{input, lengths};
    for (guint i = 0; i < format.planes; ++i)
      imports.owners[i] = std::make_shared<Owner>(lengths[i], 0x100000 + i * 4096);
    const Preproc::PlaneResolver resolver{&imports, &Imports::append};
    const auto bind = [&] {
      imports.calls = 0;
      probe::Binding result;
      okay(stage->bind(input, resolver, result, published));
      require(imports.calls == format.planes && result.inputs.size() == format.planes,
              "cached preparation bypassed current plane admission");
      for (guint i = 0; i < format.planes; ++i)
        require(result.inputs[i].owner == imports.owners[i] &&
                    result.inputs[i].length_bytes ==
                        result.operation->input_requirements(i).storage_bytes,
                "resolver-owned plane or current requirement was replaced");
      return result;
    };
    const auto pitched = bind().operation;
    require(bind().operation == pitched, "stable plane import rebuilt numerical preparation");
    // The stage may collect checked ports directly into its own indexed
    // admission records. Preproc must not depend on temporary Binding views.
    imports.collect_separately = true;
    imports.calls = 0;
    probe::Binding indexed;
    okay(stage->bind(input, resolver, indexed, published));
    require(indexed.inputs.empty() && indexed.operation == pitched &&
                imports.collected.size() == format.planes,
            "direct plane collection required temporary input views");
    for (guint i = 0; i < format.planes; ++i)
      require(imports.collected[i].owner == imports.owners[i] &&
                  imports.collected[i].length_bytes == pitched->input_requirements(i).storage_bytes,
              "direct plane collection lost its checked port");
    imports.collected.clear();
    imports.collect_separately = false;
    video->stride[0] = 8;
    const auto dense = bind().operation;
    require(dense != pitched, "changed VideoMeta pitch reused stale preparation");

    // A failure after one successfully admitted plane must not commit a new
    // semantic variant. Existing in-flight bindings retain their own owners.
    video->stride[0] = 10;
    imports.calls = 0;
    imports.reject = 1;
    probe::Binding rejected;
    const auto error = stage->bind(input, resolver, rejected, published);
    require(error && error->field == "input.loan" && imports.calls == 2,
            "plane admission failure was swallowed");
    imports.reject = SIZE_MAX;
    video->stride[0] = 12;
    require(bind().operation == pitched, "failed plane loan evicted previous preparation");
    video->stride[0] = 8;
    require(bind().operation == dense, "failed plane loan evicted current preparation");

    // Camera/pool replacement changes current memory, not the numerical
    // contract. The resolver must receive the new plane even on a cache hit.
    auto* replacement = gst_allocator_alloc(nullptr, lengths[1], nullptr);
    require(replacement != nullptr, "replacement plane allocation failed");
    gst_buffer_replace_memory(input, 1, replacement);
    require(bind().operation == dense,
            "plane replacement unnecessarily rebuilt numerical preparation");

    imports.calls = 0;
    video->stride[0] = -1;
    probe::Binding invalid;
    require(bool(stage->bind(input, resolver, invalid, published)) && imports.calls == 0,
            "invalid authoritative pitch reached allocation admission");
    video->stride[0] = 8;
    require(bool(stage->bind(input, Preproc::PlaneResolver{}, invalid, published)),
            "missing plane resolver accepted");
    gst_structure_free(published);
    gst_buffer_unref(input);
  }
  std::cout << "Preproc checked plane resolver/variant transaction PASS\n";
}

void dynamic_preparation_cache() {
  auto capture = preproc_capture();
  probe::Options options;
  options.domain = std::make_shared<probe::Domain>(capture);
  const auto policy = take(PreprocProbe::check_policy(config(3)));
  auto first_stage = take(PreprocProbe::instantiate(policy));
  auto stage = take(PreprocProbe::instantiate(policy));
  okay(first_stage->prepare(options));
  okay(stage->prepare(options));
  require(first_stage->session().context() == stage->session().context(),
          "Preproc stages did not share supplied context");
  first_stage.reset();
  auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 1, nullptr);
  set_list(meta, "preproc_roi_rects", {0, 0, 8, 4});
  set_list(meta, "preproc_roi_batch_indices", {0});
  auto* published = gst_structure_new_empty("test");
  const auto bind = [&] {
    probe::Binding result;
    okay(stage->bind(input, Storage{}, result, published));
    return result;
  };
  auto initial = bind();
  gst_structure_set(meta, "frame-id", G_TYPE_INT64, gint64(99), nullptr);
  auto repeated = bind();
  require(repeated.operation == initial.operation,
          "unchanged ROI signature rebuilt immutable preparation");
  require(repeated.inputs[0].owner != initial.inputs[0].owner,
          "reused preparation cached an invocation's storage owner");
  // Published values belong to the carrier, not the cached template. A consumer
  // may mutate them while another frame reuses this checked invocation.
  auto* retained = gst_structure_copy(published);
  gst_structure_set(published, "preproc_affine_m00", G_TYPE_DOUBLE, 99.0, "frame-id", G_TYPE_INT,
                    42, nullptr);
  set_list(published, "preproc_roi_rects", {9, 9, 9, 9});
  set_list(published, "preproc_roi_affines", {9});
  require(bind().operation == initial.operation,
          "publication mutation rebuilt the checked operation");
  require(integer(published, "frame-id") == 42,
          "cached facts overwrote current carrier correlation");
  require(gst_value_compare(gst_structure_get_value(published, "preproc_roi_rects"),
                            gst_structure_get_value(retained, "preproc_roi_rects")) ==
                  GST_VALUE_EQUAL &&
              gst_value_compare(gst_structure_get_value(published, "preproc_roi_affines"),
                                gst_structure_get_value(retained, "preproc_roi_affines")) ==
                  GST_VALUE_EQUAL &&
              gst_value_compare(gst_structure_get_value(published, "preproc_affine_m00"),
                                gst_structure_get_value(retained, "preproc_affine_m00")) ==
                  GST_VALUE_EQUAL,
          "consumer mutation corrupted cached publication");
  GValue wide_rects = G_VALUE_INIT;
  g_value_init(&wide_rects, GST_TYPE_LIST);
  for (gint64 coordinate : {0, 0, 8, 4}) {
    GValue entry = G_VALUE_INIT;
    g_value_init(&entry, G_TYPE_INT64);
    g_value_set_int64(&entry, coordinate);
    gst_value_list_append_value(&wide_rects, &entry);
    g_value_unset(&entry);
  }
  gst_structure_set_value(meta, "preproc_roi_rects", &wide_rects);
  g_value_unset(&wide_rects);
  probe::Binding invalid_type;
  require(bool(stage->bind(input, Storage{}, invalid_type, published)),
          "numerically equal wrong-typed ROI list bypassed admission through cache");
  set_list(meta, "preproc_roi_rects", {0, 0, 8, 4});
  require(bind().operation == initial.operation,
          "wrong-typed signature evicted the previous valid preparation");
  set_list(meta, "preproc_roi_rects", {1, 0, 4, 8});
  auto moved = bind();
  require(moved.operation != initial.operation && integer(published, "preproc_roi_x") == 1,
          "changed ROI reused stale preparation or facts");
  for (unsigned iteration = 0; iteration < 8; ++iteration) {
    if (iteration % 2 == 0) {
      set_list(meta, "preproc_roi_rects", {0, 0, 8, 4});
      require(bind().operation == initial.operation,
              "alternating ROI rebuilt the previous immutable variant");
    } else {
      set_list(meta, "preproc_roi_rects", {1, 0, 4, 8});
      require(bind().operation == moved.operation,
              "alternating ROI rebuilt the current immutable variant");
    }
  }
  struct RejectingStorage : Storage {
    rt::Result<probe::Buffer> resolve(GstBuffer*, GstMemory*, std::size_t, std::size_t,
                                      probe::Access) const override {
      return rt::Error{rt::ErrorCode::InvalidRequest, "storage", "rejected test loan"};
    }
  };
  set_list(meta, "preproc_roi_rects", {2, 0, 4, 8});
  probe::Binding rejected_storage;
  require(bool(stage->bind(input, RejectingStorage{}, rejected_storage, published)),
          "failed storage import unexpectedly admitted");
  set_list(meta, "preproc_roi_rects", {0, 0, 8, 4});
  require(bind().operation == initial.operation,
          "failed frame evicted the previous usable variant");
  set_list(meta, "preproc_roi_rects", {1, 0, 4, 8});
  require(bind().operation == moved.operation, "failed frame evicted the current usable variant");

  // A cached geometry never stands in for the current frame's storage check.
  for (bool split_plane : {false, true}) {
    auto* bad_input = gst_buffer_new_allocate(nullptr, 96, nullptr);
    if (split_plane) {
      auto* tail = gst_allocator_alloc(nullptr, 96, nullptr);
      gst_buffer_append_memory(bad_input, tail);
    }
    auto* bad_meta =
        gst_custom_meta_get_structure(gst_buffer_add_custom_meta(bad_input, "GstSimaMeta"));
    gst_structure_set(bad_meta, "preproc_roi_count", G_TYPE_INT, 1, nullptr);
    set_list(bad_meta, "preproc_roi_rects", {1, 0, 4, 8});
    set_list(bad_meta, "preproc_roi_batch_indices", {0});
    probe::Binding bad_storage;
    require(bool(stage->bind(bad_input, Storage{}, bad_storage, published)),
            split_plane ? "cached plane crossed independent memory spans"
                        : "cached geometry admitted insufficient current storage");
    gst_buffer_unref(bad_input);
    require(bind().operation == moved.operation,
            "bad current storage changed the valid cached variant");
  }
  gsize offsets[GST_VIDEO_MAX_PLANES]{};
  gint strides[GST_VIDEO_MAX_PLANES]{32};
  auto* video_meta = gst_buffer_add_video_meta_full(
      input, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_RGB, 8, 8, 1, offsets, strides);
  require(video_meta, "add authoritative video metadata");
  probe::Binding invalid_pitch;
  require(bool(stage->bind(input, Storage{}, invalid_pitch, published)),
          "authoritative pitched image exceeded current storage");
  require(gst_buffer_remove_meta(input, &video_meta->meta), "remove video metadata");
  require(bind().operation == moved.operation,
          "invalid per-frame video geometry replaced a valid variant");
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_STRING, "invalid", nullptr);
  probe::Binding invalid;
  require(bool(stage->bind(input, Storage{}, invalid, published)),
          "malformed changed signature escaped admission");
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 1, nullptr);
  require(bind().operation == moved.operation, "failed admission evicted the valid cached variant");
  for (int pad : {-1, 256}) {
    gst_structure_set(meta, "preproc_roi_pad_value", G_TYPE_INT, pad, nullptr);
    probe::Binding rejected_pad;
    require(bool(stage->bind(input, Storage{}, rejected_pad, published)),
            "typed translation narrowed out-of-range padding before admission");
  }
  gst_structure_set(meta, "preproc_roi_pad_value", G_TYPE_INT, 17, nullptr);
  auto padding = bind();
  require(padding.operation != moved.operation && integer(published, "preproc_roi_pad_value") == 17,
          "changed padding reused stale preparation");
  GstCaps* smaller =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", "width", G_TYPE_INT, 8,
                          "height", G_TYPE_INT, 4, "framerate", GST_TYPE_FRACTION, 30, 1, nullptr);
  okay(stage->accepts_caps(smaller));
  require(bind().operation == padding.operation &&
              integer(published, "preproc_original_height") == 8,
          "caps query changed active image geometry");
  auto* too_large = gst_caps_copy(smaller);
  gst_structure_set(gst_caps_get_structure(too_large, 0), "width", G_TYPE_INT, 9, nullptr);
  require(bool(stage->accepts_caps(too_large)), "query accepted geometry outside envelope");
  require(bool(stage->check_caps(too_large)), "event accepted geometry outside envelope");
  gst_caps_unref(too_large);
  require(bind().operation == padding.operation, "rejected caps changed the active preparation");
  okay(stage->check_caps(smaller));
  gst_caps_unref(smaller);
  auto resized = bind();
  require(resized.operation != padding.operation &&
              integer(published, "preproc_original_height") == 4,
          "changed image geometry reused stale immutable preparation");
  require(bind().operation == resized.operation, "stable changed geometry was not cached");
  auto* restored_caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", "width",
                                            G_TYPE_INT, 8, "height", G_TYPE_INT, 8, nullptr);
  okay(stage->check_caps(restored_caps));
  require(bind().operation == padding.operation,
          "small-to-large transition rebuilt the retained geometry");
  gst_structure_set(gst_caps_get_structure(restored_caps, 0), "height", G_TYPE_INT, 4, nullptr);
  okay(stage->check_caps(restored_caps));
  require(bind().operation == resized.operation,
          "large-to-small transition rebuilt the retained geometry");
  gst_caps_unref(restored_caps);
  const auto bytes = padding.operation->output_requirements(0).storage_bytes;
  padding.outputs.push_back({std::make_shared<Owner>(bytes, 0x200000), 0, bytes});
  okay(stage->session()
           .executor(0)
           ->submit(*padding.operation, padding.inputs, padding.outputs)
           .wait());
  require(!capture->packet.empty(), "destroying one stage stopped shared context");
  stage.reset();
  require(integer(retained, "preproc_roi_x") == 0 &&
              gst_value_list_get_size(gst_structure_get_value(retained, "preproc_roi_affines")) ==
                  6,
          "eviction or stage destruction invalidated published metadata");
  gst_structure_free(retained);
  gst_structure_free(published);
  gst_buffer_unref(input);
  std::cout << "Preproc bounded immutable variant cache and shared context PASS\n";
}

// Defaults are CAPS-owned, while each current frame can carry independent
// geometry/pitch. Returning to defaults must never reuse an override as a base.
void default_geometry_and_padding() {
  probe::Options options;
  options.capture = preproc_capture();
  for (const auto& [name, x, y] :
       {std::tuple{"center", 2, 2}, std::tuple{"topleft", 0, 0}, std::tuple{"topright", 4, 0},
        std::tuple{"bottomleft", 0, 4}, std::tuple{"bottomright", 4, 4}}) {
    auto cfg = config();
    cfg->padding_type = name;
    auto stage = take(PreprocProbe::create(cfg, options));
    auto* input = gst_buffer_new_allocate(nullptr, 256, nullptr);
    auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
    auto* out = gst_structure_new_empty("test");
    gst_structure_set(meta, "preproc_roi_enable", G_TYPE_BOOLEAN, TRUE, "preproc_roi_x", G_TYPE_INT,
                      0, "preproc_roi_y", G_TYPE_INT, 0, "preproc_roi_width", G_TYPE_INT, 4,
                      "preproc_roi_height", G_TYPE_INT, 8, nullptr);
    const auto bind = [&] {
      probe::Binding result;
      okay(stage->bind(input, Storage{}, result, out));
      return result;
    };
    bind();
    require(integer(out, "preproc_pad_left") == x,
            "compiled horizontal padding policy changed placement");
    gst_structure_set(meta, "preproc_roi_width", G_TYPE_INT, 8, "preproc_roi_height", G_TYPE_INT, 4,
                      nullptr);
    bind();
    require(integer(out, "preproc_pad_top") == y,
            "compiled vertical padding policy changed placement");
    gst_structure_set(meta, "preproc_roi_width", G_TYPE_INT, 4, "preproc_roi_height", G_TYPE_INT, 8,
                      "preproc_roi_source_width", G_TYPE_INT, 4, nullptr);
    bind();
    require(integer(out, "preproc_original_width") == 4,
            "current source dimension override ignored");
    gst_structure_remove_field(meta, "preproc_roi_source_width");
    const auto restored = bind().operation;
    require(integer(out, "preproc_original_width") == 8,
            "source override poisoned canonical geometry");
    // A successful pitched frame, not only rejected pitch, must leave the
    // subsequent packed frame's strides untouched.
    gsize offsets[GST_VIDEO_MAX_PLANES]{};
    gint strides[GST_VIDEO_MAX_PLANES]{32};
    auto* video = gst_buffer_add_video_meta_full(input, GST_VIDEO_FRAME_FLAG_NONE,
                                                 GST_VIDEO_FORMAT_RGB, 8, 8, 1, offsets, strides);
    require(video, "add pitched video metadata");
    require(bind().operation != restored, "current authoritative pitch reused packed preparation");
    require(gst_buffer_remove_meta(input, &video->meta), "remove pitched metadata");
    require(bind().operation == restored,
            "successful video override poisoned default pitch or cached variant");
    // A locally proven fresh destination produces exactly the same facts as
    // dirty-output publication, while preserving unknown extension fields.
    auto* fresh = gst_structure_new_empty("test");
    auto* dirty = gst_structure_new_empty("test");
    for (auto* destination : {fresh, dirty})
      gst_structure_set(destination, "preproc_vendor_extension", G_TYPE_STRING, "retained",
                        nullptr);
    gst_structure_set(dirty, "preproc_roi_count", G_TYPE_INT, 99, "preproc_roi_affines", G_TYPE_INT,
                      99, nullptr);
    probe::Binding fresh_binding, dirty_binding;
    okay(stage->bind(input, Storage{}, fresh_binding, fresh, true));
    okay(stage->bind(input, Storage{}, dirty_binding, dirty));
    require(gst_structure_is_equal(fresh, dirty),
            "fresh publication differs from stale-field cleanup");
    gst_structure_free(fresh);
    gst_structure_free(dirty);
    for (gint i = 0; i < gst_structure_n_fields(out); ++i)
      require(PreprocProbe::owns_publication_field(
                  g_quark_from_string(gst_structure_nth_field_name(out, i))),
              "publication field missing from ownership set");
    require(
        !PreprocProbe::owns_publication_field(g_quark_from_string("preproc_vendor_extension")) &&
            !PreprocProbe::owns_publication_field(g_quark_from_string("frame-id")) &&
            !PreprocProbe::owns_publication_field(0),
        "publisher claimed unknown extension or correlation metadata");
    require(PreprocProbe::owns_publication_field(
                g_quark_from_string("preproc_roi_source_stride_bytes")),
            "conditionally cleared field missing from publication ownership");
    gst_structure_free(out);
    gst_buffer_unref(input);
  }
  auto bad = config();
  bad->padding_type = "unknown";
  require(std::holds_alternative<rt::Error>(PreprocProbe::create(bad, options)),
          "invalid fixed padding policy escaped preparation");
}

void borrowed_roi_indices() {
  probe::Options options;
  options.capture = preproc_capture();
  auto stage = take(PreprocProbe::create(config(3), options));
  auto* input = gst_buffer_new_allocate(nullptr, 384, nullptr);
  auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
  auto* out = gst_structure_new_empty("test");
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 2, nullptr);
  set_list(meta, "preproc_roi_rects", {0, 0, 4, 4, 0, 0, 4, 4});
  const auto bind = [&] {
    probe::Binding result;
    return stage->bind(input, Storage{}, result, out);
  };
  for (bool array : {false, true}) {
    GValue values = G_VALUE_INIT;
    g_value_init(&values, array ? GST_TYPE_ARRAY : GST_TYPE_LIST);
    for (int index : {-1, 1}) {
      GValue value = G_VALUE_INIT;
      g_value_init(&value, G_TYPE_INT);
      g_value_set_int(&value, index);
      if (array)
        gst_value_array_append_value(&values, &value);
      else
        gst_value_list_append_value(&values, &value);
      g_value_unset(&value);
    }
    gst_structure_set_value(meta, "preproc_roi_batch_indices", &values);
    g_value_unset(&values);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      okay(bind());
      require(integer(out, "preproc_roi_input_batch_size") == 2 &&
                  integer(out, "preproc_roi_count") == 1 &&
                  integer(out, "preproc_roi_dropped_invalid") == 1,
              "borrowed ROI list changed inferred batch or invalid filtering");
    }
  }
  gst_structure_remove_field(meta, "preproc_roi_batch_indices");
  okay(bind());
  require(integer(out, "preproc_roi_input_batch_size") == 1 &&
              integer(out, "preproc_roi_count") == 2,
          "absent optional indices did not mean source frame zero");
  set_list(meta, "preproc_roi_batch_indices", {0});
  require(bool(bind()), "borrowed ROI indices ignored wrong list length");
  gst_structure_free(out);
  gst_buffer_unref(input);
}

void publication_ownership_and_replacement() {
  probe::Options options;
  options.capture = preproc_capture();
  auto stage = take(PreprocProbe::create(config(), options));
  auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 1, nullptr);
  set_list(meta, "preproc_roi_rects", {0, 0, 4, 4});
  set_list(meta, "preproc_roi_batch_indices", {0});
  auto* out =
      gst_structure_new("test", "preproc_customer_extension", G_TYPE_STRING, "keep", nullptr);
  const auto bind = [&] {
    probe::Binding binding;
    okay(stage->bind(input, Storage{}, binding, out));
  };
  bind();
  auto* retained = gst_structure_copy(out);
  auto* held = gst_structure_new_empty("test");
  {
    probe::Binding binding;
    okay(stage->bind(input, Storage{}, binding, held));
  }
  // Replacing fields on one emitted envelope must not affect the immutable
  // prepared variant or other envelopes. Wrong old types must also be replaced.
  set_list(out, "preproc_roi_rects", {99});
  gst_structure_set(out, "preproc_roi_count", G_TYPE_STRING, "stale", "preproc_roi_batch_index",
                    G_TYPE_STRING, "stale", nullptr);
  bind();
  require(integer(out, "preproc_roi_count") == 1 && integer(out, "preproc_roi_batch_index") == 0 &&
              gst_value_list_get_size(gst_structure_get_value(out, "preproc_roi_rects")) == 4,
          "cached publication retained downstream edits");
  // Empty-list mode still owns all list fields but has no selected primary ROI.
  gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 0, nullptr);
  set_list(meta, "preproc_roi_rects", {});
  set_list(meta, "preproc_roi_batch_indices", {});
  bind();
  require(integer(out, "preproc_roi_count") == 0 &&
              !gst_structure_has_field(out, "preproc_roi_batch_index"),
          "empty list retained selected ROI");
  // Leaving list mode removes only list-owned fields; extension data survives.
  gst_structure_remove_all_fields(meta);
  bind();
  for (const char* name :
       {"preproc_roi_list_enable", "preproc_roi_count", "preproc_roi_capacity",
        "preproc_valid_batch_count", "preproc_roi_batch_indices", "preproc_roi_rects",
        "preproc_roi_affines", "preproc_roi_input_batch_size", "preproc_roi_input_count",
        "preproc_roi_dropped_invalid", "preproc_roi_dropped_overflow",
        "preproc_roi_source_stride_bytes", "preproc_roi_pad_value", "preproc_roi_batch_index"})
    require(!gst_structure_has_field(out, name),
            "ordinary publication retained stale optional field");
  require(std::string_view(gst_structure_get_string(out, "preproc_customer_extension")) == "keep",
          "publication cleanup removed extension metadata");
  // The copied output outlives cache eviction and the entire stage generation.
  stage.reset();
  gst_structure_free(out);
  gst_buffer_unref(input);
  require(integer(held, "preproc_roi_count") == 1 &&
              gst_value_list_get_size(gst_structure_get_value(held, "preproc_roi_rects")) == 4,
          "original publication depends on stage lifetime");
  gst_structure_free(held);
  require(integer(retained, "preproc_roi_count") == 1 &&
              gst_value_list_get_size(gst_structure_get_value(retained, "preproc_roi_affines")) ==
                  6,
          "retained publication depends on stage lifetime");
  set_list(retained, "preproc_roi_rects", {1, 1, 2, 2});
  gst_structure_free(retained);
  std::cout << "Preproc publication replacement and retained ownership PASS\n";
}

// Incoming ROI records and output image capacity are separate limits. Exercise
// the full admission bound with one source frame, not 4096 image allocations.
void maximum_roi_admission() {
  constexpr int requested = 4096;
  auto capture = preproc_capture();
  probe::Options options;
  options.capture = capture;
  require(std::holds_alternative<rt::Error>(PreprocProbe::create(config(51), options)),
          "output ROI capacity greater than 50 admitted");
  for (const int capacity : {50, 3}) {
    auto stage = take(PreprocProbe::create(config(capacity), options));
    auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
    auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
    auto* published = gst_structure_new_empty("test");
    std::vector<int> rects, indices(requested, 0);
    rects.reserve(requested * 4);
    for (int i = 0; i < requested; ++i)
      rects.insert(rects.end(), {i % 4, (i / 4) % 4, 4, 4});
    rects[2] = 0;   // Invalid width; must not consume an output slot.
    indices[1] = 1; // Invalid source index: explicitly one source frame.
    gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, requested,
                      "preproc_roi_input_batch_size", G_TYPE_INT, 1, nullptr);
    set_list(meta, "preproc_roi_rects", rects);
    set_list(meta, "preproc_roi_batch_indices", indices);
    const auto bind = [&] {
      probe::Binding result;
      okay(stage->bind(input, Storage{}, result, published));
      return result;
    };
    const auto check_publication = [&](int incoming, int valid, int invalid, int overflow) {
      require(integer(published, "preproc_roi_input_count") == incoming &&
                  integer(published, "preproc_roi_count") == valid &&
                  integer(published, "preproc_valid_batch_count") == valid &&
                  integer(published, "preproc_roi_dropped_invalid") == invalid &&
                  integer(published, "preproc_roi_dropped_overflow") == overflow,
              "ROI admission facts do not match published counts");
      require(integer(published, "preproc_roi_capacity") == capacity &&
                  integer(published, "preproc_roi_input_batch_size") == 1,
              "output capacity confused with source batch");
      for (const auto& [name, width] :
           {std::pair{"preproc_roi_batch_indices", 1}, std::pair{"preproc_roi_rects", 4},
            std::pair{"preproc_roi_affines", 6}}) {
        const auto* list = gst_structure_get_value(published, name);
        require(list && GST_VALUE_HOLDS_LIST(list) &&
                    gst_value_list_get_size(list) == unsigned(valid * width),
                "publication retained truncated or stale ROI metadata tails");
      }
    };
    auto first = bind();
    check_publication(requested, capacity, 2, requested - 2 - capacity);
    const auto* accepted = gst_structure_get_value(published, "preproc_roi_rects");
    for (int i = 0; i < capacity * 4; ++i)
      require(g_value_get_int(gst_value_list_get_value(accepted, i)) == rects[i + 8],
              "accepted ROI order changed during bounded truncation");
    auto repeated = bind();
    require(repeated.operation == first.operation &&
                repeated.inputs[0].owner != first.inputs[0].owner,
            "maximum ROI cache hit rebuilt preparation or retained frame storage");
    check_publication(requested, capacity, 2, requested - 2 - capacity);

    // A previously valid full-size signature must not bypass the 4096 bound.
    rects.insert(rects.end(), {0, 0, 4, 4});
    indices.push_back(0);
    set_list(meta, "preproc_roi_rects", rects);
    set_list(meta, "preproc_roi_batch_indices", indices);
    gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, requested + 1, nullptr);
    probe::Binding rejected;
    const auto error = stage->bind(input, Storage{}, rejected, published);
    require(error && error->field == "preproc_roi_count",
            "4097 incoming ROIs did not fail bounded admission");
    rects.resize(requested * 4);
    indices.resize(requested);
    set_list(meta, "preproc_roi_rects", rects);
    set_list(meta, "preproc_roi_batch_indices", indices);
    gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, requested, nullptr);
    require(bind().operation == first.operation,
            "rejected oversized request replaced the valid cached preparation");
    check_publication(requested, capacity, 2, requested - 2 - capacity);

    gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 1, nullptr);
    set_list(meta, "preproc_roi_rects", {0, 0, 4, 4});
    set_list(meta, "preproc_roi_batch_indices", {0});
    bind();
    check_publication(1, 1, 0, 0);
    gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 0, nullptr);
    set_list(meta, "preproc_roi_rects", {});
    set_list(meta, "preproc_roi_batch_indices", {});
    bind();
    check_publication(0, 0, 0, 0);
    require(!gst_structure_has_field(published, "preproc_roi_batch_index"),
            "empty ROI list retained the previous primary source index");
    gst_structure_free(published);
    gst_buffer_unref(input);
  }
  require(capture->packet.empty(), "admission-only test submitted device work");
  std::cout << "Preproc maximum ROI admission and publication PASS\n";
}

// Explicit host preparation diagnostic, not an EV latency or pixel test. Frame
// metadata changes and sample reporting stay outside the measured bind call.
void benchmark_preparation() {
  constexpr std::size_t warmup = 200, samples = 4000;
  for (bool fixed_math : {false, true})
    for (const std::string_view route :
         {"fixed", "alternating", "unique", "list-fixed", "list-alternating", "list-unique"}) {
      const bool list = route.starts_with("list-");
      const auto pattern = list ? route.substr(5) : route;
      auto capture = preproc_capture();
      probe::Options options;
      options.capture = capture;
      auto cfg = config(3);
      if (fixed_math) {
        cfg->rescale_factor = 1.f / 255.f;
        cfg->scaling_type = "BILINEAR_ANTIALIASED";
      }
      auto source = cfg->input_tensors[0].descriptor();
      source.shape.sizes[0] = 480;
      source.shape.sizes[1] = 640;
      source.layout.strided.strides_bytes[0] = 640 * 3;
      cfg->input_tensors[0] = cfg->input_tensors[0].with_geometry(source);
      auto stage = take(PreprocProbe::create(cfg, options));
      auto* input = gst_buffer_new_allocate(nullptr, 640 * 480 * 3, nullptr);
      auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
      if (list) {
        gst_structure_set(meta, "preproc_roi_count", G_TYPE_INT, 3, "preproc_roi_input_batch_size",
                          G_TYPE_INT, 1, nullptr);
        set_list(meta, "preproc_roi_batch_indices", {0, 0, 0});
      } else {
        gst_structure_set(meta, "preproc_roi_enable", G_TYPE_BOOLEAN, TRUE, "preproc_roi_y",
                          G_TYPE_INT, 0, "preproc_roi_width", G_TYPE_INT, 4, "preproc_roi_height",
                          G_TYPE_INT, 4, nullptr);
      }
      auto* published = gst_structure_new_empty("test");
      std::vector<double> elapsed;
      elapsed.reserve(samples);
      for (std::size_t i = 0; i < warmup + samples; ++i) {
        const int x = pattern == "fixed" ? 0 : pattern == "alternating" ? int(i % 2) : int(i % 512);
        const int y = pattern == "unique" ? int(i / 512) : 0;
        if (list)
          set_list(meta, "preproc_roi_rects", {x, y, 4, 4, x + 4, y, 4, 4, x + 8, y, 4, 4});
        else
          gst_structure_set(meta, "preproc_roi_x", G_TYPE_INT, x, "preproc_roi_y", G_TYPE_INT, y,
                            nullptr);
        probe::Binding binding;
        const auto start = std::chrono::steady_clock::now();
        const auto error = stage->bind(input, Storage{}, binding, published);
        const auto end = std::chrono::steady_clock::now();
        okay(error);
        require(integer(published, "preproc_roi_x") == x, "stale benchmark ROI");
        if (list)
          require(integer(published, "preproc_roi_count") == 3 &&
                      gst_value_list_get_size(
                          gst_structure_get_value(published, "preproc_roi_rects")) == 12,
                  "stale benchmark ROI list");
        if (i >= warmup)
          elapsed.push_back(std::chrono::duration<double, std::micro>(end - start).count());
      }
      std::sort(elapsed.begin(), elapsed.end());
      std::cout << "preproc_bind_host_us," << (fixed_math ? "source-" : "") << route << ','
                << std::fixed << std::setprecision(3) << elapsed[samples / 2] << ','
                << elapsed[samples * 99 / 100] << ',' << elapsed.back() << '\n';
      gst_structure_free(published);
      gst_buffer_unref(input);
    }
}

// Real checked requests/packets and publication, not an EV pixel emulator.
void crop_geometry_dynamic_cache() {
  auto cfg = config();
  cfg->resize_mode = simaai::neat::ResizeMode::Crop;
  cfg->shorter_side = 10;
  auto capture = preproc_capture();
  probe::Options options;
  options.capture = capture;
  auto adapter = take(PreprocProbe::create(cfg, options));
  auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* published = gst_structure_new_empty("test");
  auto bind = [&] {
    probe::Binding binding;
    okay(adapter->bind(input, Storage{}, binding, published));
    return binding;
  };
  auto check_packet = [&](probe::Binding& binding, const std::array<std::uint32_t, 4>& expected) {
    const auto bytes = binding.operation->output_requirements(0).storage_bytes;
    binding.outputs = {{std::make_shared<Owner>(bytes, 0x200000), 0, bytes}};
    okay(adapter->session()
             .executor(0)
             ->submit(*binding.operation, binding.inputs, binding.outputs)
             .wait());
    require(capture->packet.abi == 8 && capture->packet.payload.size() >= 312,
            "Crop did not select the coordinated viewport wire ABI");
    for (std::size_t i = 0; i < expected.size(); ++i) {
      std::uint32_t actual = 0;
      std::memcpy(&actual, capture->packet.payload.data() + 296 + 4 * i, 4);
      require(actual == expected[i], "crop window did not use the actual image extent");
    }
  };
  const auto value = [&](const char* name) {
    double v = 0;
    require(gst_structure_get_double(published, name, &v), "missing crop affine");
    return v;
  };
  auto square = bind();
  check_packet(square, {10, 10, 1, 1});
  require(std::string_view(gst_structure_get_string(published, "preproc_resize_mode")) == "crop" &&
              value("preproc_affine_m00") == 0.8 && value("preproc_affine_m02") == 0.8,
          "crop publication lost the exact inverse edge transform");
  gsize offsets[GST_VIDEO_MAX_PLANES]{};
  gint strides[GST_VIDEO_MAX_PLANES]{24};
  auto* video = gst_buffer_add_video_meta_full(input, GST_VIDEO_FRAME_FLAG_NONE,
                                               GST_VIDEO_FORMAT_RGB, 8, 4, 1, offsets, strides);
  require(video, "crop video metadata allocation failed");
  auto landscape = bind();
  check_packet(landscape, {20, 10, 6, 1});
  require(square.operation != landscape.operation &&
              integer(published, "preproc_original_height") == 4 &&
              value("preproc_affine_m00") == 0.4 &&
              std::abs(value("preproc_affine_m02") - 2.4) < 1e-12,
          "actual dynamic frame geometry reused a stale crop or MPK dimensions");
  for (int i = 0; i < 8; ++i) {
    video->height = i % 2 ? 4 : 8;
    require(bind().operation == (i % 2 ? landscape.operation : square.operation),
            "alternating crop geometry did not reuse bounded immutable variants");
  }
  // Exercise the same prospective-template attachment as raw Model::run and
  // stages::Preproc ingress. Its affine must not be mistaken for applied work.
  for (const char* mode : {"crop", "letterbox"}) {
    auto* raw = gst_buffer_copy_deep(input);
    simaai::neat::InputOptions ingress;
    simaai::neat::PreprocessMetaTemplate prospective;
    prospective.enabled = true;
    prospective.target_width = prospective.target_height = 8;
    prospective.scaled_width = 20;
    prospective.scaled_height = 10;
    prospective.resize_mode = mode;
    prospective.color_in = prospective.color_out = "RGB";
    ingress.preprocess_meta = prospective;
    require(simaai::neat::apply_simaai_preprocess_meta_template(raw, ingress, 8, 4),
            "raw ingress template attachment failed");
    auto* copied = gst_buffer_new();
    std::string copy_error;
    require(simaai::neat::copy_simaai_preprocess_meta(raw, raw, &copy_error),
            "same-envelope prospective ingress copy failed");
    require(simaai::neat::copy_simaai_preprocess_meta(copied, raw, &copy_error),
            "prospective ingress copy failed");
    auto* copied_meta =
        gst_custom_meta_get_structure(gst_buffer_get_custom_meta(copied, "GstSimaMeta"));
    require(gst_structure_has_field(copied_meta, "preproc_ingress_template"),
            "copy promoted a prospective transform to applied provenance");
    const auto applied = simaai::neat::read_simaai_preprocess_meta(copied);
    require(applied && simaai::neat::write_simaai_preprocess_meta(copied, *applied) &&
                !gst_structure_has_field(copied_meta, "preproc_ingress_template"),
            "applied facts retained a prospective-template marker");
    gst_buffer_unref(copied);
    auto* source = gst_custom_meta_get_structure(gst_buffer_get_custom_meta(raw, "GstSimaMeta"));
    auto* result = gst_structure_copy(source);
    probe::Binding binding;
    okay(adapter->bind(raw, Storage{}, binding, result));
    double scale = 0., offset = 0.;
    require(binding.operation == landscape.operation &&
                gst_structure_get_double(result, "preproc_affine_m00", &scale) &&
                gst_structure_get_double(result, "preproc_affine_m02", &offset) && scale == 0.4 &&
                std::abs(offset - 2.4) < 1e-12 &&
                !gst_structure_has_field(result, "preproc_ingress_template") &&
                gst_structure_has_field(source, "preproc_ingress_template"),
            "raw ingress composed a prospective transform or leaked its marker");
    gst_structure_free(result);
    gst_buffer_unref(raw);
  }
  auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
  gst_structure_set(meta, "preproc_roi_enable", G_TYPE_BOOLEAN, TRUE, nullptr);
  probe::Binding rejected;
  require(bool(adapter->bind(input, Storage{}, rejected, published)),
          "Crop silently combined with an authored source ROI");
  gst_structure_remove_field(meta, "preproc_roi_enable");
  gst_structure_set(meta, "preproc_original_width", G_TYPE_INT, 80, "preproc_original_height",
                    G_TYPE_INT, 40, "preproc_affine_m00", G_TYPE_DOUBLE, 2.0, "preproc_affine_m01",
                    G_TYPE_DOUBLE, 0.0, "preproc_affine_m02", G_TYPE_DOUBLE, 3.0,
                    "preproc_affine_m10", G_TYPE_DOUBLE, 0.0, "preproc_affine_m11", G_TYPE_DOUBLE,
                    2.0, "preproc_affine_m12", G_TYPE_DOUBLE, 5.0, nullptr);
  auto composed = bind();
  require(composed.operation == landscape.operation &&
              integer(published, "preproc_original_width") == 80 &&
              std::abs(value("preproc_affine_m02") - 7.8) < 1e-12 &&
              std::abs(value("preproc_affine_m12") - 5.8) < 1e-12,
          "upstream transform was reapplied as ROI instructions or lost during composition");
  gst_structure_remove_field(meta, "preproc_affine_m11");
  require(bool(adapter->bind(input, Storage{}, rejected, published)),
          "incomplete upstream provenance was silently accepted");
  // Canonical compact facts are the sole authority, including absent facts.
  // A second legacy authority must fail closed, never fill or override them.
  for (bool with_facts : {false, true}) {
    auto* compact_input = gst_buffer_new_allocate(nullptr, 192, nullptr);
    require(gst_buffer_add_video_meta_full(compact_input, GST_VIDEO_FRAME_FLAG_NONE,
                                           GST_VIDEO_FORMAT_RGB, 8, 4, 1, offsets, strides),
            "compact crop video metadata failed");
    simaai::neat::test_support::attach_compact_frame_carrier(compact_input);
    simaai::gst::TensorBufferFrameMetadata frame;
    std::string error;
    if (with_facts) {
      simaai::gst::TensorBufferPreprocessFacts upstream;
      upstream.original_width = 80;
      upstream.original_height = 40;
      upstream.resized_width = 8;
      upstream.resized_height = 4;
      upstream.primary = {0, 0, 80, 40};
      upstream.placed = {0, 0, 8, 4};
      upstream.affine = {2., 0., 3., 0., 2., 5.};
      upstream.source_stride = 240;
      upstream.resize_mode = "stretch";
      upstream.color_in = upstream.color_out = "RGB";
      auto cold_config = std::make_shared<core::ProcessCvuStagePayload>(*cfg);
      auto cold_source = cold_config->input_tensors[0].descriptor();
      cold_source.shape.sizes[0] = 4;
      cold_config->input_tensors[0] = cold_config->input_tensors[0].with_geometry(cold_source);
      const auto cold_policy = take(PreprocProbe::check_policy(cold_config));
      for (const char* previous_mode : {"stretch", "crop"}) {
        upstream.resize_mode = previous_mode;
        const auto cold = take(
            lower::PreprocFrameAdapter::checked_initial_preprocess_facts(cold_policy, &upstream));
        const auto* facts = simaai::gst::tensor_buffer_preprocess_facts(cold);
        require(facts && facts->resize_mode == "crop" && facts->resized_width == 8 &&
                    facts->resized_height == 8 && facts->original_width == 80 &&
                    facts->original_height == 40 && std::abs(facts->affine[2] - 7.8) < 1e-12 &&
                    std::abs(facts->affine[5] - 5.8) < 1e-12,
                "cold composition reused old producer mode/affine instead of current Crop");
      }
      auto invalid = upstream;
      invalid.roi_list = true;
      require(
          std::holds_alternative<rt::Error>(
              lower::PreprocFrameAdapter::checked_initial_preprocess_facts(cold_policy, &invalid)),
          "cold composition accepted ambiguous ROI-list provenance");
      invalid = upstream;
      invalid.affine[2] = std::numeric_limits<double>::infinity();
      require(
          std::holds_alternative<rt::Error>(
              lower::PreprocFrameAdapter::checked_initial_preprocess_facts(cold_policy, &invalid)),
          "cold composition accepted nonfinite upstream provenance");
      frame.preprocess =
          simaai::gst::tensor_buffer_freeze_preprocess_facts(std::move(upstream), &error);
      require(bool(frame.preprocess), "compact crop facts freeze failed");
    }
    require(
        simaai::gst::tensor_buffer_attach_frame_metadata(compact_input, std::move(frame), &error),
        "compact crop frame attachment failed");
    // Inject a competing stale mirror only after valid compact publication;
    // normal attachment correctly refuses to overwrite a legacy authority.
    auto* stale =
        gst_custom_meta_get_structure(gst_buffer_add_custom_meta(compact_input, "GstSimaMeta"));
    gst_structure_set(stale, "preproc_affine_m00", G_TYPE_DOUBLE, 999.0, "preproc_roi_list_enable",
                      G_TYPE_BOOLEAN, TRUE, nullptr);
    probe::Binding compact_binding;
    require(bool(adapter->bind(compact_input, Storage{}, compact_binding, published)),
            "competing compact and legacy authorities were accepted");
    auto* competing = gst_buffer_get_custom_meta(compact_input, "GstSimaMeta");
    require(gst_buffer_remove_meta(compact_input, &competing->meta),
            "could not remove injected competing authority");
    okay(adapter->bind(compact_input, Storage{}, compact_binding, published));
    require(compact_binding.operation == landscape.operation &&
                integer(published, "preproc_original_width") == (with_facts ? 80 : 8) &&
                std::abs(value("preproc_affine_m02") - (with_facts ? 7.8 : 2.4)) < 1e-12,
            "compact upstream facts lost authority or inherited a stale legacy mirror");
    gst_buffer_unref(compact_input);
  }
  gst_structure_free(published);
  gst_buffer_unref(input);
  for (int fault = 0; fault < 4; ++fault) {
    auto bad = config();
    bad->resize_mode = simaai::neat::ResizeMode::Crop;
    if (fault == 0)
      bad->shorter_side = 7;
    if (fault == 1)
      bad->padding_type = "TOPLEFT";
    if (fault == 2)
      bad->scaled_width = 7;
    if (fault == 3) {
      bad->shorter_side = 10;
      bad->resize_mode.reset();
    }
    require(std::holds_alternative<rt::Error>(PreprocProbe::check_policy(bad)),
            "invalid Crop/legacy geometry combination escaped cold admission");
  }
  std::cout
      << "Crop exact window, actual-frame geometry, immutable cache and affine authority PASS\n";
}

void image_frontend_metadata_and_bounds() {
  auto cfg = config();
  auto policy = std::make_shared<simaai::neat::internal::ImageTransformPolicy>();
  policy->mode = simaai::neat::internal::ImageTransformMode::Normalization;
  cfg->image_transform = policy;
  cfg->output_shapes = {{8, 8, 3}};
  cfg->scaling_type = "NO_SCALING";
  cfg->aspect_ratio = 0;
  probe::Options options;
  options.capture = preproc_capture();
  auto adapter = take(PreprocProbe::create(cfg, options));
  auto* smaller = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", "width",
                                      G_TYPE_INT, 4, "height", G_TYPE_INT, 4, nullptr);
  require(adapter->accepts_caps(smaller).has_value(), "normalization accepted dynamic geometry");
  gst_caps_unref(smaller);
  auto* input = gst_buffer_new_allocate(nullptr, 192, nullptr);
  auto* meta = gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input, "GstSimaMeta"));
  gst_structure_set(meta, "preproc_original_width", G_TYPE_INT, 100, "preproc_original_height",
                    G_TYPE_INT, 50, "preproc_affine_m00", G_TYPE_DOUBLE, 2.0, "preproc_affine_m01",
                    G_TYPE_DOUBLE, 0.0, "preproc_affine_m02", G_TYPE_DOUBLE, 7.0,
                    "preproc_affine_m10", G_TYPE_DOUBLE, 0.0, "preproc_affine_m11", G_TYPE_DOUBLE,
                    3.0, "preproc_affine_m12", G_TYPE_DOUBLE, 11.0, "preproc_roi_enable",
                    G_TYPE_BOOLEAN, TRUE, "preproc_roi_x", G_TYPE_INT, 7, "preproc_roi_y",
                    G_TYPE_INT, 11, "preproc_roi_width", G_TYPE_INT, 16, "preproc_roi_height",
                    G_TYPE_INT, 24, nullptr);
  auto* published = gst_structure_new_empty("test");
  probe::Binding binding;
  tr::TensorBufferPreprocessFactsOwner held, repeated, changed;
  okay(adapter->bind(input, Storage{}, binding, published, false, &held));
  probe::Binding again;
  okay(adapter->bind(input, Storage{}, again, published, false, &repeated));
  require(held == repeated, "steady upstream spatial signature allocated new facts");
  double x = 0, y = 0;
  require(gst_structure_get_double(published, "preproc_affine_m02", &x) && x == 7.0 &&
              gst_structure_get_double(published, "preproc_affine_m12", &y) && y == 11.0,
          "normalization replaced instead of composing upstream coordinates");
  int width = 0, roi = 0;
  require(gst_structure_get_int(published, "preproc_original_width", &width) && width == 100 &&
              gst_structure_get_int(published, "preproc_roi_x", &roi) && roi == 7,
          "upstream scalar ROI provenance was lost");
  // An invalid request-like ROI above is provenance: numerical source remains the whole current
  // image.
  const auto checked = take(rt::inspect_kernel(binding.operation->executable()));
  require(checked.inputs()[0].shape[1] == 8, "result ROI was replayed as a crop request");
  gst_structure_set(meta, "preproc_affine_m02", G_TYPE_DOUBLE, 9.0, nullptr);
  probe::Binding moved;
  okay(adapter->bind(input, Storage{}, moved, published, false, &changed));
  require(changed != held && tr::tensor_buffer_preprocess_facts(held)->affine[2] == 7.0 &&
              tr::tensor_buffer_preprocess_facts(changed)->affine[2] == 9.0,
          "cached metadata mutation changed a held output");
  gsize offsets[GST_VIDEO_MAX_PLANES]{};
  gint strides[GST_VIDEO_MAX_PLANES]{12};
  auto* mismatched = gst_buffer_add_video_meta_full(
      input, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_RGB, 4, 4, 1, offsets, strides);
  probe::Binding wrong_geometry;
  require(adapter->bind(input, Storage{}, wrong_geometry, published).has_value(),
          "authoritative video metadata bypassed fixed geometry");
  gst_buffer_remove_meta(input, &mismatched->meta);
  gst_structure_set(meta, "preproc_affine_m00", G_TYPE_STRING, "bad", nullptr);
  probe::Binding bad;
  require(adapter->bind(input, Storage{}, bad, published).has_value(),
          "malformed upstream transform accepted");
  gst_structure_free(published);
  gst_buffer_unref(input);
}

void image_frontend_input_rejections() {
  auto cfg = config();
  auto policy = std::make_shared<simaai::neat::internal::ImageTransformPolicy>();
  policy->mode = simaai::neat::internal::ImageTransformMode::ColorConvert;
  cfg->image_transform = policy;
  cfg->input_img_type = cfg->output_img_type = "NV12";
  cfg->output_shapes = {{8, 8, 1}};
  cfg->output_dtype = "UINT8";
  cfg->output_tensors = {{descriptor(true), "UINT8"}};
  cfg->normalize = cfg->aspect_ratio = 0;
  cfg->scaling_type = "NO_SCALING";
  probe::Options options;
  options.capture = preproc_capture();
  auto adapter = take(PreprocProbe::create(cfg, options));
  const auto caps = [](const char* key = nullptr, const char* value = nullptr) {
    auto* result = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
                                       G_TYPE_INT, 8, "height", G_TYPE_INT, 8, nullptr);
    if (key)
      gst_caps_set_simple(result, key, G_TYPE_STRING, value, nullptr);
    return std::unique_ptr<GstCaps, decltype(&gst_caps_unref)>(result, gst_caps_unref);
  };
  const auto rejects_caps = [&](const char* key, const char* value, const char* field) {
    const auto input = caps(key, value);
    const auto error = adapter->accepts_caps(input.get());
    require(error && error->field == field, "typed image accepted contradictory source caps");
  };
  okay(adapter->accepts_caps(caps().get())); // Missing interpretation defers to authored policy.
  okay(adapter->accepts_caps(caps("chroma-site", "jpeg").get()));
  GstVideoColorimetry color{};
  color.range = GST_VIDEO_COLOR_RANGE_0_255;
  color.matrix = GST_VIDEO_COLOR_MATRIX_BT601;
  const auto color_caps = [&]() {
    std::unique_ptr<gchar, decltype(&g_free)> text(gst_video_colorimetry_to_string(&color), g_free);
    require(bool(text), "explicit colorimetry construction failed");
    return caps("colorimetry", text.get());
  };
  okay(adapter->accepts_caps(color_caps().get()));
  color.range = GST_VIDEO_COLOR_RANGE_16_235;
  require(adapter->accepts_caps(color_caps().get()).has_value(), "contradictory range accepted");
  color.range = GST_VIDEO_COLOR_RANGE_0_255;
  color.matrix = GST_VIDEO_COLOR_MATRIX_BT709;
  require(adapter->accepts_caps(color_caps().get()).has_value(), "contradictory matrix accepted");
  rejects_caps("colorimetry", "not-colorimetry", "input.colorimetry");
  rejects_caps("chroma-site", "mpeg2", "input.chroma-site");
  rejects_caps("chroma-site", "dv", "input.chroma-site");
  rejects_caps("interlace-mode", "interleaved", "input.caps");

  std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> input(
      gst_buffer_new_allocate(nullptr, 96, nullptr), gst_buffer_unref);
  gsize offsets[GST_VIDEO_MAX_PLANES]{0, 64};
  gint strides[GST_VIDEO_MAX_PLANES]{8, 8};
  auto* video = gst_buffer_add_video_meta_full(input.get(), GST_VIDEO_FRAME_FLAG_INTERLACED,
                                               GST_VIDEO_FORMAT_NV12, 8, 8, 2, offsets, strides);
  require(video != nullptr, "typed image video metadata fixture failed");
  std::unique_ptr<GstStructure, decltype(&gst_structure_free)> published(
      gst_structure_new_empty("test"), gst_structure_free);
  const auto rejects_binding = [&](const char* field) {
    probe::Binding binding;
    const auto error = adapter->bind(input.get(), Storage{}, binding, published.get());
    require(error && error->field == field, "typed image accepted unsupported source metadata");
  };
  rejects_binding("input.video-meta");
  video->flags = GST_VIDEO_FRAME_FLAG_NONE;
  auto* meta =
      gst_custom_meta_get_structure(gst_buffer_add_custom_meta(input.get(), "GstSimaMeta"));
  gst_structure_set(meta, "preproc_roi_list_enable", G_TYPE_BOOLEAN, TRUE, nullptr);
  rejects_binding("input.preprocess");
  gst_structure_remove_field(meta, "preproc_roi_list_enable");
  gst_structure_set(meta, "preproc_roi_enable", G_TYPE_BOOLEAN, TRUE, "preproc_roi_x", G_TYPE_INT,
                    0, "preproc_roi_y", G_TYPE_INT, 0, "preproc_roi_width", G_TYPE_INT, 8,
                    "preproc_roi_height", G_TYPE_INT, 8, nullptr);
  rejects_binding("input.preprocess");
  gst_structure_remove_field(meta, "preproc_roi_enable");
  probe::Binding valid;
  okay(adapter->bind(input.get(), Storage{}, valid, published.get()));
  std::cout << "Image frontend explicit color, interlace and ROI rejection PASS\n";
}

} // namespace
int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  if (!gst_meta_get_info("GstSimaMeta")) {
    const char* tags[] = {nullptr};
    gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
  }
  try {
    geometry_descriptor_compatibility();
    crop_geometry_dynamic_cache();
    image_frontend_metadata_and_bounds();
    image_frontend_input_rejections();
    cold_generic_preparation();
    immutable_policy_instances();
    contract();
    fixed_arithmetic_contract();
    typed_format_mapping();
    checked_plane_resolver();
    decoder_storage_geometry();
    dynamic_preparation_cache();
    maximum_roi_admission();
    publication_ownership_and_replacement();
    default_geometry_and_padding();
    borrowed_roi_indices();
    if (argc == 2 && std::string_view(argv[1]) == "--benchmark-bind")
      benchmark_preparation();
    else if (argc != 1)
      throw std::runtime_error("usage: preproc_contract [--benchmark-bind]");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

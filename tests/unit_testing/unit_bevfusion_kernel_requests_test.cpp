// The exact named kernels-library requests behind the SimaVoxelize, SimaPillarScatter,
// SimaBEVPoolPlan and SimaBEVPool Nodes, and the hand-computed known answers of the
// independent references their numerical tests use. Pure: no Scheduler call,
// GStreamer or device, so it runs on any host that has the Scheduler client
// value-type headers.
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include "support/bevfusion_kernel_reference.h"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
using namespace simaai::neat;
namespace kr = nodes_internal::kernel_requests;
namespace rt = ::simaai::scheduler::client;
namespace ref = ::bevfusion_test::reference;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void require_contains(const std::string& haystack, const std::string& needle,
                      const std::string& message) {
  if (haystack.find(needle) == std::string::npos)
    throw std::runtime_error(message + " (missing: " + needle + ")");
}

kr::TensorFact fact(std::vector<std::int64_t> shape, rt::DType dtype,
                    std::vector<std::int64_t> strides = {}) {
  return {std::move(shape), dtype, std::move(strides), "", {}};
}

void require_throws(const std::function<void()>& fn, const std::string& needle,
                    const std::string& what) {
  try {
    fn();
  } catch (const std::invalid_argument& error) {
    require_contains(error.what(), needle, what);
    return;
  }
  throw std::runtime_error(what + ": expected std::invalid_argument");
}

const rt::Parameter& parameter(const kr::KernelRequestPlan& plan, const std::string& name) {
  for (const auto& p : plan.parameters)
    if (p.name == name)
      return p;
  throw std::runtime_error(plan.operation + " request lacks parameter " + name);
}

template <class T> T scalar(const kr::KernelRequestPlan& plan, const std::string& name) {
  const auto& value = std::get<rt::ParameterScalar>(parameter(plan, name).value.value).value;
  return std::get<T>(value);
}

std::vector<double> floats(const kr::KernelRequestPlan& plan, const std::string& name) {
  std::vector<double> out;
  for (const auto& v : std::get<rt::ParameterArray>(parameter(plan, name).value.value))
    out.push_back(std::get<double>(v.value));
  return out;
}

// --------------------------------------------------------------------------
// The exact named requests
// --------------------------------------------------------------------------
void voxelize_request() {
  auto opt = ref::voxelize_case().options;
  const std::array facts{fact({17, 4}, rt::DType::Float32), fact({1}, rt::DType::Int32)};
  const auto plan = kr::voxelize_request(opt, facts);
  require(plan.operation == "voxelize" && plan.target == rt::Target::A65, "voxelize identity");
  require(plan.inputs.size() == 2 && plan.inputs[0].shape == std::vector<std::int64_t>{17, 4} &&
              plan.inputs[0].dtype == rt::DType::Float32 &&
              plan.inputs[1].shape == std::vector<std::int64_t>{1} &&
              plan.inputs[1].dtype == rt::DType::Int32,
          "voxelize input specs");
  require(plan.outputs.size() == 3 &&
              plan.outputs[0].shape == std::vector<std::int64_t>{1, 3, 8, 10} &&
              plan.outputs[0].dtype == rt::DType::Int8 &&
              plan.outputs[1].shape == std::vector<std::int64_t>{3, 2} &&
              plan.outputs[2].shape == std::vector<std::int64_t>{1},
          "voxelize output specs");
  for (const auto& spec : plan.outputs)
    require(std::holds_alternative<rt::Contiguous>(spec.layout), "voxelize outputs contiguous");
  require(floats(plan, "voxel_size") == std::vector<double>{1, 1, 4} &&
              floats(plan, "point_cloud_range") == std::vector<double>{0, 0, -2, 4, 4, 2},
          "voxelize grid parameters");
  require(scalar<double>(plan, "quantization_multiplier") == 4.0 &&
              scalar<std::int64_t>(plan, "quantization_zero_point") == -3 &&
              scalar<std::string>(plan, "quantization_rounding") == "nearest_even",
          "voxelize quantization parameters");
  require(plan.published[0].name == "decorated" && plan.published[0].layout == "NHWC" &&
              plan.published[1].name == "coordinates" && plan.published[2].name == "count",
          "voxelize publication names");
  opt.target = SimaKernelTarget::EV74;
  opt.quantization.rounding = SimaQuantRounding::TowardNegative;
  const auto ev = kr::voxelize_request(opt, facts);
  require(ev.target == rt::Target::EV74 &&
              scalar<std::string>(ev, "quantization_rounding") == "toward_negative",
          "voxelize EV74 target and rounding");
  require_throws([&] { kr::voxelize_request(opt, std::span(facts).first(1)); },
                 "SimaVoxelize: expects 2 input tensors", "voxelize arity");
  require_throws(
      [&] {
        const std::array bad{fact({17, 3}, rt::DType::Float32), fact({1}, rt::DType::Int32)};
        kr::voxelize_request(opt, bad);
      },
      "points must be [N,4]", "voxelize point shape");
  require_throws(
      [&] {
        const std::array bad{fact({17, 4}, rt::DType::BFloat16), fact({1}, rt::DType::Int32)};
        kr::voxelize_request(opt, bad);
      },
      "points must be FP32", "voxelize point dtype");
  require_throws(
      [&] {
        const std::array bad{fact({17, 4}, rt::DType::Float32, {32, 4}),
                             fact({1}, rt::DType::Int32)};
        kr::voxelize_request(opt, bad);
      },
      "must be dense", "voxelize rejects padded points");
  require_throws(
      [&] {
        const std::array bad{fact({17, 4}, rt::DType::Float32), fact({2}, rt::DType::Int32)};
        kr::voxelize_request(opt, bad);
      },
      "point_count must hold one INT32 value", "voxelize count shape");
}

void pillar_scatter_request() {
  const auto opt = ref::pillar_scatter_case().options;
  {
    // Only singleton axes may join the rows: [5,1,16] is [M,C], [2,3,16] is not.
    const std::array squeezed{fact({5, 1, 16}, rt::DType::Int8), fact({1, 5, 2}, rt::DType::Int32),
                              fact({1}, rt::DType::Int32)};
    require(kr::pillar_scatter_request(opt, squeezed).inputs[0].shape ==
                std::vector<std::int64_t>{5, 16},
            "pillar_scatter accepts singleton axes after M");
    const std::array split{fact({2, 3, 16}, rt::DType::Int8), fact({6, 2}, rt::DType::Int32),
                           fact({1}, rt::DType::Int32)};
    require_throws([&] { kr::pillar_scatter_request(opt, split); },
                   "features may add only singleton axes to its rows",
                   "pillar_scatter refuses rows split across axes");
  }
  // vfe publishes [1,M,1,C]; the builder squeezes it to [M,C].
  const std::array facts{fact({1, 5, 1, 16}, rt::DType::Int8), fact({5, 2}, rt::DType::Int32),
                         fact({1, 1}, rt::DType::Int32)};
  const auto plan = kr::pillar_scatter_request(opt, facts);
  require(plan.operation == "pillar_scatter", "pillar_scatter identity");
  require(plan.inputs[0].shape == std::vector<std::int64_t>{5, 16} &&
              plan.inputs[1].shape == std::vector<std::int64_t>{5, 2} &&
              plan.inputs[2].shape == std::vector<std::int64_t>{1},
          "pillar_scatter squeezes singleton axes");
  const auto* tile = std::get_if<rt::MlaTiled>(&plan.outputs[0].layout);
  require(tile && plan.outputs[0].shape == std::vector<std::int64_t>{1, 4, 6, 16} &&
              tile->axes == std::array<std::int8_t, 4>{0, 1, 2, 3} &&
              tile->tile_shape == std::array<std::int64_t, 4>{1, 1, 1, 16} &&
              tile->channels == rt::MlaTiled::Channels::Compact,
          "pillar_scatter uses full-channel cell tiles without changing canvas bytes");
  require(plan.published[0].tiled && plan.published[0].layout == "NHWC" &&
              plan.published[0].shape == std::vector<std::int64_t>{1, 4, 6, 16},
          "pillar_scatter publishes the dense NHWC canvas");
  require(scalar<std::int64_t>(plan, "zero_point") == -7 && plan.parameters.size() == 1,
          "pillar_scatter zero point");
  require_throws(
      [&] {
        const std::array bad{fact({5, 16}, rt::DType::Int8), fact({4, 2}, rt::DType::Int32),
                             fact({1}, rt::DType::Int32)};
        kr::pillar_scatter_request(opt, bad);
      },
      "features hold 5 pillars but coordinates hold 4", "pillar_scatter pillar mismatch");
  require_throws(
      [&] {
        const std::array bad{fact({5, 16}, rt::DType::Int8), fact({5, 3}, rt::DType::Int32),
                             fact({1}, rt::DType::Int32)};
        kr::pillar_scatter_request(opt, bad);
      },
      "coordinates must end in an axis of 2", "pillar_scatter coordinate pairs");
  require_throws(
      [&] {
        const std::array bad{fact({5, 16}, rt::DType::UInt8), fact({5, 2}, rt::DType::Int32),
                             fact({1}, rt::DType::Int32)};
        kr::pillar_scatter_request(opt, bad);
      },
      "features must be INT8", "pillar_scatter feature dtype");
}

void bev_pool_plan_request() {
  auto c = ref::bev_pool_plan_case();
  const std::array facts{fact({2, 4, 4}, rt::DType::Float32), fact({2, 4, 4}, rt::DType::Float32),
                         fact({2, 4, 4}, rt::DType::Float32), fact({4, 4}, rt::DType::Float32)};
  auto plan = kr::bev_pool_plan_request(c.options, facts);
  // v1 worst case: 64 + 4 * (16 cells + 1) + 4 * (2 * 2 * 2 * 3 points).
  require(plan.operation == "bev_pool_plan" && plan.outputs.size() == 1 &&
              plan.outputs[0].shape == std::vector<std::int64_t>{64 + 4 * 17 + 4 * 24} &&
              plan.outputs[0].dtype == rt::DType::UInt8,
          "bev_pool_plan v1 capacity");
  require(floats(plan, "xs") == std::vector<double>{0, 2, 4} &&
              floats(plan, "ys") == std::vector<double>{0, 2} &&
              floats(plan, "ds") == std::vector<double>{1, 2} &&
              floats(plan, "bx") == std::vector<double>{0.5, 0.5, 0} &&
              floats(plan, "dx") == std::vector<double>{1, 1, 8},
          "bev_pool_plan frustum and grid");
  const auto& nx = std::get<rt::ParameterArray>(parameter(plan, "nx").value.value);
  require(nx.size() == 3 && std::get<std::int64_t>(nx[0].value) == 4 &&
              std::get<std::int64_t>(nx[2].value) == 1,
          "bev_pool_plan integer cell counts");
  require(scalar<std::int64_t>(plan, "format") == 1 && !scalar<bool>(plan, "cache_plan") &&
              scalar<std::int64_t>(plan, "workers") == 4 &&
              scalar<std::int64_t>(plan, "tile_h") == 8,
          "bev_pool_plan defaults");
  c.options.plan_capacity_bytes = 4096;
  c.options.cache_plan = true;
  plan = kr::bev_pool_plan_request(c.options, facts);
  require(plan.outputs[0].shape == std::vector<std::int64_t>{4096} &&
              scalar<bool>(plan, "cache_plan"),
          "bev_pool_plan explicit capacity and cache");

  // BEVFusion camera-only V2 production geometry:
  // tiles 32*32, records min(1024 * 528 * 1, 2,230,272) = 540,672.
  SimaBEVPoolPlanOptions v2;
  v2.xs.assign(88, 0.0f);
  v2.ys.assign(32, 0.0f);
  v2.ds.assign(118, 1.0f);
  v2.nx = {256, 256, 1};
  v2.format = SimaBEVPoolPlanFormat::V2;
  v2.context_channels = 80;
  const auto points = std::uint64_t(6) * 118 * 32 * 88;
  require(kr::bev_pool_plan_capacity_bytes(v2, 6) ==
              64 + 8 * 16 + 8 * 1024 + 20 * std::uint64_t(540672) + 8 * points,
          "bev_pool_plan v2 worst case");
  v2.format = SimaBEVPoolPlanFormat::V1;
  require(kr::bev_pool_plan_capacity_bytes(v2, 6) == 64 + 4 * (65536 + 1) + 4 * points,
          "bev_pool_plan v1 production worst case");
  require_throws(
      [&] {
        const std::array bad{fact({2, 4, 4}, rt::DType::Float32),
                             fact({3, 4, 4}, rt::DType::Float32),
                             fact({2, 4, 4}, rt::DType::Float32), fact({4, 4}, rt::DType::Float32)};
        kr::bev_pool_plan_request(c.options, bad);
      },
      "camera_intrinsics must be [N,4,4] with the same N", "bev_pool_plan camera mismatch");
}

void bev_pool_request() {
  auto opt = ref::bev_pool_case().options;
  // Dense NHWC camera outputs [N,fH,fW,D] / [N,fH,fW,C].
  const std::array facts{fact({2, 2, 3, 2}, rt::DType::Int8), fact({2, 2, 3, 16}, rt::DType::Int8),
                         fact({228}, rt::DType::UInt8)};
  auto plan = kr::bev_pool_request(opt, facts);
  require(plan.operation == "bev_pool", "bev_pool identity");
  const auto strides = [](const rt::TensorSpec& spec) {
    const auto* s = std::get_if<rt::Strided>(&spec.layout);
    require(s != nullptr, "bev_pool channels-last ports must be Strided");
    return std::vector<std::int64_t>(s->strides_bytes.begin(), s->strides_bytes.begin() + 4);
  };
  require(plan.inputs[0].shape == std::vector<std::int64_t>{2, 2, 2, 3} &&
              strides(plan.inputs[0]) == std::vector<std::int64_t>{12, 1, 6, 2},
          "bev_pool depth is logical [N,D,fH,fW] with channels-last strides");
  require(plan.inputs[1].shape == std::vector<std::int64_t>{2, 16, 2, 3} &&
              strides(plan.inputs[1]) == std::vector<std::int64_t>{96, 1, 48, 16},
          "bev_pool context is logical [N,C,fH,fW] with channels-last strides");
  require(plan.outputs[0].shape == std::vector<std::int64_t>{1, 16, 4, 4} &&
              strides(plan.outputs[0]) == std::vector<std::int64_t>{256, 1, 64, 16} &&
              plan.published[0].stride_bytes == std::vector<std::int64_t>{256, 1, 64, 16},
          "bev_pool bev is channels-last [1,nZ*C,nX,nY]");
  require(scalar<std::int64_t>(plan, "depth_zero_point") == -128 &&
              scalar<std::int64_t>(plan, "context_zero_point") == -5 &&
              scalar<std::int64_t>(plan, "plan_format") == 1 &&
              scalar<double>(plan, "quantization_multiplier") == 0.0625 &&
              scalar<std::int64_t>(plan, "quantization_zero_point") == 3,
          "bev_pool parameters");
  // A producer that already publishes logical [N,D,fH,fW] with channels-last
  // strides is the same request.
  const std::array strided{fact({2, 2, 2, 3}, rt::DType::Int8, {12, 1, 6, 2}),
                           fact({2, 16, 2, 3}, rt::DType::Int8, {96, 1, 48, 16}),
                           fact({228}, rt::DType::UInt8)};
  const auto same = kr::bev_pool_request(opt, strided);
  require(same.inputs == plan.inputs, "bev_pool accepts channels-last strided producers");

  opt.input_storage = opt.output_storage = SimaBEVPoolStorage::Contiguous;
  const std::array nchw{fact({2, 2, 2, 3}, rt::DType::Int8), fact({2, 16, 2, 3}, rt::DType::Int8),
                        fact({228}, rt::DType::UInt8)};
  plan = kr::bev_pool_request(opt, nchw);
  require(std::holds_alternative<rt::Contiguous>(plan.inputs[0].layout) &&
              std::holds_alternative<rt::Contiguous>(plan.outputs[0].layout) &&
              plan.published[0].stride_bytes.empty(),
          "bev_pool contiguous storage");
  opt.cells[2] = 2;
  plan = kr::bev_pool_request(opt, nchw);
  require(plan.outputs[0].shape == std::vector<std::int64_t>{1, 32, 4, 4},
          "bev_pool output channels are nZ * C");
  require_throws(
      [&] {
        const std::array bad{fact({2, 2, 2, 3}, rt::DType::Int8),
                             fact({2, 16, 2, 3}, rt::DType::Int8),
                             fact({2, 114}, rt::DType::UInt8)};
        kr::bev_pool_request(opt, bad);
      },
      "plan must be a rank-1 UINT8", "bev_pool plan rank");
  opt.input_storage = SimaBEVPoolStorage::ChannelsLast;
  require_throws(
      [&] {
        const std::array bad{fact({2, 2, 3, 2}, rt::DType::Int8, {24, 12, 4, 1}),
                             fact({2, 2, 3, 16}, rt::DType::Int8), fact({228}, rt::DType::UInt8)};
        kr::bev_pool_request(opt, bad);
      },
      "neither dense channels-last", "bev_pool rejects padded depth");
}

// Malformed metadata must be rejected before any signed shape/stride arithmetic,
// including when the caller leaves dense strides implicit. No payload allocation
// or provider is needed to exercise these cold request-construction boundaries.
void checked_extent_boundaries() {
  constexpr auto limit = std::numeric_limits<std::int64_t>::max();
  const std::array largest_byte_shape{limit};
  require(kr::checked_tensor_bytes(largest_byte_shape, rt::DType::UInt8, "test") ==
              static_cast<std::uint64_t>(limit),
          "the largest representable byte extent remains valid");
  const std::array largest_bf16_shape{limit / 2};
  require(kr::checked_tensor_bytes(largest_bf16_shape, rt::DType::BFloat16, "test") ==
              static_cast<std::uint64_t>(limit - 1),
          "the largest representable BF16 byte extent remains valid");
  for (const auto& shape : {std::vector<std::int64_t>{limit / 2 + 1},
                            std::vector<std::int64_t>{631832658, 1824726041, 16}})
    require_throws([&] { kr::checked_tensor_bytes(shape, rt::DType::BFloat16, "test"); },
                   "test: tensor extent overflows", "byte extent multiplication cannot wrap");
  for (const auto bad_dimension : {std::int64_t{0}, std::int64_t{-1}}) {
    const std::array shape{bad_dimension};
    require_throws([&] { kr::checked_tensor_bytes(shape, rt::DType::Int8, "test"); },
                   "nonpositive extent", "nonpositive dimensions are not byte extents");
  }
  require_throws(
      [&] { kr::checked_tensor_bytes(largest_byte_shape, static_cast<rt::DType>(-1), "test"); },
      "unsupported tensor dtype", "an unknown dtype cannot make every extent zero");

  const auto voxelize = ref::voxelize_case().options;
  for (const auto& strides : {std::vector<std::int64_t>{}, std::vector<std::int64_t>{16, 4}}) {
    const std::array at_limit{fact({limit / 16, 4}, rt::DType::Float32, strides),
                              fact({1}, rt::DType::Int32)};
    require(kr::voxelize_request(voxelize, at_limit).inputs[0].shape == at_limit[0].shape,
            "representable point extent is left to provider capacity admission");
    for (const auto rows : {limit / 16 + 1, std::int64_t{1} << 62}) {
      const std::array overflow{fact({rows, 4}, rt::DType::Float32, strides),
                                fact({1}, rt::DType::Int32)};
      require_throws([&] { kr::voxelize_request(voxelize, overflow); },
                     "SimaVoxelize: tensor extent overflows",
                     "implicit and explicit point strides reject overflow before multiplying");
    }
  }

  auto pool = ref::bev_pool_case().options;
  pool.cells = {1, 1, 4};
  const std::array channel_overflow{fact({1, 1, 1, 1}, rt::DType::Int8),
                                    fact({1, 1, 1, std::int64_t{1} << 62}, rt::DType::Int8),
                                    fact({64}, rt::DType::UInt8)};
  require_throws([&] { kr::bev_pool_request(pool, channel_overflow); },
                 "SimaBEVPool: tensor extent overflows", "nZ times context channels is checked");

  pool.cells = {1, 1, 1};
  const std::array wide_input{fact({1, 1, 1, std::int64_t{1} << 62}, rt::DType::BFloat16),
                              fact({1, 1, 1, 16}, rt::DType::BFloat16),
                              fact({64}, rt::DType::UInt8)};
  require_throws([&] { kr::bev_pool_request(pool, wide_input); },
                 "SimaBEVPool: tensor extent overflows",
                 "a representable element count does not authorize overflowing BF16 strides");

  pool.cells = {1, 1, 2};
  pool.output_dtype = TensorDType::BFloat16;
  const std::array wide_output{fact({1, 1, 1, 1}, rt::DType::BFloat16),
                               fact({1, 1, 1, std::int64_t{1} << 61}, rt::DType::BFloat16),
                               fact({64}, rt::DType::UInt8)};
  require_throws([&] { kr::bev_pool_request(pool, wide_output); },
                 "SimaBEVPool: tensor extent overflows", "output byte extent is checked too");

  auto plan = ref::bev_pool_plan_case().options;
  for (const auto format : {SimaBEVPoolPlanFormat::V1, SimaBEVPoolPlanFormat::V2}) {
    plan.format = format;
    require_throws([&] { kr::bev_pool_plan_capacity_bytes(plan, limit); },
                   "tensor or plan extent overflows", "plan point-count multiplication is checked");
  }

  pool = ref::bev_pool_case().options;
  for (const bool input : {false, true}) {
    auto invalid = pool;
    (input ? invalid.input_storage : invalid.output_storage) = static_cast<SimaBEVPoolStorage>(-1);
    require_throws([&] { kr::validate(invalid); }, "must be Contiguous or ChannelsLast",
                   "unknown storage is not silently interpreted as a supported layout");
  }
}

// --------------------------------------------------------------------------
// The references' own known answers, computed by hand
// --------------------------------------------------------------------------
void dtype_tokens() {
  require(kr::dtype_from_token("INT8", "T") == rt::DType::Int8 &&
              kr::dtype_from_token("EVXX_INT8", "T") == rt::DType::Int8 &&
              kr::dtype_from_token("evxx_float32", "T") == rt::DType::Float32 &&
              kr::dtype_from_token("BF16", "T") == rt::DType::BFloat16 &&
              kr::dtype_from_token("INT32", "T") == rt::DType::Int32 &&
              kr::dtype_from_token("UINT8", "T") == rt::DType::UInt8,
          "dtype tokens, including EV-qualified model tokens");
  require_throws([] { kr::dtype_from_token("FP64", "SimaBEVPool"); },
                 "SimaBEVPool: unsupported input dtype", "unsupported dtype token");
  for (const auto dtype : {rt::DType::Float32, rt::DType::BFloat16, rt::DType::Int8,
                           rt::DType::UInt8, rt::DType::Int16, rt::DType::Int32, rt::DType::UInt16})
    require(kr::dtype_from_token(kr::dtype_token(dtype), "T") == dtype, "dtype token round trip");
}

bool has_parameter(const kr::KernelRequestPlan& plan, const std::string& name) {
  for (const auto& p : plan.parameters)
    if (p.name == name)
      return true;
  return false;
}

std::vector<std::int64_t> strides(const rt::TensorSpec& spec) {
  const auto* s = std::get_if<rt::Strided>(&spec.layout);
  require(s != nullptr, "expected a Strided port");
  return {s->strides_bytes.begin(), s->strides_bytes.begin() + 4};
}

void bf16_requests() {
  // Voxelize: BF16 decorated; the identity quantization is forwarded unchanged.
  auto vox = ref::voxelize_case_bf16().options;
  const std::array vf{fact({17, 4}, rt::DType::Float32), fact({1}, rt::DType::Int32)};
  auto plan = kr::voxelize_request(vox, vf);
  require(plan.outputs[0].dtype == rt::DType::BFloat16 &&
              plan.outputs[1].dtype == rt::DType::Int32 &&
              scalar<double>(plan, "quantization_multiplier") == 1.0 &&
              scalar<std::int64_t>(plan, "quantization_zero_point") == 0,
          "SimaVoxelize BF16 request");
  vox.output_dtype = TensorDType::Float32;
  require_throws([&] { kr::validate(vox); }, "SimaVoxelize: output_dtype must be Int8 or BFloat16",
                 "SimaVoxelize rejects an FP32 decorated");

  // PillarScatter: the canvas follows the BF16 features.
  const auto ps = ref::pillar_scatter_case_bf16().options;
  const std::array pf{fact({1, 5, 1, 16}, rt::DType::BFloat16), fact({5, 2}, rt::DType::Int32),
                      fact({1}, rt::DType::Int32)};
  plan = kr::pillar_scatter_request(ps, pf);
  require(plan.inputs[0].dtype == rt::DType::BFloat16 &&
              plan.inputs[0].shape == std::vector<std::int64_t>{5, 16} &&
              plan.outputs[0].dtype == rt::DType::BFloat16 &&
              std::holds_alternative<rt::MlaTiled>(plan.outputs[0].layout) &&
              scalar<std::int64_t>(plan, "zero_point") == 0,
          "SimaPillarScatter BF16 request");
  require_throws(
      [&] {
        const std::array bad{fact({5, 16}, rt::DType::UInt8), fact({5, 2}, rt::DType::Int32),
                             fact({1}, rt::DType::Int32)};
        kr::pillar_scatter_request(ps, bad);
      },
      "features must be INT8 or BF16", "SimaPillarScatter rejects UINT8 features");
  require_throws(
      [&] {
        const std::array bad{fact({5, 16}, rt::DType::BFloat16, {64, 2}),
                             fact({5, 2}, rt::DType::Int32), fact({1}, rt::DType::Int32)};
        kr::pillar_scatter_request(ps, bad);
      },
      "must be dense", "SimaPillarScatter checks BF16 density with two-byte scalars");

  // BEVPoolPlan: input_dtype is named only for BF16; format 1 is dtype-independent.
  auto bp = ref::bev_pool_plan_case().options;
  const std::array cf{fact({2, 4, 4}, rt::DType::Float32), fact({2, 4, 4}, rt::DType::Float32),
                      fact({2, 4, 4}, rt::DType::Float32), fact({4, 4}, rt::DType::Float32)};
  require(!has_parameter(kr::bev_pool_plan_request(bp, cf), "input_dtype"),
          "SimaBEVPoolPlan INT8 keeps the kernels' default input_dtype");
  bp.input_dtype = TensorDType::BFloat16;
  plan = kr::bev_pool_plan_request(bp, cf);
  require(scalar<std::string>(plan, "input_dtype") == "bfloat16" &&
              plan.outputs[0].shape == std::vector<std::int64_t>{64 + 4 * 17 + 4 * 24},
          "SimaBEVPoolPlan BF16 request and dtype-independent v1 capacity");
  bp.input_dtype = TensorDType::Float32;
  require_throws([&] { kr::validate(bp); }, "SimaBEVPoolPlan: input_dtype must be Int8 or BFloat16",
                 "SimaBEVPoolPlan rejects FP32 inputs");
  // Format 2 at fH = 128: the 16 KiB depth window holds 128 INT8 or 64 BF16 bins.
  SimaBEVPoolPlanOptions tall;
  tall.xs.assign(8, 0.0f);
  tall.ys.assign(128, 0.0f);
  tall.ds.assign(200, 1.0f);
  tall.nx = {256, 256, 1};
  tall.format = SimaBEVPoolPlanFormat::V2;
  tall.context_channels = 16;
  const std::uint64_t points = 6ULL * 200 * 128 * 8, tiles = 32 * 32;
  require(kr::bev_pool_plan_capacity_bytes(tall, 6) ==
              64 + 8 * 16 + 8 * tiles + 20 * (tiles * 48 * 2) + 8 * points,
          "SimaBEVPoolPlan INT8 v2 worst case at fH = 128");
  tall.input_dtype = TensorDType::BFloat16;
  require(kr::bev_pool_plan_capacity_bytes(tall, 6) ==
              64 + 8 * 16 + 8 * tiles + 20 * (tiles * 48 * 4) + 8 * points,
          "SimaBEVPoolPlan BF16 v2 worst case at fH = 128");

  // BEVPool: BF16 inputs with two-byte channels-last strides, BF16 or INT8 bev.
  auto pool = ref::bev_pool_case_bf16().options;
  const std::array bf{fact({2, 2, 3, 2}, rt::DType::BFloat16),
                      fact({2, 2, 3, 16}, rt::DType::BFloat16), fact({228}, rt::DType::UInt8)};
  plan = kr::bev_pool_request(pool, bf);
  require(plan.inputs[0].dtype == rt::DType::BFloat16 &&
              strides(plan.inputs[0]) == std::vector<std::int64_t>{24, 2, 12, 4} &&
              strides(plan.inputs[1]) == std::vector<std::int64_t>{192, 2, 96, 32} &&
              plan.outputs[0].dtype == rt::DType::BFloat16 &&
              strides(plan.outputs[0]) == std::vector<std::int64_t>{512, 2, 128, 32} &&
              plan.published[0].stride_bytes == std::vector<std::int64_t>{512, 2, 128, 32} &&
              scalar<std::int64_t>(plan, "depth_zero_point") == 0 &&
              scalar<double>(plan, "quantization_multiplier") == 1.0,
          "SimaBEVPool BF16 request");
  pool.output_dtype = TensorDType::Int8;
  plan = kr::bev_pool_request(pool, bf);
  require(plan.outputs[0].dtype == rt::DType::Int8 &&
              strides(plan.outputs[0]) == std::vector<std::int64_t>{256, 1, 64, 16},
          "SimaBEVPool BF16 inputs with an INT8 bev");
}

void bf16_reference_known_answers() {
  require(ref::to_bf16(1.0f) == 0x3F80 && ref::to_bf16(1.00390625f) == 0x3F80 &&
              ref::to_bf16(1.01171875f) == 0x3F82 && ref::to_bf16(-0.0f) == 0x8000 &&
              ref::to_bf16(std::numeric_limits<float>::infinity()) == 0x7F80 &&
              ref::to_bf16(std::nanf("")) == 0x7FC0 && ref::to_bf16(-std::nanf("")) == 0x7FC0 &&
              ref::to_bf16(std::numeric_limits<float>::max()) == 0x7F80,
          "BF16 nearest-even narrowing, overflow and canonical NaN");
  require(ref::from_bf16(0x3D80) == 0.0625f && std::isnan(ref::from_bf16(0x7FC1)), "BF16 widening");
  const auto v = ref::voxelize_case_bf16();
  const auto r = ref::voxelize(v.options, v.points, v.point_count);
  // Pillar 2 is (3.5, 0.5, 1.5, 0): [3.5,0.5,1.5,0,0,0,0,0,0,1.5].
  const std::array<std::uint16_t, 10> pillar2{0x4060, 0x3F00, 0x3FC0, 0, 0, 0, 0, 0, 0, 0x3FC0};
  for (std::size_t c = 0; c < 10; ++c)
    require(r.decorated_bf16[(2 * 8) * 10 + c] == pillar2[c], "SimaVoxelize BF16 decoration");
  require(r.decorated.empty() && r.decorated_bf16[(2 * 8 + 1) * 10] == 0,
          "SimaVoxelize BF16 pads with +0.0");
  // The first point's x is 2.00390625, which narrows to 2.0 (a tie, to even).
  require(r.decorated_bf16[0] == 0x4000, "SimaVoxelize BF16 rounds to nearest-even");
  const std::vector<std::uint16_t> canvas{0x1234, 0xABCD};
  require(ref::whole_tile_bf16_bytes(canvas, 2) ==
              std::vector<std::uint8_t>{0x34, 0xCD, 0x12, 0xAB},
          "MlaTiled BF16 payload: low bytes, then high bytes");
  // Pillar 0 lands in cell (0,5), 32 bytes per cell: channel 0 is 0x7FC1 and
  // channel 1 is -14.75 (0xC16C). Interleaved NHWC would put 0x7F at byte 161.
  const auto ps = ref::pillar_scatter_case_bf16();
  const auto stored = ref::pillar_scatter_bf16_bytes(ps.options, ps.features_bf16, ps.coordinates,
                                                     ps.count, ps.channels);
  require(stored.size() == 4 * 6 * 16 * 2 && stored[160] == 0xC1 && stored[161] == 0x6C &&
              stored[176] == 0x7F && stored[177] == 0xC1 &&
              std::all_of(stored.begin(), stored.begin() + 160, [](auto b) { return b == 0; }),
          "SimaPillarScatter BF16 canvas bytes are the MlaTiled payload");
  const auto acc = ref::bev_pool_case_bf16_accumulation();
  require(ref::bev_pool_bf16(acc)[(0 * 4 + 1) * 4 + 1] == 0x3F80,
          "SimaBEVPool BF16 sums in binary32: 1 + 2^-8 + 10 * 2^-30 stays on the tie");
}

void reference_known_answers() {
  const SimaKernelQuantization q{2.0f, -3, SimaQuantRounding::NearestEven};
  require(ref::quantize(1.25f, q) == -1, "Quant rounds before adding the zero point");
  require(ref::quantize(0.75f, q) == -1 && ref::quantize(-0.25f, q) == -3,
          "Quant nearest-even ties");
  require(ref::quantize(1e9f, q) == 127 && ref::quantize(-1e9f, q) == -128, "Quant saturates");
  require(ref::quantize(std::nanf(""), q) == -3, "Quant maps NaN to the zero point");

  const auto v = ref::voxelize_case();
  const auto vr = ref::voxelize(v.options, v.points, v.point_count);
  // Pillars by first appearance: cell (x=2,y=1), (0,3), (3,0); rows are (y, x).
  require(vr.count == 3 && vr.coordinates == std::vector<std::int32_t>{1, 2, 3, 0, 0, 3},
          "SimaVoxelize first-appearance pillars and (row, column) order");
  // Pillar 2 holds one point (3.5, 0.5, 1.5, 0): mean offsets are zero and the
  // centre is (3.5, 0.5, 0), so its decoration is [3.5,0.5,1.5,0,0,0,0,0,0,1.5].
  const std::array<std::int8_t, 10> pillar2{11, -1, 3, -3, -3, -3, -3, -3, -3, 3};
  for (std::size_t c = 0; c < 10; ++c)
    require(vr.decorated[(2 * 8 + 0) * 10 + c] == pillar2[c],
            "SimaVoxelize single-point decoration");
  require(vr.decorated[(2 * 8 + 1) * 10] == -3, "SimaVoxelize pads with the zero point");

  const auto p = ref::bev_pool_plan_case();
  const auto plan = ref::bev_pool_plan(p.options, p.calibration);
  // Camera 0 keeps x*d in {0, 2} and camera 1 keeps x*d + 1 in {1, 3}; y*d in
  // {0, 2}: 10 of 24 frustum points land in the 4x4 grid.
  require(plan.size() == 64 + 4 * 17 + 4 * 10, "SimaBEVPoolPlan reference keeps 10 points");
}

} // namespace

// INT8 outputs publish `real = (code - zero_point) * scale`, with scale the
// inverse of the fused multiplier times the real value of one summed unit.
void published_quantization() {
  const auto close = [](const std::optional<kr::PerTensorQuant>& q, double scale,
                        std::int64_t zero_point) {
    return q && std::abs(q->scale - scale) <= 1e-12 * scale && q->zero_point == zero_point;
  };
  auto voxelize = ref::voxelize_case().options;
  const std::array points{fact({17, 4}, rt::DType::Float32), fact({1}, rt::DType::Int32)};
  require(close(kr::voxelize_request(voxelize, points).published[0].quant, 0.25, -3),
          "voxelize decorated quant is 1/multiplier");
  require(!kr::voxelize_request(voxelize, points).published[1].quant,
          "voxelize coordinates carry no quant");
  voxelize.output_dtype = TensorDType::BFloat16;
  voxelize.quantization = {};
  require(!kr::voxelize_request(voxelize, points).published[0].quant,
          "voxelize BF16 output carries no quant");

  const auto scatter = ref::pillar_scatter_case().options;
  std::array features{fact({5, 16}, rt::DType::Int8), fact({5, 2}, rt::DType::Int32),
                      fact({1}, rt::DType::Int32)};
  require(!kr::pillar_scatter_request(scatter, features).published[0].quant,
          "pillar_scatter without feature quant publishes none");
  features[0].quant = kr::PerTensorQuant{0.125, -7};
  require(close(kr::pillar_scatter_request(scatter, features).published[0].quant, 0.125, -7),
          "pillar_scatter canvas keeps the features' quant");

  auto pool = ref::bev_pool_case().options;
  pool.quantization.multiplier = 0.5f;
  pool.quantization.zero_point = 4;
  std::array inputs{fact({2, 2, 3, 2}, rt::DType::Int8), fact({2, 2, 3, 16}, rt::DType::Int8),
                    fact({228}, rt::DType::UInt8)};
  require(!kr::bev_pool_request(pool, inputs).published[0].quant,
          "bev_pool INT8 inputs without scales publish no quant");
  inputs[0].quant = kr::PerTensorQuant{0.01, -128};
  inputs[1].quant = kr::PerTensorQuant{0.2, 3};
  require(close(kr::bev_pool_request(pool, inputs).published[0].quant, 0.01 * 0.2 / 0.5, 4),
          "bev_pool INT8 quant is depth_scale*context_scale/multiplier");
  inputs[0] = fact({2, 2, 3, 2}, rt::DType::BFloat16);
  inputs[1] = fact({2, 2, 3, 16}, rt::DType::BFloat16);
  pool.depth_zero_point = pool.context_zero_point = 0;
  require(close(kr::bev_pool_request(pool, inputs).published[0].quant, 2.0, 4),
          "bev_pool INT8 from BF16 quant is 1/multiplier");
}

int main() {
  try {
    voxelize_request();
    pillar_scatter_request();
    bev_pool_plan_request();
    bev_pool_request();
    checked_extent_boundaries();
    dtype_tokens();
    reference_known_answers();
    bf16_requests();
    bf16_reference_known_answers();
    published_quantization();
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] unit_bevfusion_kernel_requests_test: " << error.what() << "\n";
    return 1;
  }
  std::cout << "[OK] unit_bevfusion_kernel_requests_test\n";
  return 0;
}

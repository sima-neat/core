#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

namespace simaai::neat::nodes_internal::kernel_requests {
namespace {

[[noreturn]] void fail(const char* node, const std::string& message) {
  throw std::invalid_argument(std::string(node) + ": " + message);
}

std::string describe(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i)
    text += (i ? "," : "") + std::to_string(shape[i]);
  return text + "]";
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* node) {
  if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
    fail(node, "tensor or plan extent overflows");
  return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* node) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    fail(node, "tensor or plan extent overflows");
  return a + b;
}

// Core shapes and byte strides are signed 64-bit values. Bound a derived
// extent before multiplying, not after a potentially overflowing expression.
std::int64_t checked_extent_mul(std::int64_t a, std::int64_t b, const char* node) {
  if (a <= 0 || b <= 0)
    fail(node, "tensor has a nonpositive extent");
  if (b > std::numeric_limits<std::int64_t>::max() / a)
    fail(node, "tensor extent overflows");
  return a * b;
}

std::int64_t element_count(const std::vector<std::int64_t>& shape, const char* node) {
  std::uint64_t count = 1;
  for (const auto dimension : shape) {
    if (dimension <= 0)
      fail(node, "tensor " + describe(shape) + " has a nonpositive extent");
    count = checked_mul(count, static_cast<std::uint64_t>(dimension), node);
  }
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    fail(node, "tensor " + describe(shape) + " overflows");
  return static_cast<std::int64_t>(count);
}

// True when explicit strides, if any, address the shape in dense row-major order.
// The stride of an extent-1 axis never addresses data and is ignored.
bool dense(const TensorFact& fact, const char* node) {
  checked_tensor_bytes(fact.shape, fact.dtype, node);
  if (fact.stride_bytes.empty())
    return true;
  if (fact.stride_bytes.size() != fact.shape.size())
    return false;
  std::int64_t expected = static_cast<std::int64_t>(scalar_bytes(fact.dtype));
  for (std::size_t axis = fact.shape.size(); axis-- > 0;) {
    if (fact.shape[axis] != 1 && fact.stride_bytes[axis] != expected)
      return false;
    expected = checked_extent_mul(expected, fact.shape[axis], node);
  }
  return true;
}

// Channels-last byte strides of a logical [A,K,H,W] tensor: physical [A,H,W,K].
std::array<std::int64_t, 4> channels_last_strides(const std::array<std::int64_t, 4>& logical,
                                                  std::size_t scalar, const char* node) {
  const auto s = static_cast<std::int64_t>(scalar);
  const auto [a, k, h, w] = logical;
  (void)a;
  const auto cell = checked_extent_mul(k, s, node);
  const auto row = checked_extent_mul(w, cell, node);
  return {checked_extent_mul(h, row, node), s, row, cell};
}

bool same_strides(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& actual,
                  const std::array<std::int64_t, 4>& expected) {
  if (shape.size() != 4 || actual.size() != 4)
    return false;
  for (std::size_t axis = 0; axis < 4; ++axis)
    if (shape[axis] != 1 && actual[axis] != expected[axis])
      return false;
  return true;
}

void require_arity(std::span<const TensorFact> facts, std::size_t count, const char* node,
                   const char* ports) {
  if (facts.size() != count)
    fail(node, "expects " + std::to_string(count) + " input tensors (" + ports + "), got " +
                   std::to_string(facts.size()));
}

void require_dtype(const TensorFact& fact, rt::DType dtype, const char* node, const char* port) {
  if (fact.dtype != dtype)
    fail(node,
         std::string(port) + " must be " + dtype_token(dtype) + ", got " + dtype_token(fact.dtype));
}

void require_dense(const TensorFact& fact, const char* node, const char* port) {
  if (!dense(fact, node))
    fail(node, std::string(port) + " " + describe(fact.shape) +
                   " must be dense; make it contiguous before this Node");
}

// A one-element INT32 count, published by the producer as [1] or with extra
// singleton axes.
void require_count(const TensorFact& fact, const char* node, const char* port) {
  require_dtype(fact, rt::DType::Int32, node, port);
  require_dense(fact, node, port);
  if (fact.shape.empty() || element_count(fact.shape, node) != 1)
    fail(node, std::string(port) + " must hold one INT32 value, got " + describe(fact.shape));
}

// Rows of `width` elements: the trailing axis is `width`, and at most one other
// axis exceeds 1, so [M,C] may carry extra singleton axes such as [1,M,1,C].
std::int64_t rows_of(const TensorFact& fact, std::int64_t width, const char* node,
                     const char* port) {
  if (fact.shape.size() < 2 || (width > 0 && fact.shape.back() != width))
    fail(node, std::string(port) + " must end in an axis of " +
                   (width > 0 ? std::to_string(width) : std::string("C")) + " elements, got " +
                   describe(fact.shape));
  const std::vector<std::int64_t> leading(fact.shape.begin(), fact.shape.end() - 1);
  const auto rows = element_count(leading, node);
  if (std::count_if(leading.begin(), leading.end(), [](auto extent) { return extent != 1; }) > 1)
    fail(node, std::string(port) + " may add only singleton axes to its rows, got " +
                   describe(fact.shape));
  return rows;
}

rt::Parameter float_array(std::string name, std::span<const float> values) {
  rt::ParameterArray array;
  array.reserve(values.size());
  for (const auto value : values)
    array.emplace_back(value);
  return {std::move(name), rt::ParameterValue(std::move(array))};
}

rt::Parameter int_array(std::string name, std::span<const int> values) {
  rt::ParameterArray array;
  array.reserve(values.size());
  for (const auto value : values)
    array.emplace_back(static_cast<std::int64_t>(value));
  return {std::move(name), rt::ParameterValue(std::move(array))};
}

void append_quantization(std::vector<rt::Parameter>& parameters, const SimaKernelQuantization& q) {
  parameters.push_back({"quantization_multiplier", q.multiplier});
  parameters.push_back({"quantization_zero_point", static_cast<std::int64_t>(q.zero_point)});
  parameters.push_back({"quantization_rounding", rounding_name(q.rounding)});
}

// The kernel stores `round(x * multiplier) + zero_point`, so a code dequantizes
// to `x` with `scale = 1 / multiplier`; `unit` is the real value of one `x`.
PerTensorQuant stored_quant(const SimaKernelQuantization& q, double unit = 1.0) {
  return {unit / static_cast<double>(q.multiplier), q.zero_point};
}

} // namespace

rt::Target kernel_target(SimaKernelTarget target) {
  switch (target) {
  case SimaKernelTarget::A65:
    return rt::Target::A65;
  case SimaKernelTarget::EV74:
    return rt::Target::EV74;
  }
  throw std::invalid_argument("unknown SimaKernelTarget");
}

const char* rounding_name(SimaQuantRounding rounding) {
  switch (rounding) {
  case SimaQuantRounding::NearestEven:
    return "nearest_even";
  case SimaQuantRounding::TowardZero:
    return "toward_zero";
  case SimaQuantRounding::TowardPositive:
    return "toward_positive";
  case SimaQuantRounding::TowardNegative:
    return "toward_negative";
  }
  throw std::invalid_argument("unknown SimaQuantRounding");
}

std::size_t scalar_bytes(rt::DType dtype) {
  switch (dtype) {
  case rt::DType::Int8:
  case rt::DType::UInt8:
    return 1;
  case rt::DType::BFloat16:
  case rt::DType::Float16:
  case rt::DType::Int16:
  case rt::DType::UInt16:
    return 2;
  case rt::DType::Float32:
  case rt::DType::Int32:
    return 4;
  }
  return 0;
}

std::uint64_t checked_tensor_bytes(std::span<const std::int64_t> shape, rt::DType dtype,
                                   const char* node) {
  const auto scalar = scalar_bytes(dtype);
  if (!scalar)
    fail(node, "unsupported tensor dtype");
  auto bytes = static_cast<std::int64_t>(scalar);
  for (const auto dimension : shape)
    bytes = checked_extent_mul(bytes, dimension, node);
  return static_cast<std::uint64_t>(bytes);
}

std::string dtype_token(rt::DType dtype) {
  switch (dtype) {
  case rt::DType::Float32:
    return "FP32";
  case rt::DType::BFloat16:
    return "BF16";
  case rt::DType::Int8:
    return "INT8";
  case rt::DType::UInt8:
    return "UINT8";
  case rt::DType::Int16:
    return "INT16";
  case rt::DType::Int32:
    return "INT32";
  case rt::DType::Float16:
    return "FP16";
  case rt::DType::UInt16:
    return "UINT16";
  }
  return "UNKNOWN";
}

rt::DType dtype_from_token(const std::string& raw, const char* node) {
  std::string token = raw;
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  // Model stages may qualify a dtype with its EV container prefix.
  if (token.rfind("EVXX_", 0) == 0)
    token.erase(0, 5);
  if (token == "FP32" || token == "FLOAT32")
    return rt::DType::Float32;
  if (token == "BF16" || token == "BFLOAT16")
    return rt::DType::BFloat16;
  if (token == "INT8")
    return rt::DType::Int8;
  if (token == "UINT8")
    return rt::DType::UInt8;
  if (token == "INT16")
    return rt::DType::Int16;
  if (token == "UINT16")
    return rt::DType::UInt16;
  if (token == "INT32")
    return rt::DType::Int32;
  if (token == "FP16" || token == "FLOAT16")
    return rt::DType::Float16;
  fail(node, "unsupported input dtype " + raw);
}

namespace {
// The element types these kernels admit for their narrow ports.
rt::DType narrow_dtype(TensorDType dtype, const char* node, const char* field) {
  switch (dtype) {
  case TensorDType::Int8:
    return rt::DType::Int8;
  case TensorDType::BFloat16:
    return rt::DType::BFloat16;
  default:
    fail(node, std::string(field) + " must be Int8 or BFloat16");
  }
}
} // namespace

void validate(const SimaVoxelizeOptions& opt) {
  if (opt.max_pillars <= 0 || opt.max_points_per_pillar <= 0)
    fail("SimaVoxelize", "max_pillars and max_points_per_pillar must be positive");
  narrow_dtype(opt.output_dtype, "SimaVoxelize", "output_dtype");
}

void validate(const SimaPillarScatterOptions& opt) {
  if (opt.rows <= 0 || opt.columns <= 0)
    fail("SimaPillarScatter", "rows and columns must be positive");
}

void validate(const SimaBEVPoolPlanOptions& opt) {
  if (opt.xs.empty() || opt.ys.empty() || opt.ds.empty())
    fail("SimaBEVPoolPlan", "xs, ys and ds must each hold at least one frustum sample");
  if (std::any_of(opt.nx.begin(), opt.nx.end(), [](int n) { return n <= 0; }))
    fail("SimaBEVPoolPlan", "every nx cell count must be positive");
  if (opt.format != SimaBEVPoolPlanFormat::V1 && opt.format != SimaBEVPoolPlanFormat::V2)
    fail("SimaBEVPoolPlan", "format must be V1 or V2");
  if (opt.plan_capacity_bytes < 0)
    fail("SimaBEVPoolPlan", "plan_capacity_bytes must be zero (worst case) or positive");
  narrow_dtype(opt.input_dtype, "SimaBEVPoolPlan", "input_dtype");
}

void validate(const SimaBEVPoolOptions& opt) {
  if (std::any_of(opt.cells.begin(), opt.cells.end(), [](int n) { return n <= 0; }))
    fail("SimaBEVPool", "every cell count must be positive");
  if (opt.plan_format != SimaBEVPoolPlanFormat::V1 && opt.plan_format != SimaBEVPoolPlanFormat::V2)
    fail("SimaBEVPool", "plan_format must be V1 or V2");
  const auto valid_storage = [](SimaBEVPoolStorage storage) {
    return storage == SimaBEVPoolStorage::Contiguous || storage == SimaBEVPoolStorage::ChannelsLast;
  };
  if (!valid_storage(opt.input_storage) || !valid_storage(opt.output_storage))
    fail("SimaBEVPool", "input_storage and output_storage must be Contiguous or ChannelsLast");
  narrow_dtype(opt.output_dtype, "SimaBEVPool", "output_dtype");
}

KernelRequestPlan voxelize_request(const SimaVoxelizeOptions& opt,
                                   std::span<const TensorFact> facts) {
  constexpr const char* node = "SimaVoxelize";
  validate(opt);
  require_arity(facts, 2, node, "points, point_count");
  const auto& points = facts[0];
  require_dtype(points, rt::DType::Float32, node, "points");
  require_dense(points, node, "points");
  if (points.shape.size() != 2 || points.shape[1] != 4 || points.shape[0] <= 0)
    fail(node, "points must be [N,4], got " + describe(points.shape));
  require_count(facts[1], node, "point_count");

  const std::int64_t m = opt.max_pillars, t = opt.max_points_per_pillar;
  const auto decorated = narrow_dtype(opt.output_dtype, node, "output_dtype");
  KernelRequestPlan plan;
  plan.operation = "voxelize";
  plan.target = kernel_target(opt.target);
  plan.inputs = {{{points.shape[0], 4}, rt::DType::Float32}, {{1}, rt::DType::Int32}};
  // A BF16 output admits only the identity quantization, which the kernels
  // check; the fields are forwarded unchanged so a non-identity value fails.
  plan.outputs = {{{1, m, t, 10}, decorated}, {{m, 2}, rt::DType::Int32}, {{1}, rt::DType::Int32}};
  plan.parameters.push_back(float_array("voxel_size", opt.voxel_size));
  plan.parameters.push_back(float_array("point_cloud_range", opt.point_cloud_range));
  append_quantization(plan.parameters, opt.quantization);
  plan.published = {{"decorated", {1, m, t, 10}, {}, "NHWC", false, {}},
                    {"coordinates", {m, 2}, {}, "", false, {}},
                    {"count", {1}, {}, "", false, {}}};
  if (decorated == rt::DType::Int8)
    plan.published[0].quant = stored_quant(opt.quantization);
  return plan;
}

KernelRequestPlan pillar_scatter_request(const SimaPillarScatterOptions& opt,
                                         std::span<const TensorFact> facts) {
  constexpr const char* node = "SimaPillarScatter";
  validate(opt);
  require_arity(facts, 3, node, "features, coordinates, count");
  const auto& features = facts[0];
  if (features.dtype != rt::DType::Int8 && features.dtype != rt::DType::BFloat16)
    fail(node, "features must be INT8 or BF16, got " + dtype_token(features.dtype));
  require_dense(features, node, "features");
  const auto channels = features.shape.empty() ? 0 : features.shape.back();
  const auto pillars = rows_of(features, 0, node, "features");
  const auto& coordinates = facts[1];
  require_dtype(coordinates, rt::DType::Int32, node, "coordinates");
  require_dense(coordinates, node, "coordinates");
  const auto coordinate_rows = rows_of(coordinates, 2, node, "coordinates");
  if (coordinate_rows != pillars)
    fail(node, "features hold " + std::to_string(pillars) + " pillars but coordinates hold " +
                   std::to_string(coordinate_rows));
  require_count(facts[2], node, "count");

  const std::int64_t h = opt.rows, w = opt.columns;
  // Full-channel cell tiles preserve the existing canvas bytes while allowing
  // the provider to assemble bounded panels instead of one canvas-sized tile.
  // INT8 is dense NHWC; BF16 keeps each cell's low then high channel bytes.
  rt::MlaTiled cells;
  cells.axes = {0, 1, 2, 3};
  cells.tile_shape = {1, 1, 1, channels};
  cells.channels = rt::MlaTiled::Channels::Compact;
  KernelRequestPlan plan;
  plan.operation = "pillar_scatter";
  plan.target = kernel_target(opt.target);
  // The canvas keeps the features' element type; BF16 uses the MlaTiled BF16
  // payload (per cell, the channels' low bytes, then their high bytes).
  plan.inputs = {{{pillars, channels}, features.dtype},
                 {{pillars, 2}, rt::DType::Int32},
                 {{1}, rt::DType::Int32}};
  plan.outputs = {{{1, h, w, channels}, features.dtype, cells}};
  plan.parameters.push_back({"zero_point", static_cast<std::int64_t>(opt.zero_point)});
  plan.published = {{"canvas", {1, h, w, channels}, {}, "NHWC", true, {}}};
  // Features are copied unchanged, so the canvas keeps their quantization.
  if (features.dtype == rt::DType::Int8)
    plan.published[0].quant = features.quant;
  return plan;
}

std::uint64_t bev_pool_plan_capacity_bytes(const SimaBEVPoolPlanOptions& opt,
                                           std::int64_t cameras) {
  constexpr const char* node = "SimaBEVPoolPlan";
  validate(opt);
  if (cameras <= 0)
    fail(node, "camera count must be positive");
  const auto n = static_cast<std::uint64_t>(cameras);
  const auto columns = static_cast<std::uint64_t>(opt.xs.size());
  const auto rows = static_cast<std::uint64_t>(opt.ys.size());
  const auto depth = static_cast<std::uint64_t>(opt.ds.size());
  const auto points =
      checked_mul(checked_mul(checked_mul(n, depth, node), rows, node), columns, node);
  if (opt.format == SimaBEVPoolPlanFormat::V1) {
    // bev_pool::plan_layout::bytes: header, cell_start[cells + 1], point[points].
    const auto cells = checked_mul(checked_mul(static_cast<std::uint64_t>(opt.nx[0]),
                                               static_cast<std::uint64_t>(opt.nx[1]), node),
                                   static_cast<std::uint64_t>(opt.nx[2]), node);
    return checked_add(checked_add(64, checked_mul(4, checked_add(cells, 1, node), node), node),
                       checked_mul(4, points, node), node);
  }
  // bev_pool::plan_layout_v2::bytes, from the kernels' published v2 bounds:
  // header, worker table, tile headers, chunk records and point words. The
  // depth window is a byte budget, so BF16 scalars halve its bins. The kernel
  // rejects a capacity below its own bound.
  if (opt.tile_h <= 0 || opt.tile_w <= 0)
    fail(node, "tile_h and tile_w must be positive");
  constexpr std::uint64_t header_bytes = 64, max_workers = 16, depth_window_bytes = 16384,
                          max_window_bins = 256;
  const auto tile = [&](int cells, int size) {
    return (static_cast<std::uint64_t>(cells) + static_cast<std::uint64_t>(size) - 1) /
           static_cast<std::uint64_t>(size);
  };
  const auto tiles = checked_mul(tile(opt.nx[0], opt.tile_h), tile(opt.nx[1], opt.tile_w), node);
  const std::uint64_t scalar = opt.input_dtype == TensorDType::BFloat16 ? 2 : 1;
  const auto bins = std::min<std::uint64_t>(depth_window_bytes / scalar / rows, max_window_bins);
  if (bins == 0)
    fail(node, "format V2 requires at most 256 image rows");
  const auto windows = depth / bins + (depth % bins != 0);
  const auto by_tiles =
      checked_mul(checked_mul(tiles, checked_mul(n, columns, node), node), windows, node);
  const auto records = std::min(by_tiles, points);
  auto bytes = checked_add(header_bytes, 8 * max_workers, node);
  bytes = checked_add(bytes, checked_mul(8, tiles, node), node);
  bytes = checked_add(bytes, checked_mul(20, records, node), node);
  return checked_add(bytes, checked_mul(8, points, node), node);
}

KernelRequestPlan bev_pool_plan_request(const SimaBEVPoolPlanOptions& opt,
                                        std::span<const TensorFact> facts) {
  constexpr const char* node = "SimaBEVPoolPlan";
  validate(opt);
  require_arity(facts, 4, node,
                "camera2lidar, camera_intrinsics, img_aug_matrix, "
                "lidar_aug_matrix");
  constexpr const char* ports[] = {"camera2lidar", "camera_intrinsics", "img_aug_matrix",
                                   "lidar_aug_matrix"};
  for (std::size_t i = 0; i < facts.size(); ++i) {
    require_dtype(facts[i], rt::DType::Float32, node, ports[i]);
    require_dense(facts[i], node, ports[i]);
  }
  const auto cameras = facts[0].shape.empty() ? 0 : facts[0].shape[0];
  for (std::size_t i = 0; i < 3; ++i)
    if (facts[i].shape != std::vector<std::int64_t>{cameras, 4, 4} || cameras <= 0)
      fail(node, std::string(ports[i]) + " must be [N,4,4] with the same N, got " +
                     describe(facts[i].shape));
  if (facts[3].shape != std::vector<std::int64_t>{4, 4})
    fail(node, "lidar_aug_matrix must be [4,4], got " + describe(facts[3].shape));

  const std::uint64_t capacity = opt.plan_capacity_bytes > 0
                                     ? static_cast<std::uint64_t>(opt.plan_capacity_bytes)
                                     : bev_pool_plan_capacity_bytes(opt, cameras);
  if (capacity > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    fail(node, "plan capacity overflows");
  const auto bytes = static_cast<std::int64_t>(capacity);

  KernelRequestPlan plan;
  plan.operation = "bev_pool_plan";
  plan.target = kernel_target(opt.target);
  for (std::size_t i = 0; i < facts.size(); ++i)
    plan.inputs.push_back({facts[i].shape, rt::DType::Float32});
  plan.outputs = {{{bytes}, rt::DType::UInt8}};
  plan.parameters.push_back(float_array("xs", opt.xs));
  plan.parameters.push_back(float_array("ys", opt.ys));
  plan.parameters.push_back(float_array("ds", opt.ds));
  plan.parameters.push_back(float_array("bx", opt.bx));
  plan.parameters.push_back(float_array("dx", opt.dx));
  plan.parameters.push_back(int_array("nx", opt.nx));
  plan.parameters.push_back({"format", static_cast<std::int64_t>(opt.format)});
  plan.parameters.push_back({"context_channels", static_cast<std::int64_t>(opt.context_channels)});
  plan.parameters.push_back({"workers", static_cast<std::int64_t>(opt.workers)});
  plan.parameters.push_back({"tile_h", static_cast<std::int64_t>(opt.tile_h)});
  plan.parameters.push_back({"tile_w", static_cast<std::int64_t>(opt.tile_w)});
  plan.parameters.push_back({"cache_plan", opt.cache_plan});
  // Int8 is the kernels' default; only a BF16 plan names its input dtype.
  if (narrow_dtype(opt.input_dtype, node, "input_dtype") == rt::DType::BFloat16)
    plan.parameters.push_back({"input_dtype", "bfloat16"});
  plan.published = {{"plan", {bytes}, {}, "", false, {}}};
  return plan;
}

namespace {
// Logical [A,K,H,W] of a depth or context input, and its kernel layout.
rt::TensorSpec bev_input(const TensorFact& fact, SimaBEVPoolStorage storage, const char* port) {
  constexpr const char* node = "SimaBEVPool";
  if (fact.dtype != rt::DType::Int8 && fact.dtype != rt::DType::BFloat16)
    fail(node, std::string(port) + " must be INT8 or BF16, got " + dtype_token(fact.dtype));
  if (fact.shape.size() != 4)
    fail(node, std::string(port) + " must have rank 4, got " + describe(fact.shape));
  checked_tensor_bytes(fact.shape, fact.dtype, node);
  const auto& s = fact.shape;
  if (storage == SimaBEVPoolStorage::Contiguous) {
    require_dense(fact, node, port);
    return {s, fact.dtype};
  }
  const auto scalar = scalar_bytes(fact.dtype);
  std::array<std::int64_t, 4> logical{};
  if (dense(fact, node)) {
    // Dense NHWC [A,H,W,K] from the producer: the same bytes as logical
    // [A,K,H,W] with channels-last strides.
    logical = {s[0], s[3], s[1], s[2]};
  } else {
    logical = {s[0], s[1], s[2], s[3]};
    if (!same_strides(s, fact.stride_bytes, channels_last_strides(logical, scalar, node)))
      fail(node, std::string(port) + " " + describe(s) +
                     " is neither dense channels-last nor channels-last strided");
  }
  const auto strides = channels_last_strides(logical, scalar, node);
  rt::Strided layout;
  std::copy(strides.begin(), strides.end(), layout.strides_bytes.begin());
  return {{logical.begin(), logical.end()}, fact.dtype, layout};
}

} // namespace

KernelRequestPlan bev_pool_request(const SimaBEVPoolOptions& opt,
                                   std::span<const TensorFact> facts) {
  constexpr const char* node = "SimaBEVPool";
  validate(opt);
  require_arity(facts, 3, node, "depth, context, plan");
  auto depth = bev_input(facts[0], opt.input_storage, "depth");
  auto context = bev_input(facts[1], opt.input_storage, "context");
  const auto& plan_fact = facts[2];
  require_dtype(plan_fact, rt::DType::UInt8, node, "plan");
  require_dense(plan_fact, node, "plan");
  if (plan_fact.shape.size() != 1 || plan_fact.shape[0] <= 0)
    fail(node, "plan must be a rank-1 UINT8 byte tensor, got " + describe(plan_fact.shape));

  const std::int64_t nx = opt.cells[0], ny = opt.cells[1], nz = opt.cells[2];
  const auto channels = context.shape[1];
  const auto k = checked_extent_mul(nz, channels, node);
  const auto dtype = narrow_dtype(opt.output_dtype, node, "output_dtype");
  const std::array<std::int64_t, 4> logical{1, k, nx, ny};
  checked_tensor_bytes(logical, dtype, node);
  rt::TensorSpec bev{{logical.begin(), logical.end()}, dtype};
  PublishedOutput published{"bev", {logical.begin(), logical.end()}, {}, "", false, {}};
  if (opt.output_storage == SimaBEVPoolStorage::ChannelsLast) {
    const auto strides = channels_last_strides(logical, scalar_bytes(dtype), node);
    rt::Strided layout;
    std::copy(strides.begin(), strides.end(), layout.strides_bytes.begin());
    bev.layout = layout;
    published.stride_bytes.assign(strides.begin(), strides.end());
  }

  KernelRequestPlan plan;
  plan.operation = "bev_pool";
  plan.target = kernel_target(opt.target);
  plan.inputs = {std::move(depth), std::move(context), {plan_fact.shape, rt::DType::UInt8}};
  plan.outputs = {std::move(bev)};
  plan.parameters.push_back({"depth_zero_point", static_cast<std::int64_t>(opt.depth_zero_point)});
  plan.parameters.push_back(
      {"context_zero_point", static_cast<std::int64_t>(opt.context_zero_point)});
  plan.parameters.push_back({"plan_format", static_cast<std::int64_t>(opt.plan_format)});
  append_quantization(plan.parameters, opt.quantization);
  // An INT8 sum of codes is real only through both input scales; BF16 sums are
  // real already. Without both scales the output scale is unknown.
  if (dtype == rt::DType::Int8) {
    if (facts[0].dtype == rt::DType::BFloat16)
      published.quant = stored_quant(opt.quantization);
    else if (facts[0].quant && facts[1].quant)
      published.quant =
          stored_quant(opt.quantization, facts[0].quant->scale * facts[1].quant->scale);
  }
  plan.published = {std::move(published)};
  return plan;
}

} // namespace simaai::neat::nodes_internal::kernel_requests

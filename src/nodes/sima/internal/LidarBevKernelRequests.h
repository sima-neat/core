#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Pure construction of the named kernels-library requests behind the SimaVoxelize,
// SimaPillarScatter, SimaBEVPoolPlan and SimaBEVPool Nodes. No Scheduler call, GStreamer or
// runtime state is involved, so the exact request Core forwards is testable in
// isolation. Numerical admission remains the kernels provider's: these builders
// only derive tensor specifications from the upstream contract and the options.

#include "nodes/sima/SimaBEVPool.h"
#include "nodes/sima/SimaBEVPoolPlan.h"
#include "nodes/sima/SimaPillarScatter.h"
#include "nodes/sima/SimaVoxelize.h"

#include <simaai/scheduler/client/parameters.h>
#include <simaai/scheduler/client/tensor.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::nodes_internal::kernel_requests {
namespace rt = ::simaai::scheduler::client;

/// Per-tensor quantization: `real = (code - zero_point) * scale`.
struct PerTensorQuant {
  double scale = 1.0;
  std::int64_t zero_point = 0;
};

/// One upstream tensor, as the Node's contract compiler observed it.
struct TensorFact {
  std::vector<std::int64_t> shape;
  rt::DType dtype = rt::DType::Float32;
  std::vector<std::int64_t> stride_bytes; ///< Empty or per-axis byte strides.
  std::string name;
  std::optional<PerTensorQuant> quant; ///< Set when the producer declares one.
};

/// How the Node publishes one checked kernel output.
struct PublishedOutput {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride_bytes; ///< Empty for dense publication.
  std::string layout;                     ///< Coarse layout token ("" or "NHWC").
  /// The checked output is MLA-tiled; publication needs an explicit physical view.
  bool tiled = false;
  /// The quantization of an INT8 output, when the request determines it.
  std::optional<PerTensorQuant> quant;
};

/// The complete named request a Node forwards to `check_kernel`.
struct KernelRequestPlan {
  std::string operation;
  rt::Target target = rt::Target::A65;
  std::vector<rt::TensorSpec> inputs, outputs;
  std::vector<rt::Parameter> parameters;
  std::vector<PublishedOutput> published; ///< One per output, same order.
};

// Each builder throws std::invalid_argument, prefixed with the Node kind, when
// the upstream tensors cannot form the operation's request.
KernelRequestPlan voxelize_request(const SimaVoxelizeOptions&, std::span<const TensorFact>);
KernelRequestPlan pillar_scatter_request(const SimaPillarScatterOptions&,
                                         std::span<const TensorFact>);
KernelRequestPlan bev_pool_plan_request(const SimaBEVPoolPlanOptions&, std::span<const TensorFact>);
KernelRequestPlan bev_pool_request(const SimaBEVPoolOptions&, std::span<const TensorFact>);

/// Worst-case plan bytes for `cameras` cameras, from the kernels' published plan layouts.
std::uint64_t bev_pool_plan_capacity_bytes(const SimaBEVPoolPlanOptions&, std::int64_t cameras);

// Option checks shared by the Node constructors (structural fields only).
void validate(const SimaVoxelizeOptions&);
void validate(const SimaPillarScatterOptions&);
void validate(const SimaBEVPoolPlanOptions&);
void validate(const SimaBEVPoolOptions&);

rt::Target kernel_target(SimaKernelTarget);
const char* rounding_name(SimaQuantRounding);
std::size_t scalar_bytes(rt::DType);
/// Core's dtype token ("INT8", "FP32", ...) for a kernel dtype.
std::string dtype_token(rt::DType);
/// Inverse of `dtype_token`; throws on an unsupported token.
rt::DType dtype_from_token(const std::string& token, const char* node);

} // namespace simaai::neat::nodes_internal::kernel_requests

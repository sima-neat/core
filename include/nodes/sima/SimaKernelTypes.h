/**
 * @file
 * @ingroup nodes_sima
 * @brief Shared option types for Nodes that run one SiMa kernels-library operation.
 *
 * `SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan` and `SimaBEVPool` each prepare one named
 * kernels-library request through the Scheduler client. They share the execution
 * target and the fused quantization parameters declared here.
 */
#pragma once

namespace simaai::neat {

/**
 * @brief Hardware that executes a kernels-library Node.
 *
 * The selected target is final: preparation fails if the operation has no
 * implementation for it. There is no automatic fallback to another target.
 *
 * @ingroup nodes_sima
 */
enum class SimaKernelTarget {
  A65,  ///< Application CPU (Arm Cortex-A65).
  EV74, ///< EV74 vector DSP.
};

/**
 * @brief Integer rounding applied after the binary32 product of a fused quantization.
 *
 * @ingroup nodes_sima
 */
enum class SimaQuantRounding {
  NearestEven,    ///< Round half to even (default).
  TowardZero,     ///< Truncate.
  TowardPositive, ///< Ceiling.
  TowardNegative, ///< Floor.
};

/**
 * @brief Quantization fused into a kernel's store.
 *
 * The kernel computes `saturate_int8(round(binary32(x * multiplier)) + zero_point)`,
 * with `round` selected by `rounding`. `multiplier` multiplies the value; it is not a
 * dequantization scale. When the Node knows the real value of one unit of `x`, the INT8
 * output carries `Tensor::semantic.quant` with that value divided by `multiplier` as its
 * scale, and this `zero_point`.
 *
 * @ingroup nodes_sima
 */
struct SimaKernelQuantization {
  float multiplier = 1.0f; ///< Positive, finite multiplier.
  int zero_point = 0;      ///< INT8 zero point, in `[-128, 127]`.
  SimaQuantRounding rounding = SimaQuantRounding::NearestEven; ///< Integer rounding mode.
};

/**
 * @brief Layout of a `SimaBEVPoolPlan` plan, which `SimaBEVPool` must be told.
 *
 * @ingroup nodes_sima
 */
enum class SimaBEVPoolPlanFormat {
  V1 = 1, ///< Points grouped by BEV cell.
  V2 = 2, ///< Work tiles with planned cache loads, for EV74 `SimaBEVPool`.
};

} // namespace simaai::neat

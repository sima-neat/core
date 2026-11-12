#pragma once

namespace simaai::neat {
/// Sampling algorithm shared by Model preprocessing and standalone Resize.
/// Existing enumerator values are preserved.
enum class ResizeInterpolation { Nearest, Bilinear, Bicubic, BoxAverage, BilinearAntialiased };
/// Whether YUV-to-RGB conversion precedes image sampling. No transfer-function conversion.
enum class PreprocColorConversionPlacement { AfterResize = 0, BeforeResize = 1 };
/// Legacy preserves BF16 truncation. NearestEven explicitly selects BF16 nearest-even.
/// FP16 remains binary16 nearest-even in either mode.
enum class PreprocFloatingConversion { Legacy = 0, NearestEven = 1 };
} // namespace simaai::neat

#pragma once

namespace simaai::neat {
/// Whether YUV-to-RGB conversion precedes image sampling. No transfer-function conversion.
enum class PreprocColorConversionPlacement { AfterResize = 0, BeforeResize = 1 };
/// Legacy preserves BF16 truncation. NearestEven explicitly selects BF16 nearest-even.
/// FP16 remains binary16 nearest-even in either mode.
enum class PreprocFloatingConversion { Legacy = 0, NearestEven = 1 };
} // namespace simaai::neat

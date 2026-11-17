#include "nodes/sima/internal/CropGeometry.h"
#include <iostream>
#include <stdexcept>

#include <cfenv>
#include <cmath>
#include <limits>

namespace {
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
using simaai::neat::internal::checked_crop_geometry;
using simaai::neat::internal::CropGeometry;
void verify(int sw, int sh, int ow, int oh, std::optional<int> shorter,
            std::array<std::int64_t, 4> expected) {
  const auto actual = checked_crop_geometry(sw, sh, ow, oh, shorter);
  require(std::holds_alternative<CropGeometry>(actual), "valid crop geometry rejected");
  const auto& geometry = std::get<CropGeometry>(actual);
  require(geometry.window == expected, "integer resize/center rounding changed");
  const auto& a = geometry.model_to_source;
  const double sx = double(sw) / expected[0], sy = double(sh) / expected[1];
  require(a[0] == sx && a[1] == 0 && a[2] == expected[2] * sx && a[3] == 0 && a[4] == sy &&
              a[5] == expected[3] * sy,
          "box-edge inverse affine incorrectly includes sampling-center offsets");
}
} // namespace

int main() {
  try {
    verify(640, 480, 224, 224, 256, {341, 256, 58, 16});
    verify(480, 640, 224, 224, 256, {256, 341, 16, 58});
    verify(640, 480, 224, 224, {}, {299, 224, 38, 0});
    verify(480, 640, 224, 224, {}, {224, 299, 0, 38});
    verify(8, 8, 8, 8, {}, {8, 8, 0, 0});
    verify(1, 1, 1, 1, {}, {1, 1, 0, 0});
    verify(4096, 4096, 4096, 4096, {}, {4096, 4096, 0, 0});
    const auto previous = std::fegetround();
    for (const auto rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      require(std::fesetround(rounding) == 0, "cannot set host rounding mode");
      for (int difference = 0; difference <= 4095; ++difference) {
        // Independent integer parity oracle for round-to-nearest-even of d/2.
        const auto lower = difference / 2;
        const auto expected = difference % 2 == 0 ? lower : lower % 2 == 0 ? lower : lower + 1;
        require(simaai::neat::internal::center_crop_offset(difference) == expected,
                "center placement depends on FP rounding or rounds an odd tie incorrectly");
      }
    }
    require(std::fesetround(previous) == 0, "cannot restore host rounding mode");
    for (const auto bad : {0LL, -1LL, 4097LL, std::numeric_limits<long long>::max()})
      for (int dimension = 0; dimension < 4; ++dimension) {
        std::array<std::int64_t, 4> sizes{8, 8, 8, 8};
        sizes[dimension] = bad;
        require(std::holds_alternative<const char*>(
                    checked_crop_geometry(sizes[0], sizes[1], sizes[2], sizes[3], {})),
                "invalid extent accepted before cross-product arithmetic");
      }
    for (const auto shorter : {0, -1, 7, 4097})
      require(std::holds_alternative<const char*>(checked_crop_geometry(8, 8, 8, 8, shorter)),
              "invalid/undersized shorter side was silently padded or enlarged");
    require(std::holds_alternative<const char*>(checked_crop_geometry(4096, 1, 8, 8, 8)),
            "overflowing virtual extent escaped checked geometry");
    std::cout << "Crop geometry regression PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}

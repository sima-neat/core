// Regression coverage for MLA detessellation boundary transport sizing.
//
// `any_shape_on_mla` models emit detess output boundaries whose `frame_shape`
// is a low-rank flat buffer (rank < 3, e.g. `[1, N]` or `[1, N, K]` that is not
// laid out as spatial H/W/C). The transport-view sizing helper previously bailed
// to 0 for any rank < 3 shape, which made the MLA boundary transport byte-span
// check fail to load otherwise-valid models with:
//   "MLA boundary transport tensor byte span mismatch ... expected=0 tensor=<n>"
// (reported as GitHub issue #549). These cases lock in the dense-flat sizing for
// low-rank shapes while keeping the spatial (rank >= 3, channel-block aligned)
// path unchanged.

#include "pipeline/internal/sima/MpkContract.h"
#include "test_main.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

using simaai::neat::pipeline_internal::sima::expected_detess_packed_transport_bytes;
using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;

MpkPluginIoContract make_detess_stage(std::vector<std::int64_t> frame_shape,
                                      const bool channel_block_aligned) {
  MpkPluginIoContract stage;
  stage.name = "detessellate_MLA_0/tuple_get_item";
  stage.kernel = "detessellation_transform";
  stage.frame_shape = std::move(frame_shape);
  stage.has_align_c16 = channel_block_aligned;
  stage.align_c16 = channel_block_aligned;
  stage.has_cblock = channel_block_aligned;
  stage.cblock = channel_block_aligned;
  return stage;
}

} // namespace

RUN_TEST(
    "unit_detess_transport_bytes_test", ([] {
      // --- Low-rank flat outputs (the regression) -------------------------------
      // rf-detr backbone `tuple_get_item_1`: frame_shape [1, 1296] int8, whose
      // packed transport span is exactly the dense flat size 1296. Previously 0.
      require(expected_detess_packed_transport_bytes(make_detess_stage({1, 1296}, true), "int8") ==
                  1296U,
              "2D int8 detess boundary should size as dense flat bytes (1*1296*1)");

      // SigLIP2 pooler head: frame_shape [1, 768] bf16 -> 768*2 = 1536 bytes.
      require(expected_detess_packed_transport_bytes(make_detess_stage({1, 768}, true), "bf16") ==
                  1536U,
              "2D bf16 detess boundary should size as dense flat bytes (1*768*2)");

      // Bare 1D vector.
      require(expected_detess_packed_transport_bytes(make_detess_stage({1296}, false), "int8") ==
                  1296U,
              "1D int8 detess boundary should size as dense flat bytes");

      // --- Spatial outputs (must remain unchanged) ------------------------------
      // 4D [1, 36, 36, 256] int8: leading batch stripped, 256 already 16-aligned.
      require(expected_detess_packed_transport_bytes(make_detess_stage({1, 36, 36, 256}, true),
                                                     "int8") == 331776U,
              "4D spatial int8 detess boundary keeps H*W*C sizing");

      // 3D [1, 1296, 4] int8 with channel-block alignment: depth 4 rounds to 16,
      // giving 1*1296*16*1 = 20736 (rf-detr `tuple_get_item_2` transport input).
      require(expected_detess_packed_transport_bytes(make_detess_stage({1, 1296, 4}, true),
                                                     "int8") == 20736U,
              "3D spatial int8 detess boundary applies c16 channel alignment");

      // --- Degenerate inputs still report 'cannot size' (0) ---------------------
      require(expected_detess_packed_transport_bytes(make_detess_stage({}, true), "int8") == 0U,
              "empty frame_shape cannot be sized");
      require(expected_detess_packed_transport_bytes(make_detess_stage({1, 1296}, true), "") == 0U,
              "empty dtype cannot be sized");
    }));

#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "test_main.h"

RUN_TEST("unit_static_spec_builders_dtype_token_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           require(specbuilders::dtype_size_bytes_from_token("FP32") == 4U,
                   "FP32 must resolve to 4-byte elements");
           require(specbuilders::dtype_size_bytes_from_token("fp32") == 4U,
                   "FP32 resolution should be case-insensitive");
           require(specbuilders::dtype_size_bytes_from_token("FP16") == 2U,
                   "FP16 must resolve to 2-byte elements");

           const auto logical = specbuilders::build_logical_output_static_spec(
               /*logical_index=*/0,
               /*backend_output_index=*/0,
               /*physical_index=*/0,
               /*output_slot=*/0,
               /*tensor_index=*/0,
               /*shape=*/{80, 80, 64},
               /*dtype=*/"FP32",
               /*layout=*/"HWC",
               /*logical_name=*/"out0",
               /*backend_name=*/"out0",
               /*segment_name=*/"output_tensor",
               /*byte_offset=*/0,
               /*size_bytes_override=*/0U,
               /*quant=*/std::nullopt);

           require(logical.size_bytes == 80ULL * 80ULL * 64ULL * 4ULL,
                   "FP32 logical output must size from 4-byte elements");
           require(logical.dtype_source == DTypeSource::InternalContract,
                   "builder-created logical outputs must carry internal dtype provenance");
           require(logical.stride_bytes.size() == 3U,
                   "FP32 logical output must preserve rank when building strides");
           require(logical.stride_bytes[0] == 80LL * 64LL * 4LL,
                   "FP32 leading stride must scale by 4-byte elements");
           require(logical.stride_bytes[1] == 64LL * 4LL,
                   "FP32 middle stride must scale by 4-byte elements");
           require(logical.stride_bytes[2] == 4LL,
                   "FP32 trailing stride must scale by 4-byte elements");
         }));

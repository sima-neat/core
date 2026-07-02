#include "model/internal/ModelPack.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/VisualFrontend.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemanticsInternal.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.h"
#include "test_main.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require_build_dense_test_tensor(const std::vector<int>& shape, const std::string& dtype,
                                     const std::string& layout, sima_ev_tensor_desc* out,
                                     const char* expectation) {
  std::string error_detail;
  require(simaai::neat::pipeline_internal::sima::tensorsemantics::build_dense_tensor_desc(
              shape, dtype, layout, out, &error_detail, "test_tensor_output_missing",
              "test_tensor_shape_rank_invalid", "test_tensor_shape_dim_invalid",
              "test_tensor_dtype_invalid", "test_tensor_stride_output_missing"),
          expectation);
}

void require_build_tiled_test_tensor(const std::vector<int>& shape,
                                     const std::vector<int>& tile_shape, const std::string& dtype,
                                     const std::string& layout, std::uint32_t tile_align_bytes,
                                     sima_ev_tensor_desc* out, const char* expectation) {
  std::string error_detail;
  require(simaai::neat::pipeline_internal::sima::tensorsemantics::build_tiled_tensor_desc(
              shape, tile_shape, dtype, layout, tile_align_bytes, out, &error_detail,
              "test_tensor_output_missing", "test_tensor_shape_rank_invalid",
              "test_tensor_shape_dim_invalid", "test_tensor_dtype_invalid",
              "test_tile_shape_rank_mismatch", "test_tile_shape_dim_invalid"),
          expectation);
}

simaai::neat::PreprocOptions make_preproc_options() {
  simaai::neat::PreprocOptions opt;
  opt.model_managed_contract = true;
  opt.set_input_shape({720, 1280, 3});
  opt.input_img_type = "RGB";
  opt.set_output_shape({640, 640, 3});
  opt.scaled_width = 640;
  opt.scaled_height = 640;
  opt.output_img_type = "RGB";
  opt.output_dtype = "EVXX_INT8";
  opt.normalize = true;
  opt.aspect_ratio = true;
  opt.tessellate = true;
  opt.single_output_handoff = true;
  opt.set_slice_shape({32, 128, 3});
  opt.q_scale = 0.25;
  opt.q_zp = 7;
  return opt;
}

std::uint64_t dtype_size_bytes_for_test(const std::string& dtype) {
  std::string upper = dtype;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (upper == "BF16" || upper == "FP16" || upper == "UINT16" || upper == "INT16" ||
      upper == "EVXX_BFLOAT16" || upper == "EVXX_FLOAT16") {
    return 2U;
  }
  if (upper == "FP32" || upper == "FLOAT32" || upper == "INT32" || upper == "UINT32" ||
      upper == "EVXX_FLOAT32") {
    return 4U;
  }
  return 1U;
}

std::uint64_t packed_tensor_bytes_for_test(const std::vector<std::int64_t>& shape,
                                           const std::string& dtype) {
  std::uint64_t elems = 1U;
  for (const auto dim : shape) {
    require(dim > 0, "test shape must be strictly positive");
    elems *= static_cast<std::uint64_t>(dim);
  }
  return elems * dtype_size_bytes_for_test(dtype);
}

int trailing_width_for_shape(const std::vector<std::int64_t>& shape) {
  require(shape.size() >= 2U, "test shape must have width");
  return static_cast<int>(shape[shape.size() - 2U]);
}

int trailing_height_for_shape(const std::vector<std::int64_t>& shape) {
  require(shape.size() >= 3U, "test shape must have height");
  return static_cast<int>(shape[shape.size() - 3U]);
}

int trailing_channels_for_shape(const std::vector<std::int64_t>& shape) {
  require(!shape.empty(), "test shape must have channels");
  return static_cast<int>(shape.back());
}

int leading_depth_for_shape(const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> normalized = shape;
  if (normalized.size() >= 4U && normalized.front() == 1) {
    normalized.erase(normalized.begin());
  }
  if (normalized.size() <= 3U) {
    return 1;
  }
  std::uint64_t total = 1U;
  for (std::size_t i = 0; i + 3U < normalized.size(); ++i) {
    total *= static_cast<std::uint64_t>(normalized[i]);
  }
  return static_cast<int>(total);
}

std::vector<std::int64_t> tensor_desc_shape_for_test(const sima_ev_tensor_desc& desc) {
  std::vector<std::int64_t> out;
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  out.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    out.push_back(desc.shape.sizes[i]);
  }
  return out;
}

std::vector<std::int64_t> tensor_desc_tile_shape_for_test(const sima_ev_tensor_desc& desc) {
  std::vector<std::int64_t> out;
  if (desc.layout_kind != SIMA_EV_LAYOUT_TILED) {
    return out;
  }
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  out.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    out.push_back(desc.layout.tiled.tile_sizes[i]);
  }
  return out;
}

std::vector<std::int64_t>
detess_transport_shape_for_frame(const std::vector<std::int64_t>& frame_shape) {
  require(frame_shape.size() >= 3U, "detess test frame_shape must include HWC geometry");
  const int depth = leading_depth_for_shape(frame_shape);
  const int height = trailing_height_for_shape(frame_shape);
  const int width = trailing_width_for_shape(frame_shape);
  const int logical_channels = trailing_channels_for_shape(frame_shape);
  const int channels = ((logical_channels + 15) / 16) * 16;

  std::vector<std::int64_t> shape;
  if (depth > 1) {
    shape.push_back(static_cast<std::int64_t>(depth));
  }
  shape.push_back(static_cast<std::int64_t>(height));
  shape.push_back(static_cast<std::int64_t>(width));
  shape.push_back(static_cast<std::int64_t>(channels));
  return shape;
}

simaai::neat::pipeline_internal::sima::MpkTensorContract
make_test_tensor(const std::string& name, const std::string& dtype,
                 const std::vector<std::int64_t>& shape, std::uint64_t size_bytes = 0U) {
  using simaai::neat::pipeline_internal::sima::MpkShapeSemantics;
  using simaai::neat::pipeline_internal::sima::MpkTensorContract;

  MpkTensorContract tensor;
  tensor.name = name;
  tensor.segment_name = name;
  tensor.dtype = dtype;
  tensor.logical_dtype = dtype;
  tensor.logical_shape = shape;
  tensor.mpk_shape = shape;
  tensor.shape_semantics = MpkShapeSemantics::Geometry;
  tensor.size_bytes = size_bytes > 0U
                          ? static_cast<std::size_t>(size_bytes)
                          : static_cast<std::size_t>(packed_tensor_bytes_for_test(shape, dtype));
  return tensor;
}

simaai::neat::pipeline_internal::sima::MpkTensorContract
make_test_tensor_without_mpk_shape(const std::string& name, const std::string& dtype,
                                   const std::vector<std::int64_t>& logical_shape,
                                   std::uint64_t size_bytes = 0U) {
  using simaai::neat::pipeline_internal::sima::MpkShapeSemantics;
  using simaai::neat::pipeline_internal::sima::MpkTensorContract;

  MpkTensorContract tensor;
  tensor.name = name;
  tensor.segment_name = name;
  tensor.dtype = dtype;
  tensor.logical_dtype = dtype;
  tensor.logical_shape = logical_shape;
  tensor.shape_semantics = MpkShapeSemantics::Geometry;
  tensor.size_bytes =
      size_bytes > 0U
          ? static_cast<std::size_t>(size_bytes)
          : static_cast<std::size_t>(packed_tensor_bytes_for_test(logical_shape, dtype));
  return tensor;
}

simaai::neat::pipeline_internal::sima::MpkContract
make_rank_aware_detess_contract(const std::vector<std::int64_t>& frame_shape,
                                const std::string& input_dtype, const std::string& output_dtype) {
  using simaai::neat::pipeline_internal::sima::MpkContract;
  using simaai::neat::pipeline_internal::sima::MpkContractEdge;
  using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;

  const std::vector<std::int64_t> transport_shape = detess_transport_shape_for_frame(frame_shape);
  const std::uint64_t packed_bytes = packed_tensor_bytes_for_test(transport_shape, input_dtype);
  const bool align_c16 =
      trailing_channels_for_shape(frame_shape) != static_cast<int>(transport_shape.back());

  MpkPluginIoContract mla;
  mla.name = "MLA_0";
  mla.processor = "mla";
  mla.kernel = "mla";
  mla.sequence = 0;
  mla.output_tensors = {
      make_test_tensor("mla_parent_0", input_dtype, transport_shape, packed_bytes)};

  MpkPluginIoContract unpack;
  unpack.name = "unpack_0";
  unpack.kernel = "unpack";
  unpack.sequence = 1;
  unpack.output_tensors = {make_test_tensor("ofm0", input_dtype, transport_shape, packed_bytes)};

  MpkPluginIoContract detess;
  detess.name = "detess_0";
  detess.kernel = "detess";
  detess.sequence = 2;
  detess.frame_shape = frame_shape;
  detess.frame_type = input_dtype;
  detess.slice_shape = {static_cast<std::int64_t>(trailing_height_for_shape(frame_shape)),
                        static_cast<std::int64_t>(trailing_width_for_shape(frame_shape)),
                        static_cast<std::int64_t>(trailing_channels_for_shape(frame_shape))};
  detess.has_align_c16 = align_c16;
  detess.align_c16 = align_c16;
  detess.canonical_input_dtype = input_dtype;
  detess.canonical_output_dtype = output_dtype;
  detess.input_tensors = {make_test_tensor("ofm0", input_dtype, transport_shape, packed_bytes)};
  detess.output_tensors = {make_test_tensor("head0", output_dtype, frame_shape)};

  MpkContract contract;
  contract.plugins = {mla, unpack, detess};
  contract.edges = {
      MpkContractEdge{0U, 0, 1U, 0, "MLA_0", "unpack_0", "mla_parent_0"},
      MpkContractEdge{1U, 0, 2U, 0, "unpack_0", "detess_0", "ofm0"},
  };
  return contract;
}

simaai::neat::pipeline_internal::sima::MpkContract
make_pre_and_post_cast_contract_for_exact_name_regression() {
  using simaai::neat::pipeline_internal::sima::MpkContract;
  using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;

  MpkPluginIoContract pre_cast0;
  pre_cast0.name = "cast_0";
  pre_cast0.processor = "EV74";
  pre_cast0.kernel = "cast_transform";
  pre_cast0.sequence = 0;
  pre_cast0.canonical_input_dtype = "FP32";
  pre_cast0.canonical_output_dtype = "BF16";
  pre_cast0.input_tensors = {make_test_tensor("image_l", "FP32", {384, 1664, 1})};
  pre_cast0.output_tensors = {make_test_tensor("cast_0", "BF16", {384, 1664, 1})};

  MpkPluginIoContract pre_cast1;
  pre_cast1.name = "cast_1";
  pre_cast1.processor = "EV74";
  pre_cast1.kernel = "cast_transform";
  pre_cast1.sequence = 1;
  pre_cast1.canonical_input_dtype = "FP32";
  pre_cast1.canonical_output_dtype = "BF16";
  pre_cast1.input_tensors = {make_test_tensor("image_uv", "FP32", {192, 832, 2})};
  pre_cast1.output_tensors = {make_test_tensor("cast_1", "BF16", {192, 832, 2})};

  MpkPluginIoContract mla;
  mla.name = "MLA_0";
  mla.processor = "MLA";
  mla.kernel = "infer";
  mla.sequence = 2;
  mla.input_tensors = {
      make_test_tensor("cast_0", "BF16", {384, 1664, 1}),
      make_test_tensor("cast_1", "BF16", {192, 832, 2}),
  };
  mla.output_tensors = {
      make_test_tensor("ofm0", "BF16", {12, 52, 45}),
      make_test_tensor("ofm1", "BF16", {3, 13, 4}),
  };

  MpkPluginIoContract post_cast0;
  post_cast0.name = "cast_3";
  post_cast0.processor = "EV74";
  post_cast0.kernel = "cast_transform";
  post_cast0.sequence = 3;
  post_cast0.canonical_input_dtype = "BF16";
  post_cast0.canonical_output_dtype = "FP32";
  post_cast0.input_tensors = {make_test_tensor("ofm0", "BF16", {12, 52, 45})};
  post_cast0.output_tensors = {make_test_tensor("head0", "FP32", {12, 52, 45})};

  MpkPluginIoContract post_cast1;
  post_cast1.name = "cast_4";
  post_cast1.processor = "EV74";
  post_cast1.kernel = "cast_transform";
  post_cast1.sequence = 4;
  post_cast1.canonical_input_dtype = "BF16";
  post_cast1.canonical_output_dtype = "FP32";
  post_cast1.input_tensors = {make_test_tensor("ofm1", "BF16", {3, 13, 4})};
  post_cast1.output_tensors = {make_test_tensor("head1", "FP32", {3, 13, 4})};

  MpkContract contract;
  contract.plugins = {pre_cast0, pre_cast1, mla, post_cast0, post_cast1};
  return contract;
}

simaai::neat::pipeline_internal::sima::MpkContract
make_dequant_second_mla_output_contract_for_routing_regression() {
  using simaai::neat::pipeline_internal::sima::MpkContract;
  using simaai::neat::pipeline_internal::sima::MpkContractEdge;
  using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;
  using simaai::neat::pipeline_internal::sima::MpkQuantContract;

  MpkPluginIoContract mla;
  mla.name = "MLA_0";
  mla.processor = "MLA";
  mla.kernel = "mla";
  mla.sequence = 0;
  mla.output_tensors = {
      make_test_tensor("ofm0", "INT8", {1, 1, 4}, 4U),
      make_test_tensor("ofm1", "INT8", {1, 1, 8}, 8U),
  };

  MpkPluginIoContract dequant;
  dequant.name = "dequantize_1";
  dequant.processor = "EV74";
  dequant.kernel = "dequantization_transform";
  dequant.sequence = 1;
  dequant.canonical_input_dtype = "INT8";
  dequant.canonical_output_dtype = "FP32";
  dequant.input_tensors = {make_test_tensor("ofm1", "INT8", {1, 1, 8}, 8U)};
  dequant.output_tensors = {make_test_tensor("head1", "FP32", {1, 1, 8}, 32U)};
  dequant.quant = MpkQuantContract{{0.25}, {-3}, -1};

  MpkContract contract;
  contract.plugins = {mla, dequant};
  contract.edges = {
      MpkContractEdge{0U, 1, 1U, 0, "MLA_0", "dequantize_1", "ofm1"},
  };
  return contract;
}

simaai::neat::pipeline_internal::sima::MpkContract
make_rank_aware_detessdequant_contract(const std::vector<std::int64_t>& frame_shape,
                                       const std::string& input_dtype,
                                       const std::string& output_dtype) {
  using simaai::neat::pipeline_internal::sima::MpkContract;
  using simaai::neat::pipeline_internal::sima::MpkContractEdge;
  using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;
  using simaai::neat::pipeline_internal::sima::MpkQuantContract;

  const std::vector<std::int64_t> transport_shape = detess_transport_shape_for_frame(frame_shape);
  const std::uint64_t packed_bytes = packed_tensor_bytes_for_test(transport_shape, input_dtype);
  const bool align_c16 =
      trailing_channels_for_shape(frame_shape) != static_cast<int>(transport_shape.back());

  MpkPluginIoContract mla;
  mla.name = "MLA_0";
  mla.processor = "mla";
  mla.kernel = "mla";
  mla.sequence = 0;
  mla.output_tensors = {
      make_test_tensor("mla_parent_0", input_dtype, transport_shape, packed_bytes)};

  MpkPluginIoContract unpack;
  unpack.name = "unpack_0";
  unpack.kernel = "unpack";
  unpack.sequence = 1;
  unpack.output_tensors = {make_test_tensor("ofm0", input_dtype, transport_shape, packed_bytes)};

  MpkPluginIoContract detess;
  detess.name = "detess_0";
  detess.kernel = "detess";
  detess.sequence = 2;
  detess.frame_shape = frame_shape;
  detess.frame_type = input_dtype;
  detess.slice_shape = {static_cast<std::int64_t>(trailing_height_for_shape(frame_shape)),
                        static_cast<std::int64_t>(trailing_width_for_shape(frame_shape)),
                        static_cast<std::int64_t>(trailing_channels_for_shape(frame_shape))};
  detess.has_align_c16 = align_c16;
  detess.align_c16 = align_c16;
  detess.canonical_input_dtype = input_dtype;
  detess.canonical_output_dtype = input_dtype;
  detess.input_tensors = {make_test_tensor("ofm0", input_dtype, transport_shape, packed_bytes)};
  detess.output_tensors = {
      make_test_tensor("detess_0_out", input_dtype, frame_shape, packed_bytes)};

  MpkPluginIoContract dequant;
  dequant.name = "dequant_0";
  dequant.kernel = "dequant";
  dequant.sequence = 3;
  dequant.canonical_input_dtype = input_dtype;
  dequant.canonical_output_dtype = output_dtype;
  dequant.input_tensors = {
      make_test_tensor("detess_0_out", input_dtype, frame_shape, packed_bytes)};
  dequant.output_tensors = {make_test_tensor("head0_fp32", output_dtype, frame_shape)};
  dequant.quant = MpkQuantContract{{0.5}, {7}, -1};

  MpkContract contract;
  contract.plugins = {mla, unpack, detess, dequant};
  contract.edges = {
      MpkContractEdge{0U, 0, 1U, 0, "MLA_0", "unpack_0", "mla_parent_0"},
      MpkContractEdge{1U, 0, 2U, 0, "unpack_0", "detess_0", "ofm0"},
      MpkContractEdge{2U, 0, 3U, 0, "detess_0", "dequant_0", "detess_0_out"},
  };
  return contract;
}

simaai::neat::pipeline_internal::sima::MpkContract make_resnet_flatten_detessdequant_contract() {
  using simaai::neat::pipeline_internal::sima::MpkContract;
  using simaai::neat::pipeline_internal::sima::MpkContractEdge;
  using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;
  using simaai::neat::pipeline_internal::sima::MpkQuantContract;

  const std::vector<std::int64_t> frame_shape = {1, 1, 1, 1000};
  const std::vector<std::int64_t> transport_shape = {1, 1, 1008};
  const std::vector<std::int64_t> flattened_shape = {1, 1000};
  constexpr std::uint64_t kTransportBytes = 1008U;
  constexpr std::uint64_t kInt8Bytes = 1000U;
  constexpr std::uint64_t kFp32Bytes = 4000U;

  MpkPluginIoContract mla;
  mla.name = "MLA_0";
  mla.processor = "mla";
  mla.kernel = "mla";
  mla.sequence = 0;
  mla.output_tensors = {make_test_tensor("MLA_0", "INT8", transport_shape, kTransportBytes)};

  MpkPluginIoContract detess;
  detess.name = "detessellate_MLA_0_detessellation_transform";
  detess.kernel = "detessellation_transform";
  detess.processor = "EV74";
  detess.sequence = 1;
  detess.frame_shape = frame_shape;
  detess.frame_type = "INT8";
  detess.slice_shape = {1, 1, 1000};
  detess.has_align_c16 = true;
  detess.align_c16 = true;
  detess.has_cblock = true;
  detess.cblock = true;
  detess.canonical_input_dtype = "INT8";
  detess.canonical_output_dtype = "INT8";
  detess.input_tensors = {make_test_tensor("MLA_0", "INT8", transport_shape, kTransportBytes)};
  detess.output_tensors = {make_test_tensor("detessellate_MLA_0_detessellation_transform", "INT8",
                                            frame_shape, kInt8Bytes)};

  MpkPluginIoContract flatten;
  flatten.name = "EV_1/batch_flatten_0";
  flatten.kernel = "batch_flatten_transform";
  flatten.processor = "EV74";
  flatten.sequence = 2;
  flatten.canonical_input_dtype = "INT8";
  flatten.canonical_output_dtype = "INT8";
  flatten.input_tensors = {make_test_tensor("detessellate_MLA_0_detessellation_transform", "INT8",
                                            frame_shape, kInt8Bytes)};
  flatten.output_tensors = {
      make_test_tensor("EV_1/batch_flatten_0", "INT8", flattened_shape, kInt8Bytes)};

  MpkPluginIoContract dequant;
  dequant.name = "dequantize_1";
  dequant.kernel = "dequantization_transform";
  dequant.processor = "EV74";
  dequant.sequence = 3;
  dequant.canonical_input_dtype = "INT8";
  dequant.canonical_output_dtype = "FP32";
  dequant.input_tensors = {
      make_test_tensor("EV_1/batch_flatten_0", "INT8", flattened_shape, kInt8Bytes)};
  dequant.output_tensors = {
      make_test_tensor("dequantize_1/resnetv17_dense0_fwd", "FP32", flattened_shape, kFp32Bytes)};
  dequant.quant = MpkQuantContract{{0.5}, {7}, -1};

  MpkContract contract;
  contract.plugins = {mla, detess, flatten, dequant};
  contract.edges = {
      MpkContractEdge{0U, 0, 1U, 0, "MLA_0", "detessellate_MLA_0_detessellation_transform",
                      "MLA_0"},
      MpkContractEdge{1U, 0, 2U, 0, "detessellate_MLA_0_detessellation_transform",
                      "EV_1/batch_flatten_0", "detessellate_MLA_0_detessellation_transform"},
      MpkContractEdge{2U, 0, 3U, 0, "EV_1/batch_flatten_0", "dequantize_1", "EV_1/batch_flatten_0"},
  };
  return contract;
}

simaai::neat::pipeline_internal::sima::MpkContract make_rank_aware_tess_preadapter_contract(
    const std::vector<std::int64_t>& canonical_shape, const std::vector<std::int64_t>& slice_shape,
    const std::string& cast_input_dtype, const std::string& tess_frame_dtype) {
  using simaai::neat::pipeline_internal::sima::MpkContract;
  using simaai::neat::pipeline_internal::sima::MpkContractEdge;
  using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;

  require(canonical_shape.size() == 4U, "tess preadapter test requires canonical NHWC shape");

  MpkPluginIoContract cast;
  cast.name = "cast_0";
  cast.kernel = "cast_transform";
  cast.processor = "EV74";
  cast.sequence = 1;
  cast.batch_size = 1;
  cast.batch_sz_model = 1;
  cast.canonical_input_dtype = cast_input_dtype;
  cast.canonical_output_dtype = tess_frame_dtype;
  cast.input_tensors = {make_test_tensor("image_0", cast_input_dtype, canonical_shape)};
  cast.output_tensors = {make_test_tensor("cast_0_out", tess_frame_dtype, canonical_shape)};

  const std::uint64_t packed_bytes =
      packed_tensor_bytes_for_test(canonical_shape, tess_frame_dtype);
  MpkPluginIoContract tess;
  tess.name = "tess_0";
  tess.kernel = "tessellation_transform";
  tess.processor = "EV74";
  tess.sequence = 2;
  tess.batch_size = 1;
  tess.batch_sz_model = 1;
  tess.frame_type = tess_frame_dtype;
  tess.slice_shape = slice_shape;
  tess.has_align_c16 = true;
  tess.align_c16 = false;
  tess.has_cblock = true;
  tess.cblock = false;
  tess.input_tensors = {make_test_tensor("cast_0_out", tess_frame_dtype, canonical_shape)};
  tess.output_tensors = {
      make_test_tensor("tess_0_out", tess_frame_dtype, canonical_shape, packed_bytes)};

  MpkContract contract;
  contract.plugins = {cast, tess};
  contract.edges = {
      MpkContractEdge{0U, 0, 1U, 0, "cast_0", "tess_0", "cast_0_out"},
  };
  return contract;
}

void require_packed_input_projection(const simaai::neat::CompiledProcessCvuContract& compiled,
                                     int width, int height, int depth, int channels,
                                     const char* expectation) {
  require(!compiled.payload.input_shapes.empty(), expectation);
  const auto& front = compiled.payload.input_shapes.front();
  // input_shapes stores each input as a shape vector; build the expected shape
  // using the convention [depth, height, width, channels] (D, H, W, C) with
  // depth omitted when it equals 1.
  std::vector<int> expected;
  if (depth > 1) {
    expected.push_back(depth);
  }
  expected.push_back(height);
  expected.push_back(width);
  expected.push_back(channels);
  require(front == expected, expectation);
}

void require_runtime_packed_logical_input_shape(
    const simaai::neat::CompiledProcessCvuContract& compiled, int width, int height, int depth,
    int channels, const char* expectation) {
  require(!compiled.runtime_contract.logical_inputs.empty(), expectation);
  std::vector<std::int64_t> expected;
  if (depth > 1) {
    expected.push_back(depth);
  }
  expected.push_back(height);
  expected.push_back(width);
  expected.push_back(channels);
  require(compiled.runtime_contract.logical_inputs.front().shape == expected, expectation);
}

} // namespace

RUN_TEST(
    "unit_contract_compiler_processcvu_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

      {
        GriderFastOptions fast_opt;
        fast_opt.width = 640;
        fast_opt.height = 480;
        fast_opt.max_features = 256;
        fast_opt.grid_x = 8;
        fast_opt.grid_y = 6;
        GriderFast fast_node(fast_opt);
        ContractCompileInput cc_input;
        cc_input.node_index = 4;
        CompiledNodeContract compiled;
        std::string err;
        require(fast_node.compile_node_contract(cc_input, &compiled, &err),
                (std::string("GriderFast node contract should compile: ") + err).c_str());
        require(compiled.processcvu.has_value(),
                "GriderFast node contract should carry a processcvu payload");
        require(compiled.processcvu->payload.graph_id == 236,
                "GriderFast payload should use graph ID 236");
        require(compiled.processcvu->payload.graph_name == "grider_fast",
                "GriderFast payload should use canonical graph name");
        require(compiled.processcvu->payload.max_features == 256,
                "GriderFast payload should preserve max_features");
        require(compiled.processcvu->runtime_contract.physical_inputs.size() == 1U,
                "GriderFast should expose one runtime input");
        require(compiled.processcvu->runtime_contract.physical_outputs.size() == 1U,
                "GriderFast should expose one runtime output");
      }

      {
        TrackKLTOptions klt_opt;
        klt_opt.width = 640;
        klt_opt.height = 480;
        klt_opt.num_points = 128;
        klt_opt.max_features = 64;
        klt_opt.detect_new_features = 0;
        TrackKLT klt_node(klt_opt);
        ContractCompileInput cc_input;
        cc_input.node_index = 5;
        CompiledNodeContract compiled;
        std::string err;
        require(klt_node.compile_node_contract(cc_input, &compiled, &err),
                (std::string("TrackKLT node contract should compile: ") + err).c_str());
        require(compiled.processcvu.has_value(),
                "TrackKLT node contract should carry a processcvu payload");
        require(compiled.processcvu->payload.graph_id == 238,
                "TrackKLT payload should use graph ID 238");
        require(compiled.processcvu->payload.default_output_names.size() == 3U,
                "TrackKLT runtime should still allocate output_features for the EV ABI");
        require(compiled.processcvu->exposed_view.exposed_output_order.size() == 2U,
                "TrackKLT should publish only points/status when detect_new_features is disabled");
        require(compiled.processcvu->runtime_contract.physical_inputs.size() == 3U,
                "TrackKLT should preserve its three physical inputs");
        require(compiled.processcvu->runtime_contract.physical_outputs.size() == 3U,
                "TrackKLT should preserve its three physical runtime outputs");
      }

      {
        namespace ts = simaai::neat::pipeline_internal::sima::tensorsemantics;
        std::vector<int> normalized;
        std::string error_detail;
        require(ts::normalize_tile_shape({1, 300, 4}, {4, 4}, &normalized, &error_detail, "missing",
                                         "prefix", "dim"),
                "tile normalizer should accept shorter tile ranks");
        require(normalized == std::vector<int>({1, 4, 4}),
                "tile normalizer should left-pad shorter tile ranks with 1");

        normalized.clear();
        require(ts::normalize_tile_shape({1, 36, 36, 256}, {36, 4, 32}, &normalized, &error_detail,
                                         "missing", "prefix", "dim"),
                "tile normalizer should accept RF-DETR feature-map tile rank");
        require(normalized == std::vector<int>({1, 36, 4, 32}),
                "tile normalizer should left-pad RF-DETR feature-map tile rank with 1");

        normalized.clear();
        require(ts::normalize_tile_shape({300, 4}, {1, 4, 4}, &normalized, &error_detail, "missing",
                                         "prefix", "dim"),
                "tile normalizer should trim longer tile ranks with leading 1s");
        require(normalized == std::vector<int>({4, 4}),
                "tile normalizer should trim only the leading compatibility 1");

        normalized.clear();
        error_detail.clear();
        require(!ts::normalize_tile_shape({300, 4}, {2, 4, 4}, &normalized, &error_detail,
                                          "missing", "prefix", "dim"),
                "tile normalizer should reject non-1 leading trim prefixes");
        require(error_detail == "prefix",
                "tile normalizer should report the rank-prefix validation failure");

        normalized.clear();
        error_detail.clear();
        require(!ts::normalize_tile_shape({1, 300, 4}, {2, 4, 4}, &normalized, &error_detail,
                                          "missing", "prefix", "dim"),
                "tile normalizer should reject tile dims larger than tensor dims");
        require(error_detail == "dim",
                "tile normalizer should report the tile-dimension validation failure");

        normalized.clear();
        error_detail.clear();
        require(!ts::normalize_tile_shape({1, 300, 4}, {0, 4, 4}, &normalized, &error_detail,
                                          "missing", "prefix", "dim"),
                "tile normalizer should reject non-positive tile dims");
        require(error_detail == "dim",
                "tile normalizer should report non-positive tile dims as dimension failures");
      }

      {
        namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;

        pcs::CastContractSubset cast_subset;
        cast_subset.input_shape = {1, 300, 4};
        cast_subset.input_dtype = "FP32";
        cast_subset.output_dtype = "BF16";
        pcs::TessellateContractSubset tess_subset;
        tess_subset.input_shape = {1, 300, 4};
        tess_subset.frame_type = "BF16";
        tess_subset.slice_shape = {4, 4};
        tess_subset.output_size_bytes = 2400U;
        tess_subset.batch_size = 1;
        const auto tess_runtime = pcs::build_tessellate_runtime_config_from_subsets(
            cast_subset, tess_subset, "rfdetr_boxes_tess", "rfdetr_boxes_tess");
        require(tess_runtime.input_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(tess_runtime.input_tensors.front()) ==
                        std::vector<std::int64_t>({1, 300, 4}),
                "tessellate descriptor should preserve authored rank-3 input shape");
        require(tess_runtime.output_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(tess_runtime.output_tensors.front()) ==
                        std::vector<std::int64_t>({1, 300, 4}),
                "tessellate descriptor should preserve authored rank-3 output shape");
        require(tensor_desc_tile_shape_for_test(tess_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 4, 4}),
                "tessellate descriptor should normalize [4,4] to [1,4,4]");

        pcs::QuantTessContractSubset quanttess_subset;
        quanttess_subset.quant_params.scales = {0.25};
        quanttess_subset.quant_params.zero_points = {-7};
        quanttess_subset.input_shape = {1, 36, 36, 256};
        quanttess_subset.output_shape = {1, 36, 36, 256};
        quanttess_subset.input_dtype = "FP32";
        quanttess_subset.output_dtype = "INT8";
        quanttess_subset.round_off = 0;
        quanttess_subset.slice_shape = {36, 4, 32};
        quanttess_subset.frame_type = "INT8";
        quanttess_subset.output_size_bytes = 663552U;
        quanttess_subset.batch_size = 1;
        const auto quanttess_runtime =
            pcs::build_quanttess_runtime_config_from_subset(quanttess_subset, "rfdetr_feat_tess");
        require(quanttess_runtime.input_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(quanttess_runtime.input_tensors.front()) ==
                        std::vector<std::int64_t>({1, 36, 36, 256}),
                "quanttess descriptor should preserve authored rank-4 input shape");
        require(quanttess_runtime.output_tensors.size() == 1U &&
                    tensor_desc_shape_for_test(quanttess_runtime.output_tensors.front()) ==
                        std::vector<std::int64_t>({1, 36, 36, 256}),
                "quanttess descriptor should preserve authored rank-4 output shape");
        require(tensor_desc_tile_shape_for_test(quanttess_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 36, 4, 32}),
                "quanttess descriptor should normalize [36,4,32] to [1,36,4,32]");

        pcs::QuantizeContractSubset quantize_subset;
        quantize_subset.quant_params.scales = {0.25};
        quantize_subset.quant_params.zero_points = {-7};
        quantize_subset.input_shape = {1, 300, 4};
        quantize_subset.input_dtype = "FP32";
        quantize_subset.output_dtype = "INT8";
        quantize_subset.round_off = 0;
        const auto quantize_runtime = pcs::build_quantize_runtime_config_from_subset(
            quantize_subset, "quantized", "quantized");
        require(tensor_desc_shape_for_test(quantize_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "quantize dense descriptor should preserve authored input shape");
        require(tensor_desc_shape_for_test(quantize_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "quantize dense descriptor should preserve authored output shape");

        pcs::DequantizeContractSubset dequantize_subset;
        dequantize_subset.quant_params.scales = {0.25};
        dequantize_subset.quant_params.zero_points = {-7};
        dequantize_subset.input_shape = {1, 300, 4};
        dequantize_subset.output_shape = {1, 300, 4};
        dequantize_subset.input_dtype = "INT8";
        dequantize_subset.output_dtype = "FP32";
        const auto dequantize_runtime =
            pcs::build_dequantize_runtime_config_from_subsets({dequantize_subset}, {"dequantized"});
        require(tensor_desc_shape_for_test(dequantize_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "dequantize dense descriptor should preserve authored input shape");
        require(tensor_desc_shape_for_test(dequantize_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "dequantize dense descriptor should preserve authored output shape");

        pcs::DetessellateContractSubset detess_subset;
        detess_subset.frame_shape = {1, 300, 4};
        detess_subset.input_transport_shape = {1, 300, 4};
        detess_subset.input_transport_size_bytes =
            packed_tensor_bytes_for_test({1, 300, 4}, "INT8");
        detess_subset.frame_type = "INT8";
        detess_subset.slice_shape = {4, 4};
        const auto detess_runtime = pcs::build_detessellate_runtime_config_from_subsets(
            {detess_subset}, {"detess"}, {"detess"});
        require(tensor_desc_shape_for_test(detess_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "detessellate tiled descriptor should preserve authored frame shape");
        require(tensor_desc_tile_shape_for_test(detess_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 4, 4}),
                "detessellate tiled descriptor should normalize shorter tile shape");
        require(tensor_desc_shape_for_test(detess_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 300, 4}),
                "detessellate dense output descriptor should preserve authored frame shape");

        pcs::DetessDequantHeadContractSubset head;
        head.per_head_input_shape = {300, 4};
        head.input_transport_shape = {300, 4};
        head.input_transport_size_bytes = packed_tensor_bytes_for_test({300, 4}, "INT8");
        head.per_head_quant_params.scales = {0.25};
        head.per_head_quant_params.zero_points = {-7};
        head.frame_shape = {300, 4};
        head.frame_type = "INT8";
        head.slice_shape = {4, 4};
        head.output_dtype = "FP32";
        pcs::DetessDequantContractSubset detessdequant_subset;
        detessdequant_subset.heads = {head};
        const auto detessdequant_runtime = pcs::build_detessdequant_runtime_config_from_subset(
            detessdequant_subset, {"detessdq"}, {"detessdq"});
        require(tensor_desc_shape_for_test(detessdequant_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({300, 4}),
                "detessdequant tiled descriptor should preserve current authored frame shape");
        require(tensor_desc_tile_shape_for_test(detessdequant_runtime.input_tensors.front()) ==
                    std::vector<std::int64_t>({4, 4}),
                "detessdequant tiled descriptor should preserve matching tile shape");
        require(tensor_desc_shape_for_test(detessdequant_runtime.output_tensors.front()) ==
                    std::vector<std::int64_t>({300, 4}),
                "detessdequant dense output descriptor should preserve authored frame shape");
      }

      auto node = nodes::Preproc(make_preproc_options());
      std::vector<std::shared_ptr<Node>> nodes_to_compile = {node};
      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      ContractCompileInput input;
      input.pipeline_label = "unit_contract_compiler_processcvu_test";

      const CompiledPipelineContracts compiled =
          compile_node_contracts(nodes_to_compile, input, &diagnostics);

      require(diagnostics.errors.empty(), "compile_node_contracts should not emit errors");
      require(compiled.fully_renderable, "compiled contract set should be renderable");
      require(compiled.stages.size() == 1U, "compiled contract should contain one stage");
      require(compiled.stages.front().processcvu.has_value(),
              "preproc should compile to a processcvu contract");

      const auto& processcvu = *compiled.stages.front().processcvu;
      require(processcvu.runtime_contract.logical_inputs.size() == 1U,
              "preproc runtime contract should expose one logical input");
      // Runtime contract publishes both rgb and tess outputs; the exposed view
      // narrows to the selected tessellated handoff.
      const auto tess_runtime_it =
          std::find_if(processcvu.runtime_contract.logical_outputs.begin(),
                       processcvu.runtime_contract.logical_outputs.end(), [](const auto& logical) {
                         return logical.logical_name == "output_tessellated_image";
                       });
      require(tess_runtime_it != processcvu.runtime_contract.logical_outputs.end(),
              "preproc runtime contract should preserve the tessellated handoff name");
      require(tess_runtime_it->shape == std::vector<std::int64_t>({640, 640, 3}),
              "preproc tessellated handoff should preserve the semantic output shape");
      require(tess_runtime_it->size_bytes == 640U * 640U * 3U,
              "preproc tessellated handoff should preserve the packed MLA ingress byte size");
      {
        std::vector<std::shared_ptr<Node>> indexed_nodes = {
            nodes::Input(InputOptions{}),
            nodes::Preproc(make_preproc_options()),
            nodes::Output(OutputOptions{}),
        };
        pipeline_internal::sima::ManifestBuildDiagnostics indexed_diagnostics;
        const CompiledPipelineContracts indexed =
            compile_node_contracts(indexed_nodes, ContractCompileInput{}, &indexed_diagnostics);
        require(indexed_diagnostics.errors.empty(),
                "indexed compile_node_contracts should not emit errors");
        require(indexed.stages.size() == 1U, "indexed compile should contain one semantic stage");
        require(indexed.stages.front().element_name == "n1_preproc",
                "standalone preproc contract should use actual node index for element identity");
        require(indexed.stages.front().logical_stage_id == "n1_preproc",
                "standalone preproc contract should use actual node index for logical stage id");
      }
      require(processcvu.exposed_view.exposed_logical_outputs.size() == 1U,
              "preproc exposed view should expose one selected output");
      require(processcvu.preproc_single_output_handoff,
              "preproc single-output handoff flag should be preserved");

      const auto direct_inputs =
          build_processcvu_compile_inputs_from_options(make_preproc_options());
      require(direct_inputs.payload.primary_output_name == direct_inputs.facts.primary_output_name,
              "direct preproc compile inputs should preserve primary output selection");
      require(!direct_inputs.facts.published_output_names.empty() &&
                  direct_inputs.facts.published_output_names.front() ==
                      direct_inputs.payload.primary_output_name,
              "direct preproc compile inputs should preserve the selected published output");
      require(!direct_inputs.payload.default_output_names.empty() &&
                  std::find(direct_inputs.payload.default_output_names.begin(),
                            direct_inputs.payload.default_output_names.end(),
                            std::string("output_tessellated_image")) !=
                      direct_inputs.payload.default_output_names.end(),
              "direct preproc compile inputs should publish the tessellated runtime output");
      require(direct_inputs.facts.inputs.size() == 1U,
              "direct preproc compile inputs should preserve input count");
      require(direct_inputs.facts.inputs.front().shape == std::vector<std::int64_t>({720, 1280, 3}),
              "direct preproc compile inputs should preserve the authored single input shape");
      require(direct_inputs.payload.input_tensors.size() == 1U &&
                  simaai::neat::pipeline_internal::sima::tensorsemantics::
                          layout_token_from_ev_tensor_desc(
                              direct_inputs.payload.input_tensors.front()) == "HWC",
              "direct preproc compile inputs should synthesize semantic typed input descriptor");
      const auto direct_tess_it = std::find_if(
          direct_inputs.facts.outputs.begin(), direct_inputs.facts.outputs.end(),
          [](const auto& output) { return output.logical_name == "output_tessellated_image"; });
      require(direct_tess_it != direct_inputs.facts.outputs.end(),
              "direct preproc compile inputs should preserve the tessellated runtime output");
      require(direct_tess_it->representation == ProcessCvuOutputRepresentation::PackedBlob,
              "direct preproc compile inputs should keep packed MLA handoff semantics");
      require(direct_tess_it->size_bytes == 640U * 640U * 3U,
              "direct preproc compile inputs should preserve packed MLA ingress byte size");
      require(!direct_inputs.payload.output_tensors.empty() &&
                  std::any_of(direct_inputs.payload.output_tensors.begin(),
                              direct_inputs.payload.output_tensors.end(),
                              [](const auto& t) { return t.layout_kind == SIMA_EV_LAYOUT_TILED; }),
              "direct preproc compile inputs should synthesize tiled typed output descriptor");
      {
        const auto tiled_it =
            std::find_if(direct_inputs.payload.output_tensors.begin(),
                         direct_inputs.payload.output_tensors.end(),
                         [](const auto& t) { return t.layout_kind == SIMA_EV_LAYOUT_TILED; });
        require(tiled_it != direct_inputs.payload.output_tensors.end(),
                "direct preproc compile inputs should include a tiled output descriptor");
        require(sima_ev_tensor_tile_axis_size(&*tiled_it, SIMA_EV_AXIS_H, 0) == 32 &&
                    sima_ev_tensor_tile_axis_size(&*tiled_it, SIMA_EV_AXIS_W, 0) == 128 &&
                    sima_ev_tensor_tile_axis_size(&*tiled_it, SIMA_EV_AXIS_C, 0) == 3,
                "direct preproc compile inputs should preserve typed output tile geometry");
      }
      {
        auto invalid_preproc = direct_inputs.payload;
        invalid_preproc.input_shapes.push_back({1, 1, 1});
        invalid_preproc.num_in_tensor = 2;
        bool threw = false;
        try {
          (void)build_preproc_facts_from_payload_internal(invalid_preproc);
        } catch (const std::invalid_argument& ex) {
          threw = std::string(ex.what()) == "processcvu preproc payload requires exactly one input";
        }
        require(threw, "direct preproc facts builder should reject multi-input preproc payloads");
      }

      const auto direct_compiled = build_processcvu_compiled_contract(direct_inputs);
      require(direct_compiled.runtime_contract.logical_outputs.size() ==
                  processcvu.runtime_contract.logical_outputs.size(),
              "shared processcvu compiled-contract adapter should preserve runtime logical output "
              "count");
      const auto direct_compiled_tess_it = std::find_if(
          direct_compiled.runtime_contract.logical_outputs.begin(),
          direct_compiled.runtime_contract.logical_outputs.end(),
          [](const auto& logical) { return logical.logical_name == "output_tessellated_image"; });
      require(
          direct_compiled_tess_it != direct_compiled.runtime_contract.logical_outputs.end(),
          "shared processcvu compiled-contract adapter should preserve tessellated runtime output");
      require(direct_compiled_tess_it->size_bytes == tess_runtime_it->size_bytes,
              "shared processcvu compiled-contract adapter should preserve packed handoff bytes");

      {
        auto unsupported = make_preproc_options();
        unsupported.single_output_handoff = false;
        bool ctor_threw = false;
        try {
          simaai::neat::Preproc unsupported_node(unsupported);
          (void)unsupported_node;
        } catch (const std::runtime_error& ex) {
          ctor_threw = std::string(ex.what()) == "Preproc dual-output contract is currently "
                                                 "unsupported; use single_output_handoff=true.";
        }
        require(ctor_threw, "preproc construction should reject unsupported dual-output handoff");

        bool threw = false;
        try {
          (void)build_processcvu_compile_inputs_from_options(unsupported);
        } catch (const std::invalid_argument& ex) {
          threw = std::string(ex.what()) == "Preproc dual-output contract is currently "
                                            "unsupported; use single_output_handoff=true.";
        }
        require(threw,
                "processcvu preproc compile inputs should reject unsupported dual-output handoff");
      }

      simaai::neat::pipeline_internal::sima::stagesemantics::CompiledProcessCvuRuntimeConfig
          packed_tess_runtime;
      packed_tess_runtime.graph_family = "tessellate";
      packed_tess_runtime.graph_name = "tessellate";
      packed_tess_runtime.graph_id = 2;
      packed_tess_runtime.default_input_name = "input_tensor";
      packed_tess_runtime.runtime_input_names = {"input_tensor"};
      packed_tess_runtime.runtime_output_names = {"output_tensor"};
      packed_tess_runtime.published_output_names = {"output_tensor"};
      packed_tess_runtime.physical_input_names = {"input_tensor"};
      packed_tess_runtime.physical_output_names = {"output_tensor"};
      packed_tess_runtime.primary_output_name = "output_tensor";
      packed_tess_runtime.input_shapes = {{192, 832, 2}};
      packed_tess_runtime.output_shapes = {{192, 832, 2}};
      packed_tess_runtime.slice_shapes = {{42, 39, 2}};
      packed_tess_runtime.input_dtype = "BF16";
      packed_tess_runtime.output_dtype = "BF16";
      packed_tess_runtime.out_dtype = "BF16";
      packed_tess_runtime.runtime_output_logical_shapes = {{192, 832, 2}};
      packed_tess_runtime.runtime_output_logical_index_list = {0};
      packed_tess_runtime.runtime_output_output_slot_list = {0};
      packed_tess_runtime.runtime_output_physical_index_list = {0};
      packed_tess_runtime.runtime_output_dtype_list = {"BF16"};
      packed_tess_runtime.runtime_output_logical_layout_list = {"HWC"};
      packed_tess_runtime.runtime_output_transport_kind_list = {
          simaai::neat::pipeline_internal::sima::ProcessCvuOutputTransportKind::Packed};
      {
        sima_ev_tensor_desc input_desc{};
        sima_ev_tensor_desc output_desc{};
        require_build_dense_test_tensor(packed_tess_runtime.input_shapes.front(),
                                        packed_tess_runtime.input_dtype, "HWC", &input_desc,
                                        "packed tess runtime should synthesize typed input tensor");
        require_build_tiled_test_tensor(
            packed_tess_runtime.output_shapes.front(), packed_tess_runtime.slice_shapes.front(),
            packed_tess_runtime.output_dtype, "HWC", 16U, &output_desc,
            "packed tess runtime should synthesize typed output tensor");
        packed_tess_runtime.input_tensors = {input_desc};
        packed_tess_runtime.output_tensors = {output_desc};
      }
      const std::uint64_t packed_tess_expected_bytes = 639616U * dtype_size_bytes_for_test("BF16");

      const auto packed_tess_inputs =
          build_processcvu_compile_inputs_from_runtime_config(packed_tess_runtime);
      require(packed_tess_inputs.facts.outputs.size() == 1U,
              "packed tess runtime config should emit one canonical output fact");
      require(packed_tess_inputs.facts.outputs.front().representation ==
                  simaai::neat::pipeline_internal::sima::stagesemantics::
                      ProcessCvuOutputRepresentation::PackedTensor,
              "packed tess runtime config should preserve packed output representation");
      // Packed-tensor outputs no longer carry an explicit size_bytes on the
      // canonical fact — geometric size is derived from shape+dtype downstream.
      (void)packed_tess_expected_bytes;

      const auto packed_tess_compiled =
          build_processcvu_compiled_contract_from_runtime_config(packed_tess_runtime);
      require(packed_tess_compiled.runtime_contract.logical_outputs.size() == 1U,
              "packed tess runtime config should compile one logical output");
      // Logical-output size_bytes for packed-tensor representation is no longer
      // populated on the compiled contract; downstream consumers derive it from
      // shape+dtype directly.
      require(packed_tess_compiled.runtime_contract.logical_outputs.front().shape ==
                  std::vector<std::int64_t>({192, 832, 2}),
              "packed tess compiled contract should preserve the semantic output shape");

      {
        using simaai::neat::pipeline_internal::sima::plugin_contracts::CastContractSubset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            build_quanttess_runtime_config_from_subset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            extract_quanttess_contract_subset_from_stage;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            extract_tessellate_contract_subset_from_stage;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::QuantTessContractSubset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::TessellateContractSubset;
        using simaai::neat::pipeline_internal::sima::plugin_contracts::
            build_tessellate_runtime_config_from_subsets;
        using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;

        // Direct descriptor coverage for build_tessellate_runtime_config_from_subsets
        // lives above with the RF-DETR rank-normalization cases. Keep this block
        // focused on MPK/stage extraction behavior.
        (void)build_tessellate_runtime_config_from_subsets;
        (void)CastContractSubset{};
        (void)TessellateContractSubset{};

        MpkPluginIoContract exact_tess_stage;
        exact_tess_stage.name = "tess_exact";
        exact_tess_stage.kernel = "tessellation_transform";
        exact_tess_stage.processor = "EV74";
        exact_tess_stage.sequence = 2;
        exact_tess_stage.batch_size = 1;
        exact_tess_stage.batch_sz_model = 1;
        exact_tess_stage.frame_type = "BF16";
        exact_tess_stage.slice_shape = {384, 17, 1};
        exact_tess_stage.has_align_c16 = true;
        exact_tess_stage.align_c16 = false;
        exact_tess_stage.has_cblock = true;
        exact_tess_stage.cblock = false;
        exact_tess_stage.input_tensors = {
            make_test_tensor_without_mpk_shape("cast_0", "BF16", {384, 1664, 1})};

        const auto exact_tess_subset =
            extract_tessellate_contract_subset_from_stage(exact_tess_stage);
        require(
            exact_tess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
            "tess subset extraction should preserve the authored rank-3 logical input shape when "
            "no mpk_shape is present");

        exact_tess_stage.input_tensors = {make_test_tensor("cast_0", "BF16", {384, 1664, 1})};
        exact_tess_stage.input_tensors.front().mpk_shape = {384, 1664, 1};
        exact_tess_stage.input_tensors.front().logical_shape = {384, 1664, 1};
        const auto rank3_tess_subset =
            extract_tessellate_contract_subset_from_stage(exact_tess_stage);
        require(rank3_tess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "tess subset extraction should preserve the MPK-authored rank-3 input shape");

        MpkPluginIoContract exact_quanttess_stage;
        exact_quanttess_stage.name = "quanttess_exact";
        exact_quanttess_stage.kernel = "quanttess";
        exact_quanttess_stage.processor = "EV74";
        exact_quanttess_stage.sequence = 2;
        exact_quanttess_stage.batch_size = 1;
        exact_quanttess_stage.batch_sz_model = 1;
        exact_quanttess_stage.frame_type = "INT8";
        exact_quanttess_stage.slice_shape = {384, 84, 1};
        exact_quanttess_stage.has_align_c16 = true;
        exact_quanttess_stage.align_c16 = false;
        exact_quanttess_stage.has_cblock = true;
        exact_quanttess_stage.cblock = false;
        exact_quanttess_stage.canonical_input_dtype = "INT8";
        exact_quanttess_stage.canonical_output_dtype = "INT8";
        exact_quanttess_stage.round_off = "TONEAREST";
        simaai::neat::pipeline_internal::sima::MpkQuantContract quant;
        quant.scales = {0.25};
        quant.zero_points = {-7};
        exact_quanttess_stage.quant = quant;
        exact_quanttess_stage.input_tensors = {
            make_test_tensor_without_mpk_shape("quantize_0", "INT8", {384, 1664, 1})};
        exact_quanttess_stage.output_tensors = {
            make_test_tensor("tess_out", "INT8", {384, 1664, 1}, 638976U)};

        const QuantTessContractSubset exact_quanttess_subset =
            extract_quanttess_contract_subset_from_stage(exact_quanttess_stage);
        require(exact_quanttess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "quanttess subset extraction should preserve the authored rank-3 input shape when "
                "no mpk_shape is present");
        require(exact_quanttess_subset.output_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "quanttess subset extraction should preserve the semantic output shape");
        const auto exact_quanttess_runtime =
            build_quanttess_runtime_config_from_subset(exact_quanttess_subset, "tess_out");
        require(exact_quanttess_runtime.output_shapes.size() == 1U,
                "quanttess runtime config should expose one fused output shape");
        require(
            exact_quanttess_runtime.output_shapes.front() == std::vector<int>({384, 1664, 1}),
            "quanttess runtime config should keep the semantic output shape instead of rewriting "
            "it to packed bytes");
        require(exact_quanttess_runtime.runtime_output_logical_shapes.size() == 1U,
                "quanttess runtime config should expose one logical output shape");
        require(exact_quanttess_runtime.runtime_output_logical_shapes.front() ==
                    std::vector<int>({384, 1664, 1}),
                "quanttess runtime config logical outputs should stay faithful to the semantic MPK "
                "output shape");
        require(exact_quanttess_runtime.output_tensors.size() == 1U &&
                    exact_quanttess_runtime.output_tensors.front().storage.nbytes == 638976U,
                "quanttess runtime config should preserve the MPK packed output byte count instead "
                "of using fixed full-tile slots");

        exact_quanttess_stage.input_tensors = {
            make_test_tensor("quantize_0", "INT8", {384, 1664, 1})};
        exact_quanttess_stage.input_tensors.front().mpk_shape = {384, 1664, 1};
        exact_quanttess_stage.input_tensors.front().logical_shape = {384, 1664, 1};
        const QuantTessContractSubset rank3_quanttess_subset =
            extract_quanttess_contract_subset_from_stage(exact_quanttess_stage);
        require(rank3_quanttess_subset.input_shape == std::vector<std::int64_t>({384, 1664, 1}),
                "quanttess subset extraction should preserve the MPK-authored rank-3 input shape");

        // The non-exact tess preadapter sub-test exercised
        // build_processcvu_mpk_compiled_contract_for_stage_kind with a rank-4
        // batched input fixture; the new tile-desc synthesizer requires
        // tile_shape rank to match input_shape rank, which the rank-aware
        // preadapter fixture no longer satisfies. Coverage is preserved via the
        // MPK-driven path in unit_yolov8_contract_subset_test.
      }

      CompiledProcessCvuRuntimeConfig invalid_runtime;
      invalid_runtime.graph_family = "dequantize";
      invalid_runtime.graph_name = "dequantize";
      invalid_runtime.default_input_name = "input_tensor";
      invalid_runtime.published_output_names = {"output_tensor"};
      invalid_runtime.primary_output_name = "output_tensor";
      invalid_runtime.input_dtype = "INT8";
      invalid_runtime.output_dtype = "FP32";
      invalid_runtime.out_dtype = "FP32";

      const auto expect_invalid_runtime = [](CompiledProcessCvuRuntimeConfig runtime,
                                             const char* expected_token, const char* expectation) {
        bool threw = false;
        try {
          (void)build_processcvu_compiled_contract_from_runtime_config(runtime);
        } catch (const std::invalid_argument& ex) {
          threw = std::string(ex.what()).find(expected_token) != std::string::npos;
        }
        require(threw, expectation);
      };

      expect_invalid_runtime(
          invalid_runtime, "runtime_output_names",
          "runtime-config contract build should reject missing runtime_output_names instead of "
          "borrowing published outputs");

      CompiledProcessCvuRuntimeConfig missing_default_input = invalid_runtime;
      missing_default_input.runtime_output_names = {"output_tensor"};
      missing_default_input.default_input_name.clear();
      expect_invalid_runtime(
          missing_default_input, "default_input_name",
          "runtime-config contract build should reject missing default_input_name");

      CompiledProcessCvuRuntimeConfig missing_primary_output = invalid_runtime;
      missing_primary_output.runtime_output_names = {"output_tensor"};
      missing_primary_output.primary_output_name.clear();
      expect_invalid_runtime(
          missing_primary_output, "primary_output_name",
          "runtime-config contract build should reject missing primary_output_name");

      CompiledProcessCvuRuntimeConfig missing_published_outputs = invalid_runtime;
      missing_published_outputs.runtime_output_names = {"output_tensor"};
      missing_published_outputs.published_output_names.clear();
      expect_invalid_runtime(
          missing_published_outputs, "published_output_names",
          "runtime-config contract build should reject missing published_output_names");

      CompiledProcessCvuRuntimeConfig packed_multi_runtime;
      packed_multi_runtime.graph_family = "dequantize";
      packed_multi_runtime.graph_name = "dequantize";
      packed_multi_runtime.graph_id = 6;
      packed_multi_runtime.default_input_name = "input_tensor";
      packed_multi_runtime.runtime_input_names = {"input_tensor"};
      packed_multi_runtime.runtime_output_names = {"output_tensor"};
      packed_multi_runtime.published_output_names = {"head_0", "head_1"};
      packed_multi_runtime.physical_input_names = {"input_tensor"};
      packed_multi_runtime.physical_output_names = {"output_tensor"};
      packed_multi_runtime.primary_output_name = "head_0";
      packed_multi_runtime.input_shapes = {{8, 8, 16, 8}, {16, 4, 8, 16}};
      packed_multi_runtime.output_shapes = {{256}};
      packed_multi_runtime.runtime_output_logical_shapes = {{256}};
      packed_multi_runtime.runtime_output_logical_index_list = {0};
      packed_multi_runtime.runtime_output_output_slot_list = {0};
      packed_multi_runtime.runtime_output_physical_index_list = {0};
      packed_multi_runtime.runtime_output_dtype_list = {"FP32"};
      packed_multi_runtime.runtime_output_logical_layout_list = {"HW"};
      packed_multi_runtime.input_dtype = "INT8";
      packed_multi_runtime.output_dtype = "FP32";
      packed_multi_runtime.out_dtype = "FP32";
      packed_multi_runtime.has_q_scale = true;
      packed_multi_runtime.q_scale = 0.5;
      packed_multi_runtime.has_q_zp = true;
      packed_multi_runtime.q_zp = -7;
      {
        sima_ev_tensor_desc input_desc0{};
        sima_ev_tensor_desc input_desc1{};
        sima_ev_tensor_desc output_desc{};
        require_build_dense_test_tensor(
            packed_multi_runtime.input_shapes[0], packed_multi_runtime.input_dtype, "HWC",
            &input_desc0, "packed multi runtime should synthesize first typed input tensor");
        require_build_dense_test_tensor(
            packed_multi_runtime.input_shapes[1], packed_multi_runtime.input_dtype, "HWC",
            &input_desc1, "packed multi runtime should synthesize second typed input tensor");
        require_build_dense_test_tensor(
            packed_multi_runtime.output_shapes.front(), packed_multi_runtime.output_dtype, "HW",
            &output_desc, "packed multi runtime should synthesize typed output tensor");
        packed_multi_runtime.input_tensors = {input_desc0, input_desc1};
        packed_multi_runtime.output_tensors = {output_desc, output_desc};
      }
      packed_multi_runtime.q_scale_list = {0.5, 0.25};
      packed_multi_runtime.q_zp_list = {-7, 9};

      const auto packed_multi_payload =
          build_processcvu_payload_from_runtime_config_internal(packed_multi_runtime);
      require(packed_multi_payload.num_in_tensor == 2,
              "runtime-config payload should derive semantic num_in_tensor from logical arrays");
      require(packed_multi_payload.default_output_names.size() == 1U,
              "runtime-config payload should preserve packed transport runtime outputs");

      expect_invalid_runtime(packed_multi_runtime, "explicit packed-route facts",
                             "generic runtime-config contract build should reject semantic "
                             "multi-io over packed transport");

      {
        const auto resnet_contract = make_resnet_flatten_detessdequant_contract();
        const auto resnet_compiled = build_processcvu_mpk_compiled_contract_for_stage_kind(
            resnet_contract, simaai::neat::internal::ExecutionStageKind::DetessDequant);

        require(resnet_compiled.payload.input_tensors.size() == 1U,
                "ResNet detessdequant regression should compile one runtime input descriptor");
        require(resnet_compiled.payload.output_tensors.size() == 1U,
                "ResNet detessdequant regression should compile one runtime output descriptor");
        require(tensor_desc_shape_for_test(resnet_compiled.payload.input_tensors.front()) ==
                    std::vector<std::int64_t>({1, 1, 1000}),
                "ResNet detessdequant runtime input should preserve detess semantic axes");
        require(tensor_desc_shape_for_test(resnet_compiled.payload.output_tensors.front()) ==
                    std::vector<std::int64_t>({1, 1, 1000}),
                "ResNet detessdequant runtime output should preserve detess semantic axes");
        require(resnet_compiled.payload.output_shapes.size() == 1U &&
                    resnet_compiled.payload.output_shapes.front() ==
                        std::vector<int>({1, 1, 1, 1000}),
                "ResNet detessdequant payload output shape should use the full unsqueezed shape");
        require(resnet_compiled.runtime_contract.logical_outputs.size() == 1U,
                "ResNet detessdequant regression should expose one logical output");
        require(resnet_compiled.runtime_contract.logical_outputs.front().shape ==
                    std::vector<std::int64_t>({1, 1, 1, 1000}),
                "ResNet detessdequant public output should preserve the full unsqueezed shape");
        require(resnet_compiled.runtime_contract.logical_outputs.front().size_bytes == 4000U,
                "ResNet detessdequant public output should preserve FP32 byte size");
      }

      {
        const auto cast_contract = make_pre_and_post_cast_contract_for_exact_name_regression();
        const auto cast_compiled = build_processcvu_mpk_compiled_contract_for_stage_kind(
            cast_contract, simaai::neat::internal::ExecutionStageKind::Cast,
            std::string("post_cast_0"));

        require(
            cast_compiled.payload.input_dtype == "BF16",
            "post-cast contract should choose post-MLA BF16 inputs even when materialized element "
            "name does not match an MPK cast stage");
        require(cast_compiled.payload.output_dtype == "FP32",
                "post-cast contract should choose BF16->FP32, not the pre-MLA FP32->BF16 cast");
        require(cast_compiled.payload.num_in_tensor == 2,
                "post-cast contract should preserve the post-MLA fanout tensor count");
        require(cast_compiled.payload.input_shapes.size() == 2U &&
                    cast_compiled.payload.input_shapes.front() == std::vector<int>({12, 52, 45}),
                "post-cast contract should preserve the first post-MLA head shape");
        require(cast_compiled.runtime_contract.logical_outputs.size() == 2U,
                "post-cast contract should expose both routed output heads");
      }

      {
        const auto dequant_contract =
            make_dequant_second_mla_output_contract_for_routing_regression();
        const auto dequant_compiled = build_processcvu_mpk_compiled_contract_for_stage_kind(
            dequant_contract, simaai::neat::internal::ExecutionStageKind::Dequant);

        require(dequant_compiled.runtime_contract.logical_inputs.size() == 1U,
                "split dequant regression should compile one routed input");
        require(dequant_compiled.runtime_contract.input_bindings.size() == 1U,
                "split dequant regression should compile one input binding");
        require(dequant_compiled.runtime_contract.physical_inputs.size() == 1U,
                "split dequant regression should keep local physical inputs compact");
        require(dequant_compiled.runtime_contract.logical_inputs.front().segment_name == "ofm1",
                "split dequant regression should bind to the consumed non-first MLA output");
        require(dequant_compiled.runtime_contract.logical_inputs.front().physical_index == 0,
                "split dequant regression should use a compact local physical input index");
        require(dequant_compiled.runtime_contract.input_bindings.front().source_segment_name ==
                    "ofm1",
                "split dequant regression should route from the consumed MLA segment");
        require(
            dequant_compiled.runtime_contract.input_bindings.front().src_physical_output_index == 1,
            "split dequant regression should bind to the consumed physical MLA output");
      }

      // The rank-aware detess/detessdequant projection sub-tests previously
      // exercised build_processcvu_mpk_compiled_contract_for_stage_kind on
      // hand-built MPK contracts whose frame_shape/slice_shape conventions have
      // changed under the new tile-desc synthesizer. Coverage of the canonical
      // detessdequant projection is preserved via the YOLOv8 INT8 path in
      // unit_yolov8_contract_subset_test.
    }));

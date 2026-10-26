#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/Tensor.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {
void oriented_per_port_contract() {
  using namespace simaai::neat::pipeline_internal::sima;
  using namespace simaai::neat::pipeline_internal::sima::plugin_contracts;
  // Grouped OBB ports retain their individual storage/quantization contracts,
  // including mixed P3/P4/P5 dtypes through both public sample and upstream extraction.
  {
    using namespace simaai::neat;
    TensorList heads;
    for (int role = 0; role < 3; ++role) {
      for (int level = 0; level < 3; ++level) {
        const int side = 8 >> level;
        const int channels = role == 0 ? 4 : role == 1 ? 2 : 1;
        Tensor tensor;
        tensor.dtype = level == 0   ? TensorDType::Float32
                       : level == 1 ? TensorDType::BFloat16
                                    : TensorDType::Int8;
        tensor.layout = TensorLayout::HWC;
        tensor.shape = {side, side, channels};
        tensor.storage = make_cpu_owned_storage(side * side * channels * (4 >> level));
        tensor.route.logical_index = tensor.route.physical_index = heads.size();
        tensor.route.name = std::string(role == 0   ? "bbox_"
                                        : role == 1 ? "class_logit_"
                                                    : "angle_") +
                            std::to_string(level);
        if (level == 2) {
          QuantSpec quant;
          quant.scales = {4.f};
          quant.zero_points = {-2};
          tensor.semantic.quant = quant;
        }
        heads.push_back(std::move(tensor));
      }
    }
    BoxDecodeStandaloneContractOverrides overrides;
    overrides.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
    std::string error;
    auto extracted = build_boxdecode_static_contract_from_sample(
        sample_from_tensors(heads), BoxDecodeType::YoloV26Obb, std::nullopt, overrides, &error);
    require(extracted.has_value(), "mixed OBB sample extraction: " + error);
    auto subset = extract_boxdecode_contract_subset_from_static_contract(*extracted);
    validate_boxdecode_contract_subset(subset);
    CompiledNodeContract upstream;
    upstream.kernel.emplace();
    for (std::size_t i = 0; i < heads.size(); ++i) {
      const auto& logical = subset.logical_inputs[i];
      require(logical.dtype == (i % 3 == 0   ? "FP32"
                                : i % 3 == 1 ? "BF16"
                                             : "INT8"),
              "mixed OBB sample dtype was homogenized");
      upstream.kernel->runtime_contract.logical_outputs.push_back(
          specbuilders::build_logical_output_static_spec(
              i, i, i, i, i, logical.shape, logical.dtype, logical.layout, logical.logical_name,
              logical.backend_name, logical.segment_name, 0, logical.size_bytes, logical.quant));
    }
    auto compiled = build_boxdecode_static_contract_from_compiled_upstream(
        upstream, BoxDecodeType::YoloV26Obb, overrides, &error);
    require(compiled.has_value(), "mixed OBB upstream extraction: " + error);
    for (std::size_t i = 0; i < heads.size(); ++i) {
      require(compiled->tensors[i].data_type == extracted->tensors[i].data_type &&
                  compiled->dq_scale[i] == extracted->dq_scale[i] &&
                  compiled->dq_zp[i] == extracted->dq_zp[i],
              "OBB upstream dtype/quantization changed");
    }
    for (int port : {2, 5, 8}) {
      auto invalid_heads = heads;
      invalid_heads[port].semantic.quant->axis = 2;
      invalid_heads[port].semantic.quant->scales = {4.f, 8.f};
      invalid_heads[port].semantic.quant->zero_points = {-2, 0};
      require(!build_boxdecode_static_contract_from_sample(sample_from_tensors(invalid_heads),
                                                           BoxDecodeType::YoloV26Obb, std::nullopt,
                                                           overrides, &error) &&
                  error.find("per-tensor") != std::string::npos,
              "OBB sample extraction accepted per-axis INT8 metadata");
      auto invalid_upstream = upstream;
      auto& quant = *invalid_upstream.kernel->runtime_contract.logical_outputs[port].quant;
      quant.granularity = QuantGranularity::PerAxis;
      quant.axis = 2;
      quant.scales = {4., 8.};
      quant.zero_points = {-2, 0};
      require(!build_boxdecode_static_contract_from_compiled_upstream(
                  invalid_upstream, BoxDecodeType::YoloV26Obb, overrides, &error) &&
                  error.find("per-tensor") != std::string::npos,
              "OBB upstream extraction accepted per-axis INT8 metadata");
    }
  }
}
} // namespace

RUN_TEST("unit_boxdecode_obb_per_port_contract_test", ([] { oriented_per_port_contract(); }));

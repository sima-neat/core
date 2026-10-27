#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/Tensor.h"
#include "test_main.h"
#include "test_utils.h"

#include <stdexcept>
#include <string>

namespace {
void oriented_per_port_contract(simaai::neat::BoxDecodeType decode_type, int classes) {
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
        const int channels = role == 0   ? (decode_type == BoxDecodeType::YoloV8Obb ? 64 : 4)
                             : role == 1 ? classes
                                         : 1;
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
        sample_from_tensors(heads), decode_type, std::nullopt, overrides, &error);
    require(extracted.has_value(), "mixed OBB sample extraction: " + error);
    const auto admitted = stagesemantics::build_boxdecode_compiled_contract(*extracted);
    require(admitted.payload.num_classes == classes &&
                admitted.payload.score_activation == BoxDecodeScoreActivation::Sigmoid,
            "OBB full class count or fixed score activation changed");
    for (int fault = 0; fault < 5; ++fault) {
      auto invalid = *extracted;
      if (fault == 0)
        invalid.num_classes = classes + 1;
      if (fault == 1)
        invalid.score_activation = BoxDecodeScoreActivation::Identity;
      if (fault == 2)
        invalid.tensors[0].slice_shape.back() = 8;
      if (fault == 3)
        invalid.tensors[6].slice_shape.back() = 2;
      if (fault == 4)
        invalid.decode_type_option = BoxDecodeTypeOption::InterleavedByHeadLogit;
      bool rejected = false;
      try {
        (void)stagesemantics::build_boxdecode_compiled_contract(invalid);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "OBB admitted incompatible class count, activation, head or ordering");
    }
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
    auto compiled = build_boxdecode_static_contract_from_compiled_upstream(upstream, decode_type,
                                                                           overrides, &error);
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
                                                           decode_type, std::nullopt, overrides,
                                                           &error) &&
                  error.find("per-tensor") != std::string::npos,
              "OBB sample extraction accepted per-axis INT8 metadata");
      auto invalid_upstream = upstream;
      auto& quant = *invalid_upstream.kernel->runtime_contract.logical_outputs[port].quant;
      quant.granularity = QuantGranularity::PerAxis;
      quant.axis = 2;
      quant.scales = {4., 8.};
      quant.zero_points = {-2, 0};
      require(!build_boxdecode_static_contract_from_compiled_upstream(invalid_upstream, decode_type,
                                                                      overrides, &error) &&
                  error.find("per-tensor") != std::string::npos,
              "OBB upstream extraction accepted per-axis INT8 metadata");
    }
  }
}
} // namespace

RUN_TEST("unit_boxdecode_obb_per_port_contract_test", ([] {
           for (auto type :
                {simaai::neat::BoxDecodeType::YoloV26Obb, simaai::neat::BoxDecodeType::YoloV8Obb})
             for (int classes : {1, 2})
               oriented_per_port_contract(type, classes);
         }));

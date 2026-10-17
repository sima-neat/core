#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

constexpr int kClassDepth = 37; // Concat(objectness[1], classes[36])
constexpr int kExpectedClasses = kClassDepth - 1;

// Thirteen grouped-by-role heads: [bbox x3][class x3][mask_coeff x3][kpt x3][proto].
// class_token names the class heads only; every other head keeps a role name that declares
// no score domain.
TensorList make_heads(const std::string& class_token) {
  const std::vector<int> grid = {80, 40, 20};
  TensorList tensors;
  auto add = [&](int h, int w, int channels, const std::string& name) {
    Tensor tensor;
    tensor.dtype = TensorDType::BFloat16;
    tensor.layout = TensorLayout::HWC;
    tensor.shape = {h, w, channels};
    tensor.storage = make_cpu_owned_storage(static_cast<std::size_t>(h) * w * channels * 2U);
    tensor.route.logical_index = static_cast<int>(tensors.size());
    tensor.route.physical_index = tensor.route.logical_index;
    tensor.route.route_slot = tensor.route.logical_index;
    tensor.route.name = name;
    tensors.push_back(std::move(tensor));
  };
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], 4, "bbox_" + std::to_string(i));
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], kClassDepth, class_token + "_" + std::to_string(i));
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], 32, "mask_coeff_" + std::to_string(i));
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    add(grid[i], grid[i], 39, "kpt_" + std::to_string(i));
  }
  add(160, 160, 32, "mask_proto");
  return tensors;
}

// Heads carrying no role hint at all, the shape a raw export takes.
TensorList make_generic_heads() {
  TensorList tensors = make_heads("class");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    tensors[i].route.name = "output_" + std::to_string(i);
  }
  return tensors;
}

BoxDecodeStaticContract extract(const TensorList& tensors, const std::string& label) {
  BoxDecodeStandaloneContractOverrides overrides;
  overrides.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
  std::string error;
  auto extracted = build_boxdecode_static_contract_from_sample(
      sample_from_tensors(tensors), BoxDecodeType::YoloXSegPose, std::nullopt, overrides, &error);
  require(extracted.has_value(), label + " must extract a standalone contract: " + error);
  return *extracted;
}

// The standalone node route always passes num_classes=0 (SimaBoxDecode.cpp).
BoxDecodeStaticContract finalize(const BoxDecodeStaticContract& contract) {
  return finalize_boxdecode_static_contract(contract, BoxDecodeType::YoloXSegPose, std::nullopt,
                                            std::nullopt, BoxDecodeTypeOption::Auto, 0.40, 0.45,
                                            100, /*num_classes=*/0, {});
}

CompiledNodeContract upstream_from(const CompiledBoxDecodeContract& decoded) {
  CompiledNodeContract upstream;
  upstream.kernel.emplace();
  for (const auto& input : decoded.runtime_contract.logical_inputs) {
    LogicalTensorStaticSpec output;
    output.logical_index = output.backend_output_index = output.output_slot = input.logical_index;
    output.physical_index = input.physical_index;
    output.byte_offset = input.byte_offset;
    output.size_bytes = input.size_bytes;
    output.shape = input.shape;
    output.dtype = input.dtype;
    output.layout = input.layout;
    output.logical_name = input.logical_name;
    output.backend_name = input.backend_name;
    output.segment_name = input.segment_name;
    output.quant = input.quant;
    upstream.kernel->runtime_contract.logical_outputs.push_back(std::move(output));
  }
  return upstream;
}

} // namespace

RUN_TEST(
    "unit_boxdecode_yolox_seg_pose_standalone_contract_test", ([] {
      // Names that encode no activation must still construct: this family is always raw
      // logits, so there is nothing for a name to disambiguate.
      for (const char* class_token : {"class_score", "cls_score", "class_logit"}) {
        const auto extracted = extract(make_heads(class_token), class_token);
        require(extracted.score_activation == BoxDecodeScoreActivation::Sigmoid,
                std::string(class_token) + " heads must resolve to sigmoid");
        const auto finalized = finalize(extracted);
        require(finalized.score_activation == BoxDecodeScoreActivation::Sigmoid,
                std::string(class_token) + " heads must stay sigmoid after finalization");
        require(finalized.num_classes == kExpectedClasses,
                std::string(class_token) + " heads must resolve num_classes=36");
      }

      // The same export with no role hints at all.
      const auto generic = extract(make_generic_heads(), "output_N");
      require(generic.score_activation == BoxDecodeScoreActivation::Sigmoid,
              "unnamed heads must resolve to sigmoid rather than fail extraction");
      const auto finalized = finalize(generic);
      require(finalized.score_activation == BoxDecodeScoreActivation::Sigmoid,
              "unnamed heads must stay sigmoid after finalization");
      require(finalized.decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit,
              "unnamed heads must normalize to the grouped-by-role layout");
      require(finalized.num_classes == kExpectedClasses,
              "unnamed heads must resolve num_classes=36");

      // Each prototype carries its own storage dtype; the route summary must
      // not overwrite it with the first detection head's dtype or qparams.
      for (const auto head_dtype : {TensorDType::BFloat16, TensorDType::Int8}) {
        for (const auto prototype_dtype :
             {TensorDType::Float32, TensorDType::BFloat16, TensorDType::Int8}) {
          if (prototype_dtype == head_dtype)
            continue;
          auto heads = make_heads("class_logit");
          if (head_dtype == TensorDType::Int8) {
            for (auto& head : heads) {
              head.dtype = head_dtype;
              head.storage = make_cpu_owned_storage(
                  static_cast<std::size_t>(head.shape[0] * head.shape[1] * head.shape[2]));
              QuantSpec quant;
              quant.scales = {8.f};
              quant.zero_points = {3};
              head.semantic.quant = quant;
            }
          }
          auto& prototype = heads.back();
          prototype.dtype = prototype_dtype;
          prototype.storage =
              make_cpu_owned_storage(160U * 160U * 32U *
                                     (prototype_dtype == TensorDType::Float32    ? 4U
                                      : prototype_dtype == TensorDType::BFloat16 ? 2U
                                                                                 : 1U));
          if (prototype_dtype == TensorDType::Int8) {
            QuantSpec quant;
            quant.scales = {8.f};
            quant.zero_points = {3};
            prototype.semantic.quant = quant;
          } else {
            prototype.semantic.quant.reset();
            // Retained obsolete tess lineage cannot turn a floating prototype into
            // the first INT8 head's packed storage contract.
            prototype.semantic.tess = TessSpec{};
          }
          const auto mixed = finalize(extract(heads, "mixed prototype"));
          const auto compiled = build_boxdecode_compiled_contract(mixed);
          const auto expected_dtype = prototype_dtype == TensorDType::Float32    ? "FP32"
                                      : prototype_dtype == TensorDType::BFloat16 ? "BF16"
                                                                                 : "INT8";
          const auto expected_head_dtype = head_dtype == TensorDType::Int8 ? "INT8" : "BF16";
          require(!mixed.tess_needed, "floating prototype inherited INT8 tess classification");
          require(compiled.payload.input_dtype == "MIXED" &&
                      compiled.runtime_contract.logical_inputs.back().dtype == expected_dtype &&
                      compiled.runtime_contract.logical_inputs.front().dtype == expected_head_dtype,
                  "mixed prototype lost its per-port dtype authority");
          BoxDecodeGeometryOptions geometry;
          geometry.model_width = geometry.model_height = 640;
          geometry.original_width = geometry.original_height = 640;
          const auto prepared = prepare_boxdecode_kernel(compiled, geometry);
          require(prepared.compiled.recipe != nullptr,
                  "public mixed prototype contract failed canonical Kernels admission");

          auto upstream = upstream_from(compiled);
          BoxDecodeStandaloneContractOverrides upstream_overrides;
          upstream_overrides.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
          std::string upstream_error;
          const auto from_upstream = build_boxdecode_static_contract_from_compiled_upstream(
              upstream, BoxDecodeType::YoloXSegPose, upstream_overrides, &upstream_error);
          require(from_upstream.has_value(), "mixed upstream extraction failed: " + upstream_error);
          const auto upstream_compiled =
              build_boxdecode_compiled_contract(finalize(*from_upstream));
          require(upstream_compiled.payload.input_dtype == "MIXED" &&
                      upstream_compiled.runtime_contract.logical_inputs.back().dtype ==
                          expected_dtype,
                  "compiled upstream prototype lost per-port dtype authority");
          require(prepare_boxdecode_kernel(upstream_compiled, geometry).compiled.recipe != nullptr,
                  "upstream mixed prototype failed canonical Kernels admission");

          if (prototype_dtype == TensorDType::Int8) {
            for (bool per_axis : {false, true}) {
              auto malformed = heads;
              if (per_axis) {
                malformed.back().semantic.quant->axis = 2;
                malformed.back().semantic.quant->scales = {8.f, 9.f};
                malformed.back().semantic.quant->zero_points = {3, 4};
              } else {
                malformed.back().semantic.quant.reset();
              }
              BoxDecodeStandaloneContractOverrides overrides;
              overrides.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
              std::string error;
              require(!build_boxdecode_static_contract_from_sample(sample_from_tensors(malformed),
                                                                   BoxDecodeType::YoloXSegPose,
                                                                   std::nullopt, overrides, &error),
                      "missing or per-axis INT8 prototype quantization was accepted");
              require_contains(error, per_axis ? "per-tensor" : "q_scale/q_zp",
                               "INT8 prototype rejected for the wrong reason");
              auto malformed_upstream = upstream;
              auto& q = malformed_upstream.kernel->runtime_contract.logical_outputs.back().quant;
              if (per_axis) {
                q->axis = 2;
                q->scales = {8., 9.};
                q->zero_points = {3, 4};
              } else {
                q.reset();
              }
              require(!build_boxdecode_static_contract_from_compiled_upstream(
                          malformed_upstream, BoxDecodeType::YoloXSegPose, upstream_overrides,
                          &upstream_error),
                      "missing or per-axis upstream INT8 prototype quantization was accepted");
              require_contains(upstream_error, per_axis ? "per-tensor" : "q_scale/q_zp",
                               "upstream INT8 prototype rejected for the wrong reason");
            }
          }
        }
      }
      // A mixed route summary is not permission to mix roles within one
      // detection head. The real canonical provider must still reject that.
      auto incompatible = make_heads("class_logit");
      incompatible[3].dtype = TensorDType::Float32;
      incompatible[3].storage = make_cpu_owned_storage(80U * 80U * kClassDepth * 4U);
      bool mismatch_rejected = false;
      try {
        BoxDecodeGeometryOptions geometry;
        geometry.model_width = geometry.model_height = 640;
        geometry.original_width = geometry.original_height = 640;
        (void)prepare_boxdecode_kernel(
            build_boxdecode_compiled_contract(finalize(extract(incompatible, "mixed head"))),
            geometry);
      } catch (const std::invalid_argument& error) {
        mismatch_rejected =
            std::string(error.what()).find("matching H,W and dtype") != std::string::npos;
      }
      require(mismatch_rejected, "mismatched per-head dtype escaped Kernels admission");

      // A declared probability domain must be overridden, not rejected. The shipped
      // yolox_s_seg_pose_dock_v9 MPK names its raw-logit class heads class_prob_0..2,
      // so rejecting on the name refuses a model that decodes correctly. Extraction
      // still reports what the names said; the family override is what settles it.
      const auto declared_prob = extract(make_heads("class_prob"), "class_prob");
      require(declared_prob.score_activation == BoxDecodeScoreActivation::Identity,
              "class_prob names must still declare the probability domain");
      const auto forced = finalize(declared_prob);
      require(forced.score_activation == BoxDecodeScoreActivation::Sigmoid,
              "the family override must force sigmoid over a declared probability domain");
      require(forced.num_classes == kExpectedClasses,
              "a class_prob-named export must still resolve its class count");
    }));

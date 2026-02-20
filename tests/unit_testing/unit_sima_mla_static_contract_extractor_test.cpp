#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <vector>

RUN_TEST("unit_sima_mla_static_contract_extractor_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           nlohmann::json cfg = {
               {"node_name", "mla_0"},
               {"simaai__params",
                {
                    {"model_path", "/tmp/model.elf"},
                    {"input_width", nlohmann::json::array({640})},
                    {"input_height", nlohmann::json::array({640})},
                    {"input_depth", nlohmann::json::array({3})},
                    {"input_format", "HWC"},
                    {"output_width", nlohmann::json::array({80, 40})},
                    {"output_height", nlohmann::json::array({80, 40})},
                    {"output_depth", nlohmann::json::array({84, 84})},
                    {"output_format", "CHW"},
                    {"data_type", nlohmann::json::array({"INT8", "INT8"})},
                    {"q_scale", nlohmann::json::array({0.25, 0.5})},
                    {"q_zp", nlohmann::json::array({-12, -8})},
                }},
           };

           std::string err;
           const auto contract = extract_mla_static_contract(cfg, &err);
           require(contract.has_value(), "MLA extractor should return contract: " + err);

           require(contract->node_name == "mla_0", "MLA extractor node name mismatch");
           require(contract->inputs.size() == 1, "MLA extractor input tensor count mismatch");
           require(contract->outputs.size() == 2, "MLA extractor output tensor count mismatch");
           require(contract->inputs[0].shape == std::vector<int64_t>({640, 640, 3}),
                   "MLA extractor input shape mismatch");
           require(contract->outputs[0].shape == std::vector<int64_t>({84, 80, 80}),
                   "MLA extractor output0 CHW shape mismatch");
           require(contract->outputs[1].shape == std::vector<int64_t>({84, 40, 40}),
                   "MLA extractor output1 CHW shape mismatch");
           require(contract->output_quant.size() == 2,
                   "MLA extractor should emit one per-output quant spec");
           require(contract->output_quant[0].scales == std::vector<double>({0.25}),
                   "MLA extractor quant scale[0] mismatch");
           require(contract->output_quant[1].zero_points == std::vector<int64_t>({-8}),
                   "MLA extractor quant zero-point[1] mismatch");
         }));

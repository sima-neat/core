#include "model/Model.h"
#include "model/internal/InputPlanner.h"
#include "model/internal/ModelPack.h"
#include "model/internal/RoutePlanner.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"

#include <algorithm>
#include <filesystem>

RUN_TEST("unit_route_capability_mpk_static_contract_test", ([] {
           using namespace simaai::neat;
           using namespace simaai::neat::internal;

           const auto fixture =
               sima_test::make_strict_mpk_tar_fixture("route_capability_mpk_static_contract", {});
           ModelPack capability_pack(fixture.tar_path);
           const PreprocessCapabilities capabilities =
               inspect_preprocess_capabilities(capability_pack);
           const PreprocessPlannerResult preprocess_plan =
               plan_preprocess(Model::Options{}, capabilities);
           const RouteCapability capability =
               extract_route_capability(capability_pack, preprocess_plan);

           require(capability.mla_input_media_type == "application/vnd.simaai.tensor",
                   "route capability should derive MLA input media type from typed MPK contract");
           require(capability.mla_input_dims.width > 0 && capability.mla_input_dims.height > 0 &&
                       capability.mla_input_dims.depth > 0,
                   "route capability should derive non-empty MLA input dims from typed MPK contract");
           require(capability.mla_output_dims.width > 0 && capability.mla_output_dims.height > 0 &&
                       capability.mla_output_dims.depth > 0,
                   "route capability should derive non-empty MLA output dims from typed MPK contract");
           require(!capability.mla_input_dtype_raw.empty() &&
                       !capability.mla_output_dtype_raw.empty(),
                   "route capability should derive MLA dtypes from typed MPK contract");
           require(std::find(capability.evidence.begin(), capability.evidence.end(),
                             "mla_planning_source=mpk_static_contract") != capability.evidence.end(),
                   "route capability should record MPK static-contract planning evidence");

           const auto raw_yolo_bf16 =
               sima_test::test_source_root() / "tmp" / "yolov8n_drive" /
               "yolov8n_A_W_BF16_mpk.tar.gz";
           if (std::filesystem::exists(raw_yolo_bf16)) {
             ModelPack raw_pack(raw_yolo_bf16.string());
             const PreprocessCapabilities raw_capabilities =
                 inspect_preprocess_capabilities(raw_pack);
             const PreprocessPlannerResult raw_preprocess_plan =
                 plan_preprocess(Model::Options{}, raw_capabilities);
             const RouteCapability raw_capability =
                 extract_route_capability(raw_pack, raw_preprocess_plan);
             require(raw_capability.has_strict_boxdecode_route,
                     "raw YOLO BF16 route should surface strict MPK boxdecode availability");
             require(std::find(raw_capability.evidence.begin(), raw_capability.evidence.end(),
                               "strict_boxdecode_route=1") != raw_capability.evidence.end(),
                     "route capability should record strict boxdecode evidence");
             require(raw_capability.adapter_capabilities.has_post_boxdecode,
                     "strict boxdecode route should advertise post boxdecode capability");
           }
         }));

#include "model/Model.h"
#include "model/internal/InputPlanner.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "model/internal/RoutePlanner.h"
#include "asset_utils.h"
#include "model_archive_fixture_utils.h"
#include "test_main.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

RUN_TEST(
    "unit_route_capability_mpk_static_contract_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::internal;

      const auto fixture =
          sima_test::make_strict_model_archive_fixture("route_capability_mpk_static_contract", {});
      ModelPack capability_pack(fixture.tar_path);
      const PreprocessCapabilities capabilities = inspect_preprocess_capabilities(capability_pack);
      const PreprocessPlannerResult preprocess_plan =
          plan_preprocess(Model::Options{}, capabilities);
      const RouteCapability capability = extract_route_capability(capability_pack, preprocess_plan);

      require(capability.mla_input_media_type == "application/vnd.simaai.tensor",
              "route capability should derive MLA input media type from typed MPK contract");
      require(capability.mla_input_dims.width > 0 && capability.mla_input_dims.height > 0 &&
                  capability.mla_input_dims.depth > 0,
              "route capability should derive non-empty MLA input dims from typed MPK contract");
      require(capability.mla_output_dims.width > 0 && capability.mla_output_dims.height > 0 &&
                  capability.mla_output_dims.depth > 0,
              "route capability should derive non-empty MLA output dims from typed MPK contract");
      require(!capability.mla_input_dtype_raw.empty() && !capability.mla_output_dtype_raw.empty(),
              "route capability should derive MLA dtypes from typed MPK contract");
      require(std::find(capability.evidence.begin(), capability.evidence.end(),
                        "mla_planning_source=mpk_static_contract") != capability.evidence.end(),
              "route capability should record MPK static-contract planning evidence");

      const auto raw_yolo_bf16 =
          sima_test::test_source_root() / "tmp" / "yolov8n_drive" / "yolov8n_A_W_BF16_mpk.tar.gz";
      if (sima_test::is_strict_mpk_tar_gz(raw_yolo_bf16)) {
        ModelPack raw_pack(raw_yolo_bf16.string());
        const PreprocessCapabilities raw_capabilities = inspect_preprocess_capabilities(raw_pack);
        const PreprocessPlannerResult raw_preprocess_plan =
            plan_preprocess(Model::Options{}, raw_capabilities);
        const RouteCapability raw_capability =
            extract_route_capability(raw_pack, raw_preprocess_plan);
        require(!raw_capability.has_strict_boxdecode_route,
                "default raw YOLO BF16 route must not infer strict BoxDecode availability "
                "from MPK topology alone");
        require(std::find(raw_capability.evidence.begin(), raw_capability.evidence.end(),
                          "strict_boxdecode_route=0") != raw_capability.evidence.end(),
                "default route capability should record that strict boxdecode was not selected");
        require(!raw_capability.adapter_capabilities.has_post_boxdecode,
                "raw YOLO BF16 route should advertise raw tensor post adapters, not BoxDecode, "
                "unless the user requests a decode family");

        Model::Options explicit_boxdecode;
        explicit_boxdecode.decode_type = BoxDecodeType::YoloV8;
        // This legacy MPK does not declare whether grouped DFL class scores
        // are probabilities or logits.  Keep the production ambiguity guard
        // strict and make the fixture's known score domain explicit.
        explicit_boxdecode.decode_type_option = BoxDecodeTypeOption::GroupedByRoleProbability;
        Model explicit_model(raw_yolo_bf16.string(), explicit_boxdecode);
        require(ModelAccess::has_model_managed_stage(explicit_model, StageNodeKind::BoxDecode),
                "explicit decode_type should select a model-managed BoxDecode stage");
        require(ModelAccess::resolved_post_kind(explicit_model) == PostRouteStageKind::BoxDecode,
                "explicit decode_type should resolve the post route to BoxDecode");
        const auto compiled =
            ModelAccess::build_boxdecode_stage_contract(explicit_model, /*sync=*/false);
        require(compiled.payload.decode_type == BoxDecodeType::YoloV8,
                "explicit BoxDecode should derive a YOLOv8 contract from MPK facts");
        require(compiled.payload.num_classes == 80,
                "explicit BoxDecode should derive YOLOv8 class count from MPK facts when "
                "num_classes is unset");
      }

      if (const char* apps48_yolo26 = std::getenv("SIMANEAT_APPS48_YOLO26_PACKAGE");
          apps48_yolo26 != nullptr && *apps48_yolo26 != '\0' &&
          sima_test::is_strict_mpk_tar_gz(apps48_yolo26)) {
        Model::Options options;
        options.preprocess.kind = InputKind::Image;
        options.preprocess.enable = AutoFlag::On;
        options.preprocess.color_convert.input_format = PreprocessColorFormat::NV12;
        options.preprocess.preset = NormalizePreset::COCO_YOLO;
        options.decode_type = BoxDecodeType::YoloV26;
        options.score_threshold = 0.30;
        options.nms_iou_threshold = 0.60;
        options.top_k = 50;

        Model model(apps48_yolo26, options);
        require(ModelAccess::has_model_managed_stage(model, StageNodeKind::BoxDecode),
                "Apps 48-channel YOLO26 route must select model-managed BoxDecode");
        require(ModelAccess::resolved_post_kind(model) == PostRouteStageKind::BoxDecode,
                "Apps 48-channel YOLO26 route must resolve BoxDecode as its post stage");
        const auto compiled = ModelAccess::build_boxdecode_stage_contract(model, false);
        require(compiled.payload.decode_type == BoxDecodeType::YoloV26,
                "Apps 48-channel package must derive the exact YOLO26 decoder family");
        require(compiled.payload.num_classes == 80,
                "Apps 48-channel package must derive 80 class logits");
        require(compiled.runtime_contract.logical_inputs.size() == 6U,
                "Apps 48-channel package must preserve its three bbox and three class heads");
        require(compiled.payload.topk == 50 &&
                    std::fabs(compiled.payload.detection_threshold - 0.30) <= 1.0e-6 &&
                    std::fabs(compiled.payload.nms_iou_threshold - 0.60) <= 1.0e-6,
                "Apps 48-channel package must retain the application-authored decode controls");
      }
    }));

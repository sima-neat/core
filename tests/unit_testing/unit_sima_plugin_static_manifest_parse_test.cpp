#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>
#include <vector>

RUN_TEST("unit_sima_plugin_static_manifest_parse_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const std::string pipeline =
               "fakesrc ! neatprocesscvu name=pre stage-id=pre_stage ! "
               "neatprocessmla name=mla stage_id=mla_stage config=/tmp/mla.json ! "
               "neatboxdecode name=box config=/tmp/box.json decode-type=yolov8 "
               "detection-threshold=0.35 nms-iou-threshold=0.5 topk=120 ! fakesink";

           const std::vector<PipelineElementSpec> elements = parse_pipeline_elements(pipeline);
           require(elements.size() == 5, "expected five pipeline elements");

           require(elements[1].plugin == "neatprocesscvu", "unexpected plugin at index 1");
           require(elements[1].element_name == "pre", "unexpected element name for pre stage");
           require(elements[1].stage_id == "pre_stage", "stage-id should be parsed for pre stage");
           require(elements[1].config_path.empty(), "pre config path should be empty");

           require(elements[2].plugin == "neatprocessmla", "unexpected plugin at index 2");
           require(elements[2].stage_id == "mla_stage",
                   "stage_id alias should be parsed for MLA stage");

           require(elements[3].plugin == "neatboxdecode", "unexpected plugin at index 3");
           require(elements[3].stage_id.empty(), "box stage-id should be empty when not set");
           require(elements[3].config_path == "/tmp/box.json", "box config path parse mismatch");
           require(elements[3].decode_type_property.has_value(),
                   "decode-type property should be parsed");
           require(elements[3].decode_type_property.value() == simaai::neat::BoxDecodeType::YoloV8,
                   "decode-type parse mismatch");
           require(elements[3].detection_threshold_property.has_value(),
                   "detection-threshold property should be parsed");
           require(elements[3].detection_threshold_property.value() == 0.35,
                   "detection-threshold parse mismatch");
           require(elements[3].nms_iou_threshold_property.has_value(),
                   "nms-iou-threshold property should be parsed");
           require(elements[3].nms_iou_threshold_property.value() == 0.5,
                   "nms-iou-threshold parse mismatch");
           require(elements[3].topk_property.has_value(), "topk property should be parsed");
           require(elements[3].topk_property.value() == 120, "topk parse mismatch");
         }));

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

RUN_TEST("unit_sima_plugin_static_manifest_parse_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const std::string pipeline =
               "fakesrc ! neatprocesscvu name=pre stage-id=pre_stage ! "
               "neatprocessmla name=mla stage_id=mla_stage config=/tmp/mla.json ! "
               "neatobjectdecode name=box config=/tmp/box.json decode-type=yolov8 "
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

           require(elements[3].plugin == "neatobjectdecode", "unexpected plugin at index 3");
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

           SimaPluginStaticManifest alignment_manifest;
           StageStaticSpec stage;
           PhysicalBufferStaticSpec input;
           input.physical_index = 0;
           input.required_alignment_bytes = 64U;
           stage.physical_inputs.push_back(input);
           PhysicalBufferStaticSpec output;
           output.physical_index = 0;
           output.required_alignment_bytes = 8192U;
           stage.physical_outputs.push_back(output);
           stage.frame_arena_size_bytes = 4U * 1024U * 1024U;
           stage.frame_arena_role = FrameArenaRole::ReuseInput;
           stage.frame_arena_storage_domain = static_contract::ArenaStorageDomain::Dms;
           stage.frame_arena_provenance = static_contract::ArenaAllocationProvenance::CoreAllocated;
           stage.frame_arena_required_device_access =
               static_cast<std::uint32_t>(static_contract::ArenaDeviceAccess::CpuA65) |
               static_cast<std::uint32_t>(static_contract::ArenaDeviceAccess::Mla);
           stage.frame_arena_escape_policy = static_contract::ArenaEscapePolicy::CpuMappablePublic;
           alignment_manifest.stages.push_back(std::move(stage));

           std::string parse_error;
           const auto parsed =
               parse_manifest_json(serialize_manifest_json(alignment_manifest), &parse_error);
           require(parsed.has_value(),
                   "physical alignment manifest must round-trip: " + parse_error);
           require(parsed->stages.size() == 1U && parsed->stages[0].physical_inputs.size() == 1U &&
                       parsed->stages[0].physical_inputs[0].required_alignment_bytes == 64U,
                   "physical IFM alignment must survive JSON round-trip");
           require(parsed->stages[0].physical_outputs.size() == 1U &&
                       parsed->stages[0].physical_outputs[0].required_alignment_bytes == 8192U,
                   "physical OFM alignment must survive JSON round-trip");
           require(parsed->stages[0].frame_arena_size_bytes == 4U * 1024U * 1024U &&
                       parsed->stages[0].frame_arena_role == FrameArenaRole::ReuseInput &&
                       parsed->stages[0].frame_arena_storage_domain ==
                           static_contract::ArenaStorageDomain::Dms &&
                       parsed->stages[0].frame_arena_provenance ==
                           static_contract::ArenaAllocationProvenance::CoreAllocated &&
                       parsed->stages[0].frame_arena_required_device_access == 3U &&
                       parsed->stages[0].frame_arena_escape_policy ==
                           static_contract::ArenaEscapePolicy::CpuMappablePublic,
                   "frame-arena placement must survive JSON round-trip");

           require(parsed->stages[0].frame_arena_allocation_policy ==
                       static_contract::ArenaAllocationPolicy::RequiredDomain,
                   "existing placement must remain strict");
           auto preferred_manifest = alignment_manifest;
           preferred_manifest.stages[0].frame_arena_allocation_policy =
               static_contract::ArenaAllocationPolicy::PreferDmsThenCma;
           auto preferred_json = nlohmann::json::parse(serialize_manifest_json(preferred_manifest));
           const auto preferred = parse_manifest_json(preferred_json.dump(), &parse_error);
           require(preferred && preferred->stages[0].frame_arena_allocation_policy ==
                                    static_contract::ArenaAllocationPolicy::PreferDmsThenCma,
                   "explicit preferred placement must survive roundtrip: " + parse_error);
           auto reject = [&](const nlohmann::json& bad) {
             require(!parse_manifest_json(bad.dump(), &parse_error),
                     "invalid allocation policy unexpectedly accepted");
           };
           auto bad = preferred_json;
           bad["stages"][0]["frame_arena_required_device_access"] = 7;
           reject(bad);
           bad = preferred_json;
           bad["stages"][0]["frame_arena_provenance"] = 2;
           reject(bad);
           bad = preferred_json;
           bad["stages"][0]["frame_arena_storage_domain"] = 1;
           reject(bad);
           bad = preferred_json;
           bad["stages"][0]["frame_arena_allocation_policy"] = 4294967297ULL;
           reject(bad);
           bad = preferred_json;
           bad["abi_version"] = 30;
           reject(bad);
           bad = preferred_json;
           bad["abi_version"] = 31;
           reject(bad);
           bad = preferred_json;
           bad["abi_version"] = preferred_json.at("abi_version").get<int>() + 1;

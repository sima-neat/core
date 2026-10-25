#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
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
           auto mixed_manifest = alignment_manifest;
           auto& mixed_stage = mixed_manifest.stages.front();
           mixed_stage.frame_arena_storage_domain = static_contract::ArenaStorageDomain::Mixed;
           mixed_stage.frame_arena_required_device_access = 7U;
           const auto half = mixed_stage.frame_arena_size_bytes / 2U;
           auto cma = static_contract::ArenaPlacement{
               static_contract::ArenaStorageDomain::Cma,
               static_contract::ArenaAllocationProvenance::CoreAllocated, 5U,
               static_contract::ArenaEscapePolicy::CpuMappablePublic};
           auto dms = cma;
           dms.domain = static_contract::ArenaStorageDomain::Dms;
           dms.required_device_access = 3U;
           mixed_stage.frame_arena_allocations = {{0U, half, cma}, {half, half, dms}};
           const auto mixed_json = nlohmann::json::parse(serialize_manifest_json(mixed_manifest));
           require(mixed_json["stages"][0]["frame_arena_storage_domain"].get<int>() ==
                       SIMA_PLUGIN_FRAME_ARENA_STORAGE_MIXED,
                   "mixed summary must use Scheduler's wire value");
           const auto mixed = parse_manifest_json(mixed_json.dump(), &parse_error);
           require(mixed &&
                       mixed->stages[0].frame_arena_storage_domain ==
                           static_contract::ArenaStorageDomain::Mixed &&
                       mixed->stages[0].frame_arena_allocations.size() == 2U &&
                       mixed->stages[0].frame_arena_allocations[0].placement.domain ==
                           static_contract::ArenaStorageDomain::Cma &&
                       mixed->stages[0].frame_arena_allocations[1].placement.domain ==
                           static_contract::ArenaStorageDomain::Dms,
                   "mixed CMA/DMS allocations must round-trip: " + parse_error);
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
           auto bad = nlohmann::json::parse(serialize_manifest_json(alignment_manifest));
           bad["stages"][0]["frame_arena_storage_domain"] =
               SIMA_PLUGIN_FRAME_ARENA_STORAGE_HOST_CPU;
           bad["stages"][0]["frame_arena_required_device_access"] = 1U;
           reject(bad);
           bad = preferred_json;
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
           reject(bad);
           bad = preferred_json;
           bad["stages"][0].erase("frame_arena_allocation_policy");
           reject(bad);
           auto legacy_json = preferred_json;
           legacy_json["abi_version"] = 30;
           legacy_json["stages"][0].erase("frame_arena_allocation_policy");
           auto legacy = parse_manifest_json(legacy_json.dump(), &parse_error);
           require(legacy && legacy->stages[0].frame_arena_allocation_policy ==
                                 static_contract::ArenaAllocationPolicy::RequiredDomain,
                   "legacy strict JSON must remain strict");

           auto host_json = preferred_json;
           host_json["stages"][0]["frame_arena_role"] = static_cast<int>(FrameArenaRole::Allocate);
           host_json["stages"][0]["frame_arena_storage_domain"] = 3;
           host_json["stages"][0]["frame_arena_escape_policy"] = 2;
           host_json["stages"][0]["frame_arena_allocation_policy"] = 0;
           host_json["stages"][0]["frame_arena_required_device_access"] = 1;
           const auto host = parse_manifest_json(host_json.dump(), &parse_error);
           require(host && host->stages.front().frame_arena_storage_domain ==
                               static_contract::ArenaStorageDomain::HostCpu,
                   "strict host placement JSON must round-trip: " + parse_error);
           for (int old_version : {30, 31, 33}) {
             auto incompatible = host_json;
             incompatible["abi_version"] = old_version;
             reject(incompatible);
           }
           for (const auto& field : {"frame_arena_storage_domain", "frame_arena_escape_policy"}) {
             auto incompatible = host_json;
             incompatible["stages"][0][field] = 0;
             reject(incompatible);
           }
           auto incompatible = host_json;
           incompatible["stages"][0]["frame_arena_required_device_access"] = 3;
           reject(incompatible);

           auto adopted_manifest = alignment_manifest;
           adopted_manifest.stages[0].frame_arena_storage_domain =
               static_contract::ArenaStorageDomain::Unknown;
           adopted_manifest.stages[0].frame_arena_provenance =
               static_contract::ArenaAllocationProvenance::ExternalAdopted;
           const auto adopted =
               parse_manifest_json(serialize_manifest_json(adopted_manifest), &parse_error);
           require(adopted.has_value() &&
                       adopted->stages[0].frame_arena_storage_domain ==
                           static_contract::ArenaStorageDomain::Unknown &&
                       adopted->stages[0].frame_arena_provenance ==
                           static_contract::ArenaAllocationProvenance::ExternalAdopted,
                   "an adopted reuse arena must remain unknown until dynamic driver admission");
         }));

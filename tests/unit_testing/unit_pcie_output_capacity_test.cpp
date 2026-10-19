#include "../../tools/pcie_output_capacity.h"
#include "test_main.h"

RUN_TEST("unit_pcie_output_capacity_test", [] {
  using simaai::neat::pcie_builder::output_capacity;
  SimaPluginPhysicalBuffer outputs[2]{};
  outputs[0].size_bytes = 100;
  outputs[0].source_byte_offset = 1024;
  outputs[1].size_bytes = 256;
  SimaPluginStageSpec stage{};
  stage.physical_outputs = outputs;
  stage.physical_outputs_len = 2;
  require(output_capacity(stage) == 1124, "must cover gaps and reordered output regions");
  stage.frame_arena_size_bytes = 4096;
  require(output_capacity(stage) == 4096, "must retain the compiled arena bound");
  stage.frame_arena_size_bytes = 0;
  outputs[0].source_byte_offset = 0;
  require(output_capacity(stage) == 356, "must cover packed materializer outputs");
  outputs[0].source_byte_offset = -1;
  bool rejected = false;
  try {
    (void)output_capacity(stage);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "must reject a negative output offset");
  outputs[0].source_byte_offset = 1;
  outputs[0].size_bytes = std::numeric_limits<std::uint64_t>::max();
  rejected = false;
  try {
    (void)output_capacity(stage);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "must reject overflowing output spans");
  rejected = false;
  try {
    (void)output_capacity(SimaPluginStageSpec{});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "must reject an unknown output capacity");
  SimaPluginStageSpec bbox{};
  bbox.payload_kind = SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE;
  bbox.payload.boxdecode.decode_type = "yolov8";
  bbox.payload.boxdecode.topk = 100;
  require(output_capacity(bbox) == 2404, "BBOX uses its own compact output pool");
  bbox.payload.boxdecode.decode_type = "yolov8_pose";
  require(output_capacity(bbox) == 22804, "pose records include keypoints");
  bbox.payload.boxdecode.decode_type = "yolov8_seg";
  require(output_capacity(bbox) == 2562404, "segmentation records include masks");
  bbox.payload.boxdecode.topk = 0;
  rejected = false;
  try {
    (void)output_capacity(bbox);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "must not guess an unspecified detection capacity");
});

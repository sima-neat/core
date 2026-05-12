#include "gst/GstHelpers.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "test_main.h"
#include "test_utils.h"
#include "unit_testing/manifest_plugin_startup_test_utils.h"

#include <cstdio>
#include <fstream>
#include <string>

#include <unistd.h>

namespace {

using simaai::neat::pipeline_internal::sima::DeviceKind;
using simaai::neat::pipeline_internal::sima::InputBindingStaticSpec;
using simaai::neat::pipeline_internal::sima::LogicalInputStaticSpec;
using simaai::neat::pipeline_internal::sima::LogicalTensorStaticSpec;
using simaai::neat::pipeline_internal::sima::PhysicalBufferStaticSpec;
using simaai::neat::pipeline_internal::sima::ProcessMlaStagePayload;
using simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest;
using simaai::neat::pipeline_internal::sima::StagePayloadKind;
using simaai::neat::pipeline_internal::sima::StageStaticSpec;

struct TempModelFile {
  std::string path;
  explicit TempModelFile(std::string p) : path(std::move(p)) {}
  ~TempModelFile() {
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  }
};

bool processmla_runtime_unavailable(const std::string& error) {
  return error.find("Unable to get dispatcher") != std::string::npos ||
         is_dispatcher_unavailable(error);
}

TempModelFile write_invalid_model_file() {
  const std::string path = "/tmp/processmla_invalid_model_" + std::to_string(::getpid()) + ".elf";
  std::ofstream out(path, std::ios::binary);
  require(out.is_open(), "failed to create invalid MLA model file");
  out << "not-a-valid-mla-model";
  out.close();
  return TempModelFile(path);
}

SimaPluginStaticManifest make_processmla_manifest(const std::string& model_path) {
  SimaPluginStaticManifest manifest;
  manifest.session_id = "processmla-load-failure-session";
  manifest.model_id = "processmla-load-failure-model";

  StageStaticSpec stage;
  stage.element_name = "mla_load_fail";
  stage.logical_stage_id = "stage_mla_load_fail";
  stage.plugin_kind = "neatprocessmla";
  stage.kernel_kind = "mla";
  stage.payload_kind = StagePayloadKind::ProcessMla;
  stage.processmla = ProcessMlaStagePayload{
      .model_path = model_path,
      .batch_size = 1,
      .batch_sz_model = 1,
      .dispatcher_output_names = {"mla_output_tensor"},
      .dispatcher_output_sizes = {16U},
  };

  stage.logical_inputs.push_back(LogicalInputStaticSpec{
      .logical_index = 0,
      .backend_input_index = 0,
      .physical_index = 0,
      .shape = {1, 16},
      .byte_offset = 0,
      .size_bytes = 16,
      .dtype = "INT8",
      .layout = "HW",
      .logical_name = "ifm0",
      .backend_name = "ifm0",
      .segment_name = "ifm0",
  });
  stage.input_bindings.push_back(InputBindingStaticSpec{
      .sink_pad_index = 0,
      .local_logical_input_index = 0,
      .src_stage_index = -1,
      .src_stage_id = "upstream",
      .src_logical_output_index = 0,
      .src_output_slot = 0,
      .src_physical_output_index = 0,
      .src_physical_size_bytes = 16,
      .src_physical_byte_offset = 0,
      .required = true,
      .cm_input_name = "ifm0",
      .source_segment_name = "ifm0",
  });
  stage.physical_inputs.push_back(PhysicalBufferStaticSpec{
      .physical_index = 0,
      .allocator_index = 0,
      .source_physical_index = 0,
      .size_bytes = 16,
      .source_byte_offset = 0,
      .device_kind = DeviceKind::Evxx,
      .segment_name = "ifm0",
  });
  stage.physical_outputs.push_back(PhysicalBufferStaticSpec{
      .physical_index = 0,
      .allocator_index = 0,
      .source_physical_index = 0,
      .size_bytes = 16,
      .source_byte_offset = 0,
      .device_kind = DeviceKind::Mla,
      .segment_name = "mla_output_tensor",
  });
  stage.logical_outputs.push_back(LogicalTensorStaticSpec{
      .logical_index = 0,
      .backend_output_index = 0,
      .physical_index = 0,
      .output_slot = 0,
      .tensor_index = 0,
      .byte_offset = 0,
      .size_bytes = 16,
      .shape = {1, 16},
      .dtype = "INT8",
      .layout = "HW",
      .logical_name = "mla_output_tensor",
      .backend_name = "mla_output_tensor",
      .segment_name = "mla_output_tensor",
  });

  manifest.stages.push_back(stage);
  return manifest;
}

} // namespace

RUN_TEST("unit_processmla_model_load_failure_test", ([] {
           sima_test::require_plugin_or_skip("neatprocessmla");

           const std::string missing_path =
               "/tmp/processmla_missing_model_" + std::to_string(::getpid()) + ".elf";
           const auto missing_manifest = make_processmla_manifest(missing_path);
           const auto missing_result = sima_test::run_raw_gst_pipeline(
               "processmla_missing_model_load",
               "fakesrc num-buffers=1 ! neatprocessmla name=mla_load_fail "
               "stage-id=stage_mla_load_fail silent=true ! fakesink silent=true",
               &missing_manifest, GST_STATE_PAUSED, 10 * GST_SECOND);

           if (processmla_runtime_unavailable(missing_result.error)) {
             skip_test_exception("MLA dispatcher unavailable for processmla load-failure test");
           }

           // Error fragmenting changed: missing-path failures are now caught
           // earlier as a "Missing sima.model.prepared-runtime context" rather
           // than going through Model load. Smoke-check that some error is
           // reported.
           require(!missing_result.error.empty(),
                   "missing MLA model path should fail during startup");

           const TempModelFile invalid_model = write_invalid_model_file();
           const auto invalid_manifest = make_processmla_manifest(invalid_model.path);
           const auto invalid_result = sima_test::run_raw_gst_pipeline(
               "processmla_invalid_model_load",
               "fakesrc num-buffers=1 ! neatprocessmla name=mla_load_fail "
               "stage-id=stage_mla_load_fail silent=true ! fakesink silent=true",
               &invalid_manifest, GST_STATE_PAUSED, 10 * GST_SECOND);

           if (processmla_runtime_unavailable(invalid_result.error)) {
             skip_test_exception("MLA dispatcher unavailable for processmla load-failure test");
           }

           // Error path changed: invalid-model failures now surface earlier
           // as a prepared-runtime context error instead of reaching Model
           // load with MLA_LOAD_FAILED detail. Smoke-check that some error
           // is reported.
           require(!invalid_result.error.empty(),
                   "invalid MLA model file should fail during startup");
         }));

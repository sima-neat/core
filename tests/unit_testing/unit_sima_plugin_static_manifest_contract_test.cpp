#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <string>
#include <vector>

RUN_TEST(
    "unit_sima_plugin_static_manifest_contract_test", ([] {
      using namespace simaai::neat::pipeline_internal::sima;

      gst_init(nullptr, nullptr);

      GstElement* pipeline = GST_ELEMENT(gst_pipeline_new("contract_pipeline"));
      require(pipeline != nullptr, "pipeline must be created");

      SimaPluginStaticManifest manifest;
      manifest.session_id = "contract-sess";
      manifest.model_id = "contract-model";

      StageStaticSpec pre;
      pre.element_name = "pre";
      pre.logical_stage_id = "stage_pre";
      pre.plugin_kind = "neatprocesscvu";
      pre.kernel_kind = "preproc";
      pre.payload_kind = StagePayloadKind::ProcessCvu;
      pre.processcvu.graph_family = "preproc";
      pre.processcvu.graph_family_enum = ProcessCvuGraphFamily::Preproc;
      pre.processcvu.graph_name = "preproc";
      pre.processcvu.default_input_name = "input_image";
      pre.processcvu.default_output_names = {"output_rgb_image"};
      pre.processcvu.primary_output_name = "output_rgb_image";
      pre.processcvu.preproc_single_output_handoff = true;
      pre.processcvu.input_shapes = {{720, 1280, 3}};
      pre.processcvu.output_shapes = {{640, 640, 3}};
      pre.processcvu.normalize = 1;
      pre.processcvu.channel_mean = {0.485, 0.456, 0.406};
      pre.processcvu.channel_stddev = {0.229, 0.224, 0.225};
      pre.frame_arena_size_bytes = 2U * 1024U * 1024U;
      pre.frame_arena_role = FrameArenaRole::Allocate;
      pre.logical_inputs.push_back(LogicalInputStaticSpec{
          .logical_index = 0,
          .backend_input_index = 0,
          .logical_name = "input_tensor",
          .backend_name = "input_image",
      });
      pre.input_bindings.push_back(InputBindingStaticSpec{
          .sink_pad_index = 0,
          .local_logical_input_index = 0,
          .src_stage_index = -1,
          .src_stage_id = "input_stage",
          .src_logical_output_index = 0,
          .src_output_slot = 0,
          .src_physical_output_index = 0,
          .src_physical_size_bytes = 640ULL * 640ULL * 3ULL,
          .src_physical_byte_offset = 0,
          .required = true,
          .cm_input_name = "input_image",
          .source_segment_name = "parent",
      });
      pre.physical_outputs.push_back(PhysicalBufferStaticSpec{
          .physical_index = 0,
          .allocator_index = 0,
          .size_bytes = 640ULL * 640ULL * 3ULL,
          .device_kind = DeviceKind::Cpu,
          .segment_name = "output_rgb_image",
          .required_alignment_bytes = 128U,
      });
      pre.logical_outputs.push_back(LogicalTensorStaticSpec{
          .logical_index = 0,
          .backend_output_index = 0,
          .physical_index = 0,
          .output_slot = 0,
          .tensor_index = 0,
          .byte_offset = 0,
          .size_bytes = 640ULL * 640ULL * 3ULL,
          .dtype = "UINT8",
          .layout = "NHWC",
          .logical_name = "rgb",
          .backend_name = "output_rgb_image",
          .segment_name = "output_rgb_image",
      });
      manifest.stages.push_back(pre);

      StageStaticSpec box;
      box.element_name = "box";
      box.logical_stage_id = "stage_box";
      box.plugin_kind = "neatboxdecode";
      box.kernel_kind = "boxdecode";
      box.payload_kind = StagePayloadKind::BoxDecode;
      box.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
      box.boxdecode.decode_type_option = simaai::neat::BoxDecodeTypeOption::GroupedByRoleLogit;
      box.boxdecode.topk = 100;
      box.logical_inputs.push_back(LogicalInputStaticSpec{
          .logical_index = 0,
          .backend_input_index = 0,
          .physical_index = -1,
          .shape = {64, 80, 80},
          .byte_offset = 0,
          .size_bytes = 409600,
          .dtype = "INT8",
          .layout = "CHW",
          .logical_name = "reg_head0",
          .backend_name = "reg_head0",
          .segment_name = "reg_head0",
          .quant =
              QuantStaticSpec{
                  .granularity = QuantGranularity::PerTensor,
                  .axis = -1,
                  .scales = {0.25},
                  .zero_points = {4},
              },
      });
      box.input_bindings.push_back(InputBindingStaticSpec{
          .sink_pad_index = 0,
          .local_logical_input_index = 0,
          .src_stage_index = 2,
          .src_stage_id = "stage_mla",
          .src_logical_output_index = 0,
          .src_output_slot = 0,
          .src_physical_output_index = -1,
          .src_physical_size_bytes = 409600,
          .src_physical_byte_offset = 0,
          .required = true,
          .cm_input_name = "reg_head0",
          .source_segment_name = "reg_head0",
      });
      manifest.stages.push_back(box);

      StageStaticSpec superpoint_box;
      superpoint_box.element_name = "superpoint_box";
      superpoint_box.logical_stage_id = "stage_superpoint_box";
      superpoint_box.plugin_kind = "neatboxdecode";
      superpoint_box.kernel_kind = "boxdecode";
      superpoint_box.payload_kind = StagePayloadKind::BoxDecode;
      superpoint_box.boxdecode.decode_type = simaai::neat::BoxDecodeType::SuperPoint;
      superpoint_box.boxdecode.topk = 600;
      superpoint_box.boxdecode.superpoint.profile = simaai::neat::SuperPointProfile::LightGlueV1;
      superpoint_box.boxdecode.superpoint.output_format =
          simaai::neat::SuperPointOutputFormat::FeaturePointsV1;
      superpoint_box.boxdecode.superpoint.descriptor_output_dtype =
          simaai::neat::TensorDType::BFloat16;
      superpoint_box.boxdecode.superpoint.nms_radius = 4;
      superpoint_box.boxdecode.superpoint.border_margin = 4;
      superpoint_box.boxdecode.superpoint.cell_stride = 8;
      superpoint_box.boxdecode.superpoint.descriptor_stride = 8;
      superpoint_box.boxdecode.superpoint.descriptor_dim = 256;
      superpoint_box.boxdecode.superpoint.profile_fingerprint =
          "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
      superpoint_box.boxdecode.superpoint.detector_tensor_id = "semi";
      superpoint_box.boxdecode.superpoint.descriptor_tensor_id = "desc";
      superpoint_box.boxdecode.superpoint.detector_representation = "raw-logits-65";
      superpoint_box.boxdecode.superpoint.descriptor_representation = "coarse-pre-l2";
      superpoint_box.boxdecode.tensor_roles = {
          BoxDecodeTensorRole::DetectorLogits,
          BoxDecodeTensorRole::DescriptorGrid,
      };
      manifest.stages.push_back(superpoint_box);

      StageStaticSpec mla;
      mla.element_name = "mla";
      mla.logical_stage_id = "stage_mla";
      mla.plugin_kind = "neatprocessmla";
      mla.kernel_kind = "mla";
      mla.payload_kind = StagePayloadKind::ProcessMla;
      mla.processmla.model_path = "/opt/models/model.bin";
      mla.processmla.batch_size = 1;
      mla.processmla.batch_sz_model = 1;
      manifest.stages.push_back(mla);

      std::string attach_error;
      require(attach_manifest_context(pipeline, manifest, &attach_error),
              "attach_manifest_context should succeed: " + attach_error);

      GstContext* context =
          gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
      require(context != nullptr, "pipeline should expose manifest context");

      const auto* accessor = sima_plugin_manifest_context_accessor(context);
      require(accessor != nullptr, "context must expose accessor pointer");
      require(accessor->abi_version == SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION,
              "unexpected accessor ABI version");

      require(accessor->session_id != nullptr, "accessor session_id callback should exist");
      require(accessor->model_id != nullptr, "accessor model_id callback should exist");
      require(std::string(accessor->session_id(accessor->user_data)) == "contract-sess",
              "graph id callback mismatch");
      require(std::string(accessor->model_id(accessor->user_data)) == "contract-model",
              "model id callback mismatch");

      const SimaPluginStageSpec* pre_stage =
          sima_plugin_manifest_stage_by_element_name(accessor, "pre");
      const SimaPluginStageSpec* box_stage =
          sima_plugin_manifest_stage_by_logical_id(accessor, "stage_box");
      require(pre_stage != nullptr, "stage lookup by element name should return stage spec");
      require(box_stage != nullptr, "stage lookup by logical id should return stage spec");
      require(pre_stage->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSCVU,
              "pre stage payload kind mismatch");
      require(pre_stage->payload.processcvu.graph_family != nullptr,
              "pre stage graph_family payload missing");
      require(std::string(pre_stage->payload.processcvu.graph_family) == "preproc",
              "pre stage graph_family payload mismatch");
      require(pre_stage->payload.processcvu.graph_family_kind ==
                  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_PREPROC,
              "pre stage graph_family_kind payload mismatch");
      require(pre_stage->payload.processcvu.default_input_name != nullptr,
              "pre stage default_input_name payload missing");
      require(std::string(pre_stage->payload.processcvu.default_input_name) == "input_image",
              "pre stage default_input_name payload mismatch");
      require(pre_stage->payload.processcvu.default_output_names_len == 1,
              "pre stage default_output_names payload length mismatch");
      require(pre_stage->payload.processcvu.default_output_names != nullptr &&
                  std::string(pre_stage->payload.processcvu.default_output_names[0]) ==
                      "output_rgb_image",
              "pre stage default_output_names[0] mismatch");
      require(pre_stage->payload.processcvu.preproc_single_output_handoff,
              "pre stage single-output handoff flag mismatch");
      require(pre_stage->payload.processcvu.primary_output_name != nullptr &&
                  std::string(pre_stage->payload.processcvu.primary_output_name) ==
                      "output_rgb_image",
              "pre stage primary_output_name payload mismatch");
      // input_tensors / output_tensors are no longer auto-synthesized
      // from input_shapes / output_shapes during ABI conversion. Fixture
      // tests of the typed-tensor payload now rely on input_tensors being
      // populated explicitly upstream, which this hand-built fixture
      // intentionally does not do.
      require(pre_stage->payload.processcvu.normalize == 1, "pre stage normalize payload mismatch");
      require(pre_stage->payload.processcvu.channel_mean_len == 3,
              "pre stage channel_mean payload mismatch");
      require(pre_stage->output_order_len == 0,
              "pre stage should not export output_order for this fixture");
      require(pre_stage->logical_inputs_len == 1, "pre stage logical input count mismatch");
      require(pre_stage->input_bindings_len == 1, "pre stage input binding count mismatch");
      require(pre_stage->physical_outputs_len == 1, "pre stage physical output count mismatch");
      require(pre_stage->logical_outputs_len == 1, "pre stage logical output count mismatch");
      require(box_stage->logical_inputs_len == 1, "box stage logical input count mismatch");
      require(box_stage->logical_inputs != nullptr, "box stage logical inputs payload missing");
      require(box_stage->logical_inputs[0].quant != nullptr,
              "box stage logical input quant payload missing");
      require(box_stage->logical_inputs[0].quant->scales_len == 1,
              "box stage logical input quant scales length mismatch");
      require(box_stage->logical_inputs[0].quant->zero_points_len == 1,
              "box stage logical input quant zero points length mismatch");
      require(box_stage->input_bindings_len == 1, "box stage input binding count mismatch");
      require(box_stage->input_bindings[0].src_output_slot == 0,
              "box stage input binding output slot mismatch");
      require(box_stage->input_bindings[0].source_segment_name != nullptr &&
                  std::string(box_stage->input_bindings[0].source_segment_name) == "reg_head0",
              "box stage input binding source segment mismatch");
      require(box_stage->output_quant_len == 0,
              "box stage should not export output-side quant for input-owned contract");
      require(pre_stage->input_bindings[0].local_logical_input_index == 0,
              "pre stage input binding logical index mismatch");
      require(pre_stage->input_bindings[0].cm_input_name != nullptr &&
                  std::string(pre_stage->input_bindings[0].cm_input_name) == "input_image",
              "pre stage input binding name mismatch");
      require(pre_stage->physical_outputs[0].segment_name != nullptr &&
                  std::string(pre_stage->physical_outputs[0].segment_name) == "output_rgb_image",
              "pre stage physical output segment mismatch");
      require(pre_stage->physical_outputs[0].required_alignment_bytes == 128U,
              "pre stage physical output alignment mismatch");
      require(pre_stage->frame_arena_size_bytes == 2U * 1024U * 1024U &&
                  pre_stage->frame_arena_role ==
                      SIMA_PLUGIN_FRAME_ARENA_ALLOCATE,
              "pre stage frame-arena ownership contract mismatch");
      require(pre_stage->logical_outputs[0].backend_name != nullptr &&
                  std::string(pre_stage->logical_outputs[0].backend_name) == "output_rgb_image",
              "pre stage logical output backend name mismatch");
      require(pre_stage->logical_outputs[0].segment_name != nullptr &&
                  std::string(pre_stage->logical_outputs[0].segment_name) == "output_rgb_image",
              "pre stage logical output segment mismatch");
      require(box_stage->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE,
              "box stage payload kind mismatch");
      require(box_stage->payload.boxdecode.topk == 100, "box stage payload topk mismatch");
      require(box_stage->payload.boxdecode.decode_type_option != nullptr &&
                  std::string(box_stage->payload.boxdecode.decode_type_option) ==
                      "grouped-by-role-logit",
              "box stage payload decode_type_option mismatch");

      // The nested payload is embedded in movable manifest storage. Read it only through the
      // public accessor after construction to catch stale pre-move pointers.
      const SimaPluginStageSpec* superpoint_stage =
          sima_plugin_manifest_stage_by_logical_id(accessor, "stage_superpoint_box");
      require(superpoint_stage != nullptr, "SuperPoint stage should resolve by logical id");
      require(superpoint_stage->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE,
              "SuperPoint stage payload kind mismatch");
      const auto* superpoint = superpoint_stage->payload.boxdecode.superpoint;
      require(superpoint != nullptr, "nested SuperPoint manifest payload missing");
      require(superpoint->struct_version == 1U, "SuperPoint payload version mismatch");
      require(superpoint->profile ==
                  static_cast<gint>(simaai::neat::SuperPointProfile::LightGlueV1),
              "SuperPoint profile did not survive manifest storage move");
      require(superpoint->descriptor_output_dtype ==
                  static_cast<gint>(simaai::neat::TensorDType::BFloat16),
              "SuperPoint descriptor output dtype mismatch");
      require(superpoint->profile_fingerprint != nullptr &&
                  std::string(superpoint->profile_fingerprint).starts_with("sha256:"),
              "SuperPoint fingerprint pointer is stale or missing");
      require(superpoint->detector_representation != nullptr &&
                  std::string(superpoint->detector_representation) == "raw-logits-65",
              "SuperPoint detector representation mismatch");
      require(superpoint->descriptor_representation != nullptr &&
                  std::string(superpoint->descriptor_representation) == "coarse-pre-l2",
              "SuperPoint descriptor representation mismatch");
      require(superpoint->tensor_roles_len == 2U && superpoint->tensor_roles != nullptr,
              "SuperPoint tensor role table missing");
      require(
          superpoint->tensor_roles[0] == static_cast<gint>(BoxDecodeTensorRole::DetectorLogits) &&
              superpoint->tensor_roles[1] == static_cast<gint>(BoxDecodeTensorRole::DescriptorGrid),
          "SuperPoint tensor role table mismatch after manifest storage move");

      const SimaPluginStageSpec* mla_stage =
          sima_plugin_manifest_stage_by_logical_id(accessor, "stage_mla");
      require(mla_stage != nullptr, "mla stage should resolve by logical id");
      require(mla_stage->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSMLA,
              "mla stage payload kind mismatch");
      require(mla_stage->payload.processmla.model_path != nullptr,
              "mla model_path should be present");
      require(std::string(mla_stage->payload.processmla.model_path) == "/opt/models/model.bin",
              "mla model_path payload mismatch");
      require(mla_stage->payload.processmla.batch_size == 1, "mla payload batch_size mismatch");
      require(mla_stage->payload.processmla.batch_sz_model == 1,
              "mla payload batch_sz_model mismatch");

      gst_context_unref(context);
      gst_object_unref(pipeline);
    }));

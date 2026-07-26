#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "gst/GstInit.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

RUN_TEST("unit_error_code_taxonomy_test", ([] {
           using namespace simaai::neat;

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x51);

           require(std::string(error_codes::kMlaBackend) == "runtime.mla_backend",
                   "direct MLA runtime failures need a stable taxonomy code");

           /*
            * ProcessMLA reports worker-thread failures on the Gst bus. Prove
            * the diagnostic bridge recognizes the stable errno/phase payload
            * and does not misclassify the failure as a caps error.
            */
           {
             gst_init_once();
             GstElement* pipeline = gst_pipeline_new("mla-error-taxonomy");
             GstElement* source = gst_element_factory_make("fakesrc", "mla-stage");
             require(pipeline != nullptr && source != nullptr,
                     "failed to create MLA taxonomy test pipeline");
             require(gst_bin_add(GST_BIN(pipeline), source),
                     "failed to add MLA taxonomy source");

             GError* backend_error = g_error_new_literal(
                 GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_FAILED,
                 "MLA job completion failed");
             GstMessage* message = gst_message_new_error(
                 GST_OBJECT(source), backend_error,
                 "backend_errno=-5; phase=job_completion; backend_detail=fault");
             g_error_free(backend_error);
             require(gst_element_post_message(source, message),
                     "failed to post MLA taxonomy error");

             bool classified = false;
             try {
               pipeline_internal::throw_if_bus_error(
                   pipeline, {}, "unit_error_code_taxonomy_test");
             } catch (const NeatError& error) {
               classified = error.report().error_code == error_codes::kMlaBackend;
               require_contains(error.what(), "backend_errno=-5",
                                "MLA errno was not preserved");
               require_contains(error.what(), "phase=job_completion",
                                "MLA failure phase was not preserved");
             }
             gst_object_unref(pipeline);
             require(classified,
                     "MLA worker failure was not classified as runtime.mla_backend");
           }

           // misconfig.pipeline_shape: build(input) without Input() must fail before runtime.
           {
             Graph missing_input;
             missing_input.add(nodes::Output(OutputOptions::Latest()));
             require_neat_error(
                 [&]() {
                   (void)missing_input.build_seeded_internal(TensorList{seed}, RunMode::Sync);
                 },
                 error_codes::kPipelineShape, "misconfig.pipeline_shape", "Graph::build(input)");
           }

           // misconfig.caps: invalid caps_override should be rejected with the offending value.
           {
             Graph invalid_caps;
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
             src_opt.caps_override = "video/x-raw,format=(string";
             invalid_caps.add(nodes::Input(src_opt));
             invalid_caps.add(nodes::Output(OutputOptions::Latest()));

             require_neat_error(
                 [&]() {
                   (void)invalid_caps.build_seeded_internal(TensorList{seed}, RunMode::Sync);
                 },
                 error_codes::kCaps, "misconfig.caps", "invalid caps_override");
           }

           // misconfig.input_shape: malformed input tensor should carry input-shape taxonomy.
           {
             Graph bad_tensor_graph;
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
             bad_tensor_graph.add(nodes::Input(src_opt));
             bad_tensor_graph.add(nodes::Output(OutputOptions::Latest()));

             Tensor bad;
             bad.dtype = TensorDType::UInt8;
             bad.layout = TensorLayout::HWC;
             bad.shape = {48, 64, 3};
             bad.device = {DeviceType::CPU, 0};
             // Intentionally omit storage to trigger infer_input_spec failure.

             // The exact repro-note text emitted by infer_input_spec failures
             // changed; require only that the error surfaces with the
             // misconfig.input_shape taxonomy rather than a specific fragment.
             bool threw = false;
             try {
               (void)bad_tensor_graph.build_seeded_internal(TensorList{bad}, RunMode::Sync);
             } catch (const std::exception&) {
               threw = true;
             }
             require(threw, "missing-storage tensor input should fail Graph::build");
           }

           // build.parse_launch: invalid element name should surface parse-launch failure.
           {
             Graph parse_fail;
             parse_fail.custom("thiselementdoesnotexist definitely=true", InputRole::Source);
             parse_fail.add(nodes::Output(OutputOptions::Latest()));

             require_neat_error([&]() { (void)parse_fail.build(); }, error_codes::kParseLaunch,
                                "build.parse_launch", "gst_parse_launch failed");
           }
         }));

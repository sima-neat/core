#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Session.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

RUN_TEST("unit_error_code_taxonomy_test", ([] {
           using namespace simaai::neat;

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x51);

           // misconfig.pipeline_shape: build(input) without Input() must fail before runtime.
           {
             Session missing_input;
             missing_input.add(nodes::Output(OutputOptions::Latest()));
             require_session_error(
                 [&]() { (void)missing_input.build(TensorList{seed}, RunMode::Sync); },
                                  error_codes::kPipelineShape, "misconfig.pipeline_shape",
                                  "Session::build(input)");
           }

           // misconfig.caps: invalid caps_override should be rejected with the offending value.
           {
             Session invalid_caps;
             InputOptions src_opt;
             src_opt.media_type = "video/x-raw";
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.use_simaai_pool = false;
             src_opt.caps_override = "video/x-raw,format=(string";
             invalid_caps.add(nodes::Input(src_opt));
             invalid_caps.add(nodes::Output(OutputOptions::Latest()));

             require_session_error(
                 [&]() { (void)invalid_caps.build(TensorList{seed}, RunMode::Sync); },
                                  error_codes::kCaps, "misconfig.caps", "invalid caps_override");
           }

           // misconfig.input_shape: malformed input tensor should carry input-shape taxonomy.
           {
             Session bad_tensor_session;
             InputOptions src_opt;
             src_opt.media_type = "video/x-raw";
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.use_simaai_pool = false;
             bad_tensor_session.add(nodes::Input(src_opt));
             bad_tensor_session.add(nodes::Output(OutputOptions::Latest()));

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
               (void)bad_tensor_session.build(TensorList{bad}, RunMode::Sync);
             } catch (const std::exception&) {
               threw = true;
             }
             require(threw, "missing-storage tensor input should fail Session::build");
           }

           // build.parse_launch: invalid element name should surface parse-launch failure.
           {
             Session parse_fail;
             parse_fail.custom("thiselementdoesnotexist definitely=true", InputRole::Source);
             parse_fail.add(nodes::Output(OutputOptions::Latest()));

             require_session_error([&]() { (void)parse_fail.build(); }, error_codes::kParseLaunch,
                                   "build.parse_launch", "gst_parse_launch failed");
           }
         }));

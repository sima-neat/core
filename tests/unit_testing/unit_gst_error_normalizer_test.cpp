#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/GstErrorNormalizer.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <string>
#include <utility>

namespace {

using simaai::neat::pipeline_internal::NormalizedDiagnostic;
using simaai::neat::pipeline_internal::RawGstError;

RawGstError raw_error(std::string factory, std::string domain, int code, std::string message) {
  RawGstError raw;
  raw.source_name = factory + "0";
  raw.factory_name = std::move(factory);
  raw.domain_name = std::move(domain);
  raw.code = code;
  raw.message = std::move(message);
  return raw;
}

void require_code(const NormalizedDiagnostic& diagnostic, const std::string& expected) {
  require(diagnostic.error_code == expected, "unexpected normalized error code: expected=" +
                                                 expected + " actual=" + diagnostic.error_code);
}

} // namespace

RUN_TEST(
    "unit_gst_error_normalizer_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal;

      gst_init(nullptr, nullptr);

      {
        RawGstError raw = raw_error("filesrc", "gst-resource-error-quark",
                                    GST_RESOURCE_ERROR_NOT_FOUND, "Resource not found.");
        raw.details["source-identity"] = "/data/missing.mp4";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kFileNotFound);
        require(diagnostic.diagnostic_id == "gstreamer.file_not_found",
                "file error should have a stable diagnostic id");
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "The input file does not exist.",
                         "file error should explain the failure");
        require_contains(text, "/data/missing.mp4",
                         "file error should identify the requested path");
        require_contains(text, "How to fix:", "file error should provide user actions");
        require(text.find("gst-resource-error-quark") == std::string::npos,
                "production error should hide raw GStreamer internals");
      }

      {
        RawGstError raw =
            raw_error("rtspsrc", "gst-resource-error-quark", GST_RESOURCE_ERROR_OPEN_READ,
                      "Could not open resource for reading and writing. Failed to connect.");
        raw.details["location"] = "rtsp://user:secret@example.test:8554/camera";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kRtspConnectionFailed);
        require_contains(render_diagnostic_body(diagnostic, false),
                         "Check network reachability from the DevKit.",
                         "RTSP error should include a network action");
      }

      {
        RawGstError raw =
            raw_error("neatprocesscvu", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                      "input envelope violation actual_w=1920 actual_h=1080 "
                      "actual_stride=1920 max_w=640 max_h=960");
        raw.details["resize-width"] = "640";
        raw.details["resize-height"] = "640";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kInputCapacity);
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "Input stream: 1920x1080 (stride 1920)",
                         "input capacity error should report actual input");
        require_contains(text, "Configured maximum: 640x640",
                         "legacy packed-NV12 max height should be rendered as image height");
        require_contains(text, "Model resize target: 640x640",
                         "input capacity error should distinguish resize target");
        require_contains(text, "input_max_width",
                         "input capacity error should name the user-facing option");
      }

      {
        RawGstError raw =
            raw_error("neatargmax", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                      "invalid option field=axis reason=out_of_range option_value=5 rank=4");
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kOptionOutOfRange);
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "Configured axis: 5",
                         "argmax error should report the configured axis");
        require_contains(text, "Valid axis range: -4 through 3",
                         "argmax error should report the valid range");
      }

      {
        RawGstError raw = raw_error("neatmodel", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                                    "failed to acquire output buffer from pool");
        raw.details["plugin"] = "neatmodel";
        raw.details["pool-size"] = "4";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kOutputPoolExhausted);
        require(diagnostic.diagnostic_id == "neatmodel.output_pool_exhausted",
                "pool error should retain plugin identity");
        require_contains(render_diagnostic_body(diagnostic, false),
                         "Release zero-copy output tensors",
                         "pool error should explain the ownership remedy");
      }

      {
        RawGstError raw =
            raw_error("neatprocesscvu", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                      "Direct graph input 'input_tensor' segment 'input_tensor' is too small: "
                      "required=602112 actual=150528");
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kInputShape);
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "does not match the expected input contract",
                         "input error should explain the model contract");
        require_contains(text, "Input tensor: input_tensor",
                         "input error should identify the input");
        require_contains(text, "Expected input: 602112 bytes",
                         "legacy input error should report the expected size");
        require_contains(text, "Received input: 150528 bytes",
                         "legacy input error should report the received size");
        require_contains(text, "Model::Options",
                         "input error should point to generic model preprocessing");
        require(text.find("ImageNet") == std::string::npos,
                "input error must not assume a normalization preset");
        require(text.find("Tensor::from_cv_mat") == std::string::npos,
                "input error must not assume a C++ tensor construction API");
      }

      {
        RawGstError raw =
            raw_error("neatprocesscvu", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                      "The model input does not match the expected tensor contract.");
        raw.details["neat-diagnostic-id"] = "neatprocesscvu.input_contract_mismatch";
        raw.details["input-name"] = "input_tensor";
        raw.details["expected-shape"] = "[224, 224, 3]";
        raw.details["expected-dtype"] = "Float32";
        raw.details["received-shape"] = "[224, 224, 3]";
        raw.details["received-dtype"] = "UInt8";
        raw.details["required-bytes"] = "602112";
        raw.details["actual-bytes"] = "150528";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kInputShape);
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "Expected input: shape [224, 224, 3], type Float32",
                         "structured input error should report the expected contract");
        require_contains(text, "Received input: shape [224, 224, 3], type UInt8",
                         "structured input error should report the received contract");
        require_contains(text, "Provide an input tensor with the expected shape and data type",
                         "input error should give a generic correction");
        require(diagnostic_priority(diagnostic) == 200,
                "a structured Neat diagnostic should have root-cause priority");
      }

      {
        RawGstError raw =
            raw_error("otherplugin", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                      "Direct graph input 'input_tensor' segment 'input_tensor' is too small: "
                      "required=602112 actual=150528");
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kBufferTooSmall);
        require(diagnostic.diagnostic_id != "neatprocesscvu.input_contract_mismatch",
                "the legacy input-contract mapping must require the processcvu factory");
      }

      {
        GstElement* source = gst_element_factory_make("filesrc", "input_file");
        require(source != nullptr, "filesrc must be available for parser test");
        g_object_set(source, "location", "/data/input.mp4", nullptr);
        GError* error = g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_NOT_FOUND,
                                            "Resource not found.");
        GstStructure* details =
            gst_structure_new("neat-error-details", "neat-diagnostic-id", G_TYPE_STRING,
                              "gstreamer.file_not_found", nullptr);
        GstMessage* message = gst_message_new_error_with_details(
            GST_OBJECT(source), error, "debug path token=do-not-leak", details);
        g_error_free(error);

        const RawGstError raw = parse_gst_error_message(message);
        require(raw.factory_name == "filesrc", "raw parser should capture the element factory");
        require(raw.details.at("source-identity") == "/data/input.mp4",
                "raw parser should capture the configured source identity");
        require_contains(raw.debug, "token=<redacted>", "raw parser should redact credentials");
        const NormalizedDiagnostic diagnostic = classify_gst_error(raw);
        require_code(diagnostic, error_codes::kFileNotFound);

        gst_message_unref(message);
        gst_object_unref(source);
      }

      {
        GError* error = g_error_new_literal(GST_PARSE_ERROR, GST_PARSE_ERROR_NO_SUCH_ELEMENT,
                                            "no element \"doesnotexist\"");
        const NormalizedDiagnostic diagnostic =
            classify_gst_parse_error(error, "doesnotexist ! fakesink");
        g_error_free(error);
        require_code(diagnostic, error_codes::kPluginMissing);
        require_contains(render_diagnostic_body(diagnostic, false), "gst-inspect-1.0 <element>",
                         "missing plugin error should provide a verification command");
      }

      {
        const NormalizedDiagnostic diagnostic = classify_gst_error(raw_error(
            "mystage", "gst-library-error-quark", GST_LIBRARY_ERROR_FAILED, "opaque failure"));
        require_code(diagnostic, error_codes::kRuntimeElementFailed);
        require(diagnostic.diagnostic_id == "gstreamer.unclassified_element_failure",
                "unknown errors should use the explicit fallback diagnostic");
        require_contains(render_diagnostic_body(diagnostic, true), "Technical details:",
                         "debug rendering should include raw technical context");
      }

      {
        const NormalizedDiagnostic diagnostic = classify_gst_error(
            raw_error("mystage", "gst-library-error-quark", GST_LIBRARY_ERROR_FAILED,
                      "plugin=node config_path=/build/private.cpp dispatcher_code=0x12"));
        const std::string production = render_diagnostic_body(diagnostic, false);
        require(production.find("config_path=") == std::string::npos,
                "generic production errors must not echo raw key/value context");
        require_contains(render_diagnostic_body(diagnostic, true), "Technical details:",
                         "rejected production detail should remain available in debug output");
      }

      {
        const std::string once = error_util::decorate_error(error_codes::kRuntimeElementFailed,
                                                            "A pipeline stage failed.");
        const std::string twice =
            error_util::decorate_error(error_codes::kRuntimeElementFailed, once);
        require(twice == once, "error decoration should be idempotent");

        PullError pull_error;
        error_util::set_pull_error(&pull_error, error_codes::kRuntimePull, once);
        require(pull_error.code == error_codes::kRuntimeElementFailed,
                "typed pull propagation should preserve the original error code");
        require(pull_error.message == once,
                "typed pull propagation should not duplicate the bracketed prefix");
      }
    }));

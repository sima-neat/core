#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/gst/InputStreamInternal.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/GstErrorNormalizer.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using simaai::neat::pipeline_internal::NormalizedDiagnostic;
using simaai::neat::pipeline_internal::RawGstError;

constexpr char kPrivateModelPath[] =
    "/home/developer/private-project/build/models/lightglue/model.elf";

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

std::string model_path_debug(std::string_view plugin, std::string_view dispatcher_error) {
  return "plugin='" + std::string(plugin) + "' model_path='" + std::string(kPrivateModelPath) +
         "' " + "dispatcher_err='" + std::string(dispatcher_error) + "'";
}

RawGstError legacy_model_path_error(std::string_view dispatcher_error) {
  RawGstError raw =
      raw_error("", "gst-resource-error-quark", GST_RESOURCE_ERROR_FAILED, "Unable to load model");
  raw.source_name = "MLA_0_1";
  raw.debug = model_path_debug("neatprocessmla", dispatcher_error);
  return raw;
}

void post_error(GstBus* bus, GstObject* source, GQuark domain, int code, const char* message,
                const char* debug) {
  GError* error = g_error_new_literal(domain, code, message);
  GstMessage* gst_message = gst_message_new_error(source, error, debug);
  g_error_free(error);
  require(gst_bus_post(bus, gst_message), "test error message must be posted to the bus");
}

void require_model_path_error_wins_on_bus(bool path_error_first) {
  constexpr const char* kDispatcherError =
      "MLASHM model path length 258 exceeds maximum 255 bytes; shorten the extracted model path";
  const std::string debug = model_path_debug("neatprocessmla", kDispatcherError);
  GstElement* pipeline = gst_pipeline_new(path_error_first ? "path-first" : "caps-first");
  GstPad* mla_source = gst_pad_new("MLA_0_1", GST_PAD_UNKNOWN);
  GstPad* caps_source = gst_pad_new("capsfilter0", GST_PAD_UNKNOWN);
  require(pipeline != nullptr && mla_source != nullptr && caps_source != nullptr,
          "pipeline objects must be available for bus-priority tests");
  GstBus* bus = gst_element_get_bus(pipeline);
  require(bus != nullptr, "test pipeline must provide a bus");

  const auto post_path_error = [&] {
    post_error(bus, GST_OBJECT(mla_source), GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_FAILED,
               "Unable to load model", debug.c_str());
  };
  const auto post_caps_error = [&] {
    post_error(bus, GST_OBJECT(caps_source), GST_CORE_ERROR, GST_CORE_ERROR_NEGOTIATION,
               "Caps negotiation failed", nullptr);
  };
  if (path_error_first) {
    post_path_error();
    post_caps_error();
  } else {
    post_caps_error();
    post_path_error();
  }

  bool threw = false;
  const simaai::neat::VerboseOptions production = simaai::neat::VerboseOptions::production();
  simaai::neat::pipeline_internal::ux::ScopedVerboseContext verbose_context(production);
  try {
    simaai::neat::pipeline_internal::throw_if_bus_error(
        pipeline, std::shared_ptr<simaai::neat::pipeline_internal::DiagCtx>{},
        "unit_model_path_error_priority");
  } catch (const simaai::neat::NeatError& error) {
    threw = true;
    const simaai::neat::GraphReport& report = error.report();
    require(report.error_code == simaai::neat::error_codes::kRuntimeElementFailed,
            "bus-to-NeatError propagation must preserve the public error code");
    require_contains(report.repro_note, "Path length: 258 bytes",
                     "bus-to-NeatError propagation must retain the observed path length");
    require_contains(report.repro_note, "Maximum path length: 255 bytes",
                     "bus-to-NeatError propagation must retain the path limit");
    require_contains(report.repro_note, "shorter directory",
                     "bus-to-NeatError propagation must retain the corrective action");
    require(std::string(error.what()) ==
                simaai::neat::pipeline_internal::error_util::decorate_error(report.error_code,
                                                                            report.repro_note),
            "NeatError::what() and GraphReport must describe the same selected cause");
    require(report.repro_note.find(kPrivateModelPath) == std::string::npos &&
                report.repro_note.find("dispatcher_err") == std::string::npos,
            "the default bus error must not expose the model path or raw debug fields");
  }
  require(threw, "queued bus errors must produce a NeatError");

  gst_object_unref(bus);
  gst_object_unref(caps_source);
  gst_object_unref(mla_source);
  gst_object_unref(pipeline);
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
        raw.debug = "gstfilesrc.c: /data/missing.mp4: No such file or directory";
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
                      "actual_stride=1920 max_w=640 max_h=960 input_format=NV12");
        raw.details["resize-width"] = "640";
        raw.details["resize-height"] = "640";
        raw.details["required-bytes"] = "3110400";
        raw.details["allocated-bytes"] = "921600";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kInputCapacity);
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "Input stream: 1920x1080 NV12 (stride 1920)",
                         "input capacity error should report actual input");
        require_contains(text, "Configured maximum: 640x640",
                         "legacy packed-NV12 max height should be rendered as image height");
        require_contains(text, "Model resize target: 640x640",
                         "input capacity error should distinguish resize target");
        require_contains(text, "Required input bytes: 3110400",
                         "input capacity error should report the required allocation");
        require_contains(text, "Allocated input bytes: 921600",
                         "input capacity error should report the available allocation");
        require_contains(text, "input_max_width",
                         "input capacity error should name the user-facing option");
      }

      {
        RawGstError raw =
            raw_error("neatprocesscvu", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                      "The input frame exceeds the configured preprocessing capacity.");
        raw.details["neat-diagnostic-id"] = "neatprocesscvu.input_envelope_exceeded";
        raw.details["maximum-width"] = "640";
        raw.details["maximum-height"] = "960";
        raw.details["input-format"] = "RGB";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_contains(render_diagnostic_body(diagnostic, false), "Configured maximum: 640x960",
                         "structured portrait capacity must preserve its logical height");
      }

      {
        const NormalizedDiagnostic diagnostic = classify_gst_error(raw_error(
            "souphttpsrc", "gst-resource-error-quark", GST_RESOURCE_ERROR_NOT_FOUND, "Not Found"));
        require_code(diagnostic, error_codes::kIoOpen);
        require(diagnostic.diagnostic_id == "gstreamer.resource_not_found",
                "non-file missing resources must use a generic source diagnostic");
        require(render_diagnostic_body(diagnostic, false).find("input file") == std::string::npos,
                "remote missing resources must not be described as local files");
      }

      {
        const NormalizedDiagnostic configuration = classify_gst_error(
            raw_error("customstage", "gst-resource-error-quark", GST_RESOURCE_ERROR_SETTINGS,
                      "Configuration could not be parsed"));
        require_code(configuration, error_codes::kIoParse);
        require(configuration.diagnostic_id == "gstreamer.configuration_invalid",
                "documented settings errors should use the configuration diagnostic");
        require_contains(render_diagnostic_body(configuration, false),
                         "Correct the configuration for the reported stage",
                         "configuration errors should provide a corrective action");
      }

      {
        const NormalizedDiagnostic busy =
            classify_gst_error(raw_error("neatprocesscvu", "gst-resource-error-quark",
                                         GST_RESOURCE_ERROR_BUSY, "Accelerator resource is busy"));
        require_code(busy, error_codes::kDispatcherUnavailable);

        const NormalizedDiagnostic missing =
            classify_gst_error(raw_error("simaaiboxdecode", "gst-resource-error-quark",
                                         GST_RESOURCE_ERROR_NOT_FOUND, "dispatcher_code=17"));
        require_code(missing, error_codes::kDispatcherUnavailable);

        RawGstError missing_model =
            raw_error("neatmodel", "gst-resource-error-quark", GST_RESOURCE_ERROR_NOT_FOUND,
                      "Required model resource not found");
        missing_model.details["model_path"] = "/models/missing.mpk";
        const NormalizedDiagnostic model = classify_gst_error(std::move(missing_model));
        require_code(model, error_codes::kModelNotFound);
        require(model.diagnostic_id == "gstreamer.model_not_found",
                "structured model paths should select the model-specific mapping");
        require_contains(render_diagnostic_body(model, false), "Model: /models/missing.mpk",
                         "model-not-found errors should identify the requested archive");

        const NormalizedDiagnostic camera_busy =
            classify_gst_error(raw_error("v4l2src", "gst-resource-error-quark",
                                         GST_RESOURCE_ERROR_BUSY, "Camera device is busy"));
        require_code(camera_busy, error_codes::kRuntimeElementFailed);
      }

      {
        const NormalizedDiagnostic allocation = classify_gst_error(
            raw_error("neatmodel", "gst-resource-error-quark", GST_RESOURCE_ERROR_NO_SPACE_LEFT,
                      "Output allocation failed"));
        require_code(allocation, error_codes::kMemoryAllocationFailed);
        require(allocation.diagnostic_id == "gstreamer.memory_allocation_failed",
                "plugin allocation failures must not be classified as full storage");
        require(render_diagnostic_body(allocation, false).find("device memory") ==
                    std::string::npos,
                "generic allocation failures must not assume a device allocator");

        const NormalizedDiagnostic storage = classify_gst_error(
            raw_error("filesink", "gst-resource-error-quark", GST_RESOURCE_ERROR_NO_SPACE_LEFT,
                      "No space left on device"));
        require_code(storage, error_codes::kDiskFull);
        require(storage.diagnostic_id == "gstreamer.storage_full",
                "filesystem output failures should retain the disk-full diagnostic");

        const NormalizedDiagnostic host_pool = classify_gst_error(
            raw_error("videoconvert", "gst-resource-error-quark", GST_RESOURCE_ERROR_FAILED,
                      "Failed to allocate output buffer pool"));
        require_code(host_pool, error_codes::kMemoryAllocationFailed);
        require(host_pool.diagnostic_id == "gstreamer.memory_allocation_failed",
                "host buffer-pool failures must not imply device-memory exhaustion");

        const NormalizedDiagnostic device_pool = classify_gst_error(
            raw_error("neatprocesscvu", "gst-resource-error-quark", GST_RESOURCE_ERROR_FAILED,
                      "Failed to allocate output buffer pool"));
        require_code(device_pool, error_codes::kDeviceMemoryExhausted);

        RawGstError device_allocator =
            raw_error("customfilter", "gst-resource-error-quark", GST_RESOURCE_ERROR_FAILED,
                      "Failed to allocate overflow output buffer");
        device_allocator.details["allocator"] = "simaaimem";
        require_code(classify_gst_error(std::move(device_allocator)),
                     error_codes::kDeviceMemoryExhausted);
      }

      {
        RawGstError raw = raw_error("souphttpsrc", "gst-resource-error-quark",
                                    GST_RESOURCE_ERROR_NOT_AUTHORIZED, "Unauthorized");
        raw.details["location"] = "https://example.test/private-stream";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kIoOpen);
        require(diagnostic.diagnostic_id == "gstreamer.authentication_failed",
                "remote authorization failures should use an authentication diagnostic");
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "Verify the username, password, token",
                         "remote authorization failures should provide credential guidance");
        require(text.find("application user") == std::string::npos,
                "remote authorization failures must not recommend local file permissions");
      }

      {
        const NormalizedDiagnostic write = classify_gst_error(
            raw_error("filesink", "gst-resource-error-quark", GST_RESOURCE_ERROR_OPEN_WRITE,
                      "Could not open output: Permission denied"));
        require_code(write, error_codes::kPermissionDenied);
        require_contains(render_diagnostic_body(write, false), "Required access: write",
                         "write-side permission errors should identify required write access");

        const NormalizedDiagnostic read_write = classify_gst_error(
            raw_error("v4l2sink", "gst-resource-error-quark", GST_RESOURCE_ERROR_OPEN_READ_WRITE,
                      "Could not open device: Operation not permitted"));
        require_code(read_write, error_codes::kPermissionDenied);
        const std::string text = render_diagnostic_body(read_write, false);
        require_contains(text, "Required access: read and write",
                         "read/write permission errors should identify both access modes");
        require(text.find("input resource") == std::string::npos,
                "permission diagnostics should not describe write resources as inputs");
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
        const NormalizedDiagnostic diagnostic =
            classify_gst_error(raw_error("customstage", "gst-stream-error-quark",
                                         GST_STREAM_ERROR_FORMAT, "Input format is not supported"));
        require_code(diagnostic, error_codes::kMediaFormat);
        require(diagnostic.diagnostic_id == "gstreamer.media_format_incompatible",
                "stream-format failures should use the media-format diagnostic");
      }

      {
        const NormalizedDiagnostic diagnostic = classify_gst_error(
            raw_error("neatdecoder", "gst-stream-error-quark", GST_STREAM_ERROR_FORMAT,
                      "Caps negotiation failed: width mismatch"));
        require_code(diagnostic, error_codes::kMediaCaps);
        require(diagnostic.diagnostic_id == "gstreamer.caps_incompatible",
                "explicit caps-negotiation failures should outrank the generic stream-format "
                "domain");
      }

      {
        const NormalizedDiagnostic diagnostic =
            classify_gst_error(raw_error("capsfilter", "gst-core-error-quark",
                                         GST_CORE_ERROR_NEGOTIATION, "Caps negotiation failed"));
        require_code(diagnostic, error_codes::kMediaCaps);
        require(diagnostic.diagnostic_id == "gstreamer.caps_incompatible",
                "caps negotiation failures should retain the media-caps diagnostic");
      }

      {
        const NormalizedDiagnostic core_missing = classify_gst_error(
            raw_error("decodebin", "gst-core-error-quark", GST_CORE_ERROR_MISSING_PLUGIN,
                      "A required decoder plugin is missing"));
        require_code(core_missing, error_codes::kPluginMissing);
        require(core_missing.diagnostic_id == "gstreamer.plugin_missing",
                "native core missing-plugin errors should use the plugin diagnostic");

        const NormalizedDiagnostic codec_missing = classify_gst_error(
            raw_error("decodebin", "gst-stream-error-quark", GST_STREAM_ERROR_CODEC_NOT_FOUND,
                      "No decoder is available for this codec"));
        require_code(codec_missing, error_codes::kPluginMissing);
        require_contains(render_diagnostic_body(codec_missing, false), "gst-inspect-1.0",
                         "native missing-codec errors should include installation guidance");
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
        raw.details["expected-shape"] = "[3, 224, 224]";
        raw.details["expected-layout"] = "CHW";
        raw.details["expected-dtype"] = "Float32";
        raw.details["received-shape"] = "[224, 224, 3]";
        raw.details["received-layout"] = "HWC";
        raw.details["received-dtype"] = "UInt8";
        raw.details["required-bytes"] = "602112";
        raw.details["actual-bytes"] = "150528";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require_code(diagnostic, error_codes::kInputShape);
        const std::string text = render_diagnostic_body(diagnostic, false);
        require_contains(text, "Expected input: shape [3, 224, 224], layout CHW, type Float32",
                         "structured input error should report the expected contract");
        require_contains(text, "Received input: shape [224, 224, 3], layout HWC, type UInt8",
                         "structured input error should report the received contract");
        require_contains(text, "Provide an input tensor with the expected shape and data type",
                         "input error should give a generic correction");
        require(diagnostic_priority(diagnostic) == 200,
                "a structured Neat diagnostic should have root-cause priority");
      }

      {
        RawGstError raw =
            raw_error("neatprocesscvu", "gst-stream-error-quark", GST_STREAM_ERROR_FAILED,
                      "The input tensor does not match the expected input contract.");
        raw.details["diagnostic_id"] = "neatprocesscvu.input_contract_mismatch";
        const NormalizedDiagnostic diagnostic = classify_gst_error(std::move(raw));
        require(diagnostic_priority(diagnostic) == 200,
                "accepted structured diagnostic-id aliases should receive root-cause priority");
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
        const std::string long_utf8 = std::string(16383, 'a') + "\xF0\x9F\x98\x80" + "tail";
        const std::string sanitized = sanitize_gst_diagnostic_text(long_utf8);
        require(g_utf8_validate(sanitized.c_str(), static_cast<gssize>(sanitized.size()), nullptr),
                "diagnostic truncation must preserve valid UTF-8");
        require_contains(sanitized, "...<truncated>",
                         "oversized diagnostics should retain the truncation marker");
        require(sanitized.find("\xF0\x9F\x98\x80") == std::string::npos,
                "a code point crossing the limit must be removed as a whole");
      }

      {
        const std::string quoted =
            redact_gst_credentials(R"(password="abc\"def" token=following-token)");
        require(quoted.find("abc") == std::string::npos &&
                    quoted.find("def") == std::string::npos &&
                    quoted.find("following-token") == std::string::npos,
                "quoted redaction must skip escaped delimiters and remove the complete value");
        require_contains(quoted, R"(password="<redacted>")",
                         "ordinary quoted fields should preserve their delimiters");

        const std::string escaped =
            redact_gst_credentials(R"(password=\"abc\\\"def\" token=following-token)");
        require(escaped.find("abc") == std::string::npos &&
                    escaped.find("def") == std::string::npos &&
                    escaped.find("following-token") == std::string::npos,
                "escaped quoted redaction must remove embedded escaped delimiters");
        require_contains(escaped, R"(password=\"<redacted>\")",
                         "escaped quoted fields should preserve their structural delimiters");

        const std::string standalone =
            redact_gst_credentials("Bearer standalone-value trailing=visible");
        require(standalone.find("standalone-value") == std::string::npos,
                "standalone bearer credentials must be redacted: " + standalone);
        require_contains(standalone, "Bearer <redacted> trailing=visible",
                         "standalone bearer redaction should preserve surrounding text");

        const std::string basic = redact_gst_credentials("Basic dXNlcjpwYXNz trailing=visible");
        require(basic.find("dXNlcjpwYXNz") == std::string::npos,
                "standalone Basic credentials must be redacted: " + basic);
        require_contains(basic, "Basic <redacted> trailing=visible",
                         "standalone Basic redaction should preserve surrounding text");

        const std::string ordinary = redact_gst_credentials("A basic configuration is required");
        require(ordinary == "A basic configuration is required",
                "ordinary uses of basic must not be treated as credentials");

        const std::string keys =
            redact_gst_credentials("current_key=video/x-raw new_key=tensor key=direct-key-secret "
                                   "api_key=api-key-secret");
        require_contains(keys, "current_key=video/x-raw new_key=tensor",
                         "non-secret fields ending in key must retain their values");
        require(keys.find("direct-key-secret") == std::string::npos &&
                    keys.find("api-key-secret") == std::string::npos,
                "standalone and explicit API key values must remain redacted: " + keys);

        const std::string tokens = redact_gst_credentials(
            "dtype_token=FP32 observed_token=UInt8 model_signature=v2 tensor_sig=layout "
            "access_token=access-secret token=direct-token signature=direct-signature");
        require_contains(tokens,
                         "dtype_token=FP32 observed_token=UInt8 model_signature=v2 "
                         "tensor_sig=layout",
                         "non-secret fields ending in token or signature must retain values");
        require(tokens.find("access-secret") == std::string::npos &&
                    tokens.find("direct-token") == std::string::npos &&
                    tokens.find("direct-signature") == std::string::npos,
                "explicit and standalone authentication values must remain redacted: " + tokens);

        const std::string session =
            redact_gst_credentials("session='session-secret' session_count=2");
        require(session.find("session-secret") == std::string::npos,
                "bare session fields must be redacted: " + session);
        require_contains(session, "session='<redacted>' session_count=2",
                         "session redaction must preserve non-secret session metadata");

        const std::string pass = redact_gst_credentials(
            "https://example.test/stream?user=alice&pass=hunter2 bypass=enabled compass=north");
        require(pass.find("hunter2") == std::string::npos,
                "bare pass fields must be redacted: " + pass);
        require_contains(pass, "pass=<redacted> bypass=enabled compass=north",
                         "pass redaction must preserve fields that only contain the word");

        const std::string aws = redact_gst_credentials(
            "https://example.test/input?X-Amz-Signature=aws-signature-value&"
            "X-Amz-Security-Token=aws-session-value&X-Amz-Algorithm=AWS4-HMAC-SHA256");
        require(aws.find("aws-signature-value") == std::string::npos &&
                    aws.find("aws-session-value") == std::string::npos,
                "AWS signed-URL credentials must be redacted: " + aws);
        require_contains(aws, "X-Amz-Signature=<redacted>",
                         "AWS signature redaction should retain the parameter name");
        require_contains(aws, "X-Amz-Security-Token=<redacted>",
                         "AWS session-token redaction should retain the parameter name");
        require_contains(aws, "X-Amz-Algorithm=AWS4-HMAC-SHA256",
                         "AWS redaction should preserve non-secret signing metadata");

        const std::string google = redact_gst_credentials(
            "https://storage.googleapis.test/input?X-Goog-Signature=google-signature-value&"
            "X-Goog-Algorithm=GOOG4-RSA-SHA256");
        require(google.find("google-signature-value") == std::string::npos,
                "Google signed-URL signatures must be redacted: " + google);
        require_contains(google, "X-Goog-Signature=<redacted>",
                         "Google signature redaction should retain the parameter name");
        require_contains(google, "X-Goog-Algorithm=GOOG4-RSA-SHA256",
                         "Google redaction should preserve non-secret signing metadata");

        const std::string streaming =
            redact_gst_credentials("https://stream.example.test/live?auth=auth-secret&"
                                   "playback-token=playback-secret&hdnts=exp=1~hmac=hdnts-secret "
                                   "stream-key=stream-secret mode=live");
        require(streaming.find("auth-secret") == std::string::npos &&
                    streaming.find("playback-secret") == std::string::npos &&
                    streaming.find("hdnts-secret") == std::string::npos &&
                    streaming.find("stream-secret") == std::string::npos,
                "streaming authentication parameters must be redacted: " + streaming);
        require_contains(streaming, "auth=<redacted>",
                         "auth query parameters should retain only their names");
        require_contains(streaming, "playback-token=<redacted>",
                         "playback tokens should retain only their names");
        require_contains(streaming, "hdnts=<redacted>",
                         "HDNTS signed tokens should retain only their names");
        require_contains(streaming, "stream-key=<redacted>",
                         "stream keys should retain only their names");
        require_contains(streaming, "mode=live",
                         "stream redaction should preserve non-secret parameters");

        const std::string browser_vscode = redact_gst_credentials(
            "https://127.0.0.1:10000/?tkn=browser-vscode-secret&folder=/workspace");
        require(browser_vscode.find("browser-vscode-secret") == std::string::npos,
                "browser VS Code access tokens must be redacted: " + browser_vscode);
        require_contains(browser_vscode, "tkn=<redacted>",
                         "browser VS Code token parameters should retain only their names");
        require_contains(browser_vscode, "folder=/workspace",
                         "browser VS Code redaction should preserve non-secret parameters");

        const std::string contact_url = "https://api.example.test?email=user@example.test#support";
        require(redact_gst_credentials(contact_url) == contact_url,
                "query and fragment at-signs must not be mistaken for URI userinfo");

        const std::string userinfo =
            redact_gst_credentials("https://user:password@example.test?email=user@example.test");
        require_contains(userinfo, "https://<redacted>@example.test?email=user@example.test",
                         "URI userinfo should be redacted without changing the query");

        const std::string escaped_userinfo =
            redact_gst_credentials(R"({"url":"https:\/\/user:password@example.test\/path"})");
        require(escaped_userinfo.find("user:password") == std::string::npos,
                "slash-escaped URI userinfo must be redacted: " + escaped_userinfo);
        require_contains(escaped_userinfo, R"(https:\/\/<redacted>@example.test\/path)",
                         "slash-escaped URI structure should be preserved");

        const std::string repeatedly_escaped =
            redact_gst_credentials(R"(https:\\/\\/user:password@example.test\\/path)");
        require(repeatedly_escaped.find("user:password") == std::string::npos,
                "repeatedly escaped URI userinfo must be redacted: " + repeatedly_escaped);
      }

      {
        constexpr const char* kSourceNameSecret = "source-name-secret";
        GstElement* source = gst_pipeline_new("auth=source-name-secret");
        require(source != nullptr, "pipeline object must be available for parser test");
        GError* error = g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_NOT_FOUND,
                                            "Resource not found.");
        GstStructure* details = gst_structure_new(
            "neat-error-details", "neat-diagnostic-id", G_TYPE_STRING, "gstreamer.file_not_found",
            "password", G_TYPE_STRING, "hunter2", "auth-token", G_TYPE_STRING, "do-not-store",
            "sig", G_TYPE_STRING, "signed-secret", "api_key", G_TYPE_STRING, "api-secret",
            "access_key", G_TYPE_STRING, "access-secret", "private_key", G_TYPE_STRING,
            "private-secret", "user-pw", G_TYPE_STRING, "rtsp-secret", "user_pw", G_TYPE_STRING,
            "rtsp-secret-underscore", "set-cookie", G_TYPE_STRING, "sessionid=structured-secret",
            "passphrase", G_TYPE_STRING, "structured-srt-secret", "pass", G_TYPE_STRING,
            "structured-pass-secret", "accessToken", G_TYPE_STRING, "camel-access-secret",
            "refreshToken", G_TYPE_STRING, "camel-refresh-secret", "authToken", G_TYPE_STRING,
            "camel-auth-secret", "clientSecret", G_TYPE_STRING, "camel-client-secret",
            "sessionToken", G_TYPE_STRING, "structured-session-secret", "neat_auth_token",
            G_TYPE_STRING, "prefixed-auth-secret", "auth", G_TYPE_STRING, "structured-auth-secret",
            "playback-token", G_TYPE_STRING, "structured-playback-secret", "hdnts", G_TYPE_STRING,
            "structured-hdnts-secret", "stream-key", G_TYPE_STRING, "structured-stream-key-secret",
            "dtype_token", G_TYPE_STRING, "UInt8", "model_signature", G_TYPE_STRING,
            "sha256:model-metadata", "aws_access_key_id", G_TYPE_STRING, "structured-aws-key",
            "awsSecretAccessKey", G_TYPE_STRING, "structured-aws-secret", "OPENVSCODE_SERVER_TOKEN",
            G_TYPE_STRING, "structured-openvscode-secret", "AWSAccessKeyId", G_TYPE_STRING,
            "structured-acronym-aws-key", "AWSSecretAccessKey", G_TYPE_STRING,
            "structured-acronym-aws-secret", "compass", G_TYPE_STRING, "north", nullptr);
        const std::string debug =
            "debug path token=do-not-leak Authorization: Bearer abc123\npassword: hunter2 "
            "api_key='quoted-api-secret' password=\"quoted-password\" user-pw='rtsp-password' "
            "user_pw=rtsp-password-underscore pw=short-password pwd='short-password-d' "
            "json={\"password\":\"json-password\",\"access_token\":\"json-token\"} "
            "escaped={\\\"password\\\":\\\"escaped-json-password\\\","
            "\\\"access_token\\\":\\\"escaped-json-token\\\"} "
            "camel={\"accessToken\":\"camel-json-access\","
            "\"refreshToken\":\"camel-json-refresh\","
            "\"authToken\":\"camel-json-auth\",\"clientSecret\":\"camel-json-client\"} "
            "sessions={\"sessionToken\":\"camel-session-secret\","
            "\"session_token\":\"underscore-session-secret\","
            "\"session-token\":\"hyphen-session-secret\"} "
            "aws={\"SecretAccessKey\":\"camel-aws-secret\","
            "\"awsSecretAccessKey\":\"camel-prefixed-aws-secret\","
            "\"AWSAccessKeyId\":\"acronym-aws-key\","
            "\"AWSSecretAccessKey\":\"acronym-aws-secret\"} "
            "OPENVSCODE_SERVER_TOKEN=raw-openvscode-secret "
            "GITHUB_TOKEN=raw-github-secret "
            "metadata={\"dtype_token\":\"UInt8\",\"model_signature\":\"sha256:metadata\"} "
            "Cookie: sessionid=raw-cookie-secret\n{\"session_cookie\":\"json-cookie-secret\"} "
            "password=(string)\"typed password secret\" "
            "credential='raw credential secret' "
            "credentials=(string)\"typed credential secret\" "
            "X-API-Key: extension-api-secret\n"
            "X_Auth_Token: extension-auth-secret "
            "srt://example.test:9000?passphrase=srt-passphrase-secret "
            R"(multiply={\\\"password\\\":\\\"multiply-password\\\",\\\"accessToken\\\":\\\"multiply-token\\\"})";
        GstMessage* message =
            gst_message_new_error_with_details(GST_OBJECT(source), error, debug.c_str(), details);
        g_error_free(error);

        const RawGstError raw = parse_gst_error_message(message);
        require(raw.source_name.find(kSourceNameSecret) == std::string::npos,
                "raw parser must redact credentials embedded in the source name: " +
                    raw.source_name);
        require_contains(raw.source_name, "auth=<redacted>",
                         "raw parser should preserve a useful redacted source name");
        require(raw.details.at("password") == "<redacted>" &&
                    raw.details.at("auth-token") == "<redacted>" &&
                    raw.details.at("sig") == "<redacted>" &&
                    raw.details.at("api_key") == "<redacted>" &&
                    raw.details.at("access_key") == "<redacted>" &&
                    raw.details.at("private_key") == "<redacted>" &&
                    raw.details.at("user-pw") == "<redacted>" &&
                    raw.details.at("user_pw") == "<redacted>" &&
                    raw.details.at("set-cookie") == "<redacted>" &&
                    raw.details.at("passphrase") == "<redacted>" &&
                    raw.details.at("pass") == "<redacted>" &&
                    raw.details.at("accessToken") == "<redacted>" &&
                    raw.details.at("refreshToken") == "<redacted>" &&
                    raw.details.at("authToken") == "<redacted>" &&
                    raw.details.at("clientSecret") == "<redacted>" &&
                    raw.details.at("sessionToken") == "<redacted>" &&
                    raw.details.at("neat_auth_token") == "<redacted>" &&
                    raw.details.at("auth") == "<redacted>" &&
                    raw.details.at("playback-token") == "<redacted>" &&
                    raw.details.at("hdnts") == "<redacted>" &&
                    raw.details.at("stream-key") == "<redacted>" &&
                    raw.details.at("aws_access_key_id") == "<redacted>" &&
                    raw.details.at("awsSecretAccessKey") == "<redacted>" &&
                    raw.details.at("OPENVSCODE_SERVER_TOKEN") == "<redacted>" &&
                    raw.details.at("AWSAccessKeyId") == "<redacted>" &&
                    raw.details.at("AWSSecretAccessKey") == "<redacted>" &&
                    raw.details.at("dtype_token") == "UInt8" &&
                    raw.details.at("model_signature") == "sha256:model-metadata" &&
                    raw.details.at("compass") == "north",
                "raw parser should redact values whose structured field names are sensitive");
        require_contains(raw.debug, "token=<redacted>", "raw parser should redact credentials");
        require(raw.debug.find("abc123") == std::string::npos &&
                    raw.debug.find("hunter2") == std::string::npos &&
                    raw.debug.find("quoted-api-secret") == std::string::npos &&
                    raw.debug.find("quoted-password") == std::string::npos &&
                    raw.debug.find("rtsp-password") == std::string::npos &&
                    raw.debug.find("short-password") == std::string::npos &&
                    raw.debug.find("json-password") == std::string::npos &&
                    raw.debug.find("json-token") == std::string::npos &&
                    raw.debug.find("escaped-json-password") == std::string::npos &&
                    raw.debug.find("escaped-json-token") == std::string::npos &&
                    raw.debug.find("camel-json-access") == std::string::npos &&
                    raw.debug.find("camel-json-refresh") == std::string::npos &&
                    raw.debug.find("camel-json-auth") == std::string::npos &&
                    raw.debug.find("camel-json-client") == std::string::npos &&
                    raw.debug.find("camel-session-secret") == std::string::npos &&
                    raw.debug.find("underscore-session-secret") == std::string::npos &&
                    raw.debug.find("hyphen-session-secret") == std::string::npos &&
                    raw.debug.find("camel-aws-secret") == std::string::npos &&
                    raw.debug.find("camel-prefixed-aws-secret") == std::string::npos &&
                    raw.debug.find("acronym-aws-key") == std::string::npos &&
                    raw.debug.find("acronym-aws-secret") == std::string::npos &&
                    raw.debug.find("raw-openvscode-secret") == std::string::npos &&
                    raw.debug.find("raw-github-secret") == std::string::npos &&
                    raw.debug.find("multiply-password") == std::string::npos &&
                    raw.debug.find("multiply-token") == std::string::npos &&
                    raw.debug.find("raw-cookie-secret") == std::string::npos &&
                    raw.debug.find("json-cookie-secret") == std::string::npos &&
                    raw.debug.find("typed password secret") == std::string::npos &&
                    raw.debug.find("raw credential secret") == std::string::npos &&
                    raw.debug.find("typed credential secret") == std::string::npos &&
                    raw.debug.find("extension-api-secret") == std::string::npos &&
                    raw.debug.find("extension-auth-secret") == std::string::npos &&
                    raw.debug.find("srt-passphrase-secret") == std::string::npos,
                "raw parser should redact header-style, colon-separated, and quoted credentials");
        require_contains(raw.debug, "\"dtype_token\":\"UInt8\"",
                         "non-secret token-suffixed metadata should remain available");
        require_contains(raw.debug, "\"model_signature\":\"sha256:metadata\"",
                         "non-secret signature metadata should remain available");
        require_contains(raw.debug, R"(\\\"password\\\":\\\"<redacted>\\\")",
                         "multiply escaped credential fields should retain redacted structure");
        require_contains(raw.debug, "api_key='<redacted>'",
                         "quoted credential redaction should preserve only the quote delimiters");
        require_contains(raw.debug, "\"access_token\":\"<redacted>\"",
                         "JSON credential fields should retain structure without their values");
        require_contains(raw.debug, "\\\"access_token\\\":\\\"<redacted>\\\"",
                         "escaped JSON credential fields should retain escaped structure");
        require_contains(raw.debug, "password=(string)\"<redacted>\"",
                         "GStreamer type annotations should remain while values are redacted");
        require_contains(raw.debug, "credential='<redacted>'",
                         "credential aliases should redact quoted raw values");
        require_contains(raw.debug, "credentials=(string)\"<redacted>\"",
                         "plural credential aliases should redact typed raw values");
        require_contains(raw.debug, "X-API-Key: <redacted>",
                         "prefixed API key headers should retain only their field names");
        require_contains(raw.debug, "X_Auth_Token: <redacted>",
                         "prefixed auth token headers should retain only their field names");
        require_contains(raw.debug, "passphrase=<redacted>",
                         "SRT passphrases should retain only their parameter names");
        const NormalizedDiagnostic diagnostic = classify_gst_error(raw);
        require_code(diagnostic, error_codes::kIoOpen);
        require(diagnostic.diagnostic_id == "gstreamer.resource_not_found",
                "resource errors without a source factory should use the generic mapping");

        gst_message_unref(message);
        gst_object_unref(source);
      }

      {
        GstElement* source = gst_pipeline_new("versioned-wire-format");
        require(source != nullptr, "pipeline object must be available for wire-format test");
        GError* error =
            g_error_new_literal(GST_STREAM_ERROR, GST_STREAM_ERROR_FORMAT,
                                "The input tensor does not match the expected input contract.");
        GstStructure* details = gst_structure_new(
            "simaai-neat-error", "neat-schema-version", G_TYPE_UINT, 1U, "neat-diagnostic-id",
            G_TYPE_STRING, "neatprocesscvu.input_contract_mismatch", "neat-reason", G_TYPE_STRING,
            "input_contract_mismatch", "plugin", G_TYPE_STRING, "neatprocesscvu", "node",
            G_TYPE_STRING, "model_0", "stage", G_TYPE_STRING, "model_0", "graph-id", G_TYPE_INT, 17,
            "frame-id", G_TYPE_INT64, static_cast<gint64>(42), "input-name", G_TYPE_STRING,
            "input_tensor", "segment-name", G_TYPE_STRING, "parent", "required-bytes",
            G_TYPE_UINT64, static_cast<guint64>(602112), "actual-bytes", G_TYPE_UINT64,
            static_cast<guint64>(150528), "expected-shape", G_TYPE_STRING, "[3, 224, 224]",
            "expected-layout", G_TYPE_STRING, "CHW", "expected-dtype", G_TYPE_STRING, "Float32",
            "received-shape", G_TYPE_STRING, "[224, 224, 3]", "received-layout", G_TYPE_STRING,
            "HWC", "received-dtype", G_TYPE_STRING, "UInt8", nullptr);
        GstMessage* message =
            gst_message_new_error_with_details(GST_OBJECT(source), error, "wire debug", details);
        g_error_free(error);

        const RawGstError raw = parse_gst_error_message(message);
        require(raw.details.at("neat-schema-version") == "1" &&
                    raw.details.at("expected-layout") == "CHW" &&
                    raw.details.at("received-layout") == "HWC",
                "Core must consume the version-1 plugin wire format");
        const std::string rendered = render_diagnostic_body(classify_gst_error(raw), false);
        require_contains(rendered, "Expected input: shape [3, 224, 224], layout CHW, type Float32",
                         "the production wire format must reach the user-facing diagnostic");

        gst_message_unref(message);
        gst_object_unref(source);
      }

      {
        constexpr const char* kDispatcherError =
            "MLASHM model path length 258 exceeds maximum 255 bytes; shorten the extracted model "
            "path";
        GstPad* source = gst_pad_new("MLA_0_1", GST_PAD_UNKNOWN);
        require(source != nullptr, "GStreamer source object must be available for MLA error test");
        GError* error = g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_FAILED,
                                            "Unable to load model");
        const std::string debug = model_path_debug("neatprocessmla", kDispatcherError);
        GstMessage* message = gst_message_new_error(GST_OBJECT(source), error, debug.c_str());
        g_error_free(error);

        const RawGstError raw = parse_gst_error_message(message);
        const NormalizedDiagnostic diagnostic = classify_gst_error(raw);
        require_code(diagnostic, error_codes::kRuntimeElementFailed);
        require(diagnostic.diagnostic_id == "neatprocessmla.model_path_too_long",
                "legacy MLA path-limit failures should receive an actionable diagnostic; actual=" +
                    diagnostic.diagnostic_id + " factory=" + raw.factory_name);
        const std::string production = render_diagnostic_body(diagnostic, false);
        require_contains(production, "could not be loaded because its extracted path is too long",
                         "MLA path failures should explain the cause");
        require_contains(production, "Stage: MLA_0_1",
                         "MLA path failures should retain the failing stage");
        require_contains(production, "Operation: Model loading",
                         "MLA path failures should identify the operation");
        require_contains(production, "Path length: 258 bytes",
                         "MLA path failures should report the observed length");
        require_contains(production, "Maximum path length: 255 bytes",
                         "MLA path failures should report the supported limit");
        require_contains(production, "shorter directory",
                         "MLA path failures should provide a corrective action");
        require(production.find(kPrivateModelPath) == std::string::npos &&
                    production.find("dispatcher_err") == std::string::npos,
                "default MLA diagnostics must not expose the raw path or debug fields");
        require_contains(render_diagnostic_body(diagnostic, true), kPrivateModelPath,
                         "verbose diagnostics should retain the raw path for troubleshooting");

        constexpr std::size_t kLongPathBytes = 5000;
        constexpr const char* kLongPathDispatcherError =
            "MLASHM model path length 5000 exceeds maximum 255 bytes; shorten the extracted model "
            "path";
        const std::string misleading_path_prefix =
            "/models/not-negotiated/unable to get dispatcher/buffer is too small/";
        const std::string long_path =
            misleading_path_prefix +
            std::string(kLongPathBytes - misleading_path_prefix.size(), 'a');
        const std::string long_debug = "plugin='neatprocessmla' model_path='" + long_path +
                                       "' dispatcher_err='" + kLongPathDispatcherError + "'";
        GError* long_error = g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_FAILED,
                                                 "Unable to load model");
        GstMessage* long_message =
            gst_message_new_error(GST_OBJECT(source), long_error, long_debug.c_str());
        g_error_free(long_error);
        const RawGstError long_raw = parse_gst_error_message(long_message);
        require(long_raw.debug.size() <= 4096, "long GStreamer debug capture must remain bounded");
        require_contains(long_raw.debug, "...<truncated>",
                         "long GStreamer debug capture must identify omitted text");
        const NormalizedDiagnostic long_diagnostic = classify_gst_error(long_raw);
        require(long_diagnostic.diagnostic_id == "neatprocessmla.model_path_too_long",
                "the final dispatcher error must survive an oversized preceding model path");
        const std::string long_production = render_diagnostic_body(long_diagnostic, false);
        require_contains(long_production, "Path length: 5000 bytes",
                         "the preserved dispatcher error must retain its observed length");
        require(long_production.find(std::string(128, 'a')) == std::string::npos,
                "production diagnostics must not expose an oversized raw model path");
        gst_message_unref(long_message);

        GstElementFactory* mla_factory = gst_element_factory_find("neatprocessmla");
        if (!mla_factory) {
          require(gst_element_register(nullptr, "neatprocessmla", GST_RANK_NONE, GST_TYPE_BIN),
                  "test must register a stand-in neatprocessmla factory");
        } else {
          gst_object_unref(mla_factory);
        }
        GstElement* factory_source = gst_element_factory_make("neatprocessmla", "MLA_factory_0");
        require(factory_source != nullptr,
                "factory-identified MLA source must be available for debug-capture test");
        const std::string factory_debug =
            "model_path='" + long_path + "' dispatcher_err='" + kLongPathDispatcherError + "'";
        GError* factory_error = g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_FAILED,
                                                    "Unable to load model");
        GstMessage* factory_message =
            gst_message_new_error(GST_OBJECT(factory_source), factory_error, factory_debug.c_str());
        g_error_free(factory_error);
        const RawGstError factory_raw = parse_gst_error_message(factory_message);
        require(factory_raw.factory_name == "neatprocessmla",
                "the test message must establish MLA identity through its element factory");
        require(classify_gst_error(factory_raw).diagnostic_id ==
                    "neatprocessmla.model_path_too_long",
                "factory identity must preserve the final dispatcher error when plugin is absent");
        gst_message_unref(factory_message);
        gst_object_unref(factory_source);

        const NormalizedDiagnostic caps =
            classify_gst_error(raw_error("capsfilter", "gst-core-error-quark",
                                         GST_CORE_ERROR_NEGOTIATION, "Caps negotiation failed"));
        require(diagnostic_priority(diagnostic) > diagnostic_priority(caps),
                "the actionable MLA root cause should outrank a trailing caps error");

        const NormalizedDiagnostic different_limit = classify_gst_error(legacy_model_path_error(
            "MLASHM model path length 4096 exceeds maximum 1024 bytes; shorten the extracted "
            "model path"));
        const std::string different_limit_text = render_diagnostic_body(different_limit, false);
        require_contains(different_limit_text, "Path length: 4096 bytes",
                         "the classifier must report the observed value instead of a constant");
        require_contains(different_limit_text, "Maximum path length: 1024 bytes",
                         "the classifier must report the producer-supplied maximum");

        RawGstError structured = raw_error("neatprocessmla", "gst-resource-error-quark",
                                           GST_RESOURCE_ERROR_FAILED, "Unable to load model");
        structured.details["neat-diagnostic-id"] = "neatprocessmla.model_path_too_long";
        structured.details["dispatcher-error"] = kDispatcherError;
        const NormalizedDiagnostic structured_diagnostic =
            classify_gst_error(std::move(structured));
        require(structured_diagnostic.diagnostic_id == "neatprocessmla.model_path_too_long",
                "the versioned structured dispatcher field must use the same typed diagnosis");
        require_contains(render_diagnostic_body(structured_diagnostic, false),
                         "Path length: 258 bytes",
                         "the matching explicit diagnostic ID must retain validated byte facts");

        RawGstError factory_identified = legacy_model_path_error(kDispatcherError);
        factory_identified.factory_name = "neatprocessmla";
        factory_identified.debug =
            "dispatcher_err='MLASHM model path length 258 exceeds maximum 255 bytes; shorten the "
            "extracted model path'";
        require(classify_gst_error(std::move(factory_identified)).diagnostic_id ==
                    "neatprocessmla.model_path_too_long",
                "the element factory must establish the MLA source without a legacy plugin field");

        RawGstError other_plugin = raw;
        other_plugin.debug =
            "plugin='otherplugin' dispatcher_err='MLASHM model path length 258 exceeds maximum "
            "255 bytes; shorten the extracted model path'";
        require(classify_gst_error(std::move(other_plugin)).diagnostic_id ==
                    "gstreamer.unclassified_element_failure",
                "the legacy path matcher must require the neatprocessmla plugin");

        const auto require_generic = [](RawGstError candidate, const std::string& context) {
          const NormalizedDiagnostic fallback = classify_gst_error(std::move(candidate));
          require(fallback.diagnostic_id == "gstreamer.unclassified_element_failure",
                  context + " must retain the generic diagnostic ID");
          const std::string fallback_text = render_diagnostic_body(fallback, false);
          require_contains(fallback_text, "A pipeline stage stopped while processing data.",
                           context + " must retain the generic explanation");
          require(fallback_text.find("Path length:") == std::string::npos,
                  context + " must not invent path-length facts");
        };

        require_generic(
            legacy_model_path_error(
                "MLASHM model path length 255 exceeds maximum 255 bytes; shorten the extracted "
                "model path"),
            "non-exceeding path lengths");
        require_generic(
            legacy_model_path_error(
                "MLASHM model path length 18446744073709551616 exceeds maximum 255 bytes; "
                "shorten the extracted model path"),
            "overflowing path lengths");
        require_generic(
            legacy_model_path_error(
                "MLASHM model path length many exceeds maximum 255 bytes; shorten the extracted "
                "model path"),
            "nonnumeric path lengths");
        require_generic(
            legacy_model_path_error(
                "MLASHM model path length -1 exceeds maximum 255 bytes; shorten the extracted "
                "model path"),
            "negative path lengths");
        require_generic(
            legacy_model_path_error(
                "MLASHM model path length 258 exceeds maximum 255 characters; shorten the "
                "extracted model path"),
            "unexpected units");

        RawGstError lookalike_field = legacy_model_path_error(kDispatcherError);
        lookalike_field.debug =
            "plugin='neatprocessmla' other_dispatcher_err='MLASHM model path length 258 exceeds "
            "maximum 255 bytes; shorten the extracted model path'";
        require_generic(std::move(lookalike_field), "lookalike debug fields");

        RawGstError unterminated_field = legacy_model_path_error(kDispatcherError);
        unterminated_field.debug =
            "plugin='neatprocessmla' dispatcher_err='MLASHM model path length 258 exceeds maximum "
            "255 bytes; shorten the extracted model path";
        require_generic(std::move(unterminated_field), "unterminated debug fields");

        RawGstError conflicting_factory = legacy_model_path_error(kDispatcherError);
        conflicting_factory.factory_name = "otherplugin";
        require_generic(std::move(conflicting_factory), "conflicting factory identity");

        RawGstError other_operation = legacy_model_path_error(kDispatcherError);
        other_operation.message = "Unable to execute model";
        require_generic(std::move(other_operation), "other MLA operations");

        RawGstError existing_classification = legacy_model_path_error(kDispatcherError);
        existing_classification.code = GST_RESOURCE_ERROR_NOT_FOUND;
        require(classify_gst_error(std::move(existing_classification)).diagnostic_id ==
                    "neat.dispatcher_unavailable",
                "an existing classified dispatcher error must take precedence over legacy parsing");

        RawGstError explicit_diagnostic = legacy_model_path_error(kDispatcherError);
        explicit_diagnostic.details["neat-diagnostic-id"] = "producer.specific_failure";
        require(classify_gst_error(std::move(explicit_diagnostic)).diagnostic_id ==
                    "producer.specific_failure",
                "an explicit producer diagnostic ID must take precedence over legacy parsing");

        const NormalizedDiagnostic resource_failure =
            classify_gst_error(raw_error("filesrc", "gst-resource-error-quark",
                                         GST_RESOURCE_ERROR_NOT_FOUND, "Resource not found."));
        const auto require_path_marker_ignored = [&](const std::string& model_path,
                                                     const std::string& context) {
          RawGstError marked_path = legacy_model_path_error(kDispatcherError);
          marked_path.debug = "plugin='neatprocessmla' model_path='" + model_path +
                              "' dispatcher_err='" + kDispatcherError + "'";
          const NormalizedDiagnostic marked = classify_gst_error(std::move(marked_path));
          require(marked.diagnostic_id == "neatprocessmla.model_path_too_long",
                  context + " inside the model path must not override the path-limit diagnosis");
          require(diagnostic_priority(marked) == diagnostic_priority(diagnostic),
                  context + " inside the model path must not raise diagnostic priority");
          require(diagnostic_priority(marked) < diagnostic_priority(resource_failure),
                  context + " inside the model path must not hide a resource failure");
        };
        require_path_marker_ignored("/models/neat-diagnostic-id=foo/model.tar.gz",
                                    "an unquoted diagnostic-ID marker");
        require_path_marker_ignored("/models/x diagnostic_id='producer.other' y/model.tar.gz",
                                    "a quoted legacy diagnostic-ID field");

        RawGstError legacy_explicit = legacy_model_path_error(kDispatcherError);
        legacy_explicit.debug = "diagnostic_id='producer.specific_failure' " +
                                model_path_debug("neatprocessmla", kDispatcherError);
        require(classify_gst_error(std::move(legacy_explicit)).diagnostic_id ==
                    "producer.specific_failure",
                "a leading legacy producer diagnostic ID must take precedence over legacy parsing");

        RawGstError legacy_matching = legacy_model_path_error(kDispatcherError);
        legacy_matching.debug = "diagnostic_id='neatprocessmla.model_path_too_long' " +
                                model_path_debug("neatprocessmla", kDispatcherError);
        require(diagnostic_priority(classify_gst_error(std::move(legacy_matching))) == 200,
                "a leading matching producer diagnostic ID must keep explicit-ID priority");

        gst_message_unref(message);
        gst_object_unref(source);
      }

      require_model_path_error_wins_on_bus(true);
      require_model_path_error_wins_on_bus(false);

      {
        GstElement* source = gst_pipeline_new("unknown-wire-format");
        require(source != nullptr, "pipeline object must be available for schema-version test");
        GError* error =
            g_error_new_literal(GST_STREAM_ERROR, GST_STREAM_ERROR_FAILED, "Opaque failure.");
        GstStructure* details = gst_structure_new(
            "simaai-neat-error", "neat-schema-version", G_TYPE_UINT, 2U, "neat-diagnostic-id",
            G_TYPE_STRING, "neatprocesscvu.input_contract_mismatch", "expected-shape",
            G_TYPE_STRING, "[3, 224, 224]", nullptr);
        GstMessage* message =
            gst_message_new_error_with_details(GST_OBJECT(source), error, "wire debug", details);
        g_error_free(error);

        const RawGstError raw = parse_gst_error_message(message);
        require(raw.details.empty(),
                "Core must ignore structured fields from unsupported Neat schema versions");
        require(classify_gst_error(raw).diagnostic_id != "neatprocesscvu.input_contract_mismatch",
                "unsupported schema versions must fall back instead of being misinterpreted");

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
        GError* error = g_error_new_literal(GST_PARSE_ERROR, GST_PARSE_ERROR_COULD_NOT_SET_PROPERTY,
                                            "could not set property \"latency\" to \"fast\"");
        const NormalizedDiagnostic diagnostic =
            classify_gst_parse_error(error, "rtspsrc latency=fast ! fakesink");
        g_error_free(error);
        require_code(diagnostic, error_codes::kPropertyInvalid);
        const std::string production = render_diagnostic_body(diagnostic, false);
        require_contains(production, "property is unknown or invalid",
                         "invalid property values should receive property-specific guidance");
        require(production.find("latency") == std::string::npos,
                "production property errors must not echo raw parser messages");
        require_contains(render_diagnostic_body(diagnostic, true), "latency",
                         "debug property errors should retain raw parser messages");
      }

      {
        GError* error = g_error_new_literal(GST_LIBRARY_ERROR, GST_LIBRARY_ERROR_FAILED,
                                            "opaque parser failure custom-value=private");
        const NormalizedDiagnostic diagnostic = classify_gst_parse_error(error, "custom fragment");
        g_error_free(error);
        require_code(diagnostic, error_codes::kParseLaunch);
        const std::string production = render_diagnostic_body(diagnostic, false);
        require(production.find("opaque parser failure") == std::string::npos &&
                    production.find("custom-value") == std::string::npos,
                "production parse errors must not echo raw parser messages");
        const std::string debug = render_diagnostic_body(diagnostic, true);
        require_contains(debug, "opaque parser failure",
                         "debug parse errors should retain raw parser messages");
      }

      {
        const NormalizedDiagnostic diagnostic = classify_gst_error(raw_error(
            "mystage", "gst-library-error-quark", GST_LIBRARY_ERROR_FAILED, "opaque failure"));
        require_code(diagnostic, error_codes::kRuntimeElementFailed);
        require(diagnostic.diagnostic_id == "gstreamer.unclassified_element_failure",
                "unknown errors should use the explicit fallback diagnostic");
        const std::string production = render_diagnostic_body(diagnostic, false);
        require(production.find("opaque failure") == std::string::npos,
                "generic production errors must not echo raw plugin messages");
        const std::string debug = render_diagnostic_body(diagnostic, true);
        require_contains(
            debug, "Technical details:", "debug rendering should include raw technical context");
        require_contains(debug, "opaque failure",
                         "debug rendering should retain the raw plugin message");
      }

      {
        RawGstError raw = raw_error("mystage", "gst-library-error-quark", GST_LIBRARY_ERROR_FAILED,
                                    "opaque failure");
        raw.details["reason"] = "Validated plugin reason";
        const std::string production = render_diagnostic_body(classify_gst_error(raw), false);
        require_contains(production, "Reason: Validated plugin reason",
                         "generic production errors should retain vetted structured reasons");
      }

      {
        const NormalizedDiagnostic diagnostic =
            classify_gst_error(raw_error("mystage", "gst-library-error-quark",
                                         GST_LIBRARY_ERROR_FAILED, "reason=UntrustedRawReason"));
        const std::string production = render_diagnostic_body(diagnostic, false);
        require(production.find("UntrustedRawReason") == std::string::npos,
                "raw reason fields must remain out of production diagnostics");
        require_contains(render_diagnostic_body(diagnostic, true), "UntrustedRawReason",
                         "debug rendering should retain raw reason fields");
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

        PullError third_party_error;
        error_util::set_pull_error(&third_party_error, error_codes::kRuntimeElementFailed,
                                   "[json.exception.parse_error.101] invalid JSON");
        require(third_party_error.code == error_codes::kRuntimeElementFailed,
                "third-party bracketed text must not replace a stable framework code");
        require_contains(third_party_error.message, "[runtime.element_failed]",
                         "third-party bracketed text should be decorated with the supplied code");

        PullError undeclared_error;
        error_util::set_pull_error(&undeclared_error, error_codes::kRuntimeElementFailed,
                                   "[infra.dispatcher_unavailble] arbitrary exception");
        require(undeclared_error.code == error_codes::kRuntimeElementFailed,
                "undeclared framework-looking prefixes must not replace a canonical code");
        require_contains(undeclared_error.message, "[runtime.element_failed]",
                         "undeclared prefixes should be decorated with the supplied code");
      }

      {
        GraphReport report;
        report.error_code = error_codes::kInputShape;
        report.repro_note = "Expected shape [1, 224, 224, 3], received [1, 224, 224, 1].";
        NeatError source(error_util::decorate_error(report.error_code, report.repro_note), report);
        auto core = std::make_shared<runtime::RunCore>();
        core->graph_request_stop("GraphRun: pipeline push failed");
        core->graph_request_stop(source);

        const std::optional<PullError> propagated = core->graph_last_error_detail();
        require(propagated.has_value(), "graph-backed runs should retain a typed terminal error");
        require(propagated->code == error_codes::kInputShape,
                "graph-backed runs should retain the original error code");
        require(propagated->report.has_value() &&
                    propagated->report->error_code == error_codes::kInputShape,
                "a typed graph failure should replace an earlier untyped push placeholder");

        auto linear_core = std::make_shared<runtime::RunCore>();
        linear_core->set_terminal_error(source);
        const std::optional<PullError> worker_error = linear_core->last_error_detail();
        require(worker_error.has_value(),
                "linear input workers should retain a typed terminal error");
        require(worker_error->code == error_codes::kInputShape,
                "linear input workers should retain the original error code");
        require(worker_error->report.has_value() &&
                    worker_error->report->error_code == error_codes::kInputShape,
                "linear input workers should retain the original error report");
      }

      {
        auto state = std::make_shared<InputStream::State>();
        set_stream_error(*state, "InputStream::push: appsrc push failed");

        bool placeholder_had_report = false;
        try {
          throw_push_failed_with_last_error("InputStream::push", state);
        } catch (const NeatError&) {
          placeholder_had_report = true;
        } catch (const std::runtime_error&) {
        }
        require(!placeholder_had_report,
                "a reportless push placeholder must remain upgradeable by a later bus report");

        GraphReport report;
        report.error_code = error_codes::kMediaCaps;
        report.repro_note = "The input caps do not match the downstream stage.";
        set_stream_error(*state, report.error_code, report.repro_note, report);

        require(state->terminal_error.has_value() && state->terminal_error->report.has_value(),
                "a typed stream error should replace an earlier reportless push failure");
        require(state->terminal_error->code == error_codes::kMediaCaps,
                "a typed stream upgrade should retain the specific error code");
        require(state->terminal_error->report->error_code == error_codes::kMediaCaps,
                "a typed stream upgrade should retain the GraphReport");
      }

      {
        auto parent = std::make_shared<runtime::RunCore>();
        auto child = std::make_shared<runtime::RunCore>();
        PullError placeholder;
        error_util::set_pull_error(&placeholder, error_codes::kRuntimeElementFailed,
                                   "InputStream::push: appsrc push failed");
        child->set_terminal_error(std::move(placeholder));

        std::thread bus_error([child]() {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          GraphReport report;
          report.error_code = error_codes::kMediaCaps;
          report.repro_note = "The input caps do not match the downstream stage.";
          PullError typed;
          error_util::set_pull_error(&typed, report.error_code, report.repro_note,
                                     std::move(report));
          child->set_terminal_error(std::move(typed));
        });
        parent->graph_request_stop_from_pipeline(child, "GraphRun: pipeline push failed");
        bus_error.join();

        const std::optional<PullError> propagated = parent->graph_last_error_detail();
        require(propagated.has_value() && propagated->report.has_value(),
                "a graph push failure should wait for the child bus report");
        require(propagated->code == error_codes::kMediaCaps,
                "a graph push failure should propagate the normalized child error");
      }

      {
        auto direct = std::make_shared<runtime::RunCore>();
        PullError placeholder;
        error_util::set_pull_error(&placeholder, error_codes::kRuntimeElementFailed,
                                   "InputStream::push: appsrc push failed");
        direct->set_terminal_error(std::move(placeholder));

        std::thread bus_error([direct]() {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          GraphReport report;
          report.error_code = error_codes::kMediaCaps;
          report.repro_note = "The input caps do not match the downstream stage.";
          PullError typed;
          error_util::set_pull_error(&typed, report.error_code, report.repro_note,
                                     std::move(report));
          direct->set_terminal_error(std::move(typed));
        });
        const std::optional<PullError> settled =
            direct->wait_for_report_bearing_error(std::chrono::milliseconds(250));
        bus_error.join();

        require(settled.has_value() && settled->report.has_value(),
                "a direct pull should wait for the report-bearing bus error");
        require(settled->code == error_codes::kMediaCaps,
                "a direct pull should retain the normalized bus error code");
      }

      {
        constexpr const char* kPipelineSecret = "pipeline-password";
        constexpr const char* kQuerySecret = "query-token";
        constexpr const char* kBusSecret = "bus-password";
        constexpr const char* kAuthSecret = "auth-secret";
        constexpr const char* kPlaybackSecret = "playback-secret";
        constexpr const char* kHdntsSecret = "hdnts-secret";
        constexpr const char* kStreamKeySecret = "stream-secret";
        constexpr const char* kUserLabelSecret = "user-label-secret";
        constexpr const char* kElementNameSecret = "element-name-secret";
        constexpr const char* kBusSourceSecret = "bus-source-secret";
        DiagCtx diag;
        diag.pipeline_string =
            "rtspsrc location=\"rtsp://camera:pipeline-password@example.test/live?"
            "access_token=query-token&auth=auth-secret&playback-token=playback-secret&"
            "hdnts=exp=1~hmac=hdnts-secret\" stream-key=stream-secret ! fakesink";
        NodeReport node;
        node.user_label = std::string("camera auth=") + kUserLabelSecret;
        node.backend_fragment = diag.pipeline_string;
        node.elements.push_back(std::string("auth=") + kElementNameSecret);
        diag.node_reports.push_back(std::move(node));
        diag.push_bus("ELEMENT", std::string("rtspsrc0 auth=") + kBusSourceSecret,
                      "password=(string)\"bus-password\"");

        const GraphReport report = diag.snapshot_basic();
        require(diag.pipeline_string.find(kPipelineSecret) != std::string::npos,
                "report redaction must not change the executable pipeline");
        require(
            report.pipeline_string.find(kPipelineSecret) == std::string::npos &&
                report.pipeline_string.find(kQuerySecret) == std::string::npos &&
                report.pipeline_string.find(kAuthSecret) == std::string::npos &&
                report.pipeline_string.find(kPlaybackSecret) == std::string::npos &&
                report.pipeline_string.find(kHdntsSecret) == std::string::npos &&
                report.pipeline_string.find(kStreamKeySecret) == std::string::npos &&
                report.repro_gst_launch.find(kPipelineSecret) == std::string::npos &&
                report.repro_gst_launch.find(kQuerySecret) == std::string::npos &&
                report.repro_gst_launch.find(kAuthSecret) == std::string::npos &&
                report.repro_gst_launch.find(kPlaybackSecret) == std::string::npos &&
                report.repro_gst_launch.find(kHdntsSecret) == std::string::npos &&
                report.repro_gst_launch.find(kStreamKeySecret) == std::string::npos &&
                report.nodes.front().backend_fragment.find(kPipelineSecret) == std::string::npos &&
                report.nodes.front().backend_fragment.find(kAuthSecret) == std::string::npos &&
                report.nodes.front().backend_fragment.find(kPlaybackSecret) == std::string::npos &&
                report.nodes.front().backend_fragment.find(kHdntsSecret) == std::string::npos &&
                report.nodes.front().backend_fragment.find(kStreamKeySecret) == std::string::npos &&
                report.nodes.front().user_label.find(kUserLabelSecret) == std::string::npos &&
                report.nodes.front().elements.front().find(kElementNameSecret) ==
                    std::string::npos &&
                report.bus.front().src.find(kBusSourceSecret) == std::string::npos &&
                report.bus.front().detail.find(kBusSecret) == std::string::npos,
            "report-facing pipeline, node, bus, and repro fields must redact credentials");

        RawGstError source_name_raw = raw_error("identity", "gst-stream-error-quark",
                                                GST_STREAM_ERROR_FAILED, "An element failed.");
        source_name_raw.source_name = std::string("camera auth=") + kBusSourceSecret;
        const std::string rendered_source =
            render_diagnostic_body(classify_gst_error(std::move(source_name_raw)), false);
        require(rendered_source.find(kBusSourceSecret) == std::string::npos,
                "default diagnostic text must redact credentials in stage names");
        require_contains(rendered_source, "Stage: camera auth=<redacted>",
                         "default diagnostic text should keep a useful redacted stage name");

        GraphReport manually_populated;
        manually_populated.pipeline_string = diag.pipeline_string;
        manually_populated.repro_note = "password=serialized-password";
        constexpr const char* kArbitraryFieldSecret = "arbitrary-field-secret";
        const std::string arbitrary_secret = std::string("auth=") + kArbitraryFieldSecret;
        manually_populated.error_code = arbitrary_secret;
        manually_populated.nodes.push_back(NodeReport{
            0, arbitrary_secret, arbitrary_secret, arbitrary_secret, {arbitrary_secret}});
        manually_populated.bus.push_back(
            BusMessage{arbitrary_secret, arbitrary_secret, arbitrary_secret, 0});
        BoundaryFlowStats boundary;
        boundary.boundary_name = arbitrary_secret;
        manually_populated.boundaries.push_back(std::move(boundary));
        manually_populated.caps_dump = arbitrary_secret;
        manually_populated.dot_paths.push_back(arbitrary_secret);
        manually_populated.repro_env = arbitrary_secret;
        manually_populated.has_build_adaptation = true;
        manually_populated.build_adaptation.shape_policy = arbitrary_secret;
        manually_populated.build_adaptation.dynamic_capability = arbitrary_secret;
        manually_populated.build_adaptation.seed_width_origin = arbitrary_secret;
        manually_populated.build_adaptation.seed_height_origin = arbitrary_secret;
        manually_populated.build_adaptation.seed_depth_origin = arbitrary_secret;
        manually_populated.build_adaptation.max_width_origin = arbitrary_secret;
        manually_populated.build_adaptation.max_height_origin = arbitrary_secret;
        manually_populated.build_adaptation.max_depth_origin = arbitrary_secret;
        manually_populated.build_adaptation.byte_guard_origin = arbitrary_secret;
        manually_populated.build_adaptation.actions.push_back(
            BuildAdaptationAction{arbitrary_secret, false, arbitrary_secret, arbitrary_secret});
        const std::string json = manually_populated.to_json();
        require(json.find(kPipelineSecret) == std::string::npos &&
                    json.find(kQuerySecret) == std::string::npos &&
                    json.find(kAuthSecret) == std::string::npos &&
                    json.find(kPlaybackSecret) == std::string::npos &&
                    json.find(kHdntsSecret) == std::string::npos &&
                    json.find(kStreamKeySecret) == std::string::npos &&
                    json.find("serialized-password") == std::string::npos &&
                    json.find(kArbitraryFieldSecret) == std::string::npos,
                "GraphReport JSON serialization must redact every string value at the final "
                "boundary");

        const GraphReport utility_report =
            error_util::make_report(error_codes::kRuntimeElementFailed, "password=summary-password",
                                    diag.pipeline_string, "token=hint-token");
        require(utility_report.pipeline_string.find(kPipelineSecret) == std::string::npos &&
                    utility_report.repro_note.find("summary-password") == std::string::npos &&
                    utility_report.repro_note.find("hint-token") == std::string::npos,
                "generic report construction must redact report-facing credentials");
      }
    }));

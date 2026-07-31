#include "pipeline/internal/GstErrorNormalizer.h"

#include "pipeline/ErrorCodes.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>

namespace simaai::neat::pipeline_internal {
namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool contains_ci(const std::string& haystack, std::string_view needle) {
  return lower_copy(haystack).find(lower_copy(std::string(needle))) != std::string::npos;
}

std::string value_to_string(const GValue* value) {
  if (!value)
    return {};
  if (G_VALUE_HOLDS_STRING(value)) {
    const char* s = g_value_get_string(value);
    return s ? s : "";
  }
  if (G_VALUE_HOLDS_BOOLEAN(value))
    return g_value_get_boolean(value) ? "true" : "false";
  if (G_VALUE_HOLDS_INT(value))
    return std::to_string(g_value_get_int(value));
  if (G_VALUE_HOLDS_UINT(value))
    return std::to_string(g_value_get_uint(value));
  if (G_VALUE_HOLDS_INT64(value))
    return std::to_string(g_value_get_int64(value));
  if (G_VALUE_HOLDS_UINT64(value))
    return std::to_string(g_value_get_uint64(value));
  if (G_VALUE_HOLDS_LONG(value))
    return std::to_string(g_value_get_long(value));
  if (G_VALUE_HOLDS_ULONG(value))
    return std::to_string(g_value_get_ulong(value));
  if (G_VALUE_HOLDS_DOUBLE(value)) {
    std::ostringstream oss;
    oss << g_value_get_double(value);
    return oss.str();
  }
  gchar* serialized = gst_value_serialize(value);
  std::string out = serialized ? serialized : "";
  g_free(serialized);
  return out;
}

std::string redact_uri_credentials(std::string value) {
  std::size_t search_from = 0;
  while (true) {
    const std::size_t scheme = value.find("://", search_from);
    if (scheme == std::string::npos)
      break;
    const std::size_t authority = scheme + 3;
    const std::size_t slash = value.find_first_of("/ \t\r\n", authority);
    const std::size_t at = value.find('@', authority);
    if (at != std::string::npos && (slash == std::string::npos || at < slash)) {
      value.replace(authority, at - authority, "<redacted>");
      search_from = authority + std::string("<redacted>@").size();
    } else {
      search_from = authority;
    }
  }

  static constexpr std::string_view sensitive[] = {
      "token=", "key=", "secret=", "password=", "signature=", "sig=",
  };
  for (std::string_view marker : sensitive) {
    std::size_t pos = 0;
    while ((pos = lower_copy(value).find(marker, pos)) != std::string::npos) {
      const std::size_t begin = pos + marker.size();
      const std::size_t end = value.find_first_of("& \t\r\n'\"", begin);
      value.replace(begin, (end == std::string::npos ? value.size() : end) - begin, "<redacted>");
      pos = begin + std::string("<redacted>").size();
    }
  }

  static constexpr std::string_view header_secrets[] = {
      "authorization", "proxy-authorization", "password", "passwd",  "token",
      "secret",        "signature",           "sig",      "api-key", "apikey",
  };
  for (std::string_view marker : header_secrets) {
    std::size_t pos = 0;
    while ((pos = lower_copy(value).find(marker, pos)) != std::string::npos) {
      if (pos > 0 && (std::isalnum(static_cast<unsigned char>(value[pos - 1])) ||
                      value[pos - 1] == '_' || value[pos - 1] == '-')) {
        pos += marker.size();
        continue;
      }
      std::size_t separator = pos + marker.size();
      while (separator < value.size() &&
             std::isspace(static_cast<unsigned char>(value[separator])) &&
             value[separator] != '\r' && value[separator] != '\n') {
        ++separator;
      }
      if (separator >= value.size() || (value[separator] != ':' && value[separator] != '=')) {
        pos += marker.size();
        continue;
      }
      std::size_t begin = separator + 1;
      while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) &&
             value[begin] != '\r' && value[begin] != '\n') {
        ++begin;
      }
      const bool authorization = marker.find("authorization") != std::string_view::npos;
      const std::size_t end = authorization ? value.find_first_of("\r\n", begin)
                                            : value.find_first_of("&,; \t\r\n'\"", begin);
      value.replace(begin, (end == std::string::npos ? value.size() : end) - begin, "<redacted>");
      pos = begin + std::string("<redacted>").size();
    }
  }
  return value;
}

std::string truncate(std::string value, std::size_t max_bytes = 4096) {
  value = redact_uri_credentials(std::move(value));
  if (value.size() <= max_bytes)
    return value;
  value.resize(max_bytes);
  value += "...<truncated>";
  return value;
}

bool sensitive_detail_name(std::string_view name) {
  const std::string lower = lower_copy(std::string(name));
  if (lower == "key" || lower == "sig")
    return true;
  static constexpr std::string_view markers[] = {
      "password",      "passwd",  "token",  "secret",     "signature",   "credential",
      "authorization", "api-key", "apikey", "access-key", "private-key",
  };
  for (std::string_view marker : markers) {
    if (lower.find(marker) != std::string::npos)
      return true;
  }
  return false;
}

std::string combined_raw(const RawGstError& raw) {
  std::string out = raw.message;
  if (!raw.debug.empty()) {
    if (!out.empty())
      out += " ";
    out += raw.debug;
  }
  return out;
}

std::string readable_string_property(GObject* object, const char* property_name) {
  if (!object || !property_name)
    return {};
  GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), property_name);
  if (!spec || !(spec->flags & G_PARAM_READABLE) || G_PARAM_SPEC_VALUE_TYPE(spec) != G_TYPE_STRING)
    return {};

  gchar* value = nullptr;
  g_object_get(object, property_name, &value, nullptr);
  std::string out = truncate(value ? value : "");
  g_free(value);
  return out;
}

bool safe_production_reason(const std::string& reason) {
  if (reason.empty() || reason.size() > 240 || reason.find_first_of("\r\n=") != std::string::npos)
    return false;
  const std::string lower = lower_copy(reason);
  return lower.find("gst_") == std::string::npos && lower.find("0x") == std::string::npos &&
         lower.find(".cpp") == std::string::npos && lower.find(".c:") == std::string::npos;
}

std::optional<std::string> find_detail(const RawGstError& raw,
                                       std::initializer_list<std::string_view> keys) {
  for (std::string_view key : keys) {
    auto it = raw.details.find(std::string(key));
    if (it != raw.details.end() && !it->second.empty())
      return it->second;
  }

  const std::string text = combined_raw(raw);
  for (std::string_view key : keys) {
    const std::string marker = std::string(key) + "=";
    std::size_t pos = text.find(marker);
    if (pos == std::string::npos)
      continue;
    pos += marker.size();
    if (pos >= text.size())
      continue;
    char quote = '\0';
    if (text[pos] == '\'' || text[pos] == '"') {
      quote = text[pos++];
    }
    std::size_t end = pos;
    while (end < text.size()) {
      const char c = text[end];
      if ((quote != '\0' && c == quote) ||
          (quote == '\0' &&
           (std::isspace(static_cast<unsigned char>(c)) || c == '|' || c == '\'' || c == '"'))) {
        break;
      }
      ++end;
    }
    if (end > pos)
      return text.substr(pos, end - pos);
  }
  return std::nullopt;
}

std::optional<int64_t> find_integer(const RawGstError& raw,
                                    std::initializer_list<std::string_view> keys) {
  const std::optional<std::string> value = find_detail(raw, keys);
  if (!value.has_value())
    return std::nullopt;
  char* end = nullptr;
  const long long parsed = std::strtoll(value->c_str(), &end, 10);
  if (!end || end == value->c_str())
    return std::nullopt;
  return static_cast<int64_t>(parsed);
}

std::string quoted_subject(const std::string& text, std::string_view before) {
  const std::size_t marker = text.find(before);
  if (marker == std::string::npos)
    return {};
  std::size_t begin = marker + before.size();
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])))
    ++begin;
  if (begin >= text.size())
    return {};
  const char quote = (text[begin] == '\'' || text[begin] == '"') ? text[begin++] : '\0';
  std::size_t end = begin;
  while (end < text.size()) {
    const char c = text[end];
    if ((quote != '\0' && c == quote) ||
        (quote == '\0' && (std::isspace(static_cast<unsigned char>(c)) || c == '\n')))
      break;
    ++end;
  }
  return text.substr(begin, end - begin);
}

NormalizedDiagnostic base(RawGstError raw, const char* code, std::string id, std::string title) {
  NormalizedDiagnostic out;
  out.error_code = code;
  out.diagnostic_id = std::move(id);
  out.title = std::move(title);
  out.stage = raw.source_name;
  out.raw = std::move(raw);
  return out;
}

void add_fact(NormalizedDiagnostic& diagnostic, std::string label, std::string value) {
  if (!value.empty())
    diagnostic.facts.push_back({std::move(label), std::move(value)});
}

std::string dimensions(const std::optional<int64_t>& width, const std::optional<int64_t>& height) {
  if (!width.has_value() || !height.has_value() || *width <= 0 || *height <= 0)
    return {};
  return std::to_string(*width) + "x" + std::to_string(*height);
}

NormalizedDiagnostic input_capacity(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kInputCapacity, "neatprocesscvu.input_envelope_exceeded",
           "The input stream is larger than this pipeline is configured to preprocess.");
  const auto actual_width = find_integer(out.raw, {"actual-width", "actual_width", "actual_w"});
  const auto actual_height = find_integer(out.raw, {"actual-height", "actual_height", "actual_h"});
  const auto actual_stride = find_integer(out.raw, {"actual-stride", "actual_stride"});
  const auto maximum_width = find_integer(out.raw, {"maximum-width", "maximum_width", "max_w"});
  auto maximum_height = find_integer(out.raw, {"maximum-height", "maximum_height", "max_h"});
  const auto maximum_stride =
      find_integer(out.raw, {"maximum-stride", "maximum_stride", "max_stride"});
  const auto resize_width = find_integer(out.raw, {"resize-width", "resize_width"});
  const auto resize_height = find_integer(out.raw, {"resize-height", "resize_height"});
  const std::string input_format =
      find_detail(out.raw, {"input-format", "input_format", "format"}).value_or("");

  const bool has_structured_maximum_height =
      out.raw.details.find("maximum-height") != out.raw.details.end() ||
      out.raw.details.find("maximum_height") != out.raw.details.end();
  const bool legacy_nv12_height =
      !has_structured_maximum_height && contains_ci(input_format, "nv12");
  if (legacy_nv12_height && maximum_width.has_value() && maximum_height.has_value() &&
      (*maximum_height * 2) % 3 == 0 && (*maximum_height * 2) / 3 == *maximum_width) {
    maximum_height = (*maximum_height * 2) / 3;
  }

  std::string actual = dimensions(actual_width, actual_height);
  if (!actual.empty() && !input_format.empty())
    actual += " " + input_format;
  if (!actual.empty() && actual_stride.has_value())
    actual += " (stride " + std::to_string(*actual_stride) + ")";
  add_fact(out, "Input stream", actual);
  add_fact(out, "Configured maximum", dimensions(maximum_width, maximum_height));
  add_fact(out, "Model resize target", dimensions(resize_width, resize_height));
  if (maximum_stride.has_value() && !maximum_width.has_value())
    add_fact(out, "Maximum stride", std::to_string(*maximum_stride));

  out.explanation =
      "The preprocessing input capacity and model resize target are different settings. "
      "The resize target can remain unchanged when input capacity is increased.";
  out.actions = {
      "Set `ModelOptions.preprocess.input_max_width` and `input_max_height` to cover the "
      "largest resolution the source can produce, then rebuild the pipeline.",
      "Scale the source to the configured maximum or smaller before the model stage.",
  };
  return out;
}

NormalizedDiagnostic axis_out_of_range(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kOptionOutOfRange, "neatargmax.axis_out_of_range",
           "`neatargmax` received an invalid reduction axis.");
  const auto axis = find_integer(out.raw, {"option-value", "option_value", "axis"});
  const auto rank = find_integer(out.raw, {"tensor-rank", "tensor_rank", "rank"});
  if (axis.has_value())
    add_fact(out, "Configured axis", std::to_string(*axis));
  if (rank.has_value()) {
    add_fact(out, "Input rank", std::to_string(*rank));
    if (*rank > 0 && *rank <= 8) {
      add_fact(out, "Valid axis range",
               "-" + std::to_string(*rank) + " through " + std::to_string(*rank - 1));
    }
  }
  out.actions = {"Set `axis` to a valid value for the input tensor rank."};
  return out;
}

NormalizedDiagnostic dtype_missing(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kTensorDtypeMissing, "neatargmax.tensor_dtype_missing",
           "The input tensor does not specify its data type.");
  const auto rank = find_integer(out.raw, {"tensor-rank", "tensor_rank", "rank"});
  if (rank.has_value())
    add_fact(out, "Input rank", std::to_string(*rank));
  out.actions = {
      "Provide the tensor `dtype` or `format` in the upstream tensor contract.",
      "Use a data type supported by `neatargmax`.",
  };
  return out;
}

NormalizedDiagnostic file_not_found(RawGstError raw) {
  NormalizedDiagnostic out = base(std::move(raw), error_codes::kFileNotFound,
                                  "gstreamer.file_not_found", "The input file does not exist.");
  const std::string path =
      find_detail(out.raw, {"source-identity", "source_identity", "path"}).value_or("");
  add_fact(out, "Path", path);
  out.actions = {
      "Correct the file path.",
      "Confirm that the file is present on the DevKit and readable by the application.",
  };
  return out;
}

NormalizedDiagnostic resource_not_found(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kIoOpen, "gstreamer.resource_not_found",
           "The requested input resource could not be found.");
  add_fact(out, "Source",
           find_detail(out.raw, {"source-identity", "source_identity", "location"}).value_or(""));
  out.actions = {
      "Verify the source address, path, and resource name.",
      "Confirm that the remote service or device exposes the requested resource.",
  };
  return out;
}

bool remote_source(const RawGstError& raw) {
  if (raw.factory_name == "rtspsrc" || contains_ci(raw.factory_name, "http") ||
      contains_ci(raw.factory_name, "soup")) {
    return true;
  }
  const std::string source =
      find_detail(raw, {"source-identity", "source_identity", "location"}).value_or("");
  return contains_ci(source, "rtsp://") || contains_ci(source, "http://") ||
         contains_ci(source, "https://");
}

NormalizedDiagnostic authentication_failed(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kIoOpen, "gstreamer.authentication_failed",
           "The remote source rejected the supplied credentials.");
  add_fact(out, "Source",
           find_detail(out.raw, {"source-identity", "source_identity", "location"}).value_or(""));
  out.actions = {
      "Verify the username, password, token, or other credentials configured for the source.",
      "Confirm that the account is allowed to access the requested stream or resource.",
  };
  return out;
}

NormalizedDiagnostic rtsp_connection_failed(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kRtspConnectionFailed, "gstreamer.rtsp_connection_failed",
           "Neat could not connect to the RTSP source.");
  add_fact(out, "Source",
           find_detail(out.raw, {"source-identity", "source_identity", "location"}).value_or(""));
  out.actions = {
      "Confirm that the RTSP server is running and verify the host, port, and mount path.",
      "Check network reachability from the DevKit.",
      "Verify credentials if the stream requires authentication.",
  };
  return out;
}

NormalizedDiagnostic camera_not_found(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kCameraNotFound, "gstreamer.camera_not_found",
           "The requested camera is not available.");
  std::string camera = quoted_subject(combined_raw(out.raw), "camera named");
  if (camera.empty())
    camera =
        find_detail(out.raw, {"source-identity", "source_identity", "camera-name"}).value_or("");
  add_fact(out, "Requested camera", camera);
  out.actions = {
      "List the cameras available on the DevKit and pass one of the reported names.",
      "Remove the explicit camera name to use the default camera.",
  };
  return out;
}

NormalizedDiagnostic invalid_h264(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kInvalidH264Stream, "gstreamer.invalid_h264_stream",
           "No valid H.264 frames were found in the input.");
  out.actions = {
      "Confirm that the source contains H.264 video and is not empty or truncated.",
      "Use decoder settings that match the source codec.",
  };
  return out;
}

NormalizedDiagnostic device_memory_exhausted(RawGstError raw) {
  NormalizedDiagnostic out = base(
      std::move(raw), error_codes::kDeviceMemoryExhausted, "neatprocesscvu.device_memory_exhausted",
      "There is not enough contiguous device memory to allocate the pipeline buffers.");
  add_fact(out, "Resource", "device DMA/CMA memory");
  const auto required = find_detail(out.raw, {"required-bytes", "required_bytes"});
  const auto available = find_detail(out.raw, {"available-bytes", "available_bytes"});
  if (required.has_value())
    add_fact(out, "Required bytes", *required);
  if (available.has_value())
    add_fact(out, "Available bytes", *available);
  out.actions = {
      "Reduce the number of simultaneous high-resolution streams or reduce an input resolution.",
      "Reduce buffer or queue depth where the application permits it.",
      "Stop other pipelines using device memory.",
  };
  return out;
}

NormalizedDiagnostic output_pool_exhausted(RawGstError raw) {
  const std::string plugin = find_detail(raw, {"plugin"}).value_or("gstreamer");
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kOutputPoolExhausted, plugin + ".output_pool_exhausted",
           "The pipeline output pool ran out of available buffers.");
  add_fact(out, "Pool size", find_detail(out.raw, {"pool-size", "pool_size"}).value_or(""));
  add_fact(out, "Buffers in use",
           find_detail(out.raw, {"pool-in-use", "pool_in_use"}).value_or(""));
  out.actions = {
      "Release zero-copy output tensors as soon as processing is complete.",
      "Use owned/copied output when results must be retained.",
      "Reduce the source rate or increase output-pool depth when supported.",
  };
  return out;
}

NormalizedDiagnostic dispatcher_unavailable(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kDispatcherUnavailable, "neat.dispatcher_unavailable",
           "The accelerator runtime is not available.");
  add_fact(
      out, "Target",
      find_detail(out.raw, {"dispatcher-target", "dispatcher_target", "run_target"}).value_or(""));
  out.actions = {
      "Confirm that the application is running on a compatible DevKit.",
      "Stop other workloads that may exclusively own the accelerator and retry the pipeline.",
  };
  return out;
}

NormalizedDiagnostic accelerator_failed(RawGstError raw) {
  NormalizedDiagnostic out = base(std::move(raw), error_codes::kAcceleratorExecutionFailed,
                                  "neat.accelerator_execution_failed",
                                  "The accelerator could not execute this model stage.");
  add_fact(
      out, "Target",
      find_detail(out.raw, {"dispatcher-target", "dispatcher_target", "run_target"}).value_or(""));
  add_fact(out, "Frame", find_detail(out.raw, {"frame-id", "frame_id"}).value_or(""));
  out.actions = {
      "Stop and restart the pipeline.",
      "Reduce concurrent accelerator workloads and retry this model stage.",
  };
  return out;
}

NormalizedDiagnostic media_caps(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kMediaCaps, "gstreamer.caps_incompatible",
           "Two connected pipeline stages require incompatible media caps.");
  add_fact(out, "Upstream caps",
           find_detail(out.raw, {"input-caps", "input_caps", "upstream-caps"}).value_or(""));
  add_fact(out, "Downstream caps",
           find_detail(out.raw, {"output-caps", "output_caps", "downstream-caps"}).value_or(""));
  out.actions = {
      "Configure both stages to use compatible formats, resolutions, and frame rates.",
      "Insert the appropriate video conversion, scaling, or rate-conversion stage.",
  };
  return out;
}

NormalizedDiagnostic buffer_too_small(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kBufferTooSmall, "neat.buffer_too_small",
           "An input buffer is smaller than the frame or tensor contract requires.");
  add_fact(out, "Required bytes",
           find_detail(out.raw, {"required-bytes", "required_bytes", "required"}).value_or(""));
  add_fact(out, "Available bytes",
           find_detail(out.raw, {"actual-bytes", "actual_bytes", "actual"}).value_or(""));
  out.actions = {
      "Correct the upstream frame/tensor dimensions and stride metadata.",
      "Allocate a buffer large enough for the declared payload.",
  };
  return out;
}

std::string tensor_contract_summary(const RawGstError& raw, bool expected) {
  const std::string shape =
      find_detail(raw,
                  expected
                      ? std::initializer_list<std::string_view>{"expected-shape", "expected_shape"}
                      : std::initializer_list<std::string_view>{"received-shape", "received_shape",
                                                                "actual-shape", "actual_shape"})
          .value_or("");
  const std::string dtype =
      find_detail(raw,
                  expected
                      ? std::initializer_list<std::string_view>{"expected-dtype", "expected_dtype"}
                      : std::initializer_list<std::string_view>{"received-dtype", "received_dtype",
                                                                "actual-dtype", "actual_dtype"})
          .value_or("");
  const std::string bytes =
      find_detail(raw, expected
                           ? std::initializer_list<std::string_view>{"required-bytes",
                                                                     "required_bytes", "required"}
                           : std::initializer_list<std::string_view>{"actual-bytes", "actual_bytes",
                                                                     "actual"})
          .value_or("");

  std::string summary;
  if (!shape.empty())
    summary = "shape " + shape;
  if (!dtype.empty()) {
    if (!summary.empty())
      summary += ", ";
    summary += "type " + dtype;
  }
  if (!bytes.empty()) {
    if (!summary.empty())
      summary += " (" + bytes + " bytes)";
    else
      summary = bytes + " bytes";
  }
  return summary;
}

NormalizedDiagnostic input_contract_mismatch(RawGstError raw) {
  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kInputShape, "neatprocesscvu.input_contract_mismatch",
           "The input tensor does not match the expected input contract.");
  std::string input =
      find_detail(out.raw, {"input-name", "input_name", "graph-input"}).value_or("");
  if (input.empty())
    input = quoted_subject(combined_raw(out.raw), "Direct graph input");
  add_fact(out, "Input tensor", input);
  add_fact(out, "Expected input", tensor_contract_summary(out.raw, true));
  add_fact(out, "Received input", tensor_contract_summary(out.raw, false));
  out.actions = {
      "Provide an input tensor with the expected shape and data type shown above.",
      "For model input, configure the expected preprocessing input through `Model::Options` "
      "(C++) or `ModelOptions` (Python).",
  };
  return out;
}

} // namespace

RawGstError parse_gst_error_message(GstMessage* message) {
  RawGstError out;
  out.wall_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
  if (!message)
    return out;

  if (GST_MESSAGE_SRC(message) && GST_IS_OBJECT(GST_MESSAGE_SRC(message))) {
    const char* source_name = GST_OBJECT_NAME(GST_MESSAGE_SRC(message));
    out.source_name = source_name ? source_name : "";
    if (GST_IS_ELEMENT(GST_MESSAGE_SRC(message))) {
      GstElementFactory* factory = gst_element_get_factory(GST_ELEMENT(GST_MESSAGE_SRC(message)));
      if (factory) {
        const char* factory_name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
        out.factory_name = factory_name ? factory_name : "";
      }
      if (out.factory_name == "filesrc" || out.factory_name == "rtspsrc") {
        const std::string location =
            readable_string_property(G_OBJECT(GST_MESSAGE_SRC(message)), "location");
        if (!location.empty())
          out.details.emplace("source-identity", location);
      } else if (out.factory_name == "libcamerasrc") {
        std::string camera =
            readable_string_property(G_OBJECT(GST_MESSAGE_SRC(message)), "camera-name");
        if (camera.empty())
          camera = readable_string_property(G_OBJECT(GST_MESSAGE_SRC(message)), "camera");
        if (!camera.empty())
          out.details.emplace("source-identity", std::move(camera));
      }
    }
  }

  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &error, &debug);
  if (error) {
    const char* domain = g_quark_to_string(error->domain);
    out.domain_name = domain ? domain : "";
    out.code = error->code;
    out.message = truncate(error->message ? error->message : "");
    g_error_free(error);
  }
  out.debug = truncate(debug ? debug : "");
  g_free(debug);

  const GstStructure* details = nullptr;
  gst_message_parse_error_details(message, &details);
  if (details) {
    const int fields = gst_structure_n_fields(details);
    for (int i = 0; i < fields; ++i) {
      const char* name = gst_structure_nth_field_name(details, i);
      if (!name)
        continue;
      const std::string value =
          sensitive_detail_name(name)
              ? "<redacted>"
              : truncate(value_to_string(gst_structure_get_value(details, name)));
      out.details.emplace(name, value);
    }
  }
  return out;
}

std::string sanitize_gst_diagnostic_text(std::string text) {
  return truncate(std::move(text), 16384);
}

NormalizedDiagnostic classify_gst_error(RawGstError raw) {
  const std::string text = combined_raw(raw);
  const std::string diagnostic_id =
      find_detail(raw, {"neat-diagnostic-id", "neat_diagnostic_id", "diagnostic_id"}).value_or("");
  const std::string reason =
      find_detail(raw, {"neat-reason", "neat_reason", "reason"}).value_or("");

  if (diagnostic_id == "neatprocesscvu.input_envelope_exceeded" ||
      contains_ci(text, "envelope violation")) {
    return input_capacity(std::move(raw));
  }
  if (diagnostic_id == "neatargmax.axis_out_of_range" ||
      (contains_ci(text, "field=axis") && contains_ci(text, "reason=out_of_range"))) {
    return axis_out_of_range(std::move(raw));
  }
  if (diagnostic_id == "neatargmax.tensor_dtype_missing" ||
      contains_ci(text, "field=dtype|format")) {
    return dtype_missing(std::move(raw));
  }
  if (diagnostic_id.find("output_pool_exhausted") != std::string::npos ||
      contains_ci(text, "failed to acquire output buffer from pool") ||
      contains_ci(text, "output pool starved")) {
    return output_pool_exhausted(std::move(raw));
  }
  if (diagnostic_id.find("device_memory_exhausted") != std::string::npos ||
      contains_ci(text, "failed to allocate output buffer pool") ||
      contains_ci(text, "failed to allocate overflow output buffer")) {
    return device_memory_exhausted(std::move(raw));
  }
  if (diagnostic_id.find("dispatcher_unavailable") != std::string::npos ||
      contains_ci(text, "unable to get dispatcher")) {
    return dispatcher_unavailable(std::move(raw));
  }
  if (diagnostic_id.find("accelerator_execution_failed") != std::string::npos ||
      contains_ci(text, "dispatcher run failed") ||
      contains_ci(text, "dispatcher async run failed")) {
    return accelerator_failed(std::move(raw));
  }
  if (diagnostic_id == "neatprocesscvu.input_contract_mismatch" ||
      (raw.factory_name == "neatprocesscvu" && contains_ci(text, "direct graph input") &&
       contains_ci(text, "segment") && contains_ci(text, "too small"))) {
    return input_contract_mismatch(std::move(raw));
  }
  if (diagnostic_id.find("buffer_too_small") != std::string::npos ||
      contains_ci(text, "buffer is too small") ||
      (contains_ci(text, "segment") && contains_ci(text, "too small"))) {
    return buffer_too_small(std::move(raw));
  }

  if (raw.factory_name == "filesrc" && raw.domain_name == "gst-resource-error-quark" &&
      raw.code == GST_RESOURCE_ERROR_NOT_FOUND) {
    return file_not_found(std::move(raw));
  }
  if (raw.domain_name == "gst-resource-error-quark" &&
      raw.code == GST_RESOURCE_ERROR_NOT_AUTHORIZED && remote_source(raw)) {
    return authentication_failed(std::move(raw));
  }
  if (raw.factory_name == "rtspsrc" &&
      (contains_ci(text, "failed to connect") || contains_ci(text, "could not open resource"))) {
    return rtsp_connection_failed(std::move(raw));
  }
  if ((raw.factory_name == "libcamerasrc" || contains_ci(raw.factory_name, "camera")) &&
      contains_ci(text, "could not find a camera")) {
    return camera_not_found(std::move(raw));
  }
  if (raw.factory_name == "h264parse" && contains_ci(text, "no valid frames")) {
    return invalid_h264(std::move(raw));
  }

  if (raw.domain_name == "gst-resource-error-quark") {
    if (raw.code == GST_RESOURCE_ERROR_NOT_FOUND)
      return resource_not_found(std::move(raw));
    if (raw.code == GST_RESOURCE_ERROR_NOT_AUTHORIZED ||
        (raw.code == GST_RESOURCE_ERROR_OPEN_READ && contains_ci(text, "permission denied"))) {
      NormalizedDiagnostic out =
          base(std::move(raw), error_codes::kPermissionDenied, "gstreamer.permission_denied",
               "The input resource could not be opened with the required permissions.");
      out.actions = {
          "Confirm that the application user can read the file or device.",
          "Correct the source permissions and retry.",
      };
      return out;
    }
    if (raw.code == GST_RESOURCE_ERROR_NO_SPACE_LEFT) {
      NormalizedDiagnostic out =
          base(std::move(raw), error_codes::kDiskFull, "gstreamer.storage_full",
               "The output resource does not have enough free space.");
      out.actions = {"Free space on the destination or choose another output location."};
      return out;
    }
  }

  if ((raw.domain_name == "gst-core-error-quark" &&
       (raw.code == GST_CORE_ERROR_NEGOTIATION || raw.code == GST_CORE_ERROR_CAPS)) ||
      contains_ci(text, "not-negotiated") || contains_ci(text, "caps negotiation failed")) {
    return media_caps(std::move(raw));
  }
  if (raw.domain_name == "gst-stream-error-quark" && raw.code == GST_STREAM_ERROR_DECODE) {
    NormalizedDiagnostic out =
        base(std::move(raw), error_codes::kDecodeFailed, "gstreamer.decode_failed",
             "The decoder could not decode the input stream.");
    out.actions = {
        "Confirm that the configured codec matches the input stream.",
        "Check that the encoded data is complete and not corrupted.",
    };
    return out;
  }
  if (raw.domain_name == "gst-stream-error-quark" && raw.code == GST_STREAM_ERROR_ENCODE) {
    NormalizedDiagnostic out =
        base(std::move(raw), error_codes::kEncodeFailed, "gstreamer.encode_failed",
             "The encoder could not encode the input frames.");
    out.actions = {"Verify the encoder format, resolution, and bitrate settings."};
    return out;
  }

  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kRuntimeElementFailed,
           diagnostic_id.empty() ? "gstreamer.unclassified_element_failure" : diagnostic_id,
           "A pipeline stage stopped while processing data.");
  if (safe_production_reason(reason))
    add_fact(out, "Reason", reason);
  else if (safe_production_reason(out.raw.message))
    add_fact(out, "Reason", out.raw.message);
  out.actions = {
      "Correct the configuration of the reported stage and its upstream input.",
      "Restart the pipeline after correcting the first upstream failure.",
  };
  return out;
}

NormalizedDiagnostic classify_gst_parse_error(const GError* error,
                                              const std::string& pipeline_string) {
  RawGstError raw;
  if (error) {
    const char* domain = g_quark_to_string(error->domain);
    raw.domain_name = domain ? domain : "";
    raw.code = error->code;
    raw.message = truncate(error->message ? error->message : "");
  }
  raw.details["pipeline"] = truncate(pipeline_string);

  if (error && error->domain == GST_PARSE_ERROR) {
    if (error->code == GST_PARSE_ERROR_NO_SUCH_ELEMENT) {
      NormalizedDiagnostic out =
          base(std::move(raw), error_codes::kPluginMissing, "gstreamer.element_missing",
               "A required GStreamer element is not installed.");
      add_fact(out, "Missing element", quoted_subject(out.raw.message, "no element"));
      out.actions = {
          "Install the package that provides the missing element or replace it with an available "
          "element.",
          "Check availability with `gst-inspect-1.0 <element>`.",
      };
      return out;
    }
    if (error->code == GST_PARSE_ERROR_NO_SUCH_PROPERTY) {
      NormalizedDiagnostic out =
          base(std::move(raw), error_codes::kPropertyInvalid, "gstreamer.property_invalid",
               "A GStreamer element property is unknown or invalid.");
      add_fact(out, "Parser message", out.raw.message);
      out.actions = {"Correct the property name/value using `gst-inspect-1.0 <element>`."};
      return out;
    }
    if (error->code == GST_PARSE_ERROR_SYNTAX) {
      NormalizedDiagnostic out =
          base(std::move(raw), error_codes::kPipelineSyntax, "gstreamer.pipeline_syntax",
               "The custom GStreamer pipeline fragment contains invalid syntax.");
      out.actions = {
          "Correct the fragment syntax.",
          "Validate the fragment with `gst-launch-1.0` before adding it to the graph.",
      };
      return out;
    }
  }

  NormalizedDiagnostic out =
      base(std::move(raw), error_codes::kParseLaunch, "gstreamer.parse_launch_failed",
           "GStreamer could not build the generated pipeline.");
  add_fact(out, "Parser message", out.raw.message);
  out.actions = {
      "Check the custom pipeline fragment and the properties of its elements.",
      "Verify that every required GStreamer plugin is installed.",
  };
  return out;
}

std::string render_diagnostic_body(const NormalizedDiagnostic& diagnostic,
                                   bool include_debug_details) {
  std::ostringstream out;
  out << diagnostic.title;
  if (!diagnostic.stage.empty())
    out << "\n\nStage: " << diagnostic.stage;
  if (!diagnostic.source.empty())
    out << "\nSource: " << diagnostic.source;
  for (const DiagnosticFact& fact : diagnostic.facts) {
    if (!fact.label.empty() && !fact.value.empty())
      out << "\n" << fact.label << ": " << fact.value;
  }
  if (!diagnostic.explanation.empty())
    out << "\n\n" << diagnostic.explanation;
  if (!diagnostic.actions.empty()) {
    out << "\n\nHow to fix:";
    for (const std::string& action : diagnostic.actions)
      out << "\n- " << action;
  }
  if (!diagnostic.diagnostic_id.empty())
    out << "\n\nDiagnostic ID: " << diagnostic.diagnostic_id;

  if (include_debug_details) {
    out << "\n\nTechnical details:";
    if (!diagnostic.raw.domain_name.empty())
      out << "\nGStreamer error: " << diagnostic.raw.domain_name << "/" << diagnostic.raw.code;
    if (!diagnostic.raw.factory_name.empty())
      out << "\nElement factory: " << diagnostic.raw.factory_name;
    if (!diagnostic.raw.message.empty())
      out << "\nMessage: " << diagnostic.raw.message;
    if (!diagnostic.raw.debug.empty())
      out << "\nDebug: " << diagnostic.raw.debug;
    for (const auto& [key, value] : diagnostic.raw.details) {
      if (key == "pipeline")
        continue;
      out << "\n" << key << ": " << value;
    }
  }
  return out.str();
}

int diagnostic_priority(const NormalizedDiagnostic& diagnostic) {
  if (find_detail(diagnostic.raw, {"neat-diagnostic-id", "neat_diagnostic_id", "diagnostic_id"})
          .has_value()) {
    return 200;
  }

  if (diagnostic.error_code.rfind("resource.", 0) == 0 ||
      diagnostic.error_code.rfind("io.", 0) == 0 || diagnostic.error_code.rfind("codec.", 0) == 0 ||
      diagnostic.error_code.rfind("infra.", 0) == 0) {
    return 160;
  }
  if (diagnostic.error_code == error_codes::kInputCapacity ||
      diagnostic.error_code == error_codes::kInputShape ||
      diagnostic.error_code == error_codes::kTensorDtypeMissing ||
      diagnostic.error_code == error_codes::kOptionOutOfRange ||
      diagnostic.error_code == error_codes::kBufferTooSmall) {
    return 150;
  }
  if (diagnostic.error_code == error_codes::kMediaCaps)
    return 80;
  if (diagnostic.diagnostic_id.rfind("gstreamer.unclassified", 0) != 0 &&
      diagnostic.diagnostic_id.rfind("gstreamer.parse_launch_failed", 0) != 0) {
    return 100;
  }
  return 10;
}

} // namespace simaai::neat::pipeline_internal

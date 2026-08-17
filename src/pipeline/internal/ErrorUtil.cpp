#include "pipeline/internal/ErrorUtil.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/GstErrorNormalizer.h"

namespace simaai::neat::pipeline_internal::error_util {
namespace {

std::string to_owned(std::string_view s) {
  return std::string(s.begin(), s.end());
}

std::optional<std::string> leading_error_code(std::string_view message) {
  if (message.empty() || message.front() != '[')
    return std::nullopt;
  const std::size_t close = message.find(']');
  if (close == std::string_view::npos || close <= 1)
    return std::nullopt;
  return std::string(message.substr(1, close - 1));
}

bool is_framework_error_code(std::string_view code) {
  static constexpr std::string_view known_codes[] = {
      error_codes::kPipelineShape,
      error_codes::kCaps,
      error_codes::kInputShape,
      error_codes::kRuntimeAbiMismatch,
      error_codes::kGraphElementName,
      error_codes::kMediaCaps,
      error_codes::kMediaFormat,
      error_codes::kInputCapacity,
      error_codes::kTensorDtypeMissing,
      error_codes::kOptionOutOfRange,
      error_codes::kParseLaunch,
      error_codes::kPipelineSyntax,
      error_codes::kPluginMissing,
      error_codes::kPropertyInvalid,
      error_codes::kRuntimePull,
      error_codes::kRuntimeElementFailed,
      error_codes::kOutputTimeout,
      error_codes::kUnexpectedEos,
      error_codes::kIoParse,
      error_codes::kIoOpen,
      error_codes::kFileNotFound,
      error_codes::kPermissionDenied,
      error_codes::kRtspConnectionFailed,
      error_codes::kCameraNotFound,
      error_codes::kModelNotFound,
      error_codes::kSourceEnded,
      error_codes::kInvalidH264Stream,
      error_codes::kDecodeFailed,
      error_codes::kEncodeFailed,
      error_codes::kMemoryAllocationFailed,
      error_codes::kDeviceMemoryExhausted,
      error_codes::kOutputPoolExhausted,
      error_codes::kBufferTooSmall,
      error_codes::kDiskFull,
      error_codes::kDispatcherUnavailable,
      error_codes::kAcceleratorExecutionFailed,
      error_codes::kInternalPluginFailure,
      error_codes::kDispatcherUnavailableLegacy,
  };
  for (const std::string_view known : known_codes) {
    if (code == known) {
      return true;
    }
  }
  return false;
}

std::optional<std::string> leading_framework_error_code(std::string_view message) {
  const std::optional<std::string> code = leading_error_code(message);
  if (!code.has_value() || !is_framework_error_code(*code)) {
    return std::nullopt;
  }
  return code;
}

} // namespace

std::string decorate_error(std::string_view code, std::string_view message) {
  if (code.empty())
    return to_owned(message);
  const std::string prefix = "[" + to_owned(code) + "]";
  if (message.size() >= prefix.size() && message.substr(0, prefix.size()) == prefix &&
      (message.size() == prefix.size() || message[prefix.size()] == ' ')) {
    return to_owned(message);
  }

  return prefix + " " + to_owned(message);
}

std::string append_hint(std::string_view message, std::string_view hint) {
  if (hint.empty())
    return to_owned(message);
  return to_owned(message) + "\nHint: " + to_owned(hint);
}

GraphReport make_report(std::string_view code, std::string_view summary,
                        std::string_view pipeline_string, std::string_view hint) {
  GraphReport rep;
  rep.error_code = to_owned(code);
  rep.pipeline_string = redact_gst_credentials(to_owned(pipeline_string));
  if (!rep.pipeline_string.empty()) {
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + rep.pipeline_string + "'";
  }
  rep.repro_note = redact_gst_credentials(append_hint(summary, hint));
  return rep;
}

[[noreturn]] void throw_session_error(std::string_view code, std::string_view summary,
                                      std::string_view pipeline_string, std::string_view hint) {
  GraphReport rep = make_report(code, summary, pipeline_string, hint);
  const std::string msg = decorate_error(rep.error_code, rep.repro_note);
  throw NeatError(msg, std::move(rep));
}

void set_pull_error(PullError* err, std::string code, std::string message,
                    std::optional<GraphReport> report) {
  if (!err)
    return;
  if (const auto existing_code = leading_framework_error_code(message); existing_code.has_value()) {
    code = *existing_code;
  }
  err->code = std::move(code);

  if (report.has_value()) {
    if (err->code.empty()) {
      err->code = report->error_code;
    } else if (report->error_code.empty()) {
      report->error_code = err->code;
    }
    err->report = std::move(report);
  } else {
    err->report.reset();
  }

  err->message = leading_framework_error_code(message).has_value()
                     ? std::move(message)
                     : decorate_error(err->code, message);
}

} // namespace simaai::neat::pipeline_internal::error_util

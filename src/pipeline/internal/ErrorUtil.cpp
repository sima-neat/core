#include "pipeline/internal/ErrorUtil.h"

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

} // namespace

std::string decorate_error(std::string_view code, std::string_view message) {
  if (code.empty())
    return to_owned(message);
  const std::string prefix = "[" + to_owned(code) + "]";
  if (message.rfind(prefix, 0) == 0)
    return to_owned(message);
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
  rep.pipeline_string = to_owned(pipeline_string);
  if (!rep.pipeline_string.empty()) {
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + rep.pipeline_string + "'";
  }
  rep.repro_note = append_hint(summary, hint);
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
  if (const auto existing_code = leading_error_code(message); existing_code.has_value()) {
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

  err->message = leading_error_code(message).has_value() ? std::move(message)
                                                         : decorate_error(err->code, message);
}

} // namespace simaai::neat::pipeline_internal::error_util

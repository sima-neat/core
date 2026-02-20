#include "pipeline/internal/ErrorUtil.h"

namespace simaai::neat::pipeline_internal::error_util {
namespace {

std::string to_owned(std::string_view s) {
  return std::string(s.begin(), s.end());
}

} // namespace

std::string decorate_error(std::string_view code, std::string_view message) {
  if (code.empty())
    return to_owned(message);
  return "[" + to_owned(code) + "] " + to_owned(message);
}

std::string append_hint(std::string_view message, std::string_view hint) {
  if (hint.empty())
    return to_owned(message);
  return to_owned(message) + "\nHint: " + to_owned(hint);
}

SessionReport make_report(std::string_view code, std::string_view summary,
                          std::string_view pipeline_string, std::string_view hint) {
  SessionReport rep;
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
  SessionReport rep = make_report(code, summary, pipeline_string, hint);
  const std::string msg = decorate_error(rep.error_code, rep.repro_note);
  throw SessionError(msg, std::move(rep));
}

void set_pull_error(PullError* err, std::string code, std::string message,
                    std::optional<SessionReport> report) {
  if (!err)
    return;
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

  err->message = decorate_error(err->code, message);
}

} // namespace simaai::neat::pipeline_internal::error_util

#include "SshRunner.h"

#include <algorithm>
#include <chrono>
#include <string>

namespace simaai::neat::pcie::internal {

std::string SshRunner::shell_escape(const std::string& value) {
  std::string out = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out += "'";
  return out;
}

std::string SshRunner::windows_quote_argument(const std::string& value) {
  const auto requires_quotes = std::any_of(value.begin(), value.end(), [](const char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '"';
  });
  if (!value.empty() && !requires_quotes) {
    return value;
  }

  std::string out = "\"";
  std::size_t backslashes = 0;
  for (const char ch : value) {
    if (ch == '\\') {
      ++backslashes;
      continue;
    }
    if (ch == '"') {
      out.append(backslashes * 2 + 1, '\\');
      out.push_back('"');
    } else {
      out.append(backslashes, '\\');
      out.push_back(ch);
    }
    backslashes = 0;
  }
  out.append(backslashes * 2, '\\');
  out.push_back('"');
  return out;
}

std::string SshRunner::failure_detail(const std::vector<std::string>& args,
                                      const CommandResult& result) {
  std::string detail = "(exit=" + std::to_string(result.exit_code) +
                       ", timed_out=" + (result.timed_out ? "true" : "false") + "): ";
  if (!result.output.empty() || args.empty()) {
    return detail + result.output;
  }
  detail += args.front() + " printed no output";
#ifdef _WIN32
  // Windows OpenSSH exits 255 without a message when ProgramData is unset.
  if (result.exit_code == 255) {
    detail += "; check that the ProgramData environment variable is set for this process";
  }
#endif
  return detail;
}

CommandResult SshRunner::run(const std::vector<std::string>& args, const int timeout_sec) {
  const int effective_timeout = timeout_sec > 0 ? timeout_sec : 1;
  return run_for(args, std::chrono::seconds(effective_timeout));
}

} // namespace simaai::neat::pcie::internal

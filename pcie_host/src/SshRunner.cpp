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

CommandResult SshRunner::run(const std::vector<std::string>& args, const int timeout_sec) {
  const int effective_timeout = timeout_sec > 0 ? timeout_sec : 1;
  return run_for(args, std::chrono::seconds(effective_timeout));
}

} // namespace simaai::neat::pcie::internal

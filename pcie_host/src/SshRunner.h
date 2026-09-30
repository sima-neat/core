#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace simaai::neat::pcie::internal {

struct CommandResult {
  int exit_code = -1;
  bool timed_out = false;
  bool aborted = false; // should_abort() turned true; the command was killed
  std::string output;
};

class SshRunner {
public:
  static CommandResult run(const std::vector<std::string>& args, int timeout_sec);
  /// Like run(), with a millisecond timeout. If should_abort is set, it is
  /// checked about every 200 ms; when it turns true the command is killed at
  /// once and the result has aborted = true (not timed_out).
  static CommandResult run_for(const std::vector<std::string>& args,
                               std::chrono::milliseconds timeout,
                               const std::function<bool()>& should_abort = {});
  static std::string shell_escape(const std::string& value);
};

} // namespace simaai::neat::pcie::internal

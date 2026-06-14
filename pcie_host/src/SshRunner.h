#pragma once

#include <string>
#include <vector>

namespace simaai::neat::pcie::internal {

struct CommandResult {
  int exit_code = -1;
  bool timed_out = false;
  std::string output;
};

class SshRunner {
public:
  static CommandResult run(const std::vector<std::string>& args, int timeout_sec);
  static std::string shell_escape(const std::string& value);
};

} // namespace simaai::neat::pcie::internal


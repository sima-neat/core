#include "pcie_genai_history.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace fs = std::filesystem;
using simaai::neat::pcie::genai::tools::history_file_path;
using simaai::neat::pcie::genai::tools::should_record_history;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    // --- history_file_path: honours $HOME, falls back to "." -----------------
    {
      ::setenv("HOME", "/home/tester", 1);
      require(history_file_path() == fs::path("/home/tester") / ".pcie_genai_history",
              "history lives under $HOME");

      // No HOME (or empty): keep the file in the current directory rather than
      // writing to "/". A daemon or a stripped env must not lose the history to
      // a path nobody can read.
      ::unsetenv("HOME");
      require(history_file_path() == fs::path(".") / ".pcie_genai_history",
              "no HOME: history lives in the current directory");

      ::setenv("HOME", "", 1);
      require(history_file_path() == fs::path(".") / ".pcie_genai_history",
              "empty HOME: history lives in the current directory");

      ::setenv("HOME", "/home/tester", 1); // restore for any later checks
    }

    // --- should_record_history: skip empty and consecutive duplicates --------
    {
      require(!should_record_history("", "anything"), "an empty line is never recorded");
      require(!should_record_history("", ""), "an empty line with no history is not recorded");
      require(should_record_history("hello", ""), "the first real line is recorded");
      require(should_record_history("hello", "world"),
              "a new line after a different one is recorded");
      require(!should_record_history("hello", "hello"), "a line equal to the last one is skipped");
      // Only the immediately-previous line matters: an older duplicate still records.
      require(should_record_history("hello", "goodbye"), "not a consecutive duplicate: recorded");
    }

    std::cout << "[PASS] pcie-genai history helper\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

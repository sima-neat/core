#include "ModelArchiveSnapshot.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

class TempDirectory {
public:
  TempDirectory() {
    std::string directory_template =
        (fs::temp_directory_path() / "sima-neat-pcie-snapshot-test-XXXXXX").string();
    std::vector<char> writable_template(directory_template.begin(), directory_template.end());
    writable_template.push_back('\0');
    const char* created = ::mkdtemp(writable_template.data());
    if (!created) {
      throw std::runtime_error("failed to create test directory");
    }
    path_ = created;
  }

  ~TempDirectory() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  const fs::path& path() const {
    return path_;
  }

private:
  fs::path path_;
};

void write_file(const fs::path& path, const std::string& contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << contents;
  if (!output) {
    throw std::runtime_error("failed to write test file: " + path.string());
  }
}

std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_replacing_source_does_not_change_snapshot() {
  TempDirectory temp;
  const fs::path source = temp.path() / "model.tar.gz";
  write_file(source, "archive-a");

  fs::path snapshot_directory;
  {
    pcie_internal::ModelArchiveSnapshot snapshot(source.string());
    snapshot_directory = fs::path(snapshot.path()).parent_path();
    const fs::path replacement = temp.path() / "replacement.tar.gz";
    write_file(replacement, "archive-b");
    fs::rename(replacement, source);
    require(read_file(snapshot.path()) == "archive-a",
            "replacing the source changed the model snapshot");
  }
  require(!fs::exists(snapshot_directory), "model snapshot directory was not removed");
}

void test_retargeting_symlink_does_not_change_snapshot() {
  TempDirectory temp;
  const fs::path first = temp.path() / "first.tar.gz";
  const fs::path second = temp.path() / "second.tar.gz";
  const fs::path source = temp.path() / "model.tar.gz";
  write_file(first, "archive-a");
  write_file(second, "archive-b");
  fs::create_symlink(first, source);

  pcie_internal::ModelArchiveSnapshot snapshot(source.string());
  fs::remove(source);
  fs::create_symlink(second, source);
  require(read_file(snapshot.path()) == "archive-a",
          "retargeting the source symlink changed the model snapshot");
}

} // namespace

int main() {
  try {
    test_replacing_source_does_not_change_snapshot();
    test_retargeting_symlink_does_not_change_snapshot();
    std::cout << "[PASS] model archive snapshot\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

#include "HostPlatform.h"
#include "ModelArchiveSnapshot.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

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
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
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
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
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

void test_spaces_and_unicode_source_path_use_safe_snapshot_name() {
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
  const fs::path source_directory =
      temp.path() / pcie_internal::host_path_from_string("customer models \xCE\xB4");
  fs::create_directories(source_directory);
  const fs::path source =
      source_directory / pcie_internal::host_path_from_string("model archive \xCE\xB4.tar.gz");
  write_file(source, "archive-a");

  pcie_internal::ModelArchiveSnapshot snapshot(pcie_internal::host_path_string(source));
  const fs::path snapshot_path = pcie_internal::host_path_from_string(snapshot.path());
#ifdef _WIN32
  require(snapshot_path.filename() == "model.tar.gz",
          "private archive snapshot name was not normalized");
#else
  require(snapshot_path.filename() == source.filename(),
          "POSIX archive snapshot name unexpectedly changed");
#endif
  require(read_file(snapshot_path) == "archive-a",
          "spaces or Unicode changed the model snapshot contents");
}

void test_non_tar_gz_source_is_rejected_before_snapshot() {
#ifdef _WIN32
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
  const fs::path source = temp.path() / "model.zip";
  write_file(source, "archive-a");

  bool rejected = false;
  try {
    (void)pcie_internal::ModelArchiveSnapshot(pcie_internal::host_path_string(source));
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()).find("must use .tar.gz") != std::string::npos;
  }
  require(rejected, "a non-.tar.gz source acquired a normalized snapshot name");
#endif
}

} // namespace

int main() {
  try {
    test_replacing_source_does_not_change_snapshot();
    test_retargeting_symlink_does_not_change_snapshot();
    test_spaces_and_unicode_source_path_use_safe_snapshot_name();
    test_non_tar_gz_source_is_rejected_before_snapshot();
    std::cout << "[PASS] model archive snapshot\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

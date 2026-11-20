#include "HostPlatform.h"
#include "ModelArchiveSnapshot.h"
#include "model/internal/ModelArchiveLoader.h"

#include <zlib.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace pcie_internal = simaai::neat::pcie::internal;
namespace loader = simaai::neat::internal;

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

// One ustar member, as tests/tools/make_model_archive_fixtures.py writes them.
std::string tar_member(const std::string& name, const std::string& data) {
  std::string header(512, '\0');
  const auto put = [&header](const std::size_t offset, const std::string& value) {
    header.replace(offset, value.size(), value);
  };
  char size[12];
  std::snprintf(size, sizeof(size), "%011o", static_cast<unsigned>(data.size()));
  put(0, name);
  put(100, "0000644");
  put(108, "0000000");
  put(116, "0000000");
  put(124, size);
  put(136, "00000000000");
  put(148, "        ");
  put(156, "0");
  put(257, "ustar");
  put(263, "00");
  unsigned checksum = 0;
  for (const unsigned char byte : header) {
    checksum += byte;
  }
  char field[8];
  std::snprintf(field, sizeof(field), "%06o", checksum);
  put(148, std::string(field) + '\0' + ' ');
  return header + data + std::string((512 - data.size() % 512) % 512, '\0');
}

std::string gzip(const std::string& data) {
  z_stream stream{};
  require(deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 16 + MAX_WBITS, 8,
                       Z_DEFAULT_STRATEGY) == Z_OK,
          "failed to start gzip");
  std::string out(deflateBound(&stream, static_cast<uLong>(data.size())), '\0');
  stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
  stream.avail_in = static_cast<uInt>(data.size());
  stream.next_out = reinterpret_cast<Bytef*>(out.data());
  stream.avail_out = static_cast<uInt>(out.size());
  const int status = deflate(&stream, Z_FINISH);
  out.resize(stream.total_out);
  deflateEnd(&stream);
  require(status == Z_STREAM_END, "failed to gzip the test archive");
  return out;
}

// The archive, staging and output directories sit under a name no single ANSI code page holds,
// as under a non-ASCII Windows profile or TEMP.
void test_loader_extracts_under_unicode_directory() {
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
  const fs::path directory =
      temp.path() / pcie_internal::host_path_from_string("Jos\xC3\xA9 \xCE\xB4");
  fs::create_directories(directory);
  const fs::path archive = directory / "model.tar.gz";
  write_file(archive, gzip(tar_member("etc/model_mpk.json", "{}") +
                           tar_member("share/model.elf", "elf") + std::string(1024, '\0')));

  loader::ModelArchiveLoaderOptions options;
  options.staging_base = pcie_internal::host_path_string(directory);
  options.check_output_free_space = false;
  const auto extracted = loader::ModelArchiveLoader::extract(
      pcie_internal::host_path_string(archive), pcie_internal::host_path_string(directory / "out"),
      options);
  const fs::path package_root = pcie_internal::host_path_from_string(extracted.package_root);
  require(package_root == directory / "out" / "model",
          "loader returned a different package root: " + extracted.package_root);
  require(read_file(package_root / "etc" / "model_mpk.json") == "{}",
          "loader did not extract the MPK manifest under a Unicode directory");
  require(read_file(package_root / "share" / "model.elf") == "elf",
          "loader did not extract the model binary under a Unicode directory");
}

// cmd.exe would run a tar.bat from the current directory before the tar Windows ships.
void test_loader_ignores_tar_in_current_directory() {
#ifdef _WIN32
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
  write_file(temp.path() / "tar.bat", "@exit /b 7\r\n");
  const fs::path archive = temp.path() / "model.tar.gz";
  write_file(archive, gzip(tar_member("etc/model_mpk.json", "{}") +
                           tar_member("share/model.elf", "elf") + std::string(1024, '\0')));
  const fs::path previous = fs::current_path();
  fs::current_path(temp.path());
  try {
    loader::ModelArchiveLoaderOptions options;
    options.staging_base = pcie_internal::host_path_string(temp.path());
    options.check_output_free_space = false;
    const auto extracted = loader::ModelArchiveLoader::extract(
        pcie_internal::host_path_string(archive),
        pcie_internal::host_path_string(temp.path() / "out"), options);
    fs::current_path(previous);
    require(read_file(pcie_internal::host_path_from_string(extracted.package_root) / "etc" /
                      "model_mpk.json") == "{}",
            "loader did not extract with a tar.bat in the current directory");
  } catch (...) {
    fs::current_path(previous);
    throw;
  }
#endif
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
    test_loader_extracts_under_unicode_directory();
    test_loader_ignores_tar_in_current_directory();
    std::cout << "[PASS] model archive snapshot\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

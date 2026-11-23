#include "HostPlatform.h"
#include "ModelArchiveSnapshot.h"
#include "PcieModelFactsReader.h"
#include "model/internal/ModelArchiveLoader.h"

#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <thread>
#endif

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

// A cast, one MLA stage, a cast and a pass-through: the smallest model the strict decoder admits.
constexpr const char* kStrictManifest = R"json({
  "name":"synthetic","model_sdk_version":"2.0.0","input_nodes":[{"name":"input","size":16}],
  "plugins":[
    {"name":"cast0","sequence":1,"processor":"EV74","type":"sgpProcess",
     "config_params":{"desired_batch_size":1,"actual_batch_size":1,"kernel":"cast_transform",
       "params":{"out_dtype":"bfloat16","input_shapes":[[1,4]],"output_shapes":[[1,4]]}},
     "input_nodes":[{"name":"input","size":16}],"output_nodes":[{"name":"cast0","size":8}]},
    {"name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
     "config_params":{"desired_batch_size":1,"actual_batch_size":1,"number_of_quads_to_user":4},
     "input_nodes":[{"name":"cast0","size":8}],"output_nodes":[{"name":"mla0","size":8}],
     "resources":{"executable":"synthetic_mla.elf"}},
    {"name":"cast1","sequence":3,"processor":"EV74","type":"sgpProcess",
     "config_params":{"desired_batch_size":1,"actual_batch_size":1,"kernel":"cast_transform",
       "params":{"out_dtype":"float32","input_shapes":[[1,4]],"output_shapes":[[1,4]]}},
     "input_nodes":[{"name":"mla0","size":8}],"output_nodes":[{"name":"output","size":16}]},
    {"name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
     "config_params":{"desired_batch_size":1,"actual_batch_size":1,"kernel":"pass_through",
       "params":{}},
     "input_nodes":[{"name":"output","size":16}],
     "output_nodes":[{"name":"pass_through_out_0","size":16}]}
  ]
})json";

// Topology evidence for MLA_0, as tests/model_archive_fixture_utils.h writes it: one 8-byte
// data.ifm.b0 and one 8-byte data.ofm.b0 tensor section.
std::string topology_elf() {
  const std::string names("\0.shstrtab\0data.ifm.b0\0data.ofm.b0\0", 35);
  constexpr std::size_t kSections = 104;
  constexpr std::size_t kData = kSections + 4 * 64;
  std::string elf(kData + 32, '\0');
  const auto put = [&elf](const std::size_t offset, const std::uint64_t value, const int bytes) {
    for (int i = 0; i < bytes; ++i) {
      elf[offset + i] = static_cast<char>((value >> (8 * i)) & 0xffU);
    }
  };
  elf.replace(0, 7,
              "\x7f"
              "ELF\x02\x01\x01");
  put(16, 1, 2);
  put(18, 183, 2);
  put(20, 1, 4);
  put(40, kSections, 8);
  put(52, 64, 2);
  put(58, 64, 2);
  put(60, 4, 2);
  put(62, 1, 2);
  elf.replace(64, names.size(), names);
  put(kSections + 64, 1, 4);
  put(kSections + 64 + 4, 3, 4);
  put(kSections + 64 + 24, 64, 8);
  put(kSections + 64 + 32, names.size(), 8);
  put(kSections + 64 + 48, 1, 8);
  for (std::size_t i = 0; i < 2; ++i) {
    const std::size_t header = kSections + (2 + i) * 64;
    put(header, i == 0 ? 11 : 23, 4);
    put(header + 4, 0x71ba0002U, 4);
    put(header + 24, kData + 16 * i, 8);
    put(header + 32, 16, 8);
    put(kData + 16 * i, 8, 8);
    put(kData + 16 * i + 8, 1, 8);
  }
  return elf;
}

// Points the process temporary directory at another directory for one scope.
class TemporaryDirectoryOverride {
public:
  explicit TemporaryDirectoryOverride(const fs::path& directory) {
#ifdef _WIN32
    const DWORD size = GetEnvironmentVariableW(L"TMP", nullptr, 0);
    if (size != 0) {
      std::wstring value(size, L'\0');
      value.resize(GetEnvironmentVariableW(L"TMP", value.data(), size));
      previous_ = value;
    }
    require(SetEnvironmentVariableW(L"TMP", directory.c_str()) != 0, "failed to set TMP");
#else
    if (const char* value = std::getenv("TMPDIR")) {
      previous_ = value;
    }
    require(::setenv("TMPDIR", directory.c_str(), 1) == 0, "failed to set TMPDIR");
#endif
  }
  ~TemporaryDirectoryOverride() {
#ifdef _WIN32
    SetEnvironmentVariableW(L"TMP", previous_ ? previous_->c_str() : nullptr);
#else
    if (previous_) {
      ::setenv("TMPDIR", previous_->c_str(), 1);
    } else {
      ::unsetenv("TMPDIR");
    }
#endif
  }
  TemporaryDirectoryOverride(const TemporaryDirectoryOverride&) = delete;
  TemporaryDirectoryOverride& operator=(const TemporaryDirectoryOverride&) = delete;

private:
  std::optional<fs::path::string_type> previous_;
};

// read_model_facts() extracts under the temporary directory, then decodes the MPK and reads its
// MLA executable there. A Windows profile outside the ANSI code page puts TEMP under such a name.
void test_model_facts_read_under_unicode_temp_directory() {
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
  const fs::path directory =
      temp.path() / pcie_internal::host_path_from_string("\xE6\x9D\xB1\xE4\xBA\xAC \xCE\xB4");
  fs::create_directories(directory);
  const fs::path archive = directory / "model.tar.gz";
  write_file(archive,
             gzip(tar_member("etc/model_mpk.json", kStrictManifest) +
                  tar_member("share/synthetic_mla.elf", topology_elf()) + std::string(1024, '\0')));

  pcie_internal::PcieModelFacts facts;
  {
    const TemporaryDirectoryOverride override_temp(directory);
    require(fs::equivalent(fs::temp_directory_path(), directory),
            "temporary directory override was ignored");
    facts = pcie_internal::read_model_facts(pcie_internal::host_path_string(archive));
  }
  require(facts.inputs.size() == 1 && facts.outputs.size() == 1,
          "model facts under a Unicode temporary directory lost the model's tensors");
  require(facts.inputs.front().name == "input" && facts.outputs.front().name == "output",
          "model facts under a Unicode temporary directory named the wrong tensors");
}

// A killed or crashed application leaves its snapshot behind; the next one removes it, and only
// snapshots whose recorded process on this host has ended.
void test_snapshots_of_ended_processes_are_removed() {
  pcie_internal::ScopedTempDirectory temp("sima-neat-pcie-snapshot-test");
  const pcie_internal::HostProcessOwner self = pcie_internal::current_host_process_owner();
  require(!self.host_id.empty() && !self.start.empty(), "host process owner is unknown");
  const auto leave = [&temp](const std::string& name, const std::optional<std::string>& host_id,
                             const std::uint64_t pid, const std::string& start) {
    fs::create_directories(temp.path() / name);
    write_file(temp.path() / name / "model.tar.gz", "archive");
    if (host_id.has_value()) {
      write_file(temp.path() / name / "owner", "host_id=" + *host_id +
                                                   "\nhost_pid=" + std::to_string(pid) +
                                                   "\nhost_start=" + start + "\n");
    }
    return temp.path() / name;
  };
  // This process with another start time stands for an ended process whose PID was reused.
  const fs::path reused = leave("sima-neat-pcie-model-reused", self.host_id, self.pid, "1");
  const fs::path running =
      leave("sima-neat-pcie-model-running", self.host_id, self.pid, self.start);
  const fs::path other_host = leave("sima-neat-pcie-model-other", "linux:another-host", 1, "1");
  const fs::path unrecorded = leave("sima-neat-pcie-model-unrecorded", std::nullopt, 0, "");
  const fs::path other_prefix = leave("sima-neat-pcie-options-reused", self.host_id, self.pid, "1");
#ifndef _WIN32
  const pid_t child = fork();
  if (child == 0) {
    _exit(0);
  }
  waitpid(child, nullptr, 0);
  const fs::path ended = leave("sima-neat-pcie-model-ended", self.host_id,
                               static_cast<std::uint64_t>(child), self.start);
#endif
  write_file(temp.path() / "model.tar.gz", "archive");

  const TemporaryDirectoryOverride override_temp(temp.path());
  pcie_internal::ModelArchiveSnapshot snapshot(
      pcie_internal::host_path_string(temp.path() / "model.tar.gz"));
  require(!fs::exists(reused), "a snapshot of a reused PID was kept");
#ifndef _WIN32
  require(!fs::exists(ended), "a snapshot of an ended process was kept");
#endif
  require(fs::exists(running) && fs::exists(other_host) && fs::exists(unrecorded) &&
              fs::exists(other_prefix),
          "a snapshot of a running, foreign or unrecorded owner was removed");
  const std::string owner =
      read_file(pcie_internal::host_path_from_string(snapshot.path()).parent_path() / "owner");
  require(owner == "host_id=" + self.host_id + "\nhost_pid=" + std::to_string(self.pid) +
                       "\nhost_start=" + self.start + "\n",
          "the snapshot did not record its owner: " + owner);
}

#ifndef _WIN32
// The state field of /proc/<pid>/stat.
char process_state(const pid_t pid) {
  const std::string stat = read_file("/proc/" + std::to_string(pid) + "/stat");
  const auto name_end = stat.rfind(')');
  return name_end != std::string::npos && name_end + 2 < stat.size() ? stat[name_end + 2] : '?';
}

bool eventually(const std::function<bool()>& condition) {
  for (int attempt = 0; attempt < 500; ++attempt) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}
#endif

// A process whose main thread called pthread_exit while another thread runs shows its leader as
// a zombie in /proc/<pid>/stat, yet it is running. Once its last thread exits it is a zombie.
void test_process_with_exited_main_thread_is_running() {
#ifndef _WIN32
  int release[2];
  require(pipe(release) == 0, "pipe failed");
  const pid_t child = fork();
  if (child == 0) {
    close(release[1]);
    pthread_t worker;
    const auto wait_for_release = [](void* fd) -> void* {
      char byte;
      [[maybe_unused]] const ssize_t got =
          read(static_cast<int>(reinterpret_cast<std::intptr_t>(fd)), &byte, 1);
      return nullptr;
    };
    if (pthread_create(&worker, nullptr, wait_for_release,
                       reinterpret_cast<void*>(static_cast<std::intptr_t>(release[0]))) != 0) {
      _exit(1);
    }
    pthread_exit(nullptr);
  }
  close(release[0]);
  pcie_internal::HostProcessOwner owner = pcie_internal::current_host_process_owner();
  owner.pid = static_cast<std::uint64_t>(child);
  owner.start.clear(); // only the process state decides
  const bool leader_exited = eventually([child] { return process_state(child) == 'Z'; });
  const bool ended_while_running = pcie_internal::host_process_ended(owner);
  close(release[1]);
  const bool ended_after_exit =
      eventually([&owner] { return pcie_internal::host_process_ended(owner); });
  waitpid(child, nullptr, 0);
  require(leader_exited, "the main thread of the test process did not exit");
  require(!ended_while_running, "a running process whose main thread exited was taken for ended");
  require(ended_after_exit, "a zombie process was not taken for ended");
#endif
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
    test_model_facts_read_under_unicode_temp_directory();
    test_snapshots_of_ended_processes_are_removed();
    test_process_with_exited_main_thread_is_running();
    std::cout << "[PASS] model archive snapshot\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

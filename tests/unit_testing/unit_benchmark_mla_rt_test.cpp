#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/internal/MlaRt.h"
#include "test_main.h"

#include <glib.h>

#include <signal.h>

#include <fstream>
#include <functional>
#include <thread>

namespace {
using namespace simaai::neat::internal;
using namespace std::chrono_literals;

struct Fixture {
  std::filesystem::path root;
  Fixture() {
    gchar* directory = g_dir_make_tmp("neat-mla-rt-test-XXXXXX", nullptr);
    require(directory != nullptr, "create fixture");
    root = directory;
    g_free(directory);
  }
  ~Fixture() {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
  std::filesystem::path runtime(const std::string& body) {
    const auto file = root / "fake mla-rt";
    std::ofstream(file) << "#!/bin/sh\nset -eu\n" + body;
    std::filesystem::permissions(file, std::filesystem::perms::owner_all);
    return file;
  }
};
void fails(const std::function<void()>& operation, const std::string& expected) {
  try {
    operation();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(expected) != std::string::npos, error.what());
    return;
  }
  require(false, "expected failure: " + expected);
}

void output_is_combined_and_bounded() {
  Fixture fixture;
  const auto command = fixture.runtime("echo \"out $*\"\necho err >&2\n");
  require(run_mla_rt(command, {"--connect", "two words"}) == "out --connect two words\nerr\n",
          "arguments must pass through and stderr must join stdout");
  const auto noisy = fixture.runtime("head -c 10000 /dev/zero | tr '\\0' x\necho tail\n");
  const auto output = run_mla_rt(noisy, {});
  require(output.size() == 4096 && output.ends_with("xtail\n"), "keep only the last 4 KiB");
}

void failures_name_the_cause() {
  Fixture fixture;
  fails([&] { run_mla_rt(fixture.root / "missing", {}); }, "cannot start mla-rt");
  fails([&] { run_mla_rt(fixture.runtime("echo 'device busy' >&2\nexit 7\n"), {}); },
        "mla-rt failed (exit 7)\ndevice busy");
}

void timeout_kills_the_process_group() {
  Fixture fixture;
  const auto child_pid = fixture.root / "child.pid";
  // The shell's child stands in for anything mla-rt starts; it must die with mla-rt.
  const auto command =
      fixture.runtime("sleep 60 &\necho $! > '" + child_pid.string() + "'\necho started\nwait\n");
  const auto start = std::chrono::steady_clock::now();
  fails([&] { run_mla_rt(command, {}, 300ms); }, "mla-rt timed out after 300 ms\nstarted");
  require(std::chrono::steady_clock::now() - start < 5s, "the timeout must return promptly");
  pid_t pid = 0;
  std::ifstream(child_pid) >> pid;
  require(pid > 0, "the fake mla-rt must record its child");
  // Once orphaned, the killed child is reaped by init.
  bool gone = false;
  for (int i = 0; i < 100 && !gone; ++i) {
    gone = ::kill(pid, 0) != 0;
    if (!gone)
      std::this_thread::sleep_for(20ms);
  }
  require(gone, "the timeout must kill mla-rt's process group");
}
} // namespace

RUN_TEST("unit_benchmark_mla_rt_test", ([] {
           output_is_combined_and_bounded();
           failures_name_the_cause();
           timeout_kills_the_process_group();
         }));

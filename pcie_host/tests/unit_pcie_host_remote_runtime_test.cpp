// Runs the real card start script locally: a fake ssh executes it with /bin/sh, and this test
// executable, linked as pcie-pipeline-builder, stands in for the card worker.
#include "RemoteRuntime.h"
#include "SshRunner.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace pcie = simaai::neat::pcie;
namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

const std::string kHostVersion = SIMA_NEAT_PCIE_HOST_VERSION;
const std::string kHostMinor =
    kHostVersion.substr(0, kHostVersion.find('.', kHostVersion.find('.') + 1));
volatile std::sig_atomic_t g_stop = 0;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string read_file(const fs::path& path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const fs::path& path, const std::string& contents) {
  std::ofstream(path, std::ios::trunc) << contents;
}

// A worker that has exited, including one not yet reaped by its new parent.
bool running(const int pid) {
  std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
  const std::string stat((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto name_end = stat.rfind(')');
  return name_end != std::string::npos && name_end + 2 < stat.size() && stat[name_end + 2] != 'Z';
}

int ended_pid() {
  const pid_t child = fork();
  if (child == 0) {
    _exit(0);
  }
  waitpid(child, nullptr, 0);
  return child;
}

// The card worker: --version prints the fixture's version lines, as a current builder does, or
// fails as a builder that predates --version does. Otherwise it claims its queue.
int fake_builder(const int argc, char** argv) {
  const fs::path run_dir = std::getenv("SIMA_NEAT_PCIE_RUN_DIR");
  if (argc == 2 && std::string(argv[1]) == "--version") {
    if (!fs::exists(run_dir.parent_path() / "card-version")) {
      std::cerr << "unknown option '--version'\n";
      return 2;
    }
    std::cout << read_file(run_dir.parent_path() / "card-version");
    return 0;
  }
  std::string queue;
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) == "--queue") {
      queue = argv[index + 1];
    }
  }
  const fs::path pid_file = run_dir / ("q" + queue + ".pid");
  std::signal(SIGTERM, [](int) { g_stop = 1; });
  write_file(pid_file, std::to_string(getpid()) + "\n");
  for (int tick = 0; tick < 1200 && !g_stop; ++tick) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  fs::remove(pid_file);
  return 0;
}

class Card {
public:
  Card() {
    char name[] = "/tmp/neat-pcie-remote-runtime-XXXXXX";
    require(mkdtemp(name) != nullptr, "mkdtemp failed");
    root_ = name;
    fs::create_directories(root_ / "bin");
    fs::create_directories(root_ / "run");
    fs::create_directories(root_ / "log");
    write_file(root_ / "bin" / "ssh",
               "#!/bin/sh\nfor last; do :; done\nexec /bin/sh -c \"$last\"\n");
    fs::permissions(root_ / "bin" / "ssh", fs::perms::owner_all);
    fs::create_symlink(fs::read_symlink("/proc/self/exe"), root_ / "bin" / "pcie-pipeline-builder");
    if (const char* path = std::getenv("PATH")) {
      previous_path_ = path;
    }
    setenv("PATH", (root_ / "bin").string().append(":").append(previous_path_).c_str(), 1);
  }

  ~Card() {
    for (const int pid : workers_) {
      kill(pid, SIGKILL);
    }
    for (const auto& upload : uploads_) {
      std::error_code ec;
      fs::remove(upload, ec);
    }
    setenv("PATH", previous_path_.c_str(), 1);
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  pcie::ConnectionOptions connection() const {
    pcie::ConnectionOptions options;
    options.card_host = "card.invalid";
    options.card_env =
        "SIMA_NEAT_PCIE_HELPER_PATH=" + (root_ / "bin/pcie-pipeline-builder").string() +
        " SIMA_NEAT_PCIE_RUN_DIR=" + (root_ / "run").string() +
        " SIMA_NEAT_PCIE_LOG_DIR=" + (root_ / "log").string();
    return options;
  }

  std::string upload() {
    uploads_.push_back(pcie_internal::RemoteRuntime::unique_remote_upload_path("model.tar.gz"));
    write_file(uploads_.back(), "model");
    return uploads_.back();
  }

  int start(const pcie_internal::RemoteRuntime& runtime, const int queue,
            const std::string& upload) {
    const int pid = runtime.start(queue, upload, std::nullopt);
    workers_.push_back(pid);
    return pid;
  }

  // The start failure message, or "" when the start succeeded.
  std::string start_failure(const pcie_internal::RemoteRuntime& runtime, const int queue,
                            const std::string& upload) {
    try {
      start(runtime, queue, upload);
    } catch (const pcie_internal::RemoteStartError& error) {
      require(error.cleanup_safe(), "a start refused before launch must allow upload cleanup");
      return error.what();
    }
    return "";
  }

  fs::path owner(const int queue) const {
    return root_ / "run" / ("q" + std::to_string(queue) + ".owner");
  }

  // Rewrites one key of the queue's owner record, as another process or host would have it.
  void set_owner(const int queue, const std::string& key, const std::string& value) const {
    std::istringstream lines(read_file(owner(queue)));
    std::string updated;
    for (std::string line; std::getline(lines, line);) {
      updated += line.rfind(key + "=", 0) == 0 ? key + "=" + value : line;
      updated += "\n";
    }
    write_file(owner(queue), updated);
  }

  void set_card_version(const std::optional<std::string>& version) const {
    std::error_code ec;
    fs::remove(root_ / "card-version", ec);
    if (version.has_value()) {
      write_file(root_ / "card-version", *version);
    }
  }

  fs::path root() const {
    return root_;
  }

private:
  fs::path root_;
  std::string previous_path_;
  std::vector<int> workers_;
  std::vector<std::string> uploads_;
};

void test_start_records_the_host_owner(Card& card, const pcie_internal::RemoteRuntime& runtime) {
  const std::string upload = card.upload();
  const int worker = card.start(runtime, 1, upload);
  const std::string owner = read_file(card.owner(1));
  for (const std::string line : {"builder_pid=" + std::to_string(worker),
                                 "host_pid=" + std::to_string(getpid()), "upload=" + upload}) {
    require(owner.find(line + "\n") != std::string::npos,
            "queue owner record lacks " + line + ": " + owner);
  }
  require(owner.find("host_id=linux:") != std::string::npos &&
              owner.find("host_start=") != std::string::npos,
          "queue owner record lacks the host identity: " + owner);
  runtime.stop(1, worker);
}

void test_ended_owner_on_this_host_is_reclaimed(Card& card,
                                                const pcie_internal::RemoteRuntime& runtime) {
  // An ended host process, and a reused PID: this process with another start time.
  for (const auto& [key, value] : std::vector<std::pair<std::string, std::string>>{
           {"host_pid", std::to_string(ended_pid())}, {"host_start", "1"}}) {
    const std::string left_upload = card.upload();
    const int left_worker = card.start(runtime, 2, left_upload);
    card.set_owner(2, key, value);

    const std::string upload = card.upload();
    const int worker = card.start(runtime, 2, upload);
    require(worker != left_worker && running(worker), "the reclaimed queue did not start");
    require(!running(left_worker), "the leftover worker of an ended process was not stopped");
    require(!fs::exists(left_upload), "the leftover worker's upload was not removed");
    require(read_file(card.owner(2)).find("upload=" + upload + "\n") != std::string::npos,
            "the new worker did not replace the owner record");
    runtime.stop(2, worker);
  }
}

void test_running_owner_on_this_host_is_kept(Card& card,
                                             const pcie_internal::RemoteRuntime& runtime) {
  const int worker = card.start(runtime, 3, card.upload());
  const std::string error = card.start_failure(runtime, 3, card.upload());
  require(error.find("queue_busy") != std::string::npos &&
              error.find("PCIe queue 3 is in use by process " + std::to_string(getpid()) +
                         " on this host") != std::string::npos &&
              error.find("(card worker PID " + std::to_string(worker) +
                         "), which is still running") != std::string::npos,
          "a running owner was not named: " + error);
  // The owner may only look running (its PID cannot be queried), so the stop command is given.
  require(error.find("If that application has ended, stop the leftover worker and retry: ssh ") !=
                  std::string::npos &&
              error.find("pkill -f '[p]cie-pipeline-builder.*--queue 3( |$)'") != std::string::npos,
          "a running owner gave no worker stop command: " + error);
  require(running(worker), "the worker of a running process was stopped");
  runtime.stop(3, worker);
}

void test_owner_on_another_host_is_kept(Card& card, const pcie_internal::RemoteRuntime& runtime) {
  const int worker = card.start(runtime, 4, card.upload());
  card.set_owner(4, "host_id", "linux:another-host");
  card.set_owner(4, "host", "build-02");
  card.set_owner(4, "host_pid", std::to_string(ended_pid()));
  const std::string error = card.start_failure(runtime, 4, card.upload());
  require(error.find("PCIe queue 4 is in use by process ") != std::string::npos &&
              error.find(" on host build-02 (card worker PID " + std::to_string(worker) + ")") !=
                  std::string::npos &&
              error.find("pkill -f") != std::string::npos,
          "another host's owner was not named: " + error);
  require(running(worker), "the worker of another host was stopped");
  runtime.stop(4, worker);
}

// A worker started by an older host library has no owner record: the start fails as before.
void test_worker_without_owner_record_is_kept(Card& card,
                                              const pcie_internal::RemoteRuntime& runtime) {
  const int worker = card.start(runtime, 5, card.upload());
  fs::remove(card.owner(5));
  const std::string error = card.start_failure(runtime, 5, card.upload());
  require(error.find("Another card worker holds PCIe queue 5. If no application uses this queue") !=
              std::string::npos,
          "a worker without owner record changed the busy-queue message: " + error);
  require(running(worker), "a worker without owner record was stopped");
  runtime.stop(5, worker);
}

// What a current builder prints for --version: the Git-derived version, then the package version.
std::string builder_version(const std::string& version, const std::string& package_version) {
  return "sima-neat-version=" + version + "\nsima-neat-package-version=" + package_version + "\n";
}

void test_card_neat_minor_release_must_match(Card& card,
                                             const pcie_internal::RemoteRuntime& runtime) {
  // Another minor release, including one that only shares the host minor's leading digits.
  for (const std::string version : {std::string("0.0.1"), kHostMinor + "0.0"}) {
    card.set_card_version(builder_version(version, version));
    const std::string error = card.start_failure(runtime, 0, card.upload());
    require(error.find("(exit=21") != std::string::npos &&
                error.find("runs Neat " + version + " and this host runs Neat " + kHostVersion +
                           "; the card needs the same Neat minor release (" + kHostMinor +
                           ") as the host.") != std::string::npos &&
                error.find("Install the matching Neat release on the card (see Install Neat on "
                           "the card).") != std::string::npos,
            "card Neat " + version + " was not refused with both versions: " + error);
    require(!fs::exists(card.root() / "run" / "q0.pid"), "a refused start launched a worker");
  }

  // Accepted: another patch release; a tag-built card whose Git version is the tag name; a
  // builder that prints only the Git version (no release has one); a builder without --version.
  for (const std::optional<std::string> version :
       {std::optional<std::string>(builder_version(kHostMinor + ".99", kHostMinor + ".99")),
        std::optional<std::string>(builder_version("3.0.2121", kHostVersion)),
        std::optional<std::string>("sima-neat-version=0.0.1\n"), std::optional<std::string>()}) {
    card.set_card_version(version);
    const int worker = card.start(runtime, 0, card.upload());
    runtime.stop(0, worker);
  }
}

} // namespace

int main(const int argc, char** argv) {
  if (fs::path(argv[0]).filename() == "pcie-pipeline-builder") {
    return fake_builder(argc, argv);
  }
  try {
    // Adopt the detached workers and never reap them, as the `tail -f /dev/null` PID 1 of a CI
    // job container does: a worker that ended stays a zombie, which every stop must accept.
    require(prctl(PR_SET_CHILD_SUBREAPER, 1) == 0, "PR_SET_CHILD_SUBREAPER failed");
    Card card;
    const pcie_internal::RemoteRuntime runtime(card.connection());
    test_start_records_the_host_owner(card, runtime);
    test_ended_owner_on_this_host_is_reclaimed(card, runtime);
    test_running_owner_on_this_host_is_kept(card, runtime);
    test_owner_on_another_host_is_kept(card, runtime);
    test_worker_without_owner_record_is_kept(card, runtime);
    test_card_neat_minor_release_must_match(card, runtime);
    std::cout << "[PASS] PCIe remote runtime queue ownership\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

#include "model/internal/ModelPack.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
namespace fs = std::filesystem;
using simaai::neat::internal::ModelPackProcessRoot;

void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

ModelPackProcessRoot make_root(const fs::path& base, bool collect_stale) {
  require(ModelPackProcessRoot::prepare_base(base, collect_stale),
          "extraction base must be writable after stale collection");
  return ModelPackProcessRoot(base);
}

void test_roots(const fs::path& base) {
  fs::path ordinary;
  fs::path preserved;
  {
    auto first = make_root(base, true);
    auto second = make_root(base, true);
    ordinary = first.path();
    preserved = second.path();
    require(ordinary != preserved, "even owners with the same PID need distinct roots");
    require(fs::exists(ordinary), "collecting stale roots must leave a live owner alone");
    second.preserve();
  }
  require(!fs::exists(ordinary), "normal teardown must remove an unpreserved root");
  require(fs::exists(preserved), "preserved roots must survive teardown");

  // Unlocked legacy roots cannot be distinguished from live owners in another PID namespace.
  const fs::path legacy = base / "proc_1";
  fs::create_directory(legacy);
  std::ofstream(legacy / "payload") << "legacy\n";
  const fs::path initializing = base / "proc_run_initializing";
  fs::create_directory(initializing);
  std::ofstream(initializing / ".sima_modelpack_owner.pending") << "not published\n";

  // Children create their roots concurrently and retain them until the parent closes release.
  int ready[2];
  int release[2];
  require(::pipe(ready) == 0 && ::pipe(release) == 0, "create coordination pipes");
  std::vector<pid_t> children;
  for (int i = 0; i < 4; ++i) {
    const pid_t child = ::fork();
    require(child >= 0, "fork extraction owner");
    if (child == 0) {
      ::close(ready[0]);
      ::close(release[1]);
      try {
        auto root = make_root(base, true);
        std::ofstream(root.path() / "payload") << "owned\n";
        std::ofstream(base / ("child_" + std::to_string(i))) << root.path().string();
        if (::write(ready[1], "1", 1) != 1)
          ::_exit(1);
        char signal;
        if (::read(release[0], &signal, 1) != 0)
          ::_exit(1);
        // Deliberately bypass destructors, as after SIGKILL or a container crash.
        ::_exit(0);
      } catch (...) {
        ::_exit(1);
      }
    }
    children.push_back(child);
  }
  ::close(ready[1]);
  ::close(release[0]);
  for (std::size_t i = 0; i < children.size(); ++i) {
    char signal;
    require(::read(ready[0], &signal, 1) == 1, "child owner must become ready");
  }
  ::close(ready[0]);

  std::vector<fs::path> child_roots;
  {
    auto collector = make_root(base, true);
    for (std::size_t i = 0; i < children.size(); ++i) {
      std::string path;
      std::ifstream(base / ("child_" + std::to_string(i))) >> path;
      require(!path.empty(), "child must publish its root");
      child_roots.emplace_back(path);
      std::string payload;
      std::ifstream(child_roots.back() / "payload") >> payload;
      require(payload == "owned", "stale cleanup must not erase a live child's files");
      for (std::size_t j = 0; j < i; ++j)
        require(child_roots[j] != child_roots[i], "concurrent owners need distinct roots");
    }
  }
  ::close(release[1]);
  for (const pid_t child : children) {
    int status;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "child owner must exit successfully");
  }

  {
    auto no_gc = make_root(base, false);
    for (const auto& path : child_roots)
      require(fs::exists(path), "disabled stale collection must retain abandoned roots");
  }
  // A failed candidate probe must not prevent recovery of abandoned storage.
  const fs::path blocked_probe = base / ".sima_modelpack_write_probe";
  fs::create_directory(blocked_probe);
  require(!ModelPackProcessRoot::prepare_base(base, false), "blocked probe must reject the base");
  for (const auto& path : child_roots)
    require(fs::exists(path), "disabled collection must retain abandoned roots before probing");
  require(!ModelPackProcessRoot::prepare_base(base, true),
          "blocked probe must still reject the base");
  for (const auto& path : child_roots)
    require(!fs::exists(path), "stale collection must precede the candidate write probe");
  fs::remove(blocked_probe);

  {
    auto collector = make_root(base, true);
    require(fs::exists(preserved), "stale collection must honor the keep marker");
    require(fs::exists(legacy / "payload"), "legacy roots without locks must remain untouched");
    require(fs::exists(initializing), "an unpublished owner must not be mistaken for a stale root");
  }
}
} // namespace

int main(int argc, char** argv) {
  // Also used by the external PID-namespace probe; no model or accelerator is opened.
  if (argc == 3 && std::string(argv[1]) == "--hold") {
    auto root = make_root(argv[2], true);
    std::cout << root.path().string() << std::endl;
    std::cin.get();
    return 0;
  }

  std::string pattern =
      ((argc == 2 ? fs::path(argv[1]) : fs::temp_directory_path()) / "modelpack-roots-XXXXXX")
          .string();
  if (!::mkdtemp(pattern.data())) {
    std::cerr << "[FAIL] create test directory\n";
    return 1;
  }
  int status = 0;
  try {
    test_roots(pattern);
    std::cout << "[OK] unit_modelpack_process_root_test passed\n";
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    status = 1;
  }
  fs::remove_all(pattern);
  return status;
}

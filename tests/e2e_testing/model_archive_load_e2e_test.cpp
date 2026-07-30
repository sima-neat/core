// Model loading through the public Model boundary, covering the properties that the extraction
// layer must keep: archive identity decides the package directory, retained packages are published
// where a later process can find them, and concurrent loads of one archive converge on one copy.
#include "model/Model.h"

#include "asset_utils.h"
#include "model_archive_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

// The MPK contract JSON sits in <package_root>/etc, so the package root is two levels up.
std::string package_root_of(const simaai::neat::Model& model) {
  const std::string mpk_json = model.info().mpk_json_path;
  require(!mpk_json.empty(), "Model::info() should report the extracted MPK contract path");
  return fs::path(mpk_json).parent_path().parent_path().string();
}

std::vector<std::string> package_file_set(const fs::path& package_root) {
  std::vector<std::string> files;
  for (const auto& it : fs::recursive_directory_iterator(package_root)) {
    if (it.is_regular_file())
      files.push_back(fs::relative(it.path(), package_root).generic_string());
  }
  std::sort(files.begin(), files.end());
  return files;
}

// Same basename, different directory: the case where a basename-keyed package directory would make
// the second load evict the first.
fs::path copy_archive_as(const std::string& source, const fs::path& dir,
                         const std::string& basename) {
  fs::create_directories(dir);
  const fs::path target = dir / basename;
  fs::copy_file(source, target, fs::copy_options::overwrite_existing);
  return target;
}

} // namespace

RUN_TEST(
    "model_archive_load_e2e_test", ([] {
      const std::string archive = sima_test::resolve_resnet50_tar();
      if (archive.empty()) {
        // Registered in the long lane, where an unreachable modelzoo is an environment gap rather
        // than a regression in the extraction path this test covers.
        skip_test_exception(
            "ResNet50 model pack not found; set SIMA_MODEL_TAR or SIMA_RESNET50_TAR");
      }

      const fs::path scratch = fs::path(sima_test::make_temp_dir("model_archive_load_e2e"));

      // Two archives that differ only in directory. Distinct package roots is the assertion: a
      // basename-keyed directory gives both the same path, and the second extraction deletes the
      // first model's files while that model is still alive.
      const fs::path first_copy = copy_archive_as(archive, scratch / "a", "model_pack.tar.gz");
      const fs::path second_copy = copy_archive_as(archive, scratch / "b", "model_pack.tar.gz");

      simaai::neat::Model first(first_copy.string());
      simaai::neat::Model second(second_copy.string());
      const std::string first_root = package_root_of(first);
      const std::string second_root = package_root_of(second);
      require(first_root != second_root,
              "archives sharing a basename must not share a package directory");

      // <base>/proc_<pid>/pkg_<identity>/<package>, so two levels up is the per-process root.
      // The retained load below latches process cleanup off and drops a keep marker, which exempts
      // this root from both atexit cleanup and stale-root GC. Nothing else will remove it.
      const fs::path process_root = fs::path(first_root).parent_path().parent_path();

      const std::vector<std::string> first_files = package_file_set(first_root);
      require(!first_files.empty(), "first package should contain files after the second load");
      require(first_files == package_file_set(second_root),
              "both same-basename packages should hold the same extracted file set");

      // Retention publishes outside the per-process root so a later process can adopt it. A
      // per-process directory would be named proc_<pid> and would be garbage collected on exit.
      simaai::neat::Model::Options retained_opt;
      retained_opt.cleanup_extracted_model_data = false;
      const fs::path retained_copy = copy_archive_as(archive, scratch / "retained", "kept.tar.gz");
      std::string retained_root;
      {
        simaai::neat::Model retained(retained_copy.string(), retained_opt);
        retained_root = package_root_of(retained);
      }
      const std::string published_parent =
          fs::path(retained_root).parent_path().filename().string();
      require(published_parent.rfind("pkg_", 0) == 0,
              "a retained package should be published under pkg_<identity>, got " +
                  published_parent);

      // Reloading resolves to the same published package rather than extracting a second copy.
      {
        simaai::neat::Model reloaded(retained_copy.string(), retained_opt);
        require(package_root_of(reloaded) == retained_root,
                "a retained package should be reused rather than extracted again");
      }

      // Concurrent loads of one archive under default options: the extraction mutex serializes
      // them, so the first extracts and the rest resolve through the in-process cache. The
      // cross-process publish race is not reachable from a single process.
      const fs::path shared_copy = copy_archive_as(archive, scratch / "shared", "shared.tar.gz");
      constexpr int kThreads = 4;
      std::vector<std::string> roots(kThreads);
      std::vector<std::string> failures(kThreads);
      std::vector<std::thread> workers;
      for (int i = 0; i < kThreads; ++i) {
        workers.emplace_back([&, i]() {
          try {
            simaai::neat::Model model(shared_copy.string());
            roots[i] = package_root_of(model);
          } catch (const std::exception& e) {
            failures[i] = e.what();
          }
        });
      }
      for (auto& worker : workers) {
        worker.join();
      }
      for (int i = 0; i < kThreads; ++i) {
        require(failures[i].empty(),
                "concurrent load " + std::to_string(i) + " threw: " + failures[i]);
        require(roots[i] == roots[0],
                "concurrent loads of one archive should converge on a single package");
      }
      require(!package_file_set(roots[0]).empty(), "the converged package should hold files");

      // Neither the published package nor the per-process root is garbage collected once
      // retention has been requested, so this test removes both.
      std::error_code cleanup_ec;
      fs::remove_all(fs::path(retained_root).parent_path(), cleanup_ec);
      fs::remove_all(process_root, cleanup_ec);
      fs::remove_all(scratch, cleanup_ec);
    }));

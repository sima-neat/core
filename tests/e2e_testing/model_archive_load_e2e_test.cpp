// Model loading through the public Model boundary, covering the properties that the extraction
// layer must keep: archive identity decides the package directory, retained packages are published
// where a later process can find them, and concurrent loads of one archive converge on one copy.
#include "model/Model.h"

#include "asset_utils.h"
#include "model_archive_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
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

      // An unpacked directory is a first-class load source. Untar the archive flat, load both,
      // and require the same effective package: same layout, same contents, same Model contract.
      const fs::path unpacked = scratch / "unpacked";
      fs::create_directories(unpacked);
      const std::string untar = "tar -xzf '" + archive + "' -C '" + unpacked.string() + "'";
      require(std::system(untar.c_str()) == 0, "failed to untar the model pack");
      const std::vector<std::string> source_before = package_file_set(unpacked);

      simaai::neat::Model from_dir(unpacked.string());
      const std::string dir_root = package_root_of(from_dir);
      require(package_file_set(dir_root) == package_file_set(first_root),
              "a directory load should produce the same package file set as the archive");
      for (const char* sub : {"etc", "lib", "share"}) {
        require(fs::is_directory(fs::path(dir_root) / sub),
                std::string("directory load should classify into ") + sub);
      }
      require(fs::path(from_dir.info().mpk_json_path).is_absolute(),
              "a directory load should rewrite the MPK contract path to an absolute path");
      require(fs::path(from_dir.info().mpk_json_path).string().rfind(dir_root, 0) == 0,
              "the rewritten contract path should point inside the produced package");
      require(from_dir.info().model_name == first.info().model_name,
              "archive and directory loads should agree on the Model contract");

      // The caller's directory is an input. Nothing may be written into it, and the package must
      // not simply alias it.
      require(package_file_set(unpacked) == source_before,
              "loading from a directory must not modify the source directory");
      require(dir_root != unpacked.string(),
              "an unprovenanced directory must be copied, not adopted in place");

      // A Core-produced package that has not been marked ready is still just a directory: it is
      // copied and re-validated rather than trusted.
      const fs::path unmarked = scratch / "unmarked";
      fs::copy(dir_root, unmarked, fs::copy_options::recursive);
      fs::remove(unmarked / ".sima_modelpack_ready");
      simaai::neat::Model from_unmarked(unmarked.string());
      require(package_root_of(from_unmarked) != unmarked.string(),
              "a package without the ready marker must be copied, not adopted");

      // A marked package this process produced is the zero-copy path: it resolves to itself.
      simaai::neat::Model from_published(dir_root);
      require(package_root_of(from_published) == dir_root,
              "a Core-published package should load directly from its own directory");

      // Neither the published package nor the per-process root is garbage collected once
      // retention has been requested, so this test removes both.
      std::error_code cleanup_ec;
      fs::remove_all(fs::path(retained_root).parent_path(), cleanup_ec);
      fs::remove_all(process_root, cleanup_ec);
      fs::remove_all(scratch, cleanup_ec);
    }));

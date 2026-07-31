// Model loading through the public Model boundary. Archive loads stay private to this process,
// archive identity decides the package directory, and an already-organized package remains the
// zero-copy fast path.
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

RUN_TEST("model_archive_load_e2e_test", ([] {
           const std::string archive = sima_test::resolve_resnet50_tar();
           if (archive.empty()) {
             // Registered in the long lane, where an unreachable modelzoo is an environment gap
             // rather than a regression in the extraction path this test covers.
             skip_test_exception(
                 "ResNet50 model pack not found; set SIMA_MODEL_TAR or SIMA_RESNET50_TAR");
           }

           const fs::path scratch = fs::path(sima_test::make_temp_dir("model_archive_load_e2e"));

           // Two archives that differ only in directory. Distinct package roots is the assertion: a
           // basename-keyed directory gives both the same path, and the second extraction deletes
           // the first model's files while that model is still alive.
           const fs::path first_copy = copy_archive_as(archive, scratch / "a", "model_pack.tar.gz");
           const fs::path second_copy =
               copy_archive_as(archive, scratch / "b", "model_pack.tar.gz");

           simaai::neat::Model first(first_copy.string());
           simaai::neat::Model second(second_copy.string());
           const std::string first_root = package_root_of(first);
           const std::string second_root = package_root_of(second);
           require(first_root != second_root,
                   "archives sharing a basename must not share a package directory");

           const std::vector<std::string> first_files = package_file_set(first_root);
           require(!first_files.empty(),
                   "first package should contain files after the second load");
           require(first_files == package_file_set(second_root),
                   "both same-basename packages should hold the same extracted file set");

           // Reusing an extracted package needs no snapshot or output capacity. A later space
           // shortage must not invalidate an in-process cache hit.
           {
             const std::uintmax_t available = fs::space(first_root).available;
             sima_test::ScopedEnvVar space_check("SIMA_NEAT_SPACE_CHECK", "1");
             sima_test::ScopedEnvVar oversized_reserve("SIMA_MPK_EXTRACT_MIN_FREE_BYTES",
                                                       std::to_string(available + 1));
             simaai::neat::Model cached(first_copy.string());
             require(package_root_of(cached) == first_root,
                     "a cached archive should load without reserving extraction space");
           }

           if (const char* configured_base = std::getenv("SIMA_MPK_EXTRACT_ROOT");
               configured_base && *configured_base) {
             std::error_code relative_ec;
             const fs::path relative =
                 fs::relative(fs::path(first_root), fs::absolute(configured_base), relative_ec);
             require(!relative_ec && !relative.empty() && *relative.begin() != "..",
                     "archive extraction should stay under SIMA_MPK_EXTRACT_ROOT");
           }

           // Concurrent loads of one archive under default options: the extraction mutex serializes
           // them, so the first extracts and the rest resolve through the in-process cache.
           const fs::path shared_copy =
               copy_archive_as(archive, scratch / "shared", "shared.tar.gz");
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

           // The direct-directory contract is structural: an etc/lib/share package is accepted in
           // place without copying or ownership transfer. Copying the first package makes this a
           // caller-owned directory rather than another path inside Core's process root.
           const fs::path organized = scratch / "organized";
           fs::copy(first_root, organized, fs::copy_options::recursive);
           const std::vector<std::string> organized_before = package_file_set(organized);

           simaai::neat::Model from_organized(organized.string());
           require(package_root_of(from_organized) == organized.string(),
                   "an organized model package should load directly from the supplied directory");
           require(package_file_set(organized) == organized_before,
                   "direct model loading must not modify the supplied package");

           simaai::neat::Model::Options keep_opt;
           keep_opt.cleanup_extracted_model_data = false;
           simaai::neat::Model kept_organized(organized.string(), keep_opt);
           require(package_root_of(kept_organized) == organized.string(),
                   "cleanup policy must not change direct-directory ownership");
           require(package_file_set(organized) == organized_before,
                   "cleanup-disabled direct loading must not modify the supplied package");

           std::error_code cleanup_ec;
           fs::remove_all(scratch, cleanup_ec);
         }));

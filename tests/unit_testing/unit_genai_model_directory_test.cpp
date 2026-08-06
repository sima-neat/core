#include "genai/GenAIInternal.h"
#include "test_main.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

class TempDirectory {
public:
  TempDirectory()
      : path_(fs::temp_directory_path() /
              ("neat_genai_model_directory_" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    fs::create_directories(path_);
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

fs::path write_vlm(const fs::path& root, std::optional<bool> is_draft = std::nullopt) {
  fs::create_directories(root / "devkit");
  fs::create_directories(root / "elf_files");

  nlohmann::json config = {{"model_type", "llm-test"}, {"lm_cfg", nlohmann::json::object()}};
  if (is_draft.has_value()) {
    config["lm_cfg"]["speculative_decoding_cfg"] = {{"is_draft", *is_draft},
                                                    {"speculative_budget", *is_draft ? 5 : 16}};
  }
  std::ofstream(root / "devkit" / "vlm_config.json") << config.dump();
  return root;
}

fs::path write_packaged_vlm(const fs::path& package, const std::string& name,
                            std::optional<bool> is_draft) {
  return write_vlm(package / name, is_draft);
}

void require_throws_contains(const std::function<void()>& fn, const std::string& expected) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(e.what(), expected, "unexpected model-directory error");
    return;
  }
  throw std::runtime_error("expected exception containing: " + expected);
}

} // namespace

RUN_TEST(
    "unit_genai_model_directory_test", ([] {
      namespace internal = simaai::neat::genai::internal;

      TempDirectory temp;

      const auto normal_root = write_vlm(temp.path() / "normal");
      const auto normal = internal::inspect_model_directory(normal_root);
      require(normal.package_root == fs::weakly_canonical(normal_root),
              "normal package root mismatch");
      require(normal.root == normal.package_root, "normal runtime root mismatch");
      require(!normal.draft_root.has_value(), "normal model unexpectedly has a draft");

      const auto pair_root = temp.path() / "pair";
      const auto target_root = write_packaged_vlm(pair_root, "target", false);
      const auto draft_root = write_packaged_vlm(pair_root, "draft", true);
      const auto pair = internal::inspect_model_directory(pair_root);
      require(pair.package_root == fs::weakly_canonical(pair_root),
              "speculative package root mismatch");
      require(pair.root == fs::weakly_canonical(target_root), "speculative target mismatch");
      require(pair.draft_root == fs::weakly_canonical(draft_root), "speculative draft mismatch");

      require_throws_contains([&] { (void)internal::inspect_model_directory(target_root); },
                              "pass its parent directory");
      require_throws_contains([&] { (void)internal::inspect_model_directory(draft_root); },
                              "pass its parent directory");

      const auto missing_draft_root = temp.path() / "missing-draft";
      (void)write_packaged_vlm(missing_draft_root, "target", false);
      require_throws_contains([&] { (void)internal::inspect_model_directory(missing_draft_root); },
                              "one target and one draft");

      const auto missing_target_root = temp.path() / "missing-target";
      (void)write_packaged_vlm(missing_target_root, "draft", true);
      require_throws_contains([&] { (void)internal::inspect_model_directory(missing_target_root); },
                              "missing target model");

      const auto duplicate_draft_root = temp.path() / "duplicate-draft";
      (void)write_packaged_vlm(duplicate_draft_root, "target", false);
      (void)write_packaged_vlm(duplicate_draft_root, "draft-a", true);
      (void)write_packaged_vlm(duplicate_draft_root, "draft-b", true);
      require_throws_contains(
          [&] { (void)internal::inspect_model_directory(duplicate_draft_root); },
          "Multiple draft models");
    }));

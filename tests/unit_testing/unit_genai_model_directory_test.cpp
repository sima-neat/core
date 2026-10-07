#include "genai/GenAIInternal.h"
#include "genai/ScopedFileProvider.h"
#include "test_main.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

class TrackingProvider : public simaai::llima::FileProvider {
public:
  explicit TrackingProvider(const fs::path& root) : root_(root) {}
  fs::path get_path(std::string_view name) override {
    names.emplace_back(name);
    return root_ / name;
  }
  fs::path reserve(std::string_view name) override {
    names.emplace_back(name);
    return root_ / name;
  }
  bool exists(std::string_view name) override {
    names.emplace_back(name);
    return fs::is_regular_file(root_ / name);
  }
  std::unique_ptr<std::istream> open_stream(std::string_view name) override {
    names.emplace_back(name);
    auto stream = std::make_unique<std::ifstream>(root_ / name);
    if (!*stream)
      throw std::runtime_error("Missing test model asset");
    return stream;
  }
  void release(std::string_view name) override {
    released = name;
  }
  void fetch(const fs::path& path) override {
    fetched = path;
  }
  void evict(const fs::path& path) override {
    evicted = path;
  }
  bool pulls_files() const override {
    return true;
  }
  std::vector<std::string> names;
  std::string released;
  fs::path fetched, evicted;

private:
  fs::path root_;
};

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

fs::path write_vlm(const fs::path& root, std::optional<bool> is_draft = std::nullopt,
                   nlohmann::json vision_model_name = nullptr) {
  fs::create_directories(root / "devkit");
  fs::create_directories(root / "elf_files");

  nlohmann::json config = {{"model_type", "llm-test"}, {"lm_cfg", nlohmann::json::object()}};
  if (is_draft.has_value()) {
    config["lm_cfg"]["speculative_decoding_cfg"] = {{"is_draft", *is_draft},
                                                    {"speculative_budget", *is_draft ? 5 : 16}};
  }
  if (!vision_model_name.is_null()) {
    config["vm_cfg"] = nlohmann::json::object();
    config["mm_cfg"] = nlohmann::json::object();
    config["vision_model_name"] = std::move(vision_model_name);
  }

  std::ofstream out(root / "devkit" / "vlm_config.json");
  require(static_cast<bool>(out), "failed to create VLM config");
  out << config.dump();
  require(static_cast<bool>(out), "failed to write VLM config");
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
      const auto wrapped_root = temp.path() / "wrapped";
      (void)write_vlm(wrapped_root / "sima_files");
      require(internal::inspect_model_directory(wrapped_root).root == wrapped_root / "sima_files",
              "ordinary packaged models must retain runtime-root normalization");

      const auto single_vision_root =
          write_vlm(temp.path() / "single-vision", std::nullopt, "vision");
      require(internal::inspect_model_directory(single_vision_root).accepts_image,
              "single-ELF VLM should accept images");

      const auto multi_vision_root = write_vlm(temp.path() / "multi-vision", std::nullopt,
                                               nlohmann::json::array({"vision_0", "vision_1"}));
      require(internal::inspect_model_directory(multi_vision_root).accepts_image,
              "multi-ELF VLM should accept images");

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

      auto files = std::make_shared<TrackingProvider>(pair_root);
      auto target_files = std::make_shared<internal::ScopedFileProvider>(files, "target");
      auto draft_files = std::make_shared<internal::ScopedFileProvider>(files, "draft");
      const auto remote = internal::provider_model_context(target_root, target_files, draft_files);
      require(remote.info.root == target_root && remote.info.draft_root == draft_root &&
                  remote.info.accepts_text && remote.draft_files == draft_files,
              "provider-backed pair must retain both models");
      const auto target_elf = target_files->reserve("elf_files/model.elf");
      const auto draft_elf = draft_files->reserve("elf_files/model.elf");
      require(target_elf == target_root / "elf_files/model.elf" &&
                  draft_elf == draft_root / "elf_files/model.elf",
              "same ELF names must remain isolated between target and draft");
      draft_files->fetch(draft_elf);
      draft_files->evict(draft_elf);
      draft_files->release("devkit/vlm_config.json");
      require(files->fetched == draft_elf && files->evicted == draft_elf &&
                  files->released == "draft/devkit/vlm_config.json" && draft_files->pulls_files(),
              "scoped providers must preserve deferred fetch/load/evict semantics");
      for (const auto& name : files->names)
        require(name.starts_with("target/") || name.starts_with("draft/"),
                "all requests must retain the model prefix");
      require_throws_contains([&] { (void)draft_files->get_path("../target/weights.bin"); },
                              "Unsafe model asset path");
      require_throws_contains([&] { (void)draft_files->reserve("/tmp/weights.bin"); },
                              "relative model asset path");
      require_throws_contains(
          [&] { (void)internal::provider_model_context(target_root, target_files); },
          "one target and one draft");
      require_throws_contains(
          [&] { (void)internal::provider_model_context(target_root, target_files, target_files); },
          "is_draft=true");
      require_throws_contains(
          [&] { (void)internal::provider_model_context(draft_root, draft_files, target_files); },
          "one target and one draft");
      auto normal_files = std::make_shared<TrackingProvider>(normal_root);
      require(!internal::provider_model_context(normal_root, normal_files).info.draft_root,
              "ordinary provider-backed model must remain supported");
      require_throws_contains(
          [&] { (void)internal::provider_model_context(normal_root, normal_files, draft_files); },
          "one target and one draft");
      const auto asr_root = temp.path() / "asr";
      fs::create_directories(asr_root / "devkit");
      std::ofstream(asr_root / "devkit/whisper_config.json") << "{}";
      auto asr_files = std::make_shared<TrackingProvider>(asr_root);
      require(internal::provider_model_context(asr_root, asr_files).info.accepts_audio,
              "ASR provider-backed model must remain supported");
      require_throws_contains(
          [&] { (void)internal::provider_model_context(asr_root, asr_files, draft_files); },
          "one target and one draft");
    }));

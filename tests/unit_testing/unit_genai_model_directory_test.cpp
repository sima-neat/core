#include "genai/GenAIInternal.h"
#include "test_main.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

namespace {

namespace fs = std::filesystem;

class TempDirGuard {
public:
  explicit TempDirGuard(fs::path path) : path_(std::move(path)) {}

  ~TempDirGuard() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

private:
  fs::path path_;
};

void write_vlm_config(const fs::path& root, const std::string& vision_model_name) {
  std::error_code ec;
  fs::create_directories(root / "devkit", ec);
  require(!ec, "failed to create devkit directory");
  fs::create_directories(root / "elf_files", ec);
  require(!ec, "failed to create elf_files directory");

  std::ofstream out(root / "devkit" / "vlm_config.json");
  require(static_cast<bool>(out), "failed to create VLM config");
  out << R"({"vm_cfg":{},"mm_cfg":{},"vision_model_name":)" << vision_model_name << '}';
  require(static_cast<bool>(out), "failed to write VLM config");
}

} // namespace

RUN_TEST("unit_genai_model_directory_test", ([] {
           namespace internal = simaai::neat::genai::internal;

           const fs::path root = fs::temp_directory_path() / "sima_unit_genai_model_directory_test";
           TempDirGuard cleanup(root);
           std::error_code ec;
           fs::remove_all(root, ec);

           const fs::path single_model = root / "single";
           write_vlm_config(single_model, R"("vision")");
           require(internal::inspect_model_directory(single_model).accepts_image,
                   "single-ELF VLM should accept images");

           const fs::path multi_model = root / "multi";
           write_vlm_config(multi_model, R"(["vision_0","vision_1"])");
           require(internal::inspect_model_directory(multi_model).accepts_image,
                   "multi-ELF VLM should accept images");
         }));

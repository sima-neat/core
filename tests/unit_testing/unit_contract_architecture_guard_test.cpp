#include "asset_utils.h"
#include "test_main.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path);
  require(in.is_open(), "failed to open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void require_no_token(const std::string& text, const std::string& token,
                      const std::filesystem::path& path) {
  require(text.find(token) == std::string::npos,
          path.string() + " should not construct runtime spec types directly: " + token);
}

void require_token_absent_in_tree(const std::filesystem::path& root,
                                  const std::vector<std::filesystem::path>& subdirs,
                                  const std::string& token) {
  const std::filesystem::path self = std::filesystem::path(__FILE__).filename();
  for (const auto& subdir : subdirs) {
    const auto dir = root / subdir;
    if (!std::filesystem::exists(dir)) {
      continue;
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (entry.path().filename() == self) {
        continue;
      }
      const std::string text = read_text(entry.path());
      require(text.find(token) == std::string::npos,
              entry.path().string() + " should not reference deleted legacy wrapper: " + token);
    }
  }
}

void require_token_absent_in_tree_except(const std::filesystem::path& root,
                                         const std::vector<std::filesystem::path>& subdirs,
                                         const std::string& token,
                                         const std::vector<std::filesystem::path>& allowed_paths) {
  const std::filesystem::path self = std::filesystem::path(__FILE__).filename();
  for (const auto& subdir : subdirs) {
    const auto dir = root / subdir;
    if (!std::filesystem::exists(dir)) {
      continue;
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (entry.path().filename() == self) {
        continue;
      }
      const std::string text = read_text(entry.path());
      if (text.find(token) == std::string::npos) {
        continue;
      }
      const bool allowed = std::find(allowed_paths.begin(), allowed_paths.end(), entry.path()) !=
                           allowed_paths.end();
      require(allowed, entry.path().string() + " should not include restricted header: " + token);
    }
  }
}

} // namespace

RUN_TEST(
    "unit_contract_architecture_guard_test", ([] {
      const std::filesystem::path root = sima_test::test_source_root();

      const std::vector<std::filesystem::path> guarded_files = {
          root / "src/model/Model.cpp",
          root / "src/model/ModelPack.cpp",
          root / "src/pipeline/internal/contract/ContractCompiler.cpp",
          root / "src/nodes/sima/Preproc.cpp",
          root / "src/nodes/sima/Quant.cpp",
          root / "src/nodes/sima/Tess.cpp",
          root / "src/nodes/sima/QuantTess.cpp",
          root / "src/nodes/sima/Dequant.cpp",
          root / "src/nodes/sima/Detess.cpp",
          root / "src/nodes/sima/DetessDequant.cpp",
      };

      const std::vector<std::string> forbidden_tokens = {
          "LogicalInputStaticSpec ",   "LogicalTensorStaticSpec ", "InputBindingStaticSpec ",
          "PhysicalBufferStaticSpec ", "StageOutputRoute{",
      };

      for (const auto& path : guarded_files) {
        const std::string text = read_text(path);
        for (const auto& token : forbidden_tokens) {
          require_no_token(text, token, path);
        }
      }

      const std::vector<std::filesystem::path> processcvu_helper_guarded_files = {
          root / "src/model/Model.cpp",
          root / "src/model/ModelPack.cpp",
          root / "src/pipeline/internal/contract/ContractCompiler.cpp",
          root / "src/nodes/sima/Preproc.cpp",
          root / "src/nodes/sima/Dequant.cpp",
          root / "src/nodes/sima/Detess.cpp",
          root / "src/nodes/sima/DetessDequant.cpp",
      };
      const std::vector<std::string> processcvu_helper_forbidden_tokens = {
          "ProcessCvuSingleOutputFactsSpec ",
          "ProcessCvuPackedRouteEntry ",
          "CompiledProcessCvuRuntimeConfig ",
          "model/internal/ProcessCvuFamily.h",
      };
      for (const auto& path : processcvu_helper_guarded_files) {
        const std::string text = read_text(path);
        for (const auto& token : processcvu_helper_forbidden_tokens) {
          require_no_token(text, token, path);
        }
      }

      const std::vector<std::filesystem::path> wrapper_adapter_guarded_files = {
          root / "src/nodes/sima/Preproc.cpp",
          root / "src/nodes/sima/Quant.cpp",
          root / "src/nodes/sima/Tess.cpp",
          root / "src/nodes/sima/QuantTess.cpp",
      };
      const std::vector<std::string> wrapper_adapter_forbidden_tokens = {
          "build_processcvu_payload_from_options(",
          "build_processcvu_facts_from_options(",
          "build_processcvu_payload_from_runtime_config(",
          "build_processcvu_facts_from_runtime_config(",
      };
      for (const auto& path : wrapper_adapter_guarded_files) {
        const std::string text = read_text(path);
        for (const auto& token : wrapper_adapter_forbidden_tokens) {
          require_no_token(text, token, path);
        }
      }

      const std::vector<std::filesystem::path> guarded_subdirs = {
          "src",
          "include",
          "tests",
      };
      // Keep the canonical processcvu builders; only deleted compatibility wrappers belong here.
      const std::vector<std::string> deleted_wrapper_tokens = {
          "build_preproc_contract_from_mpk_contract(",
          "build_preadapter_contract_from_mpk_contract(",
          "build_detess_contract_from_mpk_contract(",
          "build_dequant_contract_from_mpk_contract(",
          "build_detessdequant_contract_from_mpk_contract(",
          "build_preproc_compiled_contract(",
          "build_processcvu_compiled_contract_from_mpk_stage(",
          "build_processcvu_compiled_contract_from_runtime(",
          "build_processcvu_node_contract_from_runtime_config(",
          "build_preproc_runtime_config_from_options(",
          "build_preproc_single_handoff_facts_from_payload(",
          "build_mla_stage_payload_from_runtime_properties(",
          "build_boxdecode_payload(",
          "build_processcvu_contract_from_mpk_preproc(",
          "build_processcvu_contract_from_mpk_preadapter(",
          "build_processcvu_contract_from_mpk_detess(",
          "build_processcvu_contract_from_mpk_dequant(",
          "build_processcvu_contract_from_mpk_detessdequant(",
          "build_model_managed_preadapter_contract(",
          "build_model_managed_preproc_contract(",
          "build_model_managed_detess_contract(",
          "build_model_managed_dequant_contract(",
          "build_model_managed_detessdequant_contract(",
          "build_model_managed_dequant_processcvu_contract(",
          "build_transport_compiled_contract(",
      };
      for (const auto& token : deleted_wrapper_tokens) {
        require_token_absent_in_tree(root, guarded_subdirs, token);
      }

      const std::vector<std::filesystem::path> adapter_header_allowed_paths = {
          root / "src/nodes/sima/Quant.cpp",
          root / "src/nodes/sima/Tess.cpp",
          root / "src/nodes/sima/QuantTess.cpp",
          root / "src/pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.cpp",
          root / "tests/unit_testing/unit_contract_compiler_processcvu_test.cpp",
          root / "tests/unit_testing/unit_dequant_processcvu_contract_equivalence_test.cpp",
          root / "tests/unit_testing/unit_dequant_node_fragment_test.cpp",
      };
      require_token_absent_in_tree_except(
          root, guarded_subdirs,
          "#include \"pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.h\"",
          adapter_header_allowed_paths);

      const std::filesystem::path modelpack_path = root / "src/model/ModelPack.cpp";
      const std::string modelpack_text = read_text(modelpack_path);
      const std::vector<std::string> modelpack_processcvu_forbidden_tokens = {
          "build_processcvu_mpk_preproc_compile_inputs(",
          "build_processcvu_mpk_preadapter_compile_inputs(",
          "build_processcvu_mpk_detess_compile_inputs(",
          "build_processcvu_mpk_dequant_compile_inputs(",
          "build_processcvu_mpk_detessdequant_compile_inputs(",
      };
      for (const auto& token : modelpack_processcvu_forbidden_tokens) {
        require_no_token(modelpack_text, token, modelpack_path);
      }
    }));

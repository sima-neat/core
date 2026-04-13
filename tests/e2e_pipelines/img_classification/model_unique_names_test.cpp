#include "model/Model.h"

#include "pipeline/Session.h"

#include "test_utils.h"

#include "asset_utils.h"
#include "cli_utils.h"

#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string extract_element_name(const std::string& pipeline, const std::string& factory) {
  const std::string needle = factory + " name=";
  size_t pos = pipeline.find(needle);
  if (pos == std::string::npos)
    return "";
  pos += needle.size();
  size_t end = pos;
  while (end < pipeline.size()) {
    const char c = pipeline[end];
    if (std::isspace(static_cast<unsigned char>(c)) || c == '!' || c == ';')
      break;
    ++end;
  }
  return pipeline.substr(pos, end - pos);
}

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  std::string tar_gz;
  sima_test::get_arg(argc, argv, "--model", tar_gz);

  if (tar_gz.empty()) {
    tar_gz = sima_test::resolve_resnet50_tar();
    if (tar_gz.empty()) {
      std::cerr << "Failed to resolve resnet50 tar.gz. "
                << "Set SIMA_RESNET50_TAR or run 'sima-cli modelzoo -v 2.0.0 get resnet_50'.\n";
      return 3;
    }
  }
  if (!fs::exists(tar_gz)) {
    std::cerr << "Model tar.gz missing: " << tar_gz << "\n";
    return 3;
  }

  try {
    constexpr int kInferWidth = 224;
    constexpr int kInferHeight = 224;

    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "RGB";
    model_opt.input_max_width = kInferWidth;
    model_opt.input_max_height = kInferHeight;
    model_opt.input_max_depth = 3;
    simaai::neat::Model model(tar_gz, model_opt);

    simaai::neat::Session p1;
    p1.add(model.session());
    const std::string pipeline1 = p1.describe_backend();
    const std::string appsrc1 = extract_element_name(pipeline1, "appsrc");
    const std::string appsink1 = extract_element_name(pipeline1, "appsink");
    require(!pipeline1.empty(), "pipeline string is empty");
    require(!appsrc1.empty(), "appsrc name not found");
    require(!appsink1.empty(), "appsink name not found");

    simaai::neat::Session p2;
    p2.add(model.session());
    const std::string pipeline2 = p2.describe_backend();
    const std::string appsrc2 = extract_element_name(pipeline2, "appsrc");
    const std::string appsink2 = extract_element_name(pipeline2, "appsink");
    require(!pipeline2.empty(), "pipeline string is empty");
    require(!appsrc2.empty(), "appsrc name not found");
    require(!appsink2.empty(), "appsink name not found");

    std::cout << "[OK] neatmodel_pipeline_names_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

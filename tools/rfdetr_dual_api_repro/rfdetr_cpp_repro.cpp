#include <neat.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace neat = simaai::neat;

namespace {

std::string env_or(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return (value && *value) ? std::string(value) : std::string(fallback);
}

int env_int_or(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) {
    return fallback;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

std::vector<float> random_vector(std::size_t count) {
  std::vector<float> out(count);
  std::mt19937 rng(12345U);
  std::uniform_real_distribution<float> dist(0.0F, 1.0F);
  for (float& value : out) {
    value = dist(rng);
  }
  return out;
}

} // namespace

int main() {
  const std::string model_path =
      env_or("RFDETR_MODEL_PATH", "/mnt/nfs/sima-neat/tmp/drive_model/"
                                  "rfdetr_576_simplified_transformer_after_gather_base_mpk.tar.gz");
  const std::string run_target = env_or("RFDETR_PROCESSCVU_TARGET", "A65");
  const int timeout_ms = env_int_or("RFDETR_TIMEOUT_MS", 60000);

  std::cout << "[cpp] model=" << model_path << "\n";
  std::cout << "[cpp] processcvu pre/post target=" << run_target << "\n";

  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Tensor;
  opt.preprocess.enable = neat::AutoFlag::Off;
  opt.processcvu.pre_run_target = run_target;
  opt.processcvu.post_run_target = run_target;

  try {
    neat::Model model(model_path, opt);
    std::cout << "[cpp] Model loaded\n";

    try {
      const auto specs = model.input_specs();
      std::cout << "[cpp] input_specs_count=" << specs.size() << "\n";
      for (std::size_t i = 0; i < specs.size(); ++i) {
        std::cout << "[cpp] input_spec[" << i << "] rank=" << specs[i].shape.size() << " shape=";
        for (std::size_t d = 0; d < specs[i].shape.size(); ++d) {
          std::cout << (d == 0 ? "" : "x") << specs[i].shape[d];
        }
        std::cout << "\n";
      }
    } catch (const std::exception& ex) {
      std::cout << "[cpp] input_specs failed: " << ex.what() << "\n";
    }

    auto backbone = random_vector(1U * 36U * 36U * 256U);
    auto gather = random_vector(1U * 300U * 4U);
    neat::TensorList inputs;
    inputs.push_back(
        neat::Tensor::from_vector(backbone, {1, 36, 36, 256}, neat::TensorMemory::EV74));
    inputs.push_back(neat::Tensor::from_vector(gather, {1, 300, 4}, neat::TensorMemory::EV74));

    std::cout << "[cpp] running input_count=" << inputs.size() << "\n";
    auto outputs = model.run(inputs, timeout_ms);
    std::cout << "[cpp] run ok output_count=" << outputs.size() << "\n";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      std::cout << "[cpp] output[" << i << "] " << outputs[i].debug_string() << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[cpp] failed: " << ex.what() << "\n";
    return 2;
  }
}

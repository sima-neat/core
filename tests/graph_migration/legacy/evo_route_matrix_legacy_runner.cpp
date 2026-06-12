#include <neat.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace simaai::neat;

namespace {

std::string arg_value(int argc, char** argv, const std::string& key) {
  const std::string prefix = key + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg.rfind(prefix, 0) == 0) {
      return arg.substr(prefix.size());
    }
    if (arg == key && i + 1 < argc) {
      return argv[++i] ? argv[i] : "";
    }
  }
  return {};
}

int int_arg(int argc, char** argv, const std::string& key, int fallback) {
  const std::string raw = arg_value(argc, argv, key);
  if (raw.empty()) {
    return fallback;
  }
  try {
    return std::stoi(raw);
  } catch (...) {
    return fallback;
  }
}

Tensor make_tensor(const TensorSpec& spec, float fill) {
  std::vector<int64_t> shape = spec.shape;
  if (shape.empty()) {
    shape = {1};
  }
  std::size_t count = 1;
  for (const auto dim : shape) {
    count *= static_cast<std::size_t>(dim > 0 ? dim : 1);
  }
  std::vector<float> data(count, fill);
  return Tensor::from_vector(data, shape, TensorMemory::EV74);
}

} // namespace

int main(int argc, char** argv) {
  const std::string model_path = arg_value(argc, argv, "--model");
  const std::string pre = arg_value(argc, argv, "--pre");
  const std::string post = arg_value(argc, argv, "--post");
  const int frames = int_arg(argc, argv, "--frames", 1);
  if (model_path.empty() || pre.empty() || post.empty() || frames <= 0) {
    std::cerr << "Usage: evo_route_matrix_legacy_runner --model <path> --pre <A65|EV74> --post "
                 "<A65|EV74> "
                 "[--frames N]\n";
    return 2;
  }

  try {
    Model::Options opt;
    opt.preprocess.kind = InputKind::Tensor;
    opt.preprocess.enable = AutoFlag::On;
    opt.processcvu.pre_run_target = pre;
    opt.processcvu.post_run_target = post;
    opt.cleanup_extracted_model_data = true;

    Model model(model_path, opt);
    std::cout << "MODEL_INIT_OK model=" << model_path << " pre=" << pre << " post=" << post
              << " frames=" << frames << "\n";

    TensorList inputs;
    const auto specs = model.input_specs();
    inputs.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
      inputs.push_back(make_tensor(specs[i], 0.01f * static_cast<float>(i + 1)));
    }

    std::size_t output_count = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int frame = 0; frame < frames; ++frame) {
      auto outputs = model.run(inputs, 20000);
      output_count += outputs.size();
      std::cout << "FRAME_OK frame=" << frame << " outputs=" << outputs.size() << "\n";
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "EVO_RESULT status=PASS frames=" << frames << " outputs=" << output_count
              << " seconds=" << seconds << " fps=" << (seconds > 0.0 ? frames / seconds : 0.0)
              << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "EVO_RESULT status=FAIL error=" << e.what() << "\n";
    return 1;
  }
}

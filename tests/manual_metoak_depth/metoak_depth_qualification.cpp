#include <neat.h>
#include <nlohmann/json.hpp>

#include <bit>
#include <chrono>
#include <csignal>
#include <pthread.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace n = simaai::neat;
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::vector<std::uint8_t> read_bytes(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  require(stream.good(), "cannot open " + path.string());
  const auto length = stream.tellg();
  require(length >= 0, "cannot measure " + path.string());
  std::vector<std::uint8_t> result(static_cast<std::size_t>(length));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
  require(stream.good(), "cannot read " + path.string());
  return result;
}
template <typename T>
n::Tensor make_tensor(const std::vector<std::uint8_t>& bytes,
                      const std::vector<std::int64_t>& shape) {
  require(bytes.size() % sizeof(T) == 0, "fixture element size mismatch");
  // Every input is a real device-backed subview; the leading sentinel must not
  // reach a destination segment or survive as its metadata byte_offset.
  std::vector<T> data(bytes.size() / sizeof(T) + 1U, T{});
  std::memcpy(data.data() + 1U, bytes.data(), bytes.size());
  auto tensor =
      n::Tensor::from_vector(data, {static_cast<std::int64_t>(data.size())}, n::TensorMemory::EV74);
  tensor.shape = shape;
  tensor.strides_bytes.clear();
  tensor.byte_offset = sizeof(T);
  return tensor;
}
struct Fixture {
  int width;
  int height;
  n::TensorList inputs;
  std::vector<std::vector<std::uint8_t>> expected;
};
Fixture load_fixture(const fs::path& dir) {
  std::ifstream stream(dir / "metadata.json");
  require(stream.good(), "fixture metadata missing");
  Json metadata;
  stream >> metadata;
  Fixture fixture{metadata.at("width").get<int>(), metadata.at("height").get<int>(), {}, {}};
  const std::vector<std::string> names{"y_src",    "u_src",     "v_src",
                                       "disp_src", "bf_mm_src", "proj_src"};
  const auto& inputs = metadata.at("inputs");
  require(inputs.size() == names.size(), "fixture must contain six inputs");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    require(inputs[i].at("name") == names[i], "fixture input order mismatch");
    const auto bytes = read_bytes(dir / inputs[i].at("file").get<std::string>());
    const auto shape = inputs[i].at("shape").get<std::vector<std::int64_t>>();
    auto tensor = i < 3    ? make_tensor<std::uint8_t>(bytes, shape)
                  : i == 3 ? make_tensor<std::uint16_t>(bytes, shape)
                           : make_tensor<float>(bytes, shape);
    tensor.route.name = tensor.route.backend_name = tensor.route.segment_name = names[i];
    tensor.route.logical_index = tensor.route.physical_index = tensor.route.route_slot =
        tensor.route.backend_output_index = static_cast<int>(i);
    tensor.route.memory_index = 0;
    if (i < 4)
      tensor.layout = n::TensorLayout::HW;
    fixture.inputs.push_back(std::move(tensor));
  }
  const std::vector<std::string> outputs{"rgb_dst", "depth_dst", "points_dst"};
  require(metadata.at("outputs").size() == 3, "fixture must contain three outputs");
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& entry = metadata.at("outputs")[i];
    require(entry.at("name") == outputs[i], "fixture output order mismatch");
    fixture.expected.push_back(read_bytes(dir / entry.at("file").get<std::string>()));
  }
  return fixture;
}
std::vector<std::uint8_t> tensor_bytes(const n::Tensor& tensor) {
  std::vector<std::uint8_t> bytes(tensor.dense_bytes_tight());
  require(tensor.copy_dense_bytes_tight_to(bytes.data(), bytes.size()), "output copy failed");
  return bytes;
}
void validate(const n::TensorList& outputs, const Fixture& fixture) {
  require(outputs.size() == 3, "must publish RGB, depth and XYZ together");
  const n::TensorDType types[] = {n::TensorDType::UInt8, n::TensorDType::UInt16,
                                  n::TensorDType::Float32};
  const std::vector<std::vector<std::int64_t>> shapes{{fixture.height, fixture.width, 3},
                                                      {fixture.height, fixture.width},
                                                      {fixture.height, fixture.width, 3}};
  for (std::size_t i = 0; i < 3; ++i) {
    require(outputs[i].dtype == types[i], "output logical dtype mismatch");
    require(outputs[i].shape == shapes[i], "output logical shape mismatch");
    const auto actual = tensor_bytes(outputs[i]);
    require(actual.size() == fixture.expected[i].size(), "output byte size mismatch");
    if (i < 2) {
      require(actual == fixture.expected[i], "RGB/depth differs from independent CPU reference");
      continue;
    }
    for (std::size_t p = 0; p < actual.size(); p += sizeof(float)) {
      float a, e;
      std::memcpy(&a, actual.data() + p, sizeof(a));
      std::memcpy(&e, fixture.expected[i].data() + p, sizeof(e));
      require(std::isnan(a) == std::isnan(e), "XYZ invalid mask mismatch");
      if (!std::isnan(e)) {
        require(std::isfinite(a) && std::abs(a - e) <= 1e-5f + 1e-5f * std::abs(e),
                "XYZ differs from independent CPU reference");
      }
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 4 && std::string(argv[1]) == "--guarded-hardware",
            "usage: metoak_depth_qualification --guarded-hardware fixture1 fixture2 [fixtureN...]\n"
            "Run only under the exclusive firmware/ownership/recovery guard. This is NOT a robot "
            "launch.");
    require(std::endian::native == std::endian::little, "fixtures require little endian");
    // Inherit this mask in pipeline threads. A shell timeout/terminal disconnect
    // must not destroy DMA owners. The external guardian retains ownership locks
    // and establishes hardware quiescence before using SIGKILL if we park.
    sigset_t blocked;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGINT);
    sigaddset(&blocked, SIGTERM);
    sigaddset(&blocked, SIGHUP);
    require(pthread_sigmask(SIG_BLOCK, &blocked, nullptr) == 0, "cannot protect DMA-owner signals");
    {
      // Keep every input alive through completion, including on an ambiguous execution failure.
      std::vector<Fixture> fixtures;
      for (int i = 2; i < argc; ++i)
        fixtures.push_back(load_fixture(argv[i]));
      for (const auto& fixture : fixtures)
        require(fixture.width == fixtures[0].width && fixture.height == fixtures[0].height,
                "one reusable Run requires identical geometry");
      n::Graph graph;
      n::InputOptions input;
      input.payload_type = n::PayloadType::Tensor;
      input.memory_policy = n::InputMemoryPolicy::Ev74;
      input.is_live = true;
      input.do_timestamp = true;
      input.block = true;
      input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                            "storage=(string)tensorbuffer";
      graph.add(n::nodes::Input(input));
      n::MetoakDepthOptions opt;
      opt.width = fixtures[0].width;
      opt.height = fixtures[0].height;
      graph.add(n::nodes::MetoakDepth(opt));
      graph.add(n::nodes::Output());
      n::RunOptions options;
      options.output_memory = n::OutputMemory::Owned;
      auto run = graph.build(fixtures[0].inputs, options);
      std::vector<n::TensorList> held;
      for (std::size_t i = 0; i < fixtures.size(); ++i) {
        n::TensorList outputs;
        const auto start = std::chrono::steady_clock::now();
        try {
          outputs = run.run(fixtures[i].inputs, 15000);
        } catch (const std::exception& error) {
          // Never retry/free input or run state after an ambiguous DMA timeout. The outer
          // guarded procedure must establish safe hardware quiescence before terminating us.
          std::cerr << "PARKED: execution failed; preserve process/memory until guarded recovery: "
                    << error.what() << std::endl;
          for (;;)
            std::this_thread::sleep_for(std::chrono::seconds(60));
        } catch (...) {
          std::cerr << "PARKED: non-standard execution exception; guarded recovery required"
                    << std::endl;
          for (;;)
            std::this_thread::sleep_for(std::chrono::seconds(60));
        }
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
        validate(outputs, fixtures[i]);
        held.push_back(std::move(outputs));
        for (std::size_t prior = 0; prior <= i; ++prior)
          validate(held[prior], fixtures[prior]);
        std::cout << "PASS frame " << i << ": all outputs, calibration and held-output lifetime"
                  << "; end_to_end_host_pipeline_elapsed_us=" << elapsed_us << std::endl;
      }
    }
    // Emit only after successful Run and all tensor-owner destruction. The
    // external guardian treats any exit without this marker as ambiguous.
    std::cout << "GUARDED_TEST_COMPLETE no_pending_dma" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}

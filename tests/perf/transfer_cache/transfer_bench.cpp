#include "pipeline/TensorCore.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>
using namespace simaai::neat;
int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 5)
      throw std::runtime_error("usage: transfer_bench bytes held iterations [cycle_sizes]");
    const auto bytes = std::stoull(argv[1]);
    const auto held_count = std::stoi(argv[2]);
    const auto iterations = std::stoi(argv[3]);
    const auto cycle_sizes = argc == 5 ? std::stoi(argv[4]) : 1;
    if (cycle_sizes < 1)
      throw std::runtime_error("cycle_sizes must be positive");
    const auto capacity = bytes + (cycle_sizes - 1) * 4096U;
    Tensor source;
    source.storage = make_cpu_owned_storage(capacity);
    source.dtype = TensorDType::Float32;
    source.shape = {static_cast<int64_t>(bytes / sizeof(float))};
    source.strides_bytes = {sizeof(float)};
    source.device = {DeviceType::CPU, 0};
    source.read_only = false;
    {
      auto map = source.map_write();
      std::memset(map.data, 77, capacity);
    }
    std::vector<Tensor> held;
    for (int i = 0; i < held_count; ++i)
      held.push_back(source.cvu());
    for (int i = 0; i < 12; ++i) {
      source.shape = {static_cast<int64_t>((bytes + (i % cycle_sizes) * 4096U) / sizeof(float))};
      auto result = source.cvu();
    }
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    std::vector<double> ms;
    ms.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      source.shape = {static_cast<int64_t>((bytes + (i % cycle_sizes) * 4096U) / sizeof(float))};
      const auto start = std::chrono::steady_clock::now();
      { auto result = source.cvu(); }
      const auto end = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    getrusage(RUSAGE_SELF, &after);
    {
      auto result = source.cvu();
      auto map = result.map_read();
      auto* p = static_cast<const uint8_t*>(map.data);
      for (std::size_t i = 0; i < bytes; ++i)
        if (p[i] != 77)
          throw std::runtime_error("payload mismatch");
    }
    for (const auto& t : held) {
      auto map = t.map_read();
      auto* p = static_cast<const uint8_t*>(map.data);
      if (p[0] != 77 || p[bytes - 1] != 77)
        throw std::runtime_error("held payload mismatch");
    }
    std::cout << "{\"bytes\":" << bytes << ",\"held\":" << held_count
              << ",\"cycle_sizes\":" << cycle_sizes << ",\"iterations\":" << iterations
              << ",\"minor_faults\":" << after.ru_minflt - before.ru_minflt
              << ",\"payload_ok\":true,\"samples_ms\":[";
    for (std::size_t i = 0; i < ms.size(); ++i)
      std::cout << (i ? "," : "") << ms[i];
    std::cout << "]}" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

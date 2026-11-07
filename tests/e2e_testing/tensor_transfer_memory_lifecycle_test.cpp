#include "pipeline/TensorCore.h"

#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h>
#include <dlfcn.h>

#include <cerrno>
#include <cstdarg>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace simaai::neat;
constexpr std::size_t mib = 1024U * 1024U;
// Acceptance bounds for the documented process-wide transfer cache.
constexpr std::size_t cache_byte_limit = 256U * mib;
constexpr std::size_t idle_allocation_limit = 128U;
thread_local std::size_t heap_allocations = 0;
thread_local int fail_heap_allocations = 0;
thread_local int injected_failures = 0;
thread_local bool capture_retry = false;
thread_local std::size_t retry_dma_bytes = 0;

void check(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Count this process's real DMA allocations, once per backing object. RSS and
// global CmaFree cannot distinguish idle cached DMA from other device workloads.
struct DmaStats {
  std::size_t bytes = 0;
  std::size_t allocations = 0;
};

DmaStats dma_stats() {
  std::set<std::pair<dev_t, ino_t>> seen;
  std::size_t total = 0;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
    const auto fd = entry.path().filename().string();
    std::ifstream info("/proc/self/fdinfo/" + fd);
    check(info.good(), "cannot read DMA accounting from /proc/self/fdinfo/" + fd);
    bool dma = false;
    std::size_t bytes = 0;
    std::string line;
    while (std::getline(info, line)) {
      if (line.starts_with("exp_name:"))
        dma = true;
      if (line.starts_with("size:"))
        bytes = std::stoull(line.substr(5));
    }
    if (!dma)
      continue;
    check(bytes > 0, "DMA exporter did not expose allocation size in fdinfo");
    struct stat status {};
    check(::stat(entry.path().c_str(), &status) == 0, "cannot identify DMA backing object");
    if (seen.emplace(status.st_dev, status.st_ino).second)
      total += bytes;
  }
  return {total, seen.size()};
}

std::size_t dma_bytes() {
  return dma_stats().bytes;
}

Tensor transfer(std::size_t bytes, bool dms, unsigned char pattern) {
  Tensor cpu;
  cpu.storage = make_cpu_owned_storage(bytes);
  cpu.dtype = TensorDType::UInt8;
  cpu.shape = {static_cast<int64_t>(bytes)};
  cpu.strides_bytes = {1};
  cpu.device = {DeviceType::CPU, 0};
  cpu.read_only = false;
  {
    auto write = cpu.map_write();
    check(write.data && write.size_bytes >= bytes, "cannot map CPU fixture");
    std::memset(write.data, pattern, bytes);
  }
  return dms ? cpu.mla(true) : cpu.cvu();
}

void verify(const Tensor& tensor, unsigned char pattern) {
  auto read = tensor.map_read();
  const auto bytes = tensor.dense_bytes_tight();
  check(read.data && read.size_bytes == bytes, "retained tensor exposes the wrong byte span");
  const auto* data = static_cast<const unsigned char*>(read.data);
  for (std::size_t i = 0; i < bytes; ++i) {
    if (data[i] != pattern)
      throw std::runtime_error("transfer reused or corrupted a still-held DMA allocation");
  }
}

void run(bool dms) {
  const auto initial = dma_bytes();
  std::vector<Tensor> held;
  for (unsigned char i = 0; i < 4; ++i)
    held.push_back(transfer(mib, dms, i + 1));
  check(dma_bytes() >= initial + 4U * mib,
        "fdinfo does not account for live DMA allocations; cannot validate retention");

  // Warm one size, then verify sequential same-size work stabilizes.
  { auto warm = transfer(mib, dms, 9); }
  const auto baseline = dma_bytes();
  for (int i = 0; i < 32; ++i) {
    auto tensor = transfer(mib, dms, 9);
    verify(tensor, 9);
  }
  const auto fixed = dma_bytes();
  check(fixed <= baseline + 4096U, "fixed-size sequential transfers did not stabilize");

  // Only one transient tensor is live. Distinct capacity classes cumulatively
  // exceed the budget, so rounding cannot hide unbounded cache growth.
  for (std::size_t i = 1; i <= 32; ++i) {
    {
      auto tensor = transfer(i * mib + 4096U, dms, 17);
      verify(tensor, 17);
    }
    if (i % 8 == 0)
      std::cout << "sizes=" << i << " dma_bytes=" << dma_bytes() << std::endl;
  }
  for (unsigned char i = 0; i < held.size(); ++i)
    verify(held[i], i + 1);
  held.clear();
  // There are now no live tensors: the final snapshot must fit the cache budget.
  const auto final = dma_bytes();
  const auto growth = final > baseline ? final - baseline : 0U;
  std::cout << "heap=" << (dms ? "DMS" : "CMA") << " baseline_bytes=" << baseline
            << " fixed_bytes=" << fixed << " final_bytes=" << final
            << " idle_growth_bytes=" << growth << " limit_bytes=" << cache_byte_limit
            << " held_tensor_integrity=passed live_tensors=0" << std::endl;
  check(final <= initial + cache_byte_limit,
        "idle DMA retention exceeded the shared 256 MiB cache budget");

  // Six nearby sizes used to consume six exact-size pools and thrash the
  // byte budget. They share one capacity class; no heap allocation is allowed
  // after warming it, even when the requested logical length changes.
  {
    auto warm = transfer(46080000U, dms, 19);
    verify(warm, 19);
  }
  const auto warmed_allocations = heap_allocations;
  for (std::size_t i = 0; i < 60; ++i) {
    const auto bytes = 46080000U + (i % 6U) * 4096U;
    auto tensor = transfer(bytes, dms, static_cast<unsigned char>(i));
    verify(tensor, static_cast<unsigned char>(i));
    check(tensor.storage->size_bytes == bytes, "capacity slack leaked into tensor storage");
  }
  check(heap_allocations == warmed_allocations,
        "nearby transfer sizes still allocate after warming a capacity class");
  std::cout << "nearby_size_reuse=passed warm_heap_allocations=0" << std::endl;

  // Allocation-pressure recovery must actually drop idle allocations before
  // retrying. Keep an outstanding tensor across both cache flush and failure.
  auto retained = transfer(3U * mib + 17U, dms, 31);
  fail_heap_allocations = 1;
  injected_failures = 0;
  {
    auto recovered = transfer(7U * mib + 123U, dms, 43);
    verify(recovered, 43);
  }
  check(injected_failures == 1 && !capture_retry,
        "pressure probe did not exercise failed allocation and retry");
  check(retry_dma_bytes <= initial + 4U * mib,
        "allocation retry retained idle DMA cache allocations");
  verify(retained, 31);
  fail_heap_allocations = 2;
  injected_failures = 0;
  bool failed = false;
  try {
    auto rejected = transfer(9U * mib + 123U, dms, 47);
  } catch (const std::runtime_error&) {
    failed = true;
  }
  check(failed && injected_failures == 2,
        "persistent allocation failure was hidden or retried without a bound");
  capture_retry = false;
  verify(retained, 31);
  retained = {};
  std::cout << "allocation_pressure_retry=passed persistent_failure=passed" << std::endl;

  // Force more live small buffers than the entry bound, independent of size
  // bucketing. Releasing them must retain at most 128 actual allocations.
  std::vector<Tensor> small_burst;
  for (std::size_t i = 0; i < 256; ++i)
    small_burst.push_back(transfer(4096U, dms, static_cast<unsigned char>(i)));
  for (std::size_t i = 0; i < small_burst.size(); ++i)
    verify(small_burst[i], static_cast<unsigned char>(i));
  small_burst.clear();
  const auto small = dma_stats();
  std::cout << "small_idle_bytes=" << small.bytes << " small_idle_allocations=" << small.allocations
            << std::endl;
  check(small.allocations <= idle_allocation_limit,
        "idle DMA pool count exceeded 128 after a small-buffer burst");

  // Reuse and retire slots concurrently. Every worker keeps a tensor alive
  // while other workers can evict its pool; exceptions propagate to the test.
  std::vector<std::future<void>> workers;
  for (std::size_t worker = 0; worker < 4; ++worker) {
    workers.push_back(std::async(std::launch::async, [dms, worker] {
      const auto pattern = static_cast<unsigned char>(79U + worker);
      auto protected_tensor = transfer(4U * mib, dms, pattern);
      for (std::size_t i = 1; i <= 64; ++i) {
        auto tensor = transfer(mib / 4U + (worker * 64U + i) * 4096U, dms, pattern);
        verify(tensor, pattern);
      }
      verify(protected_tensor, pattern);
    }));
  }
  for (auto& worker : workers)
    worker.get();
  const auto concurrent = dma_stats();
  check(concurrent.bytes <= initial + cache_byte_limit &&
            concurrent.allocations <= idle_allocation_limit,
        "concurrent eviction exceeded the idle cache bounds");
  std::cout << "concurrent_eviction=passed" << std::endl;

  // Exceed the cache budget with live tensors: no waiting or overwrites, and
  // retired pools must free their allocation when the final tensor is released.
  std::vector<Tensor> burst;
  for (int i = 0; i < 72; ++i)
    burst.push_back(transfer(4U * mib, dms, static_cast<unsigned char>(i)));
  for (std::size_t i = 0; i < burst.size(); ++i)
    verify(burst[i], static_cast<unsigned char>(i));
  burst.clear();
  const auto after_burst = dma_stats();
  std::cout << "burst_idle_bytes=" << after_burst.bytes
            << " burst_idle_allocations=" << after_burst.allocations << std::endl;
  check(after_burst.bytes <= initial + cache_byte_limit &&
            after_burst.allocations <= idle_allocation_limit,
        "concurrency burst left an oversized idle DMA cache");
}
} // namespace

// Fault injection reaches the real DMA exporter; no fake buffers replace device
// allocations. The second allocation records memory after Core's cache drain.
extern "C" int ioctl(int fd, unsigned long request, ...) noexcept {
  va_list args;
  va_start(args, request);
  void* argument = va_arg(args, void*);
  va_end(args);
  if (request == DMA_HEAP_IOCTL_ALLOC) {
    ++heap_allocations;
    if (capture_retry) {
      try {
        retry_dma_bytes = dma_bytes();
      } catch (...) {
        // Keep the syscall seam nonthrowing; the accounting assertion must fail.
        retry_dma_bytes = static_cast<std::size_t>(-1);
      }
      capture_retry = false;
    }
    if (fail_heap_allocations > 0) {
      --fail_heap_allocations;
      ++injected_failures;
      capture_retry = true;
      errno = ENOMEM;
      return -1;
    }
  }
  using Ioctl = int (*)(int, unsigned long, ...);
  static const auto real_ioctl = reinterpret_cast<Ioctl>(dlsym(RTLD_NEXT, "ioctl"));
  return real_ioctl(fd, request, argument);
}

int main(int argc, char** argv) {
  try {
    check(argc == 2, "usage: tensor_transfer_memory_lifecycle_test cma|dms|mixed");
    const std::string mode(argv[1]);
    if (mode == "mixed") {
      const auto initial = dma_bytes();
      auto cma = transfer(mib, false, 61);
      auto dms = transfer(mib, true, 67);
      for (std::size_t i = 1; i <= 32; ++i) {
        auto tensor = transfer(i * mib + 4096U, i % 2 == 0, 71);
        verify(tensor, 71);
      }
      verify(cma, 61);
      verify(dms, 67);
      cma = {};
      dms = {};
      const auto idle = dma_stats();
      std::cout << "mixed_heap_idle_bytes=" << idle.bytes
                << " mixed_heap_idle_allocations=" << idle.allocations << std::endl;
      check(idle.bytes <= initial + cache_byte_limit,
            "CMA and DMS did not share the global cache byte budget");
    } else {
      check(mode == "cma" || mode == "dms", "unknown heap selection");
      run(mode == "dms");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

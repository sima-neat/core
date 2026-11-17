#include "pipeline/TensorCore.h"

#include "test_utils.h"

#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {
// These executable-local seams observe the actual linked Core and provider.
// They neither copy TensorTransfer.cpp nor substitute a DMA allocation. All
// faults are synchronous, thread-local and one-shot; no provider thread fails.
enum class ProbeKind { Storage, Admission, Copy };
struct Probe {
  ProbeKind kind;
  std::size_t bytes;
  unsigned fail_at = 1;
  unsigned allocations = 0;
  unsigned samples = 0;
  unsigned acquisitions = 0;
  unsigned activations = 0;
  unsigned deactivations = 0;
  bool fault_armed = false;
  bool fault_consumed = false;
  bool carrier_valid = true;
  GstBufferPool* founder = nullptr; // Observed identity, never an owning reference.
  GstBuffer* acquired = nullptr;
  struct stat acquired_backing {};
};
bool read_backing(GstBuffer* buffer, struct stat* identity) {
  if (!buffer || gst_buffer_n_memory(buffer) != 1U)
    return false;
  auto* memory = gst_buffer_peek_memory(buffer, 0U);
  return gst_is_dmabuf_memory(memory) && ::fstat(gst_dmabuf_memory_get_fd(memory), identity) == 0;
}

thread_local Probe* active_probe = nullptr;
// Callbacks must outlive a failing negative-control process's leaked sample.
std::atomic<unsigned> finalized_samples{0};
std::atomic<unsigned> finalized_pools{0};

void sample_finalized(gpointer, GstMiniObject*) {
  finalized_samples.fetch_add(1);
}
void pool_finalized(gpointer, GObject*) {
  finalized_pools.fetch_add(1);
}

template <typename Function> Function next_symbol(const char* name) {
  const auto function = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
  if (!function)
    std::abort();
  return function;
}

using SampleNew = GstSample* (*)(GstBuffer*, GstCaps*, const GstSegment*, GstStructure*);
using PoolAcquire = GstFlowReturn (*)(GstBufferPool*, GstBuffer**, GstBufferPoolAcquireParams*);
using PoolSetActive = gboolean (*)(GstBufferPool*, gboolean);
SampleNew real_sample_new() {
  static const auto function = next_symbol<SampleNew>("gst_sample_new");
  return function;
}
PoolAcquire real_pool_acquire() {
  static const auto function = next_symbol<PoolAcquire>("gst_buffer_pool_acquire_buffer");
  return function;
}
PoolSetActive real_pool_set_active() {
  static const auto function = next_symbol<PoolSetActive>("gst_buffer_pool_set_active");
  return function;
}

class ProbeScope {
public:
  explicit ProbeScope(Probe& probe) {
    require(active_probe == nullptr, "nested transfer fault probe");
    active_probe = &probe;
  }
  ~ProbeScope() {
    active_probe->fault_armed = false;
    active_probe = nullptr;
  }
  ProbeScope(const ProbeScope&) = delete;
  ProbeScope& operator=(const ProbeScope&) = delete;
};
} // namespace

void* operator new(std::size_t bytes) {
  if (auto* probe = active_probe; probe && probe->fault_armed) {
    if (++probe->allocations == probe->fail_at) {
      probe->fault_armed = false;
      probe->fault_consumed = true;
      throw std::bad_alloc();
    }
  }
  if (void* memory = std::malloc(bytes == 0 ? 1 : bytes))
    return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}

extern "C" GstSample* gst_sample_new(GstBuffer* buffer, GstCaps* caps, const GstSegment* segment,
                                     GstStructure* info) {
  auto* sample = real_sample_new()(buffer, caps, segment, info);
  auto* probe = active_probe;
  if (sample && probe && probe->kind != ProbeKind::Copy) {
    ++probe->samples;
    probe->carrier_valid = probe->carrier_valid && buffer && gst_buffer_n_memory(buffer) == 1U &&
                           gst_is_dmabuf_memory(gst_buffer_peek_memory(buffer, 0U)) &&
                           gst_buffer_get_size(buffer) == probe->bytes;
    gst_mini_object_weak_ref(GST_MINI_OBJECT(sample), sample_finalized, nullptr);
    // Arm only after GstSample creation and observation: the next allocations
    // belong to the real Core storage/holder/finalization path, not this seam.
    if (probe->kind == ProbeKind::Storage)
      probe->fault_armed = true;
  }
  return sample;
}

extern "C" gboolean gst_buffer_pool_set_active(GstBufferPool* pool, gboolean active) {
  const auto result = real_pool_set_active()(pool, active);
  if (auto* probe = active_probe; probe && probe->kind == ProbeKind::Admission && result) {
    if (active) {
      ++probe->activations;
      probe->founder = pool;
    } else if (pool == probe->founder) {
      ++probe->deactivations;
    }
  }
  return result;
}

extern "C" GstFlowReturn gst_buffer_pool_acquire_buffer(GstBufferPool* pool, GstBuffer** buffer,
                                                        GstBufferPoolAcquireParams* params) {
  const auto result = real_pool_acquire()(pool, buffer, params);
  auto* probe = active_probe;
  if (result == GST_FLOW_OK && buffer && *buffer && probe) {
    ++probe->acquisitions;
    probe->acquired = *buffer;
    probe->carrier_valid = probe->carrier_valid && read_backing(*buffer, &probe->acquired_backing);
    if (probe->kind == ProbeKind::Admission && pool == probe->founder) {
      probe->carrier_valid = probe->carrier_valid && gst_buffer_get_size(*buffer) >= probe->bytes;
      g_object_weak_ref(G_OBJECT(pool), pool_finalized, nullptr);
      // Fresh initialized founder acquired: the next C++ allocation is the
      // cache's list node. Do not fault pool creation or the native allocator.
      probe->fault_armed = true;
    }
  }
  return result;
}

namespace {
using namespace simaai::neat;

Tensor make_source(std::size_t bytes, unsigned char pattern) {
  Tensor source;
  source.storage = make_cpu_owned_storage(bytes);
  source.dtype = TensorDType::UInt8;
  source.shape = {static_cast<std::int64_t>(bytes)};
  source.strides_bytes = {1};
  source.device = {DeviceType::CPU, 0};
  source.read_only = false;
  auto write = source.map_write();
  require(write.data && write.size_bytes == bytes, "cannot map CPU source fixture");
  std::memset(write.data, pattern, bytes);
  return source;
}

Tensor transfer(const Tensor& source, bool dms) {
  // The strict MLA overload must not hide a DMS failure with a CMA fallback.
  return dms ? source.mla(true) : source.cvu();
}

GstBuffer* buffer_of(const Tensor& tensor) {
  require(tensor.storage && tensor.storage->kind == StorageKind::GstSample &&
              tensor.storage->holder,
          "transfer did not publish a GstSample-backed tensor");
  return gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get()));
}

struct stat backing_of(const Tensor& tensor) {
  struct stat identity {};
  require(read_backing(buffer_of(tensor), &identity), "cannot identify transfer DMA backing");
  return identity;
}

bool same_backing(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_size == right.st_size;
}

void verify(const Tensor& tensor, std::size_t bytes, unsigned char pattern) {
  require(tensor.storage && tensor.storage->size_bytes == bytes,
          "transfer exposed capacity instead of logical bytes");
  auto read = tensor.map_read();
  require(read.data && read.size_bytes == bytes, "transfer read span does not match logical bytes");
  const auto* data = static_cast<const unsigned char*>(read.data);
  for (std::size_t index = 0; index < bytes; ++index)
    if (data[index] != pattern)
      throw std::runtime_error("transfer changed payload or a retained allocation");
}

void verify_device(const Tensor& tensor, bool dms) {
  require(tensor.device.type == (dms ? DeviceType::SIMA_MLA : DeviceType::SIMA_CVU),
          "transfer selected a different heap/device route");
}

void check_storage_exceptions(bool dms) {
  constexpr std::size_t bytes = 4099U;
  const auto source = make_source(bytes, 71);
  const auto retained = transfer(make_source(bytes, 29), dms);
  { auto warm = transfer(source, dms); }
  unsigned failures = 0;
  for (unsigned ordinal = 1; ordinal <= 32; ++ordinal) {
    Probe probe{ProbeKind::Storage, bytes, ordinal};
    const auto before = finalized_samples.load();
    Tensor output;
    std::exception_ptr error;
    {
      ProbeScope scope(probe);
      try {
        output = transfer(source, dms);
      } catch (...) {
        error = std::current_exception();
      }
    }
    require(probe.samples == 1 && probe.carrier_valid,
            "sample fault seam did not observe the actual Core DMA transfer");
    if (probe.fault_consumed) {
      require(error != nullptr && !output.storage, "storage allocation fault was not propagated");
      bool bad_alloc = false;
      try {
        std::rethrow_exception(error);
      } catch (const std::bad_alloc&) {
        bad_alloc = true;
      }
      require(bad_alloc, "storage fault changed its exception type");
      require(finalized_samples.load() == before + 1U,
              "GstSample leaked after storage/finalization allocation failure");
      ++failures;
    } else {
      if (error)
        std::rethrow_exception(error);
      verify(output, bytes, 71);
      verify_device(output, dms);
      output = {};
      require(finalized_samples.load() == before + 1U, "successful transfer retained its sample");
    }
    verify(source, bytes, 71);
    verify(retained, bytes, 29);
    {
      auto recovered = transfer(source, dms);
      verify(recovered, bytes, 71);
      verify_device(recovered, dms);
    }
    if (!probe.fault_consumed) {
      require(failures >= 2, "fault sweep did not cover holder and Storage construction");
      std::cout << "storage heap=" << (dms ? "DMS" : "CMA") << " injected=" << failures
                << " sample_finalization=passed recovery=passed\n";
      return;
    }
  }
  throw std::runtime_error("storage allocation sweep did not reach its unfaulted success control");
}

void check_copy_exceptions(bool dms) {
  constexpr std::size_t bytes = 4099U;
  const auto source = make_source(bytes, 83);
  { auto warm = transfer(source, dms); }
  for (const bool throws : {true, false}) {
    Tensor faulty = source;
    faulty.storage = std::make_shared<Storage>(*source.storage);
    unsigned calls = 0;
    faulty.storage->map_fn = [&](MapMode) -> Mapping {
      ++calls;
      if (throws)
        throw std::runtime_error("intentional source mapping exception");
      return {};
    };
    Probe probe{ProbeKind::Copy, bytes};
    bool rejected = false;
    {
      ProbeScope scope(probe);
      try {
        auto output = transfer(faulty, dms);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
    }
    require(rejected && calls == 1 && probe.acquisitions == 1 && probe.acquired &&
                probe.carrier_valid,
            "copy failure did not occur after acquiring the real destination loan");
    auto recovered = transfer(source, dms);
    require(same_backing(backing_of(recovered), probe.acquired_backing),
            "copy failure leaked its destination instead of returning the warmed pool loan");
    verify(recovered, bytes, 83);
    verify_device(recovered, dms);
  }
  std::cout << "copy heap=" << (dms ? "DMS" : "CMA") << " throw_and_false=passed recovery=passed\n";
}

void check_admission_failure(bool dms) {
  // Not a class touched by the small storage/copy cases: this must acquire a
  // freshly initialized founder, not an existing cached buffer.
  constexpr std::size_t bytes = 1024U * 1024U + 17U;
  const auto source = make_source(bytes, 97);
  const auto retained = transfer(make_source(4099U, 31), dms);
  Probe probe{ProbeKind::Admission, bytes};
  const auto pools_before = finalized_pools.load();
  const auto samples_before = finalized_samples.load();
  Tensor output;
  std::exception_ptr error;
  {
    ProbeScope scope(probe);
    try {
      output = transfer(source, dms);
    } catch (...) {
      error = std::current_exception();
    }
  }
  require(probe.activations == 1 && probe.acquisitions == 1 && probe.fault_consumed &&
              probe.allocations == 1 && probe.carrier_valid,
          "admission seam did not fail the fresh founder's first C++ node allocation");
  std::cout << "admission heap=" << (dms ? "DMS" : "CMA") << " founder_node_fault=observed\n";
  if (error)
    std::rethrow_exception(error);
  require(probe.deactivations == 1 && finalized_pools.load() == pools_before,
          "uncached founder was not deactivated while preserving its outstanding loan");
  require(probe.samples == 1, "transient admission fallback did not publish a sample");
  verify(output, bytes, 97);
  verify_device(output, dms);
  verify(retained, 4099U, 31);
  output = {};
  require(finalized_samples.load() == samples_before + 1U &&
              finalized_pools.load() == pools_before + 1U,
          "transient transfer did not release its sample and deactivated founder");
  struct stat reusable {};
  {
    auto normal = transfer(source, dms);
    verify(normal, bytes, 97);
    reusable = backing_of(normal);
  }
  auto next = transfer(source, dms);
  require(same_backing(backing_of(next), reusable),
          "admission failure disabled subsequent ordinary pool reuse");
  verify(next, bytes, 97);
  std::cout << "admission heap=" << (dms ? "DMS" : "CMA")
            << " node_bad_alloc=passed transient_release=passed later_reuse=passed\n";
}
} // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc == 1 ? "all" : argv[1];
    require(argc <= 2 &&
                (mode == "all" || mode == "storage" || mode == "admission" || mode == "copy"),
            "usage: unit_tensor_transfer_exception_test [all|storage|admission|copy]");
    // Resolve before arming faults; missing interposition is a strict failure,
    // not a skipped or fake-provider success.
    (void)real_sample_new();
    (void)real_pool_acquire();
    (void)real_pool_set_active();
    for (const bool dms : {false, true}) {
      if (mode == "all" || mode == "storage")
        check_storage_exceptions(dms);
      if (mode == "all" || mode == "copy")
        check_copy_exceptions(dms);
      if (mode == "all" || mode == "admission")
        check_admission_failure(dms);
    }
    std::cout << "[OK] tensor transfer exception ownership passed\n";
    return 0;
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
}

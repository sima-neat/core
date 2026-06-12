// SPDX-License-Identifier: MIT
// SiMa Neat — Tier-A CPU<->device cache-coherency state machine (testable core).
//
// This is the canonical Tier-A component. In production it lives inline in the
// allocator's per-allocation GstSimaaiSegmentMemory struct
// (internals/core/allocator/gstsimaaisegmentallocator.cpp) and drives the real
// dc civac/cvac CMOs in internals/sima-ai-simaai-memory/simaai_memory.c. Here it
// is dependency-light (only <atomic>/<mutex>) and routes every CMO through a
// pluggable CmoBackend so host tests can inject a counting mock and the devkit
// can inject the real simaai backend.
//
// Design invariants (each enforced by Phase-0 tests):
//   - State binds to the ALLOCATION (this struct), never to a recyclable
//     GstBuffer/GstSample wrapper.  [P0-L2-14 recycling regression]
//   - Conservative default: a never-seen allocation is treated as
//     device-written/dirty, so the first CPU read always invalidates.  [P0-L0-5]
//   - Ownership-skip: no CMO when the CPU was the last writer.            [P0-L1-6]
//   - Reader invalidate is ALWAYS whole-segment (dc civac is clean+invalidate;
//     a partial invalidate can clean a neighbour's bytes over fresh device
//     data at a shared 64B line).                                         [P0-L5-27]
//   - Partial CMO is allowed ONLY on the CPU-write clean path and ONLY when the
//     [off,len) span is 64B-aligned (no shared boundary line); otherwise it
//     falls back to whole-segment.                                        [P0-L3-19]
//   - Nested begin/end pairing: only the first begin invalidates, only the last
//     end flushes (dma-buf begin/end_cpu_access semantics).               [P0-L1-8]
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

namespace simaai::neat::coherency {

constexpr uint64_t kCacheLine = 64;

enum class Access : uint8_t { Read = 1, Write = 2, ReadWrite = 3 };
enum class Writer : uint8_t { Unknown = 0, Cpu = 1, Device = 2 };

inline bool wants_write(Access a) {
  return (static_cast<uint8_t>(a) & static_cast<uint8_t>(Access::Write)) != 0;
}
inline bool wants_read(Access a) {
  return (static_cast<uint8_t>(a) & static_cast<uint8_t>(Access::Read)) != 0;
}
inline bool aligned64(uint64_t off, uint64_t len) {
  return (off % kCacheLine) == 0 && (len % kCacheLine) == 0;
}

// Cache-maintenance backend. len==0 means "whole segment".
//   invalidate() == dc civac + dsb sy   (clean+invalidate, before a CPU read)
//   clean()      == dc cvac  + dsb st   (clean,            before a device read)
struct CmoBackend {
  virtual ~CmoBackend() = default;
  virtual void invalidate(void* seg, uint64_t off, uint64_t len) = 0;
  virtual void clean(void* seg, uint64_t off, uint64_t len) = 0;
};

// Per-allocation coherency state. One instance per physical segment; lifetime ==
// the allocation (constructed/destroyed with the GstSimaaiSegmentMemory). Non-
// copyable (holds atomics + a mutex). In production the mutex IS the allocation's
// existing map_mutex; here it is owned for isolation.
struct SegCoherency {
  void* seg = nullptr; // opaque allocation identity passed to the backend
  uint64_t size = 0;   // bytes; used for whole-segment ops

  std::atomic<uint8_t> last_writer{static_cast<uint8_t>(Writer::Unknown)};
  std::atomic<bool> cpu_cache_clean{false}; // conservative default: NOT clean
  std::atomic<uint32_t> cpu_inflight{0};    // nested begin/end refcount
  std::atomic<bool> uncached{false};        // non-cacheable buffer => CMOs are no-ops
  std::mutex lock;                          // taken only on a real transition

  SegCoherency() = default;
  SegCoherency(void* s, uint64_t n) : seg(s), size(n) {}
  SegCoherency(const SegCoherency&) = delete;
  SegCoherency& operator=(const SegCoherency&) = delete;
};

// Ownership-transfer API (the dma-buf begin/end_cpu_access analog). `be` may be
// null for the attach-fallback path, in which case the CPU side is conservative
// (handled by the caller); here a null backend simply skips the CMO.
void begin_cpu_access(SegCoherency& c, Access a, CmoBackend* be);
void end_cpu_access(SegCoherency& c, Access a, CmoBackend* be, uint64_t off = 0, uint64_t len = 0);
void end_device_access(SegCoherency& c, uint64_t off = 0, uint64_t len = 0);

} // namespace simaai::neat::coherency

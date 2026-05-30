// SPDX-License-Identifier: MIT
// Phase-0 hardware tests: Levels 2 (functional coherency) and 4 (concurrency),
// run on the Modalix A65 against REAL cacheable DDR and REAL dc civac/cvac.
//
// The "device" (a non-coherent producer/consumer that bypasses the A65 cache) is
// the SiMa allocator's own DMA copy: simaai_memcpy() drives SIMAAI_IOC_MEMCPY in
// the kernel, moving data by PHYSICAL address. So a CPU store left dirty in cache
// is invisible to it unless flushed, and a DMA write to DDR is invisible to the
// CPU unless invalidated — exactly the EV74/MLA hazard, with no /dev/mem and no
// root (group `sima` owns /dev/simaai-mem).
//
// Build (cross, on host):
//   aarch64-linux-gnu-g++ -std=c++17 -O2 -pthread segment_coherency.cpp \
//       coherency_devkit_suite.cpp -ldl -o devkit_suite
// Run on the board via devkit-run (NFS /workspace).
#include "segment_coherency.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace simaai::neat::coherency;

// Fuzzer op count, settable via argv[1] (devkit-run forwards args but not env).
uint64_t g_fuzz_ops = 0;

namespace {

constexpr int kSkip = 77;
constexpr int kTargetGeneric = 0;  // SIMAAI_MEM_TARGET_GENERIC (DDR)
constexpr int kFlagCached = 1;     // SIMAAI_MEM_FLAG_CACHED

struct simaai_memory;
using simaai_memory_t = simaai_memory;

struct Lib {
  simaai_memory_t* (*alloc_flags)(unsigned, int, int) = nullptr;
  void* (*map)(simaai_memory_t*) = nullptr;
  uint64_t (*get_phys)(simaai_memory_t*) = nullptr;
  size_t (*get_size)(simaai_memory_t*) = nullptr;
  void (*invalidate)(simaai_memory_t*) = nullptr;
  void (*invalidate_part)(simaai_memory_t*, unsigned, unsigned) = nullptr;
  void (*flush)(simaai_memory_t*) = nullptr;
  void (*flush_part)(simaai_memory_t*, unsigned, unsigned) = nullptr;
  void (*freev)(simaai_memory_t*) = nullptr;
  simaai_memory_t* (*memcpy_part)(simaai_memory_t*, uint64_t, simaai_memory_t*, uint64_t, uint64_t) = nullptr;
  bool ok() const { return alloc_flags && map && invalidate && flush && memcpy_part && freev; }
};

Lib load() {
  Lib l;
  void* h = dlopen("libsimaaimem.so.2", RTLD_LAZY | RTLD_LOCAL);
  if (!h) h = dlopen("libsimaaimem.so", RTLD_LAZY | RTLD_LOCAL);
  if (!h) return l;
  auto S = [&](const char* n) { return dlsym(h, n); };
  l.alloc_flags = reinterpret_cast<decltype(l.alloc_flags)>(S("simaai_memory_alloc_flags"));
  l.map = reinterpret_cast<decltype(l.map)>(S("simaai_memory_map"));
  l.get_phys = reinterpret_cast<decltype(l.get_phys)>(S("simaai_memory_get_phys"));
  l.get_size = reinterpret_cast<decltype(l.get_size)>(S("simaai_memory_get_size"));
  l.invalidate = reinterpret_cast<decltype(l.invalidate)>(S("simaai_memory_invalidate_cache"));
  l.invalidate_part = reinterpret_cast<decltype(l.invalidate_part)>(S("simaai_memory_invalidate_cache_part"));
  l.flush = reinterpret_cast<decltype(l.flush)>(S("simaai_memory_flush_cache"));
  l.flush_part = reinterpret_cast<decltype(l.flush_part)>(S("simaai_memory_flush_cache_part"));
  l.freev = reinterpret_cast<decltype(l.freev)>(S("simaai_memory_free"));
  l.memcpy_part = reinterpret_cast<decltype(l.memcpy_part)>(S("simaai_memcpy_part"));
  return l;
}

// Real production CMO backend: whole-segment civac/cvac on a handle.
struct SimaaiCmo : CmoBackend {
  Lib* lib; simaai_memory_t* mem;
  SimaaiCmo(Lib* l, simaai_memory_t* m) : lib(l), mem(m) {}
  void invalidate(void*, uint64_t, uint64_t) override { lib->invalidate(mem); }
  void clean(void*, uint64_t, uint64_t) override { lib->flush(mem); }
};

struct Buf {
  Lib* lib = nullptr; simaai_memory_t* mem = nullptr; volatile uint8_t* cpu = nullptr; uint64_t size = 0;
  bool alloc(Lib* l, uint64_t n) {
    lib = l; size = n;
    mem = lib->alloc_flags((unsigned)n, kTargetGeneric, kFlagCached);
    if (!mem) return false;
    cpu = static_cast<volatile uint8_t*>(lib->map(mem));
    return cpu != nullptr;
  }
  ~Buf() { if (mem && lib) lib->freev(mem); }
};

struct TF : std::runtime_error { explicit TF(const std::string& m) : std::runtime_error(m) {} };
inline void check(bool c, const std::string& m) { if (!c) throw TF(m); }

// Test rig: T = buffer under test, plus S (source) and V (verify) staging buffers
// used to drive the DMA "device".
struct Rig {
  Lib* lib; Buf T, S, V; SegCoherency coh; SimaaiCmo be;
  Rig(Lib* l, uint64_t n) : lib(l), coh(nullptr, n), be(l, nullptr) {
    if (!T.alloc(l, n) || !S.alloc(l, n) || !V.alloc(l, n)) throw std::runtime_error("skip");
    coh.seg = T.mem; coh.size = n; be.mem = T.mem;
  }
  // Device produces into T[off,len): stage in S (flushed), DMA S->T.
  void dev_write(uint64_t off, const std::vector<uint8_t>& bytes) {
    for (uint64_t i = 0; i < bytes.size(); ++i) S.cpu[off + i] = bytes[i];
    lib->flush(S.mem);  // S's dirty lines -> DDR so the DMA reads the new bytes
    lib->memcpy_part(T.mem, off, S.mem, off, bytes.size());  // DMA: DDR(S)->DDR(T)
    end_device_access(coh, off, bytes.size());
  }
  // Device consumes T[off,len): DMA T->V, invalidate V, read what the device saw.
  std::vector<uint8_t> dev_read(uint64_t off, uint64_t len) {
    lib->memcpy_part(V.mem, off, T.mem, off, len);
    lib->invalidate(V.mem);
    std::vector<uint8_t> out(len);
    for (uint64_t i = 0; i < len; ++i) out[i] = V.cpu[off + i];
    return out;
  }
  // CPU consumes T through the component and checks it byte-for-byte.
  void cpu_read_T(uint64_t off, uint64_t len, const std::vector<uint8_t>& expect, const char* where) {
    begin_cpu_access(coh, Access::Read, &be);
    int bad = -1; uint8_t g = 0;
    for (uint64_t i = 0; i < len; ++i) {
      if (T.cpu[off + i] != expect[i]) { bad = (int)i; g = T.cpu[off + i]; break; }
    }
    end_cpu_access(coh, Access::Read, &be);
    if (bad >= 0)
      throw std::runtime_error(std::string(where) + ": stale CPU read at byte " +
                               std::to_string(off + bad) + " got " + std::to_string(g) +
                               " want " + std::to_string(expect[bad]));
  }
};

struct Case { const char* id; std::function<void(Lib&)> fn; };
std::vector<Case>& reg() { static std::vector<Case> r; return r; }
struct Reg { Reg(const char* id, std::function<void(Lib&)> f) { reg().push_back({id, std::move(f)}); } };
#define HWTEST(id) static void id(Lib&); static Reg r_##id(#id, id); static void id(Lib& lib)

std::vector<uint8_t> pat(uint64_t n, uint8_t s) {
  std::vector<uint8_t> v(n);
  for (uint64_t i = 0; i < n; ++i) v[i] = uint8_t(s + i * 31 + (i >> 3));
  return v;
}

// P0-L2-11: Device->CPU stale-read hazard.
HWTEST(P0_L2_11_device_to_cpu_stale_read) {
  Rig r(&lib, 4096);
  r.dev_write(0, pat(256, 1));                          // device produces OLD
  r.cpu_read_T(0, 256, pat(256, 1), "prime");          // CPU caches OLD
  r.dev_write(0, pat(256, 9));                          // device overwrites DDR with NEW
  r.cpu_read_T(0, 256, pat(256, 9), "stale-read");      // must invalidate -> see NEW
}

// P0-L2-12: CPU->Device lost-write hazard.
HWTEST(P0_L2_12_cpu_to_device_lost_write) {
  Rig r(&lib, 4096);
  auto v = pat(256, 5);
  begin_cpu_access(r.coh, Access::Write, &r.be);
  for (uint64_t i = 0; i < 256; ++i) r.T.cpu[i] = v[i];
  end_cpu_access(r.coh, Access::Write, &r.be, 0, 256);  // flush
  auto got = r.dev_read(0, 256);
  for (uint64_t i = 0; i < 256; ++i) check(got[i] == v[i], "lost write: device missed CPU store");
}

// P0-L2-13: round-trip ping-pong with a device-side transform, byte-exact.
HWTEST(P0_L2_13_round_trip) {
  Rig r(&lib, 4096);
  for (int it = 0; it < 500; ++it) {
    auto v = pat(256, uint8_t(it));
    begin_cpu_access(r.coh, Access::Write, &r.be);
    for (uint64_t i = 0; i < 256; ++i) r.T.cpu[i] = v[i];
    end_cpu_access(r.coh, Access::Write, &r.be, 0, 256);     // CPU produces -> flush
    lib.memcpy_part(r.V.mem, 0, r.T.mem, 0, 256);            // device reads T -> V
    lib.invalidate(r.V.mem);
    for (uint64_t i = 0; i < 256; ++i) r.V.cpu[i] = uint8_t(r.V.cpu[i] + 1);  // transform
    lib.flush(r.V.mem);
    lib.memcpy_part(r.T.mem, 0, r.V.mem, 0, 256);            // device writes T
    end_device_access(r.coh, 0, 256);
    r.cpu_read_T(0, 256, [&]{ auto e = v; for (auto& b : e) b = uint8_t(b + 1); return e; }(), "round-trip");
  }
}

// P0-L2-14: THE RECYCLING REGRESSION (the original bug). State on the allocation
// survives wrapper recycling; frame 2 must not leak frame-1 data.
HWTEST(P0_L2_14_recycling_regression) {
  Rig r(&lib, 4096);
  r.dev_write(0, pat(256, 1));
  r.cpu_read_T(0, 256, pat(256, 1), "frame1");   // CPU consumes, marks clean
  // pool recycles the wrapper over the SAME segment; device produces frame 2
  r.dev_write(0, pat(256, 42));
  r.cpu_read_T(0, 256, pat(256, 42), "frame2-after-recycle");  // must NOT see frame 1
}

// P0-L2-16: multi-region coverage. Prime each region with OLD (so the cache is
// dirty/stale), then device overwrites all regions; the read must see NEW.
HWTEST(P0_L2_16_multi_region) {
  Rig r(&lib, 4096);
  for (int q = 0; q < 4; ++q) r.dev_write(q * 256, pat(256, uint8_t(q * 7 + 1)));  // OLD
  begin_cpu_access(r.coh, Access::Read, &r.be);                                    // prime cache
  for (int q = 0; q < 4; ++q) { volatile uint8_t t = r.T.cpu[q * 256]; (void)t; }
  end_cpu_access(r.coh, Access::Read, &r.be);
  for (int q = 0; q < 4; ++q) r.dev_write(q * 256, pat(256, uint8_t(q * 11 + 3)));  // NEW
  begin_cpu_access(r.coh, Access::Read, &r.be);
  for (int q = 0; q < 4; ++q) {
    auto e = pat(256, uint8_t(q * 11 + 3));
    for (uint64_t i = 0; i < 256; ++i)
      check(r.T.cpu[q * 256 + i] == e[i], "multi-region stale at q=" + std::to_string(q));
  }
  end_cpu_access(r.coh, Access::Read, &r.be);
}

// P0-L4-23: concurrent readers see fresh data; one invalidate suffices. Prime the
// cache with OLD first so a missing invalidate would yield a stale read.
HWTEST(P0_L4_23_concurrent_readers) {
  Rig r(&lib, 4096);
  r.dev_write(0, pat(256, 11));                        // OLD
  begin_cpu_access(r.coh, Access::Read, &r.be);        // prime cache with OLD
  for (uint64_t i = 0; i < 256; ++i) { volatile uint8_t t = r.T.cpu[i]; (void)t; }
  end_cpu_access(r.coh, Access::Read, &r.be);
  auto v = pat(256, 77);                               // NEW
  r.dev_write(0, v);
  std::atomic<int> bad{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t) ts.emplace_back([&] {
    begin_cpu_access(r.coh, Access::Read, &r.be);
    for (uint64_t i = 0; i < 256; ++i) if (r.T.cpu[i] != v[i]) ++bad;
    end_cpu_access(r.coh, Access::Read, &r.be);
  });
  for (auto& t : ts) t.join();
  check(bad.load() == 0, "concurrent readers saw stale data");
}

// P0-L4-24: cross-thread producer->consumer handoff (scheduler-ordered, as in the
// real pipeline). The device produces on one thread and the CPU consumes on
// another, alternating via a turn flag. Each consumed frame must be byte-exact —
// staleness would mean the cross-thread invalidate failed. (An UNORDERED
// DMA-vs-read race is a scheduler concern, not the CMO's, so we order it.)
HWTEST(P0_L4_24_producer_consumer_handoff) {
  Rig r(&lib, 4096);
  std::atomic<int> turn{0};   // 0 = producer's turn, 1 = consumer's turn
  std::atomic<int> bad{0};
  const int kFrames = 3000;
  std::thread producer([&] {
    for (int i = 1; i <= kFrames; ++i) {
      while (turn.load(std::memory_order_acquire) != 0) std::this_thread::yield();
      r.dev_write(0, pat(256, uint8_t(i)));   // synchronous DMA + end_device_access
      turn.store(1, std::memory_order_release);
    }
  });
  for (int i = 1; i <= kFrames; ++i) {
    while (turn.load(std::memory_order_acquire) != 1) std::this_thread::yield();
    try { r.cpu_read_T(0, 256, pat(256, uint8_t(i)), "handoff"); }
    catch (...) { ++bad; }
    turn.store(0, std::memory_order_release);
  }
  producer.join();
  check(bad.load() == 0, "ordered cross-thread handoff produced a stale read");
}

// P0-L5-29: real-DDR fuzzer. Random interleavings of CPU/device read/write at
// random (deliberately sub-cache-line and occasionally 64B-aligned) offsets and
// lengths on a SMALL buffer to force adjacency, every read checked against a
// byte-exact oracle held in normal host memory. This validates the state machine
// against the ACTUAL A65 cache (not a model) and hammers the partial (_part) CMO
// path next to live neighbour data — the corruption-prone case. Op count via
// SIMA_FUZZ_OPS (default 200k; set 10000000 for the full soak).
//
// Invariant the oracle relies on: after every op, DDR == oracle for all bytes
// (device writes DDR directly; every CPU write is flushed by end_cpu_access). A
// missing invalidate => a CPU read sees stale cache != oracle; a missing flush =>
// a device read sees stale DDR != oracle. Either is caught.
HWTEST(P0_L5_29_real_ddr_fuzzer) {
  const uint64_t size = 64 * 8;  // 512B / 8 lines — small to maximise adjacency
  Rig r(&lib, size);
  std::vector<uint8_t> oracle(size, 0);
  { auto v = pat(size, 0); r.dev_write(0, v); for (uint64_t i = 0; i < size; ++i) oracle[i] = v[i]; }

  uint64_t ops = g_fuzz_ops ? g_fuzz_ops : 200000;  // override via argv[1]

  std::mt19937_64 rng(0xDEADBEEFCAFEULL);
  uint8_t seed = 1;
  for (uint64_t n = 0; n < ops; ++n) {
    uint64_t off = rng() % size;
    uint64_t len = 1 + rng() % std::min<uint64_t>(96, size - off);
    if (rng() % 4 == 0) {  // ~25% snapped to 64B alignment to exercise the partial-CMO path
      off = (off / 64) * 64;
      len = ((len + 63) / 64) * 64;
      if (off + len > size) len = size - off;
      if (len == 0) len = 64;
    }
    auto data = pat(len, seed++);
    switch (rng() % 4) {
      case 0:  // CPU write
        begin_cpu_access(r.coh, Access::Write, &r.be);
        for (uint64_t i = 0; i < len; ++i) r.T.cpu[off + i] = data[i];
        end_cpu_access(r.coh, Access::Write, &r.be, off, len);
        for (uint64_t i = 0; i < len; ++i) oracle[off + i] = data[i];
        break;
      case 1: {  // CPU read, checked
        begin_cpu_access(r.coh, Access::Read, &r.be);
        int bad = -1; uint8_t g = 0;
        for (uint64_t i = 0; i < len; ++i)
          if (r.T.cpu[off + i] != oracle[off + i]) { bad = (int)i; g = r.T.cpu[off + i]; break; }
        end_cpu_access(r.coh, Access::Read, &r.be);
        if (bad >= 0) throw TF("fuzz cpu-read stale @op " + std::to_string(n) + " byte " +
                               std::to_string(off + bad) + " got " + std::to_string(g) +
                               " want " + std::to_string(oracle[off + bad]));
        break;
      }
      case 2:  // device write
        r.dev_write(off, data);
        for (uint64_t i = 0; i < len; ++i) oracle[off + i] = data[i];
        break;
      case 3: {  // device read, checked
        auto got = r.dev_read(off, len);
        for (uint64_t i = 0; i < len; ++i)
          if (got[i] != oracle[off + i])
            throw TF("fuzz dev-read stale @op " + std::to_string(n) + " byte " +
                     std::to_string(off + i) + " got " + std::to_string(got[i]) +
                     " want " + std::to_string(oracle[off + i]));
        break;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_fuzz_ops = strtoull(argv[1], nullptr, 10);  // optional fuzzer op count
  Lib lib = load();
  if (!lib.ok()) { std::printf("[SKIP] libsimaaimem not usable\n"); return kSkip; }
  int failures = 0, skipped = 0;
  for (const auto& c : reg()) {
    try { c.fn(lib); std::printf("[OK]   %s\n", c.id); }
    catch (const TF& e) { std::printf("[FAIL] %s: %s\n", c.id, e.what()); ++failures; }
    catch (const std::exception& e) {
      if (std::string(e.what()) == "skip") { std::printf("[SKIP] %s (alloc failed)\n", c.id); ++skipped; }
      else { std::printf("[FAIL] %s: %s\n", c.id, e.what()); ++failures; }
    }
  }
  std::printf("\n%zu tests, %d failures, %d skipped\n", reg().size(), failures, skipped);
  if (failures) return 1;
  if (skipped == (int)reg().size()) return kSkip;
  return 0;
}

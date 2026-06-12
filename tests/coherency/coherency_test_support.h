// SPDX-License-Identifier: MIT
// Instrumentation for the Phase-0 coherency verification ladder.
//
// Provides a faithful model of the hardware hazard so host tests can detect
// stale reads / lost writes / neighbour corruption deterministically, plus a CMO
// counter for the optimality (golden-count) tests. The model:
//
//   - ddr[]   : actual DRAM contents. Device DMA reads/writes go straight here
//               (it bypasses the A65 cache, like the EV74 uDMA engine).
//   - truth[] : ground-truth value = what the most recent writer (CPU or device)
//               intended. A CPU read is CORRECT iff it returns truth[].
//   - cache   : per-64B-line {valid, dirty, data}, a write-back CPU cache.
//               CPU writes dirty a line; CPU reads return the cached line if
//               valid else fetch ddr. civac writes back dirty lines then drops
//               them; cvac writes back dirty lines and keeps them clean.
//
// Stale-read hazard: device writes ddr+truth; CPU holds a valid cached line; a
// CPU read with no invalidate returns the stale cached byte != truth.
// Lost-write hazard: CPU writes (dirty cache) but no clean; device reads ddr
// (old) != truth. Neighbour corruption: a partial clean over a shared 64B line
// writes a dirty neighbour byte to ddr, clobbering a device-fresh byte.
#pragma once

#include "segment_coherency.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat::coherency::test {

using simaai::neat::coherency::kCacheLine;

// ---- failure reporting -----------------------------------------------------
struct TestFailure : std::runtime_error {
  explicit TestFailure(const std::string& m) : std::runtime_error(m) {}
};
inline void check(bool cond, const std::string& msg) {
  if (!cond)
    throw TestFailure(msg);
}

// ---- DDR + CPU cache model -------------------------------------------------
struct CacheModel {
  std::vector<uint8_t> ddr;
  std::vector<uint8_t> truth;
  struct Line {
    bool valid = false;
    bool dirty = false;
    std::vector<uint8_t> data; // kCacheLine bytes
  };
  std::map<uint64_t, Line> lines; // keyed by line index

  explicit CacheModel(uint64_t size) : ddr(size, 0), truth(size, 0) {}
  uint64_t size() const {
    return ddr.size();
  }
  uint64_t line_of(uint64_t off) const {
    return off / kCacheLine;
  }

  Line& line(uint64_t idx) {
    auto& l = lines[idx];
    if (l.data.empty())
      l.data.assign(kCacheLine, 0);
    return l;
  }

  // CPU store: dirties the covering cache lines.
  void cpu_write(uint64_t off, const std::vector<uint8_t>& bytes) {
    for (uint64_t i = 0; i < bytes.size(); ++i) {
      const uint64_t a = off + i;
      Line& l = line(line_of(a));
      if (!l.valid) { // allocate-on-write: fill the rest of the line from ddr
        for (uint64_t b = 0; b < kCacheLine; ++b) {
          const uint64_t la = (a / kCacheLine) * kCacheLine + b;
          l.data[b] = la < ddr.size() ? ddr[la] : 0;
        }
        l.valid = true;
      }
      l.data[a % kCacheLine] = bytes[i];
      l.dirty = true;
      truth[a] = bytes[i];
    }
  }

  // CPU load: read-allocate cache. Returns the cached line if valid, else
  // fetches the line from ddr and caches it clean (so a later stale-read hazard
  // is observable if a device write to ddr is not followed by an invalidate).
  std::vector<uint8_t> cpu_read(uint64_t off, uint64_t len) {
    std::vector<uint8_t> out(len);
    for (uint64_t i = 0; i < len; ++i) {
      const uint64_t a = off + i;
      Line& l = line(line_of(a));
      if (!l.valid) {
        for (uint64_t b = 0; b < kCacheLine; ++b) {
          const uint64_t la = (a / kCacheLine) * kCacheLine + b;
          l.data[b] = la < ddr.size() ? ddr[la] : 0;
        }
        l.valid = true;
        l.dirty = false;
      }
      out[i] = l.data[a % kCacheLine];
    }
    return out;
  }

  // Device DMA write: straight to ddr (and truth); bypasses cache.
  void device_write(uint64_t off, const std::vector<uint8_t>& bytes) {
    for (uint64_t i = 0; i < bytes.size(); ++i) {
      ddr[off + i] = bytes[i];
      truth[off + i] = bytes[i];
    }
  }
  // Device DMA read: straight from ddr.
  std::vector<uint8_t> device_read(uint64_t off, uint64_t len) {
    return std::vector<uint8_t>(ddr.begin() + off, ddr.begin() + off + len);
  }

  // ---- CMOs (what the backend drives) --------------------------------------
  void range_lines(uint64_t off, uint64_t len, uint64_t& lo, uint64_t& hi) const {
    if (len == 0) { // whole segment
      lo = 0;
      hi = (size() + kCacheLine - 1) / kCacheLine;
      return;
    }
    lo = off / kCacheLine;
    hi = (off + len + kCacheLine - 1) / kCacheLine;
  }
  // dc civac: clean (write back dirty) THEN invalidate (drop).
  uint64_t civac(uint64_t off, uint64_t len) {
    uint64_t lo, hi;
    range_lines(off, len, lo, hi);
    for (uint64_t idx = lo; idx < hi; ++idx) {
      auto it = lines.find(idx);
      if (it == lines.end() || !it->second.valid)
        continue;
      if (it->second.dirty)
        writeback(idx, it->second);
      it->second.valid = false;
      it->second.dirty = false;
    }
    return hi - lo;
  }
  // dc cvac: clean (write back dirty), keep the line valid+clean.
  uint64_t cvac(uint64_t off, uint64_t len) {
    uint64_t lo, hi;
    range_lines(off, len, lo, hi);
    for (uint64_t idx = lo; idx < hi; ++idx) {
      auto it = lines.find(idx);
      if (it == lines.end() || !it->second.valid || !it->second.dirty)
        continue;
      writeback(idx, it->second);
      it->second.dirty = false;
    }
    return hi - lo;
  }
  void writeback(uint64_t idx, Line& l) {
    for (uint64_t b = 0; b < kCacheLine; ++b) {
      const uint64_t a = idx * kCacheLine + b;
      if (a < ddr.size())
        ddr[a] = l.data[b];
    }
  }
};

// ---- counting CMO backend --------------------------------------------------
struct CountingCmo : simaai::neat::coherency::CmoBackend {
  CacheModel* cm;
  uint64_t civac_calls = 0, cvac_calls = 0;
  uint64_t civac_lines = 0, cvac_lines = 0;
  uint64_t last_off = 0, last_len = 0;
  bool last_was_whole = false;

  explicit CountingCmo(CacheModel* c) : cm(c) {}
  void invalidate(void* /*seg*/, uint64_t off, uint64_t len) override {
    ++civac_calls;
    civac_lines += cm->civac(off, len);
    last_off = off;
    last_len = len;
    last_was_whole = (len == 0);
  }
  void clean(void* /*seg*/, uint64_t off, uint64_t len) override {
    ++cvac_calls;
    cvac_lines += cm->cvac(off, len);
    last_off = off;
    last_len = len;
    last_was_whole = (len == 0);
  }
  void reset_counts() {
    civac_calls = cvac_calls = civac_lines = cvac_lines = 0;
  }
};

// ---- a fully-wired test segment -------------------------------------------
struct Seg {
  CacheModel cm;
  CountingCmo cmo;
  simaai::neat::coherency::SegCoherency coh;

  explicit Seg(uint64_t size, bool uncached = false)
      : cm(size), cmo(&cm), coh(reinterpret_cast<void*>(0x1000), size) {
    coh.uncached.store(uncached, std::memory_order_relaxed);
  }

  // CPU read transaction; returns bytes and checks them against truth.
  std::vector<uint8_t> cpu_read_checked(uint64_t off, uint64_t len, const char* where) {
    simaai::neat::coherency::begin_cpu_access(coh, Access::Read, &cmo);
    auto got = cm.cpu_read(off, len);
    simaai::neat::coherency::end_cpu_access(coh, Access::Read, &cmo);
    for (uint64_t i = 0; i < len; ++i) {
      if (got[i] != cm.truth[off + i]) {
        throw TestFailure(std::string(where) + ": stale CPU read at byte " +
                          std::to_string(off + i) + " got " + std::to_string(got[i]) + " want " +
                          std::to_string(cm.truth[off + i]));
      }
    }
    return got;
  }
  void cpu_write(uint64_t off, const std::vector<uint8_t>& bytes) {
    simaai::neat::coherency::begin_cpu_access(coh, Access::Write, &cmo);
    cm.cpu_write(off, bytes);
    simaai::neat::coherency::end_cpu_access(coh, Access::Write, &cmo, off, bytes.size());
  }
  // Device produces output: writes ddr then records authorship.
  void device_write(uint64_t off, const std::vector<uint8_t>& bytes) {
    cm.device_write(off, bytes);
    simaai::neat::coherency::end_device_access(coh, off, bytes.size());
  }
  // Device consumes input: must see CPU's writes (checks against truth).
  std::vector<uint8_t> device_read_checked(uint64_t off, uint64_t len, const char* where) {
    auto got = cm.device_read(off, len);
    for (uint64_t i = 0; i < len; ++i) {
      if (got[i] != cm.truth[off + i]) {
        throw TestFailure(std::string(where) + ": lost write, device read byte " +
                          std::to_string(off + i) + " got " + std::to_string(got[i]) + " want " +
                          std::to_string(cm.truth[off + i]));
      }
    }
    return got;
  }
};

inline std::vector<uint8_t> pattern(uint64_t len, uint8_t seed) {
  std::vector<uint8_t> v(len);
  for (uint64_t i = 0; i < len; ++i)
    v[i] = static_cast<uint8_t>(seed + i * 31 + (i >> 3));
  return v;
}

} // namespace simaai::neat::coherency::test

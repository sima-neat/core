// SPDX-License-Identifier: MIT
// Tier-A coherency state machine — see segment_coherency.h for invariants.
#include "segment_coherency.h"

namespace simaai::neat::coherency {

void begin_cpu_access(SegCoherency& c, Access a, CmoBackend* be) {
  // `a` is intentionally unused: both read and write entry must invalidate a
  // non-CPU-owned buffer. (A partial CPU write to a stale cached line would,
  // on the later clean, write the stale neighbour bytes over device-fresh data;
  // invalidating first drops the stale lines so allocate-on-write refills from
  // DDR.) A whole-buffer-write fast path could skip this, but coverage is not
  // cheaply known here, so we stay conservative for correctness.
  (void)a;
  if (c.uncached.load(std::memory_order_relaxed)) {
    // Non-cacheable buffer: nothing to maintain. (A genuine uncached/Normal-NC
    // mapping; CPU sees DDR directly.)
    c.cpu_inflight.fetch_add(1, std::memory_order_acquire);
    return;
  }

  // EVERY begin must ensure coherence — the refcount must NOT gate the
  // invalidate, or a second concurrent reader could skip it before the first
  // reader's invalidate completes. The refcount gates only the write-flush in
  // end_cpu_access. Nested same-thread siblings still invalidate once because
  // the first begin sets cpu_cache_clean and the rest hit the fast path.
  //
  // FAST PATH (no lock): already coherent for the CPU -> skip the CMO.
  if (c.cpu_cache_clean.load(std::memory_order_acquire) &&
      c.last_writer.load(std::memory_order_relaxed) != static_cast<uint8_t>(Writer::Device)) {
    c.cpu_inflight.fetch_add(1, std::memory_order_acquire);
    return;
  }

  // SLOW PATH: serialize decision+CMO+flip so concurrent first-touchers all end
  // up coherent and end_device_access cannot interleave between them.
  {
    std::lock_guard<std::mutex> g(c.lock);
    const auto w = static_cast<Writer>(c.last_writer.load(std::memory_order_relaxed));
    const bool need = (w == Writer::Device) ||
                      (w == Writer::Unknown && !c.cpu_cache_clean.load(std::memory_order_relaxed));
    if (need && be != nullptr) {
#ifndef SEGCOH_BREAK_INVALIDATE  // negative-control switch: proves the tests have teeth
      // ALWAYS whole-segment: dc civac is clean+invalidate; a partial invalidate
      // can write a neighbour's bytes over fresh device data at a shared line.
      be->invalidate(c.seg, 0, 0);
#endif
    }
    c.cpu_cache_clean.store(true, std::memory_order_release);
    c.last_writer.store(static_cast<uint8_t>(Writer::Cpu), std::memory_order_relaxed);
  }
  c.cpu_inflight.fetch_add(1, std::memory_order_acquire);
}

void end_cpu_access(SegCoherency& c, Access a, CmoBackend* be, uint64_t off, uint64_t len) {
  // Pair the refcount; only the last end acts.
  const uint32_t now = c.cpu_inflight.fetch_sub(1, std::memory_order_release) - 1;
  if (now != 0) {
    return;
  }
  if (!wants_write(a)) {
    return;  // read-only CPU access leaves nothing to flush
  }
  if (c.uncached.load(std::memory_order_relaxed)) {
    // Write-combine/uncached: no cache to clean, but a store barrier would be
    // issued here in production (dsb st). No CMO.
    c.last_writer.store(static_cast<uint8_t>(Writer::Cpu), std::memory_order_relaxed);
    return;
  }
  if (be != nullptr) {
    // Partial clean (dc cvac) is safe ONLY when the span shares no 64B line with
    // a neighbour. Require 64B alignment AND a sub-span strictly inside the
    // segment; otherwise fall back to whole-segment.
    const bool partial_ok =
        len != 0 && off + len <= c.size && aligned64(off, len) && (off != 0 || len != c.size);
#ifndef SEGCOH_BREAK_CLEAN  // negative-control switch for the lost-write hazard
    if (partial_ok) {
      be->clean(c.seg, off, len);
    } else {
      be->clean(c.seg, 0, 0);
    }
#else
    (void)partial_ok;
#endif
  }
  c.cpu_cache_clean.store(true, std::memory_order_release);
  c.last_writer.store(static_cast<uint8_t>(Writer::Cpu), std::memory_order_relaxed);
}

void end_device_access(SegCoherency& c, uint64_t /*off*/, uint64_t /*len*/) {
  // The device wrote (via uDMA, already drained before the completion signal the
  // host waits on). Record device authorship; force the next CPU read to
  // invalidate. No host-side CMO here — the host civac at the consumer's
  // begin_cpu_access is the complete maintenance.
  std::lock_guard<std::mutex> g(c.lock);
  c.last_writer.store(static_cast<uint8_t>(Writer::Device), std::memory_order_relaxed);
  c.cpu_cache_clean.store(false, std::memory_order_release);
}

}  // namespace simaai::neat::coherency

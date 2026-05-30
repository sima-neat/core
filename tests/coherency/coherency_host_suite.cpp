// SPDX-License-Identifier: MIT
// Phase-0 host-runnable coherency tests: Levels 0 (primitive), 1 (state machine),
// 3 (optimality / golden CMO counts), 5 (fuzzer + neighbour + barrier).
//
// These run anywhere (no GStreamer, no hardware) because the SegCoherency logic
// and the DDR/cache model are pure C++. The hazard-on-real-silicon tests
// (Levels 2/4/6) live in coherency_devkit_suite.cpp and need the Modalix board.
//
// Build (standalone):
//   g++ -std=c++17 -O2 -pthread segment_coherency.cpp coherency_host_suite.cpp -o host_suite
//   ./host_suite
#include "coherency_test_support.h"

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace simaai::neat::coherency;
using namespace simaai::neat::coherency::test;

namespace {

struct Case {
  const char* id;
  std::function<void()> fn;
};
std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
struct Reg {
  Reg(const char* id, std::function<void()> fn) { registry().push_back({id, std::move(fn)}); }
};
#define TEST(id) \
  static void id();                            \
  static Reg reg_##id(#id, id);                \
  static void id()

// Drive begin/end on a Seg without the checked-read wrapper (for state probing).
void begin_read(Seg& s) { begin_cpu_access(s.coh, Access::Read, &s.cmo); }
void end_read(Seg& s) { end_cpu_access(s.coh, Access::Read, &s.cmo); }

// =============================== LEVEL 0 ===================================

// P0-L0-1: CMO op selection is directional.
TEST(P0_L0_1_directional_op_selection) {
  {  // read after device write -> civac only
    Seg s(256);
    s.device_write(0, pattern(256, 1));
    s.cmo.reset_counts();
    s.cpu_read_checked(0, 256, "L0-1 read");
    check(s.cmo.civac_calls == 1, "read-after-device must issue exactly 1 civac");
    check(s.cmo.cvac_calls == 0, "read path must not clean");
  }
  {  // write to a CPU-OWNED buffer -> cvac only (no invalidate once owned)
    Seg s(256);
    s.cpu_write(0, pattern(256, 2));  // establish CPU ownership (this first write invalidates)
    s.cmo.reset_counts();
    s.cpu_write(0, pattern(256, 3));  // second write on an owned, clean buffer
    check(s.cmo.cvac_calls == 1, "write on owned buffer must issue exactly 1 cvac");
    check(s.cmo.civac_calls == 0, "write on owned, clean buffer must not invalidate");
  }
}

// P0-L0-2 / L0-3: 64B-aligned span uses partial; unaligned falls back to whole.
TEST(P0_L0_2_alignment_and_dispatch) {
  check(aligned64(0, 64) && aligned64(128, 64), "aligned64 true on 64B multiples");
  check(!aligned64(1, 64) && !aligned64(0, 30), "aligned64 false on non-64B");
  {  // aligned sub-range -> partial cvac (not whole)
    Seg s(256);
    s.cpu_write(64, pattern(64, 3));  // line 1 only
    check(!s.cmo.last_was_whole, "aligned sub-range clean must be partial");
    check(s.cmo.last_off == 64 && s.cmo.last_len == 64, "partial range must match the written span");
  }
  {  // unaligned sub-range -> whole-segment fallback
    Seg s(256);
    s.cpu_write(10, pattern(40, 4));  // unaligned
    check(s.cmo.last_was_whole, "unaligned clean must fall back to whole-segment");
  }
}

// P0-L0-4: API is the sole mutator — no CMO before any access on a fresh seg.
TEST(P0_L0_4_api_is_sole_mutator) {
  Seg s(128);
  check(s.cmo.civac_calls == 0 && s.cmo.cvac_calls == 0, "no CMO before any access");
  check(s.coh.last_writer.load() == (uint8_t)Writer::Unknown, "fresh seg is Unknown");
  check(s.coh.cpu_cache_clean.load() == false, "fresh seg is not clean");
}

// P0-L0-5: conservative default — first CPU read of a never-seen seg invalidates.
TEST(P0_L0_5_conservative_default) {
  Seg s(128);  // never written by anyone
  begin_read(s);
  end_read(s);
  check(s.cmo.civac_calls == 1, "first read of an Unknown seg MUST invalidate (conservative)");
}

// =============================== LEVEL 1 ===================================

// P0-L1-6: ownership-skip — CPU write then CPU read issues no invalidate.
TEST(P0_L1_6_ownership_skip) {
  Seg s(256);
  s.cpu_write(0, pattern(256, 5));
  s.cmo.reset_counts();
  s.cpu_read_checked(0, 256, "L1-6");
  check(s.cmo.civac_calls == 0, "CPU-owned clean buffer must not be invalidated on read");
}

// P0-L1-7: device write flips dirty — next read invalidates exactly once.
TEST(P0_L1_7_device_write_flips_dirty) {
  Seg s(256);
  s.cpu_write(0, pattern(256, 6));   // CPU owns, clean
  s.device_write(0, pattern(256, 7));  // device takes over
  s.cmo.reset_counts();
  s.cpu_read_checked(0, 256, "L1-7 first");
  check(s.cmo.civac_calls == 1, "first read after device write must invalidate");
  s.cmo.reset_counts();
  s.cpu_read_checked(0, 256, "L1-7 second");
  check(s.cmo.civac_calls == 0, "second read with no device write must NOT invalidate");
}

// P0-L1-8: nested begin/end pairing — only first invalidates, only last flushes.
TEST(P0_L1_8_nested_pairing) {
  Seg s(256);
  s.device_write(0, pattern(256, 8));
  s.cmo.reset_counts();
  begin_cpu_access(s.coh, Access::Read, &s.cmo);  // sibling tensor A (first)
  begin_cpu_access(s.coh, Access::Read, &s.cmo);  // sibling tensor B (nested)
  check(s.cmo.civac_calls == 1, "nested begins must invalidate only once");
  end_cpu_access(s.coh, Access::Read, &s.cmo);
  end_cpu_access(s.coh, Access::Read, &s.cmo);
  check(s.coh.cpu_inflight.load() == 0, "refcount returns to zero");
}

// P0-L1-9: a write inside an access still flushes at the final end.
TEST(P0_L1_9_write_upgrade) {
  Seg s(256);
  s.device_write(0, pattern(256, 9));
  s.cmo.reset_counts();
  begin_cpu_access(s.coh, Access::ReadWrite, &s.cmo);
  s.cm.cpu_write(0, pattern(256, 10));
  end_cpu_access(s.coh, Access::ReadWrite, &s.cmo, 0, 256);
  check(s.cmo.civac_calls == 1, "RW access invalidates on entry");
  check(s.cmo.cvac_calls == 1, "RW access with a write flushes on exit");
}

// P0-L1-10: attach-fallback (no persistent state) is always conservative.
// Modelled by resetting state to Unknown/!clean before each read.
TEST(P0_L1_10_attach_fallback_conservative) {
  Seg s(128);
  for (int i = 0; i < 5; ++i) {
    s.coh.last_writer.store((uint8_t)Writer::Unknown);
    s.coh.cpu_cache_clean.store(false);
    s.cmo.reset_counts();
    begin_read(s);
    end_read(s);
    check(s.cmo.civac_calls == 1, "fallback path must invalidate on every read");
  }
}

// =============================== LEVEL 3 (optimality) ======================

// P0-L3-17: cold (uncached) buffer issues zero CMO and is still correct.
TEST(P0_L3_17_cold_buffer_zero_cmo) {
  Seg s(256, /*uncached=*/true);
  s.device_write(0, pattern(256, 11));
  s.cpu_read_checked(0, 256, "L3-17");  // reads ddr directly, correct
  check(s.cmo.civac_calls == 0 && s.cmo.cvac_calls == 0, "uncached buffer must issue zero CMO");
}

// P0-L3-18: no redundant invalidate across N reads with no device write.
TEST(P0_L3_18_no_redundant_invalidate) {
  Seg s(256);
  s.device_write(0, pattern(256, 12));
  s.cmo.reset_counts();
  for (int i = 0; i < 16; ++i) s.cpu_read_checked(0, 256, "L3-18");
  check(s.cmo.civac_calls == 1, "16 reads after one device write => exactly 1 invalidate, got " +
                                    std::to_string(s.cmo.civac_calls));
}

// P0-L3-19: partial clean covers only the touched lines; unaligned -> whole.
TEST(P0_L3_19_part_minimality) {
  const uint64_t size = 64 * 16;  // 16 lines
  {  // aligned 4-line write -> 4 lines cleaned, not 16
    Seg s(size);
    s.cpu_write(64 * 2, pattern(64 * 4, 13));
    check(s.cmo.cvac_lines == 4, "aligned clean must touch exactly 4 lines, got " +
                                     std::to_string(s.cmo.cvac_lines));
  }
  {  // unaligned write -> whole segment (16 lines)
    Seg s(size);
    s.cpu_write(70, pattern(100, 14));
    check(s.cmo.last_was_whole, "unaligned clean falls back to whole-segment");
    check(s.cmo.cvac_lines == 16, "whole-segment clean touches all 16 lines, got " +
                                      std::to_string(s.cmo.cvac_lines));
  }
}

// P0-L3-20: device->device handoff with no CPU touch issues zero CPU CMO.
TEST(P0_L3_20_device_to_device_elision) {
  Seg s(256);
  s.device_write(0, pattern(256, 15));            // EV74 output
  auto out = s.device_read_checked(0, 256, "L3-20");  // MLA input, straight from ddr
  (void)out;
  check(s.cmo.civac_calls == 0 && s.cmo.cvac_calls == 0,
        "device->device with no CPU map must issue zero CPU-side CMO");
}

// P0-L3-21: directional minimality (read-only=>civac only; write-only=>cvac only).
TEST(P0_L3_21_directional_minimality) {
  {
    Seg s(256);
    s.device_write(0, pattern(256, 16));
    s.cmo.reset_counts();
    begin_cpu_access(s.coh, Access::Read, &s.cmo);
    end_cpu_access(s.coh, Access::Read, &s.cmo);
    check(s.cmo.cvac_calls == 0, "read-only must not clean");
    check(s.cmo.civac_calls == 1, "read-only after device write invalidates once");
  }
  {
    Seg s(256);
    s.cpu_write(0, pattern(256, 17));  // establish CPU ownership first
    s.cmo.reset_counts();
    begin_cpu_access(s.coh, Access::Write, &s.cmo);
    s.cm.cpu_write(0, pattern(256, 18));
    end_cpu_access(s.coh, Access::Write, &s.cmo, 0, 256);
    check(s.cmo.civac_calls == 0, "write-only on an owned, clean buffer must not invalidate");
    check(s.cmo.cvac_calls == 1, "write-only cleans once");
  }
}

// P0-L3-22: golden CMO-count matrix (the optimality gate).
TEST(P0_L3_22_golden_cmo_matrix) {
  struct Row { const char* name; std::function<void(Seg&)> seq; uint64_t civac; uint64_t cvac; };
  const std::vector<Row> matrix = {
    {"dev_write,read", [](Seg& s){ s.device_write(0, pattern(256,1)); s.cmo.reset_counts();
                                   s.cpu_read_checked(0,256,"m"); }, 1, 0},
    {"cpu_write,read", [](Seg& s){ s.cpu_write(0, pattern(256,1)); s.cmo.reset_counts();
                                   s.cpu_read_checked(0,256,"m"); }, 0, 0},
    {"dev_write,3reads",[](Seg& s){ s.device_write(0,pattern(256,1)); s.cmo.reset_counts();
                                    for(int i=0;i<3;++i) s.cpu_read_checked(0,256,"m"); }, 1, 0},
    {"fresh_write",     [](Seg& s){ s.cmo.reset_counts(); s.cpu_write(0,pattern(256,1)); }, 1, 1},
    {"owned_write_only",[](Seg& s){ s.cpu_write(0,pattern(256,1)); s.cmo.reset_counts();
                                    s.cpu_write(0,pattern(256,2)); }, 0, 1},
    {"dev_to_dev",      [](Seg& s){ s.device_write(0,pattern(256,1)); s.cmo.reset_counts();
                                    s.device_read_checked(0,256,"m"); }, 0, 0},
  };
  for (const auto& r : matrix) {
    Seg s(256);
    r.seq(s);
    check(s.cmo.civac_calls == r.civac, std::string("golden civac mismatch for ") + r.name +
              ": got " + std::to_string(s.cmo.civac_calls) + " want " + std::to_string(r.civac));
    check(s.cmo.cvac_calls == r.cvac, std::string("golden cvac mismatch for ") + r.name +
              ": got " + std::to_string(s.cmo.cvac_calls) + " want " + std::to_string(r.cvac));
  }
}

// =============================== LEVEL 5 (adversarial) =====================

// P0-L5-26: randomized fuzzer vs oracle. Every CPU read is checked against truth.
TEST(P0_L5_26_fuzzer_vs_oracle) {
  std::mt19937 rng(0xC0FFEE);
  const uint64_t size = 64 * 8;
  Seg s(size);
  auto rnd = [&](uint64_t lo, uint64_t hi) { return lo + rng() % (hi - lo + 1); };
  const int kOps = 200000;
  for (int i = 0; i < kOps; ++i) {
    const uint64_t off = rnd(0, size - 1);
    const uint64_t len = std::min<uint64_t>(rnd(1, 96), size - off);
    switch (rng() % 5) {
      case 0: s.cpu_write(off, pattern(len, (uint8_t)i)); break;
      case 1: s.cpu_read_checked(off, len, "fuzz-cpu-read"); break;
      case 2: s.device_write(off, pattern(len, (uint8_t)(i * 7))); break;
      case 3: s.device_read_checked(off, len, "fuzz-dev-read"); break;
      case 4:  // device write then immediate CPU read of the same span (hazard)
        s.device_write(off, pattern(len, (uint8_t)(i * 13)));
        s.cpu_read_checked(off, len, "fuzz-hazard-read");
        break;
    }
  }
}

// P0-L5-27: neighbour isolation — an aligned partial clean of tensor A must not
// disturb tensor B in the adjacent line; documents the 64B-alignment invariant.
TEST(P0_L5_27_neighbour_isolation) {
  Seg s(128);  // A = line0 [0,64), B = line1 [64,128)
  // Device produces both tensors fresh.
  s.device_write(0, pattern(128, 20));
  // CPU reads B (caches line1, clean) then reads A.
  s.cpu_read_checked(64, 64, "L5-27 read B");
  s.cpu_read_checked(0, 64, "L5-27 read A");
  // CPU rewrites A only (aligned) -> partial clean of line0 only.
  s.cpu_write(0, pattern(64, 21));
  check(s.cmo.last_off == 0 && s.cmo.last_len == 64 && !s.cmo.last_was_whole,
        "A's clean must be the aligned line0 only");
  // B in DDR must still equal truth (untouched by A's clean).
  s.device_read_checked(64, 64, "L5-27 B intact after A clean");
  // And a subsequent CPU read of B must still be correct.
  s.cpu_read_checked(64, 64, "L5-27 reread B");
}

// P0-L5-28: barrier/tight-loop torture (host model has no reordering, so this
// passes trivially; on the devkit it catches a missing dsb after a CMO).
TEST(P0_L5_28_tight_loop_barrier) {
  Seg s(256);
  for (int i = 0; i < 5000; ++i) {
    s.device_write(0, pattern(256, (uint8_t)i));
    s.cpu_read_checked(0, 256, "L5-28");
  }
}

}  // namespace

int main() {
  int failures = 0;
  for (const auto& c : registry()) {
    try {
      c.fn();
      std::printf("[OK]   %s\n", c.id);
    } catch (const std::exception& e) {
      std::printf("[FAIL] %s: %s\n", c.id, e.what());
      ++failures;
    }
  }
  std::printf("\n%zu tests, %d failures\n", registry().size(), failures);
  return failures == 0 ? 0 : 1;
}

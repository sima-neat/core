# Phase 0 — Cache-coherency verification ladder (Tier A)

Incrementally-harder tests that prove the Tier-A coherency rework is **correct**,
**optimal** (minimum CMOs), and an **improvement** over the recyclable-wrapper
scheme — the "perfect achievable Tier-A state."

## Layout

| File | Role |
|---|---|
| `segment_coherency.{h,cpp}` | The Tier-A component under test: inline per-allocation `SegCoherency` state + `begin_cpu_access`/`end_cpu_access`/`end_device_access`, with a pluggable `CmoBackend`. Canonical home in production: inline in `GstSimaaiSegmentMemory` ([gstsimaaisegmentallocator.cpp](../../../internals/core/allocator/gstsimaaisegmentallocator.cpp)) driving the real `dc civac/cvac` in [simaai_memory.c](../../../internals/sima-ai-simaai-memory/simaai_memory.c). |
| `coherency_test_support.h` | DDR + read/write-allocate CPU cache + device-DMA model (the hazard oracle) and the counting `CmoBackend`. |
| `coherency_host_suite.cpp` | Levels 0,1,3,5 — pure logic, runs anywhere. |
| `coherency_devkit_suite.cpp` | Levels 2,4 (+L6 hook) — real cacheable DDR + real CMOs; the "device" is a genuinely-uncached second mapping of the same physical page. Devkit-only (skips 77 off-target). |

## Run

Host (logic/optimality/fuzzer) + negative control, anywhere:
```bash
cmake -S . -B build && cmake --build build
ctest --test-dir build --output-on-failure
```

On the Modalix DevKit (real cacheable DDR, real `dc civac/cvac`, DMA "device" via
`simaai_memcpy` — no root, no `/dev/mem` needed):
```bash
./run_devkit.sh          # cross-compiles aarch64, runs host + devkit + negative control via devkit-run
```

### On-silicon results (Modalix A65, validated)

```
host suite (A65) ............... 18 tests, 0 failures
devkit suite (A65) ............. 8 tests, 0 failures        # L2-11..16, L4-23/24, L5-29 fuzzer; ~3s
negative control (CMOs off) .... 8 tests, 8 failures        # every test has teeth; fuzzer trips @op 16
```

The real-DDR fuzzer (`P0-L5-29`) runs 200k random ops in seconds by default; pass
an op count as argv to go deeper (`devkit-run …/coherency_devkit_arm 2000000`,
~21s). It validates the state machine against the *actual* A65 cache (not a
model) and hammers the partial (`_part`) CMO path next to live neighbour data.

The negative control is the key proof: with `dc civac`/`cvac` disabled, the
stale-read fails with `got 1 want 9` on real hardware — confirming the
`FLAG_CACHED` buffers are genuinely Normal-WB cacheable (regime b) and the
software CMOs are load-bearing. If those tests *passed* with CMOs off, the buffer
wouldn't be cacheable and the whole Tier-A premise would be wrong.

## Negative control (teeth)

`coherency_negative_control` builds the host suite with `-DSEGCOH_BREAK_INVALIDATE`
(invalidate disabled) and asserts it **fails** (ctest `WILL_FAIL`). With the guard
off, 11 tests fail — including the stale-read hazard and the 200k-op fuzzer —
proving the green tests actually detect coherency violations rather than passing
vacuously.

## Test → plan map

| ID | Level | Env | Proves |
|---|---|---|---|
| P0-L0-1 | 0 primitive | host | directional op selection (read→civac, write→cvac) |
| P0-L0-2/3 | 0 | host | 64B outward alignment; partial-vs-whole dispatch |
| P0-L0-4 | 0 | host | API is the sole state mutator; no CMO before access |
| P0-L0-5 | 0 | host | **conservative default** — first read of Unknown invalidates |
| P0-L1-6 | 1 state | host | **ownership-skip** — CPU-owned read issues no invalidate |
| P0-L1-7 | 1 | host | device write flips dirty; next read invalidates once |
| P0-L1-8 | 1 | host | nested begin/end pairing (one invalidate, refcount) |
| P0-L1-9 | 1 | host | write inside RW access still flushes at final end |
| P0-L1-10 | 1 | host | attach-fallback is always conservative |
| P0-L3-17 | 3 optimality | host | cold/uncached buffer → **zero CMO**, still correct |
| P0-L3-18 | 3 | host | no redundant invalidate across N reads |
| P0-L3-19 | 3 | host | partial clean = ROI lines only; unaligned → whole |
| P0-L3-20 | 3 | host | device→device elision → zero CPU CMO |
| P0-L3-21 | 3 | host | directional minimality |
| P0-L3-22 | 3 | host | **golden CMO-count matrix** (the optimality gate) |
| P0-L5-26 | 5 adversarial | host | **fuzzer vs oracle**, 200k ops, every read checked |
| P0-L5-27 | 5 | host | neighbour isolation / 64B-alignment invariant |
| P0-L5-28 | 5 | host | tight-loop barrier (real teeth on devkit: missing dsb) |
| P0-L2-11 | 2 functional | devkit ✓ | **device→CPU stale-read hazard** (real cache) |
| P0-L2-12 | 2 | devkit | **CPU→device lost-write hazard** |
| P0-L2-13 | 2 | devkit | round-trip ping-pong, byte-exact |
| P0-L2-14 | 2 | devkit | **recycling regression** (the original bug) |
| P0-L2-16 | 2 | devkit | multi-region coverage |
| P0-L4-23 | 4 concurrency | devkit ✓ | concurrent readers, single invalidate |
| P0-L4-24 | 4 | devkit ✓ | ordered cross-thread producer→consumer handoff |
| P0-L5-29 | 5 adversarial | devkit ✓ | **real-DDR fuzzer vs oracle** (actual A65 cache, partial-CMO path) |

## Level 6 (end-to-end) — wiring, not re-implemented

L6 (byte-exact model inference, differential A/B vs baseline, perf
non-regression) runs through the existing e2e harness, not this standalone
project. The hooks:

- **P0-L6-29 byte-exact:** run the validated model matrix with the new layer; diff
  outputs against the committed golden tensors (existing e2e compare path).
- **P0-L6-30 differential A/B:** the `CountingCmo` totals are the improvement
  metric — assert new-scheme CMO count ≤ baseline **and** identical outputs. Set
  `SIMA_SESSION_SYNC_CACHE_DEBUG=1` to surface per-frame counts
  ([EnvUtil.h:284](../../src/pipeline/internal/EnvUtil.h)).
- **P0-L6-31 perf:** µs/frame within noise vs baseline; use the discard-run1 /
  read-run2-6 devkit methodology.

## Exit criteria ("perfect achievable Tier A")

All host tests green, negative control fails, fuzzer ≥10⁷ ops clean on the
devkit, golden CMO matrix exact, devkit hazard + recycling tests green, e2e
byte-exact with CMO count ≤ baseline.

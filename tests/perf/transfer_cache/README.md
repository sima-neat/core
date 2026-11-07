# DMA transfer-cache performance comparison

The initial exact-size version of PR #1322 (`27bb2b15`) on
`modalix-065848` did not show a material regression on the measured
warm fixed-size workloads. It **did** make a cyclic working set just above the
256 MiB cache budget about **3.85 times slower**. The memory bound trades reuse
for fresh allocation when the working set cannot fit; it is not universally
performance-neutral.

These are private-prefix Core DSO comparisons, not Debian installation or
complete application qualification. No global device packages were changed.

## Compared implementations

- **PR #1274:** final head `f66ffe3887e860b0b7e1fc1d4e5b2cedeccc0d3a`.
- **PR #1322:** runtime at `27bb2b157dc139a66c33ef57665dfa2953d691d6`.
- **Cache-only control:** PR #1322's other code with only `TensorTransfer.cpp`
  replaced by the PR #1274 version. This isolates cache policy from unrelated
  changes between the two PR heads.

Both full heads were built with GCC 14.2.0, identical Cortex-A65 flags, and
Release `-O3 -DNDEBUG`. Processes used the same Scheduler/Runner providers and
common installed Python binding; only the selected Core shared library varied.
`ldd` and Python's `/proc/self/maps` verified loaded paths. Exact provider versions,
model/library hashes and environment output are in [provenance.json](results/provenance.json).

The device ran kernel `6.18.3-modalix #2499`, Python 3.13.5, NumPy 2.5.4 and the
common pyneat 0.6.0 binding. CPU frequency samples remained at 1.4 GHz under the
existing `userspace` governor. No governor or global affinity settings changed.
The initial comparison's sampled temperatures were 45.35–46.59 °C.

## Workloads and method

PR #1274 reports C++ 46 MiB transfer latency and CenterPoint inference with NumPy
versus a pre-created device tensor. Issue #1273 also reports YOLOv8s.

The exact custom CenterPoint MPK was unavailable. The SDK 3.0.0 model-zoo catalog
had no CenterPoint pillar encoder; CenterNet is a different model. With the
user's approval, inference tests used these model-zoo substitutes:

| Model | MPK FP32 tensor input | Input bytes |
| --- | --- | ---: |
| `yolo_v8s` | `[1,640,640,3]` | 4,915,200 |
| `yolo_v7w6` | `[1,640,640,12]` | 19,660,800 |

Only the MPK manifest JSON supplied the model contracts. The larger YOLOv7-W6
model tests the same host-tensor-to-device-to-inference path with a larger input;
it does not reproduce CenterPoint's architecture. The exact CenterPoint input
volume, **46,080,000 bytes**, was tested independently through C++ and NumPy
transfers. This differs from **46 MiB = 48,234,496 bytes**; both were covered.

Each case ran in a fresh process with 12 C++ warm-ups or 10 Python warm-ups.
Timed regions include creation/use and destruction of the returned value, but
exclude model construction, graph initialization, random-input generation and
post-measurement payload verification. NumPy inputs use a fixed random seed.
Pre-created device tensors are made before timing. The transfer-only Python
case also retains one device tensor, as the model comparison does.

- Main comparison: eight paired rounds, alternating PR order and shuffling
  workload order, with 100 timed calls per case (60 for cache pressure).
- Larger model: six balanced three-variant rounds, 50 calls per input mode.
- C++ repeat: 16 paired rounds, 100 calls, with both variants pinned to CPU 2
  through the benchmark process and its children only. This follows noisy
  unpinned C++ results and is reported separately, without discarding that data.
- Total: **18,760 timed calls**, excluding warm-ups and the exploratory byte-fixture
  pilot. All transfer payload checks and model NumPy/device comparisons passed.
  Model output hashes also match across variants.

Tables report the median of each variant's round medians. Percentage changes
and bootstrap intervals use the *paired* round ratios, so rounding or pairing can
make them differ slightly from the ratio of displayed aggregate medians. The
95% intervals resample whole paired rounds 20,000 times with seed 1322, not
individual frames. A 5% increase is a practical review threshold, not a CI timing
assertion or a universal product guarantee.

## Warm fixed-size results

| Workload | PR #1274 p50 | PR #1322 p50 | Paired change | 95% interval |
| --- | ---: | ---: | ---: | --- |
| C++ 46 MiB `.cvu()`, CPU 2 repeat | 12.440 ms | 12.467 ms | +0.19% | −0.04% to +0.68% |
| C++ 46,080,000 bytes, CPU 2 repeat | 13.112 ms | 13.119 ms | +0.06% | −4.93% to +1.18% |
| C++ 46,080,000 bytes, two held tensors | 12.259 ms | 12.268 ms | +0.08% | −0.20% to +0.72% |
| NumPy transfer, 46,080,000 bytes | 12.279 ms | 12.296 ms | +0.06% | −0.10% to +0.21% |
| YOLOv8s, NumPy input | 6.830 ms | 6.837 ms | +0.27% | −1.23% to +2.77% |
| YOLOv8s, pre-created device tensor | 5.132 ms | 5.173 ms | +0.53% | −2.54% to +3.44% |
| YOLOv7-W6, NumPy input | 37.452 ms | 37.482 ms | +0.05% | −0.63% to +0.64% |
| YOLOv7-W6, pre-created device tensor | 31.920 ms | 31.942 ms | +0.05% | −0.11% to +0.19% |

The original unpinned C++ zero-held probes had wide intervals: −8.46% to +9.38%
for 46 MiB and −0.09% to +11.43% for the CenterPoint byte count. Those probes
alone could not exclude a modest regression; the pinned repeats above address
that uncertainty. Both original and repeat samples are retained below.

### Isolating the LRU policy on the larger model

| Workload | PR #1274 cache on PR #1322 base | PR #1322 | Paired change | 95% interval |
| --- | ---: | ---: | ---: | --- |
| YOLOv7-W6, NumPy input | 37.468 ms | 37.482 ms | +0.08% | −0.51% to +0.53% |
| YOLOv7-W6, device tensor | 31.899 ms | 31.942 ms | +0.17% | −0.13% to +0.31% |

Within these working sets, the cache-policy overhead is small relative to the
copy and inference costs. These results are not a new absolute reproduction of
PR #1274's historical CenterPoint timings.

## Cache-pressure regression

The extra synthetic stress case cycles through six exact sizes:
`46,080,000 + i * 4,096` bytes, `i=0..5`. Their combined working set is
276,541,440 bytes, about **263.73 MiB**, just above the 256 MiB limit. Only one
transient device tensor is live at a time.

| Metric | PR #1274 | PR #1322 |
| --- | ---: | ---: |
| p50 transfer time | 12.296 ms | 47.317 ms |
| Paired latency change | baseline | +284.82% (3.85×) |
| 95% interval for paired change | — | +284.08% to +285.19% |
| Median minor faults per 60 transfers | 0 | 675,150 |

The unbounded cache keeps all six allocations and reuses them. The bounded cache
can keep only five; the cyclic access order repeatedly evicts the size needed
next. Fresh allocation/mapping/page-fault work returns. This is a real, measured
performance regression for this synthetic access pattern, not a claim that all
variable-size workloads slow down by this amount.

Do not describe the fix as performance-neutral without qualifying the working
set. The 256 MiB policy needs an explicit workload tradeoff decision before
release. This benchmark does not change that policy.

## Reproduce

Build both Core revisions against the same qualified Scheduler/Runner cohort,
keeping their libraries in separate directories. For the cache-only control,
replace `TensorTransfer.cpp` only in a temporary copy of the candidate source,
build it with the same flags, and preserve that third library separately. The
PR worktree itself must retain the bounded implementation.

The standalone harness uses installed public Neat headers and a shared-library
ABI compatible with both variants. In the SDK:

```sh
cmake -S tests/perf/transfer_cache -B build-transfer-perf \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSROOT="$SYSROOT" \
  -DCMAKE_PREFIX_PATH="$SYSROOT/usr"
cmake --build build-transfer-perf -j4
```

Download model assets before timing, with an authenticated CLI session:

```sh
sima-cli modelzoo --ver 3.0.0 --boardtype modalix get yolo_v8s
sima-cli modelzoo --ver 3.0.0 --boardtype modalix get yolo_v7w6
```

Run `run_pairs.py` on the DevKit through `dk`/`dk shell`, with absolute shared paths:

```sh
dk shell 'python3 /workspace/tests/perf/transfer_cache/run_pairs.py \
  --variant pr1274=/workspace/pr1274/lib \
  --variant pr1322=/workspace/pr1322/lib \
  --runtime-library-path /workspace/runtime/lib:/usr/lib/aarch64-linux-gnu/neat/runtime \
  --transfer-bench /workspace/build-transfer-perf/transfer_bench \
  --python-executable /home/sima/pyneat/bin/python \
  --model /workspace/assets/yolo_v8s_mpk.tar.gz \
  --rounds 8 --iterations 100 --output /workspace/results/matched.json'
```

Use `--model-only --model .../yolo_v7w6_mpk.tar.gz --rounds 6 --iterations 50`
and a third `--variant cache_control=...` for the larger-model comparison.
For the C++ repeat, select `--workload cpp_46MiB --workload cpp_centerpoint_bytes
--cpu 2 --rounds 16 --iterations 100`, without model cases.

Recompute the committed measurements:

```sh
python3 tests/perf/transfer_cache/summarize.py tests/perf/transfer_cache/results/matched.json
python3 tests/perf/transfer_cache/summarize.py tests/perf/transfer_cache/results/cpp-pinned.json
python3 tests/perf/transfer_cache/summarize.py tests/perf/transfer_cache/results/large-model.json
python3 tests/perf/transfer_cache/summarize.py tests/perf/transfer_cache/results/large-model.json --baseline cache_control
```

The `results/` files preserve all timed samples, round order, correctness checks,
loaded Python library paths and available sensor readings. No model archives,
credentials or compiled binaries are included. Temporary build/linkage logs and
the exploratory pilot remain under `/workspace/tmp/dma-retention-pr1321/perf` in
the shared SDK workspace. These are local benchmark observations, not full
fleet, sustained production-load, or application-level qualification.

## Fresh-board follow-up: platform blocked

On November 7, the user selected `10.252.126.2` (MAC `68:e1:54:06:8b:58`)
for a fresh comparison of all three versions. This is a different board from the
historical measurements above. Its eLxr 12 system has GLIBC 2.36 and lacks the
required DMA kernel capabilities. An isolated SDK userspace ran Python 3.13.5,
NumPy 2.2.4 and the final Core binaries, without replacing board OS libraries.

The final revision's four focused device tests reached driver admission, then
failed: CMA CPU access reports `closed owned platform capability unavailable`
(`ENOTSUP`), and `/dev/dma_heap/simaai,dms` is absent. These are platform blockers,
not completed cache correctness or performance results. **No timed calls were
completed on this board.** Its SSH endpoint subsequently became unreachable,
before baseline/old-candidate smoke tests could run.

[Raw preflight evidence](results/newboard-preflight.json) records the device,
exact test failures, final Core hash, and prepared recovery-image checksums.
The complete eLxr 3.0 B2501 netboot set was downloaded, SHA-256 verified and
prepared locally. No reflash has occurred; eMMC reflash approval and restored
access are pending before repeating the device tests and balanced benchmarks.

# SSD BoxDecode sub-millisecond execution: revisited cross-repository plan

- Status: Implemented; isolated A65 backend accepted on Modalix, full-plugin trace lane pending
  accelerator-dispatcher availability
- Date: 2026-08-01
- Issue: [sima-neat/core#513](https://github.com/sima-neat/core/issues/513)
- Scope: Core contract, Internals decoder, prepared-model fallback, correctness, and performance gates

## Implementation result

The Core and Internals changes described below are implemented in the paired worktrees. The final
optimized runtime keeps the stable `ssd` ABI token and selects only from the two exact ordered
prepared-head signatures. Unknown geometry and activation/profile conflicts fail configuration
with the observed level shapes; no model checksum or name is consulted.

The production PackedCBlock INT8 path now uses configured activation tables, raw INT8 reduction,
NEON SSD300 class-major scanning, persistent worker/candidate buffers, and exact score-ordered lazy
top-K NMS. NMS extracts only the required prefix from a max heap, decodes coordinates on demand,
uses the legacy integer IoU rule, and stops when the final top-K is provably fixed. The scalar path
and legacy NMS remain available as the differential oracle. Target-local `-O3 -fno-math-errno` is
enabled without `-ffast-math`.

Final isolated backend results on Modalix used the two captured teammate dossiers, scalar outputs
captured from the same final library with both optimization escape hatches enabled, zero score
tolerance, 200/300 warmup iterations, and 1,000 measured iterations:

| Prepared recipe | Average | p50 | p95 | p99 | Maximum | Scalar equivalence |
|---|---:|---:|---:|---:|---:|---|
| SSD-MobileNetV2 | 0.410 ms | 0.408 ms | 0.430 ms | 0.445 ms | 0.491 ms | 24/24 boxes, exact |
| SSD300 surgical | 0.739 ms | 0.733 ms | 0.768 ms | 0.778 ms | 0.838 ms | 17/17 boxes, exact |

Core now exports plugin p50/p95/p99, rejects missing/ambiguous/unreliable component rows, and gates
the exact `A65/Run/boxdecode_backend` and `A65/Exec/boxdecode_plugin_exclusive` identities with at
least 1,000 samples. The outer plugin span is implemented in Internals. Its complete full-graph
Modalix run could not be collected in this implementation session because the accelerator
dispatcher rejected the MLA stage before pipeline startup; the isolated backend result above is
not presented as plugin-exclusive evidence.

## Executive decision

Keep the current Core shape-only recipe registry, but replace the generic per-head SSD runtime path
with one configure-time, fail-closed `SsdExecutionPlan`. The primary optimized path must match the
actual teammate artifacts: both currently arrive at BoxDecode as packed **INT8** tensors. BF16 is a
secondary adapter, not the first implementation target.

The current performance problem is almost entirely score reduction:

| Prepared recipe | Current measured latency | Score-search share | Dense score work |
|---|---:|---:|---:|
| SSD-MobileNetV2 | 4.35 ms backend; 4.52 ms complete node average | 98.9% | 1,917 anchors, 172,530 foreground comparisons |
| SSD300 surgical | 46.24 ms backend; 46.65 ms complete-plugin diagnostic average | 99.7% | 8,732 anchors, 707,292 logits and exponentials |

The captured runtime inputs prove a much better exact implementation is available:

- MobileNet has 18 threshold survivors. A raw-INT8 NEON argmax plus a configured 256-entry sigmoid
  table removes per-logit float conversion and all runtime sigmoid calls.
- SSD300's exact upper-bound test leaves exactly 84 anchors on the captured surgical workload,
  matching the 84 current threshold candidates. A configured exact 256x256 raw-pair exponential
  table removes all runtime exponential calls; only `84 * 81 = 6,804` table contributions are
  summed.

The hard acceptance target is plugin-owned exclusive BoxDecode time below 1 ms. To leave room for
input mapping, packed-input resolution, output acquisition, and metadata propagation, the backend
kernel budget is 0.70 ms p99. The regular CI gate uses percentiles; a controlled Modalix release
lane additionally enforces a measured maximum below 1.00 ms.

No checksum, model-name lookup, sorted-level heuristic, or per-frame recipe detection is used.
The complete ordered logical tensor signature selects a semantic recipe. An unrecognized signature
or an unsupported storage/dtype capability fails configuration with an actionable error.

## Goal, constraints, and non-goals

### Goal

Deliver a decoder architecture that:

- is correct for the two teammate-prepared recipes already validated by Core;
- keeps SSD300's full 81-channel softmax denominator while selecting the requested 8-class prefix;
- keeps MobileNet's per-class sigmoid and anchor-major confidence semantics;
- completes plugin-owned BoxDecode work in less than 1 ms on the agreed Modalix deployment path;
- performs no heap allocation, recipe discovery, prior generation, or transcendental math in the
  steady-state INT8 hot path;
- is reusable for a future prepared MobileNetV3 recipe without adding family-name heuristics; and
- fails closed in both Core and Internals.

### Constraints

- The models are graph-surgery products. Their output contracts, not stock upstream model names,
  are authoritative.
- MLA does not support all dynamic postprocessing operations. Dynamic threshold filtering, NMS,
  top-K, and variable detection output therefore remain outside MLA unless separately proven.
- Exact NMS cost is input-density-dependent. The absolute latency contract must name the recipe,
  threshold, top-K, selected-class count, and workload envelope. Arbitrary hostile logits cannot
  have both unbounded exact NMS semantics and a hard constant-time guarantee.

### Non-goals

- A generic decoder for arbitrary SSD-like head sets.
- Inferring activation, anchors, packing, coder variances, or resize from one tensor dimension.
- Claiming MobileNetV3 support before a teammate-prepared artifact and golden output exist.
- Moving work into another stage merely to make the A65 trace row look smaller.

## Current proven state

### Core

The uncommitted Core implementation has a sound semantic foundation:

- `SsdDecodeContract.cpp` contains the two complete ordered logical H/W/C signatures.
- Registry validity and signature uniqueness are compile-time checked.
- Matching is direct and order-sensitive.
- SSD300 softmax versus MobileNet sigmoid, confidence packing, encoded class count, background,
  class-selection policy, model frame, and required Stretch resize are descriptor properties.
- Unknown shapes already produce an actionable unsupported-recipe error.

This is retained. Performance mechanics must not be added to `SsdRecipeDescriptor`; they belong to
an execution-capability plan in Internals and deployment policy in the performance matrix.

Core still has five gaps relevant to this follow-up:

1. `perf_ssd_mobilenet_boxdecode_test.cpp:163` only requires a fuzzy-matched decoder row with a
   positive average; its baseline gates full-graph latency around 10 ms and contains no component
   cap. A 4.35 ms decoder therefore passes.
2. `MeasurePluginLatency` exposes average/minimum/maximum but no percentiles, even though
   `LttngTraceParser.cpp` already retains every plugin duration.
3. Internal normalized/lowered state still represents only one `num_classes` value. Represent
   encoded count and selected count explicitly while preserving legacy ABI
   `num_classes=selected_count`.
4. A raw `resize_mode_override` can discharge unrelated preprocess metadata requirements. Replace
   it with a typed geometry proof and discharge only mode/frame facts actually proven.
5. MobileNet C++ acceptance checks selected IoU matches, SSD300 acceptance checks only nonzero
   output/class range, and Python covers only MobileNet with skip paths. Optimized score code needs
   complete score/class/box goldens for both recipes.

### Actual prepared runtime contracts

The captured plugin dossiers show the backend contract that must be optimized:

| Recipe | Confidence shapes | Runtime dtype | Activation | Selection |
|---|---|---|---|---|
| SSD-MobileNetV2 | `19x19x273,10x10x546,5x5x546,3x3x546,2x2x546,1x1x546` | INT8 PackedCBlock | sigmoid | exact 91 including background |
| SSD300 surgical | `38x38x324,19x19x486,10x10x486,5x5x486,3x3x324,1x1x324` | INT8 PackedCBlock | softmax | 81 encoded, prefix 8 selected |

Both dossiers explicitly report `source_bf16_input=false`, `quant_needed=true`, and
`tess_needed=true`. Their byte counts are one byte per aligned element; for example, MobileNet's
`19x19x273` confidence head occupies `361 * align16(273) = 103,968` bytes. These are genuinely
INT8 inputs, not BF16 inputs relabeled by transport canonicalization.

Every head has its own positive dequantization scale and zero point. The INT8 order therefore
preserves logit order, and softmax differences cancel the zero point mathematically.

### Current Internals gaps

The current runtime path is incompatible with the performance and fail-closed goals:

1. It selects MobileNet if any head contains a `2x2` level and otherwise assumes SSD300
   (`boxdecode.cpp:3469-3478`).
2. Unsupported levels or prior mismatches silently disable a head and continue
   (`boxdecode.cpp:3483-3551`).
3. `ensure_priors()` runs lazily in the first decode path instead of configuration.
4. Each score call constructs a `TesselatedTensor`; PackedCBlock construction rebuilds tile-base
   metadata and can allocate (`tesselatedaccess.cpp`, tile-cache member near line 1009).
5. Each head allocates a channel scratch `std::vector`, returns a new candidate vector, and is
   processed sequentially (`boxdecode.cpp:3579,3651,3957`).
6. MobileNet dequantizes every foreground logit before scalar comparison
   (`boxdecode.cpp:3667-3684`).
7. SSD300 dequantizes every logit twice and calls scalar `std::exp` for all 707,292 values
   (`boxdecode.cpp:3599-3633`).
8. The existing persistent worker pool and direct find/decode pattern are wired to YOLO paths, not
   SSD (`boxdecode.cpp:332-340`). Forcing `SIMA_BOXDECODE_MULTICORE=1` leaves MobileNet at 4.35 ms
   and SSD300 at 46.26 ms.
9. The accurate trace span covers only `simaai_boxdecode_run_v1`, not all plugin-owned work
   (`neatobjectdecode/gstneatboxdecode.cpp:6200-6211`).
10. Internals' activation enum has no `Softmax = 3`, and its V2 configuration path can replace the
    supplied SSD activation with `Unknown` (`boxdecode.cpp:4728-4732,5822-5823,5911-5932`). The
    runtime must preserve and validate the activation selected by the exact recipe.

The build already targets Cortex-A65 with `-mcpu=cortex-a65` and `-O2`. Compiler flags alone cannot
produce the required 5-60x reduction, and `-O3` does not turn scalar `std::exp` into a guaranteed
vector implementation.

## Resolve execution placement before implementation

Issue #513 says SSD decoding should run on EV74, but the deployed teammate path and current trace
attribute BoxDecode to A65. Freeze the production placement and metric identity before optimizing:

1. Confirm whether the product requirement is A65 backend, EV74 backend, or complete
   `neatobjectdecode` service time.
2. Use `boxdecode_plugin_exclusive` as the product SLO. It begins after steady-state configuration
   and ends after output metadata is ready, immediately before `gst_pad_push`; downstream work
   synchronously triggered by push is excluded.
3. Retain `boxdecode_backend` as an inner diagnostic and budget it separately.
4. If EV74 is authoritative, keep the same Core contract and scalar reference, but place the
   fixed-shape score reducer on EV74. The optimized A65 implementation remains the fallback and
   correctness oracle.

This decision does not block measurement or reference-kernel work because those are shared by both
placements.

## Target architecture

```text
prepared ordered heads
        |
        v
Core exact shape recipe + geometry/class proof
        |
        v
Internals configure-time exact mirror + capability check
        |
        v
immutable SsdExecutionPlan
  - prepared packed-tensor views
  - priors/coder constants
  - INT8 activation/pair tables
  - selected-class policy
  - deterministic worker shards
        |
        v
fused score -> threshold -> survivor coordinate decode
        |
        v
class-wise NMS -> Stretch inverse -> top-K -> BBOX
```

### 1. Keep semantic recipes separate from execution capability

Core's `SsdRecipeDescriptor` continues to describe semantic truth: ordered levels, model frame,
resize, activation, channel order, background, encoded classes, and class-selection policy.

Internals adds a private `SsdRuntimeRecipe` with the same two complete ordered signatures and
backend anchor/coder data. It resolves once after all 12 tensors are known. This is a defensive
runtime mirror, not a new public recipe token. The stable runtime token remains `ssd`.

After semantic resolution, a separate capability resolver selects an optimized kernel from:

```cpp
struct SsdExecutionKey {
  SsdRuntimeRecipeId recipe;
  TensorDataType dtype;
  SsdStorageKind storage;
};
```

Initially only the measured INT8 PackedCBlock keys receive a sub-millisecond production kernel.
A recognized semantic recipe with an unsupported dtype/storage combination fails with a precise
capability error rather than being mislabeled as an unknown recipe.

### 2. Build one immutable execution plan during configuration

Add an `SsdExecutionPlan` owned by `BoxDecoder`:

```cpp
struct SsdExecutionPlan {
  SsdRuntimeRecipeId recipe;
  std::array<SsdHeadPlan, 6> heads;
  SsdClassSelection classes;
  float threshold;
  int top_k;
  int worker_count;
  SsdKernelFn kernel;
  std::array<SsdWorkerState, 4> workers;
};
```

Each `SsdHeadPlan` contains tensor offsets, immutable PackedCBlock tile/stripe geometry, logical and
physical channels, anchors per cell, quantization, prebuilt score tables, prior sizes, coder
constants, and its precomputed work partitions. The frame path supplies only the buffer base.

Configuration must:

- validate the complete ordered signature, roles, logical shapes, dtype, storage, activation,
  model frame, and class policy;
- create priors and packed-layout metadata once;
- precompute thresholds and activation tables;
- reserve cache-aligned worker candidate storage; and
- choose a benchmarked worker count and kernel function.

There is no lazy `ensure_priors()`, `std::vector` construction, or layout detection in the frame
path.

### 3. Reusable packed INT8 reader

Refactor the PackedCBlock address calculation into an immutable `PreparedPackedTensorView`.
It exposes a sequential 16-channel INT8 stripe load without allocating or recalculating tile bases.

The optimized reducer consumes stripes directly. Recipe-specific template instantiations for the
small supported set of `(anchors, classes, channel-order)` values use compile-time lane maps, so
cross-stripe anchor/class boundaries do not introduce division or modulo in the inner loop.

Keep a materializing scalar reader behind the same view for differential tests. If direct fused
traversal is not initially faster than a fixed aligned per-worker cell buffer, keep the clearer
buffered kernel; the abstraction permits both without changing decoder semantics.

### 4. MobileNet exact INT8 sigmoid kernel

For every confidence head, build once:

```cpp
std::array<float, 256> sigmoid_score;
std::int8_t first_passing_raw;
```

For every anchor:

1. Ignore background channel zero.
2. Find the highest raw foreground INT8 value with the existing sign-correct NEON argmax pattern.
   Positive scale makes this identical to logit argmax.
3. Reject raw winners below `first_passing_raw`.
4. Read the final score from `sigmoid_score[uint8_t(raw)]` only for survivors.
5. Load four localization values and decode the box immediately into the worker-local candidate
   buffer.

The captured workload reduces 172,530 scalar dequantized comparisons and 1,917 sigmoids to
integer SIMD comparisons, 18 score-table loads, and 18 coordinate decodes.

### 5. SSD300 exact INT8 softmax kernel

For each head, precompute with the same dequantization, subtraction, and `std::exp` expressions as
the scalar reference:

```cpp
struct alignas(64) Int8ExpPairTable {
  std::array<float, 256 * 256> value;
};
// value[index(max_raw, raw)] = exp(dequant(raw) - dequant(max_raw))
```

The positive scale means raw INT8 comparison selects the same maxima. A pair table, rather than a
delta-only table, preserves the current float rounding of the two dequantizations and subtraction.
It costs 256 KiB per distinct head quantization, approximately 1.5 MiB for six SSD300 heads. Share
immutable tables across decoder instances keyed by the exact scale bits and zero point.

For each anchor:

1. Scan once in class-major order and find:
   - global raw maximum and its first class index over all 81 encoded channels;
   - best selected foreground raw value and first class index over `[1, selected_count)`.
2. Compute `numerator = table[global_raw][selected_raw]`.
3. Apply the exact upper bound: if `numerator < threshold`, reject because the denominator is at
   least one.
4. Otherwise set `limit = numerator / threshold` and accumulate the full 81-class denominator in
   original class order using `table[global_raw][raw[c]]`.
5. Reject early as soon as the partial denominator exceeds `limit`; remaining positive terms
   cannot restore the candidate.
6. For a survivor, emit `numerator / denominator` and decode its localization immediately.

This preserves full 81-class normalization and prefix selection. On the captured surgical input,
the exact upper bound reduces 8,732 possible denominators to 84 and replaces 707,292 runtime
exponentials with 6,804 table contributions.

The pair-table path is expected to preserve the current score bits because table construction and
denominator accumulation retain the scalar expressions and class order. Prove this with output-hash
and threshold-boundary tests. A 256-entry delta-only table is an optional later memory optimization,
not the initial correctness path; it requires explicit tolerance and threshold-decision evidence.

### 6. BF16 and FP32 adapters

Cortex-A65 has no native BF16 arithmetic. A future BF16 adapter therefore uses integer NEON on
sign-aware BF16 sort keys, then converts only winners or denominator survivors to FP32. It must
define and test NaNs, infinities, `-0/+0`, equal-logit ties, and first-index behavior.

BF16/FP32 are not enabled as performance-supported variants merely because their shapes match.
They require their own differential and sub-millisecond evidence.

### 7. Fused survivor coordinate decode

Follow the useful YOLO26 direct-path pattern: once an anchor passes score selection, read and decode
its four localization values immediately. This removes per-head returned vectors, a second tensor
accessor construction, and the current 0.02-0.10 ms separate decode pass.

Workers append to persistent cache-line-separated candidate buffers. Merge them in deterministic
head/cell/anchor order before class-wise NMS. `output_boxes.clear()` retains capacity.

### 8. Topology-aware parallel execution

Use one persistent dispatch for all six levels, not six head barriers. Partition work during
configuration by estimated `(cells * anchors * classes)` cost and retain a deterministic segment
list per worker.

Cortex-A65 is a two-thread-per-core SMT design. Benchmark one, two, and four workers pinned to
distinct physical cores; do not assume four logical CPUs provide four independent cores. The
current Modalix mapping places CPUs `(0,1)`, `(2,3)`, and so on on the same cores, so affinity must
be validated from topology rather than hard-coded by logical CPU number.

MobileNet may be faster single-threaded after SIMD because dispatch overhead becomes material.
Choose the worker count from controlled measurements at configuration/build policy time; do not
auto-tune on customer frames.

If the existing `ThreadPool` contributes material overhead, improve it generically with fixed-size
active-worker storage, worker-index-aware persistent state, and a lightweight generation barrier.
Do not create a second SSD-only thread-pool implementation.

### 9. Plugin wrapper budget

The current MobileNet node spends roughly 0.17 ms outside the backend. Add two precise nested spans:

- `boxdecode_backend`: the existing backend call;
- `boxdecode_plugin_exclusive`: steady-state plugin-owned work from input map through output
  metadata copy, ending before downstream `gst_pad_push`.

Adopt the reusable output-buffer-pool pattern already used by `neatdetess` instead of allocating a
new output buffer every frame. Preserve zero-copy TensorBuffer input when available. Continue to
report push wait separately because it belongs to downstream scheduling, not decoder service time.

## Measurement architecture

### Isolated packed-head replay

Extend Internals' existing `boxdecode_runtime_replay` rather than creating another benchmark tool.
It already consumes captured `plugin_runtime_config.json` and `input_blob.bin` dossiers through the
real runtime ABI.

Add:

- `--warmup`, `--iterations`, `--kernel=reference|optimized`, and worker-count options;
- p50, p95, p99, maximum, cycles, instructions, L1/L2 misses, branch misses, and stalled cycles;
- phase counters for packed loads, comparisons, upper-bound survivors, table contributions,
  coordinate decodes, candidates, NMS, and final boxes;
- parsed semantic comparison of class, score, and coordinates, not only output hashes; and
- deterministic normal, no-candidate, threshold-boundary, and dense-candidate dossiers.

Use `CLOCK_MONOTONIC_RAW` for wall time and the Arm64 PMU for diagnosis. Pin the benchmark to the
selected physical CPUs.

### Full graph

Core's existing performance scenario only requires a positive fuzzy-matched decoder average and
gates approximately 10 ms full-graph p50/p95. Replace that with reusable component metrics:

1. Add p50/p95/p99 to `MeasurePluginLatency`; `LttngTraceParser` already retains every span
   duration in `SpanAgg::durations_ms`.
2. Export sample count and percentile availability.
3. Match components by exact backend, phase, kernel, stage, and instance identity.
4. Add generic nested component thresholds to the performance schema, for example:

```json
"component_latency_thresholds": {
  "boxdecode_plugin_exclusive": {
    "required": true,
    "min_samples": 1000,
    "p99_max_ms": 0.90,
    "max_max_ms": 0.999
  },
  "boxdecode_backend": {
    "required": true,
    "min_samples": 1000,
    "p99_max_ms": 0.70
  }
}
```

Absolute component caps receive no percentage regression tolerance. A missing, ambiguous,
unreliable, or trace-loss-affected component row fails the scenario.

### Controlled hard-latency lane

Ordinary Linux scheduling cannot make a defensible maximum-latency claim without environmental
controls. The release performance lane must record and enforce:

- CPU and IRQ isolation for the selected physical cores;
- fixed operating point/governor and thermal range;
- device lock and no competing performance jobs;
- recipe, artifact, threshold, selected classes, top-K, and candidate counts;
- Core, Internals, compiler, runtime, and firmware revisions; and
- at least three 10,000-frame windows after warmup.

Regular CI gates p99 and regression. The controlled lane additionally requires every measured
`boxdecode_plugin_exclusive` sample to be below 1.00 ms.

## Graph-surgery / EV74 fallback

Do not assume proprietary compiler support from operator names. First compile and benchmark tiny
fixed-shape Softmax, ReduceMax, and ArgMax probes with the actual toolchain.

If the issue's EV74 target is authoritative, or if the optimized A65 backend remains above 0.70 ms
p99, move only fixed-shape score reduction upstream:

- MobileNet emits fixed per-anchor best foreground class and logit/score.
- SSD300 computes the full 81-class denominator and selected-prefix best class, then emits fixed
  per-anchor score and class.
- Localization remains fixed-shape and unchanged.
- A65 retains threshold filtering, survivor coordinate decode, exact NMS, Stretch inverse, top-K,
  and variable BBOX output.

Prefer a configurable CVU/EV74 reducer when selected class count must remain runtime-configurable.
If score reduction is fused into a prepared MLA artifact, the selected-class policy becomes part of
that artifact's exact prepared contract.

Reduced heads are a new ordered shape contract/revision and are recognized exactly like the dense
recipes. They do not use a checksum. Full-graph performance must prove the work was accelerated,
not merely moved out of the measured BoxDecode span.

## Cross-repository implementation sequence

### Gate 0: freeze placement and the SLO

- Resolve A65 versus EV74 ownership from issue #513 and the deployed application.
- Adopt exact metric identities and the plugin-exclusive boundary.
- Capture both real dossiers and goldens as controlled test assets.

Exit: one written deployment decision and reproducible current baselines.

### Gate 1: make measurement truthful

Internals:

- add backend and plugin-exclusive spans;
- add the output pool; and
- extend `boxdecode_runtime_replay` with distributions and PMU counters.

Core:

- add plugin percentiles;
- add generic component threshold comparison; and
- add SSD300 isolated/full-graph scenarios.

Exit: the current implementation fails the new sub-millisecond gates for the correct reason.

### Gate 2: fail closed in Internals

- add the exact configure-time runtime recipe registry;
- remove partial `2x2`/default-SSD300 selection;
- add and validate the runtime `Softmax = 3` activation instead of replacing SSD activation with
  `Unknown`;
- turn unsupported levels, layouts, and dtypes into configuration errors; and
- move prior/layout/table construction out of the frame path.

Exit: Core and Internals accept and reject the same captured shape matrix.

### Gate 3: introduce the immutable packed INT8 plan

- implement `PreparedPackedTensorView` and `SsdExecutionPlan`;
- precompute six per-head sigmoid or exact raw-pair exponential tables;
- allocate/reserve worker state once; and
- keep the existing scalar path as a test oracle.

Exit: optimized and reference paths are semantically equivalent on deterministic fixtures.

### Gate 4: MobileNet fast path

- add INT8 NEON anchor-major argmax;
- raw-domain thresholding and score-table lookup;
- fused coordinate decode; and
- benchmark one/two/four workers.

Exit: isolated and full-graph plugin-exclusive p99 meet budget with real MobileNet output golden.

### Gate 5: SSD300 fast path

- add vectorized class-major global and selected-prefix maxima;
- exact upper-bound and early-denominator rejection;
- exact raw-pair table denominator; and
- fused coordinate decode.

Exit: 81-to-8 semantics match the scalar reference and both latency gates pass on normal and agreed
dense workloads.

### Gate 6: placement fallback only if evidence requires it

- run fixed-shape toolchain probes;
- implement the CVU/EV74 reduced-score path only if A65 misses budget or EV74 is required; and
- validate total graph latency and power as well as BoxDecode attribution.

Exit: chosen production placement meets the same correctness and performance contract.

### Gate 7: coverage and documentation

- mandatory C++, Python, isolated, full-graph, and performance lanes;
- installed-wheel validation for both recipes;
- strict unsupported-shape/resize/capability diagnostics; and
- update public docs with supported prepared contracts and performance scope.

Exit: definition of done below is fully evidenced.

## File-level changes

### Core

| File/area | Change |
|---|---|
| `SsdDecodeContract.{h,cpp}` | Retain exact shape registry; make class selection explicit as encoded count, selected count, and `Exact`/`PrefixFromZero`. |
| `BoxDecodeStageSemantics.cpp` and manifests | Carry explicit encoded/selected state internally while lowering legacy ABI `num_classes=selected_count`. |
| `SimaBoxDecode.cpp` | Replace broad resize override discharge with a typed geometry proof that removes only proven requirements. |
| `include/pipeline/Run.h` | Add plugin p50/p95/p99 and percentile availability. |
| `LttngTraceParser.cpp`, metric JSON/export | Finalize the already retained duration vectors and export exact component identity/distributions. |
| `tests/perf/tools/perf_schema.py` | Add generic component-latency thresholds and absolute-cap comparison. |
| SSD performance tests/baselines | Gate exact BoxDecode components; add SSD300 and dense workloads. |
| C++/Python E2E | Compare complete recipe-specific detection goldens and make required artifact lanes fail, not skip. |

### Internals

| File/area | Change |
|---|---|
| `genericboxdecode/src/boxdecode.cpp` | Exact runtime recipe registry, configured `SsdExecutionPlan`, INT8 kernels, fused decode, and fail-closed errors. |
| `genericboxdecode/src/tesselatedaccess.cpp` | Immutable prepared PackedCBlock view and allocation-free stripe traversal. |
| `genericboxdecode/src/common_box_utils.cpp` | Reusable INT8 sigmoid/raw-pair tables, vector reducers, and topology-aware persistent worker state. |
| `neatobjectdecode/gstneatboxdecode.cpp` | Accurate outer span, output pool, and separate downstream push timing. |
| `gst_plugins/test/boxdecode_runtime_replay.cpp` | Reference/optimized distributions, PMU counters, semantic comparison, and workload counters. |
| `genericboxdecode/CMakeLists.txt` | Target-specific `-O3 -fno-math-errno` only after measured codegen review; no `-ffast-math`. |

## Correctness and coverage matrix

### Internals differential tests

- supported exact signatures and every malformed/reordered/unknown variant;
- INT8 PackedCBlock addressing across shortened tiles and channel padding;
- per-head quantization scales and zero points;
- negative values, `INT8_MIN/MAX`, ties, first-index behavior, and threshold equality;
- MobileNet background exclusion, anchor-major sentinels, and sigmoid lookup;
- SSD300 class-major sentinels, full 81-way denominator, and selected 8-class prefix;
- upper-bound reject, early-denominator reject, and true-survivor paths;
- optimized versus scalar class identity, score error, coordinates, NMS, and final order;
- no-candidate, normal, and dense-candidate inputs; and
- explicit rejection of unbenchmarked BF16/FP32 capabilities.

### Core and device acceptance

- MobileNet C++ golden for every emitted class/score/box within strict tolerances;
- SSD300 surgical-video golden, not merely nonzero boxes and class range;
- non-square source proving Stretch inverse geometry;
- negative Letterbox, Crop, wrong-frame, reordered-head, and unknown-shape builds;
- sync and async lifecycle; and
- actual backend/plugin component identity and reliable samples.

### Python

- parameterize the installed-wheel test over both verified prepared recipes;
- execute the real on-device pipeline;
- parse BBOX output and compare recipe-specific goldens; and
- make the required device lane fail when staged artifacts are absent.

## MobileNetV3 support gate

The optimized score engine is intentionally parameterized by activation, order, anchor count,
encoded classes, and storage, so a verified MobileNetV3 sigmoid/anchor-major recipe can reuse it.
V3 is added only after all of the following exist:

1. teammate-prepared MPK and runtime dossier;
2. unique complete ordered shape, or intentional reuse of identical prepared semantics;
3. fixed anchors, coder, activation, packing, background, class selection, and resize evidence;
4. Internals exact registry/prior support;
5. C++ and Python detection goldens; and
6. isolated and full-graph sub-millisecond evidence.

If surgery produces the same shapes as an existing recipe but different semantics, shape-only
recognition cannot distinguish them. Surgery must normalize the semantics or emit a distinct fixed
output shape. A name or checksum is not an acceptable discriminator.

## Definition of done

The work is complete only when:

- Core and Internals fail closed on every unrecognized ordered signature;
- both real INT8 PackedCBlock recipes match their scalar references and device goldens;
- SSD300 demonstrably normalizes all 81 encoded classes while selecting the 8-class prefix;
- MobileNet demonstrably uses foreground sigmoid, not SSD300 softmax;
- no steady-state decoder allocation, recipe discovery, prior construction, or transcendental call
  appears in profile/code audit;
- both isolated and full-graph scenarios collect at least 1,000 reliable component samples with no
  drops or trace loss;
- controlled Modalix runs satisfy backend p99 <= 0.70 ms, plugin-exclusive p99 <= 0.90 ms, and
  plugin-exclusive maximum < 1.00 ms;
- normal and agreed dense workloads pass;
- complete C++, Python, and performance lanes are mandatory; and
- V3 remains rejected until its separate artifact gate is satisfied.

## External technical evidence

- [Issue #513](https://github.com/sima-neat/core/issues/513) defines model-managed SSD decoding and
  names EV74 as the intended execution target; this must be reconciled with the current A65 path.
- Arm documents Cortex-A65 as an SMT processor that executes two threads per core and supports Neon:
  [Cortex-A65 product page](https://www.arm.com/products/silicon-ip-cpu/cortex-a/cortex-a65).
- The [Cortex-A65 Software Optimization Guide](https://documentation-service.arm.com/static/5ed4b9c6ca06a95ce53f916f)
  documents integer ASIMD compare/max operations, alignment/prefetch considerations, and shared SMT
  resources.
- The [Arm Neon Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)
  defines the A64 compare and horizontal-max intrinsics used by the reducers.
- GCC documents that `-mcpu=cortex-a65` selects both available instructions and performance tuning:
  [AArch64 options](https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html).
- Arm's [Optimized Routines](https://github.com/ARM-software/optimized-routines) provide a
  high-quality vector `expf`, but its documented ULP contract is not scalar bit identity. It is a
  gated fallback, not part of the primary exact INT8 path.
- Linux documents the [Arm64 PMU interface](https://docs.kernel.org/arch/arm64/perf.html) and
  [CPU isolation](https://docs.kernel.org/admin-guide/cpu-isolation.html) needed for defensible
  low-latency measurements.

# SSD BoxDecode correctness follow-up: revisited implementation plan

- Status: Implemented and validated on Modalix; MobileNetV3 intentionally remains gated
- Date: 2026-08-01
- Issue: [#513](https://github.com/sima-neat/core/issues/513)
- Current follow-up: [#624](https://github.com/sima-neat/core/pull/624), commit `9a1fbb9`

## Executive decision

PR #512 added the public `ssd` route and shipped in v0.3.0, but left the strict correctness work to
issue #513. The implementation described here completes that Core follow-up while preserving the
existing public API and runtime ABI.

Implementation note: Core recognizes profiles solely from the complete ordered logical H/W/C head
signature described below; unrecognized shapes fail with an actionable error.

The implementation uses one resolved recipe identity throughout Core and one stable runtime family
token. It follows the useful part of the YOLO26 pattern—semantic normalization happens before
lowering—while keeping recipe details out of the public API and deployed plugin ABI.

Implemented private type model:

```cpp
enum class SsdRecipeId : std::uint8_t {
  Unknown = 0,
  Ssd300V1,
  SsdMobile300V1,
};
```

The names identify post-surgery decoder contracts, not untouched upstream neural-network
architectures. `SsdRecipeId` is private and never crosses the plugin ABI. The installed teammate
decoder recognizes only `ssd` and distinguishes the two fixed implementations from the head set
(38x38 identifies SSD300; 2x2 identifies MobileNet). Therefore Core keeps `BoxDecodeType::Ssd` and
the runtime token `ssd` stable after exact validation. Emitting `ssd300-v1` or
`ssd-mobile-300-v1` through the current ABI would be rejected as an unknown decode type.

Most importantly, recognize a profile only from its complete, ordered post-surgery head signature.
Do not use a partial heuristic such as the presence of a `2x2` head, sort the levels before matching,
or infer a profile from a model name. An exact match selects the corresponding fixed descriptor;
anything else fails with an actionable unsupported-recipe error.

## Implementation and validation result

The Core change implements the two verified prepared-head contracts, preserves the deployed `ssd`
ABI token, and rejects every other ordered head signature. Validation used the actual
teammate-prepared archives, not stock model assumptions:

- the surgical SSD300 model completed three model-managed device runs with 51 decoded boxes total,
  exercising the backend's full 81-class softmax followed by the requested 8-class prefix;
- SSD-MobileNetV2 passed the C++ golden accuracy route and the installed-style Python route; the
  Python test decodes the BBOX wire result and matches the two strong person detections;
- strict contract tests pass for both recipes and reject reordered, malformed, wrong-prior, and
  unrecognized/V3 signatures, invalid class selection, non-stretch resize, and wrong model frame;
- three 100-iteration Modalix measurements produced 99.15–99.60 FPS, 9.97–10.01 ms p50,
  10.04–10.25 ms p95, 4.33–4.34 ms attributed BoxDecode time, and zero drops; and
- `ssd_mobilenet_boxdecode` is registered in the standard performance matrix with a versioned
  Modalix baseline and 10% regression tolerance.

Recipe resolution is compile-time graph work over six fixed levels. It performs no sort, heap
allocation, checksum, model-name lookup, or per-frame discovery.

## Goal and non-goals

The goal is a fail-closed SSD contract that:

- accepts only decoder profiles that Core and Internals both implement;
- preserves the exact post-surgery output binding order;
- applies SSD300 softmax and MobileNet sigmoid correctly;
- proves stretch preprocessing and the exact model frame;
- separates encoded confidence channels from selected output classes;
- resolves once during graph compilation and adds no per-frame discovery overhead;
- has real C++, Python, device, and performance evidence; and
- makes adding a verified future profile mechanical.

It is not a generic SSD decoder, a stock-model detector, or permission to accept arbitrary SSD-like
heads. MobileNetV3 is not part of the supported set until a teammate-prepared MPK, Internals profile,
and golden device output exist.

## Evidence from the prepared teammate models

Only `mpk.json` or `*_mpk.json` was treated as model-pack truth, per the repository contract. Both
current MPKs end at `PassThrough`; neither contains a BoxDecode stage. They prove ingress and output
tensor facts, but not the decoder semantics.

| Prepared artifact | Ordered localization heads | Ordered confidence heads | Derived facts |
|---|---|---|---|
| `ssd300_heads_mpk.tar.gz` | `38x38x16`, `19x19x24`, `10x10x24`, `5x5x24`, `3x3x16`, `1x1x16` | `38x38x324`, `19x19x486`, `10x10x486`, `5x5x486`, `3x3x324`, `1x1x324` | 300x300 ingress, priors/cell `{4,6,6,6,4,4}`, 8,732 priors, 81 encoded classes |
| `ssd_mobilenet_v2_heads_mpk.tar.gz` | `19x19x12`, `10x10x24`, `5x5x24`, `3x3x24`, `2x2x24`, `1x1x24` | `19x19x273`, `10x10x546`, `5x5x546`, `3x3x546`, `2x2x546`, `1x1x546` | 300x300 ingress, priors/cell `{3,6,6,6,6,6}`, 1,917 priors, 91 encoded classes |

The MPKs do not prove any of the following:

- softmax versus sigmoid;
- localization or confidence channel order;
- anchor scales, aspect ratios, offsets, or level assignment;
- box-coordinate order or coder variances;
- the background-class policy;
- the resize transform;
- whether 81 encoded classes may be reduced to 8 active classes; or
- `top_k` capacity.

Those facts currently come from the teammate applications and the matching Internals decoder:
the SSD300 application selects softmax and requests an 8-class result from 81 encoded channels;
the MobileNet application explicitly selects 300x300 Stretch, sigmoid, background class 0, and
anchor-major confidence packing. They must be captured as a versioned supported profile and proved
with numeric/golden tests. The exact ordered signature selects that descriptor; no individual
semantic is inferred independently from a partial shape.

## Current-tip gap analysis

### Merge-blocking correctness defects

1. `resolve_ssd_recipe()` sorts `(feature_size, priors_per_cell)` before comparison, but the
   compiled runtime bindings remain in their original order. A permuted set can pass validation and
   then use the wrong fixed anchors for each tensor. Compare ordered bindings directly; never sort.

2. The raw `SimaBoxDecode` constructor silently writes `ResizeMode::Stretch` when no resize proof is
   present, then removes preprocess metadata requirements. This can label a real letterbox or crop
   graph as stretch. Unknown preprocessing must fail; an explicit caller assertion is acceptable,
   but Core must never manufacture it.

3. Shape matching is treated as sufficient evidence for anchors, score activation, background,
   and channel packing. It is not sufficient after graph surgery.

4. Token-only SSD contracts temporarily default to softmax. An unresolved compatibility selector
   must stay unresolved and fail at finalization; it must not acquire a plausible default.

5. SSD normalization exists in multiple paths, including static-contract overrides and compiled-
   payload overrides. Later patches can silently overwrite an earlier authoritative decision.

6. `ssd_model_frame` is copied through several Core C++ structs but is not transported in
   `SimaPluginBoxDecodeStagePayload`. It can drift from the actual runtime profile. Derive frame
   requirements from the resolved recipe and remove this parallel field.

7. The current single `num_classes` value conflates confidence-head storage with output selection.
   If the backend uses 8 instead of 81 to index SSD300 confidence data or compute softmax, narrowing
   corrupts channel strides and probabilities.

### Evidence and coverage gaps

- `ssd300_contract_e2e_test.cpp` is a device-free contract test. It loads no MPK and executes no
  graph or plugin; move/rename it as a unit or contract-integration test.
- The Python test mainly checks enum/options mutability, can skip its useful part, and does not run a
  complete pipeline or inspect detections.
- The MobileNet performance loop measures full-graph mean wall time only. It neither isolates
  `neatobjectdecode` nor reports percentiles or a regression baseline.
- There is no actual SSD300 device E2E in the follow-up.
- The MobileNetV3 case is synthetic. There is no teammate-prepared V3 MPK or golden device output.
- The branch is five commits behind `develop`, including recent model-loading and performance work.
- The PR title/body still say two recipes while the code claims three, and its PR template/test
  evidence is incomplete.

## Target contract architecture

### One private profile registry

Add `src/pipeline/internal/sima/stagesemantics/SsdDecodeContract.{h,cpp}`. It owns a small private,
`constexpr` registry. `std::span` is appropriate only here because every span points to static
storage; it must not appear in a persisted or C ABI structure.

```cpp
struct SsdLevelSpec {
  int height;
  int width;
  int localization_channels;
  int confidence_channels;
};

enum class SsdClassSelectionPolicy {
  Exact,
  PrefixSubset,
};

struct SsdRecipeDescriptor {
  SsdRecipeId id;
  std::span<const SsdLevelSpec> ordered_levels;
  int model_width;
  int model_height;
  ResizeMode required_resize;
  BoxDecodeScoreActivation activation;
  SsdLocalizationChannelOrder localization_order;
  SsdConfidenceChannelOrder confidence_order;
  int background_class;
  int encoded_class_count;
  SsdClassCountPolicy class_count_policy;
};
```

Use a precise class policy, not `bool allow_class_narrowing`. Core does not duplicate the backend's
anchor tables or coder implementation. The recipe ID selects Core validation semantics only; the
runtime continues to receive `ssd` plus the exact validated tensor geometry.

### Separate observed facts from claimed semantics

Represent the result of MPK/upstream inspection independently:

```cpp
struct SsdObservedLevel {
  TensorBindingId loc_binding;
  TensorBindingId confidence_binding;
  TensorHwc loc;
  TensorHwc confidence;
  int priors_per_cell;
};

struct SsdObservedContract {
  std::array<SsdObservedLevel, 6> levels;
  std::size_t level_count;
  FrameSize mla_ingress;
  int encoded_class_count;
  int total_priors;
};
```

Observation may derive divisibility, priors per cell, encoded classes, and prior totals. It may not
derive activation, anchors, packing, background, or resize.

Implement three pure steps and make every route use them:

```text
observe immutable MPK/upstream facts
        -> resolve exact Core-side recipe
        -> canonicalize bindings and validate/normalize the full contract
        -> lower one immutable payload
```

`BoxDecodeStageSemantics.cpp` should orchestrate these steps rather than contain SSD tables.
Resolution is O(levels), uses fixed-size storage, performs no sorting, and happens only during graph
compilation. Runtime performs no profile discovery.

### Resolution rules

Use these strict rules:

1. Build the complete observed signature in runtime binding order: every localization and
   confidence tensor's logical H/W/C, role, level, priors per cell, and consistent encoded class
   count.
2. For `BoxDecodeType::Ssd`, compare that signature directly against every registered profile. One
   exact match resolves the recipe ID.
3. If an internal recipe ID is already attached while re-normalizing a contract, compare only
   against that descriptor and reject a mismatch.
4. The registry itself is compile-time validated to prevent two descriptors from accepting the same
   signature.
5. A malformed SSD contract or a well-formed but unrecognized signature fails during graph
   compilation. There is no generic or best-effort fallback.

This works identically for model-managed and raw/standalone routes once the complete tensor contract
is available. If future MPK role metadata permits canonicalization, reorder every parallel field
together: logical inputs, slice shapes, storage kinds, quantization metadata, physical bindings, and
names. Without explicit roles, require the exact canonical runtime order.

The unsupported-recipe diagnostic must show the evidence needed to fix the model, for example:

```text
SSD BoxDecode: unsupported ordered head signature.
Observed loc=[38x38x16,19x19x24,...], conf=[38x38x324,19x19x486,...],
encoded_classes=81, input_frame=300x300.
Supported profiles: ssd300-v1 loc=[...], conf=[...]; ssd-mobile-300-v1 loc=[...], conf=[...].
Expected binding order is loc[0..5] followed by confidence[0..5]; levels are not reordered.
```

Use a separate diagnostic for malformed inputs, identifying the first bad tensor, expected
loc/conf pairing rule, and failed invariant.

### Class-count semantics

Carry these concepts separately:

- `encoded_class_count`: derived from confidence-head storage; always drives indexing and the
  softmax/sigmoid computation;
- `class_selection`: all encoded classes, a verified contiguous prefix/range, or an explicit class
  map; and
- output class count: derived from the selection, never used as the encoded channel stride.

Before enabling SSD300's 81-to-8 behavior, audit Internals and add a sentinel-channel numeric test.
If the backend's existing `num_classes` drives indexing, bump the plugin ABI and add explicit
`encoded_num_classes`, `selected_class_count`, and `class_selection_mode` fields. If Internals
already derives the encoded count from slice geometry and uses `num_classes` only as a post-score
filter, document that contract and prove it in tests. Until one of those is true, require exact class
count. Mobile300 remains `Exact` unless an actual prepared artifact proves another policy.

### Resize as proof, not a default

Introduce one internal resolved geometry value, for example:

```cpp
struct PreprocessGeometryProof {
  ResizeMode mode;
  FrameSize input;
  FrameSize output;
  PreprocessProofSource source; // MPK plan, upstream Preproc, user assertion, or identity.
};
```

Apply the following matrix in the central validator:

| Route state | Result |
|---|---|
| Resize runs and is Stretch to the profile frame | Accept |
| Resize runs as Letterbox or Crop | Reject |
| Resize output differs from the profile frame | Reject |
| Resize is disabled and the proven input is already exactly the profile frame | Accept as identity |
| Standalone/raw route has an explicit truthful Stretch assertion and exact output frame | Accept |
| Transform or frame is unknown | Reject |

Remove the raw constructor's automatic Stretch stamp. Also stop removing the entire `preproc_*`
metadata family because only resize mode was asserted; discharge each required field only when that
specific fact has a proof. Use a non-square source such as 1280x720 in numeric tests so incorrect
x/y inverse scaling cannot hide.

### Internals contract

The audited teammate Internals implementation accepts only `ssd`. It selects SSD300 when the head
set contains 38x38 and MobileNet when it contains 2x2, then builds priors lazily once per head. Its
SSD300 score path derives the encoded class stride from confidence depth, normalizes softmax over
all emitted classes, and uses `num_classes` only to bound the reportable foreground prefix. This
makes the 81-to-8 surgical route safe without an ABI bump. MobileNet uses anchor-major confidence
channels and per-class sigmoid.

Core's exact full-signature validator is therefore the capability gate for the current ABI: only
the two head sets that Internals implements can reach `ssd`. A future recipe must update both
repositories and add device evidence before entering the registry. A future Internals revision may
replace its partial runtime selector with an explicit recipe field, but Core must not emit tokens
that the deployed decoder rejects.

## File-by-file Core changes

| File | Required change |
|---|---|
| `include/pipeline/BoxDecodeType.h` | Preserve `Ssd=22` and the stable runtime token `ssd`; document exact prepared-profile validation. |
| `src/pipeline/internal/sima/BoxDecodeTypeUtils.{h,cpp}` | Keep parsing/printing the supported `ssd` family token; recipe tokens remain internal diagnostics. |
| `src/pipeline/internal/sima/stagesemantics/SsdDecodeContract.{h,cpp}` | New private registry, observed-contract extraction, exact order-sensitive shape resolution, class policy, diagnostics, and unit-test seams. |
| `src/pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.{h,cpp}` | Become the single normalization/lowering authority; delegate SSD work to the new module; remove `SsdRecipe`, sorting, and token-only defaults. |
| `src/pipeline/internal/sima/MpkContract.{h,cpp}` | Preserve authoritative ordered output bindings and explicit tensor role/level metadata when available. |
| `src/pipeline/internal/sima/BoxDecodeStaticContractExtractor.{h,cpp}` | Extract facts only and carry the internal resolved recipe ID; remove SSD activation/profile guesses. |
| `src/pipeline/internal/sima/PluginContractSubsets.{h,cpp}` | Propagate the resolved internal recipe and encoded/selected class semantics; remove `ssd_model_frame`. |
| `src/pipeline/internal/sima/SimaPluginStaticManifest.{h,cpp}` | Carry the recipe inside Core, but lower the backend-compatible `ssd` string through the plugin ABI. |
| `include/gst/SimaPluginStaticManifestAbi.h` | No change for profile selection. Bump the ABI only if encoded and selected class counts must be transported separately. |
| `src/model/ModelPack.cpp`, `src/model/Model.cpp`, `src/model/RoutePlanner.cpp` | Route all model-managed resolution through the central normalizer; validate the resolved preprocess proof; remove SSD-specific late patches. |
| `src/nodes/sima/SimaBoxDecode.cpp` | Delete `apply_ssd_compiled_payload_overrides`, frame duplication, and automatic Stretch; require the central resolved contract and explicit raw-route proof. |
| `python/src/module.cpp` | Keep the stable `BoxDecodeType::Ssd` binding. Do not expose recipe IDs or V3 until its artifact gate passes. |
| `docs/reference/boxdecode_decode_types.md` and MPK/architecture docs | Document supported prepared profiles, exact failure behavior, explicit raw usage, compatibility resolution, and versioning. |
| CMake source/test lists | Register the new contract module and split unit, integration, device, Python, and performance lanes correctly. |

Replace all ad hoc `decode_type == BoxDecodeType::Ssd` checks with the family predicate. A repository-
wide audit is necessary because SSD checks currently appear across model, extractor, semantics,
node, Python, and tests.

## Verification plan

### Deterministic unit and reference tests

Add fixture-free strict tests for:

- registry type/token/revision uniqueness and token round trips;
- expected level count, ordered signature, encoded classes, and total priors for each verified
  profile;
- generic-family compatibility matching and exact concrete-profile mismatch;
- exact generic-SSD resolution for both supported ordered signatures and rejection of every
  unrecognized signature;
- swapped levels, reversed levels, role-group permutation, missing/extra heads, mismatched loc/conf
  grids, non-square heads, bad divisibility, inconsistent class depths, and wrong frame;
- logical semantic channel depth versus padded physical storage depth;
- token-only or raw generic SSD rejection;
- numeric SSD300 softmax versus Mobile300 sigmoid using the same logits;
- sentinel values that distinguish anchor-major from class-major confidence indexing;
- zero and non-zero box deltas against trusted prior/coder reference outputs;
- background class removal and class-ID mapping;
- encoded 81 versus selected 8 behavior without changing softmax/indexing strides; and
- the full resize decision matrix and inverse stretch mapping from a non-square source.

The current synthetic `ssd300_contract_e2e_test.cpp` belongs in this layer under a truthful name.

### MPK contract integration

Check in small, provenance-documented authoritative MPK JSON fixtures for the two teammate packs,
or stage equivalent deterministic fixtures in CI. Exercise:

```text
MPK parse -> static extraction -> profile resolution -> subset -> compiled payload -> backend fragment
```

Assert the exact original binding indices, resolved recipe ID, activation, encoded and selected
classes, frame, resize requirement, storage/slice geometry, and diagnostics. Test both an MPK-
pre-resolved internal recipe and exact-shape resolution from the public `ssd` selector.

### Device E2E

Create strict DevKit jobs with versioned, staged artifacts. Missing configured artifacts or
dispatcher support fails the job; only explicitly labelled optional/long download tests may return
skip.

- Prepared SSD300: real MPK, non-square input, softmax, 81 encoded classes, verified 8-class
  selection, and golden boxes/scores/class IDs.
- Prepared Mobile300: real MPK, non-square input, sigmoid, 1,917 priors, and golden detections.
- Cover the model-managed route and a standalone/captured-head route with an explicit resize proof.
- Cover synchronous execution and one asynchronous lifecycle/throughput run.
- Add negative build tests for letterbox, crop, wrong frame, and conflicting internal recipe.

### Python

Keep enum/token/options/import tests fixture-free and strict. Add an installed-wheel DevKit test,
parameterized by verified profile, that builds and runs the complete pipeline, parses BBOX output,
compares golden detections, and checks resize/profile rejection. Do not hardcode 300 in helpers
intended to serve a future 320 profile. Optional local artifact discovery may be a separate `long`
test, not the acceptance test.

### Performance

Remove the ad hoc stopwatch. Use `Run::start_measurement(MeasureOptions)` with warmup and
`include_plugin_latency=true`; assert that an attributed `neatobjectdecode`/BoxDecode row exists and
report its `count`, p50, p95, p99, maximum, and reliability. Also report full-pipeline single-flight
latency and async throughput.

Use two workloads:

1. captured fixed post-surgery head tensors to isolate BoxDecode; and
2. the full prepared-model graph for customer-visible cost.

Record profile token/revision, artifact version, board/platform, Core and Internals SHAs, threshold,
`top_k`, selected classes, and detection density. Keep correctness CI free of absolute timing gates.
Run performance tests under the device resource lock, repeat them, and compare median/p95 against a
board-specific baseline with a documented tolerance after a stable baseline has been collected.

## Implementation sequence

### Phase 0: freeze contracts and unblock the branch

1. Rebase PR #624 on current `develop` and resolve conflicts before modifying its design.
2. Archive the two authoritative MPK JSON contracts, compiler/SDK versions, ordered outputs,
   teammate preprocessing settings, host reference output, and device golden output.
3. Audit Internals class indexing, confidence packing, anchors, coder constants, and the legacy
   MobileNet `2x2` heuristic. Decide whether class selection requires a plugin ABI bump.
4. Confirm the deployed Internals token and selection behavior (`ssd`, verified by source audit).

### Phase 1: type and registry foundation

1. Add the private recipe ID and family helper without changing the public enum or ABI.
2. Implement the private profile registry and its pure observation/resolution/validation API.
3. Add registry, token, ordered-signature, numeric activation, channel-order, and class
   semantics unit tests.

### Phase 2: one Core normalization path

1. Route MPK, static, subset, model-managed, and standalone construction through the central API.
2. Canonicalize only from explicit role/level metadata; otherwise require exact order.
3. Remove the sorted matcher, softmax fallback, duplicated SSD overrides, and `ssd_model_frame`.
4. Improve failures to print resolved recipe, observed ordered heads, expected
   ordered heads, encoded classes, frame, resize proof source, and the first mismatch.

### Phase 3: resize and class correctness

1. Propagate `PreprocessGeometryProof` from the resolved model plan or upstream Preproc contract.
2. Make raw callers state a truthful explicit proof; remove silent Stretch.
3. Separate encoded and selected class semantics end-to-end, bumping the plugin ABI if required.
4. Complete non-square stretch and 81-to-8 reference tests before enabling narrowing.

### Phase 4: Internals compatibility

1. Audit the actual teammate decoder rather than assuming stock SSD behavior.
2. Preserve its supported `ssd` token and prove Core never lowers recipe IDs as decode types.
3. Prove SSD300 emitted-class stride/full softmax plus selected-prefix behavior with the 81-to-8
   device test.
4. Treat any future recipe or ABI expansion as a coordinated Core/Internals change.

### Phase 5: integration, E2E, Python, performance, and docs

1. Add real teammate-MPK contract integration and device golden tests.
2. Replace skippable Python smoke as acceptance evidence with strict behavior coverage.
3. Replace mean-only timing with standard measurement/plugin percentile reporting.
4. Update reference, MPK-contract, architecture, compatibility, and migration documentation.
5. Complete the PR template, attach exact commands/results, and obtain a fresh approval.

### Phase 6: MobileNetV3 artifact gate

Do not expose or document a 320 profile until all of these exist:

- the exact teammate graph-surgery recipe and prepared MPK;
- an authoritative ordered MPK head/binding contract;
- explicit anchor/coder/packing/activation/background/class semantics in Internals;
- host-reference golden decoding;
- C++ and Python device golden E2E;
- BoxDecode and full-pipeline performance evidence; and
- a coordinated Core/Internals recipe capability decision.

Upstream names are not enough. TensorFlow's MobileNetV3 SSDLite configuration uses a six-level,
320x320 sigmoid recipe with a reduced first anchor layer, while Torchvision's similarly named
`ssdlite320_mobilenet_v3_large` uses a different anchor count and applies softmax. A broad
`MobileNetV3` label would therefore be ambiguous even before SiMa graph surgery.

## Definition of done

- Only the two verified prepared profiles are documented and exposed as supported.
- `BoxDecodeType::Ssd` reaches runtime only after an exact registered ordered shape resolves to an
  internal recipe.
- An internal recipe conflict or unknown raw profile fails during graph compilation.
- Binding order is validated exactly or canonicalized from explicit role metadata as one atomic
  permutation.
- SSD300 softmax and Mobile300 sigmoid are proved numerically and by device golden detections.
- Stretch and exact model frame are proved; letterbox, crop, contradiction, and unknown all fail.
- Encoded class indexing is independent of selected output classes and 81-to-8 is proved safe.
- The current Internals `2x2`/`38x38` selector is safe only behind Core's exact full-signature gate
  and is covered by cross-repository device acceptance.
- No profile discovery, allocation, sort, or anchor generation occurs per frame.
- Strict unit, MPK integration, C++ device, Python behavior, and performance lanes pass.
- V3 remains unavailable until its artifact gate passes.
- PR #624 is rebased, its template/evidence is complete, requested changes are cleared, and issue
  #513 is closed only after the follow-up ships.

## Explicitly rejected approaches

- Matching an unordered set of feature sizes.
- Inferring sigmoid from `2x2`, MobileNet from a model name, or softmax from generic `ssd`.
- Treating a partial or unordered shape match as a supported recipe.
- Silently stamping Stretch when upstream preprocessing is unknown.
- Exposing the private recipe ID as a second independently mutable public discriminator.
- Using one `num_classes` value for both confidence storage indexing and output filtering without
  proof.
- Advertising V3 from synthetic shapes or upstream documentation alone.
- Calling a pure contract test E2E or accepting a positive finite mean as performance regression
  coverage.

## Internet and review references

- [Issue #513: Add SSD decode family to BoxDecode](https://github.com/sima-neat/core/issues/513)
- [Base implementation PR #512](https://github.com/sima-neat/core/pull/512)
- [Core v0.3.0 release](https://github.com/sima-neat/core/releases/tag/v0.3.0)
- [Deferred follow-up requirements in the PR #512 approval](https://github.com/sima-neat/core/pull/512#pullrequestreview-4696511803)
- [Open correctness follow-up PR #624](https://github.com/sima-neat/core/pull/624)
- [Order-sorting review finding](https://github.com/sima-neat/core/pull/624#discussion_r3636286732)
- [Unproven raw Stretch review finding](https://github.com/sima-neat/core/pull/624#discussion_r3664503896)
- [TensorFlow SSD MobileNetV2 configuration](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config)
- [TensorFlow SSDLite MobileNetV3 320 configuration](https://github.com/tensorflow/models/blob/4d7bdd8c170ee90850f2f9ccef0f6d19b817de35/research/object_detection/samples/configs/ssdlite_mobilenet_v3_large_320x320_coco.config)
- [TensorFlow MobileNetV3 feature-grid test](https://github.com/tensorflow/models/blob/4d7bdd8c170ee90850f2f9ccef0f6d19b817de35/research/object_detection/models/ssd_mobilenet_v3_feature_extractor_testbase.py#L74-L87)
- [Torchvision SSDLite MobileNetV3 construction](https://github.com/pytorch/vision/blob/10f68dbd78b9aa5cab9328f3b2e99cfb0b608122/torchvision/models/detection/ssdlite.py#L299-L325)
- [Torchvision SSD softmax postprocess](https://github.com/pytorch/vision/blob/10f68dbd78b9aa5cab9328f3b2e99cfb0b608122/torchvision/models/detection/ssd.py#L414-L433)

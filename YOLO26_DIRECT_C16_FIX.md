<!-- workflow wf_7f28c449-ac5, 2026-06-03 -->

All facts confirmed. Critical: BOTH the neatobjectdecode plugin AND genericboxdecode_v2 link the SAME `simaai_genboxdecode_v2` backend, built from the SAME `boxdecode.cpp`. So v2's backend reads V26 with the identical unpadded stride-4 — meaning **v2 does NOT actually decode yolo26 DIRECT correctly either**; v2's host just computes the C16 contract but the shared backend would mis-read too. The claim "v2 already handles sub-16ch correctly in direct" is unverifiable/false for the read side.

I have everything needed. Synthesizing.

# SiMa BoxDecode Bug #2 — Synthesis: yolo26 DIRECT shadow contract fix

## 1. Winner + why

**Winner: a TWO-SIDED fix derived from candidate "Port v2 align_channels_16 into v1 element-count multiply" (host half) PLUS the mandatory backend half it lists in prose but omits from its diff.**

No purely host-only candidate is correct. I verified in the real backend that the DIRECT read side genuinely walks the parent at **unpadded stride-4 / total 705600** for V26, on *both* the INT8 and BF16 paths:

- `boxdecode.cpp:2445` `tensor.channels_are_physical = (model_type == MODEL_TYPE::YOLO_V26);`
- `boxdecode.cpp:2446-2449` `storage_channels = channels_are_physical ? tensor.channels : align_channels(...)` → **4**, and `tensor.data_offset = cur_data_offset; ... cur_data_offset += tensor.tensor_size` accumulates UNPADDED → offsets `{0,25600,32000,33600,545600,673600}`, total 705600.
- BF16 op `Yolo26DirectBF16FindDecodeOp` `boxdecode.cpp:986` `bbox_channel_stride_(std::max(4, bbox.channels))` = 4; row stride `width*4` (998); per-x `x*4` (1028).
- INT8 path `tesselatedaccess.cpp:116` `aligned_tensor_channels = channels_are_physical ? tensor_channels : align_channels(tensor_channels)` → **4**; non-tess HwcAligned read steps by `aligned_tensor_channels` (240, 370).

The proven on-board parent is **806400** (C16-padded by the MLA — that is exactly what the error reports as `packed=806400`). So today **both** host (expects 705600) and backend (reads 705600) disagree with the real parent. A host-only patch flips the validator to 806400 and makes it PASS, but the backend would then dereference into an 806400 buffer using stride-4 / 705600 offsets → **contract PASSES, boxes garbled** — strictly worse than today's fail-loud. The brief's hard requirement "keep backend READ side consistent" is therefore unmet by every host-only candidate. This matches the **"flawed" verdicts** on all three candidates that shipped host-only (read-side mismatch is code-proven at 2447/986/116), and the honest **"viable, residual backend risk"** verdicts confirm the host arithmetic itself is correct.

**Important correction to the brief and to candidate claims:** the neatobjectdecode plugin and `genericboxdecode_v2` both link the **same** `simaai_genboxdecode_v2` backend, built from the **same** `vendor/.../boxdecode.cpp` (`neatobjectdecode/CMakeLists.txt:128`, `genericboxdecode_v2/CMakeLists.txt:99,102`). So the premise "v2 already decodes yolo26 DIRECT correctly" is **false for the read side** — v2 would mis-read identically. v2 is only the correct reference for the *host contract idiom*, not the backend.

**Idiom source reused:** v2's canonical `align_channels_16` (`genericboxdecode_v2/gstneatboxdecode.cpp:352-357`) and the `const int aligned_depth = align_channels_16(depth)` step inside `compute_tensor_expected_bytes` (`:380-402`); on the backend, the existing `align_channels()` helper used unconditionally everywhere else (`boxdecode.cpp:498,619,648,1086,1201,1317,1916`).

**Why the host edit goes in the byte sizer, not the element-count function:** v2's idiom pads **bytes**, never element counts (v2 has *no* `build_tensor_element_counts_from_cfg` twin — verified: grep returns nothing). The same-author placement is to align depth inside the byte computation, leaving raw element counts raw. I therefore route `build_tensor_byte_sizes_from_cfg` through `align_channels_16(d)` and leave `build_tensor_element_counts_from_cfg` untouched — which also automatically preserves the BF16-stats path (`self->bf16_tensor_elements`, populated independently at `:4374`, consumed only by debug-gated `maybe_log_bf16_input_stats` at `:2850-2859`).

## 2. Exact minimal patch

### HOST — `neatobjectdecode/gstneatboxdecode.cpp`

**EDIT 1** — add the canonical helper (verbatim port of `genericboxdecode_v2/gstneatboxdecode.cpp:352-357`) at file scope, immediately before `build_tensor_byte_sizes_from_cfg` (line 2785):

```cpp
int align_channels_16(int channels) {
  if (channels <= 0) {
    return 0;
  }
  return ((channels + 15) / 16) * 16;
}
```
*Justification:* the single canonical C16 idiom; copied byte-identical from the sibling. No bespoke arithmetic.

**EDIT 2** — route `build_tensor_byte_sizes_from_cfg` through aligned depth instead of raw element counts.

BEFORE (`gstneatboxdecode.cpp:2795-2842`, real code quoted):
```cpp
  out_sizes->clear();
  *out_total_bytes = 0U;

  std::vector<std::size_t> element_counts;
  if (!build_tensor_element_counts_from_cfg(cfg, &element_counts, error)) {
    return false;
  }

  out_sizes->reserve(element_counts.size());
  for (std::size_t i = 0; i < element_counts.size(); ++i) {
    std::string dtype = cfg.input_dtype;
    if (!cfg.data_type.empty()) {
      dtype = (i < cfg.data_type.size()) ? cfg.data_type[i] : cfg.data_type.front();
    }

    std::size_t elem_bytes = 0U;
    if (cfg.source_bf16_input) {
      elem_bytes = 2U;
    } else {
      elem_bytes = dtype_element_size_bytes(dtype);
    }
    if (elem_bytes == 0U) {
      if (error) {
        *error = "unsupported dtype in typed tensor contract for byte sizing";
      }
      return false;
    }

    std::size_t tensor_bytes = 0U;
    if (!safe_size_mul(element_counts[i], elem_bytes, &tensor_bytes)) {
      ...
```

AFTER (only the source of the volume changes; `source_bf16_input`/`elem_bytes` logic preserved verbatim):
```cpp
  out_sizes->clear();
  *out_total_bytes = 0U;

  const std::size_t n = cfg.input_shapes.size();
  if (n == 0U || cfg.input_layouts.size() != n) {
    if (error) {
      *error = "invalid typed tensor contract cardinality for byte sizing";
    }
    return false;
  }

  out_sizes->reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    int w = 0;
    int h = 0;
    int d = 0;
    if (!shape_desc_whd_from_layout(cfg.input_shapes[i], cfg.input_layouts[i], &w, &h, &d)) {
      if (error) {
        *error = "invalid tensor dimensions in typed tensor contract for byte sizing";
      }
      return false;
    }
    const int aligned_d = align_channels_16(d);

    std::string dtype = cfg.input_dtype;
    if (!cfg.data_type.empty()) {
      dtype = (i < cfg.data_type.size()) ? cfg.data_type[i] : cfg.data_type.front();
    }

    std::size_t elem_bytes = 0U;
    if (cfg.source_bf16_input) {
      elem_bytes = 2U;
    } else {
      elem_bytes = dtype_element_size_bytes(dtype);
    }
    if (elem_bytes == 0U) {
      if (error) {
        *error = "unsupported dtype in typed tensor contract for byte sizing";
      }
      return false;
    }

    std::size_t tensor_bytes = 0U;
    if (!safe_size_mul(static_cast<std::size_t>(w), static_cast<std::size_t>(h), &tensor_bytes) ||
        !safe_size_mul(tensor_bytes, static_cast<std::size_t>(aligned_d), &tensor_bytes) ||
        !safe_size_mul(tensor_bytes, elem_bytes, &tensor_bytes)) {
      if (error) {
        *error = "tensor byte size overflow in typed tensor contract";
      }
      return false;
    }
    std::size_t total_bytes = 0U;
    if (!safe_size_add(*out_total_bytes, tensor_bytes, &total_bytes)) {
      ...   // unchanged
```
*Reused helpers (all pre-existing in v1):* `shape_desc_whd_from_layout` (`:234`), `safe_size_mul` (`:2154`), `safe_size_add` (already used at `:2831`), `dtype_element_size_bytes` (`:804`), plus the verbatim `source_bf16_input`→`2U` block. Only new symbol: `align_channels_16`.
*Why not call `build_tensor_element_counts_from_cfg` anymore:* its raw counts must stay raw for the BF16-stats path at `:4374`; the v2 idiom pads bytes, not counts.

Net host: ~1 new helper (5 lines) + reworked loop (~18 lines changed). `input_segment_sizes` is single-sourced (only producer is the call at `:4296`); every offset consumer derives cumulative offsets by summing it (`expected_layout_string` `:4734`, direct walk `:4823`, packed-reconstruct `:4892`), so offsets+total shift in lockstep.

### BACKEND — `vendor/sima-ai-a65-apps/genericboxdecode/src/boxdecode.cpp` (mandatory)

**EDIT 3** — make V26 storage C16-aligned at BOTH config sites (`:2446-2447` and `:3166-3167`, identical):

BEFORE:
```cpp
      const int storage_channels =
          tensor.channels_are_physical ? tensor.channels : align_channels(tensor.channels);
```
AFTER:
```cpp
      const int storage_channels = align_channels(tensor.channels);
```
*Justification:* makes `tensor.data_offset`/`tensor.tensor_size` stride C16 → offsets `{0,102400,128000,134400,646400,774400}`, total 806400. For non-V26 `channels_are_physical` is already false so the old ternary already evaluated `align_channels(...)` → **byte-identical no-op for yolov8/v9/v10**.

**EDIT 4** — make the BF16 op read at the C16 stride so it matches the now-padded offsets (`boxdecode.cpp:986`):

BEFORE:
```cpp
        bbox_channel_stride_(std::max(4, bbox.channels)),
        score_channel_stride_(std::max(num_classes, scores.channels)) {}
```
AFTER:
```cpp
        bbox_channel_stride_(std::max(4, align_channels(bbox.channels))),
        score_channel_stride_(std::max(num_classes, align_channels(scores.channels))) {}
```
*Justification:* the op reads only the first 4 / `num_classes` lanes per cell but **steps** rows/x by this stride (`:998,1009,1028`). With data now C16-padded the step must be 16 for bbox (no-op for class: `align_channels(80)=80`). Reuses backend `align_channels`. yolov8 BF16: `align_channels(64)=64`, `align_channels(80)=80` → no-op.

**EDIT 5** — make the INT8/non-tess DIRECT reader stride C16 (`tesselatedaccess.cpp:116`):

BEFORE:
```cpp
    aligned_tensor_channels = channels_are_physical ? tensor_channels : align_channels(tensor_channels);
```
AFTER:
```cpp
    aligned_tensor_channels = align_channels(tensor_channels);
```
*Justification:* this drives the non-tess HwcAligned element offset/step (`:173,240,370`). For V26 it was 4 (unpadded); making it `align_channels` walks the C16 plane. Non-V26 already passes `channels_are_physical=false` → already `align_channels` → no-op. (Line 118 `aligned_slice_channels` and the `non_tess_linear_mode` predicate at 134-137 still see `channels_are_physical=true`, keeping `slice_channels<=tensor_channels` semantics; only the storage stride changes.)

`channels_are_physical` flag stays `true` so the BF16 fast-path gate (`:965`) and slice predicate (`:428-431`) keep firing.

## 3. Byte-math proof

yolo26 grouped reg,reg,reg,cls,cls,cls. `align_channels_16(4)=16`, `align_channels_16(80)=80`.

INT8 (elem_bytes=1): bbox `80·80·16=102400`, `40·40·16=25600`, `20·20·16=6400`; class `80·80·80=512000`, `40·40·80=128000`, `20·20·80=32000`. Sum = **806400** = packed parent. Offsets `{0,102400,128000,134400,646400,774400}`.
BF16 (elem_bytes=2): each doubles → **1612800** = BF16 packed parent.
Delta vs old 705600 = `(80·80+40·40+20·20)·(16−4)·1 = 8400·12 = 100800` (INT8), `201600` (BF16) — exactly the reported `packed−expected`.

yolov8 detect: bbox 64ch → `align_channels_16(64)=64`; class 80ch → 80. Every head already a 16-multiple → per-seg bytes, offsets, total **byte-identical** to today. Provable no-op (host EDIT 2 and backend EDITs 3-5).

`validate_segment_size_contract` (`:4307`) sum==total holds; `runtime_total_matches_expected_or_aligned` (`:4501/4616/4745`) equality branch passes; error at `:5251` clears.

## 4. Read-side consistency (proof both sides now address 806400)

- Host: `input_expected_bytes`=806400, `input_segment_sizes`={102400,25600,6400,512000,128000,32000}; packed-reconstruct (`:4892`) copies these at cumulative offsets, producing/validating the C16 parent.
- Backend storage: EDIT 3 → `data_offset` accumulates `{0,102400,128000,134400,646400,774400}` = host offsets exactly.
- Backend read: BF16 op (EDIT 4) steps bbox by 16, class by 80 from those offsets; INT8 reader (EDIT 5) `aligned_tensor_channels=16/80`. Both walk the identical 806400 C16 parent. Confirmed the only reads keyed to the old stride were `:986/998/1028` (BF16) and `tesselatedaccess.cpp:116`-derived (INT8).

No host-only desync remains, satisfying "keep backend READ side consistent."

## 5. Idiom check

v2 (`genericboxdecode_v2/gstneatboxdecode.cpp`):
```cpp
352  int align_channels_16(int channels) { if (channels<=0){return 0;} return ((channels+15)/16)*16; }
385  const int aligned_depth = align_channels_16(depth);   // inside compute_tensor_expected_bytes
394  safe_size_mul(bytes, (std::size_t)aligned_depth, &bytes);
1745 compute_tensor_expected_bytes(tensor_w, tensor_h, tensor_d, cfg.data_type[i], &tensor_bytes);
```
The host patch is the same skeleton — `shape_desc_whd_from_layout` → `align_channels_16(d)` → chained `safe_size_mul` → `safe_size_add` into running total — reading as the v1 twin v2 was always meant to have. It deliberately pads **bytes only** (never element counts), exactly like v2 (which has no element-count function). Backend reuses its own ubiquitous `align_channels()` (`boxdecode.cpp:498` `const std::size_t c16 = align_channels(tensor.channels);` etc.) — no new arithmetic on either side.

## 6. Deploy + on-board validation

Rebuild BOTH .so (shared backend means the backend edit lands for the plugin):
```
cmake --build internals_beta02_sync/build-required-gst --target neatobjectdecode simaai_genboxdecode_v2
```
- Plugin: `neatobjectdecode` → `libneatobjectdecode.so` (CMake `add_library(${PROJECT_NAME} SHARED ...)`, `:98-100`). Hot-swap by copying to the board GST plugins dir on `GST_PLUGIN_PATH` (same NFS-copy + path-override mechanism).
- Backend: `simaai_genboxdecode_v2` → `libsimaai_genboxdecode_v2.so` (`genericboxdecode_v2/CMakeLists.txt:102`). Copy to NFS runtime dir + `LD_LIBRARY_PATH` override (proven hot-swap path).

Validate:
1. yolo26 INT8 DIRECT → contract total 806400, decode **2 persons + tie**.
2. yolo26 BF16 DIRECT → total 1612800, 2 persons + tie 0.82/0.91.
3. yolov8 DIRECT → unchanged (provable no-op all 5 edits).
4. yolo26 TESS → unchanged (separate TesselatedTensor blocked-layout path; sizer change is C16-correct and tess parent is already C16; BF16 grouped-role normalizer is explicitly disabled for yolo26 at `gstneatboxdecode.cpp:1667`, so padded `input_segment_sizes` cannot corrupt it).
5. Debug fallback: `SIMA_BOXDECODE_DISABLE_YOLO26_DIRECT_FAST=1` (`:957`) forces the generic INT8/tess reader to cross-check EDIT 5 independently of the BF16 op.

## 7. Still removable

- The full `compute_tensor_expected_bytes` port is **not** needed — it keys size off `dtype` alone (`dtype_item_size_bytes`) and would drop v1's `source_bf16_input`→2U override (live where `source_bf16_input` is true but `data_type` keeps a non-bf16 token). Adding it as an unused twin is dead idiom-theater a lead would reject; the single `align_channels_16(d)` step carries the idiom. **Removed.**
- Do **not** also align `build_tensor_element_counts_from_cfg` — that would pad `bf16_tensor_elements` (`:4374→:4380`) and mis-walk `maybe_log_bf16_input_stats`. Leaving it raw is the correct minimal scope.
- Backend EDITs 4 and 5 cannot be removed: EDIT 4 covers BF16-direct, EDIT 5 covers INT8-direct (the error is shown INT8-first); each path has an independent stride. EDIT 3 cannot be split (both config sites 2446/3166 build heads).

Files cited (all under `/home/docker/sima-cli/internals_beta02_sync/gst_plugins/`):
`neatobjectdecode/gstneatboxdecode.cpp`, `genericboxdecode_v2/gstneatboxdecode.cpp`, `vendor/sima-ai-a65-apps/genericboxdecode/src/boxdecode.cpp`, `vendor/sima-ai-a65-apps/genericboxdecode/src/tesselatedaccess.cpp`, `neatobjectdecode/CMakeLists.txt`, `genericboxdecode_v2/CMakeLists.txt`.
# SIMA Plugin JSON Truth Map (Frozen)

_Last updated: 2026-02-17_

This document freezes JSON-field usage for model-pipeline SIMA stages so removals are controlled and testable.

## 1. Freeze Scope and Plugin Matrix

### 1.1 In-scope (model pipeline)

- `simaaiprocesscvu` used as:
  - preproc stage (`kernel=preproc`)
  - quant/tess stage (`kernel=quanttess`)
  - post stage wrapper for detess/dequant (`kernel=detessdequant` in model sequence, backend element is still `simaaiprocesscvu` in `src/nodes/sima/DetessDequant.cpp`)
- `simaaiprocessmla`
- `simaaiboxdecode` (generic boxdecode)
- detess/dequant/tess payload stages in `tmp/gst/*`:
  - `detessdequant`
  - `detessellate`
  - `quantize`
  - `slicedequant`

### 1.2 Out-of-scope

Generic CVU app plugins and custom graph utilities are out of scope for this truth map, including but not limited to:

- `overlay`, `genericrender`, `argmax`, `nms*`, `groupkeypoints`, `distancecalculation`, `cv_process`, `cvresize`, `fastbev*`, `PyGast-plugins/*`, deprecated plugins, and custom app/test scaffolding.

### 1.3 Source trees covered

Static extraction covered both trees requested:

- `tmp/gst_plugins_source/gst/*`
- `tmp/gst/*`

Mirror check:

- Same in both trees: `genericboxdecode`, `detessdequant`, `detessellate`, `quantize`, `slicedequant`
- Diverged: `processcvu`, `processmla`

### 1.4 Plugin matrix

| Plugin / Stage | Currently required JSON keys (current code paths) | Inferable keys | Runtime properties (should not be static JSON) | MLA-only keys |
|---|---|---|---|---|
| `simaaiprocesscvu` (preproc/quanttess/post wrapper) | infer-first: wiring comes from `ConfigManager::getBuffers()` when available; JSON `input_buffers`/`output_memory_order` is fallback-only. | `input_width`/`input_height` can come from caps/runtime for graph 200/202; wiring arrays can be synthesized from CM metadata | runtime dims are renegotiated per frame; framework build no longer rewrites per-stage JSON wiring fields | quant/tess and post paths consume MLA tensor shape fields indirectly (`input_depth`, `slice_*`) |
| `simaaiprocessmla` | `simaai__params`, `model_path`, `batch_size`; `outputs[*]` is preferred but no longer hard-required when output shape fields can infer segment sizes | output segment sizes can be inferred from `output_*`/`slice_*` + dtype | `input_segment_name` optional runtime wiring aid; model path can be pack-derived | `outputs`, `data_type`, `output_*`, `slice_*`, quant params |
| `simaaiboxdecode` | Practically required by backend config loader: `buffers.output.size`, `memory.next_cpu`, `system.out_buf_queue`; class count resolution currently depends on implementation version | `num_classes` inferable from `input_depth`/`slice_depth` + `num_in_tensor` + `decode_type` (new source logic) | `buffers.input[*].name` rewired from upstream; thresholds/topk often runtime knobs | `input_*`, `slice_*`, `data_type`, `num_in_tensor` |
| `detessdequant` (legacy standalone GST element) | `simaai__params` plus parser fields: `orig_img_width`, `orig_img_height`, `frame_width`, `frame_height`, `num_in_tensor`, `next_cpu`, `no_of_outbuf`, `out_sz`, `input_*`, `slice_*`, `q_scale`, `q_zp` | none in plugin; higher-level shape inference exists in `StageConfig` | upstream buffer naming/CPU routing are runtime in wrapper flows | `input_*`, `slice_*`, `q_scale`, `q_zp`, `num_in_tensor` |
| `detessellate` payload (`tmp/gst/detessellate`) | accepts `de_tess.*` or root/static-contract equivalents (`input_*`, `slice_*`/`output_*`); `buffers.input[0].offset` optional (defaults 0) | tensor count and dims can be synthesized from manifest stage static fields | input name/path should be runtime-wired, not static | shape/slice contract |
| `quantize` payload (`tmp/gst/quantize`) | `quant_scale`, `zero_point` (JSON fallback) | input element count inferred from incoming buffer size | now consumes `q_scale`/`q_zp` metadata from upstream first, then JSON fallback | n/a (generic quantization) |
| `slicedequant` payload (`tmp/gst/slicedequant`) | reads section (`slice_dequant`/`simaai__params`/root) for shape fallback; quant JSON is fallback-only | quant first from runtime metadata; dims can be synthesized from manifest static contract | runtime metadata (`q_scale`/`q_zp`) is preferred transport | MLA output shape/quant contract |

### 1.5 Manifest context transport (current)

- Pipeline context type: `sima.model.manifest.v1`
- ABI-safe plugin access: `manifest_accessor_v1` in `include/gst/SimaPluginStaticManifestAbi.h`
- Stage lookup keys:
  - `element_name` (default)
  - `logical_stage_id` (from `stage-id` or `stage_id` pipeline property when set)
- Legacy `manifest_json` string remains for transition fallback.
- Framework node/model fragment builders now emit `stage-id=<element-name>` for SIMA model-path
  plugins so logical lookup is deterministic even when additional name transforms are applied.

## 2. Required Key Truth Map

### 2.1 Static extraction method

Extracted from explicit access points:

- `json["..."]`
- `contains("...")`
- parser helpers (`parser_get_int`, `parser_get_double_array`, etc.)

Key evidence locations:

- `tmp/gst/processcvu/gstsimaaiprocesscvu.cpp:1667`
- `tmp/gst/processmla/gstsimaaiprocessmla.cpp:579`
- `tmp/gst/genericboxdecode/payload.cpp:61`
- `tmp/gst/detessdequant/gstsimaaidetessdequant.cpp:276`
- `tmp/gst/detessellate/detessellate.cpp:361`
- `tmp/gst/quantize/payload.cpp:124`
- `tmp/gst/slicedequant/payload.cpp:57`

Runtime wiring/inference evidence:

- `src/nodes/sima/Preproc.cpp:245`
- `src/nodes/sima/DetessDequant.cpp:238`
- `src/nodes/sima/SimaBoxDecode.cpp:158`
- `src/pipeline/runtime/StageConfig.cpp:296`
- `src/pipeline/runtime/StageConfig.cpp:411`

### 2.2 Dynamic failure-injection method used

For registered plugins (`simaaiprocesscvu`, `simaaiprocessmla`, `simaaiboxdecode`, `detessdequant`):

- baseline: `gst-launch-1.0 ... num-buffers=0`
- mutate one key at a time (remove field)
- record startup/runtime failure behavior and message
- note: dynamic results reflect the currently registered runtime plugins on this host

For non-registered payload stages (`detessellate`, `quantize`, `slicedequant`):

- no direct GST element available in this runtime (`gst-inspect-1.0` reports missing)
- dynamic key-removal was therefore limited to static/source classification for this pass

### 2.3 Categorized map (frozen)

### `simaaiprocesscvu`

- hard required:
  - either CM buffer inference success or JSON fallback with:
    - `input_buffers`
    - `output_memory_order`
    - each input `memories[*].segment_name`
    - each input `memories[*].graph_input_name`
- soft/defaultable:
  - `graph_name`
  - `input_width`, `input_height` (optional JSON dims)
- duplicate/derived:
  - `input_buffers[*].name` (runtime-wired)
  - `sink_pad_tensor_index_map` from manifest context is now preferred for deterministic multi-input mapping
  - preproc dims from caps/runtime
- debug-only:
  - `debug` style fields not required for execution

Dynamic evidence:

- if CM inference fails and JSON wiring is missing -> startup fails with bus error
- if CM inference succeeds -> `input_buffers`/`output_memory_order` can be omitted
- if context indicates a model-managed multi-input stage and `sink_pad_tensor_index_map` is missing or ambiguous -> startup fails with bus error

### `simaaiprocessmla`

- hard required:
  - `simaai__params`
  - `model_path`
  - `batch_size`
  - `outputs[*].name`
  - `outputs[*].size`
  - `batch_sz_model` when `batch_size != 1`
- soft/defaultable:
  - `input_segment_name`
- duplicate/derived:
  - output dims/types can be inferred from model metadata in higher layers
- debug-only:
  - none critical

Dynamic evidence:

- remove `model_path` -> uncaught `nlohmann::json` type error (`ec=134`)
- `batch_size=2` and remove `batch_sz_model` -> uncaught type error (`ec=134`)
- remove `outputs` -> startup can pass with `num-buffers=0`, but runtime (`num-buffers=1`) hits SIGSEGV spin path

### `simaaiboxdecode`

- hard required (current runtime behavior):
  - `buffers.output.size`
  - `memory.next_cpu`
  - `system.out_buf_queue`
- soft/defaultable (depends on implementation version):
  - `num_classes` may be inferred/fallback in newer source, but current runtime may only warn
  - `decode_type` may degrade to type-mismatch warning in current runtime
- duplicate/derived:
  - `buffers.input[*].name` runtime-wired
  - `num_classes` derivable from tensor shape (`input_depth`/`slice_depth`) for known decode families
- debug-only:
  - `system.debug`, `system.dump_data`

Dynamic evidence:

- remove `memory.next_cpu` -> uncaught type error abort (`ec=134`)
- remove `system.out_buf_queue` -> uncaught type error abort (`ec=134`)
- remove `num_classes` -> non-fatal `JSON type mismatch` warning (`ec=0`)

### `detessdequant` (legacy standalone GST plugin)

- hard required:
  - `simaai__params` object
  - parser keys: `orig_img_width`, `orig_img_height`, `frame_width`, `frame_height`,
    `num_in_tensor`, `next_cpu`, `no_of_outbuf`, `out_sz`,
    `input_height`, `input_width`, `input_depth`,
    `slice_height`, `slice_width`, `slice_depth`,
    `q_scale`, `q_zp`
- soft/defaultable:
  - none in current code
- duplicate/derived:
  - some frame/original size fields are metadata-level and should be derivable
- debug-only:
  - `debug`, `dump_data`, `inpath`, `ibufname`, `n_request`, etc.

Dynamic evidence:

- remove `simaai__params` -> SIGSEGV spin path (timeout)
- remove `num_in_tensor` -> SIGSEGV spin path (timeout)

### `detessellate` payload

- hard required:
  - resolved input/slice tensor fields (from `de_tess.*` or root/static-contract synthesized keys)
- soft/defaultable:
  - `buffers.input[0].offset` (defaults to `0`)
  - `num_in_tensor` (derived from vector sizes when omitted)
- duplicate/derived:
  - shape vectors are derivable from manifest stage static tensors
- debug-only:
  - none

### `quantize` payload

- hard required:
  - `quant_scale`
  - `zero_point`
- soft/defaultable:
  - none in current code
- duplicate/derived:
  - tensor element count derived from input byte size
- debug-only:
  - none

### `slicedequant` payload

- hard required:
  - quant params for dequantization (`q_scale`, `q_zp`) resolved from runtime metadata or fallback config
  - tensor slice dims (`input_*`, `output_depth`/`slice_depth`) resolved from section/root/static-contract synthesis
- soft/defaultable:
  - scalar vs vector encoding for quant and shape keys
- duplicate/derived:
  - quant is preferred from runtime metadata; JSON remains fallback-only
- debug-only:
  - n/a

## 3. Controlled Removal Gate (from this frozen map)

Any JSON field removal must include all of:

1. Update this truth map classification for the field.
2. Add/update a failure-injection test case:
   - startup failure must be explicit (bus error), never crash.
3. For inferable/derived fields:
   - implement inference first,
   - retain caps/plugin option fallback,
   - keep JSON as last fallback only where still required.
4. Keep MLA-only JSON minimal:
   - only unresolved MLA metadata (shape/size/quant params) remains.
5. Keep strict CI gate enabled:
   - `unit_sima_plugin_manifest_strict_model_pipeline_test`
   - `unit_sima_plugin_manifest_strict_fallback_test`
   - and the corresponding Vulcan CI test lane in `.github/workflows/vulcan-ci.yml`.

## 4. Current risks found

- Several missing-key paths still abort with uncaught `nlohmann::json` exceptions or SIGSEGV instead of bus errors.
- `detessdequant` legacy path is crash-prone for missing parser keys.
- `slicedequant` ignores JSON entirely and uses compiled constants.
- Runtime/source divergence can reappear whenever rebuilt `.so` files are not copied from `tmp/gst/*/build` to `deps/gst-plugins`.

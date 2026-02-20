# 001 Model In 5 Minutes

Track: `A` (Foundations, chapters 001-006)

## Goal
Run a model end-to-end in two passes:
1. A generic synthetic input pass to understand API contracts.
2. A real image ResNet50-style classification pass with top-k output.

## Prereqs
- Built `neat` runtime and plugins.
- ResNet50 MPK (`.tar.gz`) available.
- C++ or Python tutorial runtime environment.

## Input Contract
- Model options use RGB image input.
- Input tensors/images must be contiguous `uint8` RGB.
- Chapter options intentionally stay simple:
  - `format = RGB`
  - `input_max_width/height/depth`
  - `channel_mean/channel_stddev`
- Note: this chapter does **not** explicitly set `preproc.normalize=true`; providing mean/std is enough to enable normalization in this model path.

## Output Contract
- Generic pass emits runtime output summary:
  - `output_kind`
  - `tensor_rank`
  - `field_count`
- Real pass emits top-k classification lines with class index and probability.
- Signature is runtime-populated, not placeholder values.

## Guided Path (Default)
```bash
./tutorial_v2_001_model_in_5_minutes --mpk tmp/resnet_50_mpk.tar.gz
python3 tutorials/001_model_in_5_minutes/model_in_5_minutes.py --mpk tmp/resnet_50_mpk.tar.gz
```

## Real Image Options
```bash
# use local image
./tutorial_v2_001_model_in_5_minutes --mpk tmp/resnet_50_mpk.tar.gz --image /path/to/image.jpg
python3 tutorials/001_model_in_5_minutes/model_in_5_minutes.py --mpk tmp/resnet_50_mpk.tar.gz --image /path/to/image.jpg

# force goldfish download and verify expected class
./tutorial_v2_001_model_in_5_minutes --mpk tmp/resnet_50_mpk.tar.gz --goldfish --expect-id 1 --min-prob 0.2
python3 tutorials/001_model_in_5_minutes/model_in_5_minutes.py --mpk tmp/resnet_50_mpk.tar.gz --goldfish --expect-id 1 --min-prob 0.2
```

## Contract/Task Print
```bash
./tutorial_v2_001_model_in_5_minutes --print-contract
python3 tutorials/001_model_in_5_minutes/model_in_5_minutes.py --print-contract
```

## Degraded Fallback Policy
- Default behavior: runtime failures fail hard.
- Opt-in degraded mode: `--allow-degraded` (ignored in strict mode).
- Degraded runs emit `degraded_mode=true`.

## Expected Output
- `CHECK strict_mode_visible: PASS`
- Generic summary line:
  - `[generic] output_kind=... tensor_rank=... field_count=...`
- Real summary lines:
  - `[real] classes=... top1=... prob=...`
  - `[real] topK: ...`
- `SIGNATURE ... output_kind=... tensor_rank=... field_count=...`
- `[OK] 001_model_in_5_minutes`

## Common Failures
- Missing MPK:
  - Symptom: `SKIP: missing ResNet50 MPK (pass --mpk)`
  - Fix: supply a valid `--mpk` path.
- Missing image:
  - Symptom: `[real] skipped: no image available ...`
  - Fix: pass `--image` or use `--goldfish`.
- Runtime/plugin failure:
  - Symptom: `[FAIL] ...` or `runtime_fallback: ...` in degraded mode.
  - Fix: validate plugin install, model compatibility, and runtime environment.

## Adaptation Tasks
1. Change `--size 224` to `--size 320`; verify contract checks and signature still make sense.
2. Compare synthetic generic pass vs real image pass and explain differences in top-k meaning.
3. Inject a bad `--mpk` and document strict vs degraded behavior.

## Real App Continuity
- This chapter now includes a real classification path, not just API surface checks.
- It is the base for Track A classifier service flows and later async/pipeline chapters.

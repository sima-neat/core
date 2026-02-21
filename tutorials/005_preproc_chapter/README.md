# 005 Preproc Chapter

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | preprocessing, normalization, image |

## Concept
Preprocessing is where raw input data is transformed into the exact tensor contract your model expects.

This chapter focuses on the preproc controls you use most in real deployments:
- `format`: declares input image layout/color order (`RGB`/`BGR`/`GRAY`) expected at ingress.
- `input_max_width`, `input_max_height`, `input_max_depth`: runtime bounds for accepted dynamic inputs.
- `preproc.input_width`, `preproc.input_height`: expected source dimensions entering preproc.
- `preproc.output_width`, `preproc.output_height`: tensor dimensions produced for inference.
- `preproc.normalize`: enables value normalization before inference.
- `preproc.channel_mean`, `preproc.channel_stddev`: per-channel normalization constants that should match model training assumptions.

Use-case guidance:
- Model output is unstable or low-confidence after deployment: verify `format` and `channel_mean/stddev` first.
- Multiple cameras/sources with different resolutions: set `input_max_*` and explicit preproc in/out dimensions for predictable behavior.
- Porting a model from another framework: mirror the training-time normalization recipe with `preproc.normalize` + channel stats.
- Isolating preprocessing issues: run/inspect `model.preprocess()` path and confirm shape/dtype before debugging inference/postproc.

Reference:
- [Model](/getting-started/programming-model/model)
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Configure `Model::Options` / `ModelOptions` with explicit preproc dimensions, format, and normalization policy.
2. Build the model and inspect preprocessing-stage behavior (group composition and tensor contract cues).
3. Execute a deterministic run path and verify resulting output/type signals.
4. Confirm expected behavior through `CHECK`, `SIGNATURE`, and `[OK]` markers.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_005_preproc_chapter
python3 tutorials/005_preproc_chapter/preproc_chapter.py
```

## Source Files
- C++: `tutorials/005_preproc_chapter/preproc_chapter.cpp`
- Python: `tutorials/005_preproc_chapter/preproc_chapter.py`

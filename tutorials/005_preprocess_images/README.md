# 005 Preprocess Images Before Inference

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | preprocessing, normalization, image |

## Concept

Configure the preprocessing stage — format, dimensions, and per-channel normalization — so raw image input becomes the exact tensor your model was trained on. Accurate preprocessing is usually the difference between a model that works and a model that looks broken.

This chapter focuses on the preproc controls you use most in real deployments:
- `format`: declares input image layout/color order (`RGB`/`BGR`/`GRAY`) expected at ingress.
- `input_max_width`, `input_max_height`, `input_max_depth`: runtime bounds for accepted dynamic inputs.
- `preproc.input_width`, `preproc.input_height`: expected source dimensions entering preproc.
- `preproc.output_width`, `preproc.output_height`: tensor dimensions produced for inference.
- `preproc.normalize`: enables value normalization before inference.
- `preproc.channel_mean`, `preproc.channel_stddev`: per-channel normalization constants that should match model training assumptions.

**Use-case guidance**
- Model output is unstable or low-confidence after deployment: verify `format` and `channel_mean` / `channel_stddev` first.
- Multiple cameras/sources with different resolutions: set `input_max_*` and explicit preproc in/out dimensions for predictable behavior.
- Porting a model from another framework: mirror the training-time normalization recipe with `preproc.normalize` + channel stats.
- Isolating preprocessing issues: run/inspect `model.preprocess()` path and confirm shape/dtype before debugging inference/postproc.

**APIs introduced**
- `pyneat.ModelOptions().preproc.*` — the preprocessing sub-struct with the fields listed above.
- `model.preprocess()` — retrieves the preprocessing node group so you can inspect it in isolation.

**Prerequisites**
Chapter 001. Chapter 004 for the rest of `ModelOptions`.

**References**
- [Model](/getting-started/programming-model/model)
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Configure `Model::Options` / `ModelOptions` with explicit preproc dimensions, format, and normalization policy.
2. Build the model and inspect preprocessing-stage behavior (group composition and tensor contract cues).
3. Execute a deterministic run path and verify resulting output/type signals.

## Run

Fetch the YOLOv8-s MPK once: `sima-cli modelzoo -v 2.0.0 get yolo_v8s`.

**Python:**
```bash
python3 share/sima-neat/tutorials/005_preprocess_images/preprocess_images.py \
  --mpk /path/to/yolo_v8s.tar.gz --size 224
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_005_preprocess_images \
  --mpk /path/to/yolo_v8s.tar.gz --size 224
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_005_preprocess_images
./build/tutorials-standalone/tutorial_005_preprocess_images \
  --mpk /path/to/yolo_v8s.tar.gz --size 224
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/005_preprocess_images/preprocess_images.cpp`
- Python: `tutorials/005_preprocess_images/preprocess_images.py`

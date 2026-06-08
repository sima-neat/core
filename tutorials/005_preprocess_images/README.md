# 005 Preprocess Images Before Inference

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | resnet_50 |
| Labels | preprocessing, normalization, image |

## Concept

Configure the preprocessing stage — format, dimensions, and per-channel normalization — so raw image input becomes the exact tensor your model was trained on. Accurate preprocessing is usually the difference between a model that works and a model that looks broken.

This chapter focuses on the preprocessing controls you use most in real deployments:
- `preprocess.color_convert.input_format`: declares input image layout/color order (`RGB`/`BGR`/`GRAY8`) expected at ingress.
- `preprocess.input_max_width`, `preprocess.input_max_height`, `preprocess.input_max_depth`: runtime bounds for accepted dynamic inputs.
- `preprocess.resize.width`, `preprocess.resize.height`: tensor dimensions produced for inference.
- `preprocess.resize.mode`: stretch, letterbox, or crop behavior.
- `preprocess.normalize.enable`: enables value normalization before inference.
- `preprocess.normalize.mean`, `preprocess.normalize.stddev`: per-channel normalization constants that should match model training assumptions.

**Use-case guidance**
- Model output is unstable or low-confidence after deployment: verify `preprocess.color_convert.input_format` and normalization stats first.
- Multiple cameras/sources with different resolutions: set `input_max_*` and explicit resize targets for predictable behavior.
- Porting a model from another framework: mirror the training-time normalization recipe with `preprocess.normalize.*` channel stats.
- Isolating preprocessing issues: run/inspect `model.preprocess()` path and confirm shape/dtype before debugging inference/postproc.

**APIs introduced**
- `pyneat.ModelOptions().preprocess.*` — the preprocessing sub-struct with the fields listed above.
- `model.preprocess()` — returns the preprocessing Graph fragment so you can inspect it in isolation.

**Prerequisites**
Chapter 001. Chapter 004 for the rest of `ModelOptions`.

**References**
- [Model](/getting-started/programming-model/model)
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Configure `Model::Options` / `ModelOptions` with explicit resize dimensions, format, and normalization policy.
2. Build the model and inspect preprocessing-stage behavior (Graph-fragment composition and tensor contract cues).
3. Execute a deterministic run path and verify resulting output/type signals.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/005_preprocess_images/preprocess_images.py \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_005_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_005_preprocess_images
./build/tutorials-standalone/tutorial_005_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/005_preprocess_images/preprocess_images.cpp`
- Python: `tutorials/005_preprocess_images/preprocess_images.py`

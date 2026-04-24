# 013 Classify Images with ResNet-50

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | resnet, classification, mpk |

## Concept

Load a ResNet-50 MPK with explicit preprocessing options, feed it a 224×224 image, and read the top-1 class. The classification twin of chapter 012 — same recipe, classifier output.

If chapter 012 teaches detection flow, this one teaches classification flow. It helps new users verify model loading, preprocessing assumptions, and output tensor shape before building application logic on top.

**APIs introduced**
- `pyneat.ModelOptions()` with classification-oriented fields (`.format`, `.preproc.input_*`, `.preproc.output_*`, `.preproc.normalize`, `.preproc.channel_mean/stddev`).
- `model.run(image, timeout_ms)` — direct NumPy image input, no explicit tensor wrap needed.

**When to use this**
- First classification model integration in a new environment.
- Validating normalization/format settings for classifier accuracy.
- Establishing baseline output contracts before label mapping / top-k handling.

**Prerequisites**
Chapter 001 (Model). Chapter 005 for preprocessing options.

**References**
- [Model](/getting-started/programming-model/model)
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Resolve ResNet MPK and deterministic classifier input tensor/image.
2. Configure model input/preproc options for classification.
3. Run inference and inspect output tensor rank/shape cues.
4. Confirm expected path with `CHECK`, `SIGNATURE`, and `[OK]`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run

Fetch the ResNet-50 MPK once: `sima-cli modelzoo -v 2.0.0 get resnet_50`.

**Python:**
```bash
python3 /usr/share/sima-neat/tutorials/013_classify_images_with_resnet50/classify_images_with_resnet50.py \
  --mpk /path/to/resnet_50.tar.gz --image /path/to/frame.jpg --size 224
```

**C++:**
```bash
/usr/lib/sima-neat/tutorials/tutorial_v2_013_classify_images_with_resnet50 \
  --mpk /path/to/resnet_50.tar.gz --image /path/to/frame.jpg
```

To compile this chapter's C++ source in your own project with a custom `CMakeLists.txt` (no `sima-neat-extras.deb` required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/013_classify_images_with_resnet50/classify_images_with_resnet50.cpp`
- Python: `tutorials/013_classify_images_with_resnet50/classify_images_with_resnet50.py`

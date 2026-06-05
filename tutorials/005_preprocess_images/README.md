# 005 Preprocess Images Before Inference

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | resnet_50 |
| Labels | preprocessing, normalization, image |

## Concept

Configure the preprocessing stage — format, dimensions, and per-channel normalization — so raw image input becomes the exact tensor your model was trained on. Accurate preprocessing is usually the difference between a model that works and one that looks broken.

## Walkthrough

A compiled model expects its input in one precise shape and value range: a fixed color order, fixed dimensions, and the normalization recipe it was trained with. Preprocessing is the stage that takes a raw decoded image and turns it into exactly that tensor. Get it wrong and the model still runs — it just returns confident nonsense, which is why preprocessing is the first thing to verify when a deployed model "looks broken."

This chapter configures the preprocessing controls you reach for most — color `format`, input/output dimensions, and per-channel `mean`/`stddev` normalization — and then runs *just* the preprocessing step in isolation so you can inspect what it produced before any inference happens. By the end you will have declared a full preproc contract, attached it to a model, and confirmed the preprocessed tensor's shape and type.

### Configure the preprocessing contract {#step-configure-preproc}

These options declare the contract the preprocessing stage enforces. `format` (or `color_convert.input_format`) fixes the color order at ingress; the `input_max_*` fields bound the dynamic input the runtime will accept; the resize/output dimensions set the tensor size produced for inference; and `normalize` plus the per-channel `mean`/`stddev` constants apply the value scaling. The normalization constants must match the model's training-time recipe — mismatched stats are the most common cause of low-confidence output.

**C++:** Fields live under `Model::Options::preprocess` — `color_convert.input_format` takes a `PreprocessColorFormat` enum, `normalize.enable` is an `AutoFlag`, and `normalize.mean` / `normalize.stddev` are `std::array<float, 3>`.

**Python:** Fields live under `ModelOptions.preproc` — `format` is a string (`"RGB"`), `normalize` is a bool, and the constants are plain lists assigned to `channel_mean` / `channel_stddev`.

### Build the model {#step-load-model}

Constructing the `Model` from the archive path plus the options binds your preprocessing contract to the loaded model. From here the model carries the preprocessing definition with it, so any stage or run derived from it reuses the same recipe.

### Inspect preprocessing in isolation {#step-inspect-preproc}

Rather than running the whole model, this chapter exercises only the preprocessing step so you can confirm it produces the tensor you expect before debugging anything downstream.

**C++:** `stages::Preproc(frames, model)` runs the preprocessing step alone and returns the preprocessed `Tensor` directly — we read `pre.shape.size()` (rank) and `pre.dtype` to confirm the contract took effect.

**Python:** `model.preprocess()` returns the preprocessing `Graph` fragment so you can inspect its composition (`preproc_group.size()`); a subsequent `model.run([tensor])` then exercises the full path and reports the output `kind`.

## Run

Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

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

Expected output (the C++ build prints the preprocessed tensor's rank and dtype enum):

```text
preproc_rank=3
preproc_dtype=1
[OK] 005_preprocess_images
```

(The Python build prints `preproc_group_size=...` and `output_kind=...`.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/005_preprocess_images/preprocess_images.cpp`
- Python: `tutorials/005_preprocess_images/preprocess_images.py`

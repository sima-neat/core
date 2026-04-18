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

This tutorial loads a ResNet-50 MPK. You must supply one via `--mpk`. If the flag is omitted and no MPK is cached under `tmp/`, the tutorial prints a `SKIP` line and exits.

### Download the ResNet-50 MPK

Fetch `resnet_50.tar.gz` from the SiMa modelzoo once:

```bash
sima-cli modelzoo -v 2.0.0 get resnet_50
```

Note the absolute path to the downloaded `resnet_50.tar.gz`; you will pass it to every `--mpk` invocation below. For more on asset resolution (env vars, fallback paths), see [Assets and MPK Setup](/how-to/assets_mpk).

### eLxr SDK (C++)

From inside the paired NEAT eLxr SDK container shell, `dk` forwards execution to the DevKit while keeping paths consistent across the shared NFS workspace:

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
dk ./tutorial_v2_013_classify_images_with_resnet50 --mpk /absolute/path/to/resnet_50.tar.gz
```

### eLxr SDK (Python)

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
dk python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/013_classify_images_with_resnet50/classify_images_with_resnet50.py \
  --mpk /absolute/path/to/resnet_50.tar.gz
```

`dk` runs the script on the paired DevKit using the DevKit-side `pyneat` venv. No activation is required on the SDK side.

### DevKit (C++)

From a shell on the DevKit itself:

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
./tutorial_v2_013_classify_images_with_resnet50 --mpk /absolute/path/to/resnet_50.tar.gz
```

### DevKit (Python)

```bash
source ~/pyneat/bin/activate
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/013_classify_images_with_resnet50/classify_images_with_resnet50.py \
  --mpk /absolute/path/to/resnet_50.tar.gz
```

## Source Files
- C++: `tutorials/013_classify_images_with_resnet50/classify_images_with_resnet50.cpp`
- Python: `tutorials/013_classify_images_with_resnet50/classify_images_with_resnet50.py`

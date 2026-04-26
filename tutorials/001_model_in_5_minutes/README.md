# 001 Model in 5 Minutes

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Labels | model, mpk, inference, foundations |

## Concept
This tutorial teaches the quickest practical path to run inference with a compiled model in Neat.

A compiled model is a deployable model package (`.tar.gz`, often called an MPK) that Neat can load and execute on the target device. It contains the model artifacts and runtime metadata needed for inference. You provide input data, run inference, and consume model outputs.

After this chapter, you should understand the minimum end-to-end loop:
- Load a compiled model package.
- Prepare input data that matches model expectations.
- Run synchronous inference.
- Read and validate output behavior.

**References**
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Set up runtime inputs: parse CLI args, locate the compiled ResNet50 MPK, and prepare sample input data.
2. Build the minimal model execution path for one model and one input stream.
3. Run synchronous inference to keep behavior deterministic and easy to debug.
4. Inspect top-1 predictions and validation output (`CHECK`, `SIGNATURE`, `[OK]`).

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

From inside the paired Neat eLxr SDK container shell, `dk` forwards execution to the DevKit while keeping paths consistent across the shared NFS workspace:

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
dk ./tutorial_v2_001_model_in_5_minutes --mpk /absolute/path/to/resnet_50.tar.gz
```

### eLxr SDK (Python)

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
dk python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/001_model_in_5_minutes/model_in_5_minutes.py \
  --mpk /absolute/path/to/resnet_50.tar.gz
```

`dk` runs the script on the paired DevKit using the DevKit-side `pyneat` venv. No activation is required on the SDK side.

### DevKit (C++)

From a shell on the DevKit itself:

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
./tutorial_v2_001_model_in_5_minutes --mpk /absolute/path/to/resnet_50.tar.gz
```

### DevKit (Python)

```bash
source ~/pyneat/bin/activate
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/001_model_in_5_minutes/model_in_5_minutes.py \
  --mpk /absolute/path/to/resnet_50.tar.gz
```

## Source Files
- C++: `tutorials/001_model_in_5_minutes/model_in_5_minutes.cpp`
- Python: `tutorials/001_model_in_5_minutes/model_in_5_minutes.py`

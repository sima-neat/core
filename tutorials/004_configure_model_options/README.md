# 004 Configure Model Options

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | yolo_v8s |
| Labels | model-options, configuration, contracts |

## Concept

`ModelOptions` declares the runtime contract between your input data, the model pipeline stages, and output decoding. It is the first struct you reach for when moving past default behavior — one place to tune preprocessing, input bounds, and postprocessing together.

This chapter focuses on the options most teams use first:
- `preprocess.kind`, `preprocess.color_convert.*`: declare incoming data type (for example raw RGB/BGR image input).
- `preprocess.input_max_width`, `preprocess.input_max_height`, `preprocess.input_max_depth`: set dynamic input bounds for validation and runtime sizing.
- `preprocess.normalize.*`, `preprocess.resize.*`: control preprocessing behavior (normalization, channel stats, and resize policy overrides).
- `decode_type`, `score_threshold`, `nms_iou_threshold`, `top_k`: control detection-style postprocessing and filtering.
- `boxdecode_original_width`, `boxdecode_original_height`: provide original image geometry when postprocessing requires source-frame coordinates and no preprocess metadata is available.
- `name_suffix`, `upstream_name`: stabilize/clarify generated stage naming when composing bigger pipelines.

**Use-case guidance**
- Prototype classification quickly: set `preprocess.kind` + input max dimensions, keep postproc defaults minimal.
- Detection model bring-up (YOLO-style): set `decode_type` plus threshold/NMS/top-k options to shape final boxes.
- Mixed input sizes in one app: set `input_max_*` high enough for expected ranges to avoid runtime contract failures.
- Accuracy tuning after deployment: adjust `preprocess.normalize.enable`, `mean`, and `stddev` to match model training assumptions.
- Multi-model or hybrid pipelines: use `name_suffix` / `upstream_name` to keep pipeline graph naming explicit and debuggable.

**APIs introduced**
- `pyneat.ModelOptions()` with the fields listed above.
- `model.input_spec()`, `model.output_spec()`, `model.metadata()` — inspect the resolved contract after loading.

**Prerequisites**
Chapter 001.

**References**
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Build a `Model::Options` / `ModelOptions` config covering input, preproc, and postproc settings.
2. Instantiate a model with those options and inspect `input_spec()`, `output_spec()`, and metadata.
3. Run one deterministic inference path and observe how options influence runtime behavior.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/004_configure_model_options/configure_model_options.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_004_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_004_configure_model_options
./build/tutorials-standalone/tutorial_004_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

### Verbosity presets

Framework build/run messaging is controlled with `VerboseOptions` on `GraphOptions`, `Model::Options`, and `Model::RouteOptions`.

Current development default: `VerboseOptions::debug_all()`. Call `production()` or `quiet()` explicitly when you want less output.

| Preset | Intended use |
| --- | --- |
| `VerboseOptions::quiet()` | Suppress framework progress and detail output. |
| `VerboseOptions::production()` | Show clean phase progress only. |
| `VerboseOptions::debug_plugins()` | Keep production UX, but also surface plugin and GStreamer topics. |
| `VerboseOptions::debug_all()` | Force the full verbose/detail sweep across all topics. |

For runtime queue/throughput tuning, see [Tune Throughput and Queue Depth](/tutorials/015-tune-throughput-and-queues).

## Source Files
- C++: `tutorials/004_configure_model_options/configure_model_options.cpp`
- Python: `tutorials/004_configure_model_options/configure_model_options.py`

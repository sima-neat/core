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
- `format`, `media_type`: declare incoming data type (for example raw RGB/BGR image input).
- `input_max_width`, `input_max_height`, `input_max_depth`: set dynamic input bounds for validation and runtime sizing.
- `preproc.*`: control preprocessing behavior (normalization, channel stats, image-type and resize policy overrides).
- `decode_type`, `score_threshold`, `nms_iou_threshold`, `top_k`: control detection-style postprocessing and filtering.
- `original_width`, `original_height`: provide original image geometry when postprocessing requires source-frame coordinates.
- `name_suffix`, `upstream_name`: stabilize/clarify generated stage naming when composing bigger pipelines.

**Use-case guidance**
- Prototype classification quickly: set `format` + input max dimensions, keep postproc defaults minimal.
- Detection model bring-up (YOLO-style): set `decode_type` plus threshold/NMS/top-k options to shape final boxes.
- Mixed input sizes in one app: set `input_max_*` high enough for expected ranges to avoid runtime contract failures.
- Accuracy tuning after deployment: adjust `preproc.normalize`, `channel_mean`, `channel_stddev` to match model training assumptions.
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
  --mpk /tmp/yolo_v8s_mpk.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_004_configure_model_options \
  --mpk /tmp/yolo_v8s_mpk.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_004_configure_model_options
./build/tutorials-standalone/tutorial_004_configure_model_options \
  --mpk /tmp/yolo_v8s_mpk.tar.gz
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/004_configure_model_options/configure_model_options.cpp`
- Python: `tutorials/004_configure_model_options/configure_model_options.py`

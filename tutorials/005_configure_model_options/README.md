# 005 Configure Model Options

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | yolo_v8s |
| Labels | model-options, configuration, contracts |

## Concept

`ModelOptions` is the one struct that declares the contract between your input data, the model's pipeline stages, and its output decoding. It is the first place you reach for when moving past default behavior.

## Walkthrough

Chapter 001 loaded a model with sensible defaults. Real models — especially detection models like YOLOv8 — need you to *declare* their contract: what pixel format and size the input arrives in, how it should be normalized, and how raw network outputs become filtered boxes. `ModelOptions` groups all of that into a single struct you fill in before construction.

This chapter configures a YOLOv8 model end to end, then inspects the contract the runtime resolved from those options. By the end you will have set input, preprocessing, and postprocessing knobs, read back the resolved `input_specs()`/`output_specs()`/`metadata`, and run one deterministic frame through the configured model.

### Declare input and preprocessing {#step-set-input-preproc}

The first block describes what a frame looks like and how to prepare it for the network. `format` (`BGR` here) and the `input_max_width`/`height`/`depth` bounds set the input contract the runtime validates against and sizes buffers for. The normalization fields supply the per-channel mean and standard deviation the model was trained with, so raw pixels are scaled into the range the network expects.

**C++:** Fields live under `opt.preprocess.*`: `kind = InputKind::Image`, `color_convert.input_format = PreprocessColorFormat::BGR`, and `normalize.enable = AutoFlag::On` with `mean`/`stddev` as `std::array<float, 3>`.

**Python:** Fields live under `opt.preprocess.*`: `kind = pyneat.InputKind.Image`, `color_convert.input_format = pyneat.PreprocessColorFormat.BGR`, and `normalize.enable = pyneat.AutoFlag.On` with `mean`/`stddev` lists.

### Declare postprocessing {#step-set-postproc}

The second block shapes the detector's output. `decode_type` selects the YOLOv8 box-decode path, and `score_threshold`, `nms_iou_threshold`, and `top_k` filter the raw detections — dropping low-confidence boxes, merging overlapping ones, and capping how many survive. `boxdecode_original_width`/`boxdecode_original_height` give the decoder the source-frame geometry it needs to map normalized coordinates back to pixels, and `name_suffix` stabilizes the generated stage names so the pipeline graph stays readable when composed with others.

**C++:** `decode_type = BoxDecodeType::YoloV8`; the geometry fields are `boxdecode_original_width`/`boxdecode_original_height`.

**Python:** `decode_type = pyneat.BoxDecodeType.YoloV8`; the geometry fields are `boxdecode_original_width`/`boxdecode_original_height`.

### Load and inspect the resolved contract {#step-load-and-inspect}

Constructing the `Model` with these options resolves the contract against the archive. We then read it back: `input_specs()` and `output_specs()` report the negotiated tensor constraints, and `metadata()` exposes the key/value contract baked into the archive. Inspecting these after load confirms the runtime accepted your options and tells you the concrete shapes you will be working with.

**C++:** The specs are `TensorConstraint` values; we print the concrete shape.

**Python:** We print the shapes from `input_specs()[0]` and `output_specs()[0]`, plus `len(model.metadata())`.

### Run one frame {#step-run-inference}

Finally we synthesize one `640×640` BGR frame and run it through the configured model, confirming the whole contract executes end to end and printing how many outputs came back.

**C++:** The frame is a `cv::Mat`; `run()` returns a `TensorList` whose `size()` we print as `outputs=`.

**Python:** The frame is wrapped as a `Tensor` via `Tensor.from_numpy(...)`; `run()` returns a `TensorList`, so we print its length.

## Run

Run it and you should see the resolved spec shapes, metadata key count, and the output tally. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

**Python:**
```bash
python3 share/sima-neat/tutorials/005_configure_model_options/configure_model_options.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_005_configure_model_options
./build/tutorials-standalone/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

Expected output (shape and key counts depend on the model archive; the C++ build prints the detailed spec lines and `outputs=`, the Python build prints shapes and `output_count=`):

```text
input_specs[0]: shape=[640,640,3]
output_specs[0]: shape=[]
metadata_keys=8
outputs=1
[OK] 005_configure_model_options
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

For runtime queue/throughput tuning, see [Tune Throughput and Queue Depth](/tutorials/tune-throughput-and-queues).

## Source Files
- C++: `tutorials/005_configure_model_options/configure_model_options.cpp`
- Python: `tutorials/005_configure_model_options/configure_model_options.py`

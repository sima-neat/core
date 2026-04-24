# 007 Plug a Model Into Your Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Labels | session, composition, patterns |

## Concept

Three ways to drop a model into a `Session` — direct node composition, `model.session()`, and `model.session(options)` — so you know which pattern fits which context. All three produce a runnable session; they differ in explicitness and control.

The three composition patterns shown:
- **Direct session**: add `Input` + `Output` nodes yourself with `session.add(...)`. Most explicit; useful when you need full control over the wiring.
- **Model-default session**: `model.session()` injects the model's default pipeline group with sensible defaults. Shortest path for most cases.
- **Model-attached session**: `model.session(ModelSessionOptions)` controls appsrc/appsink inclusion and stage naming when attaching model groups into larger pipelines.

Why this matters:
- Teams often start with direct sessions for clarity, then move to model-backed composition as systems scale.
- `ModelSessionOptions` keeps graph wiring explicit in multi-camera or multi-model deployments.
- Consistent naming (`upstream_name`, `name_suffix`, `buffer_name`) improves diagnostics and backend graph readability.

**APIs introduced**
- `model.session()` — inject the default model pipeline.
- `pyneat.ModelSessionOptions()` + `model.session(opts)` — attach with explicit options.
- `session.add(group_or_node)` — what all three patterns reduce to underneath.

**Prerequisites**
Chapter 001 (Model). Chapter 002 or 003 (Session basics).

**References**
- [Session](/getting-started/programming-model/session)
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Build a minimal direct session and validate it can run end-to-end.
2. Construct model-backed session variants (`model.session()` and `model.session(options)`).
3. Compare generated backend graphs with `--print-gst` to understand composition differences.

## Run

Fetch the YOLOv8-s MPK once: `sima-cli modelzoo -v 2.0.0 get yolo_v8s`.

**Python:**
```bash
python3 share/sima-neat/tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.py \
  --mpk /path/to/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_v2_007_plug_model_into_pipeline \
  --mpk /path/to/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_v2_007_plug_model_into_pipeline
./build/tutorials-standalone/tutorial_v2_007_plug_model_into_pipeline \
  --mpk /path/to/yolo_v8s.tar.gz
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.cpp`
- Python: `tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.py`

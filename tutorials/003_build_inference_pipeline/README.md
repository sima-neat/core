# 003 Build an Inference Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Labels | session, build, run, pipeline |

## Concept

Compose a `Session` by hand — input node, output node, no model — and run one frame through it. See the pipeline primitives in isolation before a model is added to the picture.

A `Session` is where you define pipeline structure by adding nodes and node groups in order. It is not a one-off inference call; it is a reusable runtime graph definition that can be built once and executed many times.

`build(...)` turns that definition into a runnable `Run` handle — the transition from "graph description" to "executable runtime":
- Resolves the added nodes/groups into a concrete pipeline.
- Validates input/output contracts for the selected input type.
- Configures runtime behavior (sync vs async mode, output memory policy).
- Returns a `Run` object for push/pull calls.

**APIs introduced**
- `pyneat.Session()` — the composition entry point.
- `pyneat.InputOptions()`, `pyneat.nodes.input(opts)`, `pyneat.nodes.output()` — the most basic node pair.
- `session.build(tensor, pyneat.RunMode.Sync)` — materialize the pipeline.
- `run.run(tensor, timeout_ms)` — sync one-shot inference on the built pipeline.

**When to use this**
Learning the `Session` / `Run` lifecycle without a model in the loop, or building custom pipelines from primitives.

**Prerequisites**
Chapter 001.

**References**
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Create a minimal `Session` with explicit input and output nodes.
2. Build the session with a concrete sample input to materialize a runnable pipeline.
3. Execute one deterministic sync run to verify output contract behavior.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/003_build_inference_pipeline/build_inference_pipeline.py \
  --width 320 --height 240
```

**C++:**
```bash
./lib/sima-neat/tutorials/tutorial_v2_003_build_inference_pipeline \
  --width 320 --height 240
```

To compile this chapter's C++ source in your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/003_build_inference_pipeline/build_inference_pipeline.cpp`
- Python: `tutorials/003_build_inference_pipeline/build_inference_pipeline.py`

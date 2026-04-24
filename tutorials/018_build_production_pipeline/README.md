# 018 Build a Production-Ready Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Labels | production, reliability, deployment |

## Concept

Assemble a production-style run loop from the patterns earlier chapters taught: explicit `ModelOptions`, explicit `ModelSessionOptions`, explicit `RunOptions`, and one async push/pull loop. Not a full product framework — a reliable skeleton you can adapt.

The template makes three things explicit that defaults leave implicit:
- Input bounds on `ModelOptions` (`input_max_width/height/depth`) so contract failures surface at build time, not runtime.
- Stage naming (`name_suffix`) so diagnostics in a multi-model system stay readable.
- Queue policy on `RunOptions` (`queue_depth`, `overflow_policy=Block`, metrics on) so pipeline behavior under load is observable.

**APIs introduced**
- `pyneat.ModelOptions()` + `pyneat.ModelSessionOptions()` + `pyneat.RunOptions()` as a composed unit.
- `model.build(tensor, session_options, run_options)` — one-call path from `Model` to `Run`.
- `runner.push(tensor)` returning bool + `runner.pull(timeout_ms)` + `runner.close()` — the production loop.

**When to use this**
- Adapting an earlier tutorial into real deployment code.
- Establishing a consistent runtime skeleton across multiple models in the same app.
- As a starting point for apps/examples repo integrations.

**Prerequisites**
Chapters 002 (async), 004 (ModelOptions), 007 (ModelSessionOptions), 017 (RunOptions and queues).

**References**
- [Model](/getting-started/programming-model/model)
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Prepare deterministic input and shared runtime options for production-like behavior.
2. Execute model-backed blueprint when MPK exists.
3. Execute session fallback blueprint when model assets are unavailable.
4. Validate resiliency path with consistent `CHECK`, `SIGNATURE`, and `[OK]` outputs.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run

Fetch the YOLOv8-s MPK once: `sima-cli modelzoo -v 2.0.0 get yolo_v8s`.

**Python:**
```bash
python3 /usr/share/sima-neat/tutorials/018_build_production_pipeline/build_production_pipeline.py \
  --mpk /path/to/yolo_v8s.tar.gz --iters 4
```

**C++:**
```bash
/usr/lib/sima-neat/tutorials/tutorial_v2_018_build_production_pipeline \
  --mpk /path/to/yolo_v8s.tar.gz --iters 4
```

To compile this chapter's C++ source in your own project with a custom `CMakeLists.txt` (no `sima-neat-extras.deb` required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/018_build_production_pipeline/build_production_pipeline.cpp`
- Python: `tutorials/018_build_production_pipeline/build_production_pipeline.py`

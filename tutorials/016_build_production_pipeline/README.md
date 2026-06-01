# 016 Build a Production-Ready Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | resnet_50 |
| Labels | production, reliability, deployment |

## Concept

Assemble a production-style run loop from the patterns earlier chapters taught: explicit `ModelOptions`, explicit `ModelRouteOptions`, explicit `RunOptions`, and one async push/pull loop. Not a full product framework — a reliable skeleton you can adapt.

The template makes three things explicit that defaults leave implicit:
- Input bounds on `ModelOptions` (`input_max_width/height/depth`) so contract failures surface at build time, not runtime.
- Stage naming (`name_suffix`) so diagnostics in a multi-model system stay readable.
- Queue policy on `RunOptions` (`queue_depth`, `overflow_policy=Block`, metrics on) so pipeline behavior under load is observable.

**APIs introduced**
- `pyneat.ModelOptions()` + `pyneat.ModelRouteOptions()` + `pyneat.RunOptions()` as a composed unit.
- `model.build(tensor, route_options, run_options)` — one-call path from `Model` to `Run`.
- `runner.push(tensor)` returning bool + `runner.pull(timeout_ms)` + `runner.close()` — the production loop.

**When to use this**
- Adapting an earlier tutorial into real deployment code.
- Establishing a consistent runtime skeleton across multiple models in the same app.
- As a starting point for apps/examples repo integrations.

**Prerequisites**
Chapters 002 (async), 004 (ModelOptions), 007 (ModelRouteOptions), 015 (RunOptions and queues).

**References**
- [Model](/reference/programming-model/model)
- [Graph](/reference/programming-model/graph)
- [Graph](/reference/programming-model/graph)

## Learning Process
1. Prepare deterministic input and shared runtime options for production-like behavior.
2. Execute model-backed blueprint when model archive exists.
3. Execute Graph fallback blueprint when model assets are unavailable.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/016_build_production_pipeline/build_production_pipeline.py \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_016_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_016_build_production_pipeline
./build/tutorials-standalone/tutorial_016_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/016_build_production_pipeline/build_production_pipeline.cpp`
- Python: `tutorials/016_build_production_pipeline/build_production_pipeline.py`

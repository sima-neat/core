# 016 Build a Production-Ready Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | resnet_50 |
| Labels | production, reliability, deployment |

## Concept

Assemble a production-style run loop from the patterns earlier chapters taught — explicit model options, explicit route options, explicit run options, and one async push/pull loop. Not a full product framework; a reliable skeleton you can adapt.

## Walkthrough

This is the capstone chapter. Everything so far has been one concept at a time; here they come together into a single blueprint you can lift into real deployment code. The template's whole purpose is to make explicit the three things that defaults leave implicit: the model's input bounds (so contract violations fail at build time, not mid-stream), the stage naming (so diagnostics stay readable when several models share a process), and the queue policy (so behavior under load is observable rather than mysterious).

The shape is: configure run options, configure and load the model, build a runner, then drive it with a bounded async loop. By the end you will have a `Runner` executing an async pipeline with production defaults and a push/pull loop that counts successful outputs — the runtime skeleton you would standardize across multiple models in the same application.

### Configure the run options {#step-configure-run-options}

These are the production runtime defaults. `queue_depth = 8` gives a small bounded buffer; `overflow_policy = Block` makes the producer wait rather than silently drop frames (the safe choice when you care about loss); `output_memory = Owned` ensures returned tensors survive past the pull; and `enable_metrics = true` turns on the counters you would scrape in a real deployment. Setting these explicitly — instead of relying on defaults — is what makes behavior under load predictable.

### Configure and load the model {#step-configure-model}

Here we make the input contract explicit on the model. Setting `input_max_width/height/depth` to the frame's dimensions means a mismatched input fails at build time with a clear contract error, rather than producing a confusing runtime failure later. `name_suffix = "_prod"` tags this model's stages so they're identifiable in diagnostics across a multi-model app. We then construct the `Model` from the archive path and these options.

**C++:** `Model::Options` also spells out the preprocessing the model expects — `InputKind::Image`, RGB color convert, and ImageNet normalization with `has_explicit_stats = true` — because the C++ path declares preprocessing up front rather than relying on archive defaults.

**Python:** `ModelOptions` sets the input bounds and suffix; the binding applies sensible preprocessing defaults, so no normalization block is needed here.

### Build the runner {#step-build-runner}

`ModelRouteOptions` (C++ `Model::RouteOptions`) selects which boundaries the route includes — `include_input` and `include_output` both true here — and carries the same `_prod` suffix so the route's elements match the model's naming. We then call `model.build(sample, route_options, run_options)`: the one-call path that takes a `Model` straight to a runnable `Runner`, forwarding both the route and run options into the underlying pipeline. The representative sample lets the build lock in negotiated shapes.

**C++:** The sample is a `TensorList` built with `Tensor::from_cv_mat(rgb, ..., TensorMemory::EV74)`, which places the input in device-appropriate memory.

**Python:** The sample is a list containing one `Tensor` from `Tensor.from_numpy(...)`.

### Drive the production loop {#step-run-loop}

This is the loop a real service runs. For each iteration we `push(...)` an input — checking the boolean return so a rejected push (under `Block`, a transient condition) is handled rather than miscounted — then `pull(...)` with a finite timeout and count the successful outputs. After the loop, `close()` tears the runner down cleanly. This push-bool / pull-with-timeout / explicit-close pattern is the reliable async skeleton; swap in your real inputs and output handling and the structure stays the same.

## Run

This chapter needs a model archive (`resnet_50`). Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

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

Expected output:

```text
outputs=4
[OK] 016_build_production_pipeline
```

(The Python build prints `iters=4 ok=4`.)

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/016_build_production_pipeline/build_production_pipeline.cpp`
- Python: `tutorials/016_build_production_pipeline/build_production_pipeline.py`

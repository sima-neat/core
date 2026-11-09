# 010 Send Multiple Inputs in One Sample

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | None |
| Labels | multi-input, samples, sync |

## Concept

Bundle several named tensors into one `Sample` and push it as a single inference event — the pattern for models that take more than one input, such as stereo frames, image + metadata, or sensor fusion.

## Walkthrough

Many real applications carry more than one input per inference event. Neat represents this as a **bundle sample**: a single `Sample` whose `fields` list holds multiple named tensor payloads, each addressable by a `port_name`. The runtime keeps the named fields together as one logical event, so `left` and `right` (or image and metadata) stay aligned through the pipeline.

This chapter builds a tensor-in/tensor-out graph, bundles two named float tensors, pushes the bundle through, and reads the fields back out. By the end you will have constructed a multi-field sample and confirmed both fields survived the round trip, in order.

### Configure a tensor input {#step-configure-tensor-input}

This graph consumes raw tensors, not decoded images, so the input contract is declared as a tensor payload (`FP32`, with `width`/`height`/`depth`) rather than a pixel format. That tells the input node to accept tensor buffers directly.

**C++:** Set `in.payload_type = PayloadType::Tensor`.

**Python:** Set `inp.payload_type = pyneat.PayloadType.Tensor` and `inp.format = pyneat.Format.FP32`.

### Build the graph and a seed run {#step-build-seed-run}

We compose the same minimal `Input -> Output` topology from chapter 004 and `build()` it into a `Run`. `build()` needs a representative sample to lock in negotiated shapes, so we pass a single seed tensor (all zeros) of the same shape the real fields will use. The seed is only for shape negotiation — the real data comes next.

### Build the bundle {#step-make-bundle}

Now assemble the multi-input event. Each input gets a name via `make_tensor_sample(port_name, tensor)`, and those named fields are what the model addresses by port. Here `left` is filled with `1.0` and `right` with `2.0` so you can tell them apart on the way out.

**C++:** `make_bundle_sample({...})` wraps the named fields into one `Sample` whose `kind` is `Bundle`.

**Python:** The list of named samples is passed directly to `run(...)`; pyneat builds the bundle envelope for you.

### Push the bundle and read it back {#step-push-and-read}

Finally, send the bundle through and inspect the result. The output is itself a bundle `Sample`, so we read `out.fields` rather than treating it as a single tensor — `out.fields.size()` should be `2`, and each field carries a tensor payload. The fields' `port_name` does not come back yet: it is empty.

**C++:** `run.run(Sample{bundle}, timeout_ms)` returns one `Sample`. Because the logical result has multiple fields, that returned `Sample` is itself a `Bundle` — so we check `out.kind == SampleKind::Bundle` and iterate `out.fields`, not `front()` (which would mean "first field inside the bundle").

**Python:** `run.run(fields, timeout_ms=...)` pushes the named samples and returns the output sample, as in C++; Python does not expose the fields' `port_name` yet (it is empty), so the example prints each field's tensor shape by position.

## Run

Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**. This chapter needs no model archive.

**Python:**
```bash
python3 share/sima-neat/tutorials/010_feed_multi_input_model/feed_multi_input_model.py \
  --width 64 --height 48
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_010_feed_multi_input_model
./build/tutorials-standalone/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

Expected output (C++):

```text
bundle_fields=2
  field=bundle has_tensor=yes
  field=bundle has_tensor=yes
[OK] 010_feed_multi_input_model
```

(The C++ build labels a field by its `port_name`, else its `stream_label`, else its position, so with `SIMA_NEAT_BACKEND=dataflow` the labels read `field_0` and `field_1`. The Python build prints `bundle_fields=2`, `field=0 shape=[48, 64, 3]`, and `field=1 shape=[48, 64, 3]`.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

How to apply the bundle pattern beyond this two-field demo.

### Naming and routing

- `port_name` is the wiring contract: it is how a multi-input model addresses each field. Match the names to the model's declared input ports.
- The output bundle preserves field structure, so you can match results back to the inputs by name rather than position.

### Inspecting output bundles

- Always branch on `kind` first: a multi-field result is `SampleKind.Bundle`, and reading it as a single tensor will not work.
- Check tensor presence per field (`field.tensor is not None` / `field.tensor.has_value()`) before touching the payload — a field may carry metadata rather than a tensor.

## Source Files
- C++: `tutorials/010_feed_multi_input_model/feed_multi_input_model.cpp`
- Python: `tutorials/010_feed_multi_input_model/feed_multi_input_model.py`

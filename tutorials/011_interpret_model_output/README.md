# 010 Read and Interpret Model Output

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | output, patterns, sink |

## Concept

Read back from `run.pull()` or `model.run()` safely. Every run returns a `Sample` — a small sum type that may be a tensor, a bundle of named fields, or both — so you classify it before touching the payload.

## Walkthrough

Before you optimize throughput or add complex graph logic, you need a stable, defensive way to read whatever a run hands back. The output is always a `Sample`, but its shape varies: it might be a single tensor, or a bundle of named fields (as in chapter 009). Reaching for `.tensor` on a bundle — or assuming a shape that isn't there — is the bug this chapter teaches you to avoid.

We build the same minimal sync graph as before, run one frame, then *inspect* the result methodically: its `kind`, whether a tensor is present, how many fields it has, and the tensor's rank. By the end you will have a reusable output-reading pattern that works for any model the runtime serves.

### Configure the input {#step-configure-input}

Declare the input contract — pixel `format`, `width`, `height`, `depth` — matching the frame we will push. This is the same boundary contract used throughout these chapters.

### Compose and build the graph {#step-compose-graph}

Wire an input node to an output node and `build()` into a sync `Run`, passing the frame so `build()` can negotiate concrete shapes. With no model in between, the output mirrors the input — which is exactly what makes this a clean place to study output structure.

### Run one frame {#step-run-frame}

Push one frame and pull one result synchronously. The single `run(...)` call is the one-frame shortcut; what it returns is the object we are here to dissect.

**C++:** `run.run(...)` returns a `TensorList` — for a single-tensor output that means one entry, which the next step inspects via `out.size()` and `out.front()`.

**Python:** `run.run(...)` returns a `Sample`, exposing `.kind`, `.tensor`, and `.fields` directly.

### Inspect the sample {#step-inspect-sample}

This is the lesson: read the structure before the payload. Check presence and kind first, then derive rank from the tensor's `shape`. Guarding each step (non-empty output, non-empty shape) is what makes an output reader robust against models whose shape you don't control.

**C++:** Report `out.size()` and tensor presence, throw if empty or if `out.front().shape` is empty, then print `rank` from `shape.size()`. (The `fields=0` line is a placeholder — `TensorList` does not carry the bundle field structure that the Python `Sample` does.)

**Python:** Print `sample.kind`, `sample.tensor is not None`, `len(sample.fields)`, and `len(sample.tensor.shape)` — the full sum-type surface in one place. For a tensor-kind result `fields` is empty and `.tensor` is present; for a bundle you would branch on `.kind` and read `.fields` instead.

## Run

Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**. This chapter needs no model archive.

**Python:**
```bash
python3 share/sima-neat/tutorials/011_interpret_model_output/interpret_model_output.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_011_interpret_model_output
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_011_interpret_model_output
./build/tutorials-standalone/tutorial_011_interpret_model_output
```

Expected output (C++):

```text
outputs=1 has_tensor=yes fields=0
rank=3
[OK] 011_interpret_model_output
```

The Python build prints the same facts through the `Sample` surface:

```text
sample_kind=SampleKind.Tensor
has_tensor=True
num_fields=0
output_rank=3
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

A defensive checklist for reading any model's output.

### Classify before you read

- Check `kind` first. A single-tensor result is `SampleKind.Tensor`; a multi-field result is `SampleKind.Bundle`.
- For tensor-kind, `tensor` is present and `fields` is empty. For bundle-kind, read `fields` and do not assume `tensor`.

### Validate the contract

- Confirm a tensor is present before dereferencing it.
- Confirm `shape` is non-empty before computing rank or indexing dimensions.
- Inspect `tensor.dtype` when the consumer expects a specific element type.

## Source Files
- C++: `tutorials/011_interpret_model_output/interpret_model_output.cpp`
- Python: `tutorials/011_interpret_model_output/interpret_model_output.py`

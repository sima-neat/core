# 008 Pass NumPy Arrays to the Model

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | numpy, pytorch, tensor, io |

## Concept

Move data between NEAT tensors and the data structures Python users already have — NumPy arrays and PyTorch tensors — without rewriting your existing preprocessing or postprocessing stacks. Layout, dtype, and copy semantics handled in two calls.

If you are integrating NEAT into an existing Python inference stack, this is the interop surface you need. It prevents the common integration mistakes (wrong layout, silent dtype coercion, unexpected aliasing) before you build larger pipelines.

**APIs introduced**
- `pyneat.Tensor.from_numpy(array, copy=..., image_format=...)` — wrap a NumPy array as a NEAT tensor.
- `tensor.to_numpy(copy=...)` — materialize a NumPy view (or copy) of a NEAT tensor.
- `pyneat.Tensor.from_torch(tensor, copy=..., image_format=...)` — same surface for PyTorch.
- `tensor.to_torch(copy=...)` — round-trip back to PyTorch.
- `pyneat.PixelFormat.RGB` (and siblings) — explicit image-layout tag that controls interpretation.

**When to use this**
- Existing preprocessing in NumPy: keep that path, then hand off tensors to NEAT.
- Existing model/postprocessing in PyTorch: convert in/out cleanly without rewriting your whole stack.
- Interop debugging: use deterministic round-trip checks to confirm data integrity.

**Prerequisites**
Chapter 001.

**References**
- [Tensor and Sample](/getting-started/programming-model/core_types)
- [Input and Output](/getting-started/programming-model/io)

## Learning Process
1. Build deterministic tensor inputs in NumPy/PyTorch and C++ tensor storage.
2. Convert across boundaries (NumPy/PyTorch <-> NEAT tensor).
3. Verify round-trip shape/bytes/checksum behavior.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/008_pass_numpy_to_model/pass_numpy_to_model.py \
  --width 128 --height 96
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_008_pass_numpy_to_model \
  --width 128 --height 96
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_008_pass_numpy_to_model
./build/tutorials-standalone/tutorial_008_pass_numpy_to_model \
  --width 128 --height 96
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/008_pass_numpy_to_model/pass_numpy_to_model.cpp`
- Python: `tutorials/008_pass_numpy_to_model/pass_numpy_to_model.py`

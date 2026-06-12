# 009 Pass NumPy Arrays to the Model

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | numpy, pytorch, tensor, io |

## Concept

Move data between Neat tensors and the structures you already have — NumPy arrays, PyTorch tensors, or `cv::Mat` — controlling layout, dtype, and copy semantics so you can drop Neat into an existing inference stack.

## Walkthrough

If you are integrating Neat into an existing inference stack, this is the interop boundary you need: how host data becomes a Neat `Tensor`, and how a Neat `Tensor` becomes host data again. Getting it right up front prevents the classic integration bugs — wrong layout, silent dtype coercion, unexpected aliasing between the two worlds.

This is also where the two languages diverge most. Python users come from NumPy/PyTorch; C++ users come from OpenCV. The conversion *concepts* are identical, but the API names and types differ, so the per-language prose below matters. By the end you will have converted host data into a Neat tensor, inspected its payload without copying, and produced an owned copy that is safe to outlive the source buffer.

### Wrap host data as a tensor {#step-to-tensor}

The first move turns data you already hold into a Neat `Tensor`. You tag the image layout explicitly (`RGB`) so the runtime interprets the bytes correctly rather than guessing. `copy=True` (or the CPU memory choice in C++) decides whether the tensor owns its bytes or aliases the source — explicit ownership is the safe default when the source buffer may change or be freed.

**C++:** `simaai::neat::from_cv_mat(mat, ImageSpec::PixelFormat::RGB, TensorMemory::CPU)` wraps a `cv::Mat` into a CPU-backed tensor.

**Python:** `pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)` wraps an HWC `uint8` NumPy array.

### Inspect the payload {#step-map-and-inspect}

Once the data is a tensor, you can read it back. This is the round-trip half of interop: confirm shape and bytes survived the conversion before feeding anything downstream.

**C++:** `tensor.map_read()` returns a `Mapping` exposing a raw `data` pointer and `size_bytes`. It is a *view* into the tensor's storage — no copy — which is why the example can checksum the leading bytes directly.

**Python:** `tensor.to_numpy(copy=True)` materializes a NumPy array from the tensor; the example prints its `.shape` to confirm the HWC layout round-tripped intact.

### Own a copy {#step-own-a-copy}

Finally, produce data that is fully detached from the original source buffer — safe to keep after the input is gone. This is the copy you hand to long-lived consumers.

**C++:** `tensor.clone()` copies into fresh CPU-owned storage, independent of the `cv::Mat` it came from.

**Python:** the same idea is shown through PyTorch: `pyneat.Tensor.from_torch(t, copy=True, ...)` and `tensor.to_torch(copy=True)` round-trip through an owned PyTorch tensor. (Skipped gracefully if `torch` is not installed.)

## Run

Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**. This chapter needs no model archive.

**Python:**
```bash
python3 share/sima-neat/tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py \
  --width 128 --height 96
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_009_pass_numpy_to_model
./build/tutorials-standalone/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

Expected output (C++):

```text
tensor_rank=3
tensor_bytes=36864
head_checksum=4342
clone_bytes=36864
[OK] 009_pass_numpy_to_model
```

Expected output (Python, with `torch` installed):

```text
numpy_roundtrip_shape=(96, 128, 3)
torch_roundtrip_shape=(96, 128, 3)
```

(Without `torch`, the Python build prints `torch_roundtrip_skipped=True` instead of the torch line.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

The interop surface, summarized for quick reference once you move past the round-trip demo.

### Conversion API

- NumPy: `pyneat.Tensor.from_numpy(array, copy=..., image_format=...)` in; `tensor.to_numpy(copy=...)` out.
- PyTorch: `pyneat.Tensor.from_torch(tensor, copy=..., image_format=...)` in; `tensor.to_torch(copy=...)` out.
- OpenCV (C++): `simaai::neat::from_cv_mat(mat, pixel_format, memory)` in; `tensor.map_read()` for a zero-copy view; `tensor.clone()` for an owned copy.

### Copy vs view

- `copy=True` (Python) / `clone()` (C++) gives you data detached from the source — safe to keep after the source is freed or mutated.
- `copy=False` / `map_read()` gives you a view that aliases the source. Cheaper, but only valid while the source stays alive and unchanged.

### Layout and dtype

- Always pass an explicit `image_format` / `PixelFormat` for image data so layout is interpreted, not guessed.
- Neat does not silently coerce dtype — match the tensor dtype to the model's input contract before feeding it.

## Source Files
- C++: `tutorials/009_pass_numpy_to_model/pass_numpy_to_model.cpp`
- Python: `tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py`

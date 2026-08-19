# Tensors And Images

Choose between model-ready tensor transport and decoded-image transport before constructing input.
`model.info()` reports the archive's inference tensor contract in both modes. Preserve its input
contract when submitting model-ready tensors. In image mode, card-side preprocessing converts the
host's decoded image payload into that model ingress contract.

## Tensor Input

Tensor mode is the default. The host application performs required resize, color conversion,
normalization, and layout preparation before sending the model-ready tensor.
`ModelInfo.dtype` preserves the MPK contract spelling, so treat `FP32` and `FLOAT32` as aliases.

For C++, prefer `Tensor::from_vector()` when ownership simplicity matters:

```cpp
const auto& spec = model.info().inputs.at(0);
if (spec.dtype != "FP32" && spec.dtype != "FLOAT32") {
  throw std::runtime_error("example expects a float32 input");
}

std::vector<float> values(spec.size_bytes / sizeof(float), 0.0F);
pcie::Tensor input =
    pcie::Tensor::from_vector(std::move(values), spec.shape, spec.name);
```

Use `Tensor::from_external()` only when avoiding the additional caller-side copy matters. Supply a
shared owner that keeps the complete backing allocation alive until PCIe/GStreamer releases the
tensor. The backing element count describes the full allocation, and `byte_offset` selects a view
within it.

For Python, `Tensor.from_numpy(array)` is zero-copy by default and requires a C-contiguous array.
The tensor retains the NumPy owner. Use `copy=True` when the application needs an independent,
owned input:

```python
spec = model.info().inputs[0]
array = np.zeros(spec.shape, dtype=np.float32)
input_tensor = pcie.Tensor.from_numpy(
    array,
    copy=True,
    route_name=spec.name,
)
```

For multi-input models, pass one tensor per logical input in the order reported by `info().inputs`
and set each tensor's route name to the corresponding input name. Do not assume physical output
indices or memory offsets from list position alone; output `Tensor.route` carries routing metadata.

## Image Input

Image mode sends decoded pixels and enables configured card-side preprocessing. Set
`options.preprocess.kind` to `InputKind::Image` in C++ or `InputKind.Image` in Python.
The submitted image's dtype, format, and geometry can differ from `model.info().inputs`; the model
information describes the preprocessor output and MLA ingress, not the host image allocation. The
preprocessing options and submitted image metadata define the host-side image contract.

C++ applications with OpenCV can pass a `cv::Mat` directly to `run()` or `push()`. The overload
accepts non-empty 8-bit GRAY8 or BGR data and keeps or creates continuous backing storage as needed.

Python applications can use `run_image()`:

```python
outputs = model.run_image(
    bgr_image,
    timeout_ms=30000,
    format=pcie.PixelFormat.BGR,
)
```

Alternatively, construct a `Tensor` with `image_format` set. Never mix image tensors and ordinary
tensors in one submitted payload.

Keep the input media type and geometry stable after the first submission. The first payload
selects transport capacity; a later payload larger than that active capacity is rejected.

## Outputs

`run()` returns a `TensorList`; `pull()` returns an optional `TensorList`. Validate the number,
route names, dtype, shape, and byte size before interpreting storage. In Python, call
`output.to_numpy()` for a NumPy array backed by a copied Python-owned bytes buffer, or use
`run_numpy()` when direct NumPy output is appropriate. This output conversion copies; unlike
`Tensor.from_numpy(array, copy=False)`, it is not a zero-copy view of PCIe tensor storage.

`model.info().outputs` always describes the raw inference outputs recorded in the archive; runtime
postprocessing does not update `ModelInfo`. When boxdecode is enabled, runtime output is instead one
`UInt8` tensor with route name `BBOX`, not application-specific detection objects. Parse that tensor
according to the installed PCIe boxdecode tutorial rather than assuming a generic bounding-box
layout.

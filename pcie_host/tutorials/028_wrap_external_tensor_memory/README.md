# 028 Wrap External Tensor Memory

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, C++, tensor, external memory, zero-copy wrapping |

## Concept

`Tensor::from_external()` lets a C++ application wrap an existing contiguous
allocation instead of copying it into a new host-side tensor. The tensor keeps
a `std::shared_ptr<void>` owner, so its backing allocation remains alive while
PCIe and GStreamer use it. The application must also leave the bytes unchanged
until the corresponding inference result is pulled.

This avoids an additional host staging copy for a contiguous single input. It
is not end-to-end zero-copy: the PCIe transport still copies the payload into
card-owned memory.

## Walkthrough

The program creates three reusable input slots and submits eight synthetic
FP32 frames. A slot returns to the available queue only after the ordered
result for that slot is pulled.

### Inspect the model contract {#step-inspect-contract}

Construct the model and read `info().inputs` before allocating memory. This
tutorial uses the single-input YOLOv8s archive and verifies that its reported
dtype is FP32 and that its shape accounts for exactly `size_bytes` bytes.

An external view must match the corresponding `TensorInfo` dtype, shape, byte
size, and name. Do not infer these values from another model build.

### Wrap application-owned memory {#step-wrap-memory}

Each ring slot owns a `std::shared_ptr<std::vector<float>>`. The call to
`Tensor::from_external()` receives the base pointer, complete backing element
count, shared owner, model shape, and route name. Because the view is contiguous,
the PCIe host can wrap it directly instead of creating a staging allocation.

Keeping only a raw pointer is not sufficient. The shared owner is mandatory
because the transport can retain the tensor after `push()` returns.

### Build the model {#step-build-model}

Build after the model contract and ring allocations have been validated. The
example sets `max_inflight` to the ring size so the application and transport
have the same explicit bound.

### Submit and safely reuse the ring {#step-submit-ring}

Fill an available slot, call `push()`, and move that slot to the in-flight
queue. Do not modify or reuse its storage merely because `push()` returned.
The example calls `pull()` when no slot is available and returns the oldest
slot to the available queue only after its matching ordered result arrives.

Every accepted push is balanced by one pull, including the final drain. A
timeout closes the model rather than reusing memory whose request may still be
active.

## Run

Install the PCIe host package and download the tutorial bundle as described in
[Tutorial Setup](/tutorials/before-you-run). Download YOLOv8s into the extracted
PCIe extras root:

```bash
sima-cli modelzoo get yolo_v8s
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_028_wrap_external_tensor_memory
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_028_wrap_external_tensor_memory
./build/tutorials-standalone/tutorial_028_wrap_external_tensor_memory
```

The default is card 0 and queue 0. Pass `--card N` only when using another
card. A successful run prints:

```text
input=images
ring_slots=3
completed=8
[OK] 028_wrap_external_tensor_memory
```

## In Practice

The direct host wrapping path requires contiguous storage. A tensor with
non-contiguous strides is still accepted when its descriptor is valid, but the
host compacts it into a staging allocation. Multiple separately allocated
inputs are also packed into staging memory.

For a multi-input model, avoid that staging allocation only when all tensors
are consecutive views into one shared packed allocation. Submit them in the
order reported by `info().inputs`, and use each input's name and shape. For
example, when both inputs are FP32:

```cpp
const auto& first = info.inputs.at(0);
const auto& second = info.inputs.at(1);
const std::size_t first_count = first.size_bytes / sizeof(float);
const std::size_t second_count = second.size_bytes / sizeof(float);

auto packed =
    std::make_shared<std::vector<float>>(first_count + second_count);
pcie::Tensor input0 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, first.shape, first.name);
pcie::Tensor input1 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, second.shape, second.name,
    static_cast<std::int64_t>(first.size_bytes));

model.push({input0, input1});
```

This optimization removes only the host-side packing copy. PCIe still copies
the packed payload into card-owned transport memory before inference.

## Source Files

- `run_external_tensor_memory.cpp`

---
title: Preproc ROI Lists
description: Run Preproc on multiple runtime ROI windows across one or more images
sidebar_position: 7
---

# Preproc ROI Lists

Use a Preproc ROI list when your application already has one or more image windows and wants each window preprocessed exactly as a model input: resize or letterbox, color convert, normalize, quantize, and optionally tessellate.

This is the right tool for second-stage classifiers, detector cascades, trackers, or any flow where a batch of source images yields a dynamic list of crops.

## Minimal example

C++:

```cpp
#include <neat.h>
#include <opencv2/imgcodecs.hpp>

using namespace simaai::neat;

Model model("/path/to/model.tar.gz");

std::vector<cv::Mat> images = {
    cv::imread("/data/camera0.jpg", cv::IMREAD_COLOR),
    cv::imread("/data/camera1.jpg", cv::IMREAD_COLOR),
};

std::vector<PreprocessRoi> rois = {
    {0, 0, 0, 320, 240},
    {0, 120, 80, 160, 160},
    {1, -20, 30, 224, 224},
};

TensorList out = stages::Preproc(images, model, rois);

for (std::size_t i = 0; i < out.size(); ++i) {
  const auto& meta = out[i].semantic.preprocess;
  // out[i] is the preprocessed tensor for rois[i].
  // meta carries the ROI and inverse-coordinate breadcrumbs.
}
```

Python:

```python
import cv2
import pyneat

model = pyneat.Model("/path/to/model.tar.gz")

images = [
    cv2.imread("/data/camera0.jpg", cv2.IMREAD_COLOR),
    cv2.imread("/data/camera1.jpg", cv2.IMREAD_COLOR),
]

rois = [
    pyneat.PreprocessRoi(0, 0, 0, 320, 240),
    pyneat.PreprocessRoi(0, 120, 80, 160, 160),
    pyneat.PreprocessRoi(1, -20, 30, 224, 224),
]

out = pyneat.stages.preproc(
    images,
    model,
    rois=rois,
    image_format=pyneat.PixelFormat.BGR,
)

for i, tensor in enumerate(out):
    meta = tensor.semantic.preprocess
    # tensor is the preprocessed output for rois[i].
    # meta carries the ROI and inverse-coordinate breadcrumbs.
```

## Configure resize, letterbox, and normalization

ROI-list Preproc uses the same model preprocessing options as full-frame Preproc.

C++:

```cpp
Model::Options opt;
opt.preprocess.resize.enable = AutoFlag::On;
opt.preprocess.resize.width = 640;
opt.preprocess.resize.height = 640;
opt.preprocess.resize.mode = ResizeMode::Letterbox;
opt.preprocess.resize.pad_value = 114;
opt.preprocess.resize.scaling_type = "BILINEAR";
opt.preprocess.normalize.enable = AutoFlag::On;
opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
opt.preprocess.tessellate.enable = AutoFlag::Auto;

Model model("/path/to/model.tar.gz", opt);
TensorList out = stages::Preproc(images, model, rois);
```

Python:

```python
opt = pyneat.ModelOptions()
opt.preprocess.resize.enable = pyneat.AutoFlag.On
opt.preprocess.resize.width = 640
opt.preprocess.resize.height = 640
opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
opt.preprocess.resize.pad_value = 114
opt.preprocess.resize.scaling_type = "BILINEAR"
opt.preprocess.normalize.enable = pyneat.AutoFlag.On
opt.preprocess.normalize.mean = [0.0, 0.0, 0.0]
opt.preprocess.normalize.stddev = [1.0, 1.0, 1.0]
opt.preprocess.tessellate.enable = pyneat.AutoFlag.Auto

model = pyneat.Model("/path/to/model.tar.gz", opt)
out = pyneat.stages.preproc(
    images,
    model,
    rois=rois,
    image_format=pyneat.PixelFormat.BGR,
)
```

Supported `scaling_type` tokens are `BILINEAR`, `NEAREST_NEIGHBOUR`, `BICUBIC`, `INTERAREA`, and `NO_SCALING`. `NEAREST_NEIGHBOR` and `INTER_AREA` are accepted aliases.

## Batch semantics

| Input | Meaning |
| --- | --- |
| `images` | Source image batch. The vector/list must be non-empty when ROIs are supplied. Python accepts uint8 HW/HWC NumPy/Torch/`pyneat.Tensor` image inputs. |
| `rois` | Runtime ROI list. The output tensor order matches this vector order. |
| `PreprocessRoi::batch_index` | Selects which source image in `images` the ROI reads from. |
| `PreprocessRoi::x`, `y` | Signed source-frame top-left coordinate. Negative values are allowed. |
| `PreprocessRoi::width`, `height` | ROI dimensions. Both must be positive. |

All source images in a ROI-list call must have matching width, height, OpenCV type, and channel count. The stage API supports packed 8-bit RGB/BGR (`CV_8UC3`) and GRAY/GRAY8 (`CV_8UC1`) source images for ROI lists.

For Python, pass `image_format=pyneat.PixelFormat.BGR` for `cv2.imread` images, `RGB` for RGB arrays, or `GRAY8` for HW grayscale arrays. CHW tensors are rejected by `pyneat.stages.preproc`; convert them to HWC before calling the stage.

## Output semantics

`stages::Preproc(images, model, rois)` returns:

- `out.size() == rois.size()` for a valid non-empty request;
- output tensor `out[i]` produced from ROI `rois[i]`;
- the dtype and layout chosen by the model route, including BF16 or INT8 and dense or tessellated output;
- per-output `tensor.semantic.preprocess` metadata.

Each output's preprocess metadata includes the selected ROI, letterbox geometry, normalization/quantization/tessellation flags, and an affine transform that maps model/preprocessed coordinates back to the original source-frame coordinate system.

## Out-of-frame ROIs

ROIs may extend outside the source image:

```cpp
std::vector<PreprocessRoi> rois = {
    {0, -16, -16, 128, 128},
    {0, image.cols - 64, image.rows - 64, 128, 128},
};
```

The in-frame region is copied and the out-of-frame region is padded using the Preproc pad value. This keeps output shape stable and avoids special casing near image borders.

## Letterbox and aspect ratio

For `ResizeMode::Letterbox`, Preproc computes letterbox scale and padding per ROI. A tall ROI and a wide ROI may therefore produce different `scaled_width`, `scaled_height`, `pad_left`, and `pad_top` metadata even when they share the same target size.

Downstream code should read the metadata rather than recomputing padding from assumptions.

## Validation checklist

Before blaming inference output, verify:

- the image vector is non-empty;
- every image has the same size, OpenCV type, and channel count;
- Python inputs are uint8 HW/HWC images, not CHW tensors;
- each ROI has a valid `batch_index`;
- each ROI has positive `width` and `height`;
- the source image format is RGB, BGR, GRAY, or GRAY8 for ROI-list mode;
- resize mode and normalization match the model's training preprocessing;
- downstream coordinate consumers use `tensor.semantic.preprocess` metadata.

## Tests covering this behavior

The repo includes fast coverage for the user-facing path and functional ROI behavior:

- `preproc_roi_batch_functional_test`
- `preproc_roi_user_smoke_test`

These cover multiple images, multiple ROIs, out-of-frame padding, resize/letterbox behavior, normalized outputs, and dense/tessellated BF16/INT8 routes.

## See also

- [Preproc Node](/reference/nodes/preproc)
- [Preprocess Images Before Inference](/tutorials/005-preprocess-images)
- [Data Formats](/reference/data_formats)
- [BoxDecode Decode Types](/reference/boxdecode_decode_types)

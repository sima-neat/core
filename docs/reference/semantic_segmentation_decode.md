---
title: Semantic segmentation decode
description: Turn per-class scores into a resized class map on the EV74 with SimaResizeArgmax or the Model semantic_segmentation options
sidebar_position: 9
---

# Semantic segmentation decode

`SimaResizeArgmax` turns a segmentation model's per-class scores into a class map
at the size you choose. For each output pixel it bilinearly interpolates every
class score and returns the class with the largest score. This is the decode that
SegFormer and similar models need after the MLA: `interpolate(mode="bilinear")` to
the target size, then `argmax` over the classes. It gives the same class map as
Hugging Face `SegformerImageProcessor.post_process_semantic_segmentation` with
`target_sizes`, and it never stores the resized scores.

The decode runs on the EV74. There is no A65 implementation and no fallback.

## Decode a model's output

Set `Model::Options::semantic_segmentation` to replace the model's dequantize or cast
postprocess with the decode. The node then reads the MLA scores in place,
including MLA outputs whose stored channels are padded, and the model returns one
class map instead of the scores.

```cpp
#include <neat.h>

using namespace simaai::neat;

Model::Options options;
options.semantic_segmentation.enable = true;
options.semantic_segmentation.output_width = 1280; // camera frame size
options.semantic_segmentation.output_height = 720;
Model model("segformer_b0_ade_512_mpk.tar.gz", options);
```

```python
import pyneat

options = pyneat.ModelOptions()
options.semantic_segmentation.enable = True
options.semantic_segmentation.output_width = 1280  # camera frame size
options.semantic_segmentation.output_height = 720
model = pyneat.Model("segformer_b0_ade_512_mpk.tar.gz", options)
```

Combine these options with the preprocessing options your model needs. The
model output is a single class-map tensor, `[N,H,W]`, and
`model.info().selection.selected_post_kind` is `"resize_argmax"`.

- Leave `output_width` and `output_height` both at 0 to use the model input size,
  or set both to the source frame size so that every source pixel gets a class.
- `align_corners = false`, the default, uses half-pixel centers like PyTorch and
  Hugging Face. Set it to `true` only for a model trained with corner-aligned
  interpolation.
- The model's postprocess must contain only dequantize or cast stages, and
  `decode_type` must stay unset. Otherwise the `Model` constructor throws
  `std::invalid_argument` (`ValueError` in Python).

## Use the node in a graph

`nodes::SimaResizeArgmax(...)` in C++ and `pyneat.nodes.sima_resize_argmax(options)`
in Python run the same decode on any tensor that matches the contract below. This
fragment decodes one score tensor, `scores`, into a 512 by 512 class map:

```cpp
SimaResizeArgmaxOptions decode;
decode.output_width = 512;
decode.output_height = 512;

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaResizeArgmax(decode));
graph.add(nodes::Output());
auto run = graph.build(TensorList{scores});
TensorList class_map = run.run(TensorList{scores}, 1000);
```

## Input and output contract

| Port | Tensor |
| --- | --- |
| `scores` | `[h,w,C]` or `[N,h,w,C]`, classes on the last axis; INT8, UINT8, BF16 or FP32; dense, or strided with a pixel pitch larger than `C` |
| `class_map` | `[H,W]` or `[N,H,W]`, dense; UINT8 when `C` is at most 256, otherwise INT32 |

- Integer scores are compared without dequantization. That gives the same classes
  as the dequantized scores when one positive scale covers every class. Scores
  with per-axis quantization or a scale that is not positive are rejected when
  the graph is built.
- Equal scores select the lowest class index. For BF16 and FP32 scores, an
  interpolated NaN wins, and the first NaN class wins.
- Stored padding channels, such as the extra channels of a padded MLA output,
  are never read as classes.
- MLA-tiled scores are rejected; detessellate them first. Scores with logical CHW
  or NCHW axes are rejected; the classes must be on the last axis.
- The DevKit must run EV74 firmware that includes this operation. Older firmware
  fails before the first frame is submitted.

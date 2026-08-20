# Model Options

Configure only the focused PCIe schema declared by `Model.h`. Regular Neat `Model::Options`, graph
options, and plugin-private configuration are not interchangeable with these types.

## Connection Options

`ConnectionOptions` identifies the card and the physical queue owned by one `Model`.

| Field | Default | Guidance |
| --- | --- | --- |
| `card_host` | empty | Explicit SSH/SCP address. Empty derives `10.0.<card_id>.2`. |
| `card_id` | `0` | Non-negative PCIe card/plugin index. |
| `user` | `sima` | Passwordless SSH user used for card startup. |
| `queue` | `0` | Physical co-processing queue, 0 through 3. |
| `max_inflight` | `10` | Accepted requests not yet returned; `0` uses plugin-managed depth. |
| `card_env` | empty | Additional card process environment assignments. |
| `card_gst_debug` | empty | Optional card-side GStreamer diagnostics. |
| `card_gst_debug_file` | empty | Optional card-side debug log path. |

Use default addressing for ordinary card 0 setups. Set `card_host` when management addressing is
non-standard. Treat `card_env` and GStreamer fields as explicit diagnostics/configuration, not
boilerplate to add to every application.

One active `Model` owns one queue. A Modalix EV74 exposes queues 0 through 3, so do not assign the
same queue to two active models.

## Tensor Mode

`ModelOptions` defaults to tensor transport. Keep this mode when the host prepares model-ready
input. Do not enable image preprocessing merely because the source began as an image.

## Image Preprocessing

For decoded-image transport, set the input kind and then configure only the transformations the
application requires:

```cpp
pcie::ModelOptions options;
options.preprocess.kind = pcie::InputKind::Image;
options.preprocess.color_convert.input_format = pcie::ColorFormat::BGR;
options.preprocess.resize.enable = pcie::AutoFlag::On;
options.preprocess.resize.mode = pcie::ResizeMode::Letterbox;
```

The available controls cover:

- resize enablement, stretch/letterbox/crop mode, padding, and scaling type;
- input/output color format;
- normalization preset or explicit three-channel mean and standard deviation;
- optional maximum dynamic input dimensions.

The public enums expose values used across several fields, but not every combination is supported:

- `preprocess.enable` must be `Auto` or `On`; `Off` is rejected.
- `preprocess.resize.width` and `height` must remain zero because inference sizing is model-derived.
- Color-conversion `input_format` can use the declared image formats, but `output_format` is limited
  to `Auto`, `RGB`, or `BGR`.
- When color conversion is `Off`, both formats must remain `Auto`.
- When normalization is `Off`, do not set a preset or explicit mean and standard deviation.
- Any preprocessing request requires `InputKind::Image`. Tensor mode accepts only its default,
  empty preprocessing configuration.

Prefer `Auto` and model-inferred sizing when the archive contains enough information. Do not set
`input_max_width`, `input_max_height`, or `input_max_depth` for a seedless model unless the
application requires an explicit input limit. These fields bound preprocessor input; they are not
a replacement for the model's inference shape.

## Object Decode

Box decode can be selected with `decode_type`, with optional implementation, score threshold, NMS
IoU threshold, top-k, and class-count fields. Supported decode families are the enum values exposed
by the installed release, including YOLO variants. Any boxdecode request requires
`preprocess.kind = InputKind::Image`; boxdecode is rejected in tensor mode.

Zero or `Unspecified` generally leaves the model/default value unchanged. Set values intentionally
and verify the returned tensor contract. Do not assume that enabling decode produces Python or C++
objects; it produces output tensors that the application must parse.

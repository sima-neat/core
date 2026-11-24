---
title: Use a USB Camera
description: Read MJPEG or YUYV frames from a USB (UVC) webcam with UsbCameraInput, forward or decode the JPEGs, or feed YUYV frames to a model and an H.264 viewer, with no device numbers or pipeline strings.
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# Use a USB Camera

`nodes::UsbCameraInput` reads USB (UVC) webcams. By default it captures MJPEG and outputs the camera's JPEG frames; setting `format` to `UsbCameraFormat::YUYV` captures uncompressed YUYV frames instead, as described in [Capture YUYV](#capture-yuyv). Pass the camera's stable ID. `UsbCameraInputOptions` uses the field names and defaults of `CameraInputOptions` (a [MIPI camera](./mipi-camera-input.md)), except that `camera_name` is required and `format` is a `UsbCameraFormat`, `MJPEG` by default.

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model

USB camera -> UsbCameraInput (YUYV) -+-> Preproc -> model
                                     `-> ColorConvert (NV12) -> VideoSender (H.264)
```

Most webcams send 4:2:2 JPEGs, which decode to NV16. `usb_camera_info()` reports the layout a camera's JPEGs decode to: use it as the `SimaDecode` `out_format`, and set the model's `preprocess.color_convert.input_format` to `PreprocessColorFormat::NV16` when it reports `NV16`, so frames reach the model with no conversion stage.

## What Neat supports

- UVC cameras connected to a Modalix DevKit USB port, on Platform 3.0.
- MJPEG capture with 4:2:0 or 4:2:2 chroma sampling.
- YUYV capture with an even width, BT.601 or BT.709 color and full or limited range, on Platform 3.0.1.
- The exact width, height and frame rate the camera lists for the selected format. Neat does not scale or pick a nearby mode.

H.264-only cameras, other raw formats such as NV12 or UYVY, depth or IR streams, and camera controls such as exposure are not part of this path.

## Find the camera ID

Each USB camera has a stable ID built from its USB port and its vendor and product IDs, in the same format libcamera uses:

```text
/base/pci@7000000000-3.1:1.0-046d:08e5
```

The ID stays the same across reboots and plug order. It changes when the camera moves to another USB port, so two identical cameras on different ports have different IDs. List the connected cameras:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
for (const std::string& id : simaai::neat::usb_camera_ids()) {
  std::cout << id << "\n";
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

print(pyneat.usb_camera_ids())
```

</CodeTab>
</CodeTabs>

To see which MJPEG and YUYV modes a camera offers, list its formats on the DevKit. The device number is only for this check; Neat finds the device from the ID.

<ShellCommand prompt="devkit">
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --list-formats-ext
</ShellCommand>

## Build the graph

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::UsbCameraInputOptions camera;
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5";
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.buffer_name = "camera0";

neat::Graph graph("usb_camera");
graph.add(neat::nodes::UsbCameraInput(camera));
graph.add(neat::nodes::Output());

neat::Run run = graph.build();
std::optional<neat::Sample> frame = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.UsbCameraInputOptions()
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5"
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.buffer_name = "camera0"

graph = pyneat.Graph("usb_camera")
graph.add(pyneat.nodes.usb_camera_input(camera))
graph.add(pyneat.nodes.output())

run = graph.build()
frame = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

The node outputs the camera's JPEG frames. To run a model on them, decode them with `SimaDecode` as shown in the next section.

`UsbCameraInput` opens the camera once while it creates the node, to check the mode and read the first frame's chroma sampling. Close other programs that use the camera first. `usb_camera_info(camera)` does the same check and returns the capture device and `decoded_format`, `NV12` or `NV16`, without creating a node.

When you need `decoded_format` as well, call `usb_camera_info` once and pass its result to `UsbCameraInput(camera, info)`, as the next section does. The node then reuses that check instead of opening the camera again, and only confirms from sysfs that the device still belongs to the camera. Pass the same options to both calls.

After a camera is unplugged or moved to another port, call `UsbCameraInput` again. Building a Graph created earlier fails, so it never opens a different camera that took the old device number.

## Forward the camera's JPEG

To send the camera's video to a viewer and also run a model on it, connect the node to a `VideoSender` passthrough, which sends the JPEGs as RTP/JPEG without re-encoding, and to a `SimaDecode` whose `out_format` is the camera's `decoded_format`:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once

neat::SimaDecodeOptions decode;
decode.type = neat::SimaDecodeType::MJPEG;
decode.out_format = info.decoded_format;  // NV12 or NV16
decode.dec_width = 1920;
decode.dec_height = 1080;
decode.dec_fps = 30;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::GraphLinkOptions video_link;
video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

neat::Graph source;
source.add(neat::nodes::UsbCameraInput(camera, info));  // reuses info, no second open
neat::Graph decoder;
decoder.add(neat::nodes::SimaDecode(decode));

neat::Graph graph("usb_camera_forward");
graph.connect(source, decoder);
graph.connect(decoder, detector);  // your model Graph
graph.connect(source, neat::nodes::groups::VideoSender(sender), video_link);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
info = pyneat.usb_camera_info(camera)  # opens the camera once

decode = pyneat.SimaDecodeOptions()
decode.type = pyneat.SimaDecodeType.MJPEG
decode.out_format = info.decoded_format  # NV12 or NV16
decode.dec_width = 1920
decode.dec_height = 1080
decode.dec_fps = 30

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

source = pyneat.Graph()
source.add(pyneat.nodes.usb_camera_input(camera, info))  # reuses info, no second open
decoder = pyneat.Graph()
decoder.add(pyneat.nodes.sima_decode(decode))

graph = pyneat.Graph("usb_camera_forward")
graph.connect(source, decoder)
graph.connect(decoder, detector)  # your model Graph
graph.connect(source, pyneat.groups.video_sender(sender), video_link)
```

</CodeTab>
</CodeTabs>

The camera captures into CMA buffers that the hardware decoder reads in place, so no frame is copied between capture and decode. The kernel UVC driver still copies the USB packets into each frame buffer. The RTP/JPEG limits in [Send Video](./video_sender.md) apply to the sender.

## Memory and dropped frames

| Option | Effect |
| --- | --- |
| `queue_depth`, `leaky_queue` | A small leaky queue after the camera drops stale frames when the graph falls behind. The camera's capture pool holds `queue_depth` + 6 buffers, at most 64, so `queue_depth` can be at most 58. |

Webcams often pad frames past the JPEG end marker, and some frames arrive incomplete or flagged as corrupt by the USB driver. `UsbCameraInput` drops the frames the driver flags before its output, so neither `VideoSender` passthrough nor the decoder receives them. The decoder decodes each frame up to its end marker and drops incomplete or corrupt frames instead of stopping. The decoder element counts them in its `dropped-count` property.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | Use one of the listed IDs. The ID changes when the camera moves to another port. |
| `no USB cameras are connected` | Check the cable and `lsusb`. |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | Pick a listed mode. Some cameras offer high resolutions only at lower frame rates. |
| `is in use by another process` | Stop the other program that has the camera open. |
| `sent no usable MJPEG frame` | The camera sends JPEGs Neat cannot decode, such as 4:4:4 sampling, or no frames at all. |
| `is no longer /dev/videoN` | The camera was unplugged, moved or renumbered after the Graph was created. Call `UsbCameraInput` again. |
| `is not at '/dev/videoN' as its UsbCameraInfo says` | The camera was unplugged, moved or renumbered after `usb_camera_info`, or the info is for another camera. Call `usb_camera_info` again and pass the new info. |
| `has no YUYV mode 640x480 at 30/1 fps; YUYV modes: ...` | Pick a listed YUYV mode. |
| `has no YUYV capture node` | The camera does not send YUYV. Capture MJPEG. |
| `YUYV requires an even width` | Pick a YUYV mode with an even width. |
| `adjusted the requested progressive YUYV mode` or `did not accept 30/1 fps` | The camera listed the mode but did not accept it. Pick another listed mode. |
| `has unsupported YUYV matrix/range/transfer metadata` or `has unsupported YUYV color primaries` | The camera reports a color space other than BT.601 or BT.709. Capture MJPEG. |
| `capture info format does not match the selected mode` | Pass the complete `UsbCameraInfo` that `usb_camera_info` returned. |
| `capture info format does not match opt.format` | The info was checked for another `format`. Call `usb_camera_info` with the options you pass to `UsbCameraInput`. |
| `capture info describes another mode` | The options differ from those passed to `usb_camera_info`. Pass the same options to both calls. |
| A YUYV Run fails when the camera starts streaming | Run `dmesg` on the DevKit. `Failed to submit URB 0 (-28)` (`ENOSPC`) means the USB controller cannot reserve the bandwidth the uncompressed mode needs. Pick a lower resolution or frame rate, stop other USB cameras or audio devices that stream at the same time, or capture MJPEG. |
| `YUYV requires explicit input color` | Set `PreprocOptions::input_color` to `info.color`. |
| `YUYV input does not support antialiased sampling`, `FP32 transforms require supported RGB/Gray tensor output`, `crop currently requires RGB, BGR or Gray input` or `ROI lists require RGB/BGR/Gray inputs/outputs` | The model's preprocessing needs an option YUYV input does not support. See [Run a model on YUYV frames](#run-a-model-on-yuyv-frames). |
| `Strict encoder ingress requires EncoderNv12 output from ColorConvert` | Put `ColorConvert` with `ImageSurfaceLayout::EncoderNv12` and NV12 output before the strict `VideoSender`. |
| `Strict encoder ingress requires an exact integer frame rate` | Capture a mode with an integer frame rate and set the encoder's `fps` to it. |
| `Graph JSON does not store PreprocOptions::input_color or ColorConvertOptions::input_colorimetry` | Build a graph that captures YUYV in code; `Graph::save` cannot store the camera's color. |

## Capture YUYV {#capture-yuyv}

Many UVC cameras also send uncompressed YUYV: packed 4:2:2 frames, which GStreamer calls `YUY2`. YUYV frames need no decoder; the model's preprocessing and `ColorConvert` read them in place from the capture buffers. Uncompressed frames need much more USB bandwidth than MJPEG, so cameras usually list YUYV at lower resolutions or frame rates.

Set `format` to `UsbCameraFormat::YUYV`, call `usb_camera_info` and pass its result to `UsbCameraInput`. Pick a YUYV mode the camera lists, with an even width. The returned `UsbCameraInfo` also describes the frames: `row_stride` is the row pitch in bytes, `storage_bytes` the size of each capture buffer, `color` the camera's matrix (BT.601 or BT.709) and range, and `colorimetry` the same interpretation as a GStreamer colorimetry string.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
camera.format = neat::UsbCameraFormat::YUYV;
camera.width = 640;
camera.height = 480;
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once
auto source = neat::nodes::UsbCameraInput(camera, info);         // reuses info
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera.format = pyneat.UsbCameraFormat.YUYV
camera.width = 640
camera.height = 480
info = pyneat.usb_camera_info(camera)  # opens the camera once
source = pyneat.nodes.usb_camera_input(camera, info)  # reuses info
```

</CodeTab>
</CodeTabs>

The node outputs `UInt8` frames in HWC layout with two bytes per pixel. Frames keep the camera's frame rate, including fractional rates such as 30000/1001, and odd heights are allowed. With the default `UsbCameraFormat::MJPEG`, the node captures MJPEG.

### Run a model on YUYV frames {#run-a-model-on-yuyv-frames}

Set the model's `preprocess.color_convert.input_format` to `PreprocessColorFormat::YUYV` and its maximum input size to the camera mode. YUYV frames do not say which matrix and range they use, so build the model route from a `Preproc` whose `input_color` is `info.color`, followed by the model's inference and postprocessing:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Model::Options options;
options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::YUYV;
options.preprocess.input_max_width = 640;
options.preprocess.input_max_height = 480;
neat::Model model("model.tar.gz", options);

neat::PreprocOptions preproc(model);
preproc.input_color = info.color;

neat::Graph graph("usb_camera_yuyv");
graph.add(neat::nodes::UsbCameraInput(camera, info));
graph.add(neat::nodes::Preproc(preproc));
graph.add(model.inference());
graph.add(model.postprocess());
graph.add(neat::nodes::Output());
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
options = pyneat.ModelOptions()
options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.YUYV
options.preprocess.input_max_width = 640
options.preprocess.input_max_height = 480
model = pyneat.Model("model.tar.gz", options)

preproc = pyneat.PreprocOptions(model)
preproc.input_color = info.color

graph = pyneat.Graph("usb_camera_yuyv")
graph.add(pyneat.nodes.usb_camera_input(camera, info))
graph.add(pyneat.nodes.preproc(preproc))
graph.add(model.inference())
graph.add(model.postprocess())
graph.add(pyneat.nodes.output())
```

</CodeTab>
</CodeTabs>

Model preprocessing on YUYV frames does not support:

- antialiased resize sampling (`ResizeInterpolation::BilinearAntialiased`);
- fixed source arithmetic: `rescale_factor` and the fixed normalization it selects, `quantization_step`, `PreprocFloatingConversion::NearestEven` and `PreprocColorConversionPlacement::BeforeResize`;
- `ResizeMode::Crop`;
- runtime ROI lists.

A model whose preprocessing needs one of these fails when the graph is built. Capture MJPEG instead, or convert the frames first with `ColorConvert` to `ImageFormat::RGB` (set its `color` to `info.color` and its `input_colorimetry` to `info.colorimetry`) and set the model's input format to `PreprocessColorFormat::RGB`.

### View YUYV frames as H.264 {#view-yuyv-frames-as-h264}

To send the camera's video to a viewer as well, branch before `Preproc` and link the viewer branch with `RealtimeLatestByStream`, so it never holds back the model. Convert the frames to NV12 in the encoder's layout with a `ColorConvert` whose `input_colorimetry` is `info.colorimetry`, then encode them with a strict H.264 `VideoSender`, whose `SimaEncodeOptions::input_policy` is `EncoderInputPolicy::RequireDirect`:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::ColorConvertOptions convert;
convert.output_format = neat::ImageFormat::NV12;
convert.surface = neat::ImageSurfaceLayout::EncoderNv12;
convert.color = info.color;
convert.color.output_siting = {neat::ImageChromaSiting::Cosited,
                               neat::ImageChromaSiting::Centered};
convert.input_colorimetry = info.colorimetry;

neat::SimaEncodeOptions encode;
encode.fps = 30;
encode.input_policy = neat::EncoderInputPolicy::RequireDirect;
auto sender = neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
sender.host = "127.0.0.1";

neat::Graph viewer("usb_camera_yuyv_viewer");
viewer.add(neat::nodes::ColorConvert(convert));
viewer.add(neat::nodes::groups::VideoSender(sender));
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
convert = pyneat.ColorConvertOptions()
convert.output_format = pyneat.ImageFormat.NV12
convert.surface = pyneat.ImageSurfaceLayout.EncoderNv12
convert.color = info.color
convert.color.output_siting = [pyneat.ImageChromaSiting.Cosited, pyneat.ImageChromaSiting.Centered]
convert.input_colorimetry = info.colorimetry

encode = pyneat.SimaEncodeOptions()
encode.fps = 30
encode.input_policy = pyneat.EncoderInputPolicy.RequireDirect
sender = pyneat.VideoSenderOptions.from_raw(encode)
sender.host = "127.0.0.1"

viewer = pyneat.Graph("usb_camera_yuyv_viewer")
viewer.add(pyneat.nodes.color_convert(convert))
viewer.add(pyneat.groups.video_sender(sender))
```

</CodeTab>
</CodeTabs>

The strict sender never converts or copies frames on the CPU, and it signals the camera's primaries, transfer, matrix and range in the H.264 stream. It needs even width and height and an integer frame rate equal to the encoder's `fps`; it rejects other input instead of rounding. See [Send Video](./video_sender.md#strict-encoder-input).

No frame is copied between capture and the model or the encoder; the kernel UVC driver still copies the USB packets into each frame buffer. `UsbCameraInput` drops the frames the driver flags as corrupt. The driver's byte count for each frame does not reach the graph, so an incomplete frame the driver does not flag passes through with the rest of its buffer left from an earlier frame.

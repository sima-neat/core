---
title: Use a USB Camera
description: Read MJPEG frames from a USB (UVC) webcam with UsbCameraInput and forward them as JPEG or decode them for a model, with no device numbers or pipeline strings.
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# Use a USB Camera

`nodes::UsbCameraInput` reads USB (UVC) webcams that send MJPEG and outputs the camera's JPEG frames. Pass the camera's stable ID. `UsbCameraInputOptions` uses the field names and defaults of `CameraInputOptions` (a [MIPI camera](./mipi-camera-input.md)), except that `camera_name` is required and there is no `format`.

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model
```

Most webcams send 4:2:2 JPEGs, which decode to NV16. `usb_camera_info()` reports the layout a camera's JPEGs decode to: use it as the `SimaDecode` `out_format`, and set the model's `preprocess.color_convert.input_format` to `PreprocessColorFormat::NV16` when it reports `NV16`, so frames reach the model with no conversion stage.

## What Neat supports

- UVC cameras connected to a Modalix DevKit USB port, on Platform 3.0.
- MJPEG capture with 4:2:0 or 4:2:2 chroma sampling.
- The exact width, height and frame rate the camera lists for MJPEG. Neat does not scale or pick a nearby mode.

H.264-only cameras, raw YUYV/NV12 capture, depth or IR streams (for example RealSense depth), and camera controls such as exposure are not part of this path.

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

To see which MJPEG modes a camera offers, list its formats on the DevKit. The device number is only for this check; Neat finds the device from the ID.

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
| `queue_depth`, `leaky_queue` | A small leaky queue after the camera drops stale frames when the graph falls behind. |

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

# 020 MIPI Camera to UDP H.264

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | camera, v4l2, mipi, h264, udp, streaming |

## Concept
This tutorial replaces the hand-rolled `gst-launch-1.0` pipeline used for
Modalix MIPI camera bring-up with a typed NEAT session. It wires a V4L2
post-ISP source into the SiMa H.264 encoder and emits RTP/H.264 over UDP to
a configurable host and port.

The equivalent raw GStreamer pipeline looks like this:

```bash
gst-launch-1.0 v4l2src device=/dev/video0out \
  ! 'video/x-raw,format=RGB,width=1920,height=1080' \
  ! videoconvert ! 'video/x-raw,format=NV12' \
  ! neatencoder enc-width=1920 enc-height=1080 enc-bitrate=4000 enc-level=4.2 \
  ! h264parse ! rtph264pay ! udpsink host=192.168.2.1 port=9000
```

The NEAT version adds up to **five node adds**:

1. `V4L2Input` — post-ISP capture from `/dev/video0out`
2. `Queue` — basic buffering for the live source
3. `VideoConvert` — RGB -> NV12 (negotiated from the encoder's caps)
4. `H264EncodeSima` — SiMa H.264 encoder (`neatencoder`)
5. `UdpH264OutputGroup` — `h264parse` + `rtph264pay` + `udpsink` in one group

This tutorial is intentionally minimal so that downstream work (for
example `sima-neat/apps#88`, Single-Camera Modalix MIPI OpenPose) can
plug an ML model in front of the encoder without rediscovering the
source-side wiring.

## Learning Process
1. Configure `V4L2InputOptions` with device, caps, and format.
2. Add a `VideoConvert` step so downstream caps negotiation can produce NV12.
3. Hand `H264EncodeSima` the encode width/height/fps and bitrate explicitly.
4. Collapse the H.264 parse + packetize + `udpsink` tail into a single
   `UdpH264OutputGroup` node.
5. Build a source-mode session (`session.build(run_opt)` with no input
   tensor) and let the live pipeline self-drive for a bounded duration.
6. Stop cleanly and print the SIGNATURE record.

## Board Prerequisites (external to NEAT)

When targeting a Modalix MIPI post-ISP device (for example `/dev/video0out`)
the following **must be running externally** before the NEAT pipeline starts:

1. **Device tree overlay** applied at U-Boot (one-time):
   ```bash
   setenv dtbos econ-imx678-csi-0-isp-0.dtbo
   saveenv
   boot
   ```
2. **Raw sensor keepalive** — keeps the MIPI sensor active and the ISP fed:
   ```bash
   gst-launch-1.0 v4l2src device=/dev/video0raw \
     ! 'video/x-bayer,format=rggb12le,width=1920,height=1080,framerate=30/1' \
     ! fakesink &
   ```
3. **ISP 3A controller** — auto-exposure, white balance, focus:
   ```bash
   sudo isp_3a.elf &
   ```

These are board-level concerns. `V4L2Input` captures from the post-ISP
V4L2 output; it does not manage sensor initialization, ISP control, or
DTBO selection.

## Host-Side Receiver

On the host machine that should display the stream, start a GStreamer UDP
receiver **before launching the tutorial on the board**:

```bash
gst-launch-1.0 udpsrc port=9000 \
  ! application/x-rtp,encoding-name=H264,payload=96 \
  ! rtpjitterbuffer \
  ! rtph264depay \
  ! video/x-h264,stream-format=byte-stream,alignment=au \
  ! avdec_h264 \
  ! fpsdisplaysink sync=0
```

Replace the port with whatever you pass via `--port`. Make sure the board
can reach the host IP passed via `--host` (typically the laptop's address
on the Ethernet interface the Modalix devkit is connected to).

## Troubleshooting
- If `/dev/video0out` times out after a previous NEAT or GStreamer run, first
  run `bash /usr/bin/fix_devkit_runtime.sh`, then restart the raw keepalive
  pipeline and `sudo isp_3a.elf`.
- If the stream still does not recover, reboot the board before treating
  the issue as a `V4L2Input` or encoder regression. Stale Modalix runtime
  state can survive process exit.
- If you see `runtime_fallback: missing GStreamer element: ...`, the
  tutorial detected that a required plugin (`v4l2src`, `videoconvert`,
  `neatencoder`, `h264parse`, `rtph264pay`, `udpsink`) is not installed in
  the current GStreamer registry. This is the expected behavior on CI or
  on machines without the SiMa plugin set; the tutorial exits with success.
- The Modalix ISP on `/dev/video0out` often advertises 120 fps even though
  the downstream stage expects 30 fps. `--fps 0` (the default) leaves the
  V4L2 framerate caps field unconstrained, which is the safest choice. The
  encoder's nominal frame rate is controlled by `--enc-fps` and defaults to
  30.

## What To Observe
- `elements_ok=yes/no` indicates whether all required GStreamer elements
  were found.
- `streaming to udp://<host>:<port> for <N> ms` confirms the pipeline was
  built and started.
- Frames should appear on the host's `fpsdisplaysink` window within a few
  hundred milliseconds of the tutorial reaching the sleep loop.
- `CHECK stream_completed: PASS` confirms the pipeline shut down cleanly.
- `SIGNATURE` summarizes the runtime path taken.

## Run
Default values target the Modalix post-ISP device and a localhost UDP
receiver. Override the flags to match your network setup.

```bash
# Minimal run (defaults: /dev/video0out, 1920x1080, RGB, 127.0.0.1:9000).
./tutorial_v2_020_mipi_camera_to_udp

# Full Modalix run targeting a laptop on the same subnet.
./tutorial_v2_020_mipi_camera_to_udp \
  --device /dev/video0out \
  --width 1920 --height 1080 --fps 0 \
  --host 192.168.2.1 --port 9000 \
  --bitrate 4000 --enc-fps 30 \
  --duration-ms 10000

# Python mirror.
python3 tutorials/020_mipi_camera_to_udp/mipi_camera_to_udp.py \
  --device /dev/video0out \
  --host 192.168.2.1 --port 9000
```

## Source Files
- C++: `tutorials/020_mipi_camera_to_udp/mipi_camera_to_udp.cpp`
- Python: `tutorials/020_mipi_camera_to_udp/mipi_camera_to_udp.py`

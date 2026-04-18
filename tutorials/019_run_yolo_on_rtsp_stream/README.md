# 019 Run YOLO on an RTSP Stream

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | rtsp, streaming, yolo, detection, live-input |

## Concept

Pull decoded frames from a live RTSP URL, feed each to a YOLOv8 model, and see the inference output — the minimum loop for a live detection app in NEAT.

This is the first chapter where input comes from **outside the program**. Previous chapters manufactured test images or read a file from disk. Here, frames arrive continuously from a network stream, and the pipeline has to keep up. NEAT takes care of RTSP → H.264 depacketize → decode → frame internally — you hand it a URL and pull decoded frames.

Two Sessions run side by side:
- **RTSP decode session.** Takes an RTSP URL, produces decoded BGR frames on demand.
- **Model session.** Takes a frame, produces a detection output sample.

You pull a frame from the first, feed it to the second, and read what comes back. No box decoding, no drawing, no saving — that is chapter 006's job. This chapter teaches *how do I feed a live stream to a model?*

**APIs introduced**
- `pyneat.RtspDecodedInputOptions()` — configure the RTSP client (URL, transport, output format, caps).
- `pyneat.groups.rtsp_decoded_input(opts)` — the `NodeGroup` that decodes an incoming RTSP stream.
- `session.build(pyneat.RunOptions())` — the `build` overload used when the input originates inside the pipeline (no priming sample to pass).
- `run.pull(timeout_ms)` — pulls one decoded-frame `Sample` from the running RTSP session.

**When to use this pattern**
- Camera feeds streamed over RTSP from an IP camera, DVR, or another server.
- Any live inference pipeline where input comes from a pre-existing stream.
- Prototype detection apps before adding overlays, tracking, or persistence.

Not for:
- **Serving** a stream — this chapter is a consumer only. To publish a stream, run a separate RTSP server (e.g. `mediamtx`) and point `--url` at it.
- Parsing YOLO output — see chapter 006.

**Prerequisites**
- Chapter 001 — `pyneat.Model` and `model.run()`.
- Chapter 002 or 003 — how a `Session` composes nodes.
- An accessible RTSP source (URL) and a YOLOv8 MPK.

**References**
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Configure `RtspDecodedInputOptions` with the stream URL and the desired decoded output format (`BGR`).
2. Build the RTSP decode `Session` (one input group + one output node).
3. Build a YOLO `Model` with `decode_type="yolov8"` and input bounds large enough to cover the stream.
4. Loop: `pull` one decoded frame → `model.run(frame.tensor)` → print a one-line summary.

## What To Observe

For each frame the tutorial prints: `frame=N fields=<F> bbox_bytes=<B>`.
- `N` — zero-indexed frame counter.
- `F` — number of fields on the model output sample. For a YOLOv8 MPK with `decode_type=yolov8`, the BBOX payload is in the top-level tensor, so `fields` is typically `0`.
- `B` — size of the BBOX tensor in bytes. For YOLOv8, this is a packed byte buffer containing the decoded boxes — see chapter 006 for the wire format.

A pull timeout prints `frame=N rtsp_timeout` and exits. That usually means the URL is wrong or the stream is not delivering.

## Run

### Prepare an RTSP URL

If you do not have a camera handy, a reliable quick source is to publish an MP4 file through `mediamtx`:

```bash
# On any machine on your network:
./mediamtx &
ffmpeg -re -stream_loop -1 -i my_video.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/stream
```

Your URL is then `rtsp://<that-machine-ip>:8554/stream`.

### Download the YOLOv8 MPK

Fetch `yolo_v8s.tar.gz` from the SiMa modelzoo once:

```bash
sima-cli modelzoo -v 2.0.0 get yolo_v8s
```

### eLxr SDK (C++)

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
dk ./tutorial_v2_019_run_yolo_on_rtsp_stream \
  --url rtsp://host:port/stream \
  --mpk /absolute/path/to/yolo_v8s.tar.gz
```

### eLxr SDK (Python)

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
dk python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/019_run_yolo_on_rtsp_stream/run_yolo_on_rtsp_stream.py \
  --url rtsp://host:port/stream \
  --mpk /absolute/path/to/yolo_v8s.tar.gz
```

### DevKit (C++)

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
./tutorial_v2_019_run_yolo_on_rtsp_stream \
  --url rtsp://host:port/stream \
  --mpk /absolute/path/to/yolo_v8s.tar.gz
```

### DevKit (Python)

```bash
source ~/pyneat/bin/activate
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/019_run_yolo_on_rtsp_stream/run_yolo_on_rtsp_stream.py \
  --url rtsp://host:port/stream \
  --mpk /absolute/path/to/yolo_v8s.tar.gz
```

## Source Files
- C++: `tutorials/019_run_yolo_on_rtsp_stream/run_yolo_on_rtsp_stream.cpp`
- Python: `tutorials/019_run_yolo_on_rtsp_stream/run_yolo_on_rtsp_stream.py`

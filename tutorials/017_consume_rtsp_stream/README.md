# 017 Consume a Live RTSP Stream

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 5-10 minutes |
| Model | None |
| Labels | rtsp, streaming, input-group, live-input |

## Concept

Attach a live H.264 RTSP stream to a `Session` with the `RtspDecodedInput` node group. The group handles RTSP connect, depacketize, and H.264 decode — you hand it a URL and pull decoded frames.

This is the first chapter where input originates *outside the program*. Previous chapters manufactured test images or read files from disk. Here, frames arrive continuously from a network stream.

The chapter deliberately stops at "pull the decoded frames." Feeding them into a `Model` is covered by the earlier chapters (001 for a single model run, 007 for plugging a model into a pipeline, 015 for embedding a model inside a graph). This chapter is about the input group only.

**APIs introduced**
- `pyneat.RtspDecodedInputOptions()` — configure the RTSP client (URL, transport).
- `pyneat.groups.rtsp_decoded_input(opts)` — the `NodeGroup` that feeds decoded frames into a session.
- `session.build(pyneat.RunOptions())` — the `build` overload used when the input originates inside the pipeline (no priming sample to pass).

**When to use this pattern**
- Camera feeds streamed over RTSP from an IP camera, DVR, or another server.
- Any pipeline where the source is a pre-existing network stream.

Not for:
- **Serving** a stream. This chapter is a consumer only. To publish a stream, run a separate RTSP server (e.g. `mediamtx`) and point `--url` at it.

**Prerequisites**
- Chapter 003 — how a `Session` composes nodes.
- An accessible RTSP URL.

**References**
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Configure `RtspDecodedInputOptions` with the stream URL.
2. Build a `Session` composing the RTSP group + an output node.
3. Pull decoded frames from the session in a loop and inspect their tensor shape.

## What To Observe

For each frame the tutorial prints `frame=N shape=[...]`. The shape is the decoded frame in the tensor's native layout (typically `[H, W]` or `[H, W, C]` depending on the decoder output format). A pull timeout prints `frame=N rtsp_timeout` and exits — that usually means the URL is wrong or the stream is not delivering.

## Run

This chapter consumes a live RTSP stream. If you do not have a camera, publish an MP4 through `mediamtx` + `ffmpeg` and pass the URL via `--url`.

CTest reads `SIMANEAT_APPS_TEST_RTSP_URL` for this chapter's RTSP source. If you manage several sources, set `SIMANEAT_APPS_TEST_RTSP_URLS` and the first URL is used.

**Python:**
```bash
python3 share/sima-neat/tutorials/017_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --frames 5
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_017_consume_rtsp_stream \
  --url rtsp://host:port/stream --frames 5
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_017_consume_rtsp_stream
./build/tutorials-standalone/tutorial_017_consume_rtsp_stream \
  --url rtsp://host:port/stream --frames 5
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/017_consume_rtsp_stream/consume_rtsp_stream.cpp`
- Python: `tutorials/017_consume_rtsp_stream/consume_rtsp_stream.py`

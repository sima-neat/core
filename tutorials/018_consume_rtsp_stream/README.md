# 018 Consume a Live RTSP Stream

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 5-10 minutes |
| Model | None |
| Labels | rtsp, h264, h265, streaming, input-group, live-input |

## Concept

Attach a live H.264 or H.265 RTSP stream to a `Graph` with the
`RtspDecodedInput` fragment. The fragment connects to RTSP, selects the matching
RTP depacketizer and parser, and decodes the stream into raw frames.

## Walkthrough

This is the first chapter where input originates *outside the program*. Earlier chapters manufactured test images or read files from disk; here, frames arrive continuously from a network stream and you consume them as fast as you pull. The mechanism is a reusable `Graph` fragment, `RtspDecodedInput`, that bundles the whole RTSP-to-raw-frames front end behind one interface.

The chapter deliberately stops at "pull the decoded frames." Feeding them into a `Model` is covered elsewhere (001 for a single model run, 007 for plugging a model into a pipeline, 015 for embedding a model inside a graph). By the end you will have connected to an RTSP URL and printed the tensor shape of each decoded frame — proof the stream is flowing.

This is a *consumer* only. To publish a stream, run a separate RTSP server (e.g. `mediamtx`) and point `--url` at it.

### Configure the RTSP client {#step-configure-rtsp}

`RtspDecodedInputOptions` configures the source and decoder. `url` selects the
`rtsp://...` source. `codec` selects the encoded format. H.264 is the default;
the tutorial also accepts `avc`, `h265`, and `hevc`, where AVC equals H.264 and
HEVC equals H.265.

Set `source_fps` when the RTSP caps do not carry a valid frame rate; without
either value, decoder startup fails. For H.265, Neat propagates this value into
the parsed stream caps and decoder configuration. Neat does not probe the URL or
use this option to change the frame rate. The H.265 stream must use HEVC Main
profile, 8-bit, 4:2:0 input.

Setting `tcp = true` requests RTSP-over-TCP, which avoids RTP packet loss on
unreliable networks at the cost of some latency.

### Compose the graph {#step-compose-graph}

Build a `Graph` with just two stages: the `RtspDecodedInput` fragment (the source) and a bare `Output` node (the pull endpoint). Adding the fragment is a single `add(...)` — it expands internally into the connect/depacketize/decode elements, so your composition stays at the level of intent. Because the input originates *inside* the pipeline, we call the `build(RunOptions{})` overload that takes no priming sample: there is no frame to hand `build()` up front, since the stream produces them.

### Pull decoded frames {#step-pull-frames}

With the run live, loop and `pull(...)` with a timeout. Each successful pull yields a `Sample` whose tensor is one decoded frame; we print `frame=N shape=[...]`, where the shape is the frame in the decoder's native layout (typically `[H, W]` or `[H, W, C]` depending on the output format). A pull that returns nothing (or an empty tensor) prints `frame=N rtsp_timeout` and breaks the loop — that usually means the URL is wrong or the stream is not delivering. The timeout is what keeps a dead stream from hanging the program.

**C++:** A frame is extracted with `tensors_from_sample(*sample, true)`; the loop checks for an empty list before reading `shape`.

**Python:** A frame is read from the first entry in `sample.tensors` before printing its shape.

## Run

This chapter consumes a live RTSP stream, so you must supply a reachable
`--url`. If you do not have a camera, publish a compatible video through an RTSP
server and point `--url` at it. Run the **Python** and **C++ (prebuilt)** commands
from the **Neat install root** (the directory that contains `share/` and
`lib/`); run the **build from source** commands from the **repo root**.

The automated tutorial regression uses the H.264 default and reads
`SIMANEAT_TEST_RTSP_H264_URL`. If several sources are configured, it reads the
first URL from `SIMANEAT_TEST_RTSP_H264_URLS`.

**Python:**
```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --source-fps 30 --frames 5
```

For H.265:

```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --codec hevc --source-fps 30 --frames 5
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_018_consume_rtsp_stream
./build/tutorials-standalone/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

Expected output (shape depends on the stream's resolution and decoder format):

```text
frame=0 shape=[720, 1280]
frame=1 shape=[720, 1280]
frame=2 shape=[720, 1280]
frame=3 shape=[720, 1280]
frame=4 shape=[720, 1280]
```

If the stream is unreachable you will instead see `frame=0 rtsp_timeout`. To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.cpp`
- Python: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py`

# 018 Consume a Live RTSP Stream

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
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

Set `source_fps` when you already know the source cadence. If you omit it, this
tutorial opens the RTSP source with OpenCV, reads its reported FPS, and supplies
the detected value to `RtspDecodedInput`. The group itself does not probe the
URL. Only the probing path needs OpenCV, and the Python version imports it on
demand, so supplying `--source-fps` runs without it; to probe, install it with
`pip install opencv-python`. For H.265, Neat propagates this value into the
parsed stream caps and decoder configuration. It does not change the frame rate.
The H.265 stream must use HEVC Main profile, 8-bit, 4:2:0 input.

Setting `tcp = true` carries RTP over TCP. TCP preserves order and retransmits
lost segments, which can reduce visible loss compared with UDP but can increase
latency while recovering lost data.

### Compose the graph {#step-compose-graph}

Build a `Graph` with just two stages: the `RtspDecodedInput` fragment (the source) and a bare `Output` node (the pull endpoint). Adding the fragment is a single `add(...)` — it expands internally into the connect/depacketize/decode elements, so your composition stays at the level of intent. Because the input originates *inside* the pipeline, we call the `build(RunOptions{})` overload that takes no priming sample: there is no frame to hand `build()` up front, since the stream produces them.

### Pull decoded frames {#step-pull-frames}

Use C++ status-aware `pull(...)` or Python `pull_status(...)` to distinguish
frames, timeouts, closure and errors. Successful samples contain an NV12 frame
as a logical `[H, W]` tensor with Y and UV plane metadata. The examples print
its shape and stop after the requested number of frames.

Reconnect is disabled by default. Set `--reconnect-timeout-ms 10000` to retry
after ten seconds without source data. `--reconnect-attempts -1` retries
indefinitely; a positive value limits retries per outage. While reconnecting,
the examples continue across pull timeouts. Closure or error ends the example.
Supply `--source-fps` with reconnect so an initially offline camera does not
need the tutorial's OpenCV FPS probe. Existing caps requirements still apply.

## Run

This chapter consumes a live RTSP stream, so you must supply a reachable
`--url`. If you do not have a camera, publish a compatible video through an RTSP
server and point `--url` at it. Run the **Python** and **C++ (prebuilt)** commands
from the **Neat install root** (the directory that contains `share/` and
`lib/`); run the **build from source** commands from the **repo root**.

The automated tutorial regression runs both codecs. It reads the first usable
URL from `SIMANEAT_TEST_RTSP_H264_URL` or `SIMANEAT_TEST_RTSP_H264_URLS`, and
from `SIMANEAT_TEST_RTSP_H265_URL` or `SIMANEAT_TEST_RTSP_H265_URLS`. The test
probes each source and supplies its detected FPS to the RTSP group.

**Python:**
```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --source-fps 30 --frames 5
```

For automatic reconnect, add `--reconnect-timeout-ms 10000 --reconnect-attempts -1`
to either language's command. The source can start
offline and join later; the frame counter continues after recovery.

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

With reconnect disabled, an unreachable stream ends the example with a diagnostic. To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.cpp`
- Python: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py`
- Python OpenCV FPS probe: `tutorials/018_consume_rtsp_stream/probe_rtsp_fps.py`

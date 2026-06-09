# 017 Consume a Live RTSP Stream

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 5-10 minutes |
| Model | None |
| Labels | rtsp, streaming, input-group, live-input |

## Concept

Attach a live H.264 RTSP stream to a `Graph` with the `RtspDecodedInput` fragment, which handles RTSP connect, depacketize, and H.264 decode — you hand it a URL and pull decoded frames.

## Walkthrough

This is the first chapter where input originates *outside the program*. Earlier chapters manufactured test images or read files from disk; here, frames arrive continuously from a network stream and you consume them as fast as you pull. The mechanism is a reusable `Graph` fragment, `RtspDecodedInput`, that bundles the whole RTSP-to-raw-frames front end behind a single node.

The chapter deliberately stops at "pull the decoded frames." Feeding them into a `Model` is covered elsewhere (001 for a single model run, 007 for plugging a model into a pipeline, 015 for embedding a model inside a graph). By the end you will have connected to an RTSP URL and printed the tensor shape of each decoded frame — proof the stream is flowing.

This is a *consumer* only. To publish a stream, run a separate RTSP server (e.g. `mediamtx`) and point `--url` at it.

### Configure the RTSP client {#step-configure-rtsp}

`RtspDecodedInputOptions` is the configuration for the input fragment. The two fields that matter here are `url` (the `rtsp://...` source, taken from `--url`) and `tcp`, which selects the RTSP transport. Setting `tcp = true` requests RTSP-over-TCP — more robust across NAT and firewalls than the UDP default, at the cost of some latency. These options fully describe *where* frames come from and *how* they are carried.

### Compose the graph {#step-compose-graph}

Build a `Graph` with just two stages: the `RtspDecodedInput` fragment (the source) and a bare `Output` node (the pull endpoint). Adding the fragment is a single `add(...)` — it expands internally into the connect/depacketize/decode elements, so your composition stays at the level of intent. Because the input originates *inside* the pipeline, we call the `build(RunOptions{})` overload that takes no priming sample: there is no frame to hand `build()` up front, since the stream produces them.

### Pull decoded frames {#step-pull-frames}

With the run live, loop and `pull(...)` with a timeout. Each successful pull yields a `Sample` whose tensor is one decoded frame; we print `frame=N shape=[...]`, where the shape is the frame in the decoder's native layout (typically `[H, W]` or `[H, W, C]` depending on the output format). A pull that returns nothing (or an empty tensor) prints `frame=N rtsp_timeout` and breaks the loop — that usually means the URL is wrong or the stream is not delivering. The timeout is what keeps a dead stream from hanging the program.

**C++:** A frame is extracted with `tensors_from_sample(*sample, true)`; the loop checks for an empty list before reading `shape`.

**Python:** A frame is `sample.tensor`; the loop checks for `None` before reading `sample.tensor.shape`.

## Run

This chapter consumes a live RTSP stream, so you must supply a reachable `--url`; if you do not have a camera, publish an MP4 through `mediamtx` + `ffmpeg` and point `--url` at it. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

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

Expected output (shape depends on the stream's resolution and decoder format):

```text
frame=0 shape=[720, 1280, 3]
frame=1 shape=[720, 1280, 3]
frame=2 shape=[720, 1280, 3]
frame=3 shape=[720, 1280, 3]
frame=4 shape=[720, 1280, 3]
```

If the stream is unreachable you will instead see `frame=0 rtsp_timeout`. To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/017_consume_rtsp_stream/consume_rtsp_stream.cpp`
- Python: `tutorials/017_consume_rtsp_stream/consume_rtsp_stream.py`

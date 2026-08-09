---
title: Per-frame attributes
description: Capturing selected multipart HTTP headers and reading them back from the decoded frame they arrived with
sidebar_position: 4
slug: /develop-apps/advanced-concepts/frame_attributes
---

# Per-frame attributes

Some cameras describe each frame in the transport that delivers it: a sequence number, a
capture timestamp, a channel name. Neat can carry a selected set of those values through
decode and hand them back on the `Sample` for the frame they belong to.

`Sample::attributes` is a plain string-to-string map. Neat copies it, keeps it associated
with its frame, and clears it when a destination buffer is reused. It never parses, merges,
or reinterprets a value. Keys and values containing an embedded NUL byte are rejected because
the GStreamer string representation cannot preserve them.

## What is guaranteed

> Selected multipart part headers remain associated with their decoded frame through the
> default `HttpMjpegDecodedInput` path, Queue/Branch, and Core Sample-to-GStreamer
> materialized boundaries.

That is the whole promise for this release. Anything outside it is listed under
[Unsupported paths](#unsupported-paths), and Neat fails graph construction rather than
silently dropping attributes.

## Enabling capture

Capture is off by default. Naming the headers you want turns it on.

```cpp
#include "nodes/groups/HttpMjpegDecodedInput.h"

simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
opt.url = "http://camera.local/stream";
opt.header_capture.headers = {"Image-Index", "Image-Time"};

auto source = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
```

Reading them back:

```cpp
simaai::neat::Sample sample;
if (run.pull(1000, sample) == simaai::neat::PullStatus::Ok) {
  const auto it = sample.attributes.find("image-index");
  if (it != sample.attributes.end()) {
    // it->second is the value this frame was sent with.
  }
}
```

The same surface in Python, where `attributes` is a live mapping — item assignment reaches
the underlying `Sample`, and assigning a dict replaces the contents:

```python
import pyneat

opt = pyneat.HttpMjpegDecodedInputOptions()
opt.url = "http://camera.local/stream"
opt.header_capture.headers = ["Image-Index", "Image-Time"]
source = pyneat.groups.http_mjpeg_decoded_input(opt)

# ... later, on a pulled sample:
index = sample.attributes.get("image-index")

sample.attributes["image-index"] = "42"     # reaches the Sample
sample.attributes = {"image-time": "..."}   # replaces the whole map
```

## Header rules

The configured list is an **allowlist**. An empty list disables capture entirely and leaves
the existing topology and behavior untouched.

| Rule | Behavior |
| --- | --- |
| Case | Configured names and emitted keys are normalized to ASCII lowercase; matching is case-insensitive. Read attributes back with lowercase keys. |
| Duplicates in the allowlist | Collapse after normalization. |
| Header repeated within one part | Last value wins. |
| Header absent from a part | Key is omitted. It is never inherited from a previous frame. |
| Header present but empty | Preserved as an empty string. |
| Whitespace | Only surrounding SP/HTAB is trimmed. Values are not otherwise reinterpreted. |
| MIME type | Every part must declare `Content-Type: image/jpeg` (parameters are allowed). |
| JPEG payload | A part must contain exactly one complete JPEG from SOI through EOI. Truncated, empty, or concatenated images fail the stream. |
| Invalid input | Invalid header names, folded header lines, and CR/LF/NUL injection are rejected — the stream errors rather than being normalized into something safe-looking. |

Distinguish "absent" from "empty" with `count()` / `get()` rather than by testing for an
empty string.

### Limits

Parsing fails rather than truncating when any of these is exceeded:

| Limit | Value |
| --- | --- |
| `kMultipartHeaderCaptureMaxHeaders` | 64 selected header names |
| `kMultipartHeaderCaptureMaxNameBytes` | 128 bytes per name |
| `kMultipartHeaderCaptureMaxLineBytes` | 8 KiB per header line |
| `kMultipartHeaderCaptureMaxBlockBytes` | 64 KiB per part header block |
| Multipart JPEG body | 64 MiB per MIME part |

A malformed allowlist is rejected at construction with `std::invalid_argument`.

## Unsupported paths

While capture is enabled, `HttpMjpegDecodedInput` refuses to build a graph containing
`use_videoconvert`, `use_videoscale`, `use_videorate`, or `extra_fragment`. Preservation
through those elements has not been proven, and a clear construction error is better than
metadata that quietly disappears mid-stream.

Attributes are also not defined for nodes that create a new logical Sample from several
inputs — models, joins, aggregators. Those nodes do not merge attributes.

## How it stays associated

Capture-enabled graphs use a private in-process element that parses part boundaries **and**
part headers in one state machine, so a part's headers are attached to the very buffer
carrying its bytes; there is no side channel that can drift. That element emits complete,
parsed JPEG frames, so `jpegparse` is not inserted on the capture-enabled path.
If attaching the selected attributes fails, the frame is not delivered and the stream
reports an error.

Through decode, the plugin snapshots the attributes of each accepted encoded picture and
restores them onto the decoded output the decoder correlates back to it — not onto whichever
output happens to arrive next. Every accepted picture reaches exactly one terminal result,
so reordering, drops, and output-pool reuse cannot shift a value onto a different frame.
The mechanism is codec- and transport-independent, which is what lets it extend to other
encoded sources later without redesign.

## Compatibility

`Sample` and the source option structs gained appended fields. Source code that uses them by
field name or aggregate initialization keeps compiling.

The binary layout of those public structs changed, so **already-built consumers must be
rebuilt**. The Neat ABI/SOVERSION stays at **4**: 0.4.0 is unreleased, so all ABI-4
components are rebuilt and released together rather than bumping the ABI.

## Adding another source later

The decoder path is generic. A new encoded source only needs to attach the nested attribute
structure to the buffer it hands the decoder; nothing in the decoder or in the Sample
boundaries is transport-specific. What each new source still owns is its own extraction rule
and its own explicit statement of which graph shapes it guarantees.

---
title: Send JSON Metadata
description: MetadataSender UDP JSON wire contract
sidebar_position: 3
slug: /develop-apps/advanced-concepts/metadata_sender
---

# Send JSON Metadata

Use `MetadataSender` when an external viewer, recorder, or service accepts UTF-8 JSON metadata over UDP. Insight is one receiver that understands this wire contract.

## Wire Contract

- Default host: `127.0.0.1`
- Default metadata port base: `9100`
- Channel port rule: `metadata_port_base + channel`
- Default send mode: nonblocking (`MSG_DONTWAIT`)
- Payload encoding: UTF-8 JSON text
- Required top-level fields: `type`, `data`
- Maximum logical payload: 65,507 bytes

`MetadataSender` keeps each UDP payload at or below 1200 bytes. JSON payloads
up to 1200 bytes remain one unchanged datagram. Larger payloads are split into
chunks with this 12-byte binary header:

| Byte | Size | Value |
| --- | --- | --- |
| 0 | 1 | Magic byte `0x4e` |
| 1 | 1 | Protocol version `0x01` |
| 2 | 8 | Message ID as an unsigned 64-bit big-endian integer |
| 10 | 1 | Zero-based chunk index |
| 11 | 1 | Total chunk count |

Each chunk carries up to 1188 JSON bytes. A receiver reassembles chunks with
the same sender address and message ID in chunk-index order before parsing the
JSON. UDP delivery remains best effort: the sender does not retry a failed
chunk, and `send_raw_json(...)` or `send_metadata(...)` returns `false` after
the first local send failure.

Receivers should accept both unchanged JSON datagrams and versioned chunks.
Update Insight to a release with chunk reassembly before or together with this
Neat Library version. Older Insight versions continue to receive payloads up
to 1200 bytes, but cannot decode larger chunked payloads.

For Insight, pair metadata channel `N` with the video UDP stream on `9000 + N`.
If Insight or another receiver runs behind container port remapping, pass the mapped host and port explicitly from the app.

Tracking, tracklets, and other custom metadata can be sent as generic JSON. Viewer overlay support is receiver-specific; Insight tracking visualization is tracked separately in `sima-neat/insight#8`.

## C++

```cpp
simaai::neat::MetadataSenderOptions opt;
opt.host = "127.0.0.1";
opt.channel = 0;
opt.metadata_port_base = 9100;

std::string err;
simaai::neat::MetadataSender sender(opt, &err);

sender.send_metadata(
    "tracking",
    R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})",
    12345,
    "frame-7",
    &err);
```

`send_metadata(...)` validates `data_json` and builds this envelope:

```json
{
  "type": "tracking",
  "timestamp": 12345,
  "frame_id": "frame-7",
  "data": {
    "tracks": [
      {
        "id": "trk-1",
        "bbox": [10, 20, 30, 40]
      }
    ]
  }
}
```

Use `send_raw_json(...)` only when the caller already built the full top-level payload:

```cpp
sender.send_raw_json(
    R"({"type":"object-detection","data":{"objects":[{"id":"obj_1","label":"car","confidence":0.92,"bbox":[120,80,96,64]}]}})",
    &err);
```

## Real-Time Dispatch Is Nonblocking by Default

`MetadataSender` applies `MSG_DONTWAIT` to each datagram by default so a locally
congested send buffer cannot delay a thread that also dispatches video or
inference work. When the kernel cannot accept a datagram immediately, the send
returns `false` instead of waiting. Treat that metadata packet as dropped and
continue real-time work; UDP delivery is not guaranteed.

The default constructor and the default send options are equivalent:

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

Callers that explicitly prefer blocking delivery attempts can opt in:

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
send_opt.nonblocking = false;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

Use `stats()` to distinguish congestion from other failures and, in explicit
blocking mode, to detect slow calls:

```cpp
const auto stats = sender.stats();
std::cerr << "sent=" << stats.datagrams_sent
          << " would_block=" << stats.would_block
          << " enobufs=" << stats.no_buffer_space
          << " max_send_ns=" << stats.max_send_duration_ns << '\n';
```

## Python

```python
import json
import pyneat

opt = pyneat.MetadataSenderOptions()
opt.host = "127.0.0.1"
opt.channel = 0
opt.metadata_port_base = 9100

sender = pyneat.MetadataSender(opt)

sender.send_metadata(
    "object-detection",
    json.dumps(
        {
            "objects": [
                {
                    "id": "obj_1",
                    "label": "car",
                    "confidence": 0.92,
                    "bbox": [120, 80, 96, 64],
                }
            ]
        }
    ),
    12345,
    "frame-7",
)

stats = sender.stats()
print(stats.datagrams_sent, stats.would_block, stats.max_send_duration_ns)
```

As in C++, explicitly set `send_opt.nonblocking = False` and pass it as the
second constructor argument only when blocking behavior is required.

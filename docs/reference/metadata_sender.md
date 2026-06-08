---
title: Send JSON Metadata
description: MetadataSender UDP JSON wire contract
sidebar_position: 4
---

# Send JSON Metadata

Use `MetadataSender` when an external viewer, recorder, or service accepts UTF-8 JSON metadata over UDP. Insight is one receiver that understands this wire contract.

## Wire Contract

- Default host: `127.0.0.1`
- Default metadata port base: `9100`
- Channel port rule: `metadata_port_base + channel`
- Payload encoding: UTF-8 JSON text
- Required top-level fields: `type`, `data`

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
```

# Send MetadataReceiver JSON

Use `MetadataReceiverOutput` when a viewer or recorder accepts UTF-8 JSON metadata over UDP.
Insight is one MetadataReceiver implementation.

## Wire contract

- Metadata UDP port: `metadata_port_base + channel`
- Default metadata port base: `9100`
- Default host: `127.0.0.1`
- Payload encoding: UTF-8 JSON text
- Required top-level fields: `type`, `data`

For Insight, pair metadata channel `N` with the video UDP stream on `9000 + N`.
The object-detection schema renders in Insight today. Other metadata types can be transported as JSON, but they need viewer-side renderer support before overlays appear.

## C++

```cpp
simaai::neat::MetadataReceiverChannelOptions opt;
opt.host = "127.0.0.1";
opt.channel = 0;
opt.metadata_port_base = 9100;

std::string err;
simaai::neat::MetadataReceiverOutput out(opt, &err);

out.send_metadata(
    "tracking",
    R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})",
    12345,
    "frame-7",
    &err);
```

`send_metadata(...)` builds this envelope:

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
out.send_raw_json(
    R"({"type":"object-detection","data":{"objects":[{"id":"obj_1","label":"car","confidence":0.92,"bbox":[120,80,96,64]}]}})",
    &err);
```

## Python

```python
import json
import pyneat

opt = pyneat.MetadataReceiverChannelOptions()
opt.host = "127.0.0.1"
opt.channel = 0
opt.metadata_port_base = 9100

out = pyneat.MetadataReceiverOutput(opt)

out.send_metadata(
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

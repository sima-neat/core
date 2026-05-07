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

## C++: raw metadata

```cpp
simaai::neat::MetadataReceiverChannelOptions opt;
opt.host = "127.0.0.1";
opt.channel = 0;
opt.metadata_port_base = 9100;

std::string err;
simaai::neat::MetadataReceiverOutput out(opt, &err);

simaai::neat::MetadataReceiverPayload payload;
payload.type = "tracking";
payload.data_json = R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})";
payload.timestamp_ms = 12345;
payload.frame_id = "frame-7";

out.send_metadata(payload, &err);
```

## C++: object detection

```cpp
std::vector<simaai::neat::MetadataReceiverObject> objects = {
    {.x = 10, .y = 20, .w = 30, .h = 40, .score = 0.95f, .class_id = 0},
};

out.send_object_detection(12345, "frame-7", objects, {"person"}, &err);
```

This sends:

```json
{
  "type": "object-detection",
  "timestamp": 12345,
  "frame_id": "frame-7",
  "data": {
    "objects": [
      {
        "id": "obj_1",
        "label": "person",
        "confidence": 0.95,
        "bbox": [10, 20, 30, 40]
      }
    ]
  }
}
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

payload = pyneat.MetadataReceiverPayload()
payload.type = "tracking"
payload.data_json = json.dumps({"tracks": [{"id": "trk-1", "bbox": [10, 20, 30, 40]}]})
payload.timestamp_ms = 12345
payload.frame_id = "frame-7"

out.send_metadata(payload)
```

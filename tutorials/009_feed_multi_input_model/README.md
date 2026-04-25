# 009 Feed Models That Take Multiple Inputs

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | None |
| Labels | multi-input, samples, sync |

## Concept

Bundle multiple tensors into one `Sample` and push it as a single inference event. This is the pattern you need when a model takes more than one input — stereo frames, image + metadata, sensor fusion.

Many real applications carry more than one input per inference event. NEAT represents this as a **bundle sample**: a single `Sample` whose `fields` list holds multiple named tensor payloads, each addressable by port name.

**APIs introduced**
- `pyneat.Sample()` + `sample.kind = pyneat.SampleKind.Bundle` — create the bundle envelope.
- `sample.fields = [...]` — the list of named inner samples.
- `pyneat.make_tensor_sample(port_name, tensor)` — build one named field.

**When to use this**
- Stereo or paired inputs: bundle `left` and `right` together as one logical unit.
- Sensor fusion pipelines: attach related tensors/fields in one sample envelope.
- Debugging wiring issues: inspect `port_name` and field tensor presence on output.

**Prerequisites**
Chapters 001–003 (Model, Session, Run basics). Chapter 008 (Tensor interop).

**References**
- [Session](/getting-started/programming-model/session)
- [Tensor and Sample](/getting-started/programming-model/core_types)

## Learning Process
1. Define a tensor session contract for deterministic multi-field routing.
2. Build a seed run handle, then create a bundle sample with named tensor fields.
3. Push/pull the bundle and inspect output field structure.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/009_feed_multi_input_model/feed_multi_input_model.py \
  --width 64 --height 48
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_009_feed_multi_input_model \
  --width 64 --height 48
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_009_feed_multi_input_model
./build/tutorials-standalone/tutorial_009_feed_multi_input_model \
  --width 64 --height 48
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/009_feed_multi_input_model/feed_multi_input_model.cpp`
- Python: `tutorials/009_feed_multi_input_model/feed_multi_input_model.py`

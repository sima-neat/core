# 009 Feed Models That Take Multiple Inputs

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Labels | multi-input, samples, sync |

## Concept
This tutorial shows how to send and receive **multi-field samples** (bundle samples) instead of only single-tensor payloads.

Many real applications carry more than one input per inference event (for example left/right streams, image + metadata, or multi-sensor packets). This chapter teaches the sample-bundling pattern you need before building those systems.

What this chapter demonstrates:
- Building a tensor-input session contract (`FP32` tensor media).
- Constructing a bundle sample with multiple named fields.
- Pushing/pulling bundle payloads and validating field-level outputs.

Use-case guidance:
- Stereo or paired inputs: bundle `left` and `right` together as one logical unit.
- Sensor fusion pipelines: attach related tensors/fields in one sample envelope.
- Debugging wiring issues: inspect `port_name` and field tensor presence on output.

Reference:
- [Session](/getting-started/programming-model/session)
- [Tensor and Sample](/getting-started/programming-model/core_types)

## Learning Process
1. Define a tensor session contract for deterministic multi-field routing.
2. Build a seed run handle, then create a bundle sample with named tensor fields.
3. Push/pull the bundle and inspect output field structure.
4. Confirm behavior using `CHECK`, `SIGNATURE`, and `[OK]`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_009_multi_input_samples
python3 tutorials/009_multi_input_samples/multi_input_samples.py
```

## Source Files
- C++: `tutorials/009_multi_input_samples/multi_input_samples.cpp`
- Python: `tutorials/009_multi_input_samples/multi_input_samples.py`

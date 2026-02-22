# 013 Resnet Quickstart

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | resnet, classification, mpk |

## Concept
This tutorial is a quick classifier bring-up path using a ResNet-style MPK.

If YOLO quickstart teaches detection flow, this chapter teaches classification flow. It helps new users verify model loading, preprocessing assumptions, and output tensor shape before building application logic on top.

What this chapter demonstrates:
- Loading a ResNet MPK with explicit image/preproc options.
- Running a deterministic classification inference path.
- Inspecting output tensor rank and leading dimension as a sanity check.

Use-case guidance:
- First classification model integration in a new environment.
- Validating normalization/format settings for classifier accuracy.
- Establishing baseline output contracts before label mapping/top-k handling.

Reference:
- [Model](/getting-started/programming-model/model)
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Resolve ResNet MPK and deterministic classifier input tensor/image.
2. Configure model input/preproc options for classification.
3. Run inference and inspect output tensor rank/shape cues.
4. Confirm expected path with `CHECK`, `SIGNATURE`, and `[OK]`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_013_resnet_quickstart
python3 tutorials/013_resnet_quickstart/resnet_quickstart.py
```

## Source Files
- C++: `tutorials/013_resnet_quickstart/resnet_quickstart.cpp`
- Python: `tutorials/013_resnet_quickstart/resnet_quickstart.py`

# 009 Multi Input Samples

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Labels | multi-input, samples, sync |

## Concept
Run pipelines that consume bundled or multiple input samples in one flow.

## Learning Process
1. Parse flags and establish deterministic defaults.
2. Exercise the chapter's primary runtime path.
3. Emit checks and machine-parseable signature.

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

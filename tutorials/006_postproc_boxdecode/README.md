# 006 Postproc Boxdecode

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | postprocessing, boxdecode, detection |

## Concept
Use postprocessing settings to decode detection-oriented outputs correctly.

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
./tutorial_v2_006_postproc_boxdecode
python3 tutorials/006_postproc_boxdecode/postproc_boxdecode.py
```

## Source Files
- C++: `tutorials/006_postproc_boxdecode/postproc_boxdecode.cpp`
- Python: `tutorials/006_postproc_boxdecode/postproc_boxdecode.py`

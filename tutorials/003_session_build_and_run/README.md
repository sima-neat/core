# 003 Session Build And Run

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Labels | session, build, run, pipeline |

## Concept
Build a Session once and run it with deterministic runtime behavior.

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
./tutorial_v2_003_session_build_and_run
python3 tutorials/003_session_build_and_run/session_build_and_run.py
```

## Source Files
- C++: `tutorials/003_session_build_and_run/session_build_and_run.cpp`
- Python: `tutorials/003_session_build_and_run/session_build_and_run.py`

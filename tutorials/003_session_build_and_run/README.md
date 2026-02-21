# 003 Session Build And Run

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Labels | session, build, run, pipeline |

## Concept
This tutorial introduces `Session`, the runtime composition entry point in NEAT.

A `Session` is where you define pipeline structure by adding nodes and node groups in order. It is not a one-off inference call; it is a reusable runtime graph definition that can be built once and executed many times.

`build(...)` turns that definition into a runnable `Run` handle. In practice, `build(...)` is the transition from "graph description" to "executable runtime":
- Resolves the added nodes/groups into a concrete pipeline.
- Validates input/output contracts for the selected input type.
- Configures runtime behavior (for example sync vs async run mode and output memory policy).
- Returns a `Run` object that executes push/pull calls.

For the programming model behind this chapter, see:
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Create a minimal `Session` with explicit input and output nodes.
2. Build the session with a concrete sample input to materialize a runnable pipeline.
3. Execute one deterministic sync run to verify output contract behavior.
4. Read validation checkpoints (`CHECK`, `SIGNATURE`, `[OK]`) to confirm the runtime path.

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

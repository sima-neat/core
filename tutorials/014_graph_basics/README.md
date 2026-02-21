# 014 Graph Basics

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | graph, traversal, metadata |

## Concept
Build a minimal graph and verify metadata survives graph traversal.

## Learning Process
1. Build graph and push one deterministic tensor sample.
2. Prefer pipeline+stage hybrid and fallback to stage-only.
3. Validate stream/frame metadata after traversal.
4. Verify stream/frame metadata survived graph traversal.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_014_graph_basics
python3 tutorials/014_graph_basics/graph_basics.py
```

## Source Files
- C++: `tutorials/014_graph_basics/graph_basics.cpp`
- Python: `tutorials/014_graph_basics/graph_basics.py`

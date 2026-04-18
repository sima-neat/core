# 015 Embed a Model Inside a Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Labels | graph, hybrid, stage-model, mpk |

## Concept
This tutorial shows how to combine graph orchestration with model execution in one hybrid flow.

Why this chapter matters: many production systems need graph-level control (routing, scheduling, composition) while still using model execution as a stage. This tutorial demonstrates that bridge pattern with a reliable fallback path.

What this chapter demonstrates:
- Building a graph with `StageModelExecutor` when an MPK is available.
- Running a deterministic stage-only fallback when model assets are missing/unavailable.
- Validating tensor output kind/payload consistency across both paths.

Use-case guidance:
- You need graph-level composition around model execution.
- You want deterministic behavior in CI/dev even when model assets differ by environment.
- You need a safe migration path from simple stage graphs to model-backed hybrid graphs.

Reference:
- [Graph](/getting-started/programming-model/graph)
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Prepare deterministic tensor samples that match model input contracts.
2. Build and run stage-model hybrid flow when MPK is available.
3. Fall back to stage-only graph flow when model path is unavailable.
4. Compare output kind/rank behavior and validate with `CHECK` + `SIGNATURE`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_015_graph_model_hybrid
python3 tutorials/015_graph_model_hybrid/graph_model_hybrid.py
```

## Source Files
- C++: `tutorials/015_graph_model_hybrid/graph_model_hybrid.cpp`
- Python: `tutorials/015_graph_model_hybrid/graph_model_hybrid.py`

# 018 Build a Production-Ready Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Labels | production, reliability, deployment |

## Concept
This tutorial assembles a practical production blueprint from patterns introduced earlier: explicit run options, controlled async behavior, and model/session fallback paths.

The goal is not a full product framework. It is a reliable template you can adapt for deployment:
- model-backed path when MPK assets are available
- session-only fallback path when they are not
- consistent runtime/metrics/report behavior in both cases

Why this chapter is last: it combines correctness, observability, and resilience patterns from previous tutorials into one deployment-ready skeleton.

What this chapter demonstrates:
- Standardized `RunOptions` for queueing, memory ownership, and metrics.
- `Model.build(...)` blueprint path with `ModelSessionOptions`.
- Session async fallback blueprint with output/report checks.

Reference:
- [Model](/getting-started/programming-model/model)
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Prepare deterministic input and shared runtime options for production-like behavior.
2. Execute model-backed blueprint when MPK exists.
3. Execute session fallback blueprint when model assets are unavailable.
4. Validate resiliency path with consistent `CHECK`, `SIGNATURE`, and `[OK]` outputs.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_018_production_blueprint
python3 tutorials/018_production_blueprint/production_blueprint.py
```

## Source Files
- C++: `tutorials/018_production_blueprint/production_blueprint.cpp`
- Python: `tutorials/018_production_blueprint/production_blueprint.py`

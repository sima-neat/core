# 010 Read and Interpret Model Output

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | output, patterns, sink |

## Concept
This tutorial focuses on **output interpretation patterns**: how to reliably inspect what came back from a run and decide what your application should do next.

Before you optimize throughput or add complex graph logic, you need a stable way to classify outputs (`kind`, tensor presence, field count) and validate output contracts. This chapter provides that baseline pattern.

What this chapter demonstrates:
- Building a minimal sync run path.
- Reading `Sample` output summary (`kind`, tensor, fields).
- Validating output shape/rank assumptions before downstream use.

Use-case guidance:
- New model integration: verify output rank and tensor presence first.
- Mixed output types: branch behavior by `SampleKind` and field structure.
- Production robustness: fail fast when output shape contracts are unexpectedly empty.

Reference:
- [Input and Output](/getting-started/programming-model/io)
- [Tensor and Sample](/getting-started/programming-model/core_types)

## Learning Process
1. Build a deterministic sync session with explicit input/output nodes.
2. Execute one run and summarize output structure (`kind`, tensor, fields).
3. Validate output contract assumptions (non-empty tensor shape/rank).
4. Confirm run health through `CHECK`, `SIGNATURE`, and `[OK]`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_010_output_handling_patterns
python3 tutorials/010_output_handling_patterns/output_handling_patterns.py
```

## Source Files
- C++: `tutorials/010_output_handling_patterns/output_handling_patterns.cpp`
- Python: `tutorials/010_output_handling_patterns/output_handling_patterns.py`

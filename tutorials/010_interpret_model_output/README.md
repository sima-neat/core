# 010 Read and Interpret Model Output

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | output, patterns, sink |

## Concept

Read back from `run.pull()` or `model.run()` safely. Every model returns a `Sample`, but `Sample` is a small sum type — a tensor, a bundle of fields, or both. Learn which fields to check before touching the payload.

Before you optimize throughput or add complex graph logic, you need a stable way to classify outputs (`kind`, tensor presence, field count) and validate output contracts. This chapter provides that baseline.

**APIs introduced**
- `sample.kind` — `SampleKind.Tensor`, `SampleKind.Bundle`, etc.
- `sample.tensor` — present for tensor-kind samples; `None` otherwise.
- `sample.fields` — list of inner samples for bundle-kind.
- `tensor.shape`, `tensor.dtype` — inspect the payload before use.

**When to use this**
- New model integration: verify output rank and tensor presence first.
- Mixed output types: branch behavior by `SampleKind` and field structure.
- Building output readers that need to handle any model the runtime serves.

**Prerequisites**
Chapter 001.

**References**
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

**Python:**
```bash
python3 share/sima-neat/tutorials/010_interpret_model_output/interpret_model_output.py
```

**C++:**
```bash
./lib/sima-neat/tutorials/tutorial_v2_010_interpret_model_output
```

To compile this chapter's C++ source in your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/010_interpret_model_output/interpret_model_output.cpp`
- Python: `tutorials/010_interpret_model_output/interpret_model_output.py`

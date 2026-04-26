# 008 Numpy Torch Tensor IO

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | numpy, pytorch, tensor, io |

## Concept
This tutorial explains how Neat tensors map to the data structures most Python and ML users already know: NumPy arrays and PyTorch tensors.

If you are integrating Neat into an existing Python inference stack, this chapter should come handy. It helps you avoid common integration mistakes around shape/layout, dtype, and copy semantics before you build larger pipelines.

What this chapter demonstrates:
- Creating Neat tensors from NumPy and PyTorch.
- Converting Neat tensors back to NumPy/PyTorch.
- Basic C++ tensor mapping/clone flow for parity with Python paths.

Use-case guidance:
- Existing preprocessing in NumPy: keep that path, then hand off tensors to Neat.
- Existing model/postprocessing in PyTorch: convert in/out cleanly without rewriting your whole stack.
- Interop debugging: use deterministic round-trip checks to confirm data integrity.

Reference:
- [Tensor and Sample](/getting-started/programming-model/core_types)
- [Input and Output](/getting-started/programming-model/io)

## Learning Process
1. Build deterministic tensor inputs in NumPy/PyTorch and C++ tensor storage.
2. Convert across boundaries (NumPy/PyTorch <-> Neat tensor).
3. Verify round-trip shape/bytes/checksum behavior.
4. Validate completion through `CHECK`, `SIGNATURE`, and `[OK]` markers.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_008_numpy_torch_tensor_io
python3 tutorials/008_numpy_torch_tensor_io/numpy_torch_tensor_io.py
```

## Source Files
- C++: `tutorials/008_numpy_torch_tensor_io/numpy_torch_tensor_io.cpp`
- Python: `tutorials/008_numpy_torch_tensor_io/numpy_torch_tensor_io.py`

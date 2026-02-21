# 015 Graph Model Hybrid

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Labels | graph, hybrid, stage-model, mpk |

## Concept
Combine graph stages with model execution in one hybrid runtime path.

## Learning Process
1. Model-hybrid stage consumes tensor-shaped samples.
2. Use stage-model node when MPK exists, else fallback stage.
3. Inspect output rank to reason about stage boundaries.
4. Graph model stage expects tensor input with model-compatible dimensions.
5. Run stage-model hybrid when MPK exists, otherwise stage fallback.
6. Read output rank and payload shape to reason about stage boundaries.

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

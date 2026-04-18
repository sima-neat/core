# 007 Plug a Model Into Your Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Labels | session, composition, patterns |

## Concept
This chapter compares practical **session composition patterns** for building real applications.

The name "session patterns" here means different ways to construct a runnable `Session`, not different inference algorithms. In the source, you will see three common composition styles:
- **Direct session**: explicitly add `Input` + `Output` nodes yourself.
- **Model-default session**: use `model.session()` to inject the model’s default pipeline group.
- **Model-attached session**: use `model.session(session_options)` to control appsrc/appsink inclusion and naming when attaching model groups to larger systems.

Why this matters:
- Teams often start with direct sessions for clarity, then move to model-backed composition for scale.
- `Model::SessionOptions` / `ModelSessionOptions` helps keep graph wiring explicit in multi-camera or multi-model deployments.
- Consistent naming (`upstream_name`, `name_suffix`, `buffer_name`) improves diagnostics and backend graph readability.

Reference:
- [Session](/getting-started/programming-model/session)
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Build a minimal direct session and validate it can run end-to-end.
2. Construct model-backed session variants (`model.session()` and `model.session(options)`).
3. Compare generated backend graphs with `--print-gst` to understand composition differences.
4. Execute a deterministic sync run and validate completion with `CHECK`, `SIGNATURE`, and `[OK]`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_007_session_patterns
python3 tutorials/007_session_patterns/session_patterns.py
```

## Source Files
- C++: `tutorials/007_session_patterns/session_patterns.cpp`
- Python: `tutorials/007_session_patterns/session_patterns.py`

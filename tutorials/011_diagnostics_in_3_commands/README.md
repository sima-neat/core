# 011 Diagnostics In 3 Commands

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | <10 minutes |
| Labels | diagnostics, debugging, observability |

## Concept
This tutorial gives you a compact three-step diagnostics workflow you can run before deep debugging.

The goal is to answer three questions quickly:
1. Is the session contract/build valid?
2. Does one run succeed with metrics enabled?
3. What do runtime diagnostics report about performance and behavior?

This chapter is especially useful when onboarding new models or environments, because it provides a repeatable triage baseline before diving into plugin internals.

Reference:
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Validate session contract and backend parse path (`validate()`).
2. Run one deterministic frame with metrics enabled.
3. Inspect runtime stats/report/diagnostic summary outputs.
4. Use `CHECK`, `SIGNATURE`, and `[OK]` markers to gate readiness.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_011_diagnostics_in_3_commands
python3 tutorials/011_diagnostics_in_3_commands/diagnostics_in_3_commands.py
```

## Source Files
- C++: `tutorials/011_diagnostics_in_3_commands/diagnostics_in_3_commands.cpp`
- Python: `tutorials/011_diagnostics_in_3_commands/diagnostics_in_3_commands.py`

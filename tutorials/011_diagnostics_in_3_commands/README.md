# 011 Diagnostics In 3 Commands

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | <10 minutes |
| Labels | diagnostics, debugging, observability |

## Concept
Diagnose common runtime issues quickly with a compact debug workflow.

## Learning Process
1. Parse flags and establish deterministic defaults.
2. Exercise the chapter's primary runtime path.
3. Emit checks and machine-parseable signature.

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

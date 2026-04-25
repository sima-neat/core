# 011 Diagnose and Profile a Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | <10 minutes |
| Model | None |
| Labels | diagnostics, debugging, observability |

## Concept

Run three checks — `session.validate()`, one metrics-enabled `run.run()`, and `run.stats()` + `run.report()` — to answer whether a pipeline is wired correctly and how it is performing. This is the triage baseline before deep debugging.

The three checks answer three questions:
1. Is the session contract/build valid? (`validate()`)
2. Does one run succeed with metrics enabled? (`build(..., RunOptions(enable_metrics=True))` + `run.run()`)
3. What do runtime diagnostics report? (`run.stats()`, `run.report()`, `run.diagnostics_summary()`)

Especially useful when onboarding new models or environments — repeatable, fast, and catches most misconfiguration before it becomes a multi-hour debugging session.

**APIs introduced**
- `session.validate()` — contract-level check, returns a report with `error_code`.
- `pyneat.RunOptions()` with `enable_metrics=True` and `output_memory=OutputMemory.Owned`.
- `run.stats()` — `inputs_enqueued`, `outputs_pulled`, `avg/min/max_latency_ms`.
- `run.report()`, `run.diagnostics_summary()` — structured runtime diagnostics.

**Prerequisites**
Chapter 002 or 003 (Session/Run basics).

**References**
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Validate session contract and backend parse path (`validate()`).
2. Run one deterministic frame with metrics enabled.
3. Inspect runtime stats/report/diagnostic summary outputs.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/011_diagnose_a_pipeline/diagnose_a_pipeline.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_011_diagnose_a_pipeline
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_011_diagnose_a_pipeline
./build/tutorials-standalone/tutorial_011_diagnose_a_pipeline
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/011_diagnose_a_pipeline/diagnose_a_pipeline.cpp`
- Python: `tutorials/011_diagnose_a_pipeline/diagnose_a_pipeline.py`

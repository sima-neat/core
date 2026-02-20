# Diagnostics and Failure Triage

Preferred debugging sequence:

1. Build with diagnostics enabled (default runtime report path).
2. Call `Session::validate(...)` before runtime when possible.
3. For runtime issues, inspect:
   - `PipelineRun::last_error()`
   - `PipelineRun::diagnostics_summary()`
   - `PipelineRun::report(...)`

Common failure classes:

- Missing plugins / bad plugin path.
- Caps negotiation mismatch.
- Input shape/format mismatch at appsrc boundary.
- Timeout from downstream starvation.

Recommended generated error-handling pattern:

- Fail fast with explicit message on `PullStatus::Error`.
- Handle `PullStatus::Timeout` and `PullStatus::Closed` separately.
- Preserve and log `PipelineReport` context when available.

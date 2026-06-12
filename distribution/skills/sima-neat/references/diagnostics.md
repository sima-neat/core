# Diagnostics and Failure Triage

Preferred debugging sequence:

1. Build with diagnostics enabled (default runtime report path).
2. Call `Graph::validate(...)` before runtime when possible.
3. For runtime issues, inspect:
   - `Run::last_error()`
   - `Run::start_measurement()` / `MeasureReport::to_text()`
   - `NeatError::report()` when an operation throws a structured runtime error

Common failure classes:

- Missing plugins / bad plugin path.
- Caps negotiation mismatch.
- Input shape/format mismatch at appsrc boundary.
- Timeout from downstream starvation.

Recommended generated error-handling pattern:

- Fail fast with explicit message on `PullStatus::Error`.
- Handle `PullStatus::Timeout` and `PullStatus::Closed` separately.
- Preserve and log `PipelineReport` context when available.

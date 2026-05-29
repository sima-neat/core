# Runtime resource and timeout hardening plan

## Context

We investigated an intermittent failure:

```text
[ERR] [misconfig.caps] Graph::build(input): set_state timed out after 30000 ms.
```

The exact `set_state` timeout did not reproduce deterministically, but the investigation found a
separate, reproducible runtime-resource failure:

- RPMsg channel lock `/tmp/rpmsg_lock_rpmsg3` is owned by `root:root` and not writable by `sima`.
- That leaves only 3 usable EV/RPMsg channels for normal test runs.
- Stale/orphan YOLO processes can hold RPMsg lock FDs indefinitely.
- Synthetic saturation of the usable locks causes EV work to fail or time out waiting for output.
- MLASHM model load occurs inside GStreamer state transition, so a slow PipelineManager/MLASHM
  request can consume the full Graph `set_state` timeout.

This plan breaks the work into five fix streams and gives each one clear validation criteria.

## Guiding principles

1. Prefer bounded waits with explicit diagnostics over immediate failure or unbounded blocking.
2. Do not hide device/runtime health issues behind silent retries.
3. Keep timeout ownership deterministic: the innermost blocking component should time out first and
   report the root cause.
4. Cleanup must be reliable even on failure, timeout, SSH disconnect, or parent shell exit.
5. All DevKit tests must run on ARM/DevKit only. Check `file <binary>` before execution.

## Phase 0: Baseline diagnostics before changing behavior

### Goal

Create a repeatable snapshot that proves which resource is exhausted and which process owns it.

### Tasks

- Add or document a preflight command that captures:
  - `/dev/rpmsg*`
  - `/tmp/rpmsg_lock*` owner/mode
  - current lock holders via `lslocks`
  - processes with `/dev/rpmsg*` or `/tmp/rpmsg_lock*` FDs
  - `simaai-appcomplex`, `simaai-pipeline-manager`, and `ssh` service state
  - root/data filesystem free space
- Make this preflight easy to run before EVO/YOLO matrix scripts.

### Done when

- We can capture a single artifact showing channel permissions, active owners, and service state.

## Phase 1: Unblock the fourth RPMsg channel

### Goal

Make all four RPMsg channel locks usable by the `sima` runtime user.

### Likely fix locations

- Runtime/package setup for `/tmp/rpmsg_lock_rpmsg*`
- `RpmsgTransport` lock creation path:
  - `/home/docker/sima-cli/internals/sima-ai-soc-pipeline/dispatcher/src/rpmsgtransport.cc`

### Tasks

- Fix creation/ownership of `/tmp/rpmsg_lock_rpmsg3` so it matches the other lock files:

  ```text
  owner/group: sima:sima or root:sima
  mode: writable by sima group/user
  ```

- Make the setup robust when a stale root-owned lock file already exists.
- Improve diagnostics when a candidate lock file exists but cannot be opened due to permissions.
- Decide whether this belongs in:
  - service startup/tmpfiles rule,
  - package install script,
  - RPMsg transport initialization,
  - or a combination.

### Validation

- As `sima`, verify:

  ```text
  open O_RDWR /tmp/rpmsg_lock_rpmsg0..3 succeeds
  ```

- Run a synthetic 4-channel acquisition test and confirm four independent lock holders are possible.

## Phase 2: Bounded wait for RPMsg channel acquisition

### Goal

When all EV/RPMsg channels are busy, wait for a configurable deadline and then fail with a precise
resource-exhaustion error.

### Current behavior

`RpmsgTransport` already retries:

```text
SIMA_RPMSG_ACQUIRE_RETRIES default 120
SIMA_RPMSG_ACQUIRE_BACKOFF_MS default 100
```

That is roughly 12 seconds, but the error is not explicit enough and does not distinguish:

- all locks busy,
- permission denied,
- endpoint missing,
- kernel/device wedge,
- or stale holder process.

### Tasks

- Add or formalize a single acquisition deadline, for example:

  ```text
  SIMA_RPMSG_ACQUIRE_TIMEOUT_MS
  ```

- Keep retry/backoff compatible with existing envs, but report the effective deadline.
- Track and report per-candidate failure reasons:
  - `EACCES` / permission denied on lock file,
  - lock busy,
  - `/dev/rpmsg*` open failure,
  - no endpoint candidates.
- On final failure, include:
  - attempted nodes,
  - lock paths,
  - effective timeout,
  - best-effort current lock holders when debug is enabled.
- Avoid unbounded `/proc` scans on hot paths unless a debug flag is enabled.

### Validation

- Synthetic lock saturation should fail after the configured deadline with a clear message like:

  ```text
  RPMsg channel acquisition timed out after N ms: all candidates busy
  ```

- Permission-denied lock file should report permission failure, not generic channel exhaustion.

## Phase 3: Prevent stale processes from holding channels

### Goal

Ensure wrappers/tests do not leave orphaned EVO/YOLO processes holding RPMsg locks after failure,
timeout, SSH disconnect, or cleanup.

### Likely fix locations

- YOLO wrapper:
  - `/home/docker/sima-cli/tmp/run_yolov8_matrix.sh`
- EVO wrapper:
  - `/workspace/tmp/run_evo_tput_w50_m250_installed_runtime_workspace_fresh.sh`
- Potential test helper utilities if wrappers are generated from repo scripts.
- Stage runner cache:
  - `src/pipeline/runtime/StageRun.cpp`

### Tasks

- In shell wrappers:
  - run remote tests in their own process group/session,
  - use `timeout -k`,
  - trap `EXIT`, `INT`, and `TERM`,
  - kill the full child process group on failure,
  - wait/reap children before deleting remote directories,
  - collect lock/process diagnostics before cleanup.
- Add a post-run assertion:

  ```text
  no preproc_yolov8_matrix_test/yolov8_variant_route_matrix_test/evo_tput_bench process remains
  no RPMsg locks are held by those processes
  ```

- Investigate whether `StageRun`'s process-lifetime cache should support an explicit test-only
  cache release API. It currently stores `Run` objects in a static map, which intentionally keeps
  pipelines/dispatchers alive until process exit. That is acceptable for performance, but tests that
  iterate many models need a deliberate cleanup option if they are meant to release channels between
  cases.

### Validation

- Force a wrapper timeout mid-run and verify:
  - no matching test processes remain,
  - no `/tmp/rpmsg_lock_rpmsg*` locks are held by the killed test,
  - the remote directory is not removed until the child process is gone.

## Phase 4: Investigate slow or failed `RPCMLASHM::load`

### Goal

Determine why MLASHM model load can stall long enough to make Graph `set_state` hit 30 seconds.

### Why this is likely relevant

`neatprocessmla` calls model load during GStreamer startup/state transition:

```cpp
self->priv->model_handle = self->priv->dispatcher->load(self->priv->model_path.c_str());
```

`RPCMLASHM` has a default request timeout of 60 seconds:

```cpp
kDefaultMlaTimeoutMs = 60000
```

Graph `set_state` defaults to 30 seconds. Therefore a slow MLASHM/PipelineManager request can cause
Graph to time out before MLASHM returns a meaningful error.

### Tasks

- Add instrumentation or enable existing instrumentation around:
  - model path,
  - load request start/end,
  - wait time,
  - MLASHM error code/detail,
  - PipelineManager heartbeat gaps,
  - appcomplex/pipeline-manager service health.
- Correlate `RPCMLASHM::load` latency with:
  - EVO immediately before YOLO,
  - filesystem pressure,
  - MLASHM/appcomplex restarts,
  - PipelineManager MQTT heartbeat stalls,
  - remoteproc/kernel errors.
- Build a focused repro:
  - EVO warm/stress run,
  - immediate first YOLO model,
  - full debug env enabled,
  - no service restart unless diagnostics are collected first.

### Validation

- We can classify a slow load as one of:
  - MLASHM request queue backlog,
  - PipelineManager/appcomplex responsiveness issue,
  - model extraction/filesystem stall,
  - shared-memory/dispatcher fault,
  - or external device overload.

## Phase 5: Synchronize Graph and MLASHM timeouts

### Goal

Make timeout ownership deterministic so users see the true failing subsystem instead of a generic
`Graph::build(input)` `set_state` timeout.

### Current mismatch

```text
Graph set_state input timeout: 30000 ms
RPCMLASHM default timeout:     60000 ms
```

### Preferred behavior

Either:

1. MLASHM load timeout is shorter than Graph `set_state` timeout, so `neatprocessmla` posts the real
   bus error first, or
2. Graph `set_state` timeout is longer than the maximum expected MLASHM load timeout, or
3. Best option: a single state-transition deadline is propagated down to blocking dispatcher calls.

### Tasks

- Define timeout policy:
  - default Graph state timeout,
  - default MLASHM load timeout,
  - margin between inner and outer timeout,
  - env override precedence.
- Ensure timeout messages include both values.
- Investigate the detached `set_state` thread behavior:
  - today Graph throws after timeout while the detached `gst_element_set_state()` may still be
    blocked;
  - cleanup/unref while the state thread is still inside GStreamer may worsen wedges;
  - plan a safe cancellation/join strategy or defer destructive cleanup until the state operation
    has unwound.

### Validation

- Inject or simulate a slow MLASHM load:
  - MLASHM timeout should fire first with a specific dispatcher/MLASHM error, or
  - Graph timeout should explicitly state it exceeded the inner MLASHM deadline.
- No detached state-change thread should continue holding resources after timeout cleanup.

## End-to-end validation matrix

Run after each major fix, using ARM binaries only on the DevKit:

1. Clean-state YOLO:

   ```bash
   run_yolov8_matrix.sh preproc
   run_yolov8_matrix.sh both
   ```

2. EVO then YOLO:

   ```bash
   run_evo_tput_w50_m250_installed_runtime_workspace_fresh.sh
   run_yolov8_matrix.sh both
   ```

3. Synthetic RPMsg saturation:
   - hold 1, 2, 3, then 4 locks,
   - verify bounded wait and diagnostic quality.

4. Wrapper timeout cleanup:
   - kill parent SSH/wrapper mid-run,
   - verify no stale test process and no held RPMsg lock.

5. MLASHM slow-load diagnostics:
   - capture `RPCMLASHM::load` timing,
   - confirm timeout ordering.

## Suggested implementation order

1. Fix RPMsg lock permissions/setup for the fourth channel.
2. Harden wrappers so stale child processes cannot survive failures.
3. Add bounded RPMsg acquire diagnostics/deadline.
4. Align Graph/MLASHM timeouts.
5. Continue root-causing slow `RPCMLASHM::load` with the improved diagnostics.


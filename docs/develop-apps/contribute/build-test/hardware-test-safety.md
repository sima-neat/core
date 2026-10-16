---
title: Hardware Test Ownership
description: Run accelerator tests without treating a timeout as DMA retirement
---

# Hardware test ownership

A test deadline is a failure deadline, not permission to kill a DMA owner.
Hardware qualification uses `tools/hardware_test_supervisor.py` on the DevKit.
The persistent supervisor owns the actual accelerator process. CTest, SSH and
Python clients own reporting connections only.

## Configure and admit

Build Core with `NEAT_GUARDED_HARDWARE_TESTS=ON`, or configure CMake with
`SIMANEAT_GUARDED_HARDWARE_TESTS=ON`. Native CTest targets become reporting
proxies; test arguments, environment, labels, resource locks and existing
`TIMEOUT` properties remain unchanged. Installed extras ship the same supervisor
client. The performance matrix requires `SIMA_HARDWARE_SUPERVISOR_SOCKET`.
`SIMA_PERF_HOST_FIXTURES_ONLY=1` is exclusively for non-AArch64 test fixtures;
it cannot enable unguarded AArch64 matrix execution.

A privileged qualification owner supplies a new private, single-use state
folder and configuration. The supervisor refuses old admission/fail-stop folders.
The configuration pins:

- `identity`: `admission_id`, `boot_id`, `cohort_sha256`, `lease_pid`,
  `lease_start_ticks`, `observer_id`.
- `state_dir`, `socket` inside that directory, and `supervisor_sha256`.
- `checker`: exact root-owned read-only checker `command`, executable `sha256`
  and resource-free probe `timeout_sec`.
- `executables`: absolute regular-file paths and SHA256 identities, exact
  `arguments`, `deadline_sec` and protected `required_env` values. Multiple
  sanctioned command/deadline variants belong in `variants`.

Use `deadline_sec: null` only when the original test has no timeout. A CTest
proxy requests the exact approved deadline; the matrix requests its unchanged
180-second deadline. Reject loader injection and pin canonical runtime/plugin
paths and numerical-oracle settings in `required_env`.

The checker receives `--phase launch|active|retired --owner-json PATH`. It must
return fresh JSON with `schema: 1`, `status: "ready"`, the exact pinned `identity`
and `checked_at_epoch`. It must never launch accelerator work or mutate hardware.

The owner record binds PID, start ticks, boot ID, session ID, run ID and the full
request SHA256. Before allowing `exec`, the supervisor makes ownership durable
and rechecks admission and client liveness while the resource-free child waits
at a launch barrier. The active loop records session members, verifies identity,
checks live admission and scans durable logs for unretired/quarantined state.

After a real nonnegative exit, `exit.json` binds the owner, return code, deadline,
request hash and observed completion time. The `retired` checker must add:

- `retirement.proven: true`;
- `kind: "authoritative-driver-retirement"` or `"reviewed-graceful-exit"`;
- a nonempty evidence `token`;
- `scope: "whole-owner-process-tree-and-dma"`;
- `owner`: the exact owner record.

The checker implementation must justify that proof using the reviewed execution
and transport completion contract, actual ownership/descendant checks, fresh
fault observers and current driver/firmware/cohort identity. Sampled zero jobs,
process exit, elapsed time or a healthy SSH connection alone are not proof.
Pure CPU fixtures can use a specifically reviewed safe-exit contract; do not
invent driver evidence for them.

## Timeout and recovery

The supervisor never sends TERM/KILL on timeout, disconnect, cancellation,
unknown retirement or observer failure. It seals irreversible fail-stop, retains
the owner and durable logs, and rejects later launches. The matrix preserves
`ENV_BROKEN / ENV_TIMEOUT` and records remaining scenarios as **not run**; it
never fabricates passes or accepted skips. Late output cannot reverse timeout.
A naturally exited, fully retired numerical failure remains a failed case,
without being confused with unknown ownership.

An output-file completion condition is not retirement. An optional application
stop request requires an exact approved completion/argument variant and pinned
handler/drain review receipt. Only that explicitly reviewed SIGTERM request is
allowed; it is never a cleanup action and never escalates to SIGKILL. Real exit
and authoritative retirement are still required before acceptance.

Do not stop remoteproc, unload/unbind drivers, delete borrowed buffers or kill
an uncertain owner. Use the device's existing guarded recovery procedure.
After recovery, establish a new admission; the old one is never reopened.
Restore and verify the original firmware, service and recovery policy after
any temporary firmware trial.

## Clients and evidence

Python clients use `run_owned(command, cwd, env, deadline_sec, evidence_dir,
socket_path, completion=None, graceful_stop=None)`. The CLI supports
`run-json --request FILE --result FILE --socket PATH`. Results identify
`completed`, `timeout`, `fail_stop` or `refused`, owner identity, real return code,
retirement proof and durable stdout/stderr paths. Client death cannot kill the
actual hardware child; the daemon must run independently on the target, not as
an SSH foreground process subject to session cleanup.

Run harmless native host tests before hardware admission. Qualification reports
must distinguish unchanged workloads/oracles/deadlines under this reviewed
safety harness from execution of an unmodified older harness or an actual
hosted CI workflow.

For Python 3.13 and newer, PyNeat metadata selects NumPy `>=2.1,<3`; older Python
selects `>=1.24,<2`. Metadata resolution does not prove ndarray numerical parity.
Run the installed target's dtype, stride, copy/zero-copy and lifetime tests
before qualifying that Python/NumPy cohort.

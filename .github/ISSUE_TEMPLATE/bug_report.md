---
name: Bug Report
about: Report a problem with Neat Core
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

Describe the problem and its impact on your application.

## Steps to Reproduce

1. Provide the smallest C++ or Python example, tutorial, or command that reproduces the problem.
2. Include the relevant Graph/Run configuration, inputs, and execution steps.
3. Describe the observed result and how often it occurs.

For build, installation, or documentation problems, provide the failing command
or page instead. Mark sections that do not apply as not applicable.

## Expected Behavior

Describe the expected result and how it differs from the observed behavior.

## Environment

- Neat Core version, commit, or package reference:
- C++ or Python API; compiler or Python/pyneat version, if applicable:
- Host/target OS and architecture; hardware platform, if applicable:
- Neat SDK/platform version, if applicable:
- Scheduler and LLiMa package versions, if relevant and known:
- GStreamer version, if relevant:

## Pipeline and Model Context (if applicable)

- Affected Model, Graph, Run, node, or input/output path:
- Model package name/version and Model Compiler version, if relevant and known:
- Input/output tensor shapes, dtypes, and layouts; media source and format:
- Sync/Async execution, stream count, and queue settings, if relevant:
- For a regression, the last known working version or commit:

Include relevant, redacted MPK manifest fields (`mpk.json` / `*_mpk.json`) when
needed to explain model contracts. Describe how to reproduce with public or
synthetic inputs where possible.

## Diagnostics

Include the full error message and relevant `GraphReport`, build output,
GStreamer log, or stack trace. For hangs or performance regressions, include
timeouts, expected/observed latency or throughput, and the measurement setup.

Redact credentials, private stream URLs, and customer data. Include only model
metadata or artifacts you are authorized to share.

## Additional Context

Link related issues and include any workaround or other useful context.

# core-triage

Use this skill when triaging GitHub issues in the `sima-neat/core`
repository.

## Repository Scope

`core` is the SiMa NEAT framework repository. It covers the public C++ API,
Python bindings, graph/model/tensor/sample abstractions, pipeline construction,
runtime orchestration, diagnostics, tutorials, documentation, release artifacts,
and tests for the NEAT developer experience.

Core issues may describe behavior in source builds, installed SDK environments,
DevKit execution, documentation, examples, Python packaging, model archive
handling, or generated GStreamer pipelines.

## First Pass

Before classifying the issue, check whether it is likely a duplicate of an
existing issue from the provided issue context. Compare the title, error text,
affected API or tutorial, command output, model/package reference, and reported
environment against existing discussion in the issue thread and any available
repo context.

If the issue appears to duplicate another issue, do not propose closing it and
do not use the `duplicate` label. Set `needs_human_review` to true, keep the
primary category aligned with the underlying problem, and mention in the public
comment that it may overlap with an existing report if the evidence is strong.
If the duplicate relationship is uncertain, ask for the missing detail needed to
confirm whether it is the same failure mode.

## Safety And Sensitivity

If the issue appears security-sensitive, customer-confidential, or likely to
contain private customer assets, set `needs_human_review` to true, keep the
public comment conservative, and do not request extended analysis.

Do not ask the reporter to upload private models, customer videos, credentials,
tokens, proprietary logs, or full device images. Ask for redacted command output,
minimal reproductions, public model references, and version information instead.

`sima-neat/internals` is private and is not in the cross-reference allowlist. If
an issue likely requires internals/plugin/runtime implementation inspection,
route it to human review rather than requesting extended analysis.

## Classification

Use `bug` when the issue describes a command, API, tutorial, pipeline, runtime,
binding, packaging, or documentation behavior that fails or regresses from an
expected workflow.

Use `enhancement` when the issue requests new API behavior, additional node or
graph capability, better diagnostics, improved examples, broader platform
support, or developer-experience improvements.

Use `documentation` when the issue is primarily about docs, tutorials, examples,
release notes, API reference, confusing instructions, or missing explanation.

Use `question` when the issue asks how NEAT works and does not yet identify a
specific defect or requested change.

Use `help wanted` only when the issue is clearly suitable for external
contribution and does not require private hardware, private repositories,
release credentials, or internal runtime knowledge.

Use `needs-repro` when the issue is plausible but lacks the minimum command,
code snippet, environment, model/package reference, or error output needed for a
maintainer to reproduce it.

Do not propose `duplicate`, `invalid`, or `wontfix`. If an issue appears
duplicate, out of scope, or unsupported, set `needs_human_review` to true and
explain the evidence neutrally.

## Area Routing

Set `area` to one of these short names when the issue matches:

- `cpp-api`: public C++ headers under `include/`, API compatibility,
  `Graph`, `Model`, `Run`, `Tensor`, `Sample`, nodes, node groups, or builder
  APIs.
- `python-api`: `python/`, `pyneat`, nanobind bindings, Python packaging,
  NumPy/PyTorch interop, or Python tests.
- `model-contract`: model archive loading, MPK manifest handling, tensor
  contracts, routing, dtype/shape interpretation, preprocess/inference/
  postprocess fragments, or model validation errors.
- `pipeline-runtime`: graph build/run behavior, push/pull execution, async
  runs, GStreamer pipeline generation, caps negotiation, teardown, timeouts, or
  runtime lifecycle behavior.
- `diagnostics`: structured errors, `GraphReport`, `MeasureReport`, profiling,
  latency/throughput reporting, power measurement, or troubleshooting output.
- `tutorials`: tutorial source, tutorial build targets, tutorial assets,
  example commands, or generated tutorial docs.
- `docs`: documentation outside tutorials, API reference, release notes,
  getting-started pages, troubleshooting, or docs publishing.
- `build-ci`: CMake, `build.sh`, CI workflows, package build failures, tests,
  sanitizers, smoke tests, release gates, or formatting checks.
- `packaging`: Debian packages, extras tarballs, Vulcan package metadata,
  install smoke tests, artifact layout, or release artifact contents.
- `devkit-runtime`: behavior that only appears on DevKit/eLxr/Modalix hardware,
  hardware execution, device-side runtime, installed SDK paths, or board
  environment differences.
- `genai`: GenAI, LLM/VLM tutorials, LLima integration, serving workflows, or
  related model composition.
- `unknown`: not enough information to route.

## Common Missing Information

For API or source-build issues, ask for:

- core commit, branch, or package version
- host OS and architecture
- compiler version when relevant
- exact build or test command
- minimal code snippet or tutorial name
- complete error output

For runtime or DevKit issues, ask for:

- SDK image/version or installed NEAT package version
- board type and software version when available
- exact command or code path
- model archive name or public model reference
- redacted pipeline/runtime logs
- whether the issue reproduces on host, DevKit, or both

For Python binding issues, ask for:

- Python version
- install mode, such as wheel, editable install, or source build
- `pip show` or package version output
- minimal Python snippet
- full traceback

For docs/tutorial issues, ask for:

- page URL or tutorial path
- command that was run
- expected output versus actual output
- SDK/core version used

## Extended Analysis

Most issues should be triaged from the issue text, command output, and this
repo's local triage guidance. Do not request extended analysis just because the
issue mentions SDK, sima-cli, Model Compiler, or Insight.

Request extended analysis only when the issue includes enough specific detail
that checking another public repository can materially improve routing or the
next maintainer action.

Allowed cross-reference repositories:

- `sima-neat/sdk`
- `sima-neat/sima-cli`
- `sima-neat/model-compiler`
- `sima-neat/insight`

Request `sima-neat/sdk` when:

- the issue involves SDK image contents, SDK install behavior, DevKit pairing,
  SDK environment paths, packaged NEAT artifacts, or SDK workflows that surface
  a core issue.
- the issue includes an SDK image tag, branch, workflow name, install log, or
  artifact path that can be checked against SDK scripts/workflows/docs.

Request `sima-neat/sima-cli` when:

- the issue involves `sima-cli neat install`, SDK setup/start, package
  installation, Vulcan metadata lookup, playbook installation, or CLI guidance
  that routes users to core artifacts/docs.
- the issue includes a concrete CLI command, package ref, or CLI error that can
  be checked against sima-cli behavior.

Request `sima-neat/model-compiler` when:

- the issue involves compiled model archives consumed by core, BF16/INT8
  behavior, MPK generation, compiler examples, model archive layout, or
  compiler output that affects `Model`/pipeline behavior.
- the issue includes enough compiler output, model artifact name, or example
  path to compare against model-compiler docs or workflows.

Request `sima-neat/insight` when:

- the issue involves Insight using core pipelines, media/source workflows,
  browser/API behavior that depends on core output, or Insight packaging of core
  examples/artifacts.
- the issue includes Insight logs, workflow names, or package refs that can be
  checked in the public Insight repository.

Do not request extended analysis for issues that mainly require private
internals/plugin implementation inspection. Mark those as `needs_human_review`
and explain that maintainer review is needed.

When extended analysis is useful, set:

- `extended_analysis_required`: `true`
- `extended_analysis_repos`: only the specific repo or repos needed from the
  allowlist above
- `extended_analysis_reason`: a concise explanation of what should be checked

If the issue lacks concrete logs, commands, code snippets, package refs, model
names, or file paths, do not request extended analysis. Ask for the missing
information instead.

## Comment Style

Write a public triage comment with 2-4 short paragraphs:

- acknowledge the issue and state the likely area
- state whether it appears actionable, needs repro details, or needs human
  review
- mention specific evidence from the report
- ask only for missing information that would materially unblock investigation

Do not include a mechanical label list unless it is useful to the reporter. Do
not claim a root cause unless the issue text or checked public repository
context provides strong evidence.

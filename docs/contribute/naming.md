---
title: Naming Contract
description: Canonical product, API, and type naming for SiMa Neat
sidebar_position: 1
---

# Naming Contract

This document defines the canonical naming contract for the SiMa Neat codebase.

## Canonical names

- Product name: `SiMa Neat`
- CMake project: `SimaNeat`
- C++ namespace: `simaai::neat`
- Core runtime types: `Model`, `Graph`, `Run`

## Public include surface

Use headers under `include/` as the public source of truth.

Examples:

```cpp
#include "model/Model.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
```

## Legacy aliases

Legacy names are supported only as compatibility references and migration guides.

- `PipelineSession` -> `Graph`
- `PipelineRun` -> `Run`
- `NeatModel` -> `Model`
- `InputAppSrc` -> `Input`
- `OutputAppSink` -> `Output`

Do not introduce new public docs/examples using legacy symbols.

## Policy

- New user-facing docs must use canonical names.
- Legacy terms are allowed only in migration notes, compatibility comments, or
  clearly marked deprecation sections.
- CI enforces this via `scripts/ci/check_naming_and_conflicts.sh`.

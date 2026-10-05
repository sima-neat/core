---
name: neat-pcie-application-builder
description: Build host-side C++ or Python applications that run compiled Neat models on a connected Modalix PCIe Card with pcie::Model or pyneatpcie.Model. Use for application code, not pcie::Runtime, DevKit-local APIs, model compilation, or PCIe package maintenance.
---

# Neat PCIe Application Builder

## Overview

Build native host applications against the installed Neat PCIe `Model` API. Treat the installed
PCIe header, Python module, and packaged tutorials as the source of truth. This is a separate API
from the Neat Library used inside the SDK or directly on a DevKit.

The expected input is an already compiled Neat model archive and the requested C++ or Python host
application behavior. Follow explicit user requirements when they are compatible with the installed
public API; treat the defaults below as guidance.

## Choose References

- Read `references/source-of-truth.md` when generating or reviewing API use, inspecting an
  installed release, or resolving version differences.
- Read `references/model-lifecycle.md` when implementing build, inference, pipelining, cleanup, or
  multiple-model behavior.
- Read `references/tensors-and-images.md` when constructing inputs, consuming outputs, handling
  multiple inputs, or choosing tensor mode versus image mode.
- Read `references/model-options.md` when configuring a card, queue, preprocessing, MLA-only mode,
  or object decode behavior.
- Read `references/build-and-validation.md` when building, testing, or reporting validation. Do not
  claim connected-card execution from compile or import checks alone.

## Defaults

- In C++, include `<simaai/neat/pcie/Model.h>` and use the
  `simaai::neat::pcie` namespace.
- In Python, import `pyneatpcie as pcie` from the PCIe host Python environment.
- In tensor mode, inspect `model.info()` before allocating or naming model-ready inputs. In image
  mode, treat that information as the card-side preprocessing output contract, not the submitted
  image contract.
- Enable `mla_only` only when the application owns the model's dtype conversion; then
  `model.info()` is the MLA's own contract and, for INT8 tensors, its `quant` parameters are the
  only valid source for the conversion.
- Use `run()` for ordinary request/response inference. Use `push()` and `pull()` only when the
  application benefits from bounded pipelining.
- For performance-sensitive tensor pipelines, prefer a bounded ring of reusable contiguous input
  buffers. Wrap C++ storage with `Tensor::from_external()` or Python arrays with
  `Tensor.from_numpy(..., copy=False)`, and recycle a buffer only after its matching result is
  pulled. Use simpler owning or copying constructors when reuse and staging overhead do not matter.
- Use finite build and inference timeouts in applications that must fail predictably.
- Close every successfully built model on normal and error paths. Prefer a Python context manager.

## Completion

For implementation requests, provide runnable source, required dependency and build commands, a
concrete run command, and the validation level reached. State clearly when connected-card behavior
was not exercised.

## Boundaries

- Use `pcie::Model` and the supporting public types declared by `Model.h` only.
- Do not include `Runtime.h` or generate code using `pcie::Runtime`, `ModelConfig`, `ModelId`,
  `RequestId`, `Completion`, `load()`, `try_enqueue()`, or `retrieve()`.
- Multiple models are allowed as independent `Model` objects assigned to distinct physical queues.
- Do not substitute the regular Neat `Model`, `Graph`, `Node`, or `Run` APIs. They are not part of
  the PCIe host application surface.
- Do not use PCIe implementation headers, construct raw GStreamer pipelines, or launch
  `pcie-pipeline-builder` directly.
- Do not add model compilation or Model SDK workflows. The input is an already compiled Neat model
  archive.
- Verify behavior against the installed release instead of guessing from memory or another Neat
  environment.

---
name: neat-pcie-application-builder
description: Use when building host-side C++ or Python applications that run compiled Neat models on a connected Modalix PCIe Card through simaai::neat::pcie::Model or pyneatpcie.Model. Do not use for pcie::Runtime, DevKit-local Neat applications, model compilation, or PCIe package maintenance.
---

# Neat PCIe Application Builder

## Overview

Build native host applications against the installed Neat PCIe `Model` API. Treat the installed
PCIe header, Python module, and packaged tutorials as the source of truth. This is a separate API
from the Neat Library used inside the SDK or directly on a DevKit.

## Workflow

1. Confirm that the application runs on an Ubuntu host connected to a Modalix PCIe Card.
2. Establish the installed API surface by reading `references/source-of-truth.md`.
3. Read `references/model-lifecycle.md` for every application. Choose synchronous `run()` or
   pipelined `push()`/`pull()` and preserve the required build and close lifecycle.
4. Read `references/tensors-and-images.md` when constructing inputs, consuming outputs, handling
   multiple inputs, or choosing tensor mode versus image mode.
5. Read `references/model-options.md` when configuring a card, queue, preprocessing, or object
   decode behavior.
6. Before claiming success, read `references/build-and-validation.md` and run the checks possible
   on the current host. Distinguish compile/import validation from connected-card validation.

## Defaults

- In C++, include `<simaai/neat/pcie/Model.h>` and use the
  `simaai::neat::pcie` namespace.
- In Python, import `pyneatpcie as pcie` from the PCIe host Python environment.
- In tensor mode, inspect `model.info()` before allocating or naming model-ready inputs. In image
  mode, treat that information as the card-side preprocessing output contract, not the submitted
  image contract.
- Use `run()` for ordinary request/response inference. Use `push()` and `pull()` only when the
  application benefits from bounded pipelining.
- Use finite build and inference timeouts in applications that must fail predictably.
- Close every successfully built model on normal and error paths. Prefer a Python context manager.
- Keep generated applications runnable with explicit dependency, build, and run commands.

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

## References

- `references/source-of-truth.md`
- `references/model-lifecycle.md`
- `references/tensors-and-images.md`
- `references/model-options.md`
- `references/build-and-validation.md`

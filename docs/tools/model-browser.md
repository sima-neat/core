---
title: Model Browser
sidebar_position: 4
---

# Model Browser

The [Palette Model Browser](https://developer.sima.ai/models/) lists the models
that SiMa.ai compiles and measures for Modalix. Each model page shows the
performance and accuracy measured on a Modalix DevKit and offers the compiled
model as a download.

Use it when you want to:

- find a model for your task, such as object detection or text generation
- compare the performance and accuracy of models and their variants
- download a compiled model that is ready to run with Neat
- rebuild a model yourself from its public scripts

## Download a Model with sima-cli

`sima-cli` reads the same models from the terminal. You need no account.

Search for a model:

```bash
sima-cli models list -q rf_detr_nano
```

Select the model, then select a variant. A variant is one way of compiling the
model, named after the bit widths of its weights and activations. The command
shows the model card with the measured accuracy and performance, then asks
whether to download the model.

Download a model directly when you know its ID and variant:

```bash
sima-cli models download --id rf_detr_nano --variant w8a8_w16a16
```

The command saves the model under `models/<model id>/<variant>/`. Add
`--output <folder>` to save it somewhere else.

A model whose license forbids redistribution has no download. Rebuild it from
its scripts, as the next section describes.

For command details, see the [`sima-cli models`](/tools/sima-cli/models/)
reference.

## Reproduce a Model

Every model comes with the scripts that built and measured it, in the
[models repository](https://github.com/sima-neat/models). Each model has a
folder, `models/<model id>/`, with one script per step:

| Step | Script | Runs on |
| --- | --- | --- |
| Export the official model to ONNX | `convert.py` | your computer |
| Quantize and compile it | `compile.py` | Model Compiler in the Palette SDK |
| Measure performance | `performance.py` | Modalix DevKit |
| Measure accuracy | `accuracy.py` | Modalix DevKit |

The `model.yaml` file in the same folder lists the variants you can build under
`build_profiles`, and the datasets each step needs.

The [getting started guide](https://developer.sima.ai/models/docs/getting-started)
walks through every step with the commands to copy, and shows how to compare
your results with the published ones.

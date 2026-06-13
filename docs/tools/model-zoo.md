---
title: Model Zoo
sidebar_position: 5
---

# Model Zoo

The Model Zoo is a curated collection of precompiled and quantized models that
are ready to run on SiMa devices.

Use it when you want to:

- evaluate model accuracy and performance on Modalix hardware
- avoid manual compilation and quantization for a known model
- start from validated model artifacts
- select models built for a specific hardware target

The Model Zoo provides precompiled model artifacts for Neat C++ and PyNeat
applications, excluding GenAI models.

List available models:

```bash
sima-cli modelzoo list
```

Inspect a model before downloading it:

```bash
sima-cli modelzoo describe yolov5
```

Download a model artifact:

```bash
sima-cli modelzoo get yolov5s
```

Model names can vary by release. Use `sima-cli modelzoo list` first when you are
not sure which model identifier to use.

For command details, see the [`sima-cli modelzoo`](/tools/sima-cli/modelzoo/)
reference.

## GenAI Models

For GenAI, SiMa.ai provides precompiled LLM, VLM, and ASR model collections on
[Hugging Face](https://huggingface.co/simaai). Download them with the LLiMa CLI:

```bash
llima pull <model_name>
```

For example:

```bash
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

After downloading a GenAI model, use the LLiMa runtime to run it on the DevKit.
For setup and runtime commands, see [GenAI with LLiMa](/genai-llima/).

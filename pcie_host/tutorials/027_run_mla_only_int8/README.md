# 027 Run the MLA Only with INT8 Tensors

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | any archive compiled for direct MLA input and output |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

The default PCIe route sends FP32 tensors to the card, where the EV74 quantizes
them, the MLA runs, and the EV74 dequantizes the results back to FP32. An
application that already holds INT8 data, or that wants the quantization step
under its own control, can set `ModelOptions.mla_only`. The card then runs
nothing but the MLA: the host submits INT8 tensors that match the MLA ingress
contract and receives the raw INT8 heads. `model.info()` publishes the
quantization parameters of every tensor so the host can quantize and dequantize
with one equation:

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

## Walkthrough

One program quantizes an image on the host, runs the MLA-only route, dequantizes
the heads, and checks them against the default route on the same queue.

### Inspect the MLA-only contract {#step-inspect-contract}

Construct the `Model` with `mla_only` enabled. `info()` now reports INT8 inputs
and outputs, each with `quant.scales[0]` and `quant.zero_points[0]`. Inputs also
carry `input_range`, the floating-point domain the model was calibrated for. A
model with several inputs lists one INT8 tensor per input, in submission order.

### Quantize on the host {#step-quantize-on-host}

Resize the image to the ingress geometry, convert BGR to RGB, map the pixels
onto `input_range`, and apply the quantization equation with the ingress
parameters. Keep the dequantized values of the same codes: they are the exact
FP32 input the default route needs for a like-for-like comparison.

### Run the INT8 route {#step-run-int8}

`build()` starts a card pipeline that contains only the MLA. `run()` accepts the
INT8 tensors and returns one dense INT8 tensor per output, in the order and with
the names that `info().outputs` reported. The route rejects any other dtype; an
FP32 push fails instead of being quantized on the card.

### Dequantize and compare {#step-dequantize-and-compare}

Build a second `Model` without `mla_only` and send the dequantized FP32 values
through the default route. Dequantize the INT8 heads with each output's
parameters and print the largest deviation per head in units of that head's
scale. Both routes execute the same MLA program on the same codes, so the error
is zero.

## Run

Install the PCIe host package and download the tutorial bundle as described in
[Tutorial Setup](/tutorials/before-you-run).

This tutorial needs an archive that was compiled for direct MLA input and
output: the Model SDK `tessellate_parameters` with `enable_mla=True`, an `HWC`
DRAM layout on every input, and `HWC16` on every output. Model Zoo archives
tessellate on the EV74 instead and are rejected when `mla_only` is enabled:

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

Copy a qualifying archive into the extracted PCIe extras root, for example as
`model_mlatess_int8.tar.gz`, and pass its path with `--model`.

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model model_mlatess_int8.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

With a YOLOv8n archive compiled for direct MLA I/O, both versions print the
contract and a zero deviation for every head:

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00391965 zero_point=-128 range=[0, 1]
  output bbox_0 INT8 [80, 80, 64] scale=0.0828159 zero_point=-60
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 64] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

The default is card 0 and queue 0. Pass `--card N` only when using another card.

## In Practice

Enable `mla_only` when the application owns quantization: it already produces
INT8 from a sensor or an earlier model, it needs the raw INT8 heads for its own
postprocessing, or it wants to remove the EV74 stages from the card-side
latency. Read every scale, zero point, and input range from `model.info()`;
never copy them from another build of the model.

The route is all or nothing. Every input of a multi-input model must arrive as
INT8, and image preprocessing or box decode cannot be combined with `mla_only`.
Keep the default route when the host holds FP32 data and does not need to
control quantization.

For deployment diagnostics, continue with the
[PCIe model workflow](/develop-apps/development-workflow/pcie-model/).

## Source Files

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`

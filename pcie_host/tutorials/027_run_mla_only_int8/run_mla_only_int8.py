#!/usr/bin/env python3
"""Quantize on the host, run only the MLA over PCIe, and dequantize the INT8 results.

Usage:
  python3 run_mla_only_int8.py --model yolo26n-det-int8-b1.tar.gz [--card 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import pyneatpcie as pcie


BUILD_TIMEOUT_MS = 180_000
RUN_TIMEOUT_MS = 30_000
MAX_ERROR_SCALES = 0.05  # one twentieth of a quantization step
IMAGE_PATH = Path("share/sima-pcie-host/tutorials/assets/street-scene.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="yolo26n-det-int8-b1.tar.gz from the Model Zoo"
    )
    parser.add_argument("--card", type=int, default=0)
    return parser.parse_args()


def require_quant(spec: pcie.TensorInfo) -> tuple[float, int]:
    """Per-tensor parameters of the MLA-only contract.

    x = (q - zero_point) * scale
    q = clip(round(x / scale) + zero_point, -128, 127)
    """
    if spec.quant is None:
        raise RuntimeError(f"tensor '{spec.name}' publishes no quantization parameters")
    return spec.quant.scale, spec.quant.zero_point


def quantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    codes = np.rint(values.astype(np.float32) / np.float32(scale)) + zero_point
    return np.clip(codes, -128, 127).astype(np.int8)


def dequantize(codes: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (codes.astype(np.int32) - zero_point).astype(np.float32) * np.float32(scale)


def quantize_image(bgr: np.ndarray, spec: pcie.TensorInfo) -> np.ndarray:
    """Preprocessing of the reference model: one RGB HWC image with pixels in [0, 1].

    Resize, BGR to RGB, divide by 255. Another model needs its own recipe.
    """
    if len(spec.shape) != 3 or spec.shape[2] != 3:
        raise RuntimeError(f"the reference model takes a three-channel HWC input, got {spec.shape}")
    scale, zero_point = require_quant(spec)
    height, width, _ = spec.shape
    rgb = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
    return quantize(rgb.astype(np.float64) / 255.0, scale, zero_point)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model does not exist: {model_path}")
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode: {IMAGE_PATH}")
    connection = pcie.ConnectionOptions(card_id=args.card)

    # CORE LOGIC
    # STEP inspect-contract
    options = pcie.ModelOptions()
    options.mla_only = True
    with pcie.Model(str(model_path), options, connection) as model:
        info = model.info()
        if len(info.inputs) != 1:
            raise RuntimeError("the reference model has exactly one input")
        print("MLA-only contract:")
        for spec in info.inputs:
            scale, zero_point = require_quant(spec)
            print(
                f"  input {spec.name} {spec.dtype} {spec.shape} "
                f"scale={scale} zero_point={zero_point}"
            )
        for spec in info.outputs:
            scale, zero_point = require_quant(spec)
            print(
                f"  output {spec.name} {spec.dtype} {spec.shape} "
                f"scale={scale} zero_point={zero_point}"
            )
        # END STEP

        # STEP quantize-on-host
        ingress = info.inputs[0]
        codes = quantize_image(image, ingress)
        scale, zero_point = require_quant(ingress)
        fp32_input = dequantize(codes, scale, zero_point)
        int8_inputs = [pcie.Tensor.from_numpy(codes, copy=True, route_name=ingress.name)]
        # END STEP

        # STEP run-int8
        model.build(BUILD_TIMEOUT_MS)
        outputs = model.run(int8_inputs, RUN_TIMEOUT_MS)
    codes_by_head = {spec.name: output.to_numpy() for output, spec in zip(outputs, info.outputs)}
    # END STEP

    # STEP dequantize-and-compare
    with pcie.Model(str(model_path), connection=connection) as reference:
        reference.build(BUILD_TIMEOUT_MS)
        fp32_tensor = pcie.Tensor.from_numpy(fp32_input, copy=True, route_name=ingress.name)
        reference_outputs = reference.run([fp32_tensor], RUN_TIMEOUT_MS)

    print("Dequantized MLA-only outputs vs the default route (error in scale units):")
    for card, spec in zip(reference_outputs, info.outputs):
        if card.route.name != spec.name:
            raise RuntimeError(f"default route output '{spec.name}' is missing")
        scale, zero_point = require_quant(spec)
        host = dequantize(codes_by_head[spec.name], scale, zero_point)
        max_error = float(np.max(np.abs(host - card.to_numpy())) / scale)
        print(f"  {spec.name} {spec.shape} max_err={max_error:.4f}")
        if max_error > MAX_ERROR_SCALES:
            raise RuntimeError(f"output '{spec.name}' deviates from the default route")
    # END STEP
    # END CORE LOGIC

    print("[OK] 027_run_mla_only_int8")


if __name__ == "__main__":
    main()

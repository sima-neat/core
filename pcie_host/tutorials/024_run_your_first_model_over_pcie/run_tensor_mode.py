#!/usr/bin/env python3
"""Run YOLOv8s tensor-mode inference over PCIe.

Usage:
  python3 run_tensor_mode.py [--card 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import pyneatpcie as pcie


BUILD_TIMEOUT_MS = 180_000
RUN_TIMEOUT_MS = 30_000
MODEL_PATH = Path("yolo_v8s_mpk.tar.gz")
IMAGE_PATH = Path("share/sima-pcie-host/tutorials/assets/street-scene.png")


def parse_card() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0)
    return parser.parse_args().card


def make_yolo_tensor(bgr: np.ndarray, input_spec: pcie.TensorInfo) -> pcie.Tensor:
    if len(input_spec.shape) != 3 or input_spec.shape[2] != 3:
        raise RuntimeError("expected a three-channel HWC YOLO input")
    target_height, target_width, _ = input_spec.shape
    scale = min(target_width / bgr.shape[1], target_height / bgr.shape[0])
    resized_width = max(1, round(bgr.shape[1] * scale))
    resized_height = max(1, round(bgr.shape[0] * scale))
    resized = cv2.resize(bgr, (resized_width, resized_height))
    letterboxed = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    letterboxed[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    normalized = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
    return pcie.Tensor.from_numpy(
        normalized,
        copy=True,
        route_name=input_spec.name,
    )


def dtype_name(dtype: pcie.TensorDType) -> str:
    names = {
        pcie.TensorDType.UInt8: "UINT8",
        pcie.TensorDType.Int8: "INT8",
        pcie.TensorDType.UInt16: "UINT16",
        pcie.TensorDType.Int16: "INT16",
        pcie.TensorDType.Int32: "INT32",
        pcie.TensorDType.BFloat16: "BF16",
        pcie.TensorDType.Float32: "FP32",
        pcie.TensorDType.Float64: "FP64",
    }
    return names[dtype]


def main() -> None:
    card_id = parse_card()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"model does not exist: {MODEL_PATH}")
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode: {IMAGE_PATH}")

    # CORE LOGIC
    # STEP tensor-mode
    connection = pcie.ConnectionOptions(card_id=card_id)
    with pcie.Model(str(MODEL_PATH), connection=connection) as model:
        info = model.info()
        if len(info.inputs) != 1:
            raise RuntimeError("YOLOv8s must expose one input tensor")
        input_tensor = make_yolo_tensor(image, info.inputs[0])
        model.build(BUILD_TIMEOUT_MS)
        outputs = model.run([input_tensor], RUN_TIMEOUT_MS)
    if not outputs:
        raise RuntimeError("tensor mode returned no outputs")
    print("Tensor mode raw outputs:")
    for output in outputs:
        print(f"  {output.route.name} {dtype_name(output.dtype)} {output.shape}")
    # END STEP
    # END CORE LOGIC

    print("[OK] 024_run_tensor_mode")


if __name__ == "__main__":
    main()

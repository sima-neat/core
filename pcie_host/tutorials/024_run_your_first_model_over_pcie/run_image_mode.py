#!/usr/bin/env python3
"""Run YOLOv8s image-mode inference over PCIe.

Usage:
  python3 run_image_mode.py [--card 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import pyneatpcie as pcie


BUILD_TIMEOUT_MS = 180_000
RUN_TIMEOUT_MS = 30_000
MODEL_PATH = Path("yolo_v8s_mpk.tar.gz")
IMAGE_PATH = Path("share/sima-pcie-host/tutorials/assets/street-scene.png")


def parse_card() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0)
    return parser.parse_args().card


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
    # STEP image-mode
    connection = pcie.ConnectionOptions(card_id=card_id)
    options = pcie.ModelOptions()
    options.preprocess.kind = pcie.InputKind.Image
    options.preprocess.color_convert.input_format = pcie.ColorFormat.BGR
    options.preprocess.color_convert.output_format = pcie.ColorFormat.RGB
    options.preprocess.resize.enable = pcie.AutoFlag.On
    options.preprocess.resize.mode = pcie.ResizeMode.Letterbox
    options.preprocess.normalize.preset = pcie.NormalizePreset.COCO_YOLO
    with pcie.Model(str(MODEL_PATH), options, connection) as model:
        model.build(BUILD_TIMEOUT_MS)
        outputs = model.run_image(image, RUN_TIMEOUT_MS, pcie.PixelFormat.BGR)
    if not outputs:
        raise RuntimeError("image mode returned no outputs")
    print("Image mode raw outputs:")
    for output in outputs:
        print(f"  {output.route.name} {dtype_name(output.dtype)} {output.shape}")
    # END STEP
    # END CORE LOGIC

    print("[OK] 024_run_image_mode")


if __name__ == "__main__":
    main()

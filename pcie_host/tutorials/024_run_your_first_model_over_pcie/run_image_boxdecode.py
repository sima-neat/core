#!/usr/bin/env python3
"""Run YOLOv8s image inference with card-side box decode over PCIe.

Usage:
  python3 run_image_boxdecode.py [--card 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

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


# STEP parse-boxes
def parse_boxes(outputs: list[pcie.Tensor]) -> list[tuple[int, int, int, int, float, int]]:
    if len(outputs) != 1:
        raise RuntimeError("boxdecode must return one BBOX tensor")
    payload = outputs[0].to_bytes()
    if len(payload) < 4:
        raise RuntimeError("BBOX tensor is too small")
    count = struct.unpack_from("<I", payload, 0)[0]
    record_size = 24
    if count > (len(payload) - 4) // record_size:
        raise RuntimeError("BBOX detection count exceeds its payload")
    return [
        struct.unpack_from("<iiiifi", payload, 4 + index * record_size)
        for index in range(count)
    ]
# END STEP


def class_name(class_id: int) -> str:
    common_classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
    return common_classes.get(class_id, f"class_{class_id}")


def main() -> None:
    card_id = parse_card()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"model does not exist: {MODEL_PATH}")
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode: {IMAGE_PATH}")

    # CORE LOGIC
    # STEP decode-boxes
    connection = pcie.ConnectionOptions(card_id=card_id)
    options = pcie.ModelOptions()
    options.preprocess.kind = pcie.InputKind.Image
    options.preprocess.color_convert.input_format = pcie.ColorFormat.BGR
    options.preprocess.color_convert.output_format = pcie.ColorFormat.RGB
    options.preprocess.resize.enable = pcie.AutoFlag.On
    options.preprocess.resize.mode = pcie.ResizeMode.Letterbox
    options.preprocess.normalize.preset = pcie.NormalizePreset.COCO_YOLO
    options.decode_type = pcie.BoxDecodeType.YoloV8
    options.score_threshold = 0.25
    options.nms_iou_threshold = 0.45
    options.top_k = 100
    with pcie.Model(str(MODEL_PATH), options, connection) as model:
        model.build(BUILD_TIMEOUT_MS)
        outputs = model.run_image(image, RUN_TIMEOUT_MS, pcie.PixelFormat.BGR)
        boxes = parse_boxes(outputs)
    print(f"Image + boxdecode detections={len(boxes)}")
    for x, y, width, height, score, class_id in boxes[:10]:
        print(
            f"  {class_name(class_id)} score={score:.3f} "
            f"box=({x}, {y}, {width}, {height})"
        )
    if not boxes:
        raise RuntimeError("no detections passed the score threshold")
    # END STEP
    # END CORE LOGIC

    print("[OK] 024_run_image_boxdecode")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run ResNet-50 and YOLOv8s concurrently on two PCIe queues.

Usage:
  python3 run_models_on_multiple_pcie_queues.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct

import cv2
import numpy as np

import pyneatpcie as pcie


BUILD_TIMEOUT_MS = 180_000
RUN_TIMEOUT_MS = 30_000
RESNET_QUEUE = 0
YOLO_QUEUE = 1
RESNET_MODEL_PATH = Path("resnet_50_mpk.tar.gz")
YOLO_MODEL_PATH = Path("yolo_v8s_mpk.tar.gz")
RESNET_IMAGE_PATH = Path("share/sima-pcie-host/tutorials/assets/labrador.jpg")
YOLO_IMAGE_PATH = Path("share/sima-pcie-host/tutorials/assets/street-scene.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0)
    return parser.parse_args()


def connection_for(args: argparse.Namespace, queue: int) -> pcie.ConnectionOptions:
    return pcie.ConnectionOptions(
        card_id=args.card,
        queue=queue,
    )


def classification_options() -> pcie.ModelOptions:
    options = pcie.ModelOptions()
    options.preprocess.kind = pcie.InputKind.Image
    options.preprocess.color_convert.input_format = pcie.ColorFormat.BGR
    options.preprocess.color_convert.output_format = pcie.ColorFormat.RGB
    options.preprocess.resize.enable = pcie.AutoFlag.On
    options.preprocess.resize.mode = pcie.ResizeMode.Stretch
    options.preprocess.normalize.preset = pcie.NormalizePreset.ImageNet
    return options


def detection_options() -> pcie.ModelOptions:
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
    return options


def parse_boxes(outputs: list[pcie.Tensor]) -> list[tuple[int, int, int, int, float, int]]:
    if len(outputs) != 1:
        raise RuntimeError("YOLOv8 boxdecode must return one BBOX tensor")
    payload = outputs[0].to_bytes()
    if len(payload) < 4:
        raise RuntimeError("BBOX tensor is too small")
    count = struct.unpack_from("<I", payload, 0)[0]
    if count > (len(payload) - 4) // 24:
        raise RuntimeError("BBOX detection count exceeds its payload")
    return [
        struct.unpack_from("<iiiifi", payload, 4 + index * 24)
        for index in range(count)
    ]


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
    # STEP load-assets
    args = parse_args()
    for model_path in (RESNET_MODEL_PATH, YOLO_MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(f"model does not exist: {model_path}")
    labrador = cv2.imread(str(RESNET_IMAGE_PATH), cv2.IMREAD_COLOR)
    street = cv2.imread(str(YOLO_IMAGE_PATH), cv2.IMREAD_COLOR)
    if labrador is None or street is None:
        raise RuntimeError("OpenCV could not decode one of the input images")
    # END STEP

    # STEP assign-queues
    resnet = pcie.Model(
        str(RESNET_MODEL_PATH),
        classification_options(),
        connection_for(args, RESNET_QUEUE),
    )
    yolo = pcie.Model(
        str(YOLO_MODEL_PATH),
        detection_options(),
        connection_for(args, YOLO_QUEUE),
    )
    try:
        try:
            resnet.build(BUILD_TIMEOUT_MS)
        except Exception as error:
            raise RuntimeError(
                f"queue {RESNET_QUEUE} failed to build ResNet-50: {error}"
            ) from error
        try:
            yolo.build(BUILD_TIMEOUT_MS)
        except Exception as error:
            raise RuntimeError(
                f"queue {YOLO_QUEUE} failed to build YOLOv8s: {error}"
            ) from error
        # END STEP

        # CORE LOGIC
        # STEP run-concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            classification_future = executor.submit(
                resnet.run_image,
                labrador,
                RUN_TIMEOUT_MS,
                pcie.PixelFormat.BGR,
            )
            detection_future = executor.submit(
                yolo.run_image,
                street,
                RUN_TIMEOUT_MS,
                pcie.PixelFormat.BGR,
            )
            classification = classification_future.result()
            detections = detection_future.result()
        # END STEP

        # STEP read-results
        if len(classification) != 1:
            raise RuntimeError("ResNet-50 must return one output tensor")
        scores = classification[0].to_numpy().reshape(-1)
        top1 = int(np.argmax(scores))
        boxes = parse_boxes(detections)
        class_suffix = " (Labrador retriever)" if top1 == 208 else ""
        print(
            f"queue={RESNET_QUEUE} model=resnet_50 "
            f"output_shape={classification[0].shape} top1={top1}{class_suffix}"
        )
        print(
            f"queue={YOLO_QUEUE} model=yolo_v8s detections={len(boxes)}"
        )
        for x, y, width, height, score, class_id in boxes[:5]:
            print(
                f"  {class_name(class_id)} score={score:.3f} "
                f"box=({x}, {y}, {width}, {height})"
            )
        if not boxes:
            raise RuntimeError("YOLOv8s returned no street-scene detections")
        # END STEP
        # END CORE LOGIC
    finally:
        yolo.close()
        resnet.close()

    print("[OK] 026_run_models_on_multiple_pcie_queues")


if __name__ == "__main__":
    main()

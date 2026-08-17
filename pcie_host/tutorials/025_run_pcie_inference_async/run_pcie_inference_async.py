#!/usr/bin/env python3
"""Measure completed YOLOv8s detections with asynchronous PCIe push/pull.

Usage:
  python3 run_pcie_inference_async.py
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import struct
import threading
import time

import cv2
import numpy as np

import pyneatpcie as pcie


BUILD_TIMEOUT_MS = 180_000
PULL_TIMEOUT_MS = 30_000
WARMUP_FRAMES = 5
MEASURED_FRAMES = 1000
MODEL_PATH = Path("yolo_v8s_mpk.tar.gz")
IMAGE_PATH = Path("share/sima-pcie-host/tutorials/assets/street-scene.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0)
    return parser.parse_args()


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


def detection_count(outputs: list[pcie.Tensor]) -> int:
    if len(outputs) != 1:
        raise RuntimeError("boxdecode must return one BBOX tensor")
    payload = outputs[0].to_bytes()
    if len(payload) < 4:
        raise RuntimeError("BBOX tensor is too small")
    count = struct.unpack_from("<I", payload, 0)[0]
    if count > (len(payload) - 4) // 24:
        raise RuntimeError("BBOX detection count exceeds its payload")
    return count


def measure(
    model: pcie.Model, image: np.ndarray, frame_count: int
) -> tuple[int, float, float, int]:
    image_tensor = pcie.Tensor.from_numpy(
        np.ascontiguousarray(image),
        image_format=pcie.PixelFormat.BGR,
        route_name="input_image",
    )
    submitted: deque[float] = deque()
    submitted_lock = threading.Lock()
    failure_lock = threading.Lock()
    failures: list[BaseException] = []
    cancelled = threading.Event()
    latencies_ms: list[float] = []
    total_detections = 0

    def fail(error: BaseException) -> None:
        with failure_lock:
            if not failures:
                failures.append(error)
        cancelled.set()
        model.close()

    def produce() -> None:
        try:
            for index in range(frame_count):
                if cancelled.is_set():
                    return
                started = time.perf_counter()
                with submitted_lock:
                    submitted.append(started)
                if not model.push(image_tensor):
                    raise RuntimeError(f"push rejected frame {index}")
        except BaseException as error:
            fail(error)

    def consume() -> None:
        nonlocal total_detections
        try:
            for index in range(frame_count):
                if cancelled.is_set():
                    return
                outputs = model.pull(PULL_TIMEOUT_MS)
                if outputs is None:
                    raise RuntimeError(f"pull timed out for frame {index}")
                with submitted_lock:
                    if not submitted:
                        raise RuntimeError(
                            "completion arrived without a submission record"
                        )
                    started = submitted.popleft()
                total_detections += detection_count(outputs)
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
        except BaseException as error:
            fail(error)

    benchmark_start = time.perf_counter()
    producer = threading.Thread(target=produce, name="pcie-producer")
    consumer = threading.Thread(target=consume, name="pcie-consumer")
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    elapsed_seconds = time.perf_counter() - benchmark_start

    if failures:
        raise failures[0]
    if len(latencies_ms) != frame_count:
        raise RuntimeError("not every submitted frame completed")
    average_latency_ms = sum(latencies_ms) / len(latencies_ms)
    return len(latencies_ms), elapsed_seconds, average_latency_ms, total_detections


def main() -> None:
    # STEP configure-model
    args = parse_args()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"model does not exist: {MODEL_PATH}")
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode: {IMAGE_PATH}")
    connection = pcie.ConnectionOptions(
        card_id=args.card,
    )
    model = pcie.Model(str(MODEL_PATH), detection_options(), connection)
    model.build(BUILD_TIMEOUT_MS)
    # END STEP

    try:
        # STEP warm-up
        for _ in range(WARMUP_FRAMES):
            detection_count(
                model.run_image(image, PULL_TIMEOUT_MS, pcie.PixelFormat.BGR)
            )
        # END STEP

        # CORE LOGIC
        # STEP push-pull
        completed, elapsed_seconds, average_latency_ms, total_detections = measure(
            model, image, MEASURED_FRAMES
        )
        # END STEP

        # STEP report-results
        print(f"completed={completed}")
        print(f"elapsed_seconds={elapsed_seconds:.2f}")
        print(f"throughput_fps={completed / elapsed_seconds:.2f}")
        print(f"average_latency_ms={average_latency_ms:.2f}")
        print(f"total_detections={total_detections}")
        # END STEP
        # END CORE LOGIC
    finally:
        model.close()

    print("[OK] 025_run_pcie_inference_async")


if __name__ == "__main__":
    main()

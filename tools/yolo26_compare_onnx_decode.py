#!/usr/bin/env python3
"""Compare two raw-head YOLO26 ONNX models with host BoxDecode-style decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def letterbox_bgr(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_size = (round(width * ratio), round(height * ratio))
    pad_w = (size - new_size[0]) / 2
    pad_h = (size - new_size[1]) / 2
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    nchw_rgb = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return nchw_rgb, ratio, left, top


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def nms_class_aware(boxes: list[list[float]], iou_threshold: float, top_k: int) -> list[list[float]]:
    sorted_boxes = sorted(boxes, key=lambda box: box[4], reverse=True)[: max(top_k * 10, top_k)]
    keep: list[list[float]] = []
    for box in sorted_boxes:
        if all(int(box[5]) != int(prev[5]) or iou_xyxy(box, prev) < iou_threshold for prev in keep):
            keep.append(box)
        if len(keep) >= top_k:
            break
    return keep


def decode_yolo26_raw_heads(
    outputs: list[np.ndarray],
    image_shape: tuple[int, int],
    ratio: float,
    pad_left: int,
    pad_top: int,
    image_size: int,
    score_threshold: float,
    nms_iou_threshold: float,
    top_k: int,
) -> list[list[float]]:
    original_height, original_width = image_shape
    boxes: list[list[float]] = []
    for scale_index in range(3):
        bbox = outputs[scale_index][0]
        logits = outputs[scale_index + 3][0]
        _, head_h, head_w = bbox.shape
        stride_x = image_size / head_w
        stride_y = image_size / head_h
        scores = sigmoid(logits)
        class_indices, y_indices, x_indices = np.where(scores > score_threshold)
        for cls, y_cell, x_cell in zip(class_indices, y_indices, x_indices):
            score = float(scores[cls, y_cell, x_cell])
            left, top, right, bottom = [float(value) for value in bbox[:, y_cell, x_cell]]
            center_x = (x_cell + 0.5 + (right - left) / 2.0) * stride_x
            center_y = (y_cell + 0.5 + (bottom - top) / 2.0) * stride_y
            width = (left + right) * stride_x
            height = (top + bottom) * stride_y
            x1 = (center_x - width / 2.0 - pad_left) / ratio
            y1 = (center_y - height / 2.0 - pad_top) / ratio
            x2 = (center_x + width / 2.0 - pad_left) / ratio
            y2 = (center_y + height / 2.0 - pad_top) / ratio
            boxes.append(
                [
                    float(np.clip(x1, 0, original_width)),
                    float(np.clip(y1, 0, original_height)),
                    float(np.clip(x2, 0, original_width)),
                    float(np.clip(y2, 0, original_height)),
                    score,
                    float(cls),
                ]
            )
    return nms_class_aware(boxes, nms_iou_threshold, top_k)


def run_model(session: ort.InferenceSession, image_tensor: np.ndarray) -> list[np.ndarray]:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: image_tensor})


def summarize_image(
    image_path: Path,
    reference_session: ort.InferenceSession,
    candidate_session: ort.InferenceSession,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_tensor, ratio, pad_left, pad_top = letterbox_bgr(image, args.image_size)
    image_shape = image.shape[:2]
    reference_boxes = decode_yolo26_raw_heads(
        run_model(reference_session, image_tensor),
        image_shape,
        ratio,
        pad_left,
        pad_top,
        args.image_size,
        args.reference_conf,
        args.nms_iou,
        args.top_k,
    )
    candidate_boxes = decode_yolo26_raw_heads(
        run_model(candidate_session, image_tensor),
        image_shape,
        ratio,
        pad_left,
        pad_top,
        args.image_size,
        args.candidate_conf,
        args.nms_iou,
        args.top_k,
    )

    matches = []
    for reference in reference_boxes[: args.match_top_k]:
        same_class = [candidate for candidate in candidate_boxes if int(candidate[5]) == int(reference[5])]
        best_iou = 0.0
        best_candidate: list[float] | None = None
        for candidate in same_class:
            candidate_iou = iou_xyxy(reference, candidate)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_candidate = candidate
        matches.append(
            {
                "class_id": int(reference[5]),
                "reference_score": reference[4],
                "candidate_score": None if best_candidate is None else best_candidate[4],
                "best_same_class_iou": best_iou,
                "pass": best_iou >= args.min_iou,
            }
        )
    return {
        "image": str(image_path),
        "reference_count": len(reference_boxes),
        "candidate_count": len(candidate_boxes),
        "reference_top": reference_boxes[: args.match_top_k],
        "candidate_top": candidate_boxes[: args.match_top_k],
        "matches": matches,
        "pass": bool(candidate_boxes) and all(match["pass"] for match in matches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw-head YOLO26 ONNX models.")
    parser.add_argument("--reference", required=True, help="Reference raw-head YOLO26 ONNX.")
    parser.add_argument("--candidate", required=True, help="Candidate raw-head YOLO26 ONNX.")
    parser.add_argument("--images", nargs="+", required=True, help="Images to compare.")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--reference-conf", type=float, default=0.20)
    parser.add_argument("--candidate-conf", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--match-top-k", type=int, default=10)
    parser.add_argument("--min-iou", type=float, default=0.80)
    parser.add_argument("--json-output", default=None, help="Optional JSON report path.")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    reference_session = ort.InferenceSession(args.reference, providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(args.candidate, providers=["CPUExecutionProvider"])
    report = {
        "reference": args.reference,
        "candidate": args.candidate,
        "settings": {
            "reference_conf": args.reference_conf,
            "candidate_conf": args.candidate_conf,
            "nms_iou": args.nms_iou,
            "min_iou": args.min_iou,
            "match_top_k": args.match_top_k,
        },
        "images": [
            summarize_image(Path(image_path), reference_session, candidate_session, args)
            for image_path in args.images
        ],
    }
    report["pass"] = all(image["pass"] for image in report["images"])
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    if args.fail_on_mismatch and not report["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe RTSP cadence with OpenCV."""

from __future__ import annotations

import cv2


def probe_source_fps(url: str) -> int:
    capture = cv2.VideoCapture(url)
    try:
        if not capture.isOpened():
            raise RuntimeError("failed to open RTSP source for FPS probe")
        fps = int(round(capture.get(cv2.CAP_PROP_FPS)))
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError("failed to probe a positive RTSP source FPS")
    return fps

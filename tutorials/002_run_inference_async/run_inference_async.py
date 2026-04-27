#!/usr/bin/env python3
"""Feed frames to a model asynchronously from a producer thread and pull results.

Usage:
  python3 async_push_pull.py --mpk /path/to/resnet_50.tar.gz [--n 4] [--image /path/to.jpg]
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

import numpy as np
import cv2


def load_image(path: Path | None, size: int) -> np.ndarray:
  if path is None:
    return np.full((size, size, 3), 99, dtype=np.uint8)
  bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if bgr is None:
    raise RuntimeError(f"failed to read image: {path}")
  if bgr.shape[0] != size or bgr.shape[1] != size:
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  ap.add_argument("--image", type=Path)
  ap.add_argument("--n", type=int, default=4, help="number of frames to push")
  args = ap.parse_args(argv[1:])

  frame = load_image(args.image, size=224)
  model = pyneat.Model(str(args.mpk))

  # CORE LOGIC
  session = pyneat.Session()
  session.add(model.session())
  run = session.build(frame, pyneat.RunMode.Async)

  # Producer thread pushes frames; main thread pulls predictions.
  def producer() -> None:
    for _ in range(args.n):
      run.push(frame)
    run.close_input()

  thread = threading.Thread(target=producer, name="frame_producer")
  thread.start()

  pulled = 0
  while True:
    sample = run.pull(timeout_ms=2000)
    if sample is None:
      if not thread.is_alive():
        break
      continue
    top1 = int(np.argmax(sample.tensor.to_numpy().reshape(-1)))
    print(f"top1={top1}")
    pulled += 1
  # END CORE LOGIC

  thread.join()
  print(f"pulled={pulled}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))

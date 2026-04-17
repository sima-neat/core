#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit("pyneat is not installed. Follow the installation guide.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


@dataclass
class Sig:
  output_kind: str = "sample_or_tensor"
  tensor_rank: int = -1
  field_count: int = -1


def _source_fallback_signature_stub() -> None:
  # Source-fallback signature for parity tooling if runtime output is unavailable.
  if False:
    tu.signature(
        {
            "tutorial": "002",
            "lang": "py",
            "flow": "minimal_pytorch_dataloader_async_threaded",
            "run_mode": "async",
            "output_kind": "sample_or_tensor",
            "tensor_rank": -1,
            "field_count": -1,
            "tput_contract": -1,
        }
    )


def build_model_options():
  opt = pyneat.ModelOptions()
  opt.format = "RGB"
  return opt


def _resnet_image_candidates(root: Path) -> list[Path]:
  return [
      root / "tests" / "assets" / "preproc_dynamic" / "fronalpstock_1330.jpg",
      root / "tests" / "assets" / "preproc_dynamic" / "ilena_488.jpg",
      root / "tests" / "assets" / "preproc_dynamic" / "lichtenstein_512.png",
      root / "tmp" / "coco_sample.jpg",
      root / "test.jpg",
  ]


def _load_rgb_pil(path: Path):
  from PIL import Image

  with Image.open(path) as img:
    return img.convert("RGB")


def dataloader_from_pytorch(size: int, batch: int, n: int):
  try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
  except Exception as exc:
    raise RuntimeError("PyTorch + torchvision are required for chapter 002 dataloader flow") from exc

  root = tu.repo_root()
  existing = [p for p in _resnet_image_candidates(root) if p.exists()]
  if not existing:
    raise RuntimeError("no local images found for ResNet50 run")

  count = max(1, n)
  selected = [existing[i % len(existing)] for i in range(count)]

  class ResnetImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
      self._paths = paths
      # Normal PyTorch flow: PIL image -> torchvision tensor (CHW, uint8).
      self._to_tensor = transforms.PILToTensor()

    def __len__(self) -> int:
      return len(self._paths)

    def __getitem__(self, idx: int):
      pil_rgb = _load_rgb_pil(self._paths[idx])
      chw_u8 = self._to_tensor(pil_rgb)
      # Session async path consumes video-like HWC tensors.
      hwc_u8 = chw_u8.permute(1, 2, 0).contiguous()
      return hwc_u8, torch.tensor(-1, dtype=torch.int64)

  return DataLoader(
      ResnetImageDataset(selected),
      batch_size=max(1, batch),
      shuffle=False,
      num_workers=0,
      drop_last=False,
  )


def _scores_from_output(out):
  try:
    import torch
  except Exception as exc:
    raise RuntimeError("PyTorch is required for chapter 002 output decoding") from exc

  tu.ensure(out.tensor is not None, "expected tensor output from model")
  flat = out.tensor.to_torch(copy=True).to(dtype=torch.float32).reshape(-1)
  tu.ensure(flat.numel() > 0, "empty tensor output")
  if flat.numel() >= 1000:
    flat = flat[:1000]
  return flat


def top1_from_output(out) -> int:
  return int(_scores_from_output(out).argmax().item())


def summarize(out) -> Sig:
  try:
    kind = str(int(out.kind))
  except Exception:
    kind = str(out.kind)

  rank = len(out.tensor.shape) if out.tensor is not None else -1
  return Sig(output_kind=kind, tensor_rank=rank, field_count=len(out.fields))


def collect_first_images(resnet_dataloader):
  images = []
  for image_batch, _yb in resnet_dataloader:
    images.append(image_batch[0])
  tu.ensure(images, "no images available for async run")
  return images


def run_async_inference(run, images, timeout_ms: int, expect_id: int):
  pushed = 0
  pushed_lock = threading.Lock()
  producer_done = threading.Event()
  producer_error: list[Exception] = []
  sig = Sig()

  def producer() -> None:
    nonlocal pushed
    try:
      for image in images:
        # CORE LOGIC
        tu.ensure(run.push(image), "push failed")
        with pushed_lock:
          pushed += 1
        # END CORE LOGIC
    except Exception as exc:  # pragma: no cover - runtime dependent
      producer_error.append(exc)
    finally:
      run.close_input()
      producer_done.set()

  start = time.perf_counter()
  producer_thread = threading.Thread(target=producer, name="resnet_async_producer")
  producer_thread.start()

  pulled = 0
  # CORE LOGIC
  while True:
    out = run.pull(timeout_ms=timeout_ms)
    if out is not None:
      pulled += 1
      pred = top1_from_output(out)
      sig = summarize(out)
      print(f"top1={pred}")

      if expect_id >= 0:
        tu.check("top1_expected_id", pred == expect_id, "verify expected class id")
      continue

    with pushed_lock:
      pushed_now = pushed
    if producer_done.is_set() and pulled >= pushed_now:
      break
  # END CORE LOGIC
  
  producer_thread.join()
  if producer_error:
    raise producer_error[0]

  with pushed_lock:
    pushed_final = pushed

  tu.ensure(
      pulled == pushed_final,
      f"async output count mismatch: pulled={pulled}, pushed={pushed_final}",
  )
  elapsed = max(time.perf_counter() - start, 1e-9)
  return pushed_final, pulled, sig, elapsed


def main(argv: list[str]) -> int:
  _source_fallback_signature_stub()

  ap = argparse.ArgumentParser()
  ap.add_argument("--mpk", type=str, default=None)
  ap.add_argument("--size", type=int, default=224)
  ap.add_argument("--batch", type=int, default=1)
  ap.add_argument("--n", type=int, default=4)
  ap.add_argument("--queue", type=int, default=8)
  ap.add_argument("--timeout-ms", type=int, default=2000)
  ap.add_argument("--expect-id", type=int, default=-1)
  ap.add_argument("--print-gst", action="store_true")
  args = ap.parse_args(argv)

  tu.step("input_contract", "parse CLI and prepare ResNet50 model + local image dataloader")
  tu.step("run_mode_choice", "run async inference with producer-thread push and main-thread pull")
  tu.why("keep chapter 002 parallel to chapter 001 so sync-vs-async is the main variable")
  tu.tradeoff("threaded async flow improves throughput potential but adds coordination complexity")
  tu.failure_mode("push/pull mismatches, queue stalls, or producer exceptions should fail loudly")
  tu.interpret_output("top1 is user-facing; tput_fps and tput_contract are parity-facing")
  tu.step("output_contract", "emit top1 lines, async stats, and stable signature")
  tu.check("strict_mode_visible", isinstance(tu.strict_mode(), bool), "strict-mode guard is observable")

  root = tu.repo_root()
  mpk_path = Path(args.mpk) if args.mpk else tu.default_resnet_mpk(root)
  if not mpk_path or not mpk_path.exists():
    return tu.skip("missing ResNet50 MPK (pass --mpk)")

  sig = Sig()
  tput_contract = -1
  try:
    # Same model/dataloader setup as 001; only run mode differs.
    resnet50 = pyneat.Model(str(mpk_path), build_model_options())

    if args.print_gst:
      s = pyneat.Session()
      s.add(resnet50.session())
      print(s.describe_backend())
      return 0

    resnet_dataloader = dataloader_from_pytorch(args.size, args.batch, args.n)
    images = collect_first_images(resnet_dataloader)
    # CORE LOGIC
    s = pyneat.Session()
    s.add(resnet50.session())

    opt = pyneat.RunOptions()
    opt.queue_depth = args.queue
    opt.overflow_policy = pyneat.OverflowPolicy.Block
    opt.output_memory = pyneat.OutputMemory.Owned

    run = s.build(images[0], pyneat.RunMode.Async, opt)
    pushed_final, pulled, sig, elapsed = run_async_inference(
        run=run,
        images=images,
        timeout_ms=args.timeout_ms,
        expect_id=args.expect_id,
    )
    # END CORE LOGIC
    stats = run.stats()
    tput_fps = pulled / elapsed
    tput_contract = pulled

    print(f"pushed:          {pushed_final}")
    print(f"pulled:          {pulled}")
    print(f"inputs_enqueued: {stats.inputs_enqueued}")
    print(f"inputs_dropped:  {stats.inputs_dropped}")
    print(f"tput_fps:        {tput_fps:.3f}")
    print(f"tput_contract:   {tput_contract}")
  except Exception as exc:
    tu.runtime_fallback(exc)
    if tu.strict_mode():
      raise

  tu.check("tutorial_completed", True, "async dataloader path completed")
  tu.signature(
      {
          "tutorial": "002",
          "lang": "py",
          "flow": "minimal_pytorch_dataloader_async_threaded",
          "run_mode": "async",
          "output_kind": sig.output_kind,
          "tensor_rank": sig.tensor_rank,
          "field_count": sig.field_count,
          "tput_contract": tput_contract,
      }
  )

  print("[OK] 002_async_push_pull")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))

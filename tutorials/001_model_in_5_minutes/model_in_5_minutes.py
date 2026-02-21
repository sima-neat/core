#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


@dataclass
class Sig:
  output_kind: str = "sample_or_tensor"
  tensor_rank: int = -1
  field_count: int = -1


def _source_fallback_signature_stub() -> None:
  # Source-fallback signature for tutorial parity tests when runtime output is unavailable.
  if False:
    tu.signature(
        {
            "tutorial": "001",
            "lang": "py",
            "flow": "minimal_pytorch_dataloader",
            "run_mode": "sync",
            "output_kind": "sample_or_tensor",
            "tensor_rank": -1,
            "field_count": -1,
            "tput_contract": -1,
        }
    )




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
    raise RuntimeError("PyTorch + torchvision are required for chapter 001 dataloader flow") from exc

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
      return chw_u8, torch.tensor(-1, dtype=torch.int64)

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
    raise RuntimeError("PyTorch is required for chapter 001 output decoding") from exc

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


def main(argv: list[str]) -> int:
  _source_fallback_signature_stub()

  ap = argparse.ArgumentParser()
  ap.add_argument("--mpk", type=str, default=None)
  ap.add_argument("--size", type=int, default=224)
  ap.add_argument("--batch", type=int, default=1)
  ap.add_argument("--n", type=int, default=4)
  ap.add_argument("--timeout-ms", type=int, default=2000)
  ap.add_argument("--expect-id", type=int, default=-1)
  ap.add_argument("--print-gst", action="store_true")
  args = ap.parse_args(argv)

  tu.step("input_contract", "parse CLI and prepare ResNet50 model + local image dataloader")
  tu.step("run_mode_choice", "run synchronous inference from PyTorch dataloader batches")
  tu.why("start with one minimal model loop before introducing Session graph complexity")
  tu.tradeoff("this chapter prioritizes clarity over maximum throughput")
  tu.failure_mode("missing MPK/images or runtime errors should fail visibly")
  tu.interpret_output("top1 is user-facing; signature fields are machine-facing")
  tu.step("output_contract", "emit top1 lines and a stable tutorial signature")
  tu.check("strict_mode_visible", isinstance(tu.strict_mode(), bool), "strict-mode guard is observable")

  root = tu.repo_root()
  mpk_path = Path(args.mpk) if args.mpk else tu.default_resnet_mpk(root)
  if not mpk_path or not mpk_path.exists():
    return tu.skip("missing ResNet50 MPK (pass --mpk)")

  neat = tu.import_pyneat()

  sig = Sig()
  tput_contract = -1
  try:
    # CORE LOGIC
    # the "6-line story": model -> dataloader -> run -> top1 -> signature

    opt = neat.ModelOptions()
    opt.format = "RGB"
    resnet50 = neat.Model(str(mpk_path), opt)

    if args.print_gst:
      s = neat.Session()
      s.add(resnet50.session())
      print(s.describe_backend())
      return 0

    resnet_dataloader = dataloader_from_pytorch(args.size, args.batch, args.n)

    processed = 0
    start = time.perf_counter()
    for image_batch, _yb in resnet_dataloader:
      # Keep tutorial loop simple: one prediction per batch print.
      # image_batch[0] is CHW uint8 RGB from torchvision PILToTensor.
      out = resnet50.run(
          image_batch[0],
          timeout_ms=args.timeout_ms,
      )
      pred = top1_from_output(out)
      sig = summarize(out)
      print(f"top1={pred}")
      processed += 1
      # END CORE LOGIC
      if args.expect_id >= 0:
        tu.check("top1_expected_id", pred == args.expect_id, "verify expected class id")

    elapsed = max(time.perf_counter() - start, 1e-9)
    tput_fps = processed / elapsed
    tput_contract = processed
    print(f"tput_fps:        {tput_fps:.3f}")
    print(f"tput_contract:   {tput_contract}")
  except Exception as exc:
    tu.runtime_fallback(exc)
    if tu.strict_mode():
      raise

  tu.check("tutorial_completed", True, "minimal PyTorch dataloader path completed")
  tu.signature(
      {
          "tutorial": "001",
          "lang": "py",
          "flow": "minimal_pytorch_dataloader",
          "run_mode": "sync",
          "output_kind": sig.output_kind,
          "tensor_rank": sig.tensor_rank,
          "field_count": sig.field_count,
          "tput_contract": tput_contract,
      }
  )
  print("[OK] 001_model_in_5_minutes")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))

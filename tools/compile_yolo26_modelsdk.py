#!/usr/bin/env python3
"""Compile a YOLO26 raw-head ONNX with Model SDK BF16 + MLA tessellation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
import onnx
from onnxsim import simplify

from afe.apis.defines import (
    InputName,
    TensorDRAMLayout,
    TensorTessellateParameters,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    CalibrationMethod,
    RequantizationMode,
)
from afe.apis.loaded_net import load_model
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.node import node_is_tuple
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import ImporterParams, ModelFormat
from sima_utils.data.data_generator import DataGenerator


ONNX_IR_VERSION = 8


def detect_io(model_path: Path) -> tuple[list[str], list[tuple[int, ...]], list[str]]:
    model = onnx.load(model_path)
    input_names = [node.name for node in model.graph.input]
    input_shapes: list[tuple[int, ...]] = []
    for node in model.graph.input:
        dims = tuple(dim.dim_value for dim in node.type.tensor_type.shape.dim)
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"Dynamic input shape for {node.name}: {dims}")
        input_shapes.append(dims)
    output_names = [node.name for node in model.graph.output]
    return input_names, input_shapes, output_names


def prepare_onnx(model_path: Path, input_names: list[str], input_shapes: list[tuple[int, ...]], out_dir: Path) -> Path:
    shapes = {name: list(shape) for name, shape in zip(input_names, input_shapes)}
    output_path = out_dir / f"{model_path.stem}_prepared.onnx"
    model, ok = simplify(str(model_path), overwrite_input_shapes=shapes, dynamic_input_shape=False)
    if not ok:
        raise RuntimeError("onnxsim validation failed")
    del model.graph.value_info[:]
    for input_tensor in model.graph.input:
        if input_tensor.name in shapes:
            input_tensor.type.tensor_type.shape.ClearField("dim")
            for dim_size in shapes[input_tensor.name]:
                dim = input_tensor.type.tensor_type.shape.dim.add()
                dim.dim_value = int(dim_size)
    model = onnx.shape_inference.infer_shapes(model)
    model.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def dummy_calibration(input_names: list[str], input_shapes: list[tuple[int, ...]]) -> Any:
    data = {}
    rng = np.random.default_rng(0)
    for name, shape in zip(input_names, input_shapes):
        data[InputName(name)] = rng.random(shape, dtype=np.float32).transpose(0, 2, 3, 1)
    return convert_data_generator_to_iterable(DataGenerator(data))


def inspect_mpk(mpk_path: Path) -> dict[str, Any]:
    with tarfile.open(mpk_path) as tar:
        names = tar.getnames()
        mpk_json_name = next(name for name in names if name.endswith("_mpk.json"))
        mpk_json = json.load(tar.extractfile(mpk_json_name))  # type: ignore[arg-type]
        processors: dict[str, int] = {}
        for plugin in mpk_json.get("plugins", []):
            processor = plugin.get("processor") or plugin.get("type") or plugin.get("backend")
            processors[processor] = processors.get(processor, 0) + 1
        return {
            "mpk": str(mpk_path),
            "files": sorted(names),
            "mla_elf_count": sum(name.endswith("_mla.elf") for name in names),
            "so_count": sum(name.endswith(".so") for name in names),
            "process_tvm_count": sum("process_tvm" in name for name in names),
            "processor_counts": processors,
        }


def compile_model(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model)
    output_root = Path(args.build_dir) / model_path.stem
    output_root.mkdir(parents=True, exist_ok=True)
    input_names, input_shapes, output_names = detect_io(model_path)
    compile_path = (
        prepare_onnx(model_path, input_names, input_shapes, output_root)
        if args.simplify
        else model_path
    )
    _, _, output_names = detect_io(compile_path)

    loaded_net = load_model(
        ImporterParams(
            format=ModelFormat.onnx,
            file_paths=[str(compile_path)],
            input_names=input_names,
            input_shapes=input_shapes,
            input_types=[ScalarType.float32] * len(input_names),
            layout="NCHW",
            output_names=output_names,
        ),
        target=gen2_target,
    )

    bf16 = bfloat16_scheme()
    quant_config = (
        default_quantization.with_activation_quantization(bf16)
        .with_weight_quantization(bf16)
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(CalibrationMethod.from_str("mse"))
    )
    quant_model = loaded_net.quantize(
        calibration_data=dummy_calibration(input_names, input_shapes),
        quantization_config=quant_config,
        any_shape_on_mla=True,
        automatic_layout_conversion=False,
        model_name=model_path.stem,
        log_level=logging.INFO,
    )
    quant_model.save(model_name=model_path.stem, output_directory=str(output_root))

    tess_params = None
    if args.mla_tessellation:
        if "MLA_0" not in quant_model._net.nodes:
            raise RuntimeError(f"MLA_0 not found. Available nodes: {list(quant_model._net.nodes)[:30]}")
        mla_node = quant_model._net.nodes["MLA_0"]
        tess_params = {}
        input_tess = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC,
        )
        for input_name in mla_node.input_names:
            tess_params[input_name] = dataclasses.replace(input_tess)

        output_node = mla_node.ir.nodes[mla_node.ir.output_node_name]
        out_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
        output_tess = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
        )
        for output_name in out_names:
            tess_params[f"{output_name}_output"] = dataclasses.replace(output_tess)

    quant_model.compile(
        output_path=str(output_root),
        batch_size=1,
        log_level=logging.INFO,
        tessellate_parameters=tess_params,
    )
    mpks = sorted(output_root.glob("*_mpk.tar.gz"))
    if not mpks:
        raise RuntimeError(f"No *_mpk.tar.gz found under {output_root}")
    report = {
        "model_sdk_version": get_model_sdk_version(),
        "input_model": str(model_path),
        "compiled_model": str(compile_path),
        "output_root": str(output_root),
        "inputs": list(zip(input_names, input_shapes)),
        "outputs": output_names,
        "mpk": inspect_mpk(mpks[-1]),
    }
    if args.strict_one_mla:
        mpk = report["mpk"]
        if mpk["mla_elf_count"] != 1 or mpk["so_count"] != 0 or mpk["process_tvm_count"] != 0:
            raise RuntimeError(f"MPK failed one-MLA/zero-.so gate: {mpk}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile YOLO26 ONNX with BF16 MLA tessellation.")
    parser.add_argument("--model", required=True, help="Input ONNX model.")
    parser.add_argument("--build-dir", required=True, help="Output build directory.")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify")
    parser.add_argument("--no-mla-tessellation", action="store_false", dest="mla_tessellation")
    parser.add_argument("--strict-one-mla", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.set_defaults(simplify=True, mla_tessellation=True)
    args = parser.parse_args()

    report = compile_model(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""YOLO26 attention exactness/compile feasibility spike.

This tool builds small standalone ONNX models for the two YOLO26 attention
islands and tests whether mathematically equivalent rewrites can be accepted by
Model SDK as MLA-only graphs.  It is intentionally separate from
``yolo26_boxdecode_surgery.py`` so exactness experiments do not silently change
the production surgery contract.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import json
from pathlib import Path
import tarfile
import traceback
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


OPSET_VERSION = 17
IR_VERSION = 8

ATTENTION_BLOCKS = OrderedDict(
    [
        (
            "model10",
            {
                "base": "/model.10/m/m.0/attn",
                "input": "/model.10/Split_output_1",
                "output": "/model.10/m/m.0/attn/proj/conv/Conv_output_0",
            },
        ),
        (
            "model22",
            {
                "base": "/model.22/m.0/m.0.1/attn",
                "input": "/model.22/m.0/m.0.0/Add_output_0",
                "output": "/model.22/m.0/m.0.1/attn/proj/conv/Conv_output_0",
            },
        ),
    ]
)

ORIGINAL_NODE_SUFFIXES = [
    "qkv/conv/Conv",
    "Reshape",
    "Split",
    "Transpose",
    "Reshape_2",
    "MatMul",
    "pe/conv/Conv",
    "Mul",
    "Softmax",
    "Transpose_1",
    "MatMul_1",
    "Reshape_1",
    "Add",
    "proj/conv/Conv",
]


def node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def find_node(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if node.name == name:
            return node
    raise ValueError(f"Node not found: {name}")


def find_initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer
    raise ValueError(f"Initializer not found: {name}")


def tensor_shape(model: onnx.ModelProto, name: str) -> list[int]:
    for value in [*model.graph.value_info, *model.graph.input, *model.graph.output]:
        if value.name != name:
            continue
        dims: list[int] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            else:
                raise ValueError(f"Dynamic dimension found for {name}")
        return dims
    raise ValueError(f"Shape not found for {name}")


def add_initializer(initializers: list[onnx.TensorProto], name: str, array: np.ndarray) -> None:
    initializers.append(numpy_helper.from_array(array, name))


def referenced_initializers(model: onnx.ModelProto, nodes: list[onnx.NodeProto]) -> list[onnx.TensorProto]:
    graph_initializers = {initializer.name for initializer in model.graph.initializer}
    needed: set[str] = set()
    for node in nodes:
        for name in node.input:
            if name in graph_initializers:
                needed.add(name)
    return [copy.deepcopy(find_initializer(model, name)) for name in sorted(needed)]


def make_model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    output_name: str,
    output_shape: list[int],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "yolo26_attention_feasibility",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128, 20, 20])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET_VERSION)],
        producer_name="sima-yolo26-attention-feasibility",
    )
    model.ir_version = IR_VERSION
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def original_attention_model(model: onnx.ModelProto, block: dict[str, str]) -> onnx.ModelProto:
    base = block["base"]
    nodes = [copy.deepcopy(find_node(model, f"{base}/{suffix}")) for suffix in ORIGINAL_NODE_SUFFIXES]
    nodes[0].input[0] = "input"
    initializers = referenced_initializers(model, nodes)
    output_shape = tensor_shape(model, block["output"])
    return make_model(nodes, initializers, block["output"], output_shape)


def rewritten_attention_model(
    model: onnx.ModelProto,
    block_name: str,
    block: dict[str, str],
    variant: str,
) -> onnx.ModelProto:
    base = block["base"]
    qkv = find_node(model, f"{base}/qkv/conv/Conv")
    pe = find_node(model, f"{base}/pe/conv/Conv")
    proj = find_node(model, f"{base}/proj/conv/Conv")
    mul = find_node(model, f"{base}/Mul")
    scale = find_initializer(model, mul.input[1])

    initializers = [
        copy.deepcopy(find_initializer(model, qkv.input[1])),
        copy.deepcopy(find_initializer(model, qkv.input[2])),
        copy.deepcopy(find_initializer(model, pe.input[1])),
        copy.deepcopy(find_initializer(model, pe.input[2])),
        copy.deepcopy(find_initializer(model, proj.input[1])),
        copy.deepcopy(find_initializer(model, proj.input[2])),
        copy.deepcopy(scale),
    ]
    add_initializer(initializers, f"{block_name}_{variant}_qkv_shape", np.array([1, 2, 128, 400], dtype=np.int64))
    add_initializer(initializers, f"{block_name}_{variant}_v4_shape", np.array([1, 128, 20, 20], dtype=np.int64))
    add_initializer(initializers, f"{block_name}_{variant}_split", np.array([32, 32, 64], dtype=np.int64))

    qkv_out = f"{variant}/qkv"
    qkv_reshape = f"{variant}/qkv_reshape"
    q = f"{variant}/q"
    k = f"{variant}/k"
    v = f"{variant}/v"
    qk = f"{variant}/qk"
    scaled = f"{variant}/scaled"
    prob = f"{variant}/prob"
    weighted = f"{variant}/weighted"
    weighted_4d = f"{variant}/weighted_4d"
    v_4d = f"{variant}/v_4d"
    pe_out = f"{variant}/pe"
    add_out = f"{variant}/add"
    output = f"{variant}/output"

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Conv",
            ["input", qkv.input[1], qkv.input[2]],
            [qkv_out],
            name=f"/sima_feasibility/{block_name}/{variant}/qkv/Conv",
            **node_attributes(qkv),
        ),
        helper.make_node(
            "Reshape",
            [qkv_out, f"{block_name}_{variant}_qkv_shape"],
            [qkv_reshape],
            name=f"/sima_feasibility/{block_name}/{variant}/Reshape",
        ),
        helper.make_node(
            "Split",
            [qkv_reshape, f"{block_name}_{variant}_split"],
            [q, k, v],
            name=f"/sima_feasibility/{block_name}/{variant}/Split",
            axis=2,
        ),
        helper.make_node(
            "Reshape",
            [v, f"{block_name}_{variant}_v4_shape"],
            [v_4d],
            name=f"/sima_feasibility/{block_name}/{variant}/VReshape",
        ),
        helper.make_node(
            "Conv",
            [v_4d, pe.input[1], pe.input[2]],
            [pe_out],
            name=f"/sima_feasibility/{block_name}/{variant}/pe/Conv",
            **node_attributes(pe),
        ),
    ]

    if variant == "supported_einsum":
        q_t = f"{variant}/q_nhwc"
        k_t = f"{variant}/k_nhqc"
        weighted_nhwq = f"{variant}/weighted_nhwq"
        nodes.extend(
            [
                helper.make_node(
                    "Transpose",
                    [q],
                    [q_t],
                    name=f"/sima_feasibility/{block_name}/{variant}/Q/Transpose",
                    perm=[0, 1, 3, 2],
                ),
                helper.make_node(
                    "Transpose",
                    [k],
                    [k_t],
                    name=f"/sima_feasibility/{block_name}/{variant}/K/Transpose",
                    perm=[0, 1, 3, 2],
                ),
                helper.make_node(
                    "Einsum",
                    [q_t, k_t],
                    [qk],
                    name=f"/sima_feasibility/{block_name}/{variant}/QK/Einsum",
                    equation="nhwc,nhqc->nhwq",
                ),
                helper.make_node(
                    "Mul",
                    [qk, scale.name],
                    [scaled],
                    name=f"/sima_feasibility/{block_name}/{variant}/Scale/Mul",
                ),
                helper.make_node(
                    "Softmax",
                    [scaled],
                    [prob],
                    name=f"/sima_feasibility/{block_name}/{variant}/Softmax",
                    axis=-1,
                ),
                helper.make_node(
                    "Einsum",
                    [prob, v],
                    [weighted_nhwq],
                    name=f"/sima_feasibility/{block_name}/{variant}/AV/Einsum",
                    equation="nhwc,nhqc->nhwq",
                ),
                helper.make_node(
                    "Transpose",
                    [weighted_nhwq],
                    [weighted],
                    name=f"/sima_feasibility/{block_name}/{variant}/Weighted/Transpose",
                    perm=[0, 1, 3, 2],
                ),
            ]
        )
    elif variant == "einsum":
        nodes.extend(
            [
                helper.make_node(
                    "Einsum",
                    [q, k],
                    [qk],
                    name=f"/sima_feasibility/{block_name}/{variant}/QK/Einsum",
                    equation="bhcn,bhcm->bhnm",
                ),
                helper.make_node(
                    "Mul",
                    [qk, scale.name],
                    [scaled],
                    name=f"/sima_feasibility/{block_name}/{variant}/Scale/Mul",
                ),
                helper.make_node(
                    "Softmax",
                    [scaled],
                    [prob],
                    name=f"/sima_feasibility/{block_name}/{variant}/Softmax",
                    axis=-1,
                ),
                helper.make_node(
                    "Einsum",
                    [v, prob],
                    [weighted],
                    name=f"/sima_feasibility/{block_name}/{variant}/AV/Einsum",
                    equation="bhcm,bhnm->bhcn",
                ),
            ]
        )
    elif variant == "mul_reduce":
        axes = {
            "q_unsqueeze": np.array([4], dtype=np.int64),
            "k_unsqueeze": np.array([3], dtype=np.int64),
            "qk_reduce": np.array([2], dtype=np.int64),
            "v_unsqueeze": np.array([3], dtype=np.int64),
            "prob_unsqueeze": np.array([2], dtype=np.int64),
            "av_reduce": np.array([4], dtype=np.int64),
        }
        for key, array in axes.items():
            add_initializer(initializers, f"{block_name}_{variant}_{key}_axes", array)
        q_u = f"{variant}/q_unsqueeze"
        k_u = f"{variant}/k_unsqueeze"
        qk_mul = f"{variant}/qk_mul"
        v_u = f"{variant}/v_unsqueeze"
        prob_u = f"{variant}/prob_unsqueeze"
        av_mul = f"{variant}/av_mul"
        nodes.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    [q, f"{block_name}_{variant}_q_unsqueeze_axes"],
                    [q_u],
                    name=f"/sima_feasibility/{block_name}/{variant}/Q/Unsqueeze",
                ),
                helper.make_node(
                    "Unsqueeze",
                    [k, f"{block_name}_{variant}_k_unsqueeze_axes"],
                    [k_u],
                    name=f"/sima_feasibility/{block_name}/{variant}/K/Unsqueeze",
                ),
                helper.make_node(
                    "Mul",
                    [q_u, k_u],
                    [qk_mul],
                    name=f"/sima_feasibility/{block_name}/{variant}/QK/Mul",
                ),
                helper.make_node(
                    "ReduceSum",
                    [qk_mul, f"{block_name}_{variant}_qk_reduce_axes"],
                    [qk],
                    name=f"/sima_feasibility/{block_name}/{variant}/QK/ReduceSum",
                    keepdims=0,
                ),
                helper.make_node(
                    "Mul",
                    [qk, scale.name],
                    [scaled],
                    name=f"/sima_feasibility/{block_name}/{variant}/Scale/Mul",
                ),
                helper.make_node(
                    "Softmax",
                    [scaled],
                    [prob],
                    name=f"/sima_feasibility/{block_name}/{variant}/Softmax",
                    axis=-1,
                ),
                helper.make_node(
                    "Unsqueeze",
                    [v, f"{block_name}_{variant}_v_unsqueeze_axes"],
                    [v_u],
                    name=f"/sima_feasibility/{block_name}/{variant}/V/Unsqueeze",
                ),
                helper.make_node(
                    "Unsqueeze",
                    [prob, f"{block_name}_{variant}_prob_unsqueeze_axes"],
                    [prob_u],
                    name=f"/sima_feasibility/{block_name}/{variant}/Prob/Unsqueeze",
                ),
                helper.make_node(
                    "Mul",
                    [v_u, prob_u],
                    [av_mul],
                    name=f"/sima_feasibility/{block_name}/{variant}/AV/Mul",
                ),
                helper.make_node(
                    "ReduceSum",
                    [av_mul, f"{block_name}_{variant}_av_reduce_axes"],
                    [weighted],
                    name=f"/sima_feasibility/{block_name}/{variant}/AV/ReduceSum",
                    keepdims=0,
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    nodes.extend(
        [
            helper.make_node(
                "Reshape",
                [weighted, f"{block_name}_{variant}_v4_shape"],
                [weighted_4d],
                name=f"/sima_feasibility/{block_name}/{variant}/WeightedReshape",
            ),
            helper.make_node(
                "Add",
                [weighted_4d, pe_out],
                [add_out],
                name=f"/sima_feasibility/{block_name}/{variant}/Add",
            ),
            helper.make_node(
                "Conv",
                [add_out, proj.input[1], proj.input[2]],
                [output],
                name=f"/sima_feasibility/{block_name}/{variant}/proj/Conv",
                **node_attributes(proj),
            ),
        ]
    )

    return make_model(nodes, initializers, output, tensor_shape(model, block["output"]))


def write_candidates(input_model: Path, work_dir: Path) -> dict[str, dict[str, Path]]:
    source = onnx.load(input_model)
    source = onnx.shape_inference.infer_shapes(source)
    candidates: dict[str, dict[str, Path]] = {}
    for block_name, block in ATTENTION_BLOCKS.items():
        block_dir = work_dir / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        block_candidates: dict[str, onnx.ModelProto] = {
            "original": original_attention_model(source, block),
            "supported_einsum": rewritten_attention_model(
                source, block_name, block, "supported_einsum"
            ),
            "einsum": rewritten_attention_model(source, block_name, block, "einsum"),
            "mul_reduce": rewritten_attention_model(source, block_name, block, "mul_reduce"),
        }
        candidates[block_name] = {}
        for variant, model in block_candidates.items():
            path = block_dir / f"{variant}.onnx"
            onnx.save(model, path)
            candidates[block_name][variant] = path
    return candidates


def compare_candidates(candidates: dict[str, dict[str, Path]]) -> dict[str, Any]:
    rng = np.random.default_rng(26)
    report: dict[str, Any] = {}
    for block_name, variants in candidates.items():
        block_report: dict[str, Any] = {}
        x = rng.normal(0.0, 1.0, size=(1, 128, 20, 20)).astype(np.float32)
        original_session = ort.InferenceSession(str(variants["original"]), providers=["CPUExecutionProvider"])
        original_output = original_session.run(None, {"input": x})[0]
        for variant, path in variants.items():
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            output = session.run(None, {"input": x})[0]
            diff = np.abs(original_output - output)
            block_report[variant] = {
                "path": str(path),
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "allclose_atol_1e_5_rtol_1e_5": bool(
                    np.allclose(original_output, output, atol=1e-5, rtol=1e-5)
                ),
            }
        report[block_name] = block_report
    return report


def compile_candidate(model_path: Path, output_dir: Path) -> dict[str, Any]:
    try:
        import logging

        from afe.apis.defines import (
            InputName,
            bfloat16_scheme,
            CalibrationMethod,
            default_quantization,
            gen2_target,
            RequantizationMode,
        )
        from afe.apis.loaded_net import load_model
        from afe.ir.tensor_type import ScalarType
        from afe.load.importers.general_importer import ImporterParams, ModelFormat
        from sima_utils.data.data_generator import DataGenerator
        from afe.core.utils import convert_data_generator_to_iterable
    except Exception as exc:  # pragma: no cover - depends on Model SDK env
        return {"status": "skipped", "reason": f"Model SDK import failed: {exc}"}

    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_names = [output.name for output in model.graph.output]

    try:
        loaded = load_model(
            ImporterParams(
                format=ModelFormat.onnx,
                file_paths=[str(model_path)],
                input_names=[input_name],
                input_shapes=[tuple(input_shape)],
                input_types=[ScalarType.float32],
                layout="NCHW",
                output_names=output_names,
            ),
            target=gen2_target,
        )
        data = {
            InputName(input_name): np.random.default_rng(0)
            .random((1, input_shape[2], input_shape[3], input_shape[1]), dtype=np.float32)
        }
        calib = convert_data_generator_to_iterable(DataGenerator(data))
        bf16 = bfloat16_scheme()
        quant_config = (
            default_quantization.with_activation_quantization(bf16)
            .with_weight_quantization(bf16)
            .with_requantization_mode(RequantizationMode.sima)
            .with_calibration(CalibrationMethod.from_str("mse"))
        )
        model_name = model_path.stem
        quant_model = loaded.quantize(
            calibration_data=calib,
            quantization_config=quant_config,
            any_shape_on_mla=True,
            automatic_layout_conversion=False,
            model_name=model_name,
            log_level=logging.INFO,
        )
        quant_model.compile(output_path=str(output_dir), batch_size=1, log_level=logging.INFO)

        mpks = sorted(output_dir.glob("**/*_mpk.tar.gz"))
        if not mpks:
            return {"status": "failed", "reason": "compile completed but no *_mpk.tar.gz found"}
        mpk = mpks[-1]
        with tarfile.open(mpk) as tar:
            names = tar.getnames()
            mpk_json_name = next(name for name in names if name.endswith("_mpk.json"))
            mpk_json = json.load(tar.extractfile(mpk_json_name))  # type: ignore[arg-type]
            processor_counts: dict[str, int] = {}
            for plugin in mpk_json.get("plugins", []):
                processor = plugin.get("processor") or plugin.get("type") or plugin.get("backend")
                processor_counts[processor] = processor_counts.get(processor, 0) + 1
            return {
                "status": "compiled",
                "mpk": str(mpk),
                "mla_elf_count": sum(name.endswith("_mla.elf") for name in names),
                "so_count": sum(name.endswith(".so") for name in names),
                "process_tvm_count": sum("process_tvm" in name for name in names),
                "processor_counts": processor_counts,
            }
    except BaseException as exc:  # pragma: no cover - depends on Model SDK compile path
        return {
            "status": "failed",
            "reason": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }


def compile_candidates(
    candidates: dict[str, dict[str, Path]],
    work_dir: Path,
    compile_all_blocks: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for block_index, (block_name, variants) in enumerate(candidates.items()):
        if block_index > 0 and not compile_all_blocks:
            report[block_name] = {"status": "skipped", "reason": "use --compile-all-blocks to compile both blocks"}
            continue
        block_report: dict[str, Any] = {}
        for variant, path in variants.items():
            block_report[variant] = compile_candidate(path, work_dir / "compile" / block_name / variant)
        report[block_name] = block_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO26 exact attention rewrite feasibility spike.")
    parser.add_argument("--input", required=True, help="Input raw-head YOLO26 ONNX path.")
    parser.add_argument("--work-dir", required=True, help="Directory for generated candidates and report.")
    parser.add_argument("--compile", action="store_true", help="Compile candidate attention blocks with Model SDK.")
    parser.add_argument(
        "--compile-all-blocks",
        action="store_true",
        help="Compile model10 and model22 candidates. By default only model10 is compiled.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    candidates = write_candidates(Path(args.input), work_dir)
    report: dict[str, Any] = {
        "input": args.input,
        "candidates": {
            block: {variant: str(path) for variant, path in variants.items()}
            for block, variants in candidates.items()
        },
        "onnxruntime_exactness": compare_candidates(candidates),
        "static_unroll": {
            "status": "not_generated",
            "reason": (
                "Exact static unroll at YOLO26 shape requires materializing roughly "
                "400x400 dynamic dot products per head plus AV reductions, creating "
                "hundreds of thousands of nodes per attention block before Softmax. "
                "It is not a viable full-model graph-surgery target unless the compiler "
                "first proves support for the same dynamic Mul/ReduceSum/Softmax pattern."
            ),
        },
    }
    if args.compile:
        report["modelsdk_compile"] = compile_candidates(candidates, work_dir, args.compile_all_blocks)

    report_path = work_dir / "attention_feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

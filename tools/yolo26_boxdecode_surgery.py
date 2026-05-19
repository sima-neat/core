#!/usr/bin/env python3
"""Expose YOLO26 raw one2one heads for native SiMa BoxDecode-v2 postprocess.

The current YOLO26 detection head uses ``reg_max=1``: bbox heads are raw
l/t/r/b distances with 4 channels, not 64-channel DFL logits.  This surgery
therefore keeps bbox decoding out of ONNX and exposes:

    bbox_0, bbox_1, bbox_2, class_logit_0, class_logit_1, class_logit_2

in grouped-by-role order for ``BoxDecodeType::YoloV26``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx import utils as onnx_utils


BBOX_SOURCES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
]

CLASS_LOGIT_SOURCES = [
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]

ATTENTION_RESIDUAL_ADDS = [
    (
        "/model.10/m/m.0/Add",
        "/model.10/m/m.0/attn/proj/conv/Conv_output_0",
        "/sima_yolo26_mla_only/model10_attention_zero",
    ),
    (
        "/model.22/m.0/m.0.1/Add",
        "/model.22/m.0/m.0.1/attn/proj/conv/Conv_output_0",
        "/sima_yolo26_mla_only/model22_attention_zero",
    ),
]

ATTENTION_BLOCKS = [
    {
        "tag": "model10",
        "base": "/model.10/m/m.0/attn",
        "residual_add": "/model.10/m/m.0/Add",
        "attention_output": "/model.10/m/m.0/attn/proj/conv/Conv_output_0",
    },
    {
        "tag": "model22",
        "base": "/model.22/m.0/m.0.1/attn",
        "residual_add": "/model.22/m.0/m.0.1/Add",
        "attention_output": "/model.22/m.0/m.0.1/attn/proj/conv/Conv_output_0",
    },
]

SUPPORTED_EINSUM_EQUATION = "nhwc,nhqc->nhwq"


def tensor_shape(model: onnx.ModelProto, name: str) -> list[int]:
    for value in [*model.graph.value_info, *model.graph.input, *model.graph.output]:
        if value.name != name:
            continue
        shape = value.type.tensor_type.shape
        dims: list[int] = []
        for dim in shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            else:
                raise ValueError(f"Dynamic dimension found for {name}; export YOLO26 static.")
        return dims
    raise ValueError(f"Could not find inferred shape for tensor: {name}")


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


def node_attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def add_int64_initializer(model: onnx.ModelProto, name: str, values: list[int]) -> None:
    tensor = numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)
    model.graph.initializer.append(tensor)


def add_float_zero_initializer(model: onnx.ModelProto, name: str, shape: list[int]) -> None:
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Cannot create static zero initializer for dynamic shape {shape}: {name}")
    tensor = numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name)
    model.graph.initializer.append(tensor)


def bypass_attention_for_mla_only(model: onnx.ModelProto) -> None:
    """Remove YOLO26 attention islands by zeroing their residual contribution.

    Model SDK 2.0.0 does not place YOLO26's dynamic attention batch-matmul on
    MLA.  This approximate mode preserves the residual path and downstream FFN,
    but replaces each attention projection input to the residual add with a
    static zero tensor.  A later extract_model pass prunes the now-unreachable
    MatMul/Softmax attention island.
    """

    for add_name, attention_output, zero_name in ATTENTION_RESIDUAL_ADDS:
        shape = tensor_shape(model, attention_output)
        add_float_zero_initializer(model, zero_name, shape)
        add_node = find_node(model, add_name)
        replaced = False
        for index, input_name in enumerate(add_node.input):
            if input_name == attention_output:
                add_node.input[index] = zero_name
                replaced = True
                break
        if not replaced:
            raise ValueError(
                f"Expected {add_name} to consume {attention_output}, "
                f"but inputs are {list(add_node.input)}"
            )


def replace_attention_with_supported_einsum(model: onnx.ModelProto) -> None:
    """Rewrite YOLO26 attention to the compiler-supported exact Einsum form.

    The original export lowers attention as batched MatMul with layouts
    ``[N, heads, tokens, channels] @ [N, heads, channels, tokens]``.  The
    Model SDK support DB documents a Modalix-supported Einsum equation
    ``nhwc,nhqc->nhwq``.  This rewrite preserves the exact math while arranging
    Q/K and attention/V operands to use that equation.  The original attention
    branch is pruned by extract_model after the residual Add is rewired.
    """

    for block in ATTENTION_BLOCKS:
        tag = block["tag"]
        base = block["base"]
        qkv = find_node(model, f"{base}/qkv/conv/Conv")
        split = find_node(model, f"{base}/Split")
        pe = find_node(model, f"{base}/pe/conv/Conv")
        proj = find_node(model, f"{base}/proj/conv/Conv")
        mul = find_node(model, f"{base}/Mul")
        residual_add = find_node(model, block["residual_add"])

        prefix = f"/sima_yolo26_exact_attention/{tag}"
        qkv_shape = f"{prefix}/qkv_shape"
        v4_shape = f"{prefix}/v4_shape"
        add_int64_initializer(model, qkv_shape, [1, 2, 128, 400])
        add_int64_initializer(model, v4_shape, [1, 128, 20, 20])

        qkv_reshape = f"{prefix}/qkv_reshape_output_0"
        q = f"{prefix}/q_output_0"
        k = f"{prefix}/k_output_0"
        v = f"{prefix}/v_output_0"
        q_t = f"{prefix}/q_nhwc_output_0"
        k_t = f"{prefix}/k_nhqc_output_0"
        qk = f"{prefix}/qk_output_0"
        scaled = f"{prefix}/scaled_output_0"
        prob = f"{prefix}/prob_output_0"
        v4 = f"{prefix}/v4_output_0"
        pe_out = f"{prefix}/pe_output_0"
        weighted_nhwq = f"{prefix}/weighted_nhwq_output_0"
        weighted = f"{prefix}/weighted_output_0"
        weighted_4d = f"{prefix}/weighted_4d_output_0"
        add_out = f"{prefix}/add_output_0"
        proj_out = f"{prefix}/proj_output_0"

        nodes = [
            helper.make_node(
                "Reshape",
                [qkv.output[0], qkv_shape],
                [qkv_reshape],
                name=f"{prefix}/Reshape",
            ),
            helper.make_node(
                "Split",
                [qkv_reshape, split.input[1]],
                [q, k, v],
                name=f"{prefix}/Split",
                axis=2,
            ),
            helper.make_node(
                "Transpose",
                [q],
                [q_t],
                name=f"{prefix}/Q/Transpose",
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "Transpose",
                [k],
                [k_t],
                name=f"{prefix}/K/Transpose",
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "Einsum",
                [q_t, k_t],
                [qk],
                name=f"{prefix}/QK/Einsum",
                equation=SUPPORTED_EINSUM_EQUATION,
            ),
            helper.make_node(
                "Mul",
                [qk, mul.input[1]],
                [scaled],
                name=f"{prefix}/Scale/Mul",
            ),
            helper.make_node(
                "Softmax",
                [scaled],
                [prob],
                name=f"{prefix}/Softmax",
                axis=-1,
            ),
            helper.make_node(
                "Reshape",
                [v, v4_shape],
                [v4],
                name=f"{prefix}/V/Reshape",
            ),
            helper.make_node(
                "Conv",
                [v4, pe.input[1], pe.input[2]],
                [pe_out],
                name=f"{prefix}/pe/Conv",
                **node_attributes(pe),
            ),
            helper.make_node(
                "Einsum",
                [prob, v],
                [weighted_nhwq],
                name=f"{prefix}/AV/Einsum",
                equation=SUPPORTED_EINSUM_EQUATION,
            ),
            helper.make_node(
                "Transpose",
                [weighted_nhwq],
                [weighted],
                name=f"{prefix}/Weighted/Transpose",
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "Reshape",
                [weighted, v4_shape],
                [weighted_4d],
                name=f"{prefix}/Weighted/Reshape",
            ),
            helper.make_node(
                "Add",
                [weighted_4d, pe_out],
                [add_out],
                name=f"{prefix}/Add",
            ),
            helper.make_node(
                "Conv",
                [add_out, proj.input[1], proj.input[2]],
                [proj_out],
                name=f"{prefix}/proj/Conv",
                **node_attributes(proj),
            ),
        ]

        insert_index = list(model.graph.node).index(residual_add)
        for offset, node in enumerate(nodes):
            model.graph.node.insert(insert_index + offset, node)

        replaced = False
        for index, input_name in enumerate(residual_add.input):
            if input_name == block["attention_output"]:
                residual_add.input[index] = proj_out
                replaced = True
                break
        if not replaced:
            raise ValueError(
                f"Expected {block['residual_add']} to consume {block['attention_output']}, "
                f"but inputs are {list(residual_add.input)}"
            )


def add_identity_output(
    model: onnx.ModelProto,
    source: str,
    output_name: str,
    expected_rank: int,
    expected_channels: int | None,
) -> None:
    shape = tensor_shape(model, source)
    if len(shape) != expected_rank:
        raise ValueError(f"{source} shape rank {len(shape)} != expected {expected_rank}: {shape}")
    if expected_channels is not None and shape[1] != expected_channels:
        raise ValueError(
            f"{source} channel depth {shape[1]} != expected {expected_channels}: {shape}"
        )
    model.graph.node.append(
        helper.make_node(
            "Identity",
            inputs=[source],
            outputs=[output_name],
            name=f"/sima_yolo26_heads/{output_name}/Identity",
        )
    )
    model.graph.output.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
    # onnx.utils.extract_model in ONNX 1.17 indexes value_info, not graph.output.
    model.graph.value_info.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))


def expose_yolo26_heads(model: onnx.ModelProto) -> list[str]:
    del model.graph.output[:]

    output_names: list[str] = []
    for index, source in enumerate(BBOX_SOURCES):
        name = f"bbox_{index}"
        add_identity_output(model, source, name, expected_rank=4, expected_channels=4)
        output_names.append(name)

    class_depth: int | None = None
    for index, source in enumerate(CLASS_LOGIT_SOURCES):
        shape = tensor_shape(model, source)
        if class_depth is None:
            class_depth = shape[1]
        elif shape[1] != class_depth:
            raise ValueError(
                f"Class depth mismatch: {source} has {shape[1]}, expected {class_depth}"
            )
        name = f"class_logit_{index}"
        add_identity_output(model, source, name, expected_rank=4, expected_channels=class_depth)
        output_names.append(name)

    outputs_by_name = {output.name: output for output in model.graph.output}
    del model.graph.output[:]
    model.graph.output.extend(outputs_by_name[name] for name in output_names)
    return output_names


def validate_yolo26_outputs(model: onnx.ModelProto) -> None:
    expected = {
        "bbox_0": [1, 4, 80, 80],
        "bbox_1": [1, 4, 40, 40],
        "bbox_2": [1, 4, 20, 20],
        "class_logit_0": [1, 80, 80, 80],
        "class_logit_1": [1, 80, 40, 40],
        "class_logit_2": [1, 80, 20, 20],
    }
    actual_names = [output.name for output in model.graph.output]
    expected_names = list(expected)
    if actual_names != expected_names:
        raise ValueError(f"Unexpected output order: {actual_names}; expected {expected_names}")
    for name, expected_shape in expected.items():
        actual_shape = tensor_shape(model, name)
        if actual_shape != expected_shape:
            raise ValueError(f"{name} shape {actual_shape} != expected {expected_shape}")


def validate_mla_only_graph(model: onnx.ModelProto, attention_mode: str) -> None:
    op_counts = Counter(node.op_type for node in model.graph.node)
    if op_counts["MatMul"]:
        raise ValueError(f"MLA-only graph still contains MatMul nodes: {op_counts['MatMul']}")
    for node in model.graph.node:
        if node.op_type != "Einsum":
            continue
        equation = None
        for attr in node.attribute:
            if attr.name == "equation":
                equation = helper.get_attribute_value(attr)
                if isinstance(equation, bytes):
                    equation = equation.decode("utf-8")
        if equation != SUPPORTED_EINSUM_EQUATION:
            raise ValueError(f"Unsupported Einsum equation for MLA-only graph: {node.name} {equation}")
    if attention_mode == "zero" and op_counts["Softmax"]:
        raise ValueError(f"Zero-attention MLA-only graph still contains Softmax: {op_counts['Softmax']}")


def write_pruned_model(
    input_path: Path,
    output_path: Path,
    input_name: str | None,
    attention_mode: str,
    check_mla_only: bool,
) -> None:
    model = onnx.load(input_path)
    model = onnx.shape_inference.infer_shapes(model)
    if attention_mode == "zero":
        bypass_attention_for_mla_only(model)
    elif attention_mode == "supported-einsum":
        replace_attention_with_supported_einsum(model)
    actual_input_name = input_name or model.graph.input[0].name
    output_names = expose_yolo26_heads(model)
    model = onnx.shape_inference.infer_shapes(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "yolo26_heads_unpruned.onnx"
        onnx.save(model, tmp_path)
        onnx_utils.extract_model(
            str(tmp_path),
            str(output_path),
            [actual_input_name],
            output_names,
            check_model=True,
        )

    pruned = onnx.load(output_path)
    pruned = onnx.shape_inference.infer_shapes(pruned)
    onnx.checker.check_model(pruned)
    validate_yolo26_outputs(pruned)
    if check_mla_only:
        validate_mla_only_graph(pruned, attention_mode)
    onnx.save(pruned, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expose YOLO26 raw bbox/class logits for SiMa BoxDecodeType::YoloV26."
    )
    parser.add_argument("--input", required=True, help="Input YOLO26 ONNX path.")
    parser.add_argument("--output", required=True, help="Output surgically pruned ONNX path.")
    parser.add_argument(
        "--input-name",
        default=None,
        help="Model input name. Defaults to the first ONNX graph input.",
    )
    parser.add_argument(
        "--attention-mode",
        choices=["preserve", "supported-einsum", "zero"],
        default="preserve",
        help=(
            "preserve keeps original YOLO26 attention; supported-einsum rewrites attention "
            "into the exact Model SDK supported Einsum form; zero replaces attention residual "
            "contributions with zeros as an approximate fallback."
        ),
    )
    parser.add_argument(
        "--check-mla-only",
        action="store_true",
        help="Reject the output if MatMul/Einsum/Softmax remain after pruning.",
    )
    args = parser.parse_args()

    write_pruned_model(
        Path(args.input),
        Path(args.output),
        args.input_name,
        args.attention_mode,
        args.check_mla_only,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

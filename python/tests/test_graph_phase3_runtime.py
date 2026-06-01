import os
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import pyneat

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tests" / "perf" / "tools"
if _TOOLS_DIR.is_dir() and str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
try:
    import graph_run_schema
except Exception:  # pragma: no cover - source-tree tests have it; installed wheels may not.
    graph_run_schema = None


def _phase3_runtime_enabled():
    return os.environ.get("SIMA_GRAPH_PHASE3_PY_RUNTIME", "0") not in ("", "0", "false", "False", "no")


def _require_phase3_runtime():
    if not _phase3_runtime_enabled():
        pytest.skip("set SIMA_GRAPH_PHASE3_PY_RUNTIME=1 to run pyneat Graph Phase 3 runtime smoke")


def _resnet_tar_or_skip():
    raw = os.environ.get("SIMA_RESNET50_TAR") or os.environ.get("SIMA_MODEL_TAR")
    if not raw:
        pytest.skip("set SIMA_RESNET50_TAR for pyneat Graph.add(Model) runtime smoke")
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"missing ResNet model tarball: {path}")
    return path


def test_graph_connect_tensor_runtime_smoke():
    _require_phase3_runtime()

    source = pyneat.Graph()
    source.add(pyneat.nodes.input())
    sink = pyneat.Graph()
    sink.add(pyneat.nodes.output())

    app = pyneat.Graph()
    app.connect(source, sink)

    arr = np.full((24, 32, 3), 0x44, dtype=np.uint8)
    run = app.build()
    try:
        assert run.push([arr], layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB)
        out = run.pull_tensors(5000)
        assert len(out) >= 1
    finally:
        run.close()


def test_graph_named_endpoint_runtime_smoke():
    _require_phase3_runtime()

    source = pyneat.Graph("image")
    source.add(pyneat.nodes.input("image"))
    sink = pyneat.Graph("classes")
    sink.add(pyneat.nodes.output("classes"))

    app = pyneat.Graph()
    app.connect(source, sink)

    arr = np.full((24, 32, 3), 0x55, dtype=np.uint8)
    run = app.build()
    try:
        assert "image" in run.input_names()
        assert "classes" in run.output_names()
        assert run.push(
            "image",
            [arr],
            layout=pyneat.TensorLayout.HWC,
            image_format=pyneat.PixelFormat.RGB,
        )
        out = run.pull_tensors("classes", 5000)
        assert len(out) >= 1
    finally:
        run.close()


def test_run_export_runtime_smoke(tmp_path):
    _require_phase3_runtime()

    source = pyneat.Graph("image")
    source.add(pyneat.nodes.input("image"))
    sink = pyneat.Graph("classes")
    sink.add(pyneat.nodes.output("classes"))

    app = pyneat.Graph("exportable")
    app.connect(source, sink)

    auto_path = tmp_path / "auto.graph_run.json"
    build_opt = pyneat.RunOptions()
    build_opt.run_export.path = str(auto_path)
    build_opt.run_export.label = "py_auto_export"

    arr = np.full((24, 32, 3), 0x57, dtype=np.uint8)
    run = app.build(build_opt)
    try:
        assert auto_path.exists()
        auto_json = json.loads(auto_path.read_text())
        assert auto_json["schema"] == "sima.neat.graph_run"
        assert auto_json["label"] == "py_auto_export"
        if graph_run_schema is not None:
            graph_run_schema.validate_graph_run(auto_json)
        assert auto_json["graph"]["public_view"]["nodes"]
        assert any(
            edge.get("from_endpoint") == "image" and edge.get("to_endpoint") == "classes"
            for edge in auto_json["graph"]["public_view"]["edges"]
        )
        assert auto_json["graph"]["lowered_view"]["nodes"]

        assert run.push(
            "image",
            [arr],
            layout=pyneat.TensorLayout.HWC,
            image_format=pyneat.PixelFormat.RGB,
        )
        out = run.pull_tensors("classes", 5000)
        assert len(out) >= 1

        export_opt = pyneat.RunExportOptions()
        export_opt.label = "py_post_export"
        body = run.json(export_opt)
        post_json = json.loads(body)
        assert post_json["label"] == "py_post_export"
        if graph_run_schema is not None:
            graph_run_schema.validate_graph_run(post_json)
        assert post_json["run"]["stats"]["outputs_pulled"] >= 1
        assert post_json["graph"]["public_view"]["edges"]

        post_path = tmp_path / "post.graph_run.json"
        pyneat.save_run_json(run, str(post_path), export_opt)
        assert json.loads(post_path.read_text())["schema"] == "sima.neat.graph_run"
    finally:
        run.close()


def test_graph_named_output_auto_suffix_runtime_smoke():
    _require_phase3_runtime()

    source = pyneat.Graph("image")
    source.add(pyneat.nodes.input("image"))
    sinks = pyneat.Graph("classes")
    sinks.add(pyneat.nodes.output())
    sinks.add(pyneat.nodes.output())
    sinks.add(pyneat.nodes.output())

    app = pyneat.Graph()
    app.connect(source, sinks)

    arr = np.full((24, 32, 3), 0x56, dtype=np.uint8)
    run = app.build()
    try:
        assert run.output_names() == ["classes_0", "classes_1", "classes_2"]
        assert run.push(
            "image",
            [arr],
            layout=pyneat.TensorLayout.HWC,
            image_format=pyneat.PixelFormat.RGB,
        )
        for name in ("classes_0", "classes_1", "classes_2"):
            out = run.pull_tensors(name, 5000)
            assert len(out) >= 1
    finally:
        run.close()


def test_graph_stage_node_pull_all_and_bare_input_fanout_runtime_smoke():
    _require_phase3_runtime()

    seen = {"n": 0}

    def mark(sample):
        seen["n"] += 1
        sample.stream_label = "stage_ok"
        return sample

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input("image"))
    graph.add(pyneat.nodes.stage(mark, label="mark_meta"))
    graph.add(pyneat.nodes.output("pose"))

    arr = np.full((8, 8, 3), 7, dtype=np.uint8)
    run = graph.build()
    try:
        assert run.push(
            "image",
            [arr],
            layout=pyneat.TensorLayout.HWC,
            image_format=pyneat.PixelFormat.RGB,
        )
        out = run.pull_all(timeout_ms=5000)
        assert out["pose"] is not None
        assert seen["n"] == 1
    finally:
        run.close()

    # Regression: public Input fanout should not require app code to insert nodes.queue().
    source = pyneat.Graph("image")
    source.add(pyneat.nodes.input("image"))
    sinks = pyneat.Graph("classes")
    sinks.add(pyneat.nodes.output("a"))
    sinks.add(pyneat.nodes.output("b"))
    app = pyneat.Graph()
    app.connect(source, sinks)
    run = app.build()
    try:
        assert run.push(
            "image",
            [arr],
            layout=pyneat.TensorLayout.HWC,
            image_format=pyneat.PixelFormat.RGB,
        )
        out = run.pull_all(timeout_ms=5000)
        assert set(out) == {"a", "b"}
        assert all(sample is not None for sample in out.values())
    finally:
        run.close()


def test_graph_add_model_resnet_runtime_smoke():
    _require_phase3_runtime()
    tar = _resnet_tar_or_skip()

    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    opt.upstream_name = "decoder"

    model = pyneat.Model(str(tar), opt)
    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input())
    graph.add(model)
    graph.add(pyneat.nodes.output())

    img = np.full((224, 224, 3), 37, dtype=np.uint8)
    run = graph.build([img])
    try:
        assert run.push([img])
        out = run.pull_tensors(20000)
        assert len(out) >= 1
    finally:
        run.close()


def test_graph_ambiguous_default_diagnostic_python():
    _require_phase3_runtime()

    input_a = pyneat.Graph()
    input_a.add(pyneat.nodes.input())
    input_b = pyneat.Graph()
    input_b.add(pyneat.nodes.input())
    sink_a = pyneat.Graph()
    sink_a.add(pyneat.nodes.output())
    sink_b = pyneat.Graph()
    sink_b.add(pyneat.nodes.output())

    app = pyneat.Graph()
    app.connect(input_a, sink_a)
    app.connect(input_b, sink_b)

    run = app.build()
    try:
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(Exception, match="no unambiguous default input"):
            run.push([arr], layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB)
    finally:
        run.close()

import numpy as np

import pyneat


def _frame():
    return pyneat.Tensor.from_numpy(
        np.zeros((48, 64, 3), np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.CPU,
    )


def _named_graph():
    graph = pyneat.Graph("last_pipeline_named")
    graph.add(pyneat.nodes.input("in"))
    graph.add(pyneat.nodes.output("out"))
    return graph


def test_last_pipeline_is_populated_for_named_endpoints():
    # A graph with named endpoints must report the launch string of its pipeline, just like an
    # unnamed one, and keep it across push/pull and close.
    graph = _named_graph()
    assert graph.last_pipeline == ""

    frame = _frame()
    run = graph.build([frame])
    try:
        built = graph.last_pipeline
        assert built != ""
        assert "appsrc" in built
        assert "appsink" in built
        assert "format=RGB" in built
        assert "width=64" in built
        assert "height=48" in built

        assert run.push("in", [frame])
        assert run.pull("out", 10000) is not None
        assert graph.last_pipeline == built
    finally:
        run.close()
    assert graph.last_pipeline == built

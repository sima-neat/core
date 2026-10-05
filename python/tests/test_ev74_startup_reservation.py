"""EV74 reservation must fail at build, without seed execution or an output sink."""
import fcntl
import os
import struct
from pathlib import Path

import numpy as np
import pyneat
import pytest


def make_graph(source):
    graph = pyneat.Graph()
    if source:
        graph.custom_with_role(
            "videotestsrc is-live=true pattern=black ! "
            "video/x-raw,format=RGB,width=1280,height=720",
            pyneat.InputRole.Source,
        )
    else:
        options = pyneat.InputOptions()
        options.format = pyneat.FormatTag.RGB
        options.width = 1280
        options.height = 720
        options.depth = 3
        options.is_live = True
        options.memory_policy = pyneat.InputMemoryPolicy.Ev74
        options.buffer_name = "decoder"
        graph.add(pyneat.nodes.input(options))
    preproc = pyneat.PreprocOptions()
    preproc.num_buffers = 4
    preproc.model_managed_contract = source
    preproc.set_input_shape([720, 1280, 3])
    preproc.set_output_shape([640, 640, 3])
    preproc.set_slice_shape([32, 128, 3])
    preproc.scaled_width = 640
    preproc.scaled_height = 640
    preproc.input_img_type = "RGB"
    preproc.output_img_type = "RGB"
    preproc.normalize = False
    preproc.aspect_ratio = False
    preproc.output_dtype = "EVXX_INT8"
    preproc.q_scale = 0.25
    preproc.q_zp = 0
    preproc.upstream_name = "decoder"
    preproc.next_cpu = "APU"
    graph.add(pyneat.nodes.preproc(preproc))
    if not source:
        graph.add(pyneat.nodes.output())
    else:
        graph.custom("fakesink sync=false async=false")
    return graph


@pytest.mark.parametrize("source", [False, True])
def test_capacity_fails_at_build(monkeypatch, source):
    monkeypatch.setenv("SIMA_RPMSG_ACQUIRE_TIMEOUT_MS", "0")
    monkeypatch.setenv("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0")
    options = pyneat.RunOptions()
    options.startup_preflight = False
    seed = pyneat.Tensor.from_numpy(
        np.full((720, 1280, 3), 64, dtype=np.uint8),
        copy=True, image_format=pyneat.PixelFormat.RGB,
    )
    # Initialize endpoint discovery before taking locks. Closing the final
    # client must leave all channels available to this independent holder.
    control = make_graph(False).build([seed], options=options)
    control.close()
    paths = sorted(
        Path("/tmp") / ("rpmsg_lock_" + node.name)
        for node in Path("/sys/class/rpmsg").iterdir()
        if node.name.startswith("rpmsg") and node.name[5:].isdigit()
    )
    assert paths, "No RPMsg endpoints discovered"
    holders = []
    try:
        # OFD locks also conflict with the old process-owned lockf reservation.
        for path in paths:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            holders.append(fd)
            fcntl.fcntl(fd, fcntl.F_OFD_SETLK,
                        struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        graph = make_graph(source)
        with pytest.raises(pyneat.NeatError) as error:
            if source:
                graph.build_source(options=options)
            else:
                graph.build([seed], options=options)
        assert error.value.error_code == pyneat.ERROR_DISPATCHER_UNAVAILABLE
        assert "RPMsg capacity exhausted" in error.value.report_json
    finally:
        for fd in holders:
            os.close(fd)
    retry = make_graph(False).build([seed], options=options)
    retry.close()

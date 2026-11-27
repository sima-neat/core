"""Per-frame `Sample.attributes` and multipart header-capture configuration.

These cover the public Python surface only: mapping semantics, independence of copies,
and configuration validation. Association through decode is proven by the board tests.
"""

import numpy as np
import pyneat
import pytest


def test_attributes_default_empty():
    sample = pyneat.Sample()
    assert len(sample.attributes) == 0


def test_whole_dict_assignment_and_read():
    sample = pyneat.Sample()
    sample.attributes = {"image-index": "42", "image-time": "2026-08-07T12:00:00Z"}
    assert dict(sample.attributes) == {
        "image-index": "42",
        "image-time": "2026-08-07T12:00:00Z",
    }


def test_live_item_mutation_reaches_the_sample():
    """A copied dict would accept this assignment and silently lose it."""
    sample = pyneat.Sample()
    sample.attributes["image-index"] = "7"
    assert dict(sample.attributes) == {"image-index": "7"}

    sample.attributes["image-index"] = "8"
    sample.attributes["image-channel"] = "rgb"
    assert dict(sample.attributes) == {"image-index": "8", "image-channel": "rgb"}


def test_item_deletion_and_clear():
    sample = pyneat.Sample()
    sample.attributes = {"a": "1", "b": "2"}
    del sample.attributes["a"]
    assert dict(sample.attributes) == {"b": "2"}

    sample.attributes.clear()
    assert len(sample.attributes) == 0


def test_whole_dict_assignment_replaces_rather_than_merges():
    sample = pyneat.Sample()
    sample.attributes = {"a": "1", "b": "2"}
    sample.attributes = {"c": "3"}
    assert dict(sample.attributes) == {"c": "3"}


def test_membership_and_iteration():
    sample = pyneat.Sample()
    sample.attributes = {"image-index": "1", "image-time": "t"}
    assert "image-index" in sample.attributes
    assert "missing" not in sample.attributes
    assert sorted(sample.attributes) == ["image-index", "image-time"]


@pytest.mark.parametrize("method", ["push", "try_push"])
def test_push_rejects_nul_key_and_accepts_corrected_sample(method):
    frame = pyneat.Tensor.from_numpy(
        np.zeros((24, 32, 3), np.uint8), copy=True, image_format=pyneat.PixelFormat.RGB
    )
    sample = pyneat.make_tensor_sample("frame", frame)
    sample.attributes["bad\0key"] = "42"
    graph = pyneat.Graph("sample_attributes")
    graph.add(pyneat.nodes.input("in"))
    graph.add(pyneat.nodes.output("out"))
    run = graph.build([frame])
    try:
        push = getattr(run, method)
        with pytest.raises(pyneat.NeatError) as exc:
            push([sample])
        assert exc.value.error_code == pyneat.ERROR_SAMPLE_ATTRIBUTES
        assert r'"bad\u0000key"' in str(exc.value)

        del sample.attributes["bad\0key"]
        sample.attributes["image-index"] = "42"
        assert push([sample])
        result = run.pull(timeout_ms=2000)
        assert result is not None
        assert dict(result.attributes) == {"image-index": "42"}
    finally:
        run.close()


def test_distinct_samples_do_not_alias():
    first = pyneat.Sample()
    second = pyneat.Sample()
    first.attributes["shared"] = "first"
    second.attributes["shared"] = "second"
    assert first.attributes["shared"] == "first"
    assert second.attributes["shared"] == "second"


def test_assigned_dict_is_copied_not_aliased():
    source = {"k": "original"}
    sample = pyneat.Sample()
    sample.attributes = source
    source["k"] = "mutated"
    assert sample.attributes["k"] == "original"


def test_header_capture_defaults_to_disabled():
    opt = pyneat.MultipartHeaderCaptureOptions()
    assert list(opt.headers) == []
    assert opt.enabled() is False

    http = pyneat.HttpMjpegDecodedInputOptions()
    assert http.header_capture.enabled() is False


def test_header_capture_configuration_round_trips():
    http = pyneat.HttpMjpegDecodedInputOptions()
    capture = pyneat.MultipartHeaderCaptureOptions()
    capture.headers = ["Image-Index", "Image-Time"]
    http.header_capture = capture
    assert list(http.header_capture.headers) == ["Image-Index", "Image-Time"]
    assert http.header_capture.enabled() is True


def test_capture_disabled_graph_keeps_existing_topology():
    opt = pyneat.HttpMjpegDecodedInputOptions()
    opt.url = "http://example.local/mjpeg"
    graph = pyneat.groups.http_mjpeg_decoded_input(opt)
    described = graph.describe()
    assert "MultipartJpegDemux" in described
    assert "JpegParse" in described


def test_capture_enabled_graph_omits_jpegparse():
    opt = pyneat.HttpMjpegDecodedInputOptions()
    opt.url = "http://example.local/mjpeg"
    opt.header_capture.headers = ["Image-Index"]
    graph = pyneat.groups.http_mjpeg_decoded_input(opt)
    assert "JpegParse" not in graph.describe()


@pytest.mark.parametrize("bad_name", ["Bad Name", "", "with:colon"])
def test_malformed_capture_names_are_rejected(bad_name):
    opt = pyneat.HttpMjpegDecodedInputOptions()
    opt.url = "http://example.local/mjpeg"
    opt.header_capture.headers = [bad_name]
    with pytest.raises(ValueError):
        pyneat.groups.http_mjpeg_decoded_input(opt)


def test_unsupported_transform_with_capture_is_rejected():
    opt = pyneat.HttpMjpegDecodedInputOptions()
    opt.url = "http://example.local/mjpeg"
    opt.header_capture.headers = ["Image-Index"]
    opt.use_videoconvert = True
    with pytest.raises(ValueError):
        pyneat.groups.http_mjpeg_decoded_input(opt)

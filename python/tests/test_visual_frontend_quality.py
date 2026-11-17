"""Installed public TrackKLT quality option and output-contract surface."""

import math

import pyneat


def test_quality_defaults_preserve_legacy_api():
    options = pyneat.TrackKLTOptions()
    assert options.publish_error is False
    assert options.output_error_name == "output_error"
    assert options.error_threshold == -1.0
    assert options.min_eig_threshold == 0.0
    assert options.require_convergence is False


def test_quality_options_round_trip_and_summary():
    options = pyneat.TrackKLTOptions()
    options.width, options.height, options.num_points = 320, 240, 32
    options.publish_error = True
    options.output_error_name = "photometric_residual"
    options.error_threshold = 12.5
    options.min_eig_threshold = 0.00001
    options.require_convergence = True
    assert math.isclose(options.min_eig_threshold, 0.00001, rel_tol=1e-6)
    for detect in (0, 1):
        options.detect_new_features = detect
        node = pyneat.nodes.track_klt(options)
        assert node is not None
        summary = options.summary()
        assert "error_threshold=12.5" in summary
        assert "require_convergence=1" in summary
        assert "photometric_residual" in summary
        assert "published_error_shape=[1,32,1]" in summary

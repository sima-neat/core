"""ResizeSpec letterbox placement is an additive, typed Model option."""

import pytest

import pyneat


def test_letterbox_placement_defaults_to_center():
    assert pyneat.ResizeSpec().placement == pyneat.LetterboxPlacement.Center
    assert pyneat.ModelOptions().preprocess.resize.placement == pyneat.LetterboxPlacement.Center


def test_top_left_letterbox_option():
    options = pyneat.ModelOptions()
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.resize.pad_value = 116
    options.preprocess.resize.placement = pyneat.LetterboxPlacement.TopLeft
    assert options.preprocess.resize.placement == pyneat.LetterboxPlacement.TopLeft
    assert options.preprocess.resize.mode == pyneat.ResizeMode.Letterbox
    assert options.preprocess.resize.pad_value == 116
    options.preprocess.resize.placement = pyneat.LetterboxPlacement.Center
    assert options.preprocess.resize.placement == pyneat.LetterboxPlacement.Center


def test_letterbox_placement_is_typed():
    spec = pyneat.ResizeSpec()
    with pytest.raises(TypeError):
        spec.placement = "TopLeft"
    assert spec.placement == pyneat.LetterboxPlacement.Center

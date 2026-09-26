import pytest
import pyneat


def test_input_storage_layout_options():
    options = pyneat.ModelOptions()
    assert options.input_storage_layouts == {}
    options.input_storage_layouts = {
        "camera_a": pyneat.InputStorageLayout.HWC,
        "camera_b": pyneat.InputStorageLayout.HWC16,
    }
    assert options.input_storage_layouts == {
        "camera_a": pyneat.InputStorageLayout.HWC,
        "camera_b": pyneat.InputStorageLayout.HWC16,
    }
    with pytest.raises(TypeError):
        options.input_storage_layouts = {"camera_a": "HWC16"}
    assert options.input_storage_layouts["camera_b"] == pyneat.InputStorageLayout.HWC16
    options.input_storage_layouts = {}
    assert options.input_storage_layouts == {}

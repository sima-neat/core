import pyneat


def test_model_options_do_not_expose_compiled_storage_layout():
    assert not hasattr(pyneat, "InputStorageLayout")
    assert not hasattr(pyneat.ModelOptions(), "input_storage_layouts")

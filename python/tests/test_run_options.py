"""Public timeout option keeps legacy behavior unless explicitly configured."""
import pyneat


def test_output_idle_timeout_option():
    options = pyneat.RunOptions()
    assert options.output_idle_timeout_ms == -1
    for timeout in (0, 200, -1):
        options.output_idle_timeout_ms = timeout
        assert options.output_idle_timeout_ms == timeout
        assert options.input_timeout_ms == -1

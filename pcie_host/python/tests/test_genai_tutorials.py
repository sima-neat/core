"""Exercise tutorial requests with real value bindings and a fake model, never a card."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pyneatpcie import genai


TUTORIAL = Path(__file__).resolve().parents[2] / "tutorials/029_run_genai_over_pcie"
if not TUTORIAL.is_dir():
    # Installed Python tests and tutorial sources share the extras prefix.
    TUTORIAL = (Path(__file__).resolve().parents[1]
                / "share/sima-pcie-host/tutorials/029_run_genai_over_pcie")


@pytest.fixture
def run_tutorial(monkeypatch):
    calls = []
    closed = []
    result = SimpleNamespace(text="answer", language="en", no_speech_prob=0.0,
                             metrics=SimpleNamespace(generated_tokens=2))

    class Model:
        def __init__(self, path, connection):
            assert path == "host-model"
            assert connection.card_id == 1
            assert connection.startup_timeout_ms > 0
            assert connection.request_timeout_ms > 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def accepts_text(self):
            return True

        def accepts_image(self):
            return True

        def accepts_audio(self):
            return True

        def task(self):
            return genai.GenAITask.ASR if name == "run_whisper" else genai.GenAITask.VisionLanguage

        def run(self, request):
            calls.append(request)
            return result

        def stream(self, request):
            calls.append(request)
            yield SimpleNamespace(text="answer", is_final=False)
            yield SimpleNamespace(text="", is_final=True, metrics=result.metrics)

    name = ""

    def run(selected, *args):
        nonlocal name
        name = selected
        spec = importlib.util.spec_from_file_location(selected, TUTORIAL / f"{selected}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.genai = SimpleNamespace(
            GenAIModel=Model, ConnectionOptions=genai.ConnectionOptions,
            GenerationRequest=genai.GenerationRequest, GenAITask=genai.GenAITask,
            ASRTask=genai.ASRTask,
        )
        monkeypatch.setattr("sys.argv", [selected, "--model", "host-model", "--card", "1", *args])
        module.main()
        assert closed
        return calls[-1]

    return run


@pytest.mark.parametrize("stream", [False, True])
def test_llm_request_and_stream(run_tutorial, capsys, stream):
    request = run_tutorial("run_llm", "--prompt", "test prompt", "--max-tokens", "4",
                           *(["--stream"] if stream else []))
    assert request.prompt == "test prompt"
    assert request.max_new_tokens == 4
    assert "answer" in capsys.readouterr().out


def test_vlm_sends_rgb_not_bgr(run_tutorial, tmp_path):
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8))
    request = run_tutorial("run_vlm", "--image", str(image_path))
    assert len(request.images) == 1
    tensor = request.images[0]
    assert tensor.shape == [2, 3, 3]
    np.testing.assert_array_equal(tensor.to_numpy(), np.full((2, 3, 3), [30, 20, 10], dtype=np.uint8))


@pytest.mark.parametrize("translate", [False, True])
def test_whisper_host_file_and_task(run_tutorial, tmp_path, translate):
    audio = tmp_path / "speech.wav"
    audio.touch()
    request = run_tutorial("run_whisper", "--audio", str(audio), "--language", "en",
                           *(["--translate"] if translate else []))
    assert Path(request.audio_file) == audio
    assert request.language == "en"
    assert request.prompt is None
    expected = genai.ASRTask.Translate if translate else genai.ASRTask.Transcribe
    assert request.asr_task == expected

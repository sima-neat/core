"""Live H.264 and H.265 coverage for RTSP tutorial 018."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


TUTORIALS_ROOT = Path(__file__).resolve().parents[2] / "tutorials"
TUTORIAL = TUTORIALS_ROOT / "018_consume_rtsp_stream" / "consume_rtsp_stream.py"
FPS_PROBE = TUTORIAL.with_name("probe_rtsp_fps.py")
TIMEOUT_SEC = int(os.environ.get("SIMA_TUTORIAL_TIMEOUT_SEC", "180"))
REQUIRE_RUNTIME = os.environ.get("SIMA_NEAT_TUTORIAL_REQUIRE_RUNTIME") == "1"
PYNEAT_AVAILABLE = importlib.util.find_spec("pyneat") is not None


def load_fps_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tutorial_018_probe_rtsp_fps", FPS_PROBE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rtsp_fps_probe_rounds_opencv_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = load_fps_probe()

    class Capture:
        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            assert prop == probe.cv2.CAP_PROP_FPS
            return 30000 / 1001

        def release(self) -> None:
            pass

    monkeypatch.setattr(probe.cv2, "VideoCapture", lambda url: Capture())

    assert probe.probe_source_fps("rtsp://example/stream") == 30


def test_rtsp_fps_probe_releases_failed_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = load_fps_probe()
    released = False

    class Capture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            nonlocal released
            released = True

    monkeypatch.setattr(probe.cv2, "VideoCapture", lambda url: Capture())

    with pytest.raises(RuntimeError, match="failed to open RTSP source"):
        probe.probe_source_fps("rtsp://example/stream")
    assert released


def first_rtsp_url(codec: str) -> str | None:
    url = os.environ.get(f"SIMANEAT_TEST_RTSP_{codec.upper()}_URL")
    if url:
        return url
    urls = os.environ.get(f"SIMANEAT_TEST_RTSP_{codec.upper()}_URLS", "")
    return re.split(r"[ ,;]+", urls.strip(), maxsplit=1)[0] if urls.strip() else None


def skip_or_fail(reason: str) -> None:
    if REQUIRE_RUNTIME:
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_rtsp_codec_tutorial(codec: str) -> None:
    if not PYNEAT_AVAILABLE:
        skip_or_fail("pyneat is not importable in this Python environment")
    url = first_rtsp_url(codec)
    if not url:
        skip_or_fail(f"set the {codec.upper()} RTSP test URL environment variables")

    result = subprocess.run(
        [
            sys.executable,
            str(TUTORIAL),
            "--url",
            url,
            "--codec",
            codec,
            "--frames",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SEC,
    )
    assert result.returncode == 0, (
        f"tutorial 018 {codec} exited {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )
    assert "rtsp_timeout" not in result.stdout
    assert len(re.findall(r"^frame=\d+ shape=", result.stdout, re.MULTILINE)) == 10

"""Exercise the actual CMake source/test selection for Linux and Darwin."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def platform_block(path, condition):
    text = path.read_text()
    start = text.index(f"if ({condition})")
    return text[start:text.index("endif()", start) + len("endif()")]


class CameraPlatformTest(unittest.TestCase):
    def test_linux_only_implementation_and_syscall_tests(self):
        source_block = platform_block(
            ROOT / "CMakeLists.txt", 'NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"')
        test_block = platform_block(
            ROOT / "tests/CMakeLists.txt", 'CMAKE_SYSTEM_NAME STREQUAL "Linux"')
        for platform, count in (("Linux", 2), ("Darwin", 0)):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                script = Path(directory) / "platform.cmake"
                script.write_text(f"""
set(CMAKE_SYSTEM_NAME "{platform}")
set(CMAKE_CURRENT_SOURCE_DIR "/fixture")
set(SIMANEAT_SOURCES
  "/fixture/src/gst/V4L2CopyCapture.cpp"
  "/fixture/src/gst/NeatV4L2CopySource.cpp")
{source_block}
list(LENGTH SIMANEAT_SOURCES source_count)
if (NOT source_count EQUAL {count})
  message(FATAL_ERROR "V4L2 source selection is wrong for {platform}")
endif()
set(UNIT_TESTS "")
{test_block}
list(LENGTH UNIT_TESTS test_count)
if (NOT test_count EQUAL {count})
  message(FATAL_ERROR "V4L2 syscall test selection is wrong for {platform}")
endif()
""")
                result = subprocess.run(["cmake", "-P", str(script)], text=True,
                                        capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

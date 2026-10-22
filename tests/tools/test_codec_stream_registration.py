"""Prove aggregate-to-case coverage mapping without executing a native binary."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_exact_stream_children_preserve_all_http_and_sync_checks(tmp_path):
    source = (ROOT / 'tests/CMakeLists.txt').read_text()
    start = source.index('  # Keep legacy aggregate invocations')
    end = source.index('  simaneat_add_test(\n    NAME codec_runtime_bad_url_diagnostics', start)
    block = source[start:end]
    cmake = f'''cmake_minimum_required(VERSION 3.16)
project(stream_registration NONE)
enable_testing()
include("{ROOT}/cmake/NeatGuardedTests.cmake")
add_executable(codec_runtime_stream_test IMPORTED)
set_target_properties(codec_runtime_stream_test PROPERTIES IMPORTED_LOCATION /usr/bin/true)
function(append_pipeline_test_env name)
endfunction()
add_test(NAME codec_runtime_stream_test COMMAND /usr/bin/true)
add_test(NAME codec_runtime_stream_contract_test COMMAND /usr/bin/true)
'''
    (tmp_path / 'CMakeLists.txt').write_text(cmake + block)
    build = tmp_path / 'build'
    subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build), '-DSIMANEAT_GUARDED_HARDWARE_TESTS=ON'], check=True, capture_output=True)
    def inventory(*args):
        return json.loads(subprocess.check_output(['ctest', '--test-dir', str(build), '--show-only=json-v1', *args]))['tests']
    all_cases = inventory('-L', 'codec_stream')
    http_cases = inventory('-L', 'codec_stream', '-LE', 'rtsp')
    assert len(all_cases) == 16 and len(http_cases) == 4
    expected_cases = {
        'rtsp-h264-decoded', 'rtsp-h265-encoded-boundary', 'rtsp-h265-decoded',
        'rtsp-mjpeg-encoded-boundary', 'rtsp-mjpeg-decoded', 'rtsp-mjpeg-decoded-videorate',
        'http-mjpeg-decoded', 'http-mjpeg-decoded-videorate',
    }
    mapping = {}
    for test in all_cases:
        cmd = test['command']
        assert 'ctest-proxy' in cmd
        case = cmd[cmd.index('--case') + 1]
        variant = test['name'].rsplit('_', 1)[-1]
        mapping[(case, variant)] = cmd
        assert ('--sync' in cmd) == (case in {'rtsp-h264-decoded', 'http-mjpeg-decoded'})
        assert ('--determinism' in cmd) == (variant == 'contract')
        if variant == 'contract':
            assert cmd[cmd.index('--frames')+1] == '5'
            assert cmd[cmd.index('--repeat')+1] == '2'
        properties = {p['name']: p['value'] for p in test['properties']}
        assert properties['SKIP_RETURN_CODE'] == 77
        assert 'SIMA_TEST_IS_LONG=1' in properties['ENVIRONMENT']
    assert set(mapping) == {(case, variant) for case in expected_cases for variant in ('smoke', 'contract')}
    assert len(inventory('-L', 'codec_aggregate')) == 2
    implementation = (ROOT / 'tests/e2e_testing/codec_runtime_stream_test.cpp').read_text()
    assert 'if (args.sync)' in implementation
    assert 'run_decoded_source_sync(test_case, url, args.frames, args.timeout_ms, source_fps);' in implementation
    workflow = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    assert "vars.NEAT_CI_STREAM_PROFILE || 'http'" in workflow
    assert "E2E_LABEL_EXCLUDE='perf|llima|codec_stream|codec_aggregate'" in workflow
    assert '--label-exclude "perf|e2e"' in workflow  # Keep all 284 original Fast cases.

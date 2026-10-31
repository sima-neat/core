"""Progress stays visible while verbose vendor output goes to a private log."""
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading

import pytest

HERE = Path(__file__).resolve().parents[2]/'pcie_host/scripts/offline'
sys.path.insert(0, str(HERE))
from progress import Progress


class Console(io.StringIO):
    def __init__(self, tty=False):
        super().__init__()
        self.tty = tty
        self.waiting = threading.Event()

    def isatty(self):
        return self.tty

    def write(self, text):
        result = super().write(text)
        if '[WAIT]' in text:
            self.waiting.set()
        return result


@pytest.mark.parametrize('tty,no_color,expected', [(True, False, True), (False, False, False), (True, True, False)])
def test_progress_color_obeys_terminal_and_no_color(monkeypatch, tty, no_color, expected):
    monkeypatch.setenv('TERM', 'xterm')
    if no_color:
        monkeypatch.setenv('NO_COLOR', '1')
    else:
        monkeypatch.delenv('NO_COLOR', raising=False)
    console = Console(tty)
    progress = Progress(console)
    try:
        with progress.step('Checking compatibility'):
            pass
    finally:
        progress.close()
    import re
    assert bool(re.search(r'\x1b\[[0-9;]*m', console.getvalue())) == expected
    assert 'Checking compatibility' in console.getvalue()
    assert '[OK]' in console.getvalue()


def test_command_noise_is_preserved_in_private_log_not_console():
    console = Console()
    progress = Progress(console)
    try:
        with progress.step('Installing Core in SDK'):
            subprocess.run([sys.executable, '-c', "print('VERBOSE VENDOR REPORT')"],
                           stdout=progress.log, stderr=subprocess.STDOUT, check=True)
    finally:
        progress.close()
    assert 'VERBOSE VENDOR REPORT' not in console.getvalue()
    assert 'VERBOSE VENDOR REPORT' in progress.path.read_text()
    assert stat.S_IMODE(progress.path.stat().st_mode) == 0o600


def test_long_running_step_prints_heartbeat_and_stops_before_completion():
    console = Console()
    progress = Progress(console, interval=0.01)
    try:
        with progress.step('Transferring bundle'):
            assert console.waiting.wait(timeout=2)
    finally:
        progress.close()
    lines = console.getvalue().splitlines()
    assert any('[WAIT]' in line and 'still running' in line for line in lines)
    assert lines[-1].startswith('[OK]')


def test_failure_keeps_exit_status_and_full_diagnostics():
    console = Console()
    progress = Progress(console)
    try:
        with pytest.raises(subprocess.CalledProcessError) as error:
            with progress.step('Checking paired DevKit'):
                subprocess.run([sys.executable, '-c', "import sys; print('DETAILED FAILURE', file=sys.stderr); sys.exit(7)"],
                               stdout=progress.log, stderr=subprocess.STDOUT, check=True)
        assert error.value.returncode == 7
    finally:
        progress.close()
    assert '[FAIL]' in console.getvalue()
    assert 'DETAILED FAILURE' in progress.path.read_text()
    assert 'CalledProcessError' in progress.path.read_text()


def test_paired_progress_labels_transfer_check_install_and_cleanup(tmp_path, monkeypatch):
    import paired_devkit
    from types import SimpleNamespace
    monkeypatch.setenv('DEVKIT_SYNC_DEVKIT_IP', '192.0.2.2')
    monkeypatch.delenv('NEAT_INSTALLER_SKIP_DEVKIT_SYNC', raising=False)
    monkeypatch.setattr(paired_devkit.shutil, 'which', lambda name: '/usr/bin/ssh')
    console = Console()
    progress = Progress(console)
    def run(args, **kwargs):
        command = args[-1]
        if 'mktemp' in command:
            return SimpleNamespace(stdout='/tmp/neat-core-offline-ABCDEFGH\n')
        assert kwargs['stdout'] is progress.log
        assert kwargs['stderr'] == subprocess.STDOUT
        progress.log.write('vendor output\n')
    monkeypatch.setattr(paired_devkit, 'run', run)
    try:
        with paired_devkit.paired_devkit(tmp_path, progress) as device:
            device.phase('preflight')
            device.phase('install')
    finally:
        progress.close()
    output = console.getvalue()
    labels = ['Transferring offline bundle', 'Checking compatibility', 'Installing Core on paired DevKit', 'Removing temporary staging']
    assert [output.index(label) for label in labels] == sorted(output.index(label) for label in labels)
    assert 'vendor output' not in output
    assert 'vendor output' in progress.path.read_text()


@pytest.mark.parametrize('fails', [False, True])
def test_cli_reports_one_result_and_log_location(monkeypatch, fails):
    import core_install
    console = Console()
    progress = Progress(console)
    monkeypatch.setattr(core_install, 'Progress', lambda **kwargs: progress)
    monkeypatch.setattr(sys, 'argv', ['install.py', '--check'])
    def install(root, phase, devkit_only, reporter, recreate_venv):
        assert phase == 'preflight'
        assert not devkit_only
        assert not recreate_venv
        assert reporter is progress
        with reporter.step('Checking SDK compatibility'):
            reporter.log.write('internal report\n')
            if fails:
                raise ValueError('incompatible Platform version')
    monkeypatch.setattr(core_install, 'install', install)
    if fails:
        with pytest.raises(SystemExit) as error:
            core_install.main()
        assert error.value.code == 1
        assert 'sudo tail -n 60 '+str(progress.path) in console.getvalue()
        assert 'incompatible Platform version' in console.getvalue()
    else:
        core_install.main()
        assert console.getvalue().count('completed successfully') == 1
    assert progress.log.closed
    # Failed operations print a bounded log tail; successful operations stay quiet.
    assert ('internal report' in console.getvalue()) is fails
    assert 'internal report' in progress.path.read_text()


@pytest.mark.parametrize('address,skip,reason', [('', '', 'No paired DevKit configured'), ('192.0.2.2', 'ON', 'Paired deployment disabled')])
def test_sdk_only_reason_is_visible(tmp_path, monkeypatch, address, skip, reason):
    import paired_devkit
    monkeypatch.setenv('DEVKIT_SYNC_DEVKIT_IP', address)
    monkeypatch.setenv('NEAT_INSTALLER_SKIP_DEVKIT_SYNC', skip)
    console = Console()
    progress = Progress(console)
    try:
        with paired_devkit.paired_devkit(tmp_path, progress) as device:
            assert device is None
    finally:
        progress.close()
    assert reason in console.getvalue()


def test_dumb_terminal_and_untrusted_text_cannot_inject_controls(monkeypatch):
    monkeypatch.setenv('TERM', 'dumb')
    console = Console(True)
    progress = Progress(console)
    try:
        progress.message('FAIL', 'bad\033[31m\rvalue')
    finally:
        progress.close()
    assert '\033' not in console.getvalue()
    assert '\r' not in console.getvalue()


def test_standalone_panel_shows_last_ten_lines_from_current_step(monkeypatch):
    import log_progress
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.setattr(log_progress.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((80, 24)))
    console = Console(True)
    progress = Progress(console)
    try:
        progress.log.write('previous phase\n')
        with progress.step('Installing Core on paired DevKit'):
            progress.log.write(''.join(f'Package {n} installed\n' for n in range(25)))
    finally:
        progress.close()
    output = console.getvalue()
    assert 'previous phase' not in output
    assert 'Package 14 installed' not in output
    assert all(f'Package {n} installed' in output for n in range(15, 25))
    assert 'elapsed | 1. Installing Core on paired DevKit' in output
    assert output.splitlines()[-1].startswith('\033[32m[OK]') or '[OK]' in output.splitlines()[-1]


def test_standalone_panel_updates_before_command_finishes(monkeypatch):
    import log_progress
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.setattr(log_progress.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((80, 24)))
    arrived = threading.Event()
    class LiveConsole(Console):
        def write(self, text):
            result = super().write(text)
            if 'Live package output' in text:
                arrived.set()
            return result
    console = LiveConsole(True)
    progress = Progress(console, interval=0.01)
    threads = set(threading.enumerate())
    try:
        with progress.step('Checking compatibility'):
            progress.log.write('Live package output\n')
            assert arrived.wait(timeout=2)
    finally:
        progress.close()
    assert set(threading.enumerate()) == threads


def test_standalone_panel_clips_to_small_terminal(monkeypatch):
    import log_progress
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.setattr(log_progress.shutil, 'get_terminal_size', lambda fallback: os.terminal_size((22, 8)))
    console = Console(True)
    progress = Progress(console)
    try:
        with progress.step('Installing Core'):
            for n in range(8):
                progress.log.write(f'Row {n} '+('界'*30)+'\n')
    finally:
        progress.close()
    output = console.getvalue()
    assert 'Row 4 ' not in output
    assert all(f'Row {n} ' in output for n in range(5, 8))
    assert '界'*8 not in output


def test_standalone_verbose_streams_log_and_retains_failure():
    console = Console()
    progress = Progress(console, verbose=True)
    text = ''.join(f'detail {n}\n' for n in range(30))+'last partial line'
    try:
        with pytest.raises(ValueError, match='failed'):
            with progress.stream_log():
                with progress.step('Installing Core'):
                    progress.log.write(text)
                    progress.log.flush()
                    raise ValueError('failed')
    finally:
        progress.close()
    assert text in console.getvalue()
    assert '[FAIL]' in console.getvalue()
    assert text in progress.path.read_text()

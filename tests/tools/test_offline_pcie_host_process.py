"""Linux integration tests: host descendants must die before the lock is released."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parents[2]/'pcie_host/scripts/offline'
pytestmark = pytest.mark.skipif(sys.platform != 'linux' or not shutil.which('timeout'),
                                reason='Linux subreaper and GNU timeout integration')


def wait_for(predicate, process, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        assert process.poll() is None, 'supervisor exited prematurely'
        time.sleep(0.02)
    pytest.fail('timed out waiting for process fixture')


@pytest.mark.parametrize('scenario', ['ctrl_c', 'second_ctrl_c', 'timeout', 'failure', 'success'])
def test_host_tree_is_reaped_before_lock_release(tmp_path, scenario):
    child = tmp_path/'child.py'
    child.write_text('''import os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
def term(signum, frame):
    (root/'terminating').touch()
signal.signal(signal.SIGTERM, term)
pid = os.fork()
if pid == 0:
    (root/'grandchild').write_text(str(os.getpid()))
    while True: time.sleep(0.02)
(root/'child').write_text(str(os.getpid()))
while not (root/'grandchild').exists(): time.sleep(0.01)
(root/'ready').touch()
if sys.argv[2] in ('failure', 'success'):
    sys.exit(7 if sys.argv[2] == 'failure' else 0)
while True: time.sleep(0.02)
''')
    supervisor = tmp_path/'supervisor.py'
    supervisor.write_text('''import fcntl, os, signal, subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import install
root = Path(sys.argv[2])
scenario = sys.argv[3]
with (root/'lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        install.run_host(['timeout', '--foreground', '--kill-after=2',
                          '0.5' if scenario == 'timeout' else '20',
                          sys.executable, str(root/'child.py'), str(root), scenario],
                         cleanup_grace=1, stdin=subprocess.DEVNULL)
        assert scenario == 'success'
    except KeyboardInterrupt:
        assert scenario in ('ctrl_c', 'second_ctrl_c')
    except subprocess.CalledProcessError:
        assert scenario in ('timeout', 'failure')
    for name in ('child', 'grandchild'):
        pid = int((root/name).read_text())
        assert not Path('/proc', str(pid)).exists(), name+' still alive or unreaped'
    (root/'checked-under-lock').touch()
(root/'released').touch()
''')
    with (tmp_path/'output').open('w') as output:
        process = subprocess.Popen([sys.executable, str(supervisor), str(HERE), str(tmp_path), scenario],
                                   stdout=output, stderr=output, start_new_session=True)
        try:
            wait_for(lambda: (tmp_path/'ready').exists(), process)
            if scenario in ('ctrl_c', 'second_ctrl_c'):
                os.killpg(process.pid, signal.SIGINT)
            wait_for(lambda: (tmp_path/'terminating').exists(), process)
            with (tmp_path/'lock').open('w') as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if scenario == 'second_ctrl_c':
                os.killpg(process.pid, signal.SIGINT)
            assert process.wait(timeout=10) == 0, (tmp_path/'output').read_text()
            assert (tmp_path/'checked-under-lock').exists()
            assert (tmp_path/'released').exists()
            with (tmp_path/'lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for name in ('child', 'grandchild'):
                if (tmp_path/name).exists():
                    try:
                        os.kill(int((tmp_path/name).read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

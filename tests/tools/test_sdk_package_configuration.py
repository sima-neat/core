"""Run the SDK recovery shell with package/process tools replaced; no APT mutation."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/ci/configure_sdk_packages.sh'
LOCK = 'dpkg: error: dpkg frontend lock was locked by another process with pid 983'


def run_configuration(tmp_path, failures, error=LOCK, diagnostics_fail=False, status=2, later_error=None):
    tools = tmp_path / 'bin'
    tools.mkdir()
    calls = tmp_path / 'calls'
    count = tmp_path / 'count'
    scripts = {
        'dpkg': '''#!/bin/bash
printf 'dpkg %s locale=%s\\n' "$*" "$LC_ALL" >> "$CALLS"
n=0; [[ ! -f "$COUNT" ]] || read -r n < "$COUNT"
echo "$((n+1))" > "$COUNT"
if ((n > 0)) && [[ -n "$LATER_ERROR" ]]; then ERROR=$LATER_ERROR; fi
if ((n < FAILURES)); then printf '%s\\n' "$ERROR" >&2; exit "$FAILURE_STATUS"; fi
echo 'configuration completed'
''',
        'sleep': '#!/bin/bash\necho "sleep $*" >> "$CALLS"\n',
        'ps': '''#!/bin/bash
echo "ps $*" >> "$CALLS"
[[ "$DIAGNOSTICS_FAIL" != 1 ]] || exit 1
if [[ "$*" == '-p 983 -o ppid=' ]]; then echo 42
else echo '983 42 00:01 apt-get install example'; fi
''',
    }
    for name, source in scripts.items():
        path = tools / name
        path.write_text(source)
        path.chmod(0o755)
    result = subprocess.run(
        ['/bin/bash', str(SCRIPT)], text=True, capture_output=True, timeout=10,
        env={**os.environ, 'PATH': f'{tools}:/usr/bin:/bin', 'CALLS': str(calls),
             'COUNT': str(count), 'FAILURES': str(failures), 'ERROR': error,
             'DIAGNOSTICS_FAIL': str(int(diagnostics_fail)), 'LC_ALL': 'invalid-locale',
             'FAILURE_STATUS': str(status), 'LATER_ERROR': later_error or ''})
    return result, calls.read_text().splitlines()


def test_ready_sdk_does_not_wait_or_inspect_processes(tmp_path):
    result, calls = run_configuration(tmp_path, 0)
    assert result.returncode == 0, result.stderr
    assert calls == ['dpkg --configure -a locale=C']
    assert 'configuration completed' in result.stdout


def test_transient_lock_retries_and_reports_owner_and_parent(tmp_path):
    result, calls = run_configuration(tmp_path, 2)
    assert result.returncode == 0, result.stderr
    assert calls.count('dpkg --configure -a locale=C') == 3
    assert calls.count('sleep 5') == 2
    assert calls.count('ps -p 983 -o pid,ppid,etime,args') == 2
    assert calls.count('ps -p 42 -o pid,ppid,etime,args') == 2
    assert 'apt-get install example' in result.stderr
    assert 'attempt 2/25' in result.stderr


def test_persistent_lock_has_finite_budget_and_final_diagnostics(tmp_path):
    result, calls = run_configuration(tmp_path, 99)
    assert result.returncode == 2
    assert calls.count('dpkg --configure -a locale=C') == 25
    assert calls.count('sleep 5') == 24
    assert calls.count('ps -p 983 -o pid,ppid,etime,args') == 25
    assert 'retry budget exhausted' in result.stderr


@pytest.mark.parametrize('error', [
    'dpkg: error: requested operation requires superuser privilege',
    'dpkg: error processing package example (--configure): dependency problems',
    LOCK + '\ndpkg: error: unable to access database',
    'dpkg: error: unable to open lock file: Permission denied',
])
def test_unrelated_errors_fail_immediately(tmp_path, error):
    result, calls = run_configuration(tmp_path, 99, error)
    assert result.returncode == 2
    assert calls == ['dpkg --configure -a locale=C']
    assert error in result.stderr


def test_disappearing_owner_or_missing_ps_does_not_mask_failure(tmp_path):
    result, calls = run_configuration(tmp_path, 99, diagnostics_fail=True)
    assert result.returncode == 2
    assert calls.count('sleep 5') == 24
    assert 'retry budget exhausted' in result.stderr


def test_older_lock_message_without_pid_uses_process_snapshot(tmp_path):
    result, calls = run_configuration(
        tmp_path, 1, 'dpkg: error: dpkg database area is locked by another process')
    assert result.returncode == 0, result.stderr
    assert 'ps -eo pid,ppid,etime,args' in calls
    assert 'No owner PID' in result.stderr


def test_non_dpkg_exit_status_is_preserved_without_retry(tmp_path):
    result, calls = run_configuration(tmp_path, 99, status=37)
    assert result.returncode == 37
    assert calls == ['dpkg --configure -a locale=C']


def test_unrelated_error_after_contention_stops_retries(tmp_path):
    error = 'dpkg: error: unable to access database'
    result, calls = run_configuration(tmp_path, 99, later_error=error)
    assert result.returncode == 2
    assert calls.count('dpkg --configure -a locale=C') == 2
    assert calls.count('sleep 5') == 1
    assert error in result.stderr

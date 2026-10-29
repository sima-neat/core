#!/usr/bin/env python3
"""Persistent board-side hardware child ownership; deadlines never kill owners.

The configured checker is a root-reviewed, resource-free admission/retirement
probe. This module deliberately cannot infer DMA retirement from process exit,
empty sampled queues, elapsed time, or a successful SSH command.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import select
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid

SCHEMA = 1
IDENTITY_FIELDS = ('admission_id', 'boot_id', 'cohort_sha256', 'lease_pid',
                   'lease_start_ticks', 'observer_id')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    return {'pid': pid, 'start_ticks': int(text[text.rfind(')') + 2:].split()[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def trusted_file(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise ValueError(f'Untrusted control file: {path}')
    return path


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.module = trusted_file(Path(__file__).resolve())
        if sha(self.module) != config['supervisor_sha256']:
            raise ValueError('Supervisor source identity mismatch')
        self.identity = config['identity']
        if set(self.identity) != set(IDENTITY_FIELDS):
            raise ValueError('Incomplete pinned admission identity')
        if self.identity['boot_id'] != process_identity(os.getpid())['boot_id']:
            raise ValueError('Admission belongs to another boot')
        self.root = Path(config['state_dir'])
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError('State directory must be private and owned by supervisor UID')
        self.lock = (self.root / 'exclusive-launch.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.root / 'fail-stop.json').exists() or (self.root / 'admission.json').exists():
            raise ValueError('Admission directories are single-use; never reopen old state')
        atomic_json(self.root / 'admission.json', self.identity)
        self.failed = None
        self.owner = None
        self.retained = None
        self.child = None
        self.checker = trusted_file(config['checker']['command'][0])
        if sha(self.checker) != config['checker']['sha256']:
            raise ValueError('Checker identity mismatch')

    def fail(self, reason):
        if self.failed is None:
            self.failed = {'schema': SCHEMA, 'reason': reason,
                           'identity': self.identity, 'owner': self.owner,
                           'retirement_unknown': self.child is not None,
                           'epoch': time.time()}
            atomic_json(self.root / 'fail-stop.json', self.failed)
        self.retained = self.child

    def check(self, phase, owner_file):
        # Only the approved NON-hardware read-only checker may be killed by this
        # bounded probe. Hardware owners never enter subprocess.run().
        if sha(trusted_file(self.checker)) != self.config['checker']['sha256']:
            raise ValueError('Checker changed during admission')
        command = self.config['checker']['command'] + ['--phase', phase, '--owner-json', str(owner_file)]
        probe = subprocess.run(command, capture_output=True, text=True, check=False,
                               timeout=self.config['checker'].get('timeout_sec', 2))
        if probe.returncode:
            raise ValueError('Admission checker failed: ' + probe.stderr[-500:])
        receipt = json.loads(probe.stdout)
        if receipt.get('schema') != SCHEMA or receipt.get('status') != 'ready':
            raise ValueError('Admission checker did not approve current phase')
        if receipt.get('identity') != self.identity:
            raise ValueError('Lease/observer/boot/cohort identity changed')
        age = time.time() - receipt['checked_at_epoch']
        if not 0 <= age <= self.config.get('max_checker_age_sec', 3):
            raise ValueError('Stale admission/observer evidence')
        if phase == 'retired':
            proof = receipt.get('retirement', {})
            if (proof.get('proven') is not True or not proof.get('token') or
                proof.get('kind') not in ('authoritative-driver-retirement', 'reviewed-graceful-exit') or
                proof.get('owner') != self.owner or
                proof.get('scope') != 'whole-owner-process-tree-and-dma'):
                raise ValueError('Authoritative owner-bound retirement is unproven')
        atomic_json(owner_file.with_name(phase + '-checker.json'), receipt)
        return receipt

    def approved_command(self, request):
        command = request['command']
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise ValueError('Command must be an explicit argv list')
        executable = Path(command[0])
        if not executable.is_absolute() or executable.is_symlink():
            raise ValueError('Executable must be an absolute regular file')
        approved = self.config['executables'].get(str(executable))
        if not approved or sha(trusted_file(executable)) != approved['sha256']:
            raise ValueError('Executable absent from immutable approved cohort')
        variants = approved.get('variants', [approved])
        matches = [item for item in variants if item.get('arguments') == command[1:]]
        if len(matches) != 1:
            raise ValueError('Exact argument vector absent/ambiguous in approved full-test selection')
        approved = {**approved, **matches[0]}
        if not Path(request['cwd']).is_absolute() or not Path(request['cwd']).is_dir():
            raise ValueError('Explicit existing absolute cwd required')
        if not isinstance(request['env'], dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                         for k, v in request['env'].items()):
            raise ValueError('Explicit string environment required')
        if request['deadline_sec'] == 'approved':
            request['deadline_sec'] = approved['deadline_sec']
        deadline = request['deadline_sec']
        if deadline != approved['deadline_sec']:
            raise ValueError('Deadline differs from approved unchanged test deadline')
        if deadline is not None and (not isinstance(deadline, (int, float)) or deadline <= 0):
            raise ValueError('Invalid test deadline')
        for key in ('LD_PRELOAD', 'LD_AUDIT'):
            if request['env'].get(key):
                raise ValueError('Dynamic loader injection is forbidden')
        for key, value in approved.get('required_env', {}).items():
            if request['env'].get(key) != value:
                raise ValueError('Noncanonical protected environment: ' + key)
        if request.get('graceful_stop') and approved.get('graceful_stop') != request['graceful_stop']:
            raise ValueError('Graceful stop requires exact root-reviewed handler policy')
        if request.get('completion'):
            if not approved.get('graceful_stop') or request['completion'] != approved.get('completion'):
                raise ValueError('Output completion requires an exact reviewed stop/completion variant')
            policy = approved['graceful_stop']
            if policy.get('signal') != 'SIGTERM' or not re.fullmatch(r'[0-9a-f]{64}', policy.get('review_receipt_sha256', '')):
                raise ValueError('Graceful handler/drain review receipt must be pinned by SHA256')
        return approved

    def release_gate(self, gate):
        # Serialize signal-handler fail-stop transitions with the final admission
        # commit. Signals already pending are refused, not delayed into execution.
        blocked = {signal.SIGTERM, signal.SIGINT}
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            if self.failed or signal.sigpending() & blocked:
                raise ValueError('Fail-stop/pending cancellation at atomic exec barrier')
            os.write(gate, b'1')
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def run(self, request, connection):
        if self.failed:
            return {'schema': SCHEMA, 'status': 'refused', 'fail_stop': self.failed}
        run_id = uuid.uuid4().hex
        directory = self.root / run_id
        directory.mkdir(mode=0o700)
        owner_file = directory / 'owner.json'
        atomic_json(owner_file, None)
        result = {'schema': SCHEMA, 'status': 'fail_stop', 'run_id': run_id,
                  'stdout_path': str(directory / 'stdout.log'), 'stderr_path': str(directory / 'stderr.log'),
                  'retirement_proven': False, 'completion_condition_met': False}
        output = error = None
        read_gate = write_gate = None
        try:
            approved = self.approved_command(request)
            # GitHub runner process cleanup uses this cookie even for an
            # independently supervised process. It must not own hardware children.
            if 'RUNNER_TRACKING_ID' in request['env']:
                request['env'] = dict(request['env'])
                del request['env']['RUNNER_TRACKING_ID']
                request['safety_env_removed'] = ['RUNNER_TRACKING_ID']
            self.check('launch', owner_file)
            if sha(trusted_file(self.module)) != self.config['supervisor_sha256']:
                raise ValueError('Supervisor source changed before child barrier')
            if self.failed:
                raise ValueError('Fail-stop sealed before child launch')
            output = open(result['stdout_path'], 'xb', buffering=0)
            error = open(result['stderr_path'], 'xb', buffering=0)
            launch_file = directory / 'request.json'
            atomic_json(launch_file, request)
            read_gate, write_gate = os.pipe()
            launch_environment = dict(os.environ)
            launch_environment.pop('RUNNER_TRACKING_ID', None)
            self.child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_launch',
                                           str(read_gate), str(launch_file)],
                                          pass_fds=(read_gate,), stdout=output, stderr=error,
                                          start_new_session=True, env=launch_environment)
            os.close(read_gate)
            read_gate = None
            request_digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.owner = {**process_identity(self.child.pid), 'run_id': run_id, 'request_sha256': request_digest, 'session_id': self.child.pid}
            atomic_json(owner_file, self.owner)
            # The launched child is still a resource-free Python gate. Recheck
            # after ownership is durable, before allowing accelerator exec.
            self.check('launch', owner_file)
            if self.failed:
                raise ValueError('Fail-stop sealed while child waited at exec barrier')
            ready, _, _ = select.select([connection], [], [], 0)
            if ready and not connection.recv(4096):
                raise ValueError('Client disconnected before exec admission')
            # Reverify approved executable after the second admission check.
            self.approved_command(request)
            self.release_gate(write_gate)
            os.close(write_gate)
            write_gate = None
            started = time.monotonic()
            result.update(owner=self.owner, deadline_sec=request['deadline_sec'])
            graceful_requested = False
            last_heartbeat = started
            while True:
                if self.failed:
                    raise ValueError('Fail-stop already sealed')
                if process_identity(self.child.pid) != {k: self.owner[k] for k in ('pid', 'start_ticks', 'boot_id')}:
                    raise ValueError('Owner PID/start identity changed')
                ready, _, _ = select.select([connection], [], [], 0)
                if ready:
                    heartbeat = connection.recv(4096)
                    if not heartbeat:
                        raise ValueError('Reporting client disconnected; ownership retained')
                    last_heartbeat = time.monotonic()
                if time.monotonic() - last_heartbeat > self.config.get('client_heartbeat_max_age_sec', 3):
                    raise ValueError('Reporting client heartbeat expired; ownership retained')
                elapsed = time.monotonic() - started
                if request['deadline_sec'] is not None and elapsed >= request['deadline_sec']:
                    result['status'] = 'timeout'
                    self.fail('Original test deadline expired; retirement unknown')
                    break
                session_members = []
                for procdir in Path('/proc').glob('[0-9]*'):
                    try:
                        fields = (procdir / 'stat').read_text()
                        tail = fields[fields.rfind(')') + 2:].split()
                        if int(tail[3]) == self.owner['session_id']:
                            session_members.append(process_identity(int(procdir.name)))
                    except (OSError, ValueError, IndexError):
                        continue
                atomic_json(directory / 'owner-session-members.json', session_members)
                for path in (Path(result['stdout_path']), Path(result['stderr_path'])):
                    text = path.read_text(errors='replace')
                    patterns = self.config.get('fault_patterns', [r'UNRETIRED', r'DMA_RETIREMENT_UNKNOWN',
                                                                 r'quarantined until platform recovery', r'HARDWARE_FAIL_STOP'])
                    if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                        raise ValueError('Owner reported unretired/quarantined hardware state')
                self.check('active', owner_file)
                if request['deadline_sec'] is not None and time.monotonic() - started >= request['deadline_sec']:
                    result['status'] = 'timeout'
                    self.fail('Original test deadline expired during admission probe')
                    break
                code = self.child.poll()
                if code is not None:
                    if code < 0:
                        raise ValueError('Owner died by signal; retirement unknown')
                    atomic_json(directory / 'exit.json', {
                        'owner': self.owner, 'returncode': code, 'deadline_sec': request['deadline_sec'],
                        'finished_monotonic': time.monotonic(), 'elapsed_sec': time.monotonic() - started,
                        'request_sha256': self.owner['request_sha256']})
                    self.check('retired', owner_file)
                    result.update(status='completed', returncode=code, retirement_proven=True)
                    self.child = None
                    break
                completion = request.get('completion')
                if completion and not graceful_requested:
                    root = Path(completion['output_dir'])
                    count = sum(p.is_file() and not p.is_symlink() and p.name != 'config.yaml' and p.stat().st_size > 0
                                for p in root.rglob('*')) if root.is_dir() else 0
                    if count >= completion['expected_files']:
                        result['completion_condition_met'] = True
                        policy = request.get('graceful_stop') or approved['graceful_stop']
                        if policy.get('signal') != 'SIGTERM' or not policy.get('review_receipt_sha256'):
                            raise ValueError('Unsupported graceful-stop policy')
                        # This is an EXPLICIT reviewed application stop request,
                        # never a deadline/fault/disconnect cleanup action.
                        if process_identity(self.child.pid) != {k: self.owner[k] for k in ('pid', 'start_ticks', 'boot_id')}:
                            raise ValueError('Owner identity changed before approved graceful stop')
                        os.kill(self.child.pid, signal.SIGTERM)
                        graceful_requested = True
                        result['graceful_stop_requested'] = True
                time.sleep(self.config.get('poll_interval_sec', 0.1))
        except BaseException as exc:
            self.fail(type(exc).__name__ + ': ' + str(exc))
            result['error'] = str(exc)
        finally:
            # Closing our direct log descriptors cannot SIGPIPE the owner. Never
            # signal/wait/reset the hardware child in exception cleanup.
            for fd in (read_gate, write_gate):
                if fd is not None:
                    os.close(fd)
            for stream in (output, error):
                if stream is not None:
                    stream.close()
            if self.failed:
                result['fail_stop'] = self.failed
            for key in ('stdout', 'stderr'):
                path = Path(result[key + '_path'])
                result[key] = path.read_text(errors='replace') if path.exists() else ''
            atomic_json(directory / 'result.json', result)
            atomic_json(self.root / 'latest-result.json', result)
        return result


def serve(config_path):
    if os.geteuid() != 0:
        raise ValueError('Board hardware supervisor requires root')
    config = json.loads(trusted_file(config_path).read_text())
    supervisor = Supervisor(config)
    path = Path(config['socket'])
    if path.parent != supervisor.root or path.exists():
        raise ValueError('Socket must be a new path inside private single-use state root')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o600)
    server.listen(8)
    signal.signal(signal.SIGTERM, lambda *_: supervisor.fail('Supervisor received SIGTERM; owners retained'))
    signal.signal(signal.SIGINT, lambda *_: supervisor.fail('Supervisor received SIGINT; owners retained'))
    atomic_json(supervisor.root / 'supervisor.json', process_identity(os.getpid()))
    while True:
        connection, _ = server.accept()
        try:
            connection.settimeout(config.get('client_request_timeout_sec', 3))
            data = bytearray()
            while b'\n' not in data:
                part = connection.recv(4096)
                if not part or len(data) > 1024 * 1024:
                    raise ValueError('Incomplete or oversized client request')
                data.extend(part)
            request = json.loads(data.split(b'\n', 1)[0])
            connection.setblocking(True)
            if request.get('action') == 'shutdown':
                # No forced shutdown of a live/uncertain owner. After fail-stop,
                # root must recover first; old admission is never reopened.
                if supervisor.retained is not None:
                    code = supervisor.retained.poll()
                    if code is None:
                        raise ValueError('Cannot shut down retained live owner')
                    if code < 0:
                        raise ValueError('Retained owner died by signal; retirement remains unknown')
                    directory = supervisor.root / supervisor.owner['run_id']
                    request = json.loads((directory / 'request.json').read_text())
                    atomic_json(directory / 'exit.json', {
                        'owner': supervisor.owner, 'returncode': code,
                        'deadline_sec': request['deadline_sec'], 'finished_monotonic': time.monotonic(),
                        'request_sha256': supervisor.owner['request_sha256'],
                        'post_fail_stop_observation': True})
                    owner_file = supervisor.root / supervisor.owner['run_id'] / 'owner.json'
                    supervisor.check('retired', owner_file)
                connection.sendall(b'{"status":"shutdown"}\n')
                break
            response = supervisor.run(request, connection)
            connection.sendall(json.dumps(response).encode() + b'\n')
        except (BrokenPipeError, ConnectionResetError):
            supervisor.fail('Reporting connection disappeared; owner retained')
        except BaseException as exc:
            supervisor.fail('Control connection failed: ' + str(exc))
            try:
                connection.sendall(json.dumps({'schema': SCHEMA, 'status': 'refused', 'error': str(exc)}).encode() + b'\n')
            except OSError:
                pass
        finally:
            connection.close()
    server.close()
    path.unlink()
    supervisor.lock.close()


def run_owned(command, cwd, env, deadline_sec, evidence_dir=None, socket_path=None,
              completion=None, graceful_stop=None):
    """Client death/timeout NEVER owns or kills the hardware process."""
    endpoint = socket_path or os.environ.get('SIMA_HARDWARE_SUPERVISOR_SOCKET')
    if not endpoint:
        raise ValueError('Explicit guarded hardware supervisor socket required')
    request = {'command': command, 'cwd': str(cwd), 'env': env, 'deadline_sec': deadline_sec}
    if completion is not None:
        request['completion'] = completion
    if graceful_stop is not None:
        request['graceful_stop'] = graceful_stop
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(endpoint)
    client.sendall(json.dumps(request).encode() + b'\n')
    pending = bytearray()
    try:
        while b'\n' not in pending:
            ready, _, _ = select.select([client], [], [], 0.2)
            if ready:
                part = client.recv(65536)
                if not part:
                    raise RuntimeError('Supervisor disconnected; ownership uncertain, stop scheduling')
                pending.extend(part)
            else:
                client.sendall(b'H')
        result = json.loads(pending.split(b'\n', 1)[0])
        if evidence_dir:
            directory = Path(evidence_dir)
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / (result.get('run_id', uuid.uuid4().hex) + '.json'), result)
        return result
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    daemon = sub.add_parser('serve'); daemon.add_argument('--config', required=True)
    run = sub.add_parser('run-json'); run.add_argument('--request', required=True); run.add_argument('--result', required=True)
    run.add_argument('--socket')
    proxy = sub.add_parser('ctest-proxy'); proxy.add_argument('--test-name', required=True)
    proxy.add_argument('command', nargs=argparse.REMAINDER)
    launch = sub.add_parser('_launch'); launch.add_argument('fd', type=int); launch.add_argument('request')
    args = parser.parse_args()
    if args.action == '_launch':
        token = os.read(args.fd, 1); os.close(args.fd)
        if token != b'1':
            return 125
        request = json.loads(Path(args.request).read_text())
        os.chdir(request['cwd'])
        os.execvpe(request['command'][0], request['command'], request['env'])
    if args.action == 'serve':
        serve(args.config)
        return 0
    if args.action == 'ctest-proxy':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        result = run_owned(command=command, cwd=os.getcwd(), env=dict(os.environ), deadline_sec='approved')
        sys.stdout.write(result.get('stdout', ''))
        sys.stderr.write(result.get('stderr', ''))
        if result['status'] != 'completed':
            print('Hardware supervisor fail-stop; owner retained: ' + json.dumps(result), file=sys.stderr)
        return result.get('returncode', 124) if result['status'] == 'completed' else 124
    request = json.loads(Path(args.request).read_text())
    result = run_owned(**request, socket_path=args.socket)
    atomic_json(args.result, result)
    return result.get('returncode', 124) if result['status'] == 'completed' else 124


if __name__ == '__main__':
    sys.exit(main())

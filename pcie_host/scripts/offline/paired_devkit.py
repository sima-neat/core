"""Deploy the standalone offline bundle to the SDK's configured paired DevKit."""
from contextlib import contextmanager, nullcontext
import ipaddress
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile


def run(args, **kwargs):
    return subprocess.run(args, check=True, timeout=1800, **kwargs)


class PairedDevKit:
    def __init__(self, address, user, progress=None):
        self.progress = progress
        self.recreate_venv = False
        self.target = user+'@'+str(ipaddress.ip_address(address))
        if not re.fullmatch(r'[a-z_][a-z0-9_-]*', user):
            raise ValueError('Invalid DEVKIT_DEPLOY_USER')
        ssh = shutil.which('ssh') or shutil.which('sima-ssh')
        if not ssh:
            raise ValueError('Paired DevKit deployment requires ssh or sima-ssh')
        self.command = [ssh, '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.target]
        # sudo should not change which SDK user's provisioned SSH identity is used.
        uid = int(os.environ.get('SUDO_UID', os.getuid()))
        if uid != os.getuid():
            self.command = ['runuser', '-u', pwd.getpwuid(uid).pw_name, '--', *self.command]
        self.directory = None
        self.uncertain = False

    def ssh(self, command, **kwargs):
        try:
            return run([*self.command, command], **kwargs)
        except subprocess.CalledProcessError as exc:
            if self.progress and exc.stderr:
                self.progress.log.write(exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else exc.stderr)
            raise

    def phase(self, phase):
        option = {'preflight': ['--check'], 'install': [], 'verify': ['--verify']}[phase]
        command = ['sudo', '-S', '-p', '', 'python3', '-B', self.directory+'/neat-core-offline/install.py', '--devkit-only', *option]
        if self.recreate_venv:
            command.append('--recreate-venv')
        password = os.environ.get('SUDO_PASSWORD') or os.environ.get('DEVKIT_PASSWORD') or 'edgeai'
        # Never put the sudo password in argv, artifacts, or diagnostic messages.
        label = {'preflight': 'Checking compatibility and offline dependencies on paired DevKit',
                 'install': 'Installing Core on paired DevKit and verifying installation',
                 'verify': 'Verifying paired DevKit installation'}[phase]+' ('+self.target+')'
        with self.progress.step(label) if self.progress else nullcontext():
            self.uncertain = phase == 'install'
            try:
                logs = {'stdout': self.progress.log, 'stderr': subprocess.STDOUT} if self.progress else {}
                self.ssh(shlex.join(command), input=password+'\n', text=True, **logs)
            except subprocess.CalledProcessError as exc:
                # A reported remote failure has finished; SSH status 255 can mean
                # the connection disappeared while the installer is still running.
                if exc.returncode != 255:
                    self.uncertain = False
                raise
            else:
                self.uncertain = False


@contextmanager
def paired_devkit(root, progress=None):
    address = os.environ.get('DEVKIT_SYNC_DEVKIT_IP', '')
    if not address or os.environ.get('NEAT_INSTALLER_SKIP_DEVKIT_SYNC') == 'ON':
        if progress:
            reason = 'Paired deployment disabled by NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON' if address else 'No paired DevKit configured'
            progress.message('INFO', reason+'; operating on SDK sysroot only')
        yield None
        return
    device = PairedDevKit(address, os.environ.get('DEVKIT_DEPLOY_USER', 'sima'), progress)
    try:
        with progress.step('Transferring offline bundle to paired DevKit ('+device.target+')') if progress else nullcontext():
            result = device.ssh('umask 077; mktemp -d /tmp/neat-core-offline-XXXXXXXX',
                                capture_output=True, text=True)
            device.directory = result.stdout.strip()
            if not re.fullmatch(r'/tmp/neat-core-offline-[A-Za-z0-9]{8}', device.directory):
                device.directory = None
                raise ValueError('Unexpected paired DevKit staging directory')
            with tempfile.TemporaryFile() as stream:
                with tarfile.open(fileobj=stream, mode='w') as archive:
                    archive.add(root, arcname='neat-core-offline')
                stream.seek(0)
                device.ssh('tar --no-same-owner -xf - -C '+shlex.quote(device.directory), stdin=stream,
                           **({'stdout': progress.log, 'stderr': subprocess.STDOUT} if progress else {}))
        yield device
    finally:
        if device.directory:
            if device.uncertain:
                message = 'Paired DevKit installation outcome unknown; inspect '+device.target+':'+device.directory
                if progress:
                    progress.message('WARN', message)
                else:
                    print(message)
            else:
                with progress.step('Removing temporary staging from paired DevKit ('+device.target+')') if progress else nullcontext():
                    device.ssh('rm -rf -- '+shlex.quote(device.directory),
                               **({'stdout': progress.log, 'stderr': subprocess.STDOUT} if progress else {}))

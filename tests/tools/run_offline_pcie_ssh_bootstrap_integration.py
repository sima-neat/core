#!/usr/bin/env python3
"""Real OpenSSH bootstrap/retry/changed-key regression in a disposable Linux container.

This uses loopback instead of PCI hardware. It exercises the same shared
pcie-setup.sh as online installation with no existing credentials or host trust.
It does not install Neat packages or prove application inference acceptance.
See pcie_host/scripts/offline/ASSEMBLY.md for the network-isolated invocation.
"""
import os
import pathlib
import pty
import pwd
import select
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pcie_host/scripts/offline"))
import install as installer

if len(sys.argv) > 1:
    account = pwd.getpwnam("hostowner")
    home = pathlib.Path(account.pw_dir)
    ops = installer.Operations(
        ROOT / "pcie_host/scripts",
        "sima",
        home / ".ssh/sima_neat_pcie_ed25519",
        home / ".ssh/known_hosts",
    )
    ops.topology = lambda: ["test"]
    ops.route = lambda card: None
    installer.prepare_ssh(
        [
            {
                "bdf": "test-pcie",
                "ip": "127.0.0.1",
                "local_ip": "127.0.0.1",
                "interface": "lo",
            }
        ],
        ["test"],
        ops,
        account,
        installer.Progress(),
        create_key=True,
    )
    sys.exit(0)

if not pathlib.Path("/.dockerenv").exists() or os.geteuid() != 0:
    raise RuntimeError("Run only as root inside a fresh disposable test container")

for name in ["hostowner", "sima"]:
    if not subprocess.run(
        ["id", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode:
        raise RuntimeError(
            "This test requires a fresh disposable container without " + name
        )
for name in ["hostowner", "sima"]:
    subprocess.run(["useradd", "-m", "-s", "/bin/bash", name], check=True)
subprocess.run(
    ["chpasswd"], input="sima:temporary-smoke-password\n", text=True, check=True
)
pathlib.Path("/run/sshd").mkdir(exist_ok=True)
hostkey = pathlib.Path("/tmp/test-ssh-host-key")
config = pathlib.Path("/tmp/test-sshd.conf")
config.write_text(
    "Port 22\nListenAddress 127.0.0.1\nHostKey "
    + str(hostkey)
    + "\nPasswordAuthentication yes\nUsePAM no\nPermitRootLogin no\nPidFile /tmp/test-sshd.pid\n"
)


def server():
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(hostkey)], check=True
    )
    proc = subprocess.Popen(
        ["/usr/sbin/sshd", "-D", "-e", "-f", str(config)], stderr=subprocess.DEVNULL
    )
    time.sleep(0.3)
    assert proc.poll() is None
    return proc


def attempt(expect):
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-B", __file__, "child"])
    text = ""
    answered = 0
    passwords = 0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if select.select([fd], [], [], 0.2)[0]:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            text += chunk.decode(errors="replace")
            if "Are you sure you want to continue connecting" in text and not answered:
                os.write(fd, b"yes\n")
                answered += 1
            if text.count("password:") > passwords:
                os.write(fd, b"temporary-smoke-password\n")
                passwords += 1
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
    else:
        os.kill(pid, 9)
        raise AssertionError("SSH bootstrap timed out")
    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    os.close(fd)
    code = os.waitstatus_to_exitcode(status)
    assert (code == 0) == expect, text
    return text, answered, passwords


hostkey.unlink(missing_ok=True)
hostkey.with_suffix(".pub").unlink(missing_ok=True)
sshd = server()
try:
    text, trust, passwords = attempt(True)
    assert trust == 1 and passwords == 1, text
    account = pwd.getpwnam("hostowner")
    root = pathlib.Path(account.pw_dir) / ".ssh"
    for filename in [
        "sima_neat_pcie_ed25519",
        "sima_neat_pcie_ed25519.pub",
        "known_hosts",
    ]:
        assert (root / filename).stat().st_uid == account.pw_uid
    assert (root / "sima_neat_pcie_ed25519").stat().st_mode & 0o777 == 0o600
    known = root / "known_hosts"
    authorized = pathlib.Path("/home/sima/.ssh/authorized_keys")
    before = (
        known.read_bytes(),
        authorized.read_bytes(),
        known.stat().st_mtime_ns,
        authorized.stat().st_mtime_ns,
    )
    text, trust, passwords = attempt(True)
    assert trust == 0 and passwords == 0, text
    assert before == (
        known.read_bytes(),
        authorized.read_bytes(),
        known.stat().st_mtime_ns,
        authorized.stat().st_mtime_ns,
    )
    sshd.terminate()
    sshd.wait(timeout=5)
    hostkey.unlink()
    hostkey.with_suffix(".pub").unlink()
    sshd = server()
    text, trust, passwords = attempt(False)
    assert "REMOTE HOST IDENTIFICATION HAS CHANGED" in text
    assert trust == 0 and passwords == 0
    assert before == (
        known.read_bytes(),
        authorized.read_bytes(),
        known.stat().st_mtime_ns,
        authorized.stat().st_mtime_ns,
    )
    print(
        "PASS: first-run trust/password setup; user-owned credentials; prompt-free retry; changed host key rejected without mutation"
    )
finally:
    sshd.terminate()
    sshd.wait(timeout=5)

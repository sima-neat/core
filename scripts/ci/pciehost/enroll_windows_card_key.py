#!/usr/bin/env python3
"""Enroll the card lane's SSH public key with the card password, for Windows.

The Linux card lane's pcie-setup.sh does this with sshpass and ssh-copy-id.
Windows OpenSSH has no non-interactive password input, so this helper uses
paramiko from the card wheelhouse. It reads the password from
SIMAPCIE_SSH_PASSWORD and requires the host key that pcie-setup.ps1 already
recorded, then appends the key with the remote script pcie-setup.ps1 uses.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import paramiko

REMOTE_SCRIPT = """umask 077
mkdir -p ~/.ssh && cd ~/.ssh && touch authorized_keys || exit 1
head -n1 | tr -d '\\r' > .neat-pcie-key || exit 1
if grep -Eqx '(ssh-(ed25519|rsa)|ecdsa-sha2-nistp(256|384|521)|sk-(ssh-ed25519|ecdsa-sha2-nistp256)@openssh[.]com) [A-Za-z0-9+/]+={0,3}( .*)?' .neat-pcie-key; then
  grep -qxFf .neat-pcie-key authorized_keys || cat .neat-pcie-key >> authorized_keys
  status=$?
else
  echo 'The enrolment input is not one OpenSSH public key line.' >&2
  status=1
fi
rm -f .neat-pcie-key
chmod 700 . && chmod 600 authorized_keys || status=1
exit $status
"""


def enroll(host: str, user: str, public_key: Path, known_hosts: Path, password: str) -> None:
  key = public_key.read_text(encoding="utf-8").strip()
  client = paramiko.SSHClient()
  client.load_host_keys(str(known_hosts))
  client.set_missing_host_key_policy(paramiko.RejectPolicy())
  try:
    client.connect(host, username=user, password=password, timeout=10,
                   allow_agent=False, look_for_keys=False)
    stdin, stdout, stderr = client.exec_command(REMOTE_SCRIPT, timeout=30)
    stdin.write(key + "\n")
    stdin.channel.shutdown_write()
    status = stdout.channel.recv_exit_status()
    if status != 0:
      raise RuntimeError(f"key enrollment failed on {user}@{host}: "
                         + stderr.read().decode("utf-8", errors="replace").strip())
  finally:
    client.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", required=True)
  parser.add_argument("--user", required=True)
  parser.add_argument("--key", type=Path, required=True)
  args = parser.parse_args()
  password = os.environ.get("SIMAPCIE_SSH_PASSWORD", "")
  if not password:
    raise SystemExit("SIMAPCIE_SSH_PASSWORD is required to enroll the card key")
  enroll(args.host, args.user, Path(f"{args.key}.pub"), args.key.parent / "known_hosts",
         password)
  print(f"Enrolled {args.key}.pub on {args.user}@{args.host}.")


if __name__ == "__main__":
  main()

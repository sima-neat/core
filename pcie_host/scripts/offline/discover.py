#!/usr/bin/env python3
"""Collect PCIe discovery evidence; never authorize or perform installation."""
import argparse
import ipaddress
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

GROUP = '239.255.42.1'
PORT = 50000
SOURCE_PORT = 60000
VENDOR = '0x1f06'


def inventory(root=Path('/sys/bus/pci/devices')):
    """Retain every SiMa PCI function, including functions without netdevs.

    BDF identifies an attachment in this snapshot, not a persistent card serial.
    Device IDs are evidence only until the supported firmware matrix is qualified.
    """
    cards = []
    for device in sorted(root.iterdir()):
        if (device / 'vendor').read_text().strip().lower() != VENDOR:
            continue
        net = device / 'net'
        cards.append({
            'bdf': device.name,
            'device_id': (device / 'device').read_text().strip().lower(),
            'interfaces': sorted(p.name for p in net.iterdir()) if net.exists() else [],
        })
    return cards


def interface_ipv4(name):
    result = subprocess.run(
        ['ip', '-j', '-4', 'address', 'show', 'dev', name],
        check=True, capture_output=True, text=True, timeout=5,
    )
    addresses = []
    for interface in json.loads(result.stdout):
        if interface.get('ifname') != name:
            raise ValueError('ip returned a different interface')
        if 'UP' not in interface.get('flags', []):
            raise ValueError('interface is down')
        for address in interface.get('addr_info', []):
            if address.get('family') == 'inet':
                addresses.append(str(ipaddress.IPv4Address(address['local'])))
    if len(addresses) != 1:
        raise ValueError('expected exactly one provisioned IPv4 address')
    return addresses[0]


def probe(interface, local_ip, timeout, socket_factory=socket.socket, clock=time.monotonic):
    """Probe only the PCI-owned netdev; replies remain untrusted observations."""
    responses = []
    seen = set()
    malformed = 0
    with socket_factory(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        # Linux SO_BINDTODEVICE is essential: never fall back to a general LAN scan.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode() + b'\0')
        sock.bind((local_ip, SOURCE_PORT))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        sock.sendto(b'DISCOVER', (GROUP, PORT))
        deadline = clock() + timeout
        # Bound time and output even if a responder floods the socket.
        for _ in range(256):
            remaining = deadline - clock()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, sender = sock.recvfrom(4097)
            except socket.timeout:
                break
            try:
                if len(data) > 4096:
                    raise ValueError('oversized reply')
                payload = json.loads(data.decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('reply is not an object')
                key = (sender[0], json.dumps(payload, sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    responses.append({'source_ip': sender[0], 'payload': payload})
            except (ValueError, UnicodeError):
                malformed += 1
        else:
            raise ValueError('discovery packet limit exceeded')
    return {'responses': responses, 'malformed_packets': malformed}


def collect(root, timeout):
    before = inventory(root)
    errors = []
    observations = []
    if not before:
        errors.append('No SiMa PCI functions found')
    for card in before:
        entry = dict(card, probes=[])
        observations.append(entry)
        if len(card['interfaces']) != 1:
            errors.append(f"{card['bdf']}: expected exactly one PCI-owned netdev")
            continue
        name = card['interfaces'][0]
        try:
            local_ip = interface_ipv4(name)
            result = probe(name, local_ip, timeout)
            entry['probes'].append(dict(result, interface=name, local_ip=local_ip))
            if len(result['responses']) != 1:
                errors.append(f"{card['bdf']}: missing or ambiguous discovery response")
            if result['malformed_packets']:
                errors.append(f"{card['bdf']}: malformed discovery replies")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"{card['bdf']}: {exc}")
    if inventory(root) != before:
        errors.append('PCI inventory changed during discovery; rerun with stable topology')
    return {
        'schema_version': 1,
        'installation_authorized': False,
        'identity_status': 'unverified',
        'cards': observations,
        'errors': errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='seconds per interface (0 < timeout <= 30)')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 30:
        parser.error('timeout must be finite and in (0, 30]')
    if sys.platform != 'linux':
        parser.error('requires a Linux PCIe host')
    try:
        result = collect(Path('/sys/bus/pci/devices'), args.timeout)
    except (OSError, ValueError) as exc:
        print(json.dumps({'installation_authorized': False, 'errors': [str(exc)]}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

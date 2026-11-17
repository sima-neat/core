"""Host-only tests for the non-installing PCIe qualification collector."""
import importlib.util
from pathlib import Path
import socket
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    'offline_discovery', Path(__file__).resolve().parents[2] / 'pcie_host/scripts/offline/discover.py')
discovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(discovery)


def pci(root, bdf, vendor='0x1f06', interface=None):
    device = root / bdf
    device.mkdir()
    (device / 'vendor').write_text(vendor)
    (device / 'device').write_text('0x0121')
    if interface:
        (device / 'net' / interface).mkdir(parents=True)


def test_inventory_retains_card_without_network_and_excludes_other_vendor(tmp_path):
    pci(tmp_path, '0000:02:00.0', interface='enp2s0')
    pci(tmp_path, '0000:01:00.0')
    pci(tmp_path, '0000:03:00.0', vendor='0x8086', interface='eth0')
    cards = discovery.inventory(tmp_path)
    assert [c['bdf'] for c in cards] == ['0000:01:00.0', '0000:02:00.0']
    assert cards[0]['interfaces'] == []
    assert cards[1]['interfaces'] == ['enp2s0']


class Socket:
    def __init__(self, packets):
        self.packets = iter(packets)
        self.options = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def setsockopt(self, *args):
        self.options.append(args)
    def bind(self, address):
        self.bound = address
    def sendto(self, data, address):
        self.sent = (data, address)
    def settimeout(self, timeout):
        assert 0 < timeout <= 2
    def recvfrom(self, size):
        try:
            return next(self.packets), ('10.0.0.2', 50000)
        except StopIteration:
            raise socket.timeout


def test_protocol_scoping_dedup_and_untrusted_advertised_ip(monkeypatch):
    monkeypatch.setattr(socket, 'SO_BINDTODEVICE', 25, raising=False)
    sock = Socket([b'bad', b'[]', b'{"ip":"192.0.2.99"}', b'{"ip":"192.0.2.99"}'])
    result = discovery.probe('enp2s0', '10.0.0.1', 2, lambda *args: sock)
    assert sock.sent == (b'DISCOVER', ('239.255.42.1', 50000))
    assert sock.bound == ('10.0.0.1', 60000)
    assert (socket.SOL_SOCKET, 25, b'enp2s0\0') in sock.options
    assert result['malformed_packets'] == 2
    assert len(result['responses']) == 1
    assert result['responses'][0]['source_ip'] == '10.0.0.2'
    assert sock.closed


def test_no_general_interface_fallback(monkeypatch):
    monkeypatch.setattr(socket, 'SO_BINDTODEVICE', 25, raising=False)
    sock = Socket([])
    sock.setsockopt = lambda *args: (_ for _ in ()).throw(PermissionError('denied'))
    with pytest.raises(PermissionError):
        discovery.probe('enp2s0', '10.0.0.1', 2, lambda *args: sock)
    assert not hasattr(sock, 'sent')
    assert sock.closed


def test_sequential_all_card_collection_continues_after_failure(tmp_path):
    pci(tmp_path, '0000:01:00.0', interface='card1')
    pci(tmp_path, '0000:02:00.0', interface='card2')
    calls = []
    def probe(interface, *args):
        calls.append(interface)
        if interface == 'card1':
            raise OSError('unreachable')
        return {'responses': [{'source_ip': '10.0.1.2', 'payload': {}}], 'malformed_packets': 0}
    with patch.object(discovery, 'interface_ipv4', return_value='10.0.0.1'), patch.object(discovery, 'probe', side_effect=probe):
        result = discovery.collect(tmp_path, 2)
    assert calls == ['card1', 'card2']
    assert len(result['cards']) == 2
    assert len(result['errors']) == 1
    assert result['installation_authorized'] is False
    assert result['identity_status'] == 'unverified'


def test_empty_inventory_fails(tmp_path):
    assert discovery.collect(tmp_path, 2)['errors']


def test_topology_change_fails():
    with patch.object(discovery, 'inventory', side_effect=[[], [{'bdf': 'new'}]]):
        assert 'PCI inventory changed' in discovery.collect(Path('/unused'), 2)['errors'][-1]


@pytest.mark.parametrize('responses', [[], [{}, {}]])
def test_missing_or_ambiguous_response_fails(tmp_path, responses):
    pci(tmp_path, '0000:01:00.0', interface='card1')
    with patch.object(discovery, 'interface_ipv4', return_value='10.0.0.1'), patch.object(
        discovery, 'probe', return_value={'responses': responses, 'malformed_packets': 0}
    ):
        result = discovery.collect(tmp_path, 2)
    assert 'missing or ambiguous' in result['errors'][0]


def test_packet_flood_is_bounded(monkeypatch):
    monkeypatch.setattr(socket, 'SO_BINDTODEVICE', 25, raising=False)
    sock = Socket([b'{}'] * 257)
    with pytest.raises(ValueError, match='packet limit'):
        discovery.probe('card1', '10.0.0.1', 2, lambda *args: sock, lambda: 0)
    assert sock.closed

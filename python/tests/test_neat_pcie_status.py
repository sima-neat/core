"""PCIe status preserves partial inventory and never mutates card software."""
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('neat_pcie_status', ROOT/'scripts/neat_pcie_status.py')
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def report():
    return {'board': {'MACHINE': 'modalix', 'SIMA_BUILD_VERSION': '3.0.0_B2372'},
            'os': {'ID': 'elxr', 'VERSION_ID': '26.04'},
            'packages': {'sima-neat': {'version': '0.6.0', 'architecture': 'arm64'},
                         'simaai-scheduler': {'version': '0.2.1', 'architecture': 'arm64'}},
            'python': {'pyneat': {'version': '0.6.0', 'error': None}}}


def pci(tmp_path, bdf, interface=None, vendor='0x1f06'):
    device = tmp_path/bdf
    device.mkdir()
    (device/'vendor').write_text(vendor)
    (device/'device').write_text('0x1000')
    if interface:
        (device/'net'/interface).mkdir(parents=True)


@pytest.mark.parametrize('failure', ['host-key', 'timeout'])
def test_all_cards_remain_visible_when_one_ssh_probe_fails(tmp_path, failure):
    pci(tmp_path, '0000:01:00.0', 'enp1s0')
    pci(tmp_path, '0000:02:00.0', 'enp2s0')
    pci(tmp_path, '0000:03:00.0')
    pci(tmp_path, '0000:04:00.0', 'eth0', vendor='0x1234')
    commands = []
    def command(args, **kwargs):
        commands.append(args)
        assert 'sudo' not in str(args)
        assert 'install' not in str(args)
        assert 'StrictHostKeyChecking=yes' in args
        assert 'BatchMode=yes' in args
        assert args[args.index('-b')+1].startswith('10.0.')
        if 'sima@10.0.1.2' in args:
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(args, 8)
            raise subprocess.CalledProcessError(255, args, stderr='Host key verification failed')
        assert kwargs['input'] == (ROOT/'scripts/neat_pcie_status.py').read_text()
        return json.dumps(report())
    with patch.object(status, 'software', return_value=report()), \
         patch.object(status, 'endpoint', side_effect=[('10.0.0.1', '10.0.0.2'), ('10.0.1.1', '10.0.1.2')]), \
         patch.object(status, 'command', side_effect=command):
        result = status.collect(tmp_path)
    assert len(result['cards']) == 3
    assert [card['state'] for card in result['cards']] == ['ok', 'unavailable', 'unavailable']
    assert result['cards'][0]['software']['packages']['sima-neat']['version'] == '0.6.0'
    assert ('Host key' if failure == 'host-key' else 'timed out') in result['cards'][1]['error']
    assert len(commands) == 2


@pytest.mark.parametrize('remote', [{'board': 'wrong'}, {'board': {'MACHINE': 'other'}},
                                   {'board': {'MACHINE': 'modalix'}, 'packages': [], 'python': {}},
                                   {'board': {'MACHINE': 'modalix'}, 'packages': {'x': None}, 'python': {}}])
def test_unknown_or_malformed_remote_identity_is_not_a_healthy_card(tmp_path, remote):
    pci(tmp_path, '0000:01:00.0', 'enp1s0')
    with patch.object(status, 'software', return_value=report()), \
         patch.object(status, 'endpoint', return_value=('10.0.0.1', '10.0.0.2')), \
         patch.object(status, 'command', return_value=json.dumps(remote)):
        card = status.collect(tmp_path)['cards'][0]
    assert card['state'] == 'unavailable'
    assert card['software'] is None


@pytest.mark.parametrize('wrong_route', [False, True])
def test_endpoint_uses_only_provisioned_pci_interface(wrong_route):
    address = [{'ifname': 'enp1s0', 'flags': ['UP'], 'addr_info': [{'family': 'inet', 'local': '10.0.2.1'}]}]
    route = [{'dev': 'eth0' if wrong_route else 'enp1s0'}]
    with patch.object(status, 'command', side_effect=[json.dumps(address), json.dumps(route)]):
        if wrong_route:
            with pytest.raises(ValueError, match='PCI-owned'):
                status.endpoint('enp1s0')
        else:
            assert status.endpoint('enp1s0') == ('10.0.2.1', '10.0.2.2')


def test_unconfigured_interface_is_not_used_to_scan_other_networks():
    address = [{'ifname': 'enp1s0', 'flags': ['UP'], 'addr_info': [{'family': 'inet', 'local': '192.168.1.1'}]}]
    with patch.object(status, 'command', return_value=json.dumps(address)) as command:
        with pytest.raises(ValueError, match='10.0.N.1'):
            status.endpoint('enp1s0')
    assert command.call_count == 1


def test_changed_topology_is_reported(tmp_path):
    with patch.object(status, 'software', return_value=report()), \
         patch.object(status, 'inventory', side_effect=[[], [{'bdf': 'new'}]]):
        result = status.collect(tmp_path)
    assert 'changed' in result['errors'][0]


def test_package_probe_ignores_uninstalled_and_unrelated_packages(monkeypatch):
    output = 'sima-neat\t0.6.0\tarm64\tinstalled\nneat-old\t0.1\tarm64\tconfig-files\nbash\t5\tarm64\tinstalled\n'
    monkeypatch.setattr(status.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ''))
    monkeypatch.setattr(status.importlib.metadata, 'distributions', lambda **kwargs: [])
    assert status.software()['packages'] == {'sima-neat': {'version': '0.6.0', 'architecture': 'arm64'}}


def test_render_keeps_unavailable_cards_and_sanitizes_diagnostics(capsys):
    status.render({'host': report(), 'errors': [], 'cards': [
        {'bdf': '0000:01:00.0', 'interfaces': [], 'address': None, 'software': None, 'error': '\x1b[2JOffline\ncard'}]})
    text = capsys.readouterr().out
    assert '0000:01:00.0' in text and 'Software unavailable' in text
    assert '\x1b' not in text


@pytest.mark.parametrize('json_output', [False, True])
def test_packaged_neat_detects_host_and_reports_without_registry_queries(tmp_path, json_output):
    import os
    import shutil
    import sys
    bindir = tmp_path/'usr/bin'
    shared = tmp_path/'usr/share/sima-pcie-host'
    bindir.mkdir(parents=True)
    shared.mkdir(parents=True)
    shutil.copy2(ROOT/'scripts/neat', bindir/'neat')
    shutil.copy2(ROOT/'scripts/neat_pcie_status.py', shared/'neat_pcie_status.py')
    value = {'host': report(), 'cards': [], 'errors': []}
    value['host']['packages'] = {'sima-pcie-host': {'version': '0.6.0', 'architecture': 'amd64'},
                                'simaai-scheduler-pcie-host': {'version': '0.2.1', 'architecture': 'amd64'}}
    value['host']['python'] = {'pyneatpcie': {'version': '0.6.0', 'error': None}}
    fixture = tmp_path/'status.json'
    fixture.write_text(json.dumps(value))
    def executable(name, source):
        path = bindir/name
        path.write_text('#!/bin/sh\n'+source)
        path.chmod(0o755)
    executable('dpkg-query', 'printf installed')
    executable('python3', f'if [ "${{2##*/}}" = "neat_pcie_status.py" ] && [ "$#" = 2 ]; then cat "{fixture}"; else exec "{sys.executable}" "$@"; fi')
    executable('curl', 'echo unexpected-network >&2; exit 99')
    env = {**os.environ, 'PATH': str(bindir)+':'+os.environ['PATH'],
           'ELXR_SDK_RELEASE_FILE': str(tmp_path/'absent'), 'NEAT_BUILDINFO_FILE': str(tmp_path/'absent'),
           'SYSROOT': str(tmp_path/'cross-compile-sysroot'), 'HOME': str(tmp_path)}
    Path(env['SYSROOT']).mkdir()
    result = subprocess.run([str(bindir/'neat'), *(['--json'] if json_output else [])], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    if json_output:
        value = json.loads(result.stdout)
        assert value['environment']['mode'] == 'pcie-host'
        assert value['pcie']['cards'] == []
        assert value['updateCheck']['status'] == 'skipped'
        assert value['components']['pcieHost']['version'] == '0.6.0'
        assert value['components']['pyneatpcie']['version'] == '0.6.0'
    else:
        assert 'Ubuntu PCIe host' in result.stdout
        assert 'No SiMa PCIe cards detected' in result.stdout
        assert 'sima-pcie-host' in result.stdout and 'pyneatpcie' in result.stdout
    update = subprocess.run([str(bindir/'neat'), 'update', '--yes'], env=env, text=True, capture_output=True)
    assert update.returncode != 0
    assert 'core/pciehost/' in update.stderr

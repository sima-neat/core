from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NEAT = REPO_ROOT / "scripts" / "neat"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def write_exe(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def run_neat(tmp_path: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    merged.setdefault("NO_COLOR", "1")
    return subprocess.run(
        [str(NEAT), *args],
        cwd=REPO_ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def base_env(tmp_path: Path, bin_dir: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    return {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "HOME": str(home),
        "ELXR_SDK_RELEASE_FILE": str(tmp_path / "missing-sdk-release"),
        "NEAT_BUILDINFO_FILE": str(tmp_path / "missing-buildinfo"),
        "NEAT_ARTIFACTS_BASE_URL": "https://core.test",
        "NEAT_INSIGHT_BASE_URL": "https://insight.test",
        "NEAT_PORT_MAP_FILE": str(tmp_path / "missing-port-map.json"),
        "NEAT_INSIGHT_VENV_DIR": "",
        "SIMA_CLI_BIN": "",
        "SIMA_CLI_REGISTRY_FILE": str(home / ".sima-cli" / "registry.json"),
        "SYSROOT": str(tmp_path / "missing-sysroot"),
    }


def write_neat_debs(
    cache: Path,
    version: str,
    *,
    core: bool = True,
    dev: bool = True,
) -> None:
    if core:
        (cache / f"sima-neat-{version}-Linux-core.deb").write_text("", encoding="utf-8")
    if dev:
        (cache / f"sima-neat-{version}-Linux-dev.deb").write_text("", encoding="utf-8")


def component_line(stdout: str, name: str) -> str:
    return next(line for line in stdout.splitlines() if ANSI_RE.sub("", line).strip().startswith(name))


def make_curl(bin_dir: Path, log_path: Path | None = None) -> None:
    log_line = f'echo "$url" >> "{log_path}"' if log_path else ":"
    write_exe(
        bin_dir / "curl",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        url="${{@: -1}}"
        {log_line}
        case "$url" in
          https://core.test/main/latest.tag) printf 'abcdef0\\n' ;;
          https://core.test/main/abcdef0/metadata-minimal.json) printf '{{"version":"0.0.0+main-abcdef0"}}\\n' ;;
          https://core.test/develop/1234567/metadata-minimal.json) printf '{{"version":"0.0.0+develop-1234567"}}\\n' ;;
          https://insight.test/main/latest.tag) printf '7654321\\n' ;;
          https://insight.test/main/7654321/metadata.json) printf '{{"version":"0.0.0+main.7654321"}}\\n' ;;
          *) echo "unexpected curl url: $url" >&2; exit 22 ;;
        esac
        """,
    )


def make_pip(path: Path, package: str, version: str) -> None:
    write_exe(
        path,
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${{1:-}}" == "show" && "${{2:-}}" == "{package}" ]]; then
          printf 'Name: {package}\\nVersion: {version}\\n'
          exit 0
        fi
        exit 1
        """,
    )


def make_sima_cli_with_playbooks(bin_dir: Path, output: str = "sima-neat\nuse-neat-insight\n") -> None:
    write_exe(
        bin_dir / "sima-cli",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${{1:-}}" == "playbooks" && "${{2:-}}" == "list" ]]; then
          printf '%s' {output!r}
          exit 0
        fi
        echo "unexpected sima-cli $*" >&2
        exit 2
        """,
    )


def make_sima_cli_executable(path: Path, output: str = "sima-neat\nuse-neat-insight\n") -> None:
    write_exe(
        path,
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${{1:-}}" == "playbooks" && "${{2:-}}" == "list" ]]; then
          printf '%s' {output!r}
          exit 0
        fi
        if [[ "${{1:-}}" == "playbooks" && "${{2:-}}" == "update" ]]; then
          exit 0
        fi
        if [[ "${{1:-}}" == "install" ]]; then
          exit 0
        fi
        echo "unexpected sima-cli $*" >&2
        exit 2
        """,
    )


def make_insight_admin(bin_dir: Path, calls: Path | None = None, running: bool = True) -> None:
    log_line = f'echo "insight-admin $*" >> "{calls}"' if calls else ":"
    status_line = (
        "neat-insight                     RUNNING   pid 18108, uptime 0:13:28"
        if running
        else "neat-insight                     STOPPED   Not started"
    )
    write_exe(
        bin_dir / "insight-admin",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        {log_line}
        if [[ "${{1:-}}" == "status" ]]; then
          printf '%s\\n' {status_line!r}
          exit 0
        fi
        if [[ "${{1:-}}" == "update" ]]; then
          exit 0
        fi
        echo "unexpected insight-admin $*" >&2
        exit 2
        """,
    )


def test_sdk_status_parses_sysroot_cache_and_checks_updates(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_curl(bin_dir)
    make_sima_cli_with_playbooks(bin_dir)
    make_insight_admin(bin_dir, running=True)
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    cache = sysroot / "neat-install-packages"
    cache.mkdir(parents=True)
    write_neat_debs(cache, "0.0.0+main-1111111")
    (cache / "pyneat-0.0.0-cp311-cp311-linux_aarch64.whl").write_text("", encoding="utf-8")
    (cache / "neat-runtime_0.0.1-main-2222222_arm64.deb").write_text("", encoding="utf-8")
    (cache / "neat-gst-plugins_0.0.1-main-2222222_arm64.deb").write_text("", encoding="utf-8")
    insight_venv = tmp_path / "insight" / "venv"
    (insight_venv / "bin").mkdir(parents=True)
    make_pip(insight_venv / "bin" / "pip3", "neat-insight", "0.0.0+main.3333333")
    env.update(
        {
            "ELXR_SDK_RELEASE_FILE": str(sdk_release),
            "SYSROOT": str(sysroot),
            "NEAT_INSIGHT_VENV_DIR": str(insight_venv),
        }
    )

    proc = run_neat(tmp_path, ["--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Mode               Neat SDK" in proc.stdout
    assert "SDK Version        10.0.0.244" in proc.stdout
    assert "Neat core" in proc.stdout
    assert "0.0.0+main-1111111" in proc.stdout
    assert "latest=abcdef0" in proc.stdout
    assert "PyNeat" in proc.stdout and "0.0.0" in proc.stdout
    assert "neat-insight" in proc.stdout and "0.0.0+main.3333333" in proc.stdout
    assert "status=Running" in proc.stdout
    assert (
        proc.stdout.count(
            "Update available: yellow component versions indicate available updates. "
            "Run: neat update."
        )
        == 1
    )
    assert "Coding Agent Playbooks" in proc.stdout
    assert "sima-neat" in proc.stdout
    assert "use-neat-insight" in proc.stdout


def test_status_update_available_note_is_yellow(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_curl(bin_dir)
    env = base_env(tmp_path, bin_dir)
    env["NO_COLOR"] = ""
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    cache = sysroot / "neat-install-packages"
    cache.mkdir(parents=True)
    write_neat_debs(cache, "0.0.0+main-1111111")
    env.update({"ELXR_SDK_RELEASE_FILE": str(sdk_release), "SYSROOT": str(sysroot)})

    proc = run_neat(tmp_path, ["--color=always"], env)

    assert proc.returncode == 0, proc.stderr
    assert "\x1b[0;33m0.0.0+main-1111111" in proc.stdout
    assert (
        "\x1b[0;33mUpdate available: yellow component versions indicate "
        "available updates. Run: neat update.\x1b[0m"
    ) in proc.stdout


def test_sdk_status_requires_neat_core_and_dev_debs(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    env["NO_COLOR"] = ""
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    env["ELXR_SDK_RELEASE_FILE"] = str(sdk_release)

    cases = (
        ("missing-dev", {"dev": False}, "missing dev artifact"),
        ("missing-core", {"core": False}, "missing core artifact"),
        (
            "mismatch",
            {"core": True, "dev": False},
            "core/dev version mismatch core=0.0.0+main-1111111 dev=0.0.0+main-2222222",
        ),
    )
    for name, deb_kwargs, detail in cases:
        sysroot = tmp_path / name / "sysroot"
        cache = sysroot / "neat-install-packages"
        cache.mkdir(parents=True)
        write_neat_debs(cache, "0.0.0+main-1111111", **deb_kwargs)
        if name == "mismatch":
            write_neat_debs(cache, "0.0.0+main-2222222", core=False)
        env["SYSROOT"] = str(sysroot)

        proc = run_neat(tmp_path, ["--offline", "--color=always"], env)

        assert proc.returncode == 0, proc.stderr
        neat_core_line = component_line(proc.stdout, "Neat core")
        assert "\x1b[0;33mnot installed" in neat_core_line
        assert detail in neat_core_line


def test_sdk_status_prints_exposed_ports_from_port_map(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    (sysroot / "neat-install-packages").mkdir(parents=True)
    port_map = tmp_path / "neat-port-map.json"
    port_map.write_text(
        json.dumps(
            {
                "cert": {
                    "certFile": "/sdk-cert/neat-sdk.pem",
                    "keyFile": "/sdk-cert/neat-sdk-key.pem",
                    "mount": "/sdk-cert",
                },
                "mainUI": {"container": 9900, "host": 9900, "protocol": "tcp"},
                "metadataUDP": {
                    "containerEnd": 9179,
                    "containerStart": 9100,
                    "hostEnd": 9179,
                    "hostStart": 9100,
                    "protocol": "udp",
                },
                "rtsp": {"tcp": {"container": 8554, "host": 8554}},
                "schema": "sima.neat.port-map.v1",
                "videoUDP": {
                    "containerEnd": 9079,
                    "containerStart": 9000,
                    "hostEnd": 9079,
                    "hostStart": 9000,
                    "protocol": "udp",
                },
            }
        ),
        encoding="utf-8",
    )
    env.update(
        {
            "ELXR_SDK_RELEASE_FILE": str(sdk_release),
            "SYSROOT": str(sysroot),
            "NEAT_PORT_MAP_FILE": str(port_map),
            "CONTAINER_HOST_IP": "192.0.2.10",
            "NFS_SERVER_HOST_IP": "192.0.2.20",
            "OPENVSCODE_SERVER_HTTPS_PORT": "10000",
            "OPENVSCODE_SERVER_TOKEN": "test-token",
            "OPENVSCODE_WORKSPACE": "/workspace",
        }
    )

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Web Access" in proc.stdout
    assert any(
        line.split() == ["Insight", "(local)", "https://127.0.0.1:9900"]
        for line in proc.stdout.splitlines()
    )
    assert any(
        line.split() == ["Insight", "(remote)", "https://192.0.2.20:9900"]
        for line in proc.stdout.splitlines()
    )
    assert any(
        line.split()
        == ["VS", "Code", "(local)", "https://127.0.0.1:10000/?tkn=test-token&folder=/workspace"]
        for line in proc.stdout.splitlines()
    )
    assert any(
        line.split()
        == ["VS", "Code", "(remote)", "https://192.0.2.20:10000/?tkn=test-token&folder=/workspace"]
        for line in proc.stdout.splitlines()
    )
    assert "Local access uses a browser on this SDK host; remote access uses another machine." in proc.stdout
    assert "Prefer the local URL on this host; it remains valid if the host network IP changes." in proc.stdout
    assert "VS Code URLs contain an access token; do not share them." in proc.stdout
    assert "Exposed Ports" in proc.stdout
    assert "Name               Protocol Host Port (Start) Host Port (End)" in proc.stdout
    assert any(line.split() == ["mainUI", "tcp", "9900", "-"] for line in proc.stdout.splitlines())
    assert any(
        line.split() == ["metadataUDP", "udp", "9100", "9179"]
        for line in proc.stdout.splitlines()
    )
    assert any(line.split() == ["rtsp.tcp", "tcp", "8554", "-"] for line in proc.stdout.splitlines())
    assert any(
        line.split() == ["videoUDP", "udp", "9000", "9079"] for line in proc.stdout.splitlines()
    )
    assert "containerStart" not in proc.stdout
    assert "certFile" not in proc.stdout


def test_sdk_status_colorizes_insight_web_ui_url(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    env["NO_COLOR"] = ""
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    (sysroot / "neat-install-packages").mkdir(parents=True)
    port_map = tmp_path / "neat-port-map.json"
    port_map.write_text(
        json.dumps({"mainUI": {"container": 9900, "host": 9900, "protocol": "tcp"}}),
        encoding="utf-8",
    )
    env.update(
        {
            "ELXR_SDK_RELEASE_FILE": str(sdk_release),
            "SYSROOT": str(sysroot),
            "NEAT_PORT_MAP_FILE": str(port_map),
            "CONTAINER_HOST_IP": "192.0.2.10",
            "NFS_SERVER_HOST_IP": "192.0.2.20",
            "OPENVSCODE_SERVER_HTTPS_PORT": "10000",
            "OPENVSCODE_SERVER_TOKEN": "test-token",
        }
    )

    proc = run_neat(tmp_path, ["--offline", "--color=always"], env)

    assert proc.returncode == 0, proc.stderr
    assert "\x1b[1m\x1b[0;32mInsight (local)" in proc.stdout
    assert "\x1b[1m\x1b[0;32mhttps://127.0.0.1:9900\x1b[0m" in proc.stdout
    assert "\x1b[1m\x1b[0;32mVS Code (remote)" in proc.stdout
    assert "\x1b[1m\x1b[0;32mhttps://192.0.2.20:10000/?tkn=test-token&folder=/workspace\x1b[0m" in proc.stdout


def test_sdk_status_prints_vscode_urls_without_port_map(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    env.update(
        {
            "ELXR_SDK_RELEASE_FILE": str(sdk_release),
            "NFS_SERVER_HOST_IP": "192.0.2.20",
            "OPENVSCODE_SERVER_HTTPS_PORT": "10000",
            "OPENVSCODE_SERVER_TOKEN": "test-token",
            "OPENVSCODE_WORKSPACE": "/workspace",
        }
    )

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Insight (local)" not in proc.stdout
    assert "https://127.0.0.1:10000/?tkn=test-token&folder=/workspace" in proc.stdout
    assert "https://192.0.2.20:10000/?tkn=test-token&folder=/workspace" in proc.stdout
    assert "Exposed Ports" not in proc.stdout


def test_json_status_exports_components_and_ports(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_curl(bin_dir)
    make_insight_admin(bin_dir, running=True)
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    cache = sysroot / "neat-install-packages"
    cache.mkdir(parents=True)
    write_neat_debs(cache, "0.0.0+main-1111111")
    (cache / "pyneat-0.0.0-cp311-cp311-linux_aarch64.whl").write_text("", encoding="utf-8")
    (cache / "neat-runtime_0.0.1-main-2222222_arm64.deb").write_text("", encoding="utf-8")
    (cache / "neat-gst-plugins_0.0.1-main-2222222_arm64.deb").write_text("", encoding="utf-8")
    insight_venv = tmp_path / "insight" / "venv"
    (insight_venv / "bin").mkdir(parents=True)
    make_pip(insight_venv / "bin" / "pip3", "neat-insight", "0.0.0+main.3333333")
    port_map = tmp_path / "neat-port-map.json"
    port_map.write_text(
        json.dumps(
            {
                "mainUI": {"container": 9900, "host": 9900, "protocol": "tcp"},
                "metadataUDP": {
                    "containerEnd": 9179,
                    "containerStart": 9100,
                    "hostEnd": 9179,
                    "hostStart": 9100,
                    "protocol": "udp",
                },
                "rtsp": {"tcp": {"container": 8554, "host": 8554}},
                "schema": "sima.neat.port-map.v1",
            }
        ),
        encoding="utf-8",
    )
    env.update(
        {
            "ELXR_SDK_RELEASE_FILE": str(sdk_release),
            "SYSROOT": str(sysroot),
            "NEAT_INSIGHT_VENV_DIR": str(insight_venv),
            "NEAT_PORT_MAP_FILE": str(port_map),
            "CONTAINER_HOST_IP": "192.0.2.10",
        }
    )

    proc = run_neat(tmp_path, ["--json"], env)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "sima.neat.status.v1"
    assert payload["environment"]["mode"] == "elxr-sdk"
    assert payload["environment"]["sdkVersion"] == "10.0.0.244"
    assert payload["updateCheck"] == {"offline": False, "status": "ok"}
    assert payload["components"]["core"]["version"] == "0.0.0+main-1111111"
    assert payload["components"]["core"]["latestTag"] == "abcdef0"
    assert payload["components"]["core"]["updateAvailable"] is True
    assert payload["components"]["pyneat"]["version"] == "0.0.0"
    assert payload["components"]["insight"]["version"] == "0.0.0+main.3333333"
    assert payload["components"]["insight"]["serviceState"] == "Running"
    assert payload["insight"]["webUiUrl"] == "https://192.0.2.10:9900"
    assert {
        "name": "metadataUDP",
        "protocol": "udp",
        "hostPortStart": 9100,
        "hostPortEnd": 9179,
    } in payload["exposedPorts"]
    assert {
        "name": "rtsp.tcp",
        "protocol": "tcp",
        "hostPortStart": 8554,
        "hostPortEnd": None,
    } in payload["exposedPorts"]
    assert "Coding Agent Playbooks" not in proc.stdout


def test_json_status_offline_skips_network(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exe(
        bin_dir / "curl",
        """\
        #!/usr/bin/env bash
        echo "curl should not be called" >&2
        exit 99
        """,
    )
    env = base_env(tmp_path, bin_dir)

    proc = run_neat(tmp_path, ["--json", "--offline"], env)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["environment"]["sdkVersion"] is None
    assert payload["updateCheck"] == {"offline": True, "status": "skipped"}
    assert payload["components"]["core"]["latestTag"] is None


def test_status_finds_sima_cli_in_user_install_location(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    env["PATH"] = f"{bin_dir}:/bin:/usr/bin"
    make_sima_cli_with_playbooks(bin_dir, output="global-playbook\n")
    sima_cli = Path(env["HOME"]) / ".sima-cli" / ".venv" / "bin" / "sima-cli"
    sima_cli.parent.mkdir(parents=True)
    make_sima_cli_executable(sima_cli)

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Coding Agent Playbooks" in proc.stdout
    assert "sima-cli not found" not in proc.stdout
    assert "sima-neat" in proc.stdout
    assert "use-neat-insight" in proc.stdout
    assert "global-playbook" not in proc.stdout


def test_insight_status_reports_stopped_when_insight_admin_is_not_running(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_insight_admin(bin_dir, running=False)
    env = base_env(tmp_path, bin_dir)
    insight_venv = tmp_path / "insight" / "venv"
    (insight_venv / "bin").mkdir(parents=True)
    make_pip(insight_venv / "bin" / "pip3", "neat-insight", "0.0.0+main.3333333")
    env["NEAT_INSIGHT_VENV_DIR"] = str(insight_venv)

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "neat-insight" in proc.stdout
    assert "status=Stopped" in proc.stdout


def test_insight_status_colorizes_running_and_stopped(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_insight_admin(bin_dir, running=True)
    env = base_env(tmp_path, bin_dir)
    env["NO_COLOR"] = ""
    insight_venv = tmp_path / "insight" / "venv"
    (insight_venv / "bin").mkdir(parents=True)
    make_pip(insight_venv / "bin" / "pip3", "neat-insight", "0.0.0+main.3333333")
    env["NEAT_INSIGHT_VENV_DIR"] = str(insight_venv)

    proc = run_neat(tmp_path, ["--offline", "--color=always"], env)

    assert proc.returncode == 0, proc.stderr
    assert "status=\x1b[0;32mRunning\x1b[0m" in proc.stdout

    make_insight_admin(bin_dir, running=False)

    proc = run_neat(tmp_path, ["--offline", "--color=always"], env)

    assert proc.returncode == 0, proc.stderr
    assert "status=\x1b[0;33mStopped\x1b[0m" in proc.stdout


def test_offline_status_does_not_contact_network(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exe(
        bin_dir / "curl",
        """\
        #!/usr/bin/env bash
        echo "curl should not be called" >&2
        exit 99
        """,
    )
    env = base_env(tmp_path, bin_dir)

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Update check       offline" in proc.stdout
    assert "latest=" not in proc.stdout


def test_status_update_check_failure_prints_download_reason(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exe(
        bin_dir / "curl",
        """\
        #!/usr/bin/env bash
        echo "curl: (28) Operation timed out after 8000 milliseconds" >&2
        exit 28
        """,
    )
    env = base_env(tmp_path, bin_dir)

    proc = run_neat(tmp_path, ["--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "update check failed" in proc.stdout
    assert "Neat core latest tag: https://core.test/" in proc.stdout
    assert "/latest.tag failed" in proc.stdout
    assert "neat-insight latest tag: https://insight.test/" in proc.stdout
    assert "curl: (28) Operation timed out after 8000 milliseconds" in proc.stdout
    assert "Try --offline to skip update checks" in proc.stdout


def test_status_update_check_failure_details_are_yellow(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exe(
        bin_dir / "curl",
        """\
        #!/usr/bin/env bash
        printf ''
        """,
    )
    env = base_env(tmp_path, bin_dir)
    env["NO_COLOR"] = ""

    proc = run_neat(tmp_path, ["--color=always"], env)

    assert proc.returncode == 0, proc.stderr
    assert "\x1b[0;33mupdate check failed\x1b[0m" in proc.stdout
    assert "\x1b[0;33mNeat core latest tag: https://core.test/" in proc.stdout
    assert "/latest.tag returned an empty response\x1b[0m" in proc.stdout
    assert (
        "\x1b[0;33mTry --offline to skip update checks when network access is unavailable.\x1b[0m"
    ) in proc.stdout


def test_json_status_update_check_failure_exports_errors(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exe(
        bin_dir / "curl",
        """\
        #!/usr/bin/env bash
        echo "curl: (28) Operation timed out after 8000 milliseconds" >&2
        exit 28
        """,
    )
    env = base_env(tmp_path, bin_dir)

    proc = run_neat(tmp_path, ["--json"], env)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["updateCheck"]["status"] == "failed"
    assert any(
        err.startswith("Neat core latest tag: https://core.test/")
        and "/latest.tag failed" in err
        for err in payload["updateCheck"]["errors"]
    )
    assert any(
        "curl: (28) Operation timed out" in err for err in payload["updateCheck"]["errors"]
    )


def test_status_infers_non_main_channel_from_core_version(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    cache = sysroot / "neat-install-packages"
    cache.mkdir(parents=True)
    write_neat_debs(cache, "0.0.0+feature-dash-name-9999999")
    env.update({"ELXR_SDK_RELEASE_FILE": str(sdk_release), "SYSROOT": str(sysroot)})

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "channel=feature-dash-name" in proc.stdout
    assert "(assumed)" not in proc.stdout.split("Neat core", 1)[1].splitlines()[0]


def test_model_compiler_installed_from_extension_dir(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    model_compiler = tmp_path / "sdk-extensions" / "model-compiler"
    site_packages = model_compiler / "lib" / "python3.10" / "site-packages"
    dist_info = site_packages / "sima_frontend-2.0.0.dev0+master.371.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: sima-frontend\n"
        "Version: 2.0.0.dev0+master.371\n",
        encoding="utf-8",
    )
    (model_compiler / "bin").mkdir(parents=True)
    write_exe(
        model_compiler / "bin" / "python",
        f"""\
        #!/usr/bin/env bash
        PYTHONPATH="{site_packages}" python3 "$@"
        """,
    )
    env["NEAT_MODEL_COMPILER_DIR"] = str(model_compiler)

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Model Compiler" in proc.stdout
    assert "2.0.0.dev0+master.371" in proc.stdout
    assert f"source {model_compiler}/bin/activate to activate" in proc.stdout

    proc = run_neat(tmp_path, ["--json", "--offline"], env)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["components"]["modelCompiler"] == {
        "name": "Model Compiler",
        "installed": True,
        "version": "2.0.0.dev0+master.371",
        "detail": f"source {model_compiler}/bin/activate to activate",
    }


def test_model_compiler_installed_from_legacy_sima_cli_registry(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    registry = Path(env["HOME"]) / ".sima-cli" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            [
                {
                    "package_name": "sima-neat-model-sdk",
                    "version": "2.0.0.neat+main-1ebbc39",
                    "state": "installed",
                    "install_path": "/home/jim",
                    "updated_at": "2026-05-05T20:09:34.188287Z",
                    "metadata": {"version": "2.0.0.neat+main-1ebbc39"},
                }
            ]
        ),
        encoding="utf-8",
    )

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Model Compiler" in proc.stdout
    assert "2.0.0.neat+main-1ebbc39" in proc.stdout
    assert "run activate-model-compiler to activate" in proc.stdout
    assert "sima-cli registry" not in proc.stdout
    assert "install_path=/home/jim" not in proc.stdout


def test_model_compiler_ignores_non_installed_registry_state(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    registry = Path(env["HOME"]) / ".sima-cli" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            [
                {
                    "package_name": "sima-neat-model-sdk",
                    "version": "2.0.0.neat+main-1ebbc39",
                    "state": "aborted",
                }
            ]
        ),
        encoding="utf-8",
    )

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    model_compiler_line = next(line for line in proc.stdout.splitlines() if "Model Compiler" in line)
    assert "not installed" in model_compiler_line


def test_model_compiler_ignores_activate_function_without_install(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = base_env(tmp_path, bin_dir)
    bashrc = Path(env["HOME"]) / ".bashrc"
    bashrc.write_text("activate-model-compiler() { :; }\n", encoding="utf-8")

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    model_compiler_line = next(line for line in proc.stdout.splitlines() if "Model Compiler" in line)
    assert "not installed" in model_compiler_line


def test_devkit_status_uses_dpkg_and_pyneat_venv(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    buildinfo = tmp_path / "buildinfo"
    buildinfo.write_text(
        "DISTRO = eLxr\n"
        "DISTRO_VERSION = 2.0.0\n"
        "MACHINE = modalix\n"
        "SIMA_BUILD_VERSION = 2.0.0_master_B827\n",
        encoding="utf-8",
    )
    write_exe(
        bin_dir / "dpkg-query",
        """\
        #!/usr/bin/env bash
        pkg="${@: -1}"
        case "$pkg" in
          sima-neat) printf '0.0.0+main-4444444' ;;
          sima-neat-dev) printf '0.0.0+main-4444444' ;;
          neat-runtime) printf '0.0.1-main-5555555' ;;
          neat-gst-plugins) printf '0.0.1-main-5555555' ;;
          *) exit 1 ;;
        esac
        """,
    )
    env = base_env(tmp_path, bin_dir)
    env["NEAT_BUILDINFO_FILE"] = str(buildinfo)
    pyneat_venv = Path(env["HOME"]) / "pyneat"
    (pyneat_venv / "bin").mkdir(parents=True)
    make_pip(pyneat_venv / "bin" / "pip3", "pyneat", "0.0.0")

    proc = run_neat(tmp_path, ["--offline", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Mode               Modalix DevKit" in proc.stdout
    assert "DevKit Version     2.0.0_master_B827" in proc.stdout
    assert "0.0.0+main-4444444" in proc.stdout
    assert "0.0.1-main-5555555" in proc.stdout
    assert "PyNeat" in proc.stdout and "0.0.0" in proc.stdout

    proc = run_neat(tmp_path, ["--json", "--offline"], env)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["environment"]["mode"] == "modalix-board"
    assert payload["environment"]["devkitBuildVersion"] == "2.0.0_master_B827"


def test_update_yes_dispatches_core_and_insight_commands(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_curl(bin_dir)
    calls = tmp_path / "calls.log"
    write_exe(
        bin_dir / "sima-cli",
        f"""\
        #!/usr/bin/env bash
        echo "sima-cli $*" >> "{calls}"
        """,
    )
    make_insight_admin(bin_dir, calls=calls, running=True)
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    cache = sysroot / "neat-install-packages"
    cache.mkdir(parents=True)
    write_neat_debs(cache, "0.0.0+main-1111111")
    insight_venv = tmp_path / "insight" / "venv"
    (insight_venv / "bin").mkdir(parents=True)
    make_pip(insight_venv / "bin" / "pip3", "neat-insight", "0.0.0+main.3333333")
    env.update(
        {
            "ELXR_SDK_RELEASE_FILE": str(sdk_release),
            "SYSROOT": str(sysroot),
            "NEAT_INSIGHT_VENV_DIR": str(insight_venv),
        }
    )

    proc = run_neat(tmp_path, ["update", "--yes", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Update Plan" in proc.stdout
    call_text = calls.read_text(encoding="utf-8")
    assert "sima-cli install -m https://core.test/main/abcdef0/metadata-minimal.json" in call_text
    assert "sima-cli playbooks update" in call_text
    assert "insight-admin update main 7654321" in call_text


def test_update_skips_current_core_and_insight_but_updates_playbooks(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_curl(bin_dir)
    calls = tmp_path / "calls.log"
    write_exe(
        bin_dir / "sima-cli",
        f"""\
        #!/usr/bin/env bash
        echo "sima-cli $*" >> "{calls}"
        """,
    )
    make_insight_admin(bin_dir, calls=calls, running=True)
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    cache = sysroot / "neat-install-packages"
    cache.mkdir(parents=True)
    write_neat_debs(cache, "0.0.0+main-abcdef0")
    insight_venv = tmp_path / "insight" / "venv"
    (insight_venv / "bin").mkdir(parents=True)
    make_pip(insight_venv / "bin" / "pip3", "neat-insight", "0.0.0+main.7654321")
    env.update(
        {
            "ELXR_SDK_RELEASE_FILE": str(sdk_release),
            "SYSROOT": str(sysroot),
            "NEAT_INSIGHT_VENV_DIR": str(insight_venv),
        }
    )

    proc = run_neat(tmp_path, ["update", "--yes", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "Neat core     0.0.0+main-abcdef0 (already current)" in proc.stdout
    assert "neat-insight  0.0.0+main.7654321 (already current)" in proc.stdout
    assert "only playbooks will be checked" in proc.stdout
    call_text = calls.read_text(encoding="utf-8")
    assert "sima-cli playbooks update" in call_text
    assert "sima-cli install -m" not in call_text
    assert "insight-admin update" not in call_text


def test_update_core_only_current_exits_without_confirmation_or_sima_cli(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_curl(bin_dir)
    calls = tmp_path / "calls.log"
    write_exe(
        bin_dir / "sima-cli",
        f"""\
        #!/usr/bin/env bash
        echo "sima-cli $*" >> "{calls}"
        """,
    )
    env = base_env(tmp_path, bin_dir)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 10.0.0.244\neLXr Version = 2.0.0\n", encoding="utf-8")
    sysroot = tmp_path / "sysroot"
    cache = sysroot / "neat-install-packages"
    cache.mkdir(parents=True)
    write_neat_debs(cache, "0.0.0+main-abcdef0")
    env.update({"ELXR_SDK_RELEASE_FILE": str(sdk_release), "SYSROOT": str(sysroot)})

    proc = run_neat(tmp_path, ["update", "--core-only", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "All selected components are already current." in proc.stdout
    assert not calls.exists()


def test_update_channel_and_tag_override_core_only(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_curl(bin_dir)
    calls = tmp_path / "calls.log"
    write_exe(
        bin_dir / "sima-cli",
        f"""\
        #!/usr/bin/env bash
        echo "sima-cli $*" >> "{calls}"
        """,
    )
    env = base_env(tmp_path, bin_dir)
    buildinfo = tmp_path / "buildinfo"
    buildinfo.write_text("MACHINE = modalix\n", encoding="utf-8")
    env["NEAT_BUILDINFO_FILE"] = str(buildinfo)
    write_exe(
        bin_dir / "dpkg-query",
        """\
        #!/usr/bin/env bash
        case "${@: -1}" in
          sima-neat|sima-neat-dev) printf '0.0.0+main-1111111' ;;
          *) exit 1 ;;
        esac
        """,
    )

    proc = run_neat(
        tmp_path,
        ["update", "--yes", "--core-only", "--channel", "develop", "--tag", "1234567", "--color=never"],
        env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "https://core.test/develop/1234567/metadata-minimal.json" in calls.read_text(
        encoding="utf-8"
    )


def test_vulcan_buildinfo_selects_core_update_environment(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    calls = tmp_path / "calls.log"
    write_exe(
        bin_dir / "curl",
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        url="${{@: -1}}"
        echo "$url" >> "{curl_log}"
        case "$url" in
          https://artifacts.neat.paconsultings.com/core/feat%2Fx/latest.tag) printf 'abcdef123456\\n' ;;
          https://artifacts.neat.paconsultings.com/core/feat%2Fx/abcdef123456/metadata-minimal.json) printf '{{"version":"2.0.0+feat.x.abcdef1"}}\\n' ;;
          *) echo "unexpected curl url: $url" >&2; exit 22 ;;
        esac
        """,
    )
    write_exe(
        bin_dir / "sima-cli",
        f"""\
        #!/usr/bin/env bash
        echo "sima-cli $*" >> "{calls}"
        """,
    )
    write_exe(
        bin_dir / "dpkg-query",
        """\
        #!/usr/bin/env bash
        case "${@: -1}" in
          sima-neat|sima-neat-dev) printf '2.0.0+feat.x.1111111' ;;
          *) exit 1 ;;
        esac
        """,
    )

    package_buildinfo = tmp_path / "package-buildinfo.json"
    package_buildinfo.write_text(
        json.dumps(
            {
                "format_version": 1,
                "package": "sima-neat",
                "version": "2.0.0+feat.x.1111111",
                "source": "vulcan",
                "vulcan": {
                    "environment": "dev",
                    "repository": "core",
                    "ref": "feat/x",
                    "ref_key": "feat%2Fx",
                    "spec": "111111111111",
                },
            }
        ),
        encoding="utf-8",
    )

    env = base_env(tmp_path, bin_dir)
    env.pop("NEAT_ARTIFACTS_BASE_URL")
    buildinfo = tmp_path / "buildinfo"
    buildinfo.write_text("MACHINE = modalix\n", encoding="utf-8")
    env["NEAT_BUILDINFO_FILE"] = str(buildinfo)
    env["NEAT_PACKAGE_BUILDINFO_FILE"] = str(package_buildinfo)

    proc = run_neat(tmp_path, ["update", "--yes", "--core-only", "--color=never"], env)

    assert proc.returncode == 0, proc.stderr
    assert "https://artifacts.neat.paconsultings.com/core/feat%2Fx/latest.tag" in curl_log.read_text(
        encoding="utf-8"
    )
    assert "https://artifacts.neat.paconsultings.com/core/feat%2Fx/abcdef123456/metadata-minimal.json" in calls.read_text(
        encoding="utf-8"
    )

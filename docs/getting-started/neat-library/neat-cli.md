---
title: Neat CLI
description: Use the neat command to inspect and update an installed Neat environment
sidebar_position: 4
---

The installed library provides the `neat` environment command. Run it from either
the SDK or DevKit to see installed component versions, installed `sima-cli`
playbooks, and whether newer artifacts are available.

In the SDK, the status output also includes Insight host-port mappings from
`$HOME/.insight-config/neat-port-map.json`.

<ShellCommand prompt="sdk-or-devkit">
neat
</ShellCommand>

Example output:

```text
Neat Environment
  Mode               Neat SDK
  Sysroot            /opt/toolchain/aarch64/modalix
  Update check       online

Components
  Neat core              0.0.0+main-6735c35 channel=main latest=6735c35
  PyNeat                 0.0.0
  neat-runtime           0.0.1-main-e4b19553e07d
  neat-gst-plugins       0.0.1-main-e4b19553e07d
  neat-insight           0.0.0+main.74ae0b5 channel=main latest=74ae0b5 status=Running venv=/opt/neat-insight/venv
  Model Compiler        2.0.0.neat+main-1ebbc39 run activate-model-compiler to activate

Exposed Ports
  Insight Web UI     https://10.0.0.22:9900

  Name               Protocol Host Port (Start) Host Port (End)
  ------------------ -------- ----------------- ---------------
  mainUI             tcp      9900              -
  metadataUDP        udp      9100              9179
  rtsp.tcp           tcp      8554              -
  videoUDP           udp      9000              9079
  videoUI            tcp      8081              -
  webRTC             udp      40000             40199
  webSSH             tcp      8022              -
```

## JSON Output

For automation and tool integrations, use JSON output:

<ShellCommand prompt="sdk-or-devkit">
neat --json
</ShellCommand>

## Update Installed Components

To update the Neat Library runtime, `neat-insight`, and installed `sima-cli`
playbooks from the detected channel, run:

<ShellCommand prompt="sdk-or-devkit">
neat update
</ShellCommand>

## Next Step

Continue with [Hello Neat!](/develop-apps/hello-neat/minimal) to run your first
Neat inference.

---
title: Neat CLI
description: Use the neat command to inspect and update an installed Neat environment
sidebar_position: 5
---

The installed library provides the `neat` environment command. Run it from either
the SDK or DevKit to see installed component versions, installed `sima-cli`
playbooks, and whether newer artifacts are available.

In the SDK, the status output also includes Insight host-port mappings from
`$HOME/.insight-config/neat-port-map.json`.

Missing PyNeat or Insight installations do not prevent status reporting. If a
version probe fails, the component shows `unknown` and `neat`
writes the provider diagnostic to standard error. JSON output keeps the existing
schema and reports unavailable versions as `null`.

Split installations report `simaai-runner` and `simaai-scheduler`. On the DevKit, versions come
from the installed package database; in the SDK, they come from the cached DEB control metadata.
Legacy `neat-runtime` and `neat-gst-plugins` rows remain visible when those packages are present
or no split component is detected. JSON adds `components.runner` and `components.scheduler`
while preserving the legacy `runtime` and `gstPlugins` keys; absent versions remain `null`.

<ShellCommand prompt="sdk|devkit">
neat
</ShellCommand>

Example output from a legacy installation:

```text
Neat Environment
  Mode               Neat SDK
  Sysroot            /opt/toolchain/aarch64/modalix
  Update check       online

Components
  Neat core              %%release.core%% channel=release latest=%%release.core%%
  PyNeat                 %%release.core%%
  neat-runtime           %%release.core%%
  neat-gst-plugins       %%release.core%%
  neat-insight           %%release.insight%% channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         %%release.model_compiler%% run activate-model-compiler to activate

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

<ShellCommand prompt="sdk|devkit">
neat --json
</ShellCommand>

## Update Installed Components

To update the Neat Library runtime, `neat-insight`, and installed `sima-cli`
playbooks from the detected channel, run:

<ShellCommand prompt="sdk|devkit">
neat update
</ShellCommand>

To update only the Neat Library runtime, without inspecting or updating Insight
or playbooks, run:

<ShellCommand prompt="sdk|devkit">
neat update --core-only
</ShellCommand>

## Next Step

Continue to [Compile a Model](/compile-a-model/) to prepare a model for Modalix —
this runs in the SDK and needs no DevKit. To run a full application on hardware,
see [Hello Neat!](/develop-apps/hello-neat/minimal/), which requires a paired
DevKit.

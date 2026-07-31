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
  Neat core              0.2.2 channel=release latest=0.2.2
  PyNeat                 0.2.2
  neat-runtime           0.2.2
  neat-gst-plugins       0.2.2
  neat-insight           0.0.3 channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         2.1.2 run activate-model-compiler to activate

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

## Model Archives

These commands run on Modalix and in AMD64 or ARM64 Neat SDKs.

Validate a compiled model archive with the same archive checks used by `Model`:

<ShellCommand prompt="sdk-or-devkit">
neat model validate yolo_v8s_mpk.tar.gz
</ShellCommand>

```text
yolo_v8s_mpk: valid model archive (7 entries, 162.4 MiB extracted)
```

A rejected archive prints the loader error and exits nonzero. Validation does not
run inference or check model output.

```text
neat-model-archive: invalid_archive: failed to decompress archive: yolo_v8s_mpk.tar.gz
```

Extract the archive into the package layout accepted by the loader's directory
fast path:

<ShellCommand prompt="sdk-or-devkit">
neat model extract yolo_v8s_mpk.tar.gz --output ./yolo_v8s_pkg
</ShellCommand>

The output contains `etc`, `lib`, and `share`. The command refuses an existing
output path and removes incomplete output after a failure.

Pass the produced directory to `Model` in place of the archive:

```cpp
simaai::neat::Model yolo("./yolo_v8s_pkg", opt);
```

The extracted package is an ordinary user-owned directory. Neat does not make it
immutable or verify it again. Do not edit or move it; re-extract the archive when
needed. It can be deleted normally when no longer in use. Loading does not write
to it, so it may be shared by multiple processes or made read-only.

Model loading reports the selected package location, for example:

```text
Model loaded: yolo_v8s_mpk (package storage: NVMe, root: /media/nvme/.../yolo_v8s_mpk)
```

## Next Step

Continue to [Compile a Model](/compile-a-model/) to prepare a model for Modalix —
this runs in the SDK and needs no DevKit. To run a full application on hardware,
see [Hello Neat!](/develop-apps/hello-neat/minimal/), which requires a paired
DevKit.

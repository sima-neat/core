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

To check a compiled `.tar.gz` model package before an application tries to load
it, run:

<ShellCommand prompt="sdk-or-devkit">
neat model validate yolo_v8s_mpk.tar.gz
</ShellCommand>

```text
yolo_v8s_mpk: valid model archive (7 entries, 162.4 MiB extracted)
```

This applies the same archive rules the runtime applies when a `Model` loads the
same file, so an archive that validates here loads at runtime and one that fails
here fails there with the same error. It does not run inference, so it says
nothing about whether the model produces correct results on a given target. A
rejected archive prints the reason and exits nonzero:

```text
neat-model-archive: invalid_archive: failed to decompress archive: yolo_v8s_mpk.tar.gz
```

To unpack an archive into the package layout the runtime uses, so that loading it
skips extraction, run:

<ShellCommand prompt="sdk-or-devkit">
neat model extract yolo_v8s_mpk.tar.gz --output ./yolo_v8s_pkg
</ShellCommand>

`--output` is the package root itself, and it holds `etc`, `lib`, and `share`
once the command succeeds. The command refuses to run if that path already
exists, and a failed run leaves nothing behind.

Pass the produced directory to `Model` in place of the archive:

```cpp
simaai::neat::Model yolo("./yolo_v8s_pkg", opt);
```

This is the faster way to load a model, and the reason to extract one at all.
Constructing a `Model` from an archive decompresses, validates, organizes and
path-rewrites it first. Constructing from an extracted directory does none of
that — the package is loaded in place.

The saving is per process, not per load. Within one process the extraction is
reused, so only the first archive load pays for it. A new process starts over.
Applications that load a model once at startup therefore gain the most. On a
Modalix DevKit, loading a 163 MiB YOLO26-L package:

| Input | First load in a process | Later loads |
| --- | --- | --- |
| `.tar.gz` archive | 2.49 s | 0.01 s |
| extracted directory | 0.01 s (0.27 s cold cache) | 0.01 s |

Treat the figures as an illustration — they scale with model size and with the
machine. Whether the package sits on eMMC or NVMe makes no measurable
difference.

Only a directory produced by `neat model extract` works this way. Unpacking the
archive yourself with `tar -xzf` produces a different layout that `Model` does
not accept.

### Keeping an extracted package valid

The output is a plain directory that you own, so delete it with `rm -rf` when you
no longer want it. Nothing else refers to it.

Keeping it correct is your responsibility. Neat does not verify a package after
extraction: it checks that `etc`, `lib`, and `share` are present and that `etc`
holds configuration, and it does not re-check the contents against the archive.
Editing a config, deleting or truncating a model binary, or moving the directory
will not be reported when the `Model` is constructed — the paths written into the
package JSON are absolute, so a moved package points at files that are no longer
there, and the failure surfaces later. Re-run `neat model extract` instead of
editing or relocating an extracted package.

Loading never writes into the package, so one package can back several processes
and can be kept read-only.

## Next Step

Continue to [Compile a Model](/compile-a-model/) to prepare a model for Modalix —
this runs in the SDK and needs no DevKit. To run a full application on hardware,
see [Hello Neat!](/develop-apps/hello-neat/minimal/), which requires a paired
DevKit.
